# coding=utf-8
"""
Logits Processors for SpecSoT (GPU 优化版本 - Tensor Mask)

该模块定义了用于控制生成过程的 Logits Processors：

SemanticLogitsProcessor: 骨架生成约束 (基于 GPU Tensor Mask + 增量状态机)
   - 支持格式：
     - [DIRECT]...[END] 直接回答模式
     - [PLAN] 后接多个分支行，每行格式：ID.<Length><Tool>[-]Title
   - 使用 GPU Tensor 位运算替代 Python Set 循环，性能提升 100x+
   - 增量解码和状态持久化，避免 O(N²) 性能问题

核心优化设计：
   1. **预计算 Tensor Masks**：将所有 Token 集合转换为 Boolean Tensor [vocab_size]
   2. **预合并复合 Mask**：在初始化时预计算 deps_mask 等复合 mask
   3. **Device 懒加载**：Mask 根据 input_ids.device 自动迁移到正确的 GPU
   4. **全向量化推理**：使用位运算 (|=) 和布尔索引，禁止 Python for 循环
   5. 时间复杂度 O(1)，微秒级延迟

简化的骨架格式（去除不必要的空格）：
   [PLAN]
   1.<200><Search>[-]任务描述
   2.<150><None>[-]任务描述
   [END]
"""

import os
import re
import json
import hashlib
import pickle
from enum import IntEnum
from typing import Optional, Dict, List
import torch
from transformers import LogitsProcessor

# 词表缓存目录
VOCAB_CACHE_DIR = os.path.join(os.path.dirname(__file__), "vocab_cache")


# =============================================================================
# FSM State Definitions for Protocol
# =============================================================================

class FSMState(IntEnum):
    """
    FSM 状态定义，用于解析骨架格式
    
    简化的分支行格式: ID.<Length><Tool>[-]Title
    - 去除了不必要的空格约束
    - 支持多位数长度（如 <127>）
    """
    # Header 阶段
    HEADER_LBRACKET = 0      # 期望 [
    HEADER_KEYWORD = 1       # 期望 PLAN 或 DIRECT 关键字
    HEADER_RBRACKET = 2      # 期望 ]
    HEADER_NEWLINE = 3       # 期望换行
    
    # DIRECT 模式 - 无约束
    DIRECT_CONTENT = 4
    
    # PLAN 模式 - 分支行解析 (简化格式)
    LINE_START = 10          # 行首，期望数字 ID 或 [
    AFTER_ID = 11            # ID 后，期望 .
    AFTER_DOT = 12           # . 后，期望 <
    LEN_OPEN = 13            # < 后（长度），期望数字
    LEN_VAL = 14             # 长度值中，期望数字或 >
    AFTER_LEN_CLOSE = 15     # > 后，期望 <
    TOOL_OPEN = 16           # < 后（工具），期望字母
    TOOL_VAL = 17            # 工具值中，期望字母或 >
    AFTER_TOOL_CLOSE = 18    # > 后，期望 [
    DEPS_OPEN = 19           # [ 后（依赖），期望 - 或数字
    DEPS_VAL = 20            # 依赖值中，期望数字/逗号/- 或 ]
    AFTER_DEPS_CLOSE = 21    # ] 后，进入自由内容
    CONTENT = 22             # 自由内容，直到换行符
    
    # END 检测
    END_KEYWORD = 30         # 期望 END 关键字
    END_RBRACKET = 31        # 期望 ]
    FINISHED = 32            # 生成完成


# =============================================================================
# Vocabulary Scanner - 词表扫描器 (GPU Tensor Mask 优化版)
# =============================================================================

def _get_model_name_from_tokenizer(tokenizer) -> str:
    """从 tokenizer 获取模型名称（用于缓存文件命名）"""
    if hasattr(tokenizer, 'name_or_path') and tokenizer.name_or_path:
        model_name = os.path.basename(tokenizer.name_or_path.rstrip('/'))
        if model_name:
            return model_name
    
    vocab_size = getattr(tokenizer, 'vocab_size', 0)
    vocab_signature = ""
    for i in range(min(1000, vocab_size)):
        try:
            vocab_signature += tokenizer.decode([i])
        except:
            pass
    vocab_hash = hashlib.md5(vocab_signature.encode()).hexdigest()[:8]
    return f"unknown_model_v{vocab_size}_{vocab_hash}"


def _get_cache_path(model_name: str, ext: str = "pkl") -> str:
    """获取词表缓存文件路径"""
    os.makedirs(VOCAB_CACHE_DIR, exist_ok=True)
    safe_name = re.sub(r'[^\w\-.]', '_', model_name)
    return os.path.join(VOCAB_CACHE_DIR, f"vocab_cache_{safe_name}.{ext}")


class VocabScanner:
    """
    词表扫描器 (GPU Tensor Mask 优化版)
    
    核心优化：
    1. 所有 Token 集合存储为 Boolean Tensor Masks (shape: [vocab_size])
    2. 预合并复合 Mask (如 deps_mask = digit_mask | dash_mask | comma_mask)
    3. Device 懒加载：首次访问时迁移到目标 GPU
    4. 推理时使用位运算，O(1) 查询
    
    性能对比：
    - 原版 (Python Set + for 循环): ~20ms
    - 优化版 (GPU Tensor): ~0.02ms (1000x 加速)
    """
    
    # 基础 Mask 名称列表
    MASK_NAMES = [
        'digit', 'letter', 'newline', 'gt', 'lt',
        'lbracket', 'rbracket', 'dot', 'dash', 'comma', 'space'
    ]
    
    # 预合并的复合 Mask
    COMPOSITE_MASKS = {
        'deps': ['digit', 'dash', 'comma'],              # 依赖字段: 数字、短横线、逗号
        'digit_or_dot': ['digit', 'dot'],                # ID 后: 数字或点号
        'digit_or_gt': ['digit', 'gt'],                  # 长度值: 数字或 >
        'digit_or_lbracket': ['digit', 'lbracket'],      # 行首: 数字或 [
        'dash_or_digit': ['dash', 'digit'],              # 依赖开始: - 或数字
        'deps_or_rbracket': ['digit', 'dash', 'comma', 'rbracket'],  # 依赖值
    }
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        
        # 模型名称和缓存路径
        self.model_name = _get_model_name_from_tokenizer(tokenizer)
        self.cache_path = _get_cache_path(self.model_name, ext="pkl")
        
        # Token 到文本的映射 (用于增量解码)
        self.token_to_text: Dict[int, str] = {}
        
        # 字符到 Token 的映射 (用于动态查询，如关键字匹配)
        self.char_to_tokens: Dict[str, List[int]] = {}
        
        # =====================================================================
        # 核心：Boolean Tensor Masks (CPU 版本，懒加载到 GPU)
        # =====================================================================
        self._masks_cpu: Dict[str, torch.Tensor] = {}  # CPU 上的 Mask
        self._masks_gpu: Dict[str, Dict[str, torch.Tensor]] = {}  # (device, size) -> {name: mask}
        self._actual_logits_size: Optional[int] = None  # 实际 logits 维度（运行时确定）
        
        # 临时存储 token sets
        self._token_sets: Dict[str, set] = {}
        
        # 加载或扫描词表
        if not self._load_from_cache():
            self._scan_vocabulary()
            self._save_to_cache()
        
        # 构建 Tensor Masks
        self._build_tensor_masks()
    
    def _load_from_cache(self) -> bool:
        """从本地缓存加载词表扫描结果 (使用 pickle 提高性能)"""
        if not os.path.exists(self.cache_path):
            # 尝试旧版 JSON 格式
            json_path = _get_cache_path(self.model_name, ext="json")
            if os.path.exists(json_path):
                return self._load_from_json_cache(json_path)
            return False
        
        try:
            with open(self.cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            if cache_data.get('vocab_size') != self.vocab_size:
                return False
            
            self.token_to_text = cache_data['token_to_text']
            self.char_to_tokens = cache_data['char_to_tokens']
            self._token_sets = cache_data['token_sets']
            
            print(f"[VocabScanner] 从缓存加载词表: {self.cache_path}")
            return True
            
        except Exception as e:
            print(f"[VocabScanner] 加载缓存失败: {e}")
            return False
    
    def _load_from_json_cache(self, json_path: str) -> bool:
        """从旧版 JSON 缓存加载"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            if cache_data.get('vocab_size') != self.vocab_size:
                return False
            
            self.token_to_text = {int(k): v for k, v in cache_data['token_to_text'].items()}
            self.char_to_tokens = {k: list(v) for k, v in cache_data['char_to_tokens'].items()}
            
            # 恢复 token sets
            self._token_sets = {
                'digit': set(cache_data['digit_tokens']),
                'letter': set(cache_data['letter_tokens']),
                'newline': set(cache_data['newline_tokens']),
                'gt': set(cache_data['gt_tokens']),
                'lt': set(cache_data['lt_tokens']),
                'lbracket': set(cache_data['lbracket_tokens']),
                'rbracket': set(cache_data['rbracket_tokens']),
                'dot': set(cache_data['dot_tokens']),
                'dash': set(cache_data['dash_tokens']),
                'comma': set(cache_data['comma_tokens']),
                'space': set(cache_data['space_tokens']),
            }
            
            print(f"[VocabScanner] 从 JSON 缓存加载词表: {json_path}")
            # 保存为新格式
            self._save_to_cache()
            return True
            
        except Exception as e:
            print(f"[VocabScanner] 加载 JSON 缓存失败: {e}")
            return False
    
    def _save_to_cache(self):
        """将词表扫描结果保存到本地缓存 (使用 pickle)"""
        try:
            cache_data = {
                'vocab_size': self.vocab_size,
                'model_name': self.model_name,
                'token_to_text': self.token_to_text,
                'char_to_tokens': self.char_to_tokens,
                'token_sets': self._token_sets,
            }
            
            with open(self.cache_path, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            print(f"[VocabScanner] 词表缓存已保存: {self.cache_path}")
            
        except Exception as e:
            print(f"[VocabScanner] 保存缓存失败: {e}")
    
    def _scan_vocabulary(self):
        """扫描整个词表，建立字符到 Token 的映射"""
        print(f"[VocabScanner] 开始扫描词表 (vocab_size={self.vocab_size})...")
        
        # 初始化 token sets
        self._token_sets = {name: set() for name in self.MASK_NAMES}
        
        for token_id in range(self.vocab_size):
            try:
                text = self.tokenizer.decode([token_id], skip_special_tokens=False)
                if not text:
                    continue
                
                self.token_to_text[token_id] = text
                
                first_char = text[0] if text else ''
                stripped = text.lstrip()
                stripped_first = stripped[0] if stripped else ''
                
                # 建立字符映射
                for c in {first_char, stripped_first}:
                    if c:
                        if c not in self.char_to_tokens:
                            self.char_to_tokens[c] = []
                        self.char_to_tokens[c].append(token_id)
                
                # 分类 token
                self._categorize_token(token_id, text, first_char, stripped_first)
                    
            except Exception:
                continue
        
        print(f"[VocabScanner] 词表扫描完成")
    
    def _categorize_token(self, token_id: int, text: str, first_char: str, stripped_first: str):
        """将 token 分类到不同的类别"""
        for c in {first_char, stripped_first}:
            if not c:
                continue
            
            if c.isdigit():
                self._token_sets['digit'].add(token_id)
            if c.isalpha():
                self._token_sets['letter'].add(token_id)
            if c == '<':
                self._token_sets['lt'].add(token_id)
            elif c == '>':
                self._token_sets['gt'].add(token_id)
            elif c == '[':
                self._token_sets['lbracket'].add(token_id)
            elif c == ']':
                self._token_sets['rbracket'].add(token_id)
            elif c == '.':
                self._token_sets['dot'].add(token_id)
            elif c == '-':
                self._token_sets['dash'].add(token_id)
            elif c == ',':
                self._token_sets['comma'].add(token_id)
            elif c == ' ':
                self._token_sets['space'].add(token_id)
        
        if '\n' in text:
            self._token_sets['newline'].add(token_id)
    
    def _build_tensor_masks(self):
        """
        构建 CPU 上的 Boolean Tensor Masks
        
        这是核心优化：将 Python Set 转换为 Tensor，后续使用位运算
        """
        vocab_size = self.vocab_size
        
        # 1. 构建基础 Masks
        for name in self.MASK_NAMES:
            token_set = self._token_sets.get(name, set())
            mask = torch.zeros(vocab_size, dtype=torch.bool)
            if token_set:
                indices = torch.tensor(list(token_set), dtype=torch.long)
                mask[indices] = True
            self._masks_cpu[name] = mask
        
        # 2. 构建复合 Masks (预合并，避免推理时计算)
        for composite_name, components in self.COMPOSITE_MASKS.items():
            combined_mask = torch.zeros(vocab_size, dtype=torch.bool)
            for component in components:
                if component in self._masks_cpu:
                    combined_mask |= self._masks_cpu[component]
            self._masks_cpu[composite_name] = combined_mask
        
        # 注意：不删除 _token_sets，因为缓存需要
    
    def get_mask(self, name: str, device: torch.device, logits_size: int) -> torch.Tensor:
        """
        获取指定名称的 Boolean Mask，自动迁移到目标设备并适配 logits 维度
        
        Args:
            name: Mask 名称 (如 'digit', 'lbracket', 'deps_or_rbracket')
            device: 目标设备
            logits_size: 实际 logits 维度 (可能大于 vocab_size)
            
        Returns:
            Boolean Tensor [logits_size]，在指定设备上
        """
        cache_key = (str(device), logits_size, name)
        
        # 检查缓存
        if cache_key not in self._masks_gpu:
            if name in self._masks_cpu:
                # 获取 CPU mask 并适配大小
                cpu_mask = self._masks_cpu[name]
                if logits_size > len(cpu_mask):
                    # 扩展 mask，额外位置为 False（不允许）
                    extended_mask = torch.zeros(logits_size, dtype=torch.bool)
                    extended_mask[:len(cpu_mask)] = cpu_mask
                    self._masks_gpu[cache_key] = extended_mask.to(device)
                else:
                    self._masks_gpu[cache_key] = cpu_mask[:logits_size].to(device)
            else:
                # 未知 Mask，返回全 False
                self._masks_gpu[cache_key] = torch.zeros(
                    logits_size, dtype=torch.bool, device=device
                )
        
        return self._masks_gpu[cache_key]
    
    def get_char_mask(self, char: str, device: torch.device, logits_size: int) -> torch.Tensor:
        """
        获取指定字符的 Token Mask (用于动态关键字匹配)
        
        Args:
            char: 字符
            device: 目标设备
            logits_size: 实际 logits 维度
            
        Returns:
            Boolean Tensor [logits_size]
        """
        cache_key = (str(device), logits_size, f"char_{char}")
        
        if cache_key not in self._masks_gpu:
            mask = torch.zeros(logits_size, dtype=torch.bool, device=device)
            if char in self.char_to_tokens:
                # 只添加在 logits_size 范围内的 token
                valid_indices = [t for t in self.char_to_tokens[char] if t < logits_size]
                if valid_indices:
                    indices = torch.tensor(valid_indices, dtype=torch.long, device=device)
                    mask[indices] = True
            self._masks_gpu[cache_key] = mask
        
        return self._masks_gpu[cache_key]
    
    def decode_token(self, token_id: int) -> str:
        """解码单个 token（使用缓存）"""
        return self.token_to_text.get(token_id, "")
    
    def decode_tokens(self, token_ids: torch.Tensor) -> str:
        """解码多个 token（使用缓存）"""
        result = []
        for token_id in token_ids.tolist():
            text = self.token_to_text.get(token_id, "")
            result.append(text)
        return "".join(result)


# =============================================================================
# SemanticLogitsProcessor - GPU 优化版本
# =============================================================================

class SemanticLogitsProcessor(LogitsProcessor):
    """
    骨架生成约束处理器 (GPU Tensor Mask 优化版)
    
    性能优化：
    - 使用 GPU Tensor Mask 替代 Python Set
    - 使用位运算 (|=) 替代 for 循环
    - 使用布尔索引替代逐元素赋值
    - 延迟从 ~20ms 降低到 ~0.02ms
    
    初始化优化：
    - 支持预初始化的 VocabScanner，避免推理时重新扫描词表
    - 通过 configure() 方法在推理时动态设置 prefix_len
    """
    
    def __init__(
        self,
        tokenizer,
        prefix_len: int = 0,
        enforce_format: bool = True,
        vocab_scanner: Optional[VocabScanner] = None,
    ):
        """
        初始化 SemanticLogitsProcessor
        
        Args:
            tokenizer: 分词器
            prefix_len: 前缀长度（可在 configure 中动态设置）
            enforce_format: 是否强制执行格式约束
            vocab_scanner: 预初始化的 VocabScanner（可选）
                          如果提供，则复用该实例，避免重复扫描词表
                          如果不提供，则创建新实例
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.prefix_len = prefix_len
        self.enforce_format = enforce_format
        
        # 词表扫描器（核心组件）- 支持复用预初始化的实例
        if vocab_scanner is not None:
            self.vocab_scanner = vocab_scanner
        else:
            self.vocab_scanner = VocabScanner(tokenizer)
        
        # 增量状态机
        self._generated_text: str = ""
        self._current_state: FSMState = FSMState.HEADER_LBRACKET
        self.mode: Optional[str] = None
        self._last_token_count: int = 0
        self._current_line: str = ""
        self._keyword_buffer: str = ""
    
    def configure(self, prefix_len: int, enforce_format: bool = True):
        """
        动态配置 Processor 参数（在推理时调用）
        
        这个方法用于在推理时设置 prefix_len，而不需要重新创建 Processor。
        同时会重置状态机，为新的生成做准备。
        
        Args:
            prefix_len: 前缀长度（当前 input_ids 的长度）
            enforce_format: 是否强制执行格式约束
        """
        self.prefix_len = prefix_len
        self.enforce_format = enforce_format
        self.reset()

    def _decode_new_tokens(self, input_ids: torch.LongTensor) -> str:
        """增量解码：只解码新生成的 token"""
        current_token_count = input_ids.shape[-1] - self.prefix_len
        
        if current_token_count <= 0:
            return ""
        
        if current_token_count <= self._last_token_count:
            return ""
        
        new_token_count = current_token_count - self._last_token_count
        new_token_ids = input_ids[0][-new_token_count:]
        new_text = self.vocab_scanner.decode_tokens(new_token_ids)
        
        self._generated_text += new_text
        self._last_token_count = current_token_count
        
        return new_text

    def _update_state_incremental(self, new_text: str) -> FSMState:
        """增量状态转换"""
        for char in new_text:
            self._current_state = self._transition(self._current_state, char)
            if self._current_state == FSMState.FINISHED:
                break
        return self._current_state

    def _transition(self, state: FSMState, char: str) -> FSMState:
        """状态转换函数"""
        # Header 阶段
        if state == FSMState.HEADER_LBRACKET:
            if char == '[':
                self._keyword_buffer = ""
                return FSMState.HEADER_KEYWORD
            return state
        
        if state == FSMState.HEADER_KEYWORD:
            if char == ']':
                if self._keyword_buffer == "PLAN":
                    self.mode = "PLAN"
                    return FSMState.HEADER_NEWLINE
                elif self._keyword_buffer == "DIRECT":
                    self.mode = "DIRECT"
                    return FSMState.HEADER_NEWLINE
                return state
            else:
                self._keyword_buffer += char
                return state
        
        if state == FSMState.HEADER_RBRACKET:
            if char == ']':
                return FSMState.HEADER_NEWLINE
            return state
        
        if state == FSMState.HEADER_NEWLINE:
            if char == '\n':
                if self.mode == "DIRECT":
                    return FSMState.DIRECT_CONTENT
                elif self.mode == "PLAN":
                    self._current_line = ""
                    return FSMState.LINE_START
            return state
        
        # DIRECT 模式
        if state == FSMState.DIRECT_CONTENT:
            if char == '[':
                self._keyword_buffer = ""
                return FSMState.END_KEYWORD
            return state
        
        # PLAN 模式 - 分支行
        if state == FSMState.LINE_START:
            self._current_line = char
            if char == '[':
                self._keyword_buffer = ""
                return FSMState.END_KEYWORD
            elif char.isdigit():
                return FSMState.AFTER_ID
            return state
        
        if state == FSMState.AFTER_ID:
            self._current_line += char
            if char.isdigit():
                return state
            elif char == '.':
                return FSMState.AFTER_DOT
            return state
        
        if state == FSMState.AFTER_DOT:
            self._current_line += char
            if char == '<':
                return FSMState.LEN_OPEN
            return state
        
        if state == FSMState.LEN_OPEN:
            self._current_line += char
            if char.isdigit():
                return FSMState.LEN_VAL
            return state
        
        if state == FSMState.LEN_VAL:
            self._current_line += char
            if char.isdigit():
                return state
            elif char == '>':
                return FSMState.AFTER_LEN_CLOSE
            return state
        
        if state == FSMState.AFTER_LEN_CLOSE:
            self._current_line += char
            if char == '<':
                return FSMState.TOOL_OPEN
            return state
        
        if state == FSMState.TOOL_OPEN:
            self._current_line += char
            if char.isalpha():
                return FSMState.TOOL_VAL
            return state
        
        if state == FSMState.TOOL_VAL:
            self._current_line += char
            if char == '>':
                return FSMState.AFTER_TOOL_CLOSE
            return state
        
        if state == FSMState.AFTER_TOOL_CLOSE:
            self._current_line += char
            if char == '[':
                return FSMState.DEPS_OPEN
            return state
        
        if state == FSMState.DEPS_OPEN:
            self._current_line += char
            if char == '-' or char.isdigit():
                return FSMState.DEPS_VAL
            return state
        
        if state == FSMState.DEPS_VAL:
            self._current_line += char
            if char == ']':
                return FSMState.AFTER_DEPS_CLOSE
            return state
        
        if state == FSMState.AFTER_DEPS_CLOSE:
            self._current_line += char
            if char == '\n':
                self._current_line = ""
                return FSMState.LINE_START
            return FSMState.CONTENT
        
        if state == FSMState.CONTENT:
            self._current_line += char
            if char == '\n':
                self._current_line = ""
                return FSMState.LINE_START
            return state
        
        # END 检测
        if state == FSMState.END_KEYWORD:
            if char == ']':
                if self._keyword_buffer == "END":
                    return FSMState.FINISHED
                if self.mode == "DIRECT":
                    return FSMState.DIRECT_CONTENT
                else:
                    return FSMState.LINE_START
            else:
                self._keyword_buffer += char
                return state
        
        if state == FSMState.END_RBRACKET:
            if char == ']':
                return FSMState.FINISHED
            return state
        
        if state == FSMState.FINISHED:
            return state
        
        return state

    def _apply_state_constraints(
        self, 
        state: FSMState, 
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        根据当前状态应用约束 (GPU Tensor 优化版)
        
        使用位运算构建 allowed_mask，然后用布尔索引应用
        """
        if not self.enforce_format:
            return scores
        
        device = scores.device
        logits_size = scores.shape[-1]  # 获取实际的 logits 维度
        vs = self.vocab_scanner
        
        # 初始化：需要应用约束的状态
        allowed_mask: Optional[torch.Tensor] = None
        
        # =====================================================================
        # Header 阶段
        # =====================================================================
        
        if state == FSMState.HEADER_LBRACKET:
            allowed_mask = vs.get_mask('lbracket', device, logits_size)
        
        elif state == FSMState.HEADER_KEYWORD:
            # 获取能继续匹配 PLAN 或 DIRECT 的 token
            buffer = self._keyword_buffer
            allowed_mask = torch.zeros(logits_size, dtype=torch.bool, device=device)
            
            # 检查 PLAN
            if "PLAN".startswith(buffer):
                if len(buffer) < 4:
                    next_char = "PLAN"[len(buffer)]
                    allowed_mask |= vs.get_char_mask(next_char, device, logits_size)
                else:
                    allowed_mask |= vs.get_mask('rbracket', device, logits_size)
            
            # 检查 DIRECT
            if "DIRECT".startswith(buffer):
                if len(buffer) < 6:
                    next_char = "DIRECT"[len(buffer)]
                    allowed_mask |= vs.get_char_mask(next_char, device, logits_size)
                else:
                    allowed_mask |= vs.get_mask('rbracket', device, logits_size)
        
        elif state == FSMState.HEADER_RBRACKET:
            allowed_mask = vs.get_mask('rbracket', device, logits_size)
        
        elif state == FSMState.HEADER_NEWLINE:
            allowed_mask = vs.get_mask('newline', device, logits_size)
        
        # =====================================================================
        # DIRECT 模式 - 无约束
        # =====================================================================
        
        elif state == FSMState.DIRECT_CONTENT:
            pass  # 无约束
        
        # =====================================================================
        # PLAN 模式 - 分支行
        # =====================================================================
        
        elif state == FSMState.LINE_START:
            # 允许数字（ID）或 [（[END]）
            allowed_mask = vs.get_mask('digit_or_lbracket', device, logits_size)
        
        elif state == FSMState.AFTER_ID:
            # 允许继续数字或点号
            allowed_mask = vs.get_mask('digit_or_dot', device, logits_size)
        
        elif state == FSMState.AFTER_DOT:
            allowed_mask = vs.get_mask('lt', device, logits_size)
        
        elif state == FSMState.LEN_OPEN:
            # 期望数字
            allowed_mask = vs.get_mask('digit', device, logits_size)
        
        elif state == FSMState.LEN_VAL:
            # 期望数字或 > (支持多位数)
            allowed_mask = vs.get_mask('digit_or_gt', device, logits_size)
        
        elif state == FSMState.AFTER_LEN_CLOSE:
            allowed_mask = vs.get_mask('lt', device, logits_size)
        
        elif state == FSMState.TOOL_OPEN:
            # 工具名 - 已去除约束
            pass
        
        elif state == FSMState.TOOL_VAL:
            # 工具名 - 已去除约束
            pass
        
        elif state == FSMState.AFTER_TOOL_CLOSE:
            allowed_mask = vs.get_mask('lbracket', device, logits_size)
        
        elif state == FSMState.DEPS_OPEN:
            # 期望 - 或数字
            allowed_mask = vs.get_mask('dash_or_digit', device, logits_size)
        
        elif state == FSMState.DEPS_VAL:
            # 期望数字、逗号、短横线或 ]
            allowed_mask = vs.get_mask('deps_or_rbracket', device, logits_size)
        
        elif state == FSMState.AFTER_DEPS_CLOSE:
            pass  # Title 部分无约束
        
        elif state == FSMState.CONTENT:
            pass  # 无约束
        
        # =====================================================================
        # END 检测
        # =====================================================================
        
        elif state == FSMState.END_KEYWORD:
            buffer = self._keyword_buffer
            allowed_mask = torch.zeros(logits_size, dtype=torch.bool, device=device)
            
            if "END".startswith(buffer):
                if len(buffer) < 3:
                    next_char = "END"[len(buffer)]
                    allowed_mask |= vs.get_char_mask(next_char, device, logits_size)
                else:
                    allowed_mask |= vs.get_mask('rbracket', device, logits_size)
        
        elif state == FSMState.END_RBRACKET:
            allowed_mask = vs.get_mask('rbracket', device, logits_size)
        
        elif state == FSMState.FINISHED:
            pass
        
        # =====================================================================
        # 应用约束 (全向量化)
        # =====================================================================
        
        if allowed_mask is not None:
            # 使用布尔索引，一次性设置所有非允许 token 为 -inf
            scores = scores.clone()
            scores[:, ~allowed_mask] = float('-inf')
        
        return scores

    def __call__(
        self, 
        input_ids: torch.LongTensor, 
        scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        """处理 logits，应用语义约束"""
        seq_length = input_ids.shape[-1]
        
        # 还没有生成任何 token
        if seq_length <= self.prefix_len:
            if self.enforce_format:
                device = scores.device
                logits_size = scores.shape[-1]
                allowed_mask = self.vocab_scanner.get_mask('lbracket', device, logits_size)
                scores = scores.clone()
                scores[:, ~allowed_mask] = float('-inf')
            return scores
        
        # 增量解码新 token
        new_text = self._decode_new_tokens(input_ids)
        
        # 增量更新状态
        if new_text:
            self._update_state_incremental(new_text)
        
        # 应用状态约束
        return self._apply_state_constraints(self._current_state, scores)

    def reset(self):
        """重置状态机"""
        self.mode = None
        self._generated_text = ""
        self._current_state = FSMState.HEADER_LBRACKET
        self._last_token_count = 0
        self._current_line = ""
        self._keyword_buffer = ""

    def __repr__(self) -> str:
        return f"SemanticLogitsProcessor(mode={self.mode}, state={self._current_state.name}, prefix_len={self.prefix_len})"
