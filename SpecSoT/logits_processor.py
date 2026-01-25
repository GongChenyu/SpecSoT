# coding=utf-8
"""
Logits Processors for SpecSoT (词表扫描版本 v2)

该模块定义了用于控制生成过程的 Logits Processors：

SemanticLogitsProcessor: 骨架生成约束 (基于词表扫描的通用实现)
   - 支持格式：
     - [DIRECT]...[END] 直接回答模式
     - [PLAN] 后接多个分支行，每行格式：ID.<Length><Tool>[-]Title
   - 通过词表扫描解决不同分词器的兼容性问题
   - 使用增量字符串匹配，而非硬编码 Token ID

核心设计理念：
   1. 初始化时扫描整个词表，建立"字符->Token集合"的映射
   2. FSM 状态机只关心"下一个应该是什么字符"
   3. 使用增量字符串匹配处理多 token 关键字

简化的骨架格式（去除不必要的空格）：
   [PLAN]
   1.<200><Search>[-]任务描述
   2.<150><None>[-]任务描述
   [END]
"""

import re
from enum import IntEnum
from typing import Optional, Set, Dict, List, Tuple
import torch
from transformers import LogitsProcessor


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
# Vocabulary Scanner - 词表扫描器 (一劳永逸解决分词器差异)
# =============================================================================

class VocabScanner:
    """
    词表扫描器：在初始化时扫描整个词表，建立通用的字符到 Token 的映射
    
    核心设计：
    1. 一次性遍历整个词表（约100-500ms）
    2. 建立多维度的映射关系
    3. 推理时 O(1) 查询
    
    这种方式完全解决了不同分词器（Llama/Vicuna/Qwen/Mistral）的 Token 差异问题
    """
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        
        # 核心映射表
        self.char_to_tokens: Dict[str, Set[int]] = {}      # 首字符 -> Token集合
        self.token_to_text: Dict[int, str] = {}            # Token ID -> 解码文本
        
        # 特殊字符类别
        self.digit_tokens: Set[int] = set()       # 以数字开头的 token
        self.letter_tokens: Set[int] = set()      # 以字母开头的 token
        self.newline_tokens: Set[int] = set()     # 包含换行的 token
        self.gt_tokens: Set[int] = set()          # 以 > 开头的 token
        self.lt_tokens: Set[int] = set()          # 以 < 开头的 token
        self.lbracket_tokens: Set[int] = set()    # 以 [ 开头的 token
        self.rbracket_tokens: Set[int] = set()    # 以 ] 开头的 token
        self.dot_tokens: Set[int] = set()         # 以 . 开头的 token
        self.dash_tokens: Set[int] = set()        # 以 - 开头的 token
        self.comma_tokens: Set[int] = set()       # 以 , 开头的 token
        self.space_tokens: Set[int] = set()       # 以空格开头的 token
        
        # 执行词表扫描
        self._scan_vocabulary()
    
    def _scan_vocabulary(self):
        """
        扫描整个词表，建立字符到 Token 的映射
        
        这是解决分词器差异的核心：不管是 Llama、Vicuna 还是 Qwen，
        我们都通过解码每个 token 来确定它代表什么字符
        """
        for token_id in range(self.vocab_size):
            try:
                # 解码单个 token
                text = self.tokenizer.decode([token_id], skip_special_tokens=False)
                
                if not text:
                    continue
                
                self.token_to_text[token_id] = text
                
                # 获取第一个非空字符（处理 SentencePiece 的空格前缀）
                first_char = text[0] if text else ''
                stripped = text.lstrip()
                stripped_first = stripped[0] if stripped else ''
                
                # 建立首字符映射
                if first_char:
                    if first_char not in self.char_to_tokens:
                        self.char_to_tokens[first_char] = set()
                    self.char_to_tokens[first_char].add(token_id)
                
                # 对于带空格前缀的 token，也记录去除空格后的首字符
                # 这对于 SentencePiece 分词器非常重要
                if stripped_first and stripped_first != first_char:
                    if stripped_first not in self.char_to_tokens:
                        self.char_to_tokens[stripped_first] = set()
                    self.char_to_tokens[stripped_first].add(token_id)
                
                # 分类到特殊类别
                self._categorize_token(token_id, text, first_char, stripped_first)
                    
            except Exception:
                continue
    
    def _categorize_token(self, token_id: int, text: str, first_char: str, stripped_first: str):
        """将 token 分类到不同的类别"""
        # 检查所有可能的首字符
        check_chars = {first_char, stripped_first}
        
        for c in check_chars:
            if not c:
                continue
                
            if c.isdigit():
                self.digit_tokens.add(token_id)
            
            if c.isalpha():
                self.letter_tokens.add(token_id)
            
            if c == '<':
                self.lt_tokens.add(token_id)
            elif c == '>':
                self.gt_tokens.add(token_id)
            elif c == '[':
                self.lbracket_tokens.add(token_id)
            elif c == ']':
                self.rbracket_tokens.add(token_id)
            elif c == '.':
                self.dot_tokens.add(token_id)
            elif c == '-':
                self.dash_tokens.add(token_id)
            elif c == ',':
                self.comma_tokens.add(token_id)
            elif c == ' ':
                self.space_tokens.add(token_id)
        
        # 换行符需要特殊处理：只要包含换行就算
        if '\n' in text:
            self.newline_tokens.add(token_id)
    
    def get_tokens_matching_char(self, char: str) -> Set[int]:
        """获取所有以指定字符开头的 token（公开接口）"""
        return self.char_to_tokens.get(char, set()).copy()
    
    def get_tokens_containing_string(self, target: str) -> Set[int]:
        """获取解码后包含指定字符串的所有 token"""
        result = set()
        for token_id, text in self.token_to_text.items():
            if target in text:
                result.add(token_id)
        return result
    
    def decode_token(self, token_id: int) -> str:
        """解码单个 token（使用缓存）"""
        return self.token_to_text.get(token_id, "")


# =============================================================================
# SemanticLogitsProcessor - 基于词表扫描的通用实现
# =============================================================================

class SemanticLogitsProcessor(LogitsProcessor):
    """
    骨架生成约束处理器 (词表扫描版本)
    
    简化的骨架格式：
    
    格式一（直接回答）：
    ```
    [DIRECT]
    (内容，无约束)
    [END]
    ```
    
    格式二（规划模式）：
    ```
    [PLAN]
    1.<200><Search>[-]搜索最新的篮球比赛规则
    2.<150><None>[-]分析投篮动作的物理原理
    3.<300><None>[-]总结提高命中率的训练技巧
    [END]
    ```
    
    特点：
    - 使用词表扫描，一劳永逸解决分词器差异
    - 简化格式，减少不必要的空格
    - 支持多位数长度预测 (1-999+)
    """
    
    def __init__(
        self,
        tokenizer,
        prefix_len: int,
        enforce_format: bool = True,
    ):
        """
        初始化语义约束处理器
        
        Args:
            tokenizer: 分词器
            prefix_len: 输入前缀长度
            enforce_format: 是否强制执行格式约束
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.prefix_len = prefix_len
        self.enforce_format = enforce_format
        
        # 记录模式：None, 'PLAN', 'DIRECT'
        self.mode: Optional[str] = None
        
        # 词表扫描器（核心组件）
        self.vocab_scanner = VocabScanner(tokenizer)
        
        # 预构建常用字符集合（加速推理）
        self._build_char_sets()
    
    def _build_char_sets(self):
        """预构建常用的字符集合"""
        vs = self.vocab_scanner
        
        # 工具名允许的字符：字母、数字、下划线
        self.tool_tokens = vs.letter_tokens.copy()
        for c in "0123456789_":
            self.tool_tokens.update(vs.get_tokens_matching_char(c))
        
        # 依赖允许的字符：数字、逗号、短横线
        self.deps_tokens = vs.digit_tokens.copy()
        self.deps_tokens.update(vs.dash_tokens)
        self.deps_tokens.update(vs.comma_tokens)
        
        # 数字 token（用于长度值）
        self.digit_tokens = vs.digit_tokens.copy()
        
        # 字母 token（用于关键字匹配）
        self.letter_tokens = vs.letter_tokens.copy()

    def _get_state_from_text(self, full_text: str) -> FSMState:
        """
        根据完整生成文本确定当前 FSM 状态
        
        核心状态判定逻辑：通过分析已生成的文本来确定约束
        """
        if not full_text:
            return FSMState.HEADER_LBRACKET
        
        # =====================================================================
        # Header 阶段检测
        # =====================================================================
        
        if full_text == "[":
            return FSMState.HEADER_KEYWORD
        
        # 检查是否是部分关键字
        if full_text.startswith("[") and not full_text.startswith("[PLAN]") and not full_text.startswith("[DIRECT]"):
            content = full_text[1:]
            
            # 完整匹配检查
            if content == "PLAN" or content == "DIRECT":
                return FSMState.HEADER_RBRACKET
            
            # 前缀匹配检查
            if "PLAN".startswith(content) or "DIRECT".startswith(content):
                return FSMState.HEADER_KEYWORD
        
        # Header 完成但没换行
        if full_text == "[PLAN]" or full_text == "[DIRECT]":
            return FSMState.HEADER_NEWLINE
        
        # =====================================================================
        # 确定模式
        # =====================================================================
        
        if full_text.startswith("[DIRECT]"):
            self.mode = "DIRECT"
        elif full_text.startswith("[PLAN]"):
            self.mode = "PLAN"
        
        # =====================================================================
        # DIRECT 模式
        # =====================================================================
        
        if self.mode == "DIRECT":
            if "[END]" in full_text:
                return FSMState.FINISHED
            return FSMState.DIRECT_CONTENT
        
        # =====================================================================
        # PLAN 模式
        # =====================================================================
        
        if self.mode == "PLAN":
            if "[END]" in full_text:
                return FSMState.FINISHED
            
            # 获取当前行
            lines = full_text.split('\n')
            current_line = lines[-1] if lines else ""
            
            if not current_line:
                return FSMState.LINE_START
            
            # 检查是否正在输入 [END]
            if current_line.startswith("["):
                content = current_line[1:]
                if not content:
                    return FSMState.END_KEYWORD
                if content == "END":
                    return FSMState.END_RBRACKET
                if "END".startswith(content):
                    return FSMState.END_KEYWORD
            
            # 解析分支行
            return self._parse_branch_line_state(current_line)
        
        return FSMState.HEADER_LBRACKET

    def _parse_branch_line_state(self, line: str) -> FSMState:
        """
        解析分支行，确定当前状态
        
        简化格式：ID.<Length><Tool>[-]Title
        例如：1.<200><Search>[-]搜索内容
        
        支持多位数长度 (如 <127>, <1500>)
        """
        # 1. 检查 ID（支持多位数 ID）
        id_match = re.match(r'^(\d+)', line)
        if not id_match:
            return FSMState.LINE_START
        
        pos = id_match.end()
        remaining = line[pos:]
        
        # 2. 检查点号
        if not remaining:
            return FSMState.AFTER_ID
        
        if remaining[0] != '.':
            return FSMState.AFTER_ID
        
        remaining = remaining[1:]
        
        # 3. 检查 <Length>
        if not remaining:
            return FSMState.AFTER_DOT
        
        if remaining[0] != '<':
            return FSMState.AFTER_DOT
        
        # 找到第一个 >
        gt_pos = remaining.find('>')
        if gt_pos == -1:
            # 还在 <...> 内部
            content = remaining[1:]
            if not content:
                return FSMState.LEN_OPEN
            # 验证内容是否全为数字
            if content.isdigit():
                return FSMState.LEN_VAL
            return FSMState.LEN_VAL  # 继续期望数字或 >
        
        remaining = remaining[gt_pos + 1:]
        
        # 4. 检查 <Tool>
        if not remaining:
            return FSMState.AFTER_LEN_CLOSE
        
        if remaining[0] != '<':
            return FSMState.AFTER_LEN_CLOSE
        
        gt_pos = remaining.find('>')
        if gt_pos == -1:
            content = remaining[1:]
            if not content:
                return FSMState.TOOL_OPEN
            return FSMState.TOOL_VAL
        
        remaining = remaining[gt_pos + 1:]
        
        # 5. 检查 [Deps]
        if not remaining:
            return FSMState.AFTER_TOOL_CLOSE
        
        if remaining[0] != '[':
            return FSMState.AFTER_TOOL_CLOSE
        
        rb_pos = remaining.find(']')
        if rb_pos == -1:
            content = remaining[1:]
            if not content:
                return FSMState.DEPS_OPEN
            return FSMState.DEPS_VAL
        
        remaining = remaining[rb_pos + 1:]
        
        # 6. Title 部分
        if not remaining:
            return FSMState.AFTER_DEPS_CLOSE
        
        return FSMState.CONTENT

    def _apply_state_constraints(
        self, 
        state: FSMState, 
        scores: torch.FloatTensor,
        full_text: str = "",
    ) -> torch.FloatTensor:
        """根据当前状态应用约束"""
        if not self.enforce_format:
            return scores
        
        scores = scores.clone()
        vs = self.vocab_scanner
        
        # =====================================================================
        # Header 阶段
        # =====================================================================
        
        if state == FSMState.HEADER_LBRACKET:
            self._mask_except(scores, vs.lbracket_tokens)
        
        elif state == FSMState.HEADER_KEYWORD:
            # 获取能继续匹配 PLAN 或 DIRECT 的 token
            allowed = set()
            
            content = full_text[1:] if full_text.startswith("[") else ""
            
            # 检查 PLAN
            if "PLAN".startswith(content):
                next_char = "PLAN"[len(content)] if len(content) < 4 else ""
                if next_char:
                    allowed.update(vs.get_tokens_matching_char(next_char))
            
            # 检查 DIRECT
            if "DIRECT".startswith(content):
                next_char = "DIRECT"[len(content)] if len(content) < 6 else ""
                if next_char:
                    allowed.update(vs.get_tokens_matching_char(next_char))
            
            if allowed:
                self._mask_except(scores, allowed)
        
        elif state == FSMState.HEADER_RBRACKET:
            # 允许 ] 及其组合（如 ]\n）
            allowed = vs.rbracket_tokens.copy()
            self._mask_except(scores, allowed)
        
        elif state == FSMState.HEADER_NEWLINE:
            self._mask_except(scores, vs.newline_tokens)
        
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
            allowed = vs.digit_tokens.copy()
            allowed.update(vs.lbracket_tokens)
            self._mask_except(scores, allowed)
        
        elif state == FSMState.AFTER_ID:
            self._mask_except(scores, vs.dot_tokens)
        
        elif state == FSMState.AFTER_DOT:
            self._mask_except(scores, vs.lt_tokens)
        
        elif state == FSMState.LEN_OPEN:
            # 期望数字
            self._mask_except(scores, vs.digit_tokens)
        
        elif state == FSMState.LEN_VAL:
            # 期望数字或 > (支持多位数)
            allowed = vs.digit_tokens.copy()
            allowed.update(vs.gt_tokens)
            self._mask_except(scores, allowed)
        
        elif state == FSMState.AFTER_LEN_CLOSE:
            self._mask_except(scores, vs.lt_tokens)
        
        elif state == FSMState.TOOL_OPEN:
            # 期望字母（工具名首字符）
            self._mask_except(scores, vs.letter_tokens)
        
        elif state == FSMState.TOOL_VAL:
            # 期望字母、数字、下划线或 >
            allowed = self.tool_tokens.copy()
            allowed.update(vs.gt_tokens)
            self._mask_except(scores, allowed)
        
        elif state == FSMState.AFTER_TOOL_CLOSE:
            self._mask_except(scores, vs.lbracket_tokens)
        
        elif state == FSMState.DEPS_OPEN:
            # 期望 - 或数字
            allowed = vs.dash_tokens.copy()
            allowed.update(vs.digit_tokens)
            self._mask_except(scores, allowed)
        
        elif state == FSMState.DEPS_VAL:
            # 期望数字、逗号、短横线或 ]
            allowed = self.deps_tokens.copy()
            allowed.update(vs.rbracket_tokens)
            self._mask_except(scores, allowed)
        
        elif state == FSMState.AFTER_DEPS_CLOSE:
            pass  # Title 部分无约束
        
        elif state == FSMState.CONTENT:
            pass  # 无约束
        
        # =====================================================================
        # END 检测
        # =====================================================================
        
        elif state == FSMState.END_KEYWORD:
            # 获取当前行
            lines = full_text.split('\n')
            current_line = lines[-1] if lines else ""
            content = current_line[1:] if current_line.startswith("[") else ""
            
            allowed = set()
            if "END".startswith(content):
                next_char = "END"[len(content)] if len(content) < 3 else ""
                if next_char:
                    allowed.update(vs.get_tokens_matching_char(next_char))
            
            if allowed:
                self._mask_except(scores, allowed)
        
        elif state == FSMState.END_RBRACKET:
            self._mask_except(scores, vs.rbracket_tokens)
        
        elif state == FSMState.FINISHED:
            pass
        
        return scores

    def _mask_except(self, scores: torch.FloatTensor, allowed_ids: Set[int]):
        """将除 allowed_ids 之外的所有 token 设为 -inf"""
        if not allowed_ids:
            return
        
        mask = torch.ones(scores.shape[-1], dtype=torch.bool, device=scores.device)
        for token_id in allowed_ids:
            if 0 <= token_id < scores.shape[-1]:
                mask[token_id] = False
        scores[:, mask] = float('-inf')

    def __call__(
        self, 
        input_ids: torch.LongTensor, 
        scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        """处理 logits，应用语义约束"""
        seq_length = input_ids.shape[-1]
        
        # 还没有生成任何 token
        if seq_length <= self.prefix_len:
            vs = self.vocab_scanner
            if self.enforce_format:
                scores = scores.clone()
                self._mask_except(scores, vs.lbracket_tokens)
            return scores
        
        # 解码已生成的文本
        generated_ids = input_ids[0][self.prefix_len:]
        full_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # 确定当前状态
        state = self._get_state_from_text(full_text)
        
        # 应用状态约束
        return self._apply_state_constraints(state, scores, full_text)

    def reset(self):
        """重置状态机"""
        self.mode = None

    def __repr__(self) -> str:
        return f"SemanticLogitsProcessor(mode={self.mode}, prefix_len={self.prefix_len})"
