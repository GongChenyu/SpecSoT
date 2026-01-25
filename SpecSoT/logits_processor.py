# coding=utf-8
"""
Logits Processors for SpecSoT (词表扫描版本 v3 - 增量优化)

该模块定义了用于控制生成过程的 Logits Processors：

SemanticLogitsProcessor: 骨架生成约束 (基于词表扫描 + 增量状态机)
   - 支持格式：
     - [DIRECT]...[END] 直接回答模式
     - [PLAN] 后接多个分支行，每行格式：ID.<Length><Tool>[-]Title
   - 通过词表扫描解决不同分词器的兼容性问题
   - 使用增量解码和状态持久化，避免 O(N²) 性能问题

核心设计理念 (v3 优化)：
   1. 初始化时扫描整个词表，建立"字符->Token集合"的映射
   2. 增量解码：维护 generated_text 缓存，每次只解码新 token
   3. 状态持久化：维护 current_state，根据新字符增量转换状态
   4. 时间复杂度从 O(N²) 降低到 O(N)

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
    
    def decode_tokens(self, token_ids: torch.Tensor) -> str:
        """
        解码多个 token（使用缓存，避免调用 tokenizer.decode）
        
        这是增量解码的核心方法，直接从缓存中拼接文本
        比调用 tokenizer.decode 快得多
        
        Args:
            token_ids: token ID 张量
            
        Returns:
            解码后的文本
        """
        result = []
        for token_id in token_ids.tolist():
            text = self.token_to_text.get(token_id, "")
            result.append(text)
        return "".join(result)


# =============================================================================
# SemanticLogitsProcessor - 增量状态机实现 (v3)
# =============================================================================

class SemanticLogitsProcessor(LogitsProcessor):
    """
    骨架生成约束处理器 (词表扫描 + 增量状态机版本)
    
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
    
    性能优化 (v3)：
    - 增量解码：维护 generated_text 缓存，每次只解码新 token，O(1)
    - 状态持久化：维护 current_state，增量转换，避免从头解析
    - 总时间复杂度从 O(N²) 降低到 O(N)
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
        
        # 词表扫描器（核心组件）
        self.vocab_scanner = VocabScanner(tokenizer)
        
        # 预构建常用字符集合（加速推理）
        self._build_char_sets()
        
        # =====================================================================
        # 增量状态机 (v3 优化核心)
        # =====================================================================
        
        # 已生成文本缓存（增量解码）
        self._generated_text: str = ""
        
        # 当前 FSM 状态
        self._current_state: FSMState = FSMState.HEADER_LBRACKET
        
        # 模式：None, 'PLAN', 'DIRECT'
        self.mode: Optional[str] = None
        
        # 上次处理的 token 数量（用于检测新 token）
        self._last_token_count: int = 0
        
        # 当前行缓存（用于分支行解析）
        self._current_line: str = ""
        
        # 关键字匹配缓冲区 (用于 PLAN/DIRECT/END)
        self._keyword_buffer: str = ""
    
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

    def _decode_new_tokens(self, input_ids: torch.LongTensor) -> str:
        """
        增量解码：只解码新生成的 token
        
        这是 v3 优化的核心：避免每次都解码整个序列
        时间复杂度从 O(N) 降低到 O(1)（假设每次只生成少量 token）
        
        Args:
            input_ids: 当前完整的 input_ids
            
        Returns:
            新增的文本
        """
        current_token_count = input_ids.shape[-1] - self.prefix_len
        
        if current_token_count <= 0:
            return ""
        
        if current_token_count <= self._last_token_count:
            # 没有新 token（可能是重置或回退场景）
            return ""
        
        # 计算新增的 token 数量
        new_token_count = current_token_count - self._last_token_count
        
        # 只解码新增的 token
        new_token_ids = input_ids[0][-new_token_count:]
        new_text = self.vocab_scanner.decode_tokens(new_token_ids)
        
        # 更新缓存
        self._generated_text += new_text
        self._last_token_count = current_token_count
        
        return new_text

    def _update_state_incremental(self, new_text: str) -> FSMState:
        """
        增量状态转换：根据新生成的字符更新状态
        
        这是 v3 优化的核心：不需要从头解析整个文本
        只需要根据当前状态和新字符确定下一个状态
        
        Args:
            new_text: 新增的文本
            
        Returns:
            更新后的状态
        """
        for char in new_text:
            self._current_state = self._transition(self._current_state, char)
            
            # 如果已完成，提前退出
            if self._current_state == FSMState.FINISHED:
                break
        
        return self._current_state

    def _transition(self, state: FSMState, char: str) -> FSMState:
        """
        状态转换函数：根据当前状态和输入字符确定下一个状态
        
        这是标准的 FSM 状态转换实现
        
        Args:
            state: 当前状态
            char: 输入字符
            
        Returns:
            下一个状态
        """
        # =====================================================================
        # Header 阶段
        # =====================================================================
        
        if state == FSMState.HEADER_LBRACKET:
            if char == '[':
                self._keyword_buffer = ""
                return FSMState.HEADER_KEYWORD
            return state  # 忽略无效字符
        
        if state == FSMState.HEADER_KEYWORD:
            if char == ']':
                # 检查关键字
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
        
        # =====================================================================
        # DIRECT 模式
        # =====================================================================
        
        if state == FSMState.DIRECT_CONTENT:
            # 检测 [END]
            if char == '[':
                self._keyword_buffer = ""
                return FSMState.END_KEYWORD
            return state
        
        # =====================================================================
        # PLAN 模式 - 分支行
        # =====================================================================
        
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
                return state  # 多位数 ID
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
                return state  # 多位数长度
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
            return state  # 继续工具名
        
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
            return state  # 继续依赖内容
        
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
        
        # =====================================================================
        # END 检测
        # =====================================================================
        
        if state == FSMState.END_KEYWORD:
            if char == ']':
                if self._keyword_buffer == "END":
                    return FSMState.FINISHED
                # 不是 END，回到对应模式
                if self.mode == "DIRECT":
                    return FSMState.DIRECT_CONTENT
                else:
                    # 可能是新分支行的开始（虽然不太可能）
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
            buffer = self._keyword_buffer
            
            # 检查 PLAN
            if "PLAN".startswith(buffer):
                next_char = "PLAN"[len(buffer)] if len(buffer) < 4 else ""
                if next_char:
                    allowed.update(vs.get_tokens_matching_char(next_char))
                elif len(buffer) == 4:
                    # 已经是 PLAN，允许 ]
                    allowed.update(vs.rbracket_tokens)
            
            # 检查 DIRECT
            if "DIRECT".startswith(buffer):
                next_char = "DIRECT"[len(buffer)] if len(buffer) < 6 else ""
                if next_char:
                    allowed.update(vs.get_tokens_matching_char(next_char))
                elif len(buffer) == 6:
                    # 已经是 DIRECT，允许 ]
                    allowed.update(vs.rbracket_tokens)
            
            if allowed:
                self._mask_except(scores, allowed)
        
        elif state == FSMState.HEADER_RBRACKET:
            self._mask_except(scores, vs.rbracket_tokens)
        
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
            # 允许继续数字或点号
            allowed = vs.digit_tokens.copy()
            allowed.update(vs.dot_tokens)
            self._mask_except(scores, allowed)
        
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
            buffer = self._keyword_buffer
            allowed = set()
            
            if "END".startswith(buffer):
                next_char = "END"[len(buffer)] if len(buffer) < 3 else ""
                if next_char:
                    allowed.update(vs.get_tokens_matching_char(next_char))
                elif len(buffer) == 3:
                    # 已经是 END，允许 ]
                    allowed.update(vs.rbracket_tokens)
            
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
        """
        处理 logits，应用语义约束
        
        v3 优化：使用增量解码和状态持久化，避免 O(N²) 性能问题
        """
        seq_length = input_ids.shape[-1]
        
        # 还没有生成任何 token
        if seq_length <= self.prefix_len:
            vs = self.vocab_scanner
            if self.enforce_format:
                scores = scores.clone()
                self._mask_except(scores, vs.lbracket_tokens)
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
