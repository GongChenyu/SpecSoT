# coding=utf-8
"""
Logits Processors for SpecSoT

该模块定义了用于控制生成过程的 Logits Processors：

SemanticLogitsProcessor: 骨架生成约束 (FSM 状态机)
   - 支持格式：ID. <Length> <Tool> [Deps] Title
   - 基于有限状态机(FSM)实现精确的格式控制

"""

import re
from enum import IntEnum
from typing import Optional, Set, Dict
import torch
from transformers import LogitsProcessor


# =============================================================================
# FSM State Definitions for Protocol
# =============================================================================

class FSMState(IntEnum):
    """
    FSM 状态定义，用于解析骨架格式：ID. <Length> <Tool> [Deps] Title
    
    状态流转：
    LINE_START -> AFTER_ID -> AFTER_DOT -> LEN_OPEN -> LEN_VAL -> LEN_CLOSE
    -> TOOL_OPEN -> TOOL_VAL -> TOOL_CLOSE -> DEPS_OPEN -> DEPS_VAL 
    -> DEPS_CLOSE -> CONTENT -> (遇到换行符) -> LINE_START
    """
    LINE_START = 0    # 行首，期望数字 ID
    AFTER_ID = 1      # ID 后，期望 .
    AFTER_DOT = 2     # . 后，期望空格和 <
    LEN_OPEN = 3      # < 打开（长度），期望数字
    LEN_VAL = 4       # 长度值中，期望数字或 >
    LEN_CLOSE = 5     # > 关闭长度，期望空格和 <
    TOOL_OPEN = 6     # < 打开（工具），期望字母
    TOOL_VAL = 7      # 工具值中，期望字母或 >
    TOOL_CLOSE = 8    # > 关闭工具，期望空格和 [
    DEPS_OPEN = 9     # [ 打开（依赖），期望 -
    DEPS_VAL = 10     # 依赖值中，期望 - 或 ]
    DEPS_CLOSE = 11   # ] 关闭依赖，进入自由内容
    CONTENT = 12      # 自由内容，直到换行符


class SemanticLogitsProcessor(LogitsProcessor):
    """
    骨架生成约束处理器 (FSM 状态机版本)
    
    支持的骨架格式：
    ```
    [PLAN]
    1. <200> <Search> [-] 搜索最新的篮球比赛规则
    2. <150> <None> [-] 分析投篮动作的物理原理
    3. <300> <None> [-] 总结提高命中率的训练技巧
    [END]
    ```
    
    FSM 状态流转确保生成符合格式的输出。
    
    Attributes:
        tokenizer: 用于获取 token ID 的分词器
        prefix_len: 输入前缀长度
        state: 当前 FSM 状态
        token_ids: 关键 token 的 ID 映射
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
            tokenizer: 分词器（用于获取 token IDs）
            prefix_len: 输入前缀长度
            enforce_format: 是否强制执行格式约束
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.prefix_len = prefix_len
        self.enforce_format = enforce_format
        self.state = FSMState.LINE_START
        
        # 缓存关键 token IDs
        self._init_token_ids()
        
        # 记录当前行的生成内容（用于正则匹配）
        self.current_line = ""
        
    def _init_token_ids(self):
        """初始化关键 token 的 IDs"""
        self.token_ids = {}
        
        # 基础符号
        key_tokens = {
            "dot": ".",
            "lt": "<",
            "gt": ">",
            "lbracket": "[",
            "rbracket": "]",
            "dash": "-",
            "newline": "\n",
            "space": " ",
            # 特殊标记
            "plan_start": "[PLAN]",
            "plan_end": "[END]",
            "direct": "[DIRECT]",
        }
        
        for name, token in key_tokens.items():
            try:
                # 尝试直接编码
                ids = self.tokenizer.encode(token, add_special_tokens=False)
                if ids:
                    self.token_ids[name] = ids[0] if len(ids) == 1 else ids
            except Exception:
                pass
        
        # 获取数字 token IDs (0-9)
        self.digit_token_ids: Set[int] = set()
        for d in "0123456789":
            try:
                ids = self.tokenizer.encode(d, add_special_tokens=False)
                if ids:
                    self.digit_token_ids.add(ids[0])
            except Exception:
                pass
        
        # 获取字母 token IDs (用于工具名)
        self.letter_token_ids: Set[int] = set()
        for c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ":
            try:
                ids = self.tokenizer.encode(c, add_special_tokens=False)
                if ids:
                    self.letter_token_ids.add(ids[0])
            except Exception:
                pass

    def _get_current_line_from_ids(self, input_ids: torch.Tensor) -> str:
        """从 input_ids 中提取当前行的文本"""
        # 找到最后一个换行符的位置
        ids_list = input_ids[0].tolist()
        newline_id = self.token_ids.get("newline")
        
        last_newline_pos = -1
        if newline_id is not None:
            for i in range(len(ids_list) - 1, -1, -1):
                if ids_list[i] == newline_id:
                    last_newline_pos = i
                    break
        
        # 提取当前行的 token IDs
        current_line_ids = ids_list[last_newline_pos + 1:]
        
        # 解码为文本
        try:
            return self.tokenizer.decode(current_line_ids, skip_special_tokens=True)
        except Exception:
            return ""

    def _determine_state(self, current_line: str) -> FSMState:
        """
        根据当前行内容确定 FSM 状态
        
        使用正则表达式匹配确定当前所处的解析状态
        """
        line = current_line.strip()
        
        if not line:
            return FSMState.LINE_START
        
        # 匹配各个阶段
        # 完整格式：ID. <Length> <Tool> [Deps] Title
        
        # 检查是否在标题内容区域（已完成所有格式部分）
        if re.match(r"^\d+\.\s*<\d+>\s*<[^>]+>\s*\[[^\]]+\]\s*.+", line):
            return FSMState.CONTENT
        
        # 检查依赖关闭后
        if re.match(r"^\d+\.\s*<\d+>\s*<[^>]+>\s*\[[^\]]+\]$", line):
            return FSMState.DEPS_CLOSE
        
        # 检查依赖值中
        if re.match(r"^\d+\.\s*<\d+>\s*<[^>]+>\s*\[[^\]]*$", line):
            return FSMState.DEPS_VAL
        
        # 检查依赖打开
        if re.match(r"^\d+\.\s*<\d+>\s*<[^>]+>\s*\[$", line):
            return FSMState.DEPS_OPEN
        
        # 检查工具关闭后
        if re.match(r"^\d+\.\s*<\d+>\s*<[^>]+>$", line):
            return FSMState.TOOL_CLOSE
        
        # 检查工具值中
        if re.match(r"^\d+\.\s*<\d+>\s*<[^>]*$", line):
            return FSMState.TOOL_VAL
        
        # 检查工具打开
        if re.match(r"^\d+\.\s*<\d+>\s*<$", line):
            return FSMState.TOOL_OPEN
        
        # 检查长度关闭后
        if re.match(r"^\d+\.\s*<\d+>$", line):
            return FSMState.LEN_CLOSE
        
        # 检查长度值中
        if re.match(r"^\d+\.\s*<\d+$", line):
            return FSMState.LEN_VAL
        
        # 检查长度打开
        if re.match(r"^\d+\.\s*<$", line):
            return FSMState.LEN_OPEN
        
        # 检查点号后
        if re.match(r"^\d+\.\s*$", line):
            return FSMState.AFTER_DOT
        
        # 检查 ID 后
        if re.match(r"^\d+$", line):
            return FSMState.AFTER_ID
        
        # 默认：行首或自由内容
        return FSMState.LINE_START

    def _apply_state_constraints(
        self, 
        state: FSMState, 
        scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        根据当前状态应用约束
        
        Args:
            state: 当前 FSM 状态
            scores: 模型输出的 logits
            
        Returns:
            约束后的 logits
        """
        if not self.enforce_format:
            return scores
            
        scores = scores.clone()
        
        # 根据状态应用 mask
        if state == FSMState.LINE_START:
            # 期望数字开头
            self._mask_except(scores, self.digit_token_ids)
            
        elif state == FSMState.AFTER_ID:
            # 期望 .
            dot_id = self.token_ids.get("dot")
            if dot_id is not None:
                self._mask_except(scores, {dot_id})
                
        elif state == FSMState.AFTER_DOT:
            # 期望空格或 <
            allowed = set()
            if "space" in self.token_ids:
                allowed.add(self.token_ids["space"])
            if "lt" in self.token_ids:
                allowed.add(self.token_ids["lt"])
            if allowed:
                self._mask_except(scores, allowed)
                
        elif state == FSMState.LEN_OPEN:
            # 期望数字
            self._mask_except(scores, self.digit_token_ids)
            
        elif state == FSMState.LEN_VAL:
            # 期望数字或 >
            allowed = self.digit_token_ids.copy()
            if "gt" in self.token_ids:
                allowed.add(self.token_ids["gt"])
            self._mask_except(scores, allowed)
            
        elif state == FSMState.LEN_CLOSE:
            # 期望空格或 <
            allowed = set()
            if "space" in self.token_ids:
                allowed.add(self.token_ids["space"])
            if "lt" in self.token_ids:
                allowed.add(self.token_ids["lt"])
            if allowed:
                self._mask_except(scores, allowed)
                
        elif state == FSMState.TOOL_OPEN:
            # 期望字母
            self._mask_except(scores, self.letter_token_ids)
            
        elif state == FSMState.TOOL_VAL:
            # 期望字母或 >
            allowed = self.letter_token_ids.copy()
            if "gt" in self.token_ids:
                allowed.add(self.token_ids["gt"])
            self._mask_except(scores, allowed)
            
        elif state == FSMState.TOOL_CLOSE:
            # 期望空格或 [
            allowed = set()
            if "space" in self.token_ids:
                allowed.add(self.token_ids["space"])
            if "lbracket" in self.token_ids:
                allowed.add(self.token_ids["lbracket"])
            if allowed:
                self._mask_except(scores, allowed)
                
        elif state == FSMState.DEPS_OPEN:
            # 期望 - (当前强制纯并行)
            if "dash" in self.token_ids:
                self._mask_except(scores, {self.token_ids["dash"]})
                
        elif state == FSMState.DEPS_VAL:
            # 期望 - 或 ]
            allowed = set()
            if "dash" in self.token_ids:
                allowed.add(self.token_ids["dash"])
            if "rbracket" in self.token_ids:
                allowed.add(self.token_ids["rbracket"])
            if allowed:
                self._mask_except(scores, allowed)
                
        elif state == FSMState.DEPS_CLOSE:
            # 期望空格，进入自由内容
            if "space" in self.token_ids:
                self._mask_except(scores, {self.token_ids["space"]})
                
        # CONTENT 状态：不做约束，允许自由生成
        # 遇到换行符后重置到 LINE_START
        
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
        
        Args:
            input_ids: 当前生成的 token 序列 [batch, seq_len]
            scores: 模型输出的 logits [batch, vocab_size]
            
        Returns:
            处理后的 logits [batch, vocab_size]
        """
        seq_length = input_ids.shape[-1]
        
        # 首个生成 token 特殊处理
        if seq_length <= self.prefix_len:
            return scores
        
        # 获取当前行内容
        current_line = self._get_current_line_from_ids(input_ids)
        
        # 确定当前状态
        state = self._determine_state(current_line)
        self.state = state
        
        # 应用状态约束
        return self._apply_state_constraints(state, scores)

    def reset(self):
        """重置状态机"""
        self.state = FSMState.LINE_START
        self.current_line = ""

    def __repr__(self) -> str:
        return f"SemanticLogitsProcessor(state={self.state.name}, prefix_len={self.prefix_len})"


