# coding=utf-8
"""
Logits Processors for SpecSoT

该模块定义了用于控制生成过程的 Logits Processors：

1. SemanticLogitsProcessor: 骨架生成约束
   - 强制生成骨架格式的特殊 token 序列
   - 确保格式：####标题:...\n####标题:...\n####%%%%
"""

import torch
from transformers import LogitsProcessor


class SemanticLogitsProcessor(LogitsProcessor):
    """
    语义约束 Logits Processor
    
    用于骨架生成阶段，强制模型输出符合骨架格式的 token 序列。
    
    骨架格式示例：
    ```
    ####背景介绍(100):...
    ####核心论点分析(350):...
    ####总结与建议(200):...
    ####%%%%
    ```
    
    约束规则：
    1. 首个 token 不能是特殊分隔符
    2. #### 后不能紧跟 ####
    3. : 或 ：后必须紧跟 ...
    4. ... 后必须紧跟换行 \n\n
    5. ... + \n\n 后必须紧跟 ####
    
    Attributes:
        para_begin_token_id: "####" 的 token ID
        para_end_token_id: "%%%%" 的 token ID
        ellipsis_token_id: "......" 的 token ID
        line_break_token_id: "\n\n" 的 token ID
        colon_token_id: ":" 的 token ID
        cn_colon_token_id: "：" 的 token ID (中文冒号)
        colon_new_line_token_id: ":\n" 的 token ID
        prefix_len: 输入前缀长度
    """

    def __init__(
        self,
        para_end_token_id: int,
        ellipsis_token_id: int,
        line_break_token_id: int,
        para_begin_token_id: int,
        colon_token_id: int,
        cn_colon_token_id: int,
        colon_new_line_token_id: int,
        prefix_len: int,
    ):
        """
        初始化语义约束处理器
        
        Args:
            para_end_token_id: 骨架结束符 "%%%%" 的 token ID
            ellipsis_token_id: 省略号 "......" 的 token ID
            line_break_token_id: 换行 "\n\n" 的 token ID
            para_begin_token_id: 分支开始符 "####" 的 token ID
            colon_token_id: 英文冒号 ":" 的 token ID
            cn_colon_token_id: 中文冒号 "：" 的 token ID
            colon_new_line_token_id: ":\n" 的 token ID
            prefix_len: 输入前缀长度（用于判断首个生成位置）
        """
        super().__init__()
        self.para_end_token_id = para_end_token_id
        self.ellipsis_token_id = ellipsis_token_id
        self.line_break_token_id = line_break_token_id
        self.para_begin_token_id = para_begin_token_id
        self.colon_token_id = colon_token_id
        self.cn_colon_token_id = cn_colon_token_id
        self.colon_new_line_token_id = colon_new_line_token_id
        self.prefix_len = prefix_len

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        处理 logits，应用语义约束
        
        Args:
            input_ids: 当前生成的 token 序列 [batch, seq_len]
            scores: 模型输出的 logits [batch, vocab_size]
            
        Returns:
            处理后的 logits [batch, vocab_size]
        """
        # 推理模式下不允许修改原始 scores
        scores = scores.clone()
        seq_length = input_ids.shape[-1]

        # =================================================================
        # 规则 1: 首个 token 不能是特殊分隔符
        # =================================================================
        if seq_length == self.prefix_len:
            scores[:, self.para_end_token_id] = float('-inf')
            scores[:, self.ellipsis_token_id] = float('-inf')
            scores[:, self.line_break_token_id] = float('-inf')
            scores[:, self.para_begin_token_id] = float('-inf')
            scores[:, self.colon_token_id] = float('-inf')
            scores[:, self.cn_colon_token_id] = float('-inf')
            scores[:, self.colon_new_line_token_id] = float('-inf')
            return scores

        # 获取上一个 token
        last_token = input_ids[0, -1].item()
        
        # 获取上上个 token (如果存在)
        prev_prev_token = input_ids[0, -2].item() if seq_length > 1 else None

        # =================================================================
        # 规则 2: #### 后不能紧跟 ####
        # =================================================================
        if last_token == self.para_begin_token_id:
            scores[:, self.para_begin_token_id] = float('-inf')

        # =================================================================
        # 规则 3: ... + \n\n 后必须是 ####
        # 即：上上个是 ...，上一个是 \n\n，则强制下一个是 ####
        # =================================================================
        elif (prev_prev_token == self.ellipsis_token_id and 
              last_token == self.line_break_token_id):
            scores[:, :] = float('-inf')
            scores[:, self.para_begin_token_id] = 0

        # =================================================================
        # 规则 4: : 或 ：后必须是 ...
        # =================================================================
        elif last_token == self.colon_token_id:
            scores[:, :] = float('-inf')
            scores[:, self.ellipsis_token_id] = 0

        elif last_token == self.cn_colon_token_id:
            scores[:, :] = float('-inf')
            scores[:, self.ellipsis_token_id] = 0

        elif last_token == self.colon_new_line_token_id:
            scores[:, :] = float('-inf')
            scores[:, self.ellipsis_token_id] = 0

        # =================================================================
        # 规则 5: ... 后必须是 \n\n
        # =================================================================
        elif last_token == self.ellipsis_token_id:
            scores[:, :] = float('-inf')
            scores[:, self.line_break_token_id] = 0

        return scores

    def __repr__(self) -> str:
        return (
            f"SemanticLogitsProcessor("
            f"para_begin={self.para_begin_token_id}, "
            f"para_end={self.para_end_token_id}, "
            f"ellipsis={self.ellipsis_token_id}, "
            f"line_break={self.line_break_token_id})"
        )
