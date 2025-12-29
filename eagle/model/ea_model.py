# coding=utf-8
import copy
import json
import time

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
import os
from transformers import LogitsProcessor
from transformers import PreTrainedModel, PretrainedConfig, AutoConfig

from .modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
from .modeling_mixtral_kv import MixtralForCausalLM as KVMixtralForCausalLM
#from .modeling_qwen2_kv import LlamaForCausalLM as KVQwen2ForCausalLM
from .modeling_qwen2_kv import Qwen2ForCausalLM as KVQwen2ForCausalLM
from .modeling_qwen3_kv import Qwen3ForCausalLM as KVQwen3ForCausalLM
from .utils import *
from .kv_cache import initialize_past_key_values

from .cnets import Model
from .cnets1 import Model as Model1
from .configs import EConfig
from ..prompts import *

# ======================= 添加内容 ================
class SemanticLogitsProcessor(LogitsProcessor):
    def __init__(self, para_end_token_id, ellipsis_token_id, line_break_token_id, para_begin_token_id, colon_token_id, cn_colon_token_id, colon_new_line_token_id, prefix_len):
        self.para_end_token_id = para_end_token_id
        self.ellipsis_token_id = ellipsis_token_id
        self.line_break_token_id = line_break_token_id
        self.para_begin_token_id = para_begin_token_id
        self.colon_token_id = colon_token_id
        self.cn_colon_token_id = cn_colon_token_id # 增加中文冒号的支持
        self.colon_new_line_token_id = colon_new_line_token_id
        self.prefix_len = prefix_len

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 推理模式下不允许修改原始 scores
        scores = scores.clone()
        # 获取当前生成的序列长度
        seq_length = input_ids.shape[-1]

        # 首个token不可以是para_end_token_id，ellipsis_token_id，line_break_token_id，para_begin_token_id
        if seq_length == self.prefix_len:
            scores[:, self.para_end_token_id] = float('-inf')
            scores[:, self.ellipsis_token_id] = float('-inf')
            scores[:, self.line_break_token_id] = float('-inf')
            scores[:, self.para_begin_token_id] = float('-inf')

        # 如果上一个token是para_begin_token_id，下一个不能是para_begin_token_id
        if seq_length > 0 and input_ids[0, -1] == self.para_begin_token_id:
            scores[:, self.para_begin_token_id] = float('-inf')

        # 如果上一个token是 line_break_token_id 且上上个为ellipsis_token_id，则next token 强制为para_begin_token_id
        if seq_length > 1 and input_ids[0, -2] == self.ellipsis_token_id and input_ids[0, -1] == self.line_break_token_id:
            scores[:, :] = float('-inf')
            scores[:, self.para_begin_token_id] = 0

        # 如果上一个token是colon_token_id，则next token 强制为ellipsis_token_id
        elif seq_length > 0 and input_ids[0, -1] == self.colon_token_id:
            scores[:, :] = float('-inf')
            scores[:, self.ellipsis_token_id] = 0

        elif seq_length > 0 and input_ids[0, -1] == self.cn_colon_token_id:
            scores[:, :] = float('-inf')
            scores[:, self.ellipsis_token_id] = 0

        # 如果上一个token是colon_new_line_token_id，则next token 强制为line_break_token_id
        elif seq_length > 0 and input_ids[0, -1] == self.colon_new_line_token_id:
            scores[:, :] = float('-inf')
            scores[:, self.ellipsis_token_id] = 0

        # 如果上一个token是ellipsis_token_id，则下一个强制为 line_break_token_id
        elif seq_length > 0 and input_ids[0, -1] == self.ellipsis_token_id:
            scores[:, :] = float('-inf')
            scores[:, self.line_break_token_id] = 0

        return scores


class EaModel(nn.Module):

    def __init__(self, use_eagle3, base_model, base_model_name_or_path, ea_model_path,
                 total_token, depth, top_k, threshold, ea_layer_state_dict,):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.hidden_size = base_model.lm_head.weight.shape[-1]
        self.vocab_size = base_model.lm_head.weight.shape[0]
        self.base_model_name_or_path = base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path, use_fast=False)
        self.use_eagle3 = use_eagle3
        config = EConfig.from_pretrained(ea_model_path)
        with open(ea_model_path, "r") as f:
            con = json.loads(f.read())
        try:
            bias = con["bias"]
        except:
            bias = True

        if use_eagle3:
            self.ea_layer = Model(config, bias=bias, total_tokens=total_token, depth=depth, top_k=top_k,
                                 threshold=threshold, path=base_model_name_or_path, load_emb=True)
        else:
            self.ea_layer = Model1(config, bias=bias, total_tokens=total_token, depth=depth, top_k=top_k,
                                  threshold=threshold, path=base_model_name_or_path, load_emb=True)

        low_memory = False

        device = base_model.model.layers[-1].self_attn.q_proj.weight.device
        if device != base_model.lm_head.weight.device:
            self.ea_layer.diff_device = True
            if not low_memory:
                self.ea_layer.headweight = base_model.lm_head.weight.clone().to(device)
            else:
                self.ea_layer.layer_device = device
        else:
            self.ea_layer.diff_device = False

        if self.use_eagle3 and config.vocab_size == config.draft_vocab_size:
            del self.ea_layer.d2t, self.ea_layer.t2d

        self.ea_layer.load_state_dict(ea_layer_state_dict, strict=False) 
        self.ea_layer.to(self.base_model.dtype).to(device)
        self.ea_layer.init_tree()

        self.reset_state()


    def get_tokenizer(self):
        """Get the tokenizer of the base model.

        Returns:
            Tokenizer: The tokenizer of the base model.
        """
        return self.tokenizer


    @classmethod
    def from_pretrained(
        cls,
        use_eagle3=True,
        base_model_path=None,
        ea_model_path=None,
        total_token=60,
        depth=7,
        top_k=10,
        threshold=1.0,
        **kwargs,
    ):
        # assert Type=="LLaMA" or "Mixtral"
        Type = AutoConfig.from_pretrained(base_model_path).architectures[0]

        if Type == 'LlamaForCausalLM':
            base_model = KVLlamaForCausalLM.from_pretrained(base_model_path, **kwargs)
        elif Type == 'Qwen2ForCausalLM':
            base_model = KVQwen2ForCausalLM.from_pretrained(base_model_path, **kwargs)
        elif Type == 'Qwen3ForCausalLM':
            base_model = KVQwen3ForCausalLM.from_pretrained(base_model_path, **kwargs)
        else:
            base_model = KVMixtralForCausalLM.from_pretrained(base_model_path, **kwargs)

        configpath = os.path.join(ea_model_path, "config.json")
        if not os.path.exists(configpath):
            configpath = hf_hub_download(ea_model_path, "config.json")

        try:   # 这里是什么意思，加载模型参数还是eagle参数？
            load_model_path = os.path.join(ea_model_path, "pytorch_model.bin")
            if not os.path.exists(load_model_path):
                load_model_path = hf_hub_download(ea_model_path, "pytorch_model.bin")
            ea_layer_state_dict = torch.load(load_model_path, map_location=base_model.device)
        except:
            from safetensors.torch import load_file
            load_model_path = os.path.join(ea_model_path, "model.safetensors")
            if not os.path.exists(load_model_path):
                load_model_path = hf_hub_download(ea_model_path, "model.safetensors")
            ea_layer_state_dict = load_file(load_model_path)
        model = cls(
            use_eagle3,
            base_model,
            base_model_path,
            configpath,
            total_token,
            depth,
            top_k,
            threshold,
            ea_layer_state_dict
        )

        if total_token == -1:
            device = model.base_model.model.layers[0].self_attn.q_proj.weight.device
            cans = [40, 48, 50, 56, 60]
            x = [1, 1.05, 1.07, 1.1, 1.13]
            times = []

            for i in range(len(cans)):
                length = cans[i]
                input_ids = torch.randint(0, model.config.vocab_size - 200, (1, length)).to(device)
                torch.cuda.synchronize()
                start_time = time.time()
                for _ in range(20):
                    torch.cuda.synchronize()
                    with torch.no_grad():
                        outputs = model.base_model(input_ids)
                    torch.cuda.synchronize()
                torch.cuda.synchronize()
                end_time = time.time()
                times.append((end_time - start_time) / x[i])
            total_token = cans[times.index(min(times))]
            model.ea_layer.total_tokens = total_token - 1

        return model


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        output_orig=False,
        position_ids=None,
    ):
        with torch.inference_mode():
            # Pass input through the base model
            outputs = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
            if output_orig:
                orig = self.base_model.lm_head(outputs[0])
            hidden_states = outputs[0]

        if output_orig:
            return outputs, orig, hidden_states
        else:
            return outputs, hidden_states


    def reset_state(self):
        self.ea_layer.reset_kv()
        self.past_key_values = None
        self.past_key_values_data = None
        self.current_length_data = None

        # 维护一个BIM,和一个保存draft输入的padding信息，因为如果接受长度不同的话，需要补齐输入的batch
        # 同时维护一个全量的位置编码
        self.branch_index_map = None
        self.full_position_ids = None

        # self.batched_draft_input = None
        self.ea_layer.cache_padding_mask = None
        self.ea_layer.full_position_ids = None

        # 保存结果
        self.skeleton_output = None
        self.parallel_branches_output = None

        self.active_branches = None

    # ================= 核心修改：三阶段生成逻辑 =================
    @torch.no_grad()
    def eagenerate(self, task_prompt, max_new_tokens=2048, temperature=0.0, top_p=0.0, top_k=0.0, enable_parallel=True, para_token_ids=None):
        
        self.reset_state()

        if not enable_parallel:
            # 如果不开启并行，回退到原始 EAGLE 逻辑
            input_ids = self.tokenizer([task_prompt], return_tensors="pt").input_ids.to(self.base_model.device)
            return self._eagenerate_loop(input_ids, max_new_tokens, temperature, top_p, top_k, enable_parallel)

        # 准备输入
        task_input = base_prompt.format(user_question=task_prompt)
        task_input_ids = self.tokenizer([task_input], return_tensors="pt").input_ids.to(self.base_model.device)
        skeleton_input_ids = self.tokenizer([skeleton_trigger_zh], return_tensors="pt").input_ids.to(self.base_model.device)
        input_ids = torch.cat([task_input_ids, skeleton_input_ids], dim=-1).to(self.base_model.device)

        # === Stage 1: Skeleton Generation (生成大纲) ===
        print(">>> Stage 1: Generating Skeleton...")

        # 构造 Logits Processor 强制生成骨架
        sp_processor = SemanticLogitsProcessor(
            para_end_token_id=para_token_ids['para_end_token_id'],
            ellipsis_token_id=para_token_ids['ellipsis_token_id'],
            line_break_token_id=para_token_ids['line_break_token_id'],
            para_begin_token_id=para_token_ids['para_begin_token_id'],
            colon_token_id=para_token_ids['colon_token_id'],
            cn_colon_token_id=para_token_ids['cn_colon_token_id'],
            colon_new_line_token_id=para_token_ids['colon_new_line_token_id'],
            prefix_len=input_ids.shape[1]
        )
        
        # 调用 EAGLE 生成骨架 (直到生成 %%%%)
        skeleton_ids = self._eagenerate_loop(
            input_ids=input_ids, 
            max_new_tokens=max_new_tokens, 
            temperature=temperature,
            enable_parallel=enable_parallel,
            logits_processor=LogitsProcessorList([sp_processor]),   
            stop_token_id=para_token_ids['para_end_token_id'],
            return_kv=False,
            para_token_ids=para_token_ids 
        )
        print("Generated Skeleton:", self.tokenizer.decode(skeleton_ids[0]))
        self.skeleton_output = skeleton_ids.clone()  # 保存skeleton

        # === Stage 2: Parse & Prepare (解析与准备) ===
        print(">>> Stage 2: Parsing Skeleton & Preparing Parallel Branches...")
        
        # 调用封装好的解析函数
        clean_branches, instruction_len = self._parse_skeleton_branches(skeleton_ids, para_token_ids)
        
        if not clean_branches:
            print("Parsing failed or no branches found. Returning skeleton.")
            return skeleton_ids

        num_para = len(clean_branches)
        if num_para == 0:
            return skeleton_ids
        
        self.parallel_branches_output = [list(br) for br in clean_branches]  # 保存branches结果
        print(f"Detected {num_para} parallel branches.")

        # === Stage 3: Parallel Decoding (并行解码) ===
        print(">>> Stage 3: Parallel Decoding...")

        # 调用并行解码
        _ = self._eagenerate_parallel(
            prefix_ids=task_input_ids,
            branches_prompts=clean_branches,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            instruction_len=instruction_len,
        )
        
        # 拼接结果
        skeleton_part = self.skeleton_output[0].tolist()
        skeleton_part.append(para_token_ids['line_break_token_id'])
        print("Skeleton:", self.tokenizer.decode(torch.tensor(skeleton_part)))

        parallel_part = []
        for i in range(num_para):
            branch_output = self.parallel_branches_output[i][instruction_len[i]:]
            parallel_part.extend(branch_output)
            parallel_part.append(para_token_ids['line_break_token_id'])
            # print(f"Branch {i}:", self.tokenizer.decode(torch.tensor(branch_output)))
        
        merged_ids = skeleton_part + parallel_part
        merged_ids.append(para_token_ids['para_end_token_id'])
        
        return torch.tensor([merged_ids], device=input_ids.device)


    # ================= 内部方法: 原始 EAGLE 循环 (Stage 1) =================
    def _eagenerate_loop(self, input_ids, max_new_tokens=2048, temperature=0.0, top_p=0.0, top_k=0.0, enable_parallel=False,
                        logits_processor=None, stop_token_id=None, return_kv=False, para_token_ids=None):
        # 初始化
        if temperature > 1e-5:
            lp = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
            if logits_processor:
                lp.extend(logits_processor)
            logits_processor = lp
        
        self.ea_layer.reset_kv()
        
        # KV Cache 初始化
        max_kv_len = input_ids.shape[1] + max_new_tokens + 100
        (past_key_values, past_key_values_data, current_length_data) = initialize_past_key_values(self.base_model, max_length=max_kv_len)
        
        self.past_key_values = past_key_values
        self.past_key_values_data = past_key_values_data
        self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        
        # Prefill
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids, logits, hidden_state, sample_token = initialize_tree(
            input_ids, self, past_key_values, logits_processor
        )
        
        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        
        # Decode Loop
        for idx in range(max_new_tokens):
            self.base_model.model.tree_mask = tree_mask # 关键：设置 Base Model 的 Mask
            draft_tokens = draft_tokens.to(input_ids.device)
            
            # Base Model Forward
            logits, hidden_state_new, outputs = tree_decoding(self, draft_tokens, past_key_values, tree_position_ids, input_ids, retrieve_indices)

            draft_tokens = torch.cat((draft_tokens, padding), dim=1)
            candidates = draft_tokens[0, retrieve_indices]  # 全都采用batch维度。  [1, Leaf, Depth] 
            logits_for_eval = logits #  [1, Leaf, Depth, Vocab]

            # Verification
            best_candidate, accept_length, sample_p = evaluate_posterior(logits_for_eval, candidates, logits_processor, para_token_ids)
            # print(f"Accept length: {accept_length.item()}")

            # 为了兼容 update_inference_inputs (它处理 Batch)，我们需要把 best_candidate 和 accept_length 包装回 Batch=1
            if isinstance(accept_length, int):
                accept_length = torch.tensor(accept_length, device=input_ids.device)
            if best_candidate.ndim == 0:
                best_candidate = best_candidate.unsqueeze(0) # [1]
            if accept_length.ndim == 0:
                accept_length = accept_length.unsqueeze(0)   # [1]
            
            # Update Inputs for next step (Generates next Draft Tree)
            input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids = update_inference_inputs(
                input_ids, candidates, best_candidate, accept_length, retrieve_indices, logits_processor, self, hidden_state_new, sample_p)
            
            # Stop Conditions
            if stop_token_id and stop_token_id in input_ids[0, input_len:].tolist():
                break
            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if self.current_length_data[0].item() == max_kv_len - 100:  # 防止超出cache(每个轮次会分配60)
                break
            if idx >= 50 and enable_parallel:  # skeleton 的长度控制不超过50
                break
                 
        if return_kv:  # 这里需要返回self.ea_layer.stable_kv来复用eagle layer的cache
            return input_ids[:, input_len:], (past_key_values, past_key_values_data, current_length_data)
        return input_ids[:, input_len:]


    # ================= 内部方法: 并行解码 (Stage 3) =================
    def _eagenerate_parallel(self, prefix_ids, branches_prompts, max_new_tokens, temperature=0, top_p=0, top_k=0, instruction_len=None, logits_processor=None):
        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)

        device = self.base_model.device
        prefix_len = prefix_ids.shape[1]
        
        # 初始化状态 (Input, Cache, BIM)  
        input_ids, tips_indices, branch_begins, branch_lengths, draft_input_ids = self._init_parallel_state(
            prefix_ids, branches_prompts, max_new_tokens)

        # Call Initialize Tree  
        input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, hidden_states = initialize_tree_packed(
            prefix_len, input_ids, self, tips_indices, branch_begins, branch_lengths, draft_input_ids, logits_processor)
               
        print(f"skeleton_output {self.tokenizer.decode(self.skeleton_output[0].tolist())}")
        # 3. Decoding Loop
        total_accept_len = torch.zeros(1, dtype=torch.long, device=device)
        for step in range(max_new_tokens):
            # 构建 Mask, History(使用 BIM) + Draft(Tree mask)
            current_length = self.current_length_data[0].item()
            num_nodes = draft_tokens.shape[1] # 这里绝对不能去掉root啊，因为root是生成的，还没有输入推理         
            combined_mask = self._construct_verify_mask(current_length, tree_mask, num_nodes, device, torch.float32)
            
            # Verify (Base Model Forward)
            logits, hidden_states = self._parallel_verify_step(draft_tokens, tree_position_ids, combined_mask, num_nodes)
            
            # Evaluate (Compare candidates & logits)  
            best_candidate, accept_length, sample_logits = self._evaluate_wrapper(logits, draft_tokens, retrieve_indices, logits_processor)
            # print(f"accept length: {accept_length}")
            total_accept_len += accept_length.sum()
            if torch.sum(accept_length) > 0 and step % 10 == 0:
                print(f"Step: {step}, Accepted lengths: {accept_length.tolist()}")

            prob = sample_logits 
            if logits_processor is not None:
                sample_tokens = torch.multinomial(prob, 1)
                sample_tokens = sample_tokens[None] if sample_tokens.ndim == 1 else sample_tokens # Ensure [B, 1]
            else:
                sample_tokens = torch.argmax(prob, dim=-1, keepdim=True) # [B, 1]

            # Update State (Cache Management & Token Commit)
            next_tips_hidden, next_tips_tokens = self._update_parallel_state(best_candidate, accept_length, draft_tokens, 
                retrieve_indices, hidden_states, sample_tokens, num_nodes)
            
            if self.active_branches == None or len(self.active_branches) == 0:
                print("All branches finished generation.")
                break

            # Next Draft
            draft_tokens, retrieve_indices, tree_mask, tree_position_ids = self.ea_layer.topK_generate(
                next_tips_hidden, next_tips_tokens, active_branch=self.active_branches)
            
            if step > 300: break # Safety break
        print(f"Avg accepted lengths: {total_accept_len.item()/step}")
        return None


    def _init_parallel_state(self, prefix_ids, branches_prompts, max_new_tokens):
        device = self.base_model.device
        num_para = len(branches_prompts)
        prefix_len = prefix_ids.shape[1]
        self.active_branches = list(range(num_para))

        # === Cache Init (Base & Draft) === 
        self.current_length_data.fill_(prefix_len)  # base model,只保留prefix的cache就好
        if self.ea_layer.stable_kv is not None:
            k_draft, v_draft = self.ea_layer.stable_kv[0]  # 提取 K, V
            k_prefix_draft = k_draft[..., :prefix_len, :].clone()  # 切片 (保留到 para_begin_idx)
            v_prefix_draft = v_draft[..., :prefix_len, :].clone()
            # Expand + Clone (物理复制以支持后续 Append)
            k_expanded = k_prefix_draft.expand(num_para, -1, -1, -1).clone()
            v_expanded = v_prefix_draft.expand(num_para, -1, -1, -1).clone()
            self.ea_layer.stable_kv = ((k_expanded, v_expanded),)
        
        # === Base Model Inputs (Packed) === 
        flat_branch_ids = []
        branch_index_list = [-1] * prefix_len # Prefix 标记为 -1, 用于记录每个token的branch的归属问题。-1表示共享prefix部分。

        current_offset = prefix_len
        tips_indices = []
        branch_begins = []
        branch_lengths = []
        pos_ids_list_base = list(range(prefix_len))

        draft_branch_list = []
        draft_pos_list = [] # 用于 Draft 的 Position ID
        
        for i, br in enumerate(branches_prompts):
            current_len = len(br)
            branch_begins.append(current_offset)
            flat_branch_ids.extend(br)

            branch_index_list.extend([i] * current_len) 
            branch_lengths.append(current_len)
            current_offset += current_len
            tips_indices.append(current_offset - 1)  # Tips 是这一段的最后一个词
            # base model logic
            curr_pos_ids = list(range(prefix_len, prefix_len + len(br)))
            pos_ids_list_base.extend(curr_pos_ids)  # 更新 pos_ids

            # Draft Model Logic (Preparation)
            draft_branch_list.append(torch.tensor(br, device=device, dtype=torch.long))  # 收集 token tensor
            draft_pos_list.append(torch.tensor(curr_pos_ids, device=device, dtype=torch.long))   # 收集 Position IDs (后续要 Pad)
        
        branches_tensor = torch.tensor([flat_branch_ids], device=device, dtype=torch.long) # [1, Br_Len]
        input_ids = torch.cat([prefix_ids, branches_tensor], dim=1)
        tips_indices = torch.tensor(tips_indices, device=device)
        self.full_position_ids = torch.tensor(pos_ids_list_base, device=device)
        
        # Branch Index Map (BIM) - [Total_Capacity]
        total_capacity = input_ids.shape[1] + max_new_tokens + 128
        self.branch_index_map = torch.full((total_capacity,), -2, dtype=torch.long, device=device) # -2: Empty
        self.branch_index_map[:len(branch_index_list)] = torch.tensor(branch_index_list, device=device)

        # ===  Draft Batch Padding ===
        #  Token 用 pad_token_id，Position ID 用 0 (Mask 会盖住，数值不重要)
        draft_input_ids, branch_mask = stack_with_left_padding(draft_branch_list, 
            pad_id=self.tokenizer.pad_token_id, device=device, return_mask=True) # [Num_Para, Max_Len]
        padded_branch_pos = stack_with_left_padding(draft_pos_list, pad_id=0, device=device) # [Num_Para, Max_Len]

        prefix_mask = torch.ones((num_para, prefix_len), dtype=torch.long, device=device)
        prefix_pos = torch.arange(prefix_len, device=device, dtype=torch.long).unsqueeze(0).expand(num_para, -1)
        self.ea_layer.cache_padding_mask = torch.cat([prefix_mask, branch_mask], dim=1)
        self.ea_layer.full_position_ids = torch.cat([prefix_pos, padded_branch_pos], dim=1)
            
        return input_ids, tips_indices, branch_begins, branch_lengths, draft_input_ids


    def _construct_verify_mask(self, history_len, tree_mask, num_nodes, device, dtype=torch.float32):
        """
        利用 Branch Index Map (BIM) 构建高效 Mask
        mask由两部分构成，一部分是根据历史得到的块对角矩阵，另一部分是不同branch的tree mask拼成的块对角阵
        """
        history_bim = self.branch_index_map[:history_len]
        total_history = history_bim.shape[0]
        
        active_ids_list = self.active_branches
        num_para = len(self.active_branches)
        packed_draft_len = num_para * num_nodes
        
        # Cross Mask [1, 1, Draft, History]
        cross_mask = torch.full((1, 1, packed_draft_len, total_history), torch.finfo(dtype).min, device=device)
        
        # 获取token的branch归属：[0,0,0...,1,1,1,...,2,2,2....]
        active_ids_tensor = torch.tensor(active_ids_list, device=device)
        draft_branch_ids = active_ids_tensor.repeat_interleave(num_nodes)
        # draft_branch_ids = torch.arange(num_para, device=device).repeat_interleave(num_nodes) # [Packed_Draft]
        
        # Prefix 全部可见
        is_prefix = (history_bim == -1).view(1, 1, 1, -1) # [1, 1, 1, Hist]
        cross_mask.masked_fill_(is_prefix, 0)

        draft_ids_view = draft_branch_ids.view(1, 1, -1, 1)
        hist_ids_view = history_bim.view(1, 1, 1, -1)
        
        # branch对应可见
        is_self = (draft_ids_view == hist_ids_view)
        cross_mask.masked_fill_(is_self, 0)
        
        # tree Mask
        converted_tree_mask = torch.where(tree_mask == 1, 0.0, torch.finfo(dtype).min)
        draft_block_mask = torch.full((packed_draft_len, packed_draft_len), torch.finfo(dtype).min, device=device)
        for i in range(num_para):
            st, ed = i*num_nodes, (i+1)*num_nodes
            draft_block_mask[st:ed, st:ed] = converted_tree_mask[i, 0, :, :] # 包括root，且可见和不可见用0和-inf表示，重新赋值操作
            
        draft_block_mask = draft_block_mask.unsqueeze(0).unsqueeze(0)
        block_tree_mask = torch.cat([cross_mask, draft_block_mask], dim=-1)
        
        return block_tree_mask


    def _update_parallel_state(self, best_candidate, accept_length, draft_tokens, retrieve_indices, 
        hidden_states, sample_tokens, num_nodes):
        """
        核心管理函数：
        1. 搬运 Accepted KV 到 History 末尾 (Compact)
        2. 更新 BIM
        3. 准备下一轮 Tips
        4. 重置 Cache 指针，实现回滚
        """
        device = self.base_model.device
        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id

        # ========== 1. 确定当前有效历史长度（BIM 中第一个 -2 的位置）==========
        valid_history_len = (self.branch_index_map != -2).sum().item()
        dst_ptr = valid_history_len  # KV 写入指针
        last_pos = self.ea_layer.full_position_ids[:, -1].tolist()

        new_bim_entries = []
        new_pos_list = []

        # 用于更新 EAGLE Draft Model 的输入（左填充）
        draft_update_tokens_list = []  # 每个分支的 Bonus Token（或 pad）
        draft_update_hiddens_list = [] # 对应 hidden state

        # 标记本轮分支存活状态 (初始化全为 True)
        keep_mask_list = [True] * len(self.active_branches)

        # ========== 3. 遍历每个分支 ==========
        for i, branch_idx in enumerate(self.active_branches):

            # --- 提取当前分支的验证结果 ---
            acc_len_i = accept_length[i].item()  # 接受的 token 数（不含 Root）
            best_idx_i = best_candidate[i].item()

            # retrieve_indices[i, best_idx_i, :] 形状: [Depth]
            # 路径：[Root, t1, t2, ..., t_acc_len, ...]
            select_indices = retrieve_indices[i, best_idx_i, :acc_len_i + 1]  # Root + acc_len 
            seq_tokens = draft_tokens[i][select_indices]  # 这里边包括root

            # Bonus Token 需要通过sample logits计算
            bonus_token = sample_tokens[i].item()
            is_now_finished = (bonus_token == eos_token_id)

            # --- 更新 final_sequences（已接受的 token，不含 bonus）---
            tokens_to_add = seq_tokens.tolist()  
            if is_now_finished:
                print(f"Branch {branch_idx} (curr_idx {i}) finished with EOS.")
                keep_mask_list[i] = False
                tokens_to_add.append(bonus_token)  # 如果 Bonus 是 EOS，必须加上并结束
            self.parallel_branches_output[branch_idx].extend(tokens_to_add)

            # 更新 KV Cache
            cache_indices = select_indices + num_nodes * i  # [关键]转换为全局索引
            for past_key_values in self.past_key_values_data:
                # abs_select_indices = select_indices + dst_ptr
                tgt = past_key_values.index_select(dim=-2, index=(cache_indices + valid_history_len).to(device))
                tgt_len = tgt.shape[-2]
                dst = past_key_values[..., dst_ptr: dst_ptr + tgt_len, :]
                dst.copy_(tgt, non_blocking=True)
                self.current_length_data.fill_(dst_ptr + tgt_len)
            dst_ptr = dst_ptr + tgt_len  # 补充一个指针，这个指针是下一个branch的起点
            new_bim_entries.extend([branch_idx] * tgt_len)
            
            pos = torch.tensor(list(range(last_pos[i] + 1, last_pos[i] + 1 + tgt_len)), device=device)
            self.full_position_ids = torch.cat([self.full_position_ids, pos])

            if not is_now_finished:
                # --- 准备下一轮 Draft 输入 ---
                hidden_states_this_branch = hidden_states[:, i*num_nodes: (i+1)*num_nodes]
                seq_hiddens = hidden_states_this_branch[0, select_indices, :] 

                draft_tokens_tensor = torch.cat([seq_tokens[1:], torch.tensor([bonus_token], device=device)])  # 这里需要去掉root错位对齐
                draft_update_tokens_list.append(draft_tokens_tensor)
                draft_update_hiddens_list.append(seq_hiddens)

                new_pos_list.append(pos)

        # ========== Base Model 状态更新 ==========
        if new_bim_entries:  #  更新 BIM 和 position id
            self.branch_index_map[valid_history_len : dst_ptr] = torch.tensor(new_bim_entries, device=device)
        self.branch_index_map[dst_ptr:] = -2 
        self.current_length_data.fill_(dst_ptr)  # 更新 cache指针

        # ========== Draft Model 动态剪枝与更新 ==========
        # 1. 生成掩码
        keep_mask_tensor = torch.tensor(keep_mask_list, device=device, dtype=torch.bool)
        
        # 2. 【剪枝】如果本轮有分支结束，立即从 ea_layer 物理删除
        # 这步操作保证了 ea_layer 的行数与 active_branches 保持一致
        if not torch.all(keep_mask_tensor):
            self.ea_layer.full_position_ids = self.ea_layer.full_position_ids[keep_mask_tensor]
            self.ea_layer.cache_padding_mask = self.ea_layer.cache_padding_mask[keep_mask_tensor]
            
            # 同步更新 Python 端的 ID 列表
            self.active_branches = [b for b, keep in zip(self.active_branches, keep_mask_list) if keep]

            # 针对stable kv进行剪枝
            if self.ea_layer.stable_kv is not None:
                k_stable, v_stable = self.ea_layer.stable_kv[0]
                k_stable = k_stable[keep_mask_tensor]
                v_stable = v_stable[keep_mask_tensor]
                self.ea_layer.stable_kv = ((k_stable, v_stable),)

        # if len(self.active_branches) == 1:
        #     print("Only one branch remaining, switching to original EAGLE decoding.")

        # 3. 检查是否全部结束
        if not self.active_branches:
            print("All branches finished.")
            return None, None

        # ========== Draft Model 局状态更新 ==========
        # token, cache, positon 对齐
        batched_draft_tokens, batch_mask = stack_with_left_padding(draft_update_tokens_list, pad_id=pad_token_id, device=device, return_mask=True)
        batched_draft_hiddens = stack_with_left_padding(draft_update_hiddens_list, pad_id=0, device=device)
        batched_new_pos = stack_with_left_padding(new_pos_list, pad_id=0, device=device)
        self.ea_layer.cache_padding_mask = torch.cat([self.ea_layer.cache_padding_mask, batch_mask], dim=1)  # 更新 Eagle Mask   
        self.ea_layer.full_position_ids = torch.cat([self.ea_layer.full_position_ids, batched_new_pos], dim=1)  # 更新 position id   

        return batched_draft_hiddens, batched_draft_tokens
        

    def _construct_parallel_mask(self, prefix_len, branch_len, dtype=torch.float32):
        """
        构建并行 Prefill 阶段的 Mask
        Branches Prompts (Length = input_len)
        Prefix + Branches Prompts (Length = prefix_len + input_len)
        """
        device = self.base_model.device
        # dtype = self.base_model.dtype
        total_len = prefix_len + branch_len

        total_ids = self.branch_index_map[:total_len] # Total Sequence
        branch_ids = self.branch_index_map[prefix_len : total_len] # Input Sequence
        
        # 初始化 Mask (全不可见)
        mask = torch.full((1, 1, branch_len, total_len), torch.finfo(dtype).min, device=device)
        
        # prefix填充，全部可见
        is_prefix = (total_ids == -1).unsqueeze(0)  
        mask.masked_fill_(is_prefix, 0) 
        
        # branch部分填充，两部分组成
        # Block Constraint
        branch_ids_view = branch_ids.unsqueeze(1) # [Input_Len, 1]
        total_ids_view = total_ids.unsqueeze(0) # [1, Total_Len]
        block_mask = (branch_ids_view == total_ids_view) # [Input_Len, Total_Len]
        
        # Causal Constraint
        branch_idx = torch.arange(prefix_len, total_len, device=device).unsqueeze(1) # [Input, 1]
        total_idx = torch.arange(total_len, device=device).unsqueeze(0)    # [1, Total]
        causal_mask = (total_idx <= branch_idx)
        
        # 块掩码和因果掩码合并取交集
        valid_mask = block_mask & causal_mask   
        
        mask.masked_fill_(valid_mask, 0)
        
        return mask


    def _parallel_verify_step(self, draft_tokens, tree_position_ids, combined_mask, num_nodes):
        """
        执行 Base Model Forward 进行验证
        """
        # 1. 准备 Input, Position IDs，千万不能切掉root，因为root只是生成了，还没有前向推理
        num_para = draft_tokens.shape[0]
        flat_draft_tokens = draft_tokens.reshape(1, -1)

        current_branch_tip_pos = self.ea_layer.full_position_ids[:,-1].unsqueeze(-1)  # 计算每个 Branch 当前的 Tip 绝对位置 (含Prefix)
        abs_draft_pos = tree_position_ids + current_branch_tip_pos + 1 # 最后的+1是因为要加上树的root
        flat_draft_pos = abs_draft_pos.view(1, -1)
        
        # 3. Forward
        outputs, hidden_states = self(
            flat_draft_tokens,
            past_key_values=self.past_key_values,
            attention_mask=combined_mask,
            position_ids=flat_draft_pos,
            output_orig=False
        )
        
        # 4. Reshape Logits
        logits = self.base_model.lm_head(hidden_states)
        logits = logits.view(num_para, num_nodes, -1) # [Num_Para, Tree_Size, V]
        
        ea_device = self.ea_layer.lm_head.weight.device
        if outputs["hidden_states"][0].device != ea_device:
            outputs["hidden_states"] = [x.to(ea_device) for x in outputs["hidden_states"]]
        choosen_hidden_states=torch.cat(outputs["hidden_states"],dim=-1)

        return logits, choosen_hidden_states
    

    def _evaluate_wrapper(self, logits, draft_tokens, retrieve_indices, logits_processor):
        """
        准备数据并调用 evaluate_posterior
        """
        num_para = logits.shape[0]
        device = logits.device

        retrieve_indices = retrieve_indices.to(device)
        draft_tokens = draft_tokens.to(device)
        
        # 1. Gather Candidates (含 Root)
        # draft_tokens 是原始含 Root 的 [B, 61]
        padding_mask = (retrieve_indices == -1)
        safe_indices = retrieve_indices.clone()
        safe_indices[padding_mask] = 0  # 避免负索引
        
        candidates = torch.gather(draft_tokens.unsqueeze(1).expand(-1, retrieve_indices.size(1), -1), 2, safe_indices)
        candidates.masked_fill_(padding_mask, 0)
        
        # 2. Prepare Logits
        vocab_size = logits.size(-1)
        flat_indices = safe_indices.view(num_para, -1).unsqueeze(-1).expand(-1, -1, vocab_size)
        candidate_logits = torch.gather(logits, 1, flat_indices).view(num_para, retrieve_indices.size(1), retrieve_indices.size(2), -1)
        
        # 3. Evaluate
        best_candidate, accept_length, sample_logits = evaluate_posterior(candidate_logits, candidates, logits_processor)
        
        return best_candidate, accept_length, sample_logits


    def _parse_skeleton_branches(self, skeleton_ids, para_token_ids):
        """
        解析 Skeleton，提取公共前缀和并行分支
        返回: (prefix_ids, branches_prompts)
        """
        seq_list = skeleton_ids[0].tolist()
        para_begin_token_id = para_token_ids['para_begin_token_id']
        para_end_token_id = para_token_ids['para_end_token_id']
        
        # 准备冒号 ID 列表
        colon_ids = [para_token_ids['colon_token_id']]
        colon_ids.append(para_token_ids['cn_colon_token_id'])
        colon_ids.append(para_token_ids['colon_new_line_token_id'])

        try:
            para_begin_idx = seq_list.index(para_begin_token_id)
            # 搜索结束符
            try:
                para_end_idx = seq_list.index(para_end_token_id, para_begin_idx)
            except ValueError:
                para_end_idx = len(seq_list)
        except ValueError:
            print("Warning: No '####' found in generated output.")
            return None, None

        # 1. 提取公共前缀 (Shared Prefix)
        prefix_ids = skeleton_ids[:, :para_begin_idx]
        # 2. 提取并行片段并分割, 把结束符的两个符号"####" "%%%%" 都去掉
        para_segment = seq_list[para_begin_idx : para_end_idx-1]
        
        # 手动 split 逻辑 (保留分隔符 ####)
        raw_branches = []
        current_branch = []
        for token in para_segment:
            if token == para_begin_token_id:
                if current_branch:
                    raw_branches.append(current_branch)
                current_branch = [token]
            else:
                current_branch.append(token)
        if current_branch:
            raw_branches.append(current_branch)

        # 3. 清洗分支 (截取到冒号)
        clean_branches = []
        for br in raw_branches:
            cut_idx = -1
            for i, token in enumerate(br):
                if token in colon_ids:
                    cut_idx = i
                    break
            
            if cut_idx != -1:
                # 保留到冒号 [####, ..., :]
                clean_branches.append(br[:cut_idx+1])
            else:
                # 没找到冒号，保留全部作为 fallback
                clean_branches.append(br)
        
        result_skeleton = []
        for br in clean_branches:
            result_skeleton.extend(br)
        result_skeleton_str = self.tokenizer.decode(result_skeleton)

        instruction_len = []
        for i, br in enumerate(clean_branches):
            current_branch_str = self.tokenizer.decode(br)
            instruction_text = parallel_trigger_zh.format(skeleton_context = result_skeleton_str, current_point=current_branch_str)
            instruction_ids = self.tokenizer.encode(instruction_text, add_special_tokens=False)
            clean_branches[i] = instruction_ids + br
            instruction_len.append(len(instruction_ids))

        return clean_branches, instruction_len