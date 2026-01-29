       
    def generate_eagle(
        self,
        task_prompt: str,
        max_new_tokens: int,
        logits_processor: Optional[LogitsProcessorList] = None,
    ) -> Tuple[torch.Tensor, float, int, float, float, float]:
        """
        标准投机解码生成（不使用骨架）
        
        流程：
        1. 初始化: 准备 input_ids, KV Cache
        2. Prefill: 初始化 KV Cache 和第一轮 Draft Tree (支持分布式)
        3. Decode Loop: 循环执行 decode_step_single 直到停止
        """
        device = self.base_model.device
        input_ids = self.tokenizer([task_prompt], return_tensors="pt").input_ids.to(device)
        input_len = input_ids.shape[1]
        
        # =====================================================================
        # 1. 初始化 KV Cache
        # =====================================================================
        max_kv_len = input_len + max_new_tokens + 100
        self.past_key_values, self.past_key_values_data, self.current_length_data = \
            initialize_past_key_values(self.base_model, max_length=max_kv_len)
        
        # =====================================================================
        # 2. Prefill 阶段 (支持分布式)
        # =====================================================================
        if self.is_distributed():
            # 分布式Prefill
            draft_tokens, retrieve_indices, tree_mask, tree_position_ids, _, _, _ = \
                self.distributed_prefill_manager.prefill_single_distributed(
                    input_ids, self.past_key_values, logits_processor
                )
        else:
            # 普通Prefill
            draft_tokens, retrieve_indices, tree_mask, tree_position_ids, _, _, _ = \
                self.prefill_single(input_ids, logits_processor)
        set_random_seed(self.seed)

        # =====================================================================
        # 3. Decode 循环
        # =====================================================================
        # 计时器
        total_accept_len = torch.zeros(1, dtype=torch.long, device=device)
        total_draft_time = 0.0
        total_update_time = 0.0
        total_verify_time = 0.0
        evt_start = torch.cuda.Event(enable_timing=True)
        evt_after_verify = torch.cuda.Event(enable_timing=True)
        evt_after_update = torch.cuda.Event(enable_timing=True)
        evt_after_draft = torch.cuda.Event(enable_timing=True)
        
        stop_token_id = None
        eos_token_id = self.tokenizer.eos_token_id
        
        for step in range(max_new_tokens):
            evt_start.record()
            
            # 执行单步 decode
            (
                input_ids,
                draft_tokens,
                retrieve_indices,
                tree_mask,
                tree_position_ids,
                accept_length,
            ) = self.decode_step_single(
                input_ids=input_ids,
                draft_tokens=draft_tokens,
                retrieve_indices=retrieve_indices,
                tree_mask=tree_mask,
                tree_position_ids=tree_position_ids,
                logits_processor=logits_processor,
                evt_after_verify=evt_after_verify,
                evt_after_update=evt_after_update,
            )
            evt_after_draft.record()
            
            # 统计
            total_accept_len += accept_length.sum()
            
            # 计时统计
            torch.cuda.synchronize()
            total_verify_time += evt_start.elapsed_time(evt_after_verify) / 1000
            total_update_time += evt_after_verify.elapsed_time(evt_after_update) / 1000
            total_draft_time += evt_after_update.elapsed_time(evt_after_draft) / 1000
            
            # 停止条件检查
            if check_stop_conditions(
                input_ids, input_len, stop_token_id, eos_token_id,
                self.current_length_data[0].item(), max_kv_len,
                tokens_per_step=self.eagle_layer.total_tokens + 1
            ):
                break
        
        # 计算平均值
        num_steps = max(step, 1)
        output_ids = input_ids[:, input_len:]
        avg_accept_len = total_accept_len.item() / num_steps
        avg_draft_time = total_draft_time / num_steps
        avg_update_time = total_update_time / num_steps
        avg_verify_time = total_verify_time / num_steps
        
        return output_ids, avg_accept_len, 0, avg_draft_time, avg_update_time, avg_verify_time

    def generate_specsot(
        self,
        task_prompt: str,
        max_new_tokens: int,
        logits_processor: Optional[LogitsProcessorList] = None,
        use_semantic_constraint: bool = False,
    ) -> Tuple[torch.Tensor, float, int, float, float, float, float, float]:
        """
        骨架并行生成模式
        
        三阶段流程：
        1. Skeleton Generation - 生成回答骨架
        2. Skeleton Parsing - 使用 parser 解析骨架
        3. Parallel Decoding - 并行解码各分支
        
        Args:
            task_prompt: 用户输入的任务描述
            max_new_tokens: 最大生成 token 数
            logits_processor: 采样相关的 logits processor
            use_semantic_constraint: 是否使用语义约束 (FSM 状态机)
        
        Returns:
            output_ids, avg_accept_len, num_para, avg_draft_time, avg_update_time, avg_verify_time, skeleton_time, parallel_time
        """
        device = self.base_model.device
        model_type = self.get_model_type()
        
        # 阶段时间统计
        skeleton_time = 0.0
        parallel_time = 0.0
        evt_start_skeleton = torch.cuda.Event(enable_timing=True) # 骨架开始
        evt_end_skeleton = torch.cuda.Event(enable_timing=True)   # 骨架结束
        evt_start_parallel = torch.cuda.Event(enable_timing=True) # 并行开始 
        evt_end_parallel = torch.cuda.Event(enable_timing=True)   # 并行结束 
        
        # =====================================================================
        # Stage 1: Skeleton Generation
        # =====================================================================
        # 记录Skeleton阶段开始时间
        evt_start_skeleton.record()

        input_ids, task_input_ids = prepare_skeleton_input(
            self.tokenizer, task_prompt, model_type, device
        )
        input_len = input_ids.shape[1]
        
        # 根据参数决定是否使用 FSM 状态机约束
        if use_semantic_constraint:
            self._semantic_processor.configure(prefix_len=input_len, enforce_format=True)
            skeleton_logits_processor = LogitsProcessorList([self._semantic_processor])
            if logits_processor is not None:
                for p in logits_processor:
                    skeleton_logits_processor.append(p)
            self.logger.info("使用 FSM 语义约束进行骨架生成")
        else:
            skeleton_logits_processor = logits_processor
            self.logger.info("不使用 FSM 语义约束，直接生成骨架")
        
        # 初始化 KV Cache
        max_kv_len = input_len + max_new_tokens + 100
        self.past_key_values, self.past_key_values_data, self.current_length_data = \
            initialize_past_key_values(self.base_model, max_length=max_kv_len)
        
        # ---------------------------------------------------------------------
        # Stage 1.1: Prefill 阶段
        # ---------------------------------------------------------------------
        if self.is_distributed():
            draft_tokens, retrieve_indices, tree_mask, tree_position_ids, _, _, _ = \
                self.distributed_prefill_manager.prefill_single_distributed(
                    input_ids, self.past_key_values, skeleton_logits_processor
                )
        else:
            draft_tokens, retrieve_indices, tree_mask, tree_position_ids, _, _, _ = \
                self.prefill_single(input_ids, skeleton_logits_processor)
        set_random_seed(self.seed)

        # ---------------------------------------------------------------------
        # Stage 1.2: Skeleton Decode 循环
        # ---------------------------------------------------------------------
        # 骨架的停止条件：检测到 [END] 标记或 EOS
        eos_token_id = self.tokenizer.eos_token_id
        max_steps = 200  # 骨架长度限制
        
        for step in range(max_steps):
            (
                input_ids,
                draft_tokens,
                retrieve_indices,
                tree_mask,
                tree_position_ids,
                _,
            ) = self.decode_step_single(
                input_ids=input_ids,
                draft_tokens=draft_tokens,
                retrieve_indices=retrieve_indices,
                tree_mask=tree_mask,
                tree_position_ids=tree_position_ids,
                logits_processor=skeleton_logits_processor,
            )
            
            # 骨架的停止条件：检测是否生成了 [END] 或遇到 EOS
            generated_text = self.tokenizer.decode(input_ids[0, input_len:])
            if check_skeleton_stop(
                generated_text, eos_token_id, input_ids, input_len, 
                self.current_length_data[0].item(), max_kv_len,
                self.eagle_layer.total_tokens + 1
            ):
                break
        
        skeleton_ids = input_ids[:, input_len:]
        skeleton_text = self.tokenizer.decode(skeleton_ids[0], skip_special_tokens=False)
        self.logger.info(f"Generated Skeleton: {skeleton_text}")
        self.skeleton_output = skeleton_ids.clone()
        
        # 记录Skeleton阶段时间
        evt_end_skeleton.record()
        torch.cuda.synchronize()
        skeleton_time = evt_start_skeleton.elapsed_time(evt_end_skeleton) / 1000.0  # 转换为秒
        self.logger.info(f"Skeleton generation completed in {skeleton_time:.3f}s")

        # =====================================================================
        # Stage 2: Skeleton Parsing (骨架解析)
        # =====================================================================
        mode, content = parse_skeleton_output(skeleton_text)
        
        if mode == "direct":
            # 直接回答模式：不需要并行处理，直接返回骨架输出
            self.logger.info("Direct answer mode detected, returning skeleton output")
            return skeleton_ids, 0.0, 0, 0.0, 0.0, 0.0, skeleton_time, 0.0
        
        elif mode == "error":
            self.logger.warning(f"Skeleton parsing error: {content}")
            return skeleton_ids, 0.0, 0, 0.0, 0.0, 0.0, skeleton_time, 0.0
        
        # mode == "plan": 规划模式
        tasks = content
        num_para = len(tasks)
        self.logger.info(f"Detected {num_para} parallel tasks: {[t['title'] for t in tasks]}")
        
        # 准备并行分支输入
        clean_branches, instruction_len = prepare_parallel_branches(
            self.tokenizer, tasks, skeleton_text, model_type, task_prompt
        )
        
        self.parallel_branches_output = [list(br) for br in clean_branches]
        self.instruction_len = instruction_len
        predicted_branch_lengths = [t['length'] for t in tasks]
        self.logger.info(f"Predicted lengths: {predicted_branch_lengths}")

        # =====================================================================
        # Stage 3: Parallel Decoding (并行分支解码)
        # =====================================================================
        evt_start_parallel.record()
        # ---------------------------------------------------------------------
        # Stage 3.1: 前缀复用 - 复用 Skeleton KV Cache + 初始化并行状态
        # ---------------------------------------------------------------------
        input_ids, tips_indices, branch_begins, branch_lengths_actual, draft_input_ids = \
            self.reuse_prefix_for_parallel(task_input_ids, clean_branches, max_new_tokens)
        
        # ---------------------------------------------------------------------
        # Stage 3.2: Prefill 阶段 (并行)
        # ---------------------------------------------------------------------
        prefix_len = task_input_ids.shape[1]
        input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, hidden_states = \
            self.prefill_parallel(
                prefix_len, input_ids, tips_indices,
                branch_begins, branch_lengths_actual, draft_input_ids, logits_processor
            )
        
        # ---------------------------------------------------------------------
        # Stage 3.3: Parallel Decode 循环
        # ---------------------------------------------------------------------
        total_accept_len_parallel = torch.zeros(1, dtype=torch.long, device=device)
        total_verify_time_parallel = 0.0
        total_update_time_parallel = 0.0
        total_draft_time_parallel = 0.0
        
        tokens_per_branch = self.eagle_layer.total_tokens + 1
        
        evt_start_p = torch.cuda.Event(enable_timing=True)
        evt_after_verify_p = torch.cuda.Event(enable_timing=True)
        evt_after_update_p = torch.cuda.Event(enable_timing=True)
        evt_after_draft_p = torch.cuda.Event(enable_timing=True)
        
        for step_parallel in range(max_new_tokens):
            if check_stop_conditions_parallel(
                current_length=self.current_length_data[0].item(),
                max_kv_len=max_kv_len,
                num_active_branches=len(self.active_branches),
                tokens_per_branch=tokens_per_branch,
            ):
                self.logger.warning(f"Incomplete branches due to KV cache limit: {self.active_branches}")
                break
            
            if step_parallel % 50 == 0:
                self.logger.debug(f"Parallel Decoding Step {step_parallel + 1}")

            evt_start_p.record()
            
            (
                draft_tokens,
                retrieve_indices,
                tree_mask,
                tree_position_ids,
                accept_length,
                all_finished,
            ) = self.decode_step_parallel(
                draft_tokens=draft_tokens,
                retrieve_indices=retrieve_indices,
                tree_mask=tree_mask,
                tree_position_ids=tree_position_ids,
                logits_processor=logits_processor,
                evt_after_verify=evt_after_verify_p,
                evt_after_update=evt_after_update_p,
            )
            evt_after_draft_p.record()
            
            total_accept_len_parallel += accept_length.sum()
            
            torch.cuda.synchronize()
            total_verify_time_parallel += evt_start_p.elapsed_time(evt_after_verify_p) / 1000
            total_update_time_parallel += evt_after_verify_p.elapsed_time(evt_after_update_p) / 1000
            total_draft_time_parallel += evt_after_update_p.elapsed_time(evt_after_draft_p) / 1000
            
            if all_finished:
                self.logger.info("All branches finished generation.")
                break
        
        # 计算统计
        num_steps_parallel = max(step_parallel, 1)
        avg_accept_len = total_accept_len_parallel.item() / num_steps_parallel
        avg_draft_time = total_draft_time_parallel / num_steps_parallel
        avg_update_time = total_update_time_parallel / num_steps_parallel
        avg_verify_time = total_verify_time_parallel / num_steps_parallel
        
        self.logger.info(f"Avg accepted lengths: {avg_accept_len:.2f}, "
                        f"Avg draft time: {avg_draft_time:.4f}s, "
                        f"Avg update time: {avg_update_time:.4f}s, "
                        f"Avg verify time: {avg_verify_time:.4f}s")
        
        # 记录Parallel阶段时间
        evt_end_parallel.record()
        torch.cuda.synchronize()
        parallel_time = evt_start_parallel.elapsed_time(evt_end_parallel) / 1000.0  # 转换为秒
        self.logger.info(f"Parallel decoding completed in {parallel_time:.3f}s")

        # 合并结果
        merged_ids = merge_outputs(
            skeleton_output=self.skeleton_output,
            parallel_branches_output=self.parallel_branches_output,
            instruction_len=self.instruction_len,
            device=device,
            tasks=tasks,
            tokenizer=self.tokenizer,
        )
        
        return merged_ids, avg_accept_len, num_para, avg_draft_time, avg_update_time, avg_verify_time, skeleton_time, parallel_time

 