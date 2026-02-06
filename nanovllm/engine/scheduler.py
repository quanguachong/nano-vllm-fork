from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:
    """
    调度器（Scheduler）：负责两阶段调度（prefill/decode）与抢占（preempt）。
    
    核心职责：
    - 维护 waiting/running 两个队列，管理序列的生命周期
    - prefill 阶段：从 waiting 队列选择序列，分配 KV cache blocks，加入 running
    - decode 阶段：从 running 队列选择序列继续生成；资源不足时抢占其他序列
    - 后处理：检查停止条件（EOS/max_tokens），释放已完成的序列资源
    """

    def __init__(self, config: Config):
        # 【关键】调度约束参数
        self.max_num_seqs = config.max_num_seqs  # 单批次最大序列数
        self.max_num_batched_tokens = config.max_num_batched_tokens  # 单批次最大 token 数（prefill 阶段使用）
        self.eos = config.eos  # 结束符 token_id（用于停止条件判断）
        
        # 【关键】KV cache 块管理器：负责分配/释放 blocks，判断资源是否充足
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        
        # 【关键】两个队列：
        # - waiting: 等待调度的序列（尚未分配 KV cache blocks）
        # - running: 正在运行的序列（已分配 blocks，正在 prefill 或 decode）
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        """判断是否所有请求都已完成（waiting 和 running 队列都为空）"""
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        """添加新请求到等待队列（由 LLMEngine.add_request 调用）"""
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        """
        【关键】核心调度逻辑：两阶段调度（prefill 优先，decode 次之）
        
        调度策略：
        1. Prefill 阶段：优先处理 waiting 队列中的新请求
           - 约束：max_num_seqs、max_num_batched_tokens、BlockManager.can_allocate
           - 分配 KV cache blocks，将序列从 waiting 移到 running
        2. Decode 阶段：处理 running 队列中的序列继续生成
           - 约束：max_num_seqs、BlockManager.can_append
           - 资源不足时触发抢占（preempt）释放其他序列的 blocks
        
        Returns:
            (scheduled_seqs, is_prefill): 本轮调度的序列列表，以及是否为 prefill 阶段
        """
        # 【关键】Prefill 阶段：从 waiting 队列选择序列进行首次处理
        scheduled_seqs = []  # 本轮调度的序列列表（用于返回给模型执行器）
        
        # 【关键】num_seqs：当前批次中已调度的序列数量（累加器）
        # 
        # 作用：
        # - 用于约束批次大小，确保不超过 max_num_seqs 限制
        # - 在 prefill 和 decode 两个阶段都会使用（共享同一个计数器）
        # - prefill 阶段：从 waiting 队列选择序列，每选择一个 num_seqs += 1
        # - decode 阶段：从 running 队列选择序列，每选择一个 num_seqs += 1
        num_seqs = 0
        # 【关键】num_batched_tokens：当前 prefill 批次中"需要处理的 token 总数"（累加器）
        # 
        # 作用：
        # 1. 用于约束 prefill 阶段的批次大小，确保不超过 max_num_batched_tokens 限制
        # 2. 只统计"未缓存的 token 数"（len(seq) - seq.num_cached_tokens）
        #    因为已缓存的 token 通过 prefix cache 复用，不需要在 prefill 中重新计算
        # 
        # 示例：
        # - 序列 A：100 tokens，0 cached → 贡献 100 tokens
        # - 序列 B：200 tokens，150 cached → 贡献 50 tokens（只处理未缓存部分）
        # - 总计：num_batched_tokens = 150 tokens（而不是 300）
        num_batched_tokens = 0
        
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]  # 查看队列头部（FIFO）
            
            # 检查约束：token 数限制 + KV cache blocks 是否可分配
            # 【关键】判断：如果加入当前序列后，总 token 数是否超过限制
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break  # 不满足约束，停止调度
            
            # 满足约束，分配资源并加入运行队列
            num_seqs += 1
            self.block_manager.allocate(seq)  # 分配 KV cache blocks
            # 【关键】累加"未缓存的 token 数"到批次统计中
            # 注意：这里用 len(seq) 而不是 (len(seq) - seq.num_cached_tokens) 来判断是否超限，
            # 但在累加时只加未缓存部分，这是为了简化判断逻辑（保守估计）
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        
        if scheduled_seqs:
            return scheduled_seqs, True  # 返回 prefill 批次

        # 【关键】Decode 阶段：从 running 队列选择序列继续生成
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()  # FIFO 顺序
            
            # 【关键】抢占逻辑：如果当前序列无法追加 token（KV cache blocks 不足）
            while not self.block_manager.can_append(seq):
                if self.running:
                    # 抢占 running 队列中的其他序列（从尾部开始，LIFO）
                    self.preempt(self.running.pop())
                else:
                    # running 队列已空，抢占当前序列本身（极端情况）
                    self.preempt(seq)
                    break
            else:
                # 资源充足，可以追加 token
                num_seqs += 1
                self.block_manager.may_append(seq)  # 预分配下一个 block（如果需要）
                scheduled_seqs.append(seq)
        
        assert scheduled_seqs  # decode 阶段必须至少调度一个序列
        # 【关键】将调度的序列重新放回 running 队列头部（保持 FIFO 顺序）
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False  # 返回 decode 批次

    def preempt(self, seq: Sequence):
        """
        【关键】抢占（preempt）：释放序列的 KV cache blocks，将其重新加入 waiting 队列
        
        触发场景：decode 阶段资源不足时，抢占其他序列以释放 blocks。
        被抢占的序列会丢失已缓存的 KV（需要重新 prefill），但 prompt token_ids 保留。
        
        Args:
            seq: 被抢占的序列
        """
        seq.status = SequenceStatus.WAITING  # 重置状态
        self.block_manager.deallocate(seq)  # 释放 KV cache blocks
        self.waiting.appendleft(seq)  # 重新加入 waiting 队列头部（优先调度）

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        """
        【关键】后处理：将模型生成的 token 追加到序列，检查停止条件，清理已完成的序列
        
        停止条件（满足任一即完成）：
        1. 生成到 EOS token（且未设置 ignore_eos）
        2. 达到最大生成长度（max_tokens）
        
        Args:
            seqs: 本轮调度的序列列表
            token_ids: 模型生成的 token_id 列表（与 seqs 一一对应）
        
        Returns:
            每个序列是否已完成的布尔值列表（当前实现未使用返回值）
        """
        for seq, token_id in zip(seqs, token_ids):
            # 追加新生成的 token
            seq.append_token(token_id)
            
            # 【关键】检查停止条件
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                # 序列已完成，清理资源
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)  # 释放 KV cache blocks
                self.running.remove(seq)  # 从 running 队列移除
