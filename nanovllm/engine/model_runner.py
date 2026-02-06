import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:
    """
    模型执行器：负责在 GPU 上运行模型，处理 prefill 和 decode 阶段的前向计算。
    
    核心功能：
    - 多进程/多卡支持：通过 SharedMemory + Event 实现主进程与子进程的通信
    - KV cache 管理：一次性预分配所有 KV cache blocks 的显存
    - CUDA Graph 优化：在 decode 阶段使用 CUDA Graph 加速（可选）
    - 采样：从模型输出的 logits 中采样生成 token
    """

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        """
        初始化模型执行器
        
        Args:
            config: 配置对象，包含模型路径、并行度等参数
            rank: 当前进程的 rank（0 为主进程，>0 为子进程）
            event: 进程间同步事件（rank=0 时为事件列表，rank>0 时为单个事件）
        """
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size  # KV cache block 大小（通常 256）
        self.enforce_eager = config.enforce_eager  # 是否禁用 CUDA Graph 优化
        self.world_size = config.tensor_parallel_size  # 张量并行度（GPU 数量）
        self.rank = rank  # 当前进程的 rank
        self.event = event  # 进程间同步事件

        # 【关键】初始化分布式进程组（用于张量并行）
        # 使用 NCCL 后端，通过 TCP 连接（localhost:2333）
        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)  # 设置当前进程使用的 GPU
        
        # 临时切换到模型的数据类型和设备
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        
        # 加载模型和权重
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()  # 采样器（用于从 logits 生成 token）
        
        # 预热模型（触发 CUDA kernel 编译，避免首次运行延迟）
        self.warmup_model()
        
        # 【关键】分配 KV cache 显存（一次性预分配所有 blocks）
        self.allocate_kv_cache()
        
        # 如果启用 CUDA Graph，捕获计算图（用于 decode 阶段加速）
        if not self.enforce_eager:
            self.capture_cudagraph()
        
        # 恢复默认数据类型和设备
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        # 【关键】多进程通信设置
        if self.world_size > 1:
            if rank == 0:
                # rank=0（主进程）：创建共享内存，用于向子进程发送指令
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)  # 1MB 共享内存
                dist.barrier()  # 等待所有进程同步
            else:
                # rank>0（子进程）：连接到共享内存，进入循环等待主进程指令
                dist.barrier()  # 等待主进程创建共享内存
                self.shm = SharedMemory(name="nanovllm")
                self.loop()  # 进入循环，等待并执行主进程发送的指令

    def exit(self):
        """
        清理资源：关闭共享内存、销毁 CUDA Graph、销毁进程组
        """
        if self.world_size > 1:
            self.shm.close()  # 关闭共享内存连接
            dist.barrier()  # 等待所有进程完成清理
            if self.rank == 0:
                self.shm.unlink()  # rank=0 负责删除共享内存
        if not self.enforce_eager:
            del self.graphs, self.graph_pool  # 释放 CUDA Graph 资源
        torch.cuda.synchronize()  # 等待所有 CUDA 操作完成
        dist.destroy_process_group()  # 销毁分布式进程组

    def loop(self):
        """
        【关键】子进程主循环：持续等待主进程发送的指令并执行
        
        只在 rank > 0 的子进程中调用，主进程不会进入此循环。
        子进程通过共享内存接收主进程发送的方法调用指令，执行后继续等待。
        """
        while True:
            method_name, args = self.read_shm()  # 从共享内存读取方法名和参数
            self.call(method_name, *args)  # 执行方法
            if method_name == "exit":
                break  # 收到退出指令，退出循环

    def read_shm(self):
        """
        【关键】从共享内存读取主进程发送的方法调用指令（仅子进程调用）
        
        协议格式：
        - 前 4 字节：数据长度（小端）
        - 后续字节：pickle 序列化的 [method_name, *args]
        
        Returns:
            (method_name, args): 方法名和参数元组
        """
        assert self.world_size > 1 and self.rank > 0  # 仅子进程调用
        self.event.wait()  # 等待主进程发送信号
        n = int.from_bytes(self.shm.buf[0:4], "little")  # 读取数据长度
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])  # 反序列化数据
        self.event.clear()  # 清除事件标志，准备下次等待
        return method_name, args

    def write_shm(self, method_name, *args):
        """
        【关键】将方法调用指令写入共享内存（仅主进程调用）
        
        协议格式：
        - 前 4 字节：数据长度（小端）
        - 后续字节：pickle 序列化的 [method_name, *args]
        
        写入后通过 Event 通知所有子进程读取并执行。
        
        Args:
            method_name: 要调用的方法名（字符串）
            *args: 传递给方法的参数
        """
        assert self.world_size > 1 and self.rank == 0  # 仅主进程调用
        data = pickle.dumps([method_name, *args])  # 序列化方法名和参数
        n = len(data)  # 计算数据长度
        self.shm.buf[0:4] = n.to_bytes(4, "little")  # 写入长度（前 4 字节）
        self.shm.buf[4:n+4] = data  # 写入实际数据
        for event in self.event:
            event.set()  # 通知所有子进程有新数据

    def call(self, method_name, *args):
        """
        【关键】统一的方法调用接口：支持单进程和多进程场景
        
        单进程（world_size == 1）：直接调用方法
        多进程（world_size > 1）：
            - rank=0：写入共享内存通知子进程，然后本地执行
            - rank>0：已在 loop() 中通过 read_shm 获取指令，直接执行
        
        Args:
            method_name: 方法名（字符串）
            *args: 方法参数
            
        Returns:
            方法返回值
        """
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)  # 主进程：通知子进程
        method = getattr(self, method_name, None)  # 获取方法对象
        return method(*args)  # 执行方法

    def warmup_model(self):
        """
        预热模型：运行一次前向传播，触发 CUDA kernel 编译
        
        避免首次运行时因 kernel 编译导致的延迟。使用最大序列长度和最大批次大小
        进行预热，确保后续运行都能命中已编译的 kernel。
        """
        torch.cuda.empty_cache()  # 清空 CUDA 缓存
        torch.cuda.reset_peak_memory_stats()  # 重置峰值内存统计
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]  # 创建虚拟序列
        self.run(seqs, True)  # 运行一次 prefill 预热
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        """
        【关键】分配 KV cache 显存：一次性预分配所有 blocks 的物理存储
        
        计算逻辑：
        1. 计算每个 block 的字节数：2（K+V）* num_layers * block_size * num_kv_heads * head_dim * dtype_size
        2. 根据 GPU 显存利用率计算可分配的 block 数量
        3. 创建大 tensor：[2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
        4. 将每个 attention 层的 k_cache 和 v_cache 指向这个大 tensor 的对应切片
        
        注意：这是物理显存分配，逻辑上的 block 分配由 BlockManager 管理。
        """
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()  # 获取 GPU 显存信息
        used = total - free  # 已使用的显存
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]  # 峰值显存
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]  # 当前显存
        
        # 计算每个 rank 的 KV heads 数量（张量并行时每个 GPU 只处理部分 heads）
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        
        # 计算每个 block 的字节数：K cache + V cache
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize
        
        # 根据 GPU 显存利用率计算可分配的 block 数量
        # 公式：可用显存 = total * utilization - used - peak + current
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0  # 确保至少能分配一个 block
        
        # 【关键】一次性分配所有 KV cache blocks 的显存
        # 形状：[2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
        # 2 表示 K cache 和 V cache
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        
        # 将每个 attention 层的 k_cache 和 v_cache 指向这个大 tensor 的对应切片
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]  # K cache: [num_blocks, block_size, num_kv_heads, head_dim]
                module.v_cache = self.kv_cache[1, layer_id]  # V cache: [num_blocks, block_size, num_kv_heads, head_dim]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        """
        准备 block_table：将序列的逻辑 block_id 列表转换为 tensor
        
        每个序列的 block_table 长度可能不同，需要补齐到相同长度（用 -1 填充），
        以便批处理。block_table 用于 flash-attn 查找 KV cache 的物理位置。
        
        Args:
            seqs: 序列列表
            
        Returns:
            block_tables: [batch_size, max_num_blocks] 的 tensor，每行是一个序列的 block_table
        """
        max_len = max(len(seq.block_table) for seq in seqs)  # 找到最长的 block_table
        # 补齐到相同长度，用 -1 填充（-1 表示无效 block）
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        """
        【关键】准备 prefill 阶段的输入：处理 prompt tokens，设置 KV cache 写入位置
        
        核心逻辑：
        1. 收集所有序列的未缓存 token（跳过已通过 prefix cache 缓存的 token）
        2. 计算每个序列的 query 和 key 长度（query 只包含未缓存部分，key 包含全部）
        3. 计算 slot_mapping：将每个 token 映射到 KV cache 的物理 slot 位置
        4. 如果存在 prefix cache，准备 block_table 用于 flash-attn
        
        Args:
            seqs: 序列列表
            
        Returns:
            (input_ids, positions): 输入 token IDs 和位置编码
        """
        input_ids = []
        positions = []
        cu_seqlens_q = [0]  # query 的累积长度（用于 flash-attn）
        cu_seqlens_k = [0]  # key 的累积长度（用于 flash-attn）
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []  # token 到 KV cache slot 的映射
        block_tables = None
        
        for seq in seqs:
            seqlen = len(seq)
            # 只处理未缓存的 token（已缓存的通过 prefix cache 复用，不需要重新计算）
            input_ids.extend(seq[seq.num_cached_tokens:])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            
            seqlen_q = seqlen - seq.num_cached_tokens  # query 长度（未缓存部分）
            seqlen_k = seqlen  # key 长度（全部，包括缓存的）
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            
            if not seq.block_table:  # warmup 阶段，没有 block_table
                continue
            
            # 计算 slot_mapping：将每个 token 映射到 KV cache 的物理 slot
            # slot = block_id * block_size + offset_in_block
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size  # block 的起始 slot
                if i != seq.num_blocks - 1:
                    end = start + self.block_size  # 完整 block
                else:
                    end = start + seq.last_block_num_tokens  # 最后一个 block 可能不满
                slot_mapping.extend(list(range(start, end)))
        
        # 如果存在 prefix cache（cu_seqlens_k > cu_seqlens_q），需要 block_table
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
            block_tables = self.prepare_block_tables(seqs)
        
        # 转换为 tensor 并传输到 GPU
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        
        # 设置全局上下文（供 attention 层使用）
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        """
        【关键】准备 decode 阶段的输入：每个序列只处理最后一个 token
        
        核心逻辑：
        1. 每个序列只取最后一个 token（decode 阶段每次生成一个 token）
        2. 计算每个序列的 context_len（总长度，用于 flash-attn）
        3. 计算 slot_mapping：新 token 在 KV cache 中的写入位置
        4. 准备 block_table：用于 flash-attn 查找历史 KV cache
        
        Args:
            seqs: 序列列表
            
        Returns:
            (input_ids, positions): 输入 token IDs 和位置编码
        """
        input_ids = []
        positions = []
        slot_mapping = []  # 新 token 的 KV cache 写入位置
        context_lens = []  # 每个序列的总长度（用于 flash-attn）
        
        for seq in seqs:
            input_ids.append(seq.last_token)  # 只取最后一个 token
            positions.append(len(seq) - 1)  # 最后一个 token 的位置
            context_lens.append(len(seq))  # 序列总长度
            
            # 计算新 token 的 KV cache 写入位置
            # slot = 最后一个 block 的起始位置 + block 内的偏移
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)
        
        # 转换为 tensor 并传输到 GPU
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        
        # 设置全局上下文（供 attention 层使用）
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        """
        准备采样参数：收集每个序列的 temperature
        
        Args:
            seqs: 序列列表
            
        Returns:
            temperatures: [batch_size] 的 tensor，每个序列的 temperature
        """
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        """
        【关键】运行模型前向传播：根据阶段选择不同的执行路径
        
        执行路径：
        1. Prefill 阶段：直接运行（因为序列长度变化大，不适合 CUDA Graph）
        2. Decode 阶段 + 启用 CUDA Graph + 批次大小 <= 512：使用 CUDA Graph 加速
        3. 其他情况：直接运行（eager mode）
        
        Args:
            input_ids: 输入 token IDs
            positions: 位置编码
            is_prefill: 是否为 prefill 阶段
            
        Returns:
            logits: 模型输出的 logits [batch_size, vocab_size]（decode）或 [total_tokens, vocab_size]（prefill）
        """
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            # Prefill 阶段或禁用 CUDA Graph 或批次过大：直接运行
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            # Decode 阶段 + CUDA Graph：使用预捕获的计算图加速
            bs = input_ids.size(0)
            context = get_context()
            
            # 选择合适大小的 CUDA Graph（选择 >= bs 的最小 graph）
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            
            # 更新 graph 的输入变量
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            
            # 重放 CUDA Graph（执行预捕获的计算图，避免 Python 开销）
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        """
        【关键】运行模型并采样生成 token：prefill 和 decode 的统一入口
        
        执行流程：
        1. 准备输入（prefill 或 decode）
        2. 运行模型前向传播
        3. 采样生成 token（仅 rank=0，其他 rank 返回 None）
        4. 重置上下文
        
        Args:
            seqs: 序列列表
            is_prefill: 是否为 prefill 阶段
            
        Returns:
            token_ids: 每个序列新生成的 token ID 列表（rank=0）或 None（rank>0）
        """
        # 准备输入：根据阶段选择不同的准备方法
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        
        # 准备采样参数（仅 rank=0 需要，其他 rank 不参与采样）
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        
        # 运行模型前向传播
        logits = self.run_model(input_ids, positions, is_prefill)
        
        # 采样生成 token（仅 rank=0，其他 rank 返回 None）
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        
        # 重置全局上下文
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        """
        【关键】捕获 CUDA Graph：为不同批次大小预捕获计算图，用于 decode 阶段加速
        
        CUDA Graph 优化原理：
        - 将模型的计算流程捕获为静态计算图
        - 运行时直接重放（replay）计算图，避免 Python 解释器开销
        - 显著提升 decode 阶段的吞吐量（通常 2-3x 加速）
        
        捕获策略：
        - 为多个批次大小（1, 2, 4, 8, 16, 32, ...）分别捕获 graph
        - 运行时选择 >= 实际批次大小的最小 graph
        - 使用 graph pool 共享内存，减少内存占用
        
        注意：只在 decode 阶段使用 CUDA Graph，prefill 阶段序列长度变化大，不适合。
        """
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)  # 最大批次大小（限制为 512）
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size  # 最大 block 数
        
        # 创建固定大小的输入/输出 tensor（CUDA Graph 需要固定大小的 tensor）
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        
        # 定义要捕获的批次大小列表：[1, 2, 4, 8, 16, 32, 48, 64, ...]
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}  # 存储不同批次大小的 graph
        self.graph_pool = None  # graph 内存池（共享内存）

        # 从大到小捕获 graph（确保 graph_pool 足够大）
        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            
            # 设置上下文（decode 模式）
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            
            # Warmup：运行一次，确保所有 CUDA kernel 已编译
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
            
            # 捕获计算图
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
            
            # 第一个 graph 创建 pool，后续 graph 共享这个 pool
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            
            self.graphs[bs] = graph  # 保存 graph
            torch.cuda.synchronize()  # 确保捕获完成
            reset_context()

        # 保存 graph 变量（运行时通过修改这些变量来更新输入）
        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
