from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    """
    序列状态枚举：用于跟踪每条生成请求在调度器中的生命周期。
    
    - WAITING：等待调度（刚加入队列，尚未分配 KV cache blocks）
    - RUNNING：正在运行（已分配 blocks，正在 prefill 或 decode）
    - FINISHED：已完成（达到停止条件，已释放 KV cache blocks）
    """
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    """
    序列（Sequence）：表示一条生成请求的完整状态。
    
    维护了 token 序列、KV cache 块表、缓存状态、采样参数等核心信息，
    是调度器、块管理器、模型执行器之间传递请求状态的主要数据结构。
    """
    # 【关键】硬编码的块大小：每个 KV cache block 固定包含 256 个 token
    # 注意：这个值必须与 Config.kvcache_block_size 的"256 倍数约束"保持一致
    block_size = 256
    # 全局计数器：为每条序列分配唯一 ID（用于跨进程标识与最终输出排序）
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        # 分配唯一序列 ID
        self.seq_id = next(Sequence.counter)
        # 初始状态：等待调度器分配 KV cache blocks
        self.status = SequenceStatus.WAITING
        # 【关键】完整的 token 序列（包含 prompt + 已生成的 completion tokens）
        # 使用 copy 避免外部修改影响内部状态
        self.token_ids = copy(token_ids)
        # 最后一个 token（用于 decode 阶段的输入，也用于跨进程序列化优化）
        self.last_token = token_ids[-1]
        # 当前总 token 数（prompt + completion）
        self.num_tokens = len(self.token_ids)
        # prompt token 数量（用于区分 prompt 与 completion 部分）
        self.num_prompt_tokens = len(token_ids)
        # 【关键】已缓存的 token 数量（用于 prefix cache 复用判断）
        # 当 num_cached_tokens < num_prompt_tokens 时，prefill 阶段需要处理未缓存部分
        self.num_cached_tokens = 0
        # 【关键】KV cache 块表（逻辑页表）：每个元素是 block_id，表示该序列占用的 KV cache blocks
        # 由 BlockManager 分配与更新，用于 flash-attn 的 block_table 参数
        self.block_table = []
        # 采样参数（从 SamplingParams 中提取，避免每次访问都查对象）
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        """使 Sequence 可以像列表一样使用 len() 获取总 token 数"""
        return self.num_tokens

    def __getitem__(self, key):
        """使 Sequence 可以像列表一样使用索引访问 token_ids"""
        return self.token_ids[key]

    @property
    def is_finished(self):
        """判断序列是否已完成（达到停止条件）"""
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        """已生成的 completion token 数量（不包含 prompt）"""
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        """返回 prompt 部分的 token_ids（用于 prefix cache 的 hash 计算等）"""
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        """返回 completion 部分的 token_ids（用于最终输出）"""
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self):
        """已缓存的 KV cache blocks 数量（用于 prefix cache 复用判断）"""
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self):
        """【关键】当前序列占用的总 block 数量（向上取整）
        
        用于 BlockManager 判断是否需要分配新 block，以及 flash-attn 的 block_table 长度。
        """
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        """最后一个 block 中的 token 数量（可能不满 256）
        
        用于判断是否需要分配新 block：当新 token 进入下一个 block 的第 1 个位置时分配。
        """
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        """获取第 i 个 block 的 token_ids（用于 prefix cache 的 hash 计算）
        
        Args:
            i: block 索引（0-based）
        
        Returns:
            该 block 对应的 token_ids 切片
        """
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        """【关键】追加新生成的 token（在 decode 阶段调用）
        
        更新 token_ids、last_token 和 num_tokens，为下一轮 decode 做准备。
        """
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        """【关键】跨进程序列化优化：自定义 pickle 序列化逻辑
        
        在 TP 多进程场景下，rank0 需要将 Sequence 状态广播给其他 rank。
        为了减少传输开销：
        - prefill 阶段（num_completion_tokens == 0）：传输完整的 token_ids（需要用于 prefill）
        - decode 阶段（num_completion_tokens > 0）：只传输 last_token（decode 只需要最后一个 token）
        
        Returns:
            序列化后的元组：(num_tokens, num_prompt_tokens, num_cached_tokens, block_table, token_ids_or_last_token)
        """
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table,
                self.token_ids if self.num_completion_tokens == 0 else self.last_token)

    def __setstate__(self, state):
        """【关键】跨进程序列化优化：自定义 pickle 反序列化逻辑
        
        根据序列化时的策略，恢复 Sequence 状态：
        - prefill 阶段：恢复完整的 token_ids
        - decode 阶段：只恢复 last_token（token_ids 的其他部分不需要）
        
        Args:
            state: 由 __getstate__ 返回的元组
        """
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table = state[:-1]
        if self.num_completion_tokens == 0:
            # prefill 阶段：恢复完整 token_ids
            self.token_ids = state[-1]
        else:
            # decode 阶段：只恢复 last_token（token_ids 的其他部分在 rank0 本地维护，不需要传输）
            self.last_token = state[-1]
