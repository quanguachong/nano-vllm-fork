"""
张量并行（TP）线性层实现。

本模块提供多种线性层，用于在多 GPU 张量并行下切分权重与计算：
- ReplicatedLinear：不切分，每 rank 持有一份完整权重（用于不需并行的层，如部分归一化后的投影）。
- ColumnParallelLinear：沿输出维（dim=0）切分，每 rank 只存、只算自己的输出分片；上游输入完整，下游需配合 RowParallel 或 all_reduce。
- RowParallelLinear：沿输入维（dim=1）切分，每 rank 只存、只算自己的输入分片；forward 后对输出做 all_reduce 得到完整结果。
- MergedColumnParallelLinear：多个「输出块」合并成一个 ColumnParallel 线性层，加载时按块（如 gate/up）分别传入权重。
- QKVParallelLinear：将 Q、K、V 三部分合并为一个 ColumnParallel 线性层，按 head 维切分到各 rank，用于 Attention 的 qkv_proj。
"""
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist


def divide(numerator, denominator):
    """整除，且要求能除尽（用于 TP 时保证各 rank 分片大小一致）。"""
    assert numerator % denominator == 0
    return numerator // denominator


class LinearBase(nn.Module):
    """
    TP 线性层的基类：维护 input_size、output_size、tp_dim，以及可选的 weight_loader 钩子。
    
    - tp_dim：参与 TP 切分的维度，0 表示沿 output 维切（ColumnParallel），1 表示沿 input 维切（RowParallel），None 表示不切分。
    - weight_loader：在加载权重时由 loader 工具调用，子类可重写以实现「只加载本 rank 分片」等逻辑。
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
    ):
        super().__init__()
        self.tp_dim = tp_dim  # 张量并行切分维度：0=输出维，1=输入维，None=不切分
        self.tp_rank = dist.get_rank()  # 当前进程在 TP 组内的 rank
        self.tp_size = dist.get_world_size()  # TP 总进程数（GPU 数）
        # 权重形状为 (output_size, input_size)，与 F.linear 约定一致
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader  # 供外部 loader 按层类型调用
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ReplicatedLinear(LinearBase):
    """
    复制线性层：不参与 TP 切分，每个 rank 持有一份完整的 weight/bias。
    
    用于不需要并行的层（如部分 LayerNorm 后的投影）。tp_dim 为 None，父类中 output_size 即完整维度。
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size, bias, tp_dim=None)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """直接整份拷贝，不做分片。"""
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class ColumnParallelLinear(LinearBase):
    """
    列并行线性层：沿**输出维**（第 0 维）切分到各 rank。
    
    每个 rank 的 weight 形状为 (output_size // tp_size, input_size)，即只存、只算自己的输出分片。
    输入 x 在各 rank 上相同（完整），输出为分片，通常下游会接 RowParallelLinear，在那一层做 all_reduce 得到完整向量。
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        # 父类中 output_size 传「本 rank 的分片大小」；tp_dim=0 表示沿第 0 维（输出维）切分
        super().__init__(input_size, divide(output_size, tp_size), bias, tp_dim=0)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """
        加载本 rank 对应的权重分片。
        loaded_weight 为完整权重 (full_output, input_size)，按第 0 维切为 tp_size 份，本 rank 取第 tp_rank 份。
        """
        param_data = param.data  # 本 rank 的 (shard_output, input_size)
        shard_size = param_data.size(self.tp_dim)  # 本 rank 在输出维上的长度
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., input_size) → 输出 (..., output_size // tp_size)，为完整输出的一个分片。"""
        return F.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):
    """
    合并多输出的列并行线性层：将多个「输出块」在输出维上拼成一个大矩阵，整体做 ColumnParallel 切分。
    
    例如 gate_proj + up_proj 合并为 (gate_size + up_size, hidden)，再按 tp 切分。
    加载权重时按块传入（loaded_shard_id 区分是第几个块），每块再按 tp 切分后写入对应区间。
    """

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ):
        self.output_sizes = output_sizes  # 各块的输出维，如 [gate_size, up_size]
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        """
        加载第 loaded_shard_id 块对应的分片。
        param 在输出维上对应整块合并后的矩阵；块内再按 tp_rank 取分片，写入 param 的对应 narrow 区间。
        """
        param_data = param.data
        # 当前块在「合并后输出维」上的偏移（各块也按 tp 切分，所以偏移要除以 tp_size）
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):
    """
    Q/K/V 合并的列并行线性层：用于 Attention 的 qkv_proj，输出维为 [Q; K; V] 拼接。
    
    按 head 维切分：每个 rank 拥有 total_num_heads/tp_size 个 Q head、total_num_kv_heads/tp_size 个 K/V head，
    权重加载时根据 loaded_shard_id in ["q","k","v"] 分别从完整权重的对应段切出本 rank 分片。
    """

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)  # 本 rank 的 Q head 数
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)  # 本 rank 的 K/V head 数
        # 合并输出维 = Q + K + V（按 head 数 * head_size）
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        """
        按 Q/K/V 分别加载：loaded_shard_id 为 "q"、"k" 或 "v"。
        完整权重中 Q/K/V 在输出维上连续排列，各自再按 tp 切分后写入 param 的对应区间。
        """
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class RowParallelLinear(LinearBase):
    """
    行并行线性层：沿**输入维**（第 1 维）切分到各 rank。
    
    每个 rank 的 weight 形状为 (output_size, input_size // tp_size)，即只存、只算自己那部分输入；
    上游通常为 ColumnParallelLinear，各 rank 的输入 x 已是分片。forward 后对输出做 all_reduce，
    得到完整的 (..., output_size)，且只在 rank 0 上保留 bias（避免重复加）。
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        # 父类中 input_size 传「本 rank 的输入分片大小」；tp_dim=1 表示沿第 1 维（输入维）切分
        super().__init__(divide(input_size, tp_size), output_size, bias, tp_dim=1)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """
        加载本 rank 对应的权重分片。
        完整权重 (output_size, full_input_size) 按第 1 维切为 tp_size 份，本 rank 取第 tp_rank 份。
        """
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        各 rank 用本分片权重计算后得到相同形状的 (..., output_size)，再 all_reduce 求和得到完整输出。
        bias 仅由 rank 0 持有，避免多 rank 重复加 bias。
        """
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y
