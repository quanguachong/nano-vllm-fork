"""
Attention 层对 KV cache 的使用方式：

1) 写 cache（本步算出的 K/V 落盘）：
   - 每层 forward 得到 q,k,v 后，若已分配 k_cache/v_cache，则调用 store_kvcache(k, v, k_cache, v_cache, slot_mapping)。
   - slot_mapping[i] 表示第 i 个 token 的 K/V 应写入的物理 slot 下标；Triton kernel 按 slot 写入 k_cache/v_cache。

2) 读 cache（用历史 K/V 做 attention）：
   - Prefill：本步的 K/V 在当次前向中已算得，直接做 flash_attn_varlen_func(q,k,v,...)；若有 prefix cache，
     则通过 block_tables 从 cache 中取历史 K/V 与当前 K/V 一起参与计算。
   - Decode：本步只有 Q（当前 token），历史 K/V 存在 k_cache/v_cache 里。context_lens 与 block_tables 不存 K/V，
     而是寻址参数：block_tables[i] 表示第 i 条序列的 token 分布在哪些物理 block；context_lens[i] 表示该序列
     当前长度（要读的 token 数）。Flash Attention 根据二者从 k_cache/v_cache 中定位并读取各序列的历史 K/V，与 Q 做 attention。
"""
import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        # 占位，numel()==0 时 forward 不写 cache。真实 KV cache 由 ModelRunner.allocate_kv_cache() 注入：
        # 遍历 model 中带 k_cache/v_cache 的模块，赋值为预分配大 tensor 的切片（见 model_runner.py）
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        # 写 cache：把本步算出的 K/V 按 slot_mapping 写入 k_cache/v_cache（prefill 写多 token，decode 写 1 token）
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            # Prefill：用本步的 q,k,v 做变长 attention；若有 prefix cache 则 K/V 来自 cache（block_tables 定位）
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode：历史 K/V 在 k_cache/v_cache；context_lens/block_tables 告诉 flash-attn 各序列在 cache 中的位置与长度
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True)
        return o
