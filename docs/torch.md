# torch

## package dist

dist.all_reduce: 所有进程把各自的 tensor 做一次“聚合计算”，然后每个进程都拿到相同的结果

dist.barrier: 所有进程在这里“集合”，谁都不能往下走，直到全部到齐

dist.gather: 把所有 rank 的 tensor 收集到 指定 rank（root）

dist.get_world_size: 当前分布式进程组里一共有多少个进程