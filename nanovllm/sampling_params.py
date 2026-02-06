from dataclasses import dataclass


@dataclass
class SamplingParams:
    """
    采样参数（Sampling Parameters）。

    用于控制文本生成时“如何从模型输出的概率分布中选 token”，以及生成长度/停止相关行为。
    这些参数会被挂在每条请求的 `Sequence` 上，供调度与模型执行阶段读取。
    """
    temperature: float = 1.0
    # 最多生成多少个 completion token（不包含 prompt token）
    max_tokens: int = 64
    # 是否忽略 eos（若为 True，即使生成到 eos 也继续生成，直到 max_tokens）
    ignore_eos: bool = False

    def __post_init__(self):
        # 【关键】本项目不允许 greedy（temperature≈0）：
        # 通过限制 temperature 下界，避免退化到确定性 argmax 采样路径。
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
