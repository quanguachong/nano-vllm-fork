import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:
    """
    nano-vllm 的核心引擎：负责把「请求」交给调度器，驱动模型运行并收集输出。

    - 【关键】多进程/多卡：通过 torch.multiprocessing + spawn 启动子进程做张量并行
    - 【关键】调度：Scheduler 决定本轮运行哪些 Sequence（prefill 或 decode）
    - 【关键】执行：ModelRunner 负责真正调用模型前向、返回生成 token
    """

    def __init__(self, model, **kwargs):
        # 从 kwargs 里筛出 Config 支持的字段，避免传入无关参数
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)

        # 【关键】多进程张量并行：rank=0 在主进程，其余 rank 用子进程启动
        # 使用 spawn 能避免 fork 带来的 CUDA 上下文/线程状态继承问题（更安全、更通用）
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            # 事件用于进程间的简单同步/信号通知（由 ModelRunner 的实现决定如何使用）
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)

        # rank=0 的执行器在主进程中创建；同时把事件列表交给它管理/协调
        self.model_runner = ModelRunner(config, 0, self.events)

        # tokenizer 用于：把字符串 prompt 编码成 token_ids；以及把输出 token_ids 解码成文本
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        # eos 由 tokenizer 决定（不同模型可能不同）；写回 config 供调度/停止条件使用
        config.eos = self.tokenizer.eos_token_id

        # 【关键】调度器：维护所有请求的状态与 KV/块资源分配，并决定每一步的执行批次
        self.scheduler = Scheduler(config)

        # 注册退出清理：确保进程能正确 join，避免僵尸进程
        atexit.register(self.exit)

    def exit(self):
        # 通知模型执行器退出（具体清理逻辑由 ModelRunner 实现）
        self.model_runner.call("exit")
        del self.model_runner
        # 等待所有子进程退出
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        # 支持两种输入：
        # - str：自动用 tokenizer 编码
        # - list[int]：用户已提前编码好的 token_ids
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        # 【关键】Sequence 表示“一条生成请求”的完整状态（prompt、采样参数、已生成 token 等）
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        # 【关键】一次引擎步进：调度 -> 执行 -> 后处理（更新序列状态/完成情况）
        seqs, is_prefill = self.scheduler.schedule()
        # 调用模型：prefill 阶段会“吃掉”整段 prompt；decode 阶段通常每个序列生成 1 个 token
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        # 后处理：把新 token 写回 Sequence，检查是否满足停止条件等
        self.scheduler.postprocess(seqs, token_ids)

        # 只返回已完成的序列（seq.is_finished 为 True）
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]

        # 【关键】吞吐统计：
        # - prefill：统计本轮“处理了多少 token”（通常是 prompt token 数），用正数表示
        # - decode：统计本轮“生成了多少 token”（通常每序列 1 个），这里用负数表示以便区分阶段
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens

    def is_finished(self):
        # 是否所有请求都已完成
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        # 【关键】tqdm 是进度条库：这里的 total 是请求数（不是 token 数）
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)

        # sampling_params 支持：
        # - 单个 SamplingParams：复制成与 prompts 等长（每条请求用同一套采样参数）
        # - list[SamplingParams]：每条请求一套采样参数（需与 prompts 对齐）
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)

        # 把所有请求加入调度器队列
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)

        # outputs 用 seq_id 做 key：便于最终按请求加入顺序稳定输出
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            # 用 perf_counter 计时：用于计算本轮 prefill/decode 的 tok/s
            t = perf_counter()
            output, num_tokens = self.step()
            if use_tqdm:
                # 【关键】num_tokens 的正负号区分阶段（见 step() 注释）
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                # 在进度条右侧动态显示吞吐（tok/s）
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    # 【关键】这里按“完成的请求数”推进进度条（每完成 1 条请求 +1）
                    pbar.update(1)
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        # 最终输出：同时提供解码后的文本与 token_ids（方便调试/评估）
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        if use_tqdm:
            pbar.close()
        return outputs
