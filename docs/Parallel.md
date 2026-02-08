# nano-vllm 并行设计

本文档说明 nano-vllm 如何解决**多卡模型并行**与**多请求批并行**，以及 prefill/decode 的执行路径差异。

---

## 1. 概述

nano-vllm 的并行可从三个维度理解：

| 维度 | 作用 | 主要实现位置 |
|------|------|--------------|
| **张量并行（TP）** | 多张 GPU 共同承载一个模型，按层/head/词表切分 | `llm_engine.py` 多进程；`model_runner.py` 通信；`layers/linear.py`、`embed_head.py` 分片与通信 |
| **请求/批并行** | 多条请求同时进入一批 prefill 或 decode，提高吞吐 | `scheduler.py` 的 waiting/running 队列与调度/抢占逻辑 |
| **Prefill vs Decode** | 同一入口、不同数据准备与执行路径（eager vs CUDA Graph） | `model_runner.py` 的 `prepare_*`、`run_model` |

---

## 2. 张量并行（多 GPU）

### 2.1 进程与设备

- **配置**：`Config.tensor_parallel_size`（默认 1，最大 8）。
- **进程模型**：rank 0 在主进程；rank 1 ～ N-1 通过 `torch.multiprocessing` 的 **spawn** 启动子进程，避免 fork 带来的 CUDA 上下文问题。
- **设备绑定**：每个 rank 调用 `torch.cuda.set_device(rank)`，各进程占一张 GPU。
- **进程组**：`dist.init_process_group("nccl", "tcp://localhost:2333", world_size, rank)`，用于后续 all_reduce、gather 等。

参见：`llm_engine.py`（spawn 子进程）、`model_runner.py`（`__init__` 里 init_process_group、set_device）。

### 2.2 多进程协调（主进程发令、子进程执行）

- **单进程**（`world_size == 1`）：`ModelRunner.call(method_name, *args)` 直接在本进程执行对应方法。
- **多进程**（`world_size > 1`）：
  - **rank 0**：把 `(method_name, *args)` 序列化写入 **SharedMemory**，通过 **Event** 通知各子进程，然后在本进程执行同一方法。
  - **rank > 0**：在 `ModelRunner.loop()` 中阻塞在 `event.wait()`；被唤醒后从 SharedMemory 读出 `(method_name, args)`，执行 `call(method_name, *args)`。

这样，每次 `model_runner.call("run", seqs, is_prefill)` 时，所有 rank 都会执行同一批 `seqs`、同一阶段（prefill 或 decode），实现多卡步调一致。

参见：`model_runner.py` 的 `call`、`write_shm`、`read_shm`、`loop`。

### 2.3 模型与权重的分片方式

- **ColumnParallelLinear**（如 QKV 投影）：输出维按 rank 切分，每个 rank 只存、只算自己的输出分片；加载权重时只加载对应分片。
- **RowParallelLinear**（如 o_proj）：输入维按 rank 切分，各 rank 算完后对输出做 **all_reduce**，得到完整向量。
- **QKVParallelLinear**：Q/K/V 的 head 维按 rank 切分，每个 rank 只拥有部分 heads 的权重与计算。
- **VocabParallelEmbedding**：词表按 rank 切分；forward 时用 mask 只取本 rank 词表对应的 embedding，再 **all_reduce** 得到完整 embedding。
- **ParallelLMHead**：每个 rank 只算本 rank 词表对应的 logits；forward 末尾在 rank 0 上 **gather** 所有 rank 的 logits 并拼成完整词表维，**仅 rank 0 做采样**并返回 token，其他 rank 返回 None（由调度层保证只使用 rank 0 的返回值）。

KV cache：每层 Attention 的 `k_cache`/`v_cache` 在 `allocate_kv_cache` 时按 **num_kv_heads // world_size** 分配，即每个 rank 只存自己那部分 KV heads，attention 计算无需在 KV 上做跨 rank 通信。

参见：`layers/linear.py`（Column/Row/QKVParallelLinear）、`layers/embed_head.py`（VocabParallelEmbedding、ParallelLMHead）、`model_runner.py` 的 `allocate_kv_cache`。

---

## 3. 请求/批并行（调度与抢占）

### 3.1 队列与状态

- **waiting**：已通过 `add_request` 加入、尚未分配 KV cache blocks 的序列（Sequence）。
- **running**：已分配 blocks，正在 prefill 或 decode 的序列。

每条序列有状态：`WAITING` → `RUNNING` → `FINISHED`（或被打回 `WAITING`，见抢占）。

参见：`scheduler.py`（`waiting`/`running`、`SequenceStatus`）。

### 3.2 Prefill 调度

- 从 **waiting** 头部按 FIFO 取序列。
- 约束：`num_seqs < max_num_seqs`，且 `num_batched_tokens + len(seq) <= max_num_batched_tokens`（或类似上界），且 `block_manager.can_allocate(seq)`。
- 满足则：`block_manager.allocate(seq)`，序列状态改为 RUNNING，从 waiting 移到 running，加入本轮的 `scheduled_seqs`；累加未缓存 token 数。
- 一旦不满足则停止本阶段；若 `scheduled_seqs` 非空，返回 `(scheduled_seqs, True)` 表示本步做 prefill。

### 3.3 Decode 调度

- 若本步没有做 prefill，则从 **running** 取序列做 decode。
- 从 running 头部按 FIFO 取序列；约束：`num_seqs < max_num_seqs`，且 `block_manager.can_append(seq)`（当前序列能再追加一个 token 所需 block）。
- 若某序列无法 append（blocks 不足），则触发 **抢占（preempt）**：从 running 尾部开始释放其他序列的 blocks（`block_manager.deallocate`），被抢占序列状态改回 WAITING 并重新加入 waiting 头部，直到当前序列能 append 或 running 被掏空。
- 本步被选中的序列会调 `block_manager.may_append(seq)`（必要时预占下一个 block），然后返回 `(scheduled_seqs, False)` 表示本步做 decode；running 顺序通过 `extendleft(reversed(scheduled_seqs))` 保持 FIFO。

### 3.4 后处理与结束

- `postprocess(seqs, token_ids)`：对每条序列追加对应生成的 token，若命中 EOS 或达到 `max_tokens`，则 `deallocate` 并从 running 移除，状态设为 FINISHED。

这样，**多条请求在同一批中并行 prefill 或 decode**；资源紧张时通过抢占释放 blocks，保证至少部分请求能继续推进。

参见：`scheduler.py` 的 `schedule`、`preempt`、`postprocess`。

---

## 4. Prefill 与 Decode 执行路径

### 4.1 统一入口

- `LLMEngine.step()` 调用 `scheduler.schedule()` 得到 `(seqs, is_prefill)`，再调用 `model_runner.call("run", seqs, is_prefill)`。
- `ModelRunner.run(seqs, is_prefill)` 根据 `is_prefill` 选择不同的 prepare 与执行方式。

### 4.2 Prefill

- **输入准备**：`prepare_prefill(seqs)` 拼出变长的 `input_ids`、`positions`，以及 `cu_seqlens_q/cu_seqlens_k`、`max_seqlen_q/max_seqlen_k`、`slot_mapping`、可选的 `block_tables`（prefix cache 时），并 `set_context(is_prefill=True, ...)`。
- **执行**：`run_model(..., is_prefill=True)` 走 **eager**，直接 `model(input_ids, positions)` + `compute_logits`，不用 CUDA Graph（序列长度与 batch 变化大）。

### 4.3 Decode

- **输入准备**：`prepare_decode(seqs)` 只取各序列的 last token 与对应 positions，以及本步的 `slot_mapping`、`context_lens`、`block_tables`，并 `set_context(is_prefill=False, ...)`。
- **执行**：在未禁用 CUDA Graph 且 batch 较小时，`run_model(..., is_prefill=False)` 使用预捕获的 **CUDA Graph** 重放，只更新 `graph_vars` 中的 input_ids、positions、slot_mapping、context_lens、block_tables，然后 replay；否则仍走 eager。

因此，**并行**体现在多请求一起调度；**加速**体现在 decode 阶段用固定 batch 的 CUDA Graph 降低 kernel 启动开销。

参见：`model_runner.py` 的 `run`、`prepare_prefill`、`prepare_decode`、`run_model`、`capture_cudagraph`。

---

## 5. 小结

- **多卡模型并行**：多进程 + NCCL + SharedMemory 同步；Linear/Embedding/LMHead 按输出或词表分片，配合 all_reduce/gather；KV cache 按 head 分片。
- **多请求并行**：Scheduler 维护 waiting/running，prefill 按 token/seq 上限组批，decode 按 seq 上限组批；block 不足时通过 preempt 释放其他序列的 blocks。
- **Prefill vs Decode**：同一 `run()` 入口，由 `is_prefill` 选择 prepare 与 run_model 路径；decode 在条件满足时使用 CUDA Graph 复用小 batch 计算图。

相关文件：

- `nanovllm/engine/llm_engine.py`：多进程启动、step 循环。
- `nanovllm/engine/model_runner.py`：TP 初始化、通信、KV cache、prepare_*、run_model、CUDA Graph。
- `nanovllm/engine/scheduler.py`：waiting/running、schedule、preempt、postprocess。
- `nanovllm/layers/linear.py`：Column/Row/QKVParallelLinear。
- `nanovllm/layers/embed_head.py`：VocabParallelEmbedding、ParallelLMHead。
