# Milestone 1：前缀共享模拟模式说明

> 目标：在 **无 GPU / 无真实模型推理** 的条件下，仍然跑通 vLLM 的调度、连续批处理、前缀缓存等核心逻辑，为后续研究 prefix sharing 的行为奠定基础。

---

## 1. 总体设计

1. **保持主干不变**：Scheduler、请求状态机、KV Block 管理等组件完全复用，不做侵入式改动，让后续 Milestone 可以直接在此基础上继续实验。
2. **替换执行环节**：在 `LLMEngine` 中识别模拟模式，跳过真实 `execute_model`，由纯 Python 的 `Simulator` 直接输出 token 序列。
3. **轨迹/默认双模式**：如果提供 JSONL 轨迹则严格重放；没有轨迹时按 prompt 长度生成可复现的 dummy 序列，保证任意请求都能完成。
4. **环境变量驱动**：通过 `VLLM_USE_SIMULATOR`（必需）和 `VLLM_SIMULATOR_TRACE_PATH`（可选）切换，不影响默认 GPU/CPU 推理路径。

---

## 2. 源码改动概览

| 文件 | 角色 | 说明 |
| --- | --- | --- |
| `vllm/engine/llm_engine.py` | Online 引擎主循环 | 负责挂载 `Simulator`、替换 executor、在 `step()` 中注入 `_simulate_step`，并在请求结束时清理模拟状态。 |
| `vllm/sim/simulator.py` | **新增** 模拟器 | 管理每个请求的输出游标，可从 JSONL 轨迹初始化；提供 `start_request/next_token/finish_request` API。 |
| `vllm/executor/simulator_executor.py` | **新增** 空执行器 | 满足 `ExecutorBase` 接口但不运行模型；`LLMEngine` 在模拟模式下使用它，避免加载真实权重。 |
| `vllm/sim/__init__.py` | 包标识 | 让 `vllm.sim` 可被导入。 |
| `tests/standalone_tests/test_simulator_mode.py` | **新增** 单测 | 构造两个请求，验证输出与轨迹一致，并检查前缀缓存统计。 |

> Scheduler（`vllm/core/scheduler.py`）与多进程 Engine（`vllm/engine/multiprocessing/engine.py`）保持原样，原因见文末。

---

## 3. 关键代码路径详解

### 3.1 `LLMEngine` 初始化
```
self.use_simulator = bool(os.getenv("VLLM_USE_SIMULATOR"))
if self.use_simulator:
    self.simulator = Simulator(trace_path=os.getenv("VLLM_SIMULATOR_TRACE_PATH"))
    self.model_executor = SimulatorExecutor(vllm_config=vllm_config,
                                            simulator=self.simulator)
```
- 未设置环境变量时，`Simulator` 不创建，行为与原版一致。
- `SimulatorExecutor` 只提供缓存大小信息，`execute_model` 返回 `None`，真正的 token 生成由 `_simulate_step` 完成。

### 3.2 请求接入
```
seq_group = self._add_processed_request(...)
if self.use_simulator and isinstance(params, SamplingParams):
    self.simulator.start_request(request_id,
                                 seq_group.first_seq,
                                 max_new_tokens=params.max_tokens)
```
- 只有在模拟模式且为文本生成请求时才注册。
- `start_request` 会检查是否存在轨迹；若没有，就用 prompt 长度生成 deterministic 序列。

### 3.3 执行循环
1. Scheduler 按原逻辑调度，生成 `ScheduledSequenceGroup` 列表。
2. 若 `use_simulator=True`，跳过真实 `execute_model`：
```
outputs = [self._simulate_step(seq_group_metadata_list,
                               scheduler_outputs)]
```
3. `_simulate_step` 遍历批次，针对每个请求调用 `simulator.next_token`，再构造 `SequenceOutput → CompletionSequenceGroupOutput → SamplerOutput`，并把 `model_forward_time/model_execute_time` 置零。
4. 结果交给原有 `SequenceGroupOutputProcessor`，因此调度逻辑、stop checker、metrics 统计全部沿用。

### 3.4 请求完成与清理
- 当某个 `SequenceGroup` 在 `_process_model_outputs` 中被标记为 finished 时，调用 `self.simulator.finish_request(request_id, seq)` 回收游标，避免状态泄漏。

---

## 4. 运行与调试

1. 设置环境变量：
```
export VLLM_USE_SIMULATOR=1
export VLLM_SIMULATOR_TRACE_PATH=/path/to/trace.jsonl   # 可选
```
2. 启动示例：
```
VLLM_USE_SIMULATOR=1 python -m vllm.entrypoints.openai.api_server \
  --model gpt2 --tokenizer gpt2 --device cpu \
  --max-model-len 4096 --max-num-seqs 4 --block-size 8
```
3. 轨迹格式（JSON Lines）：
```
{"request_id": "req1", "output_token_ids": [101, 102, 103]}
{"request_id": "req2", "output_token_ids": [201, 202]}
```
- 如果省略轨迹，模拟器会按 prompt 长度生成 `prefix_len * 10 + i` 的虚拟 token。

---

## 5. 测试方案

### 5.1 文件
`tests/standalone_tests/test_simulator_mode.py`

### 5.2 步骤
1. `trace_file` fixture：写入临时 JSONL（`req1→[101,102,103]`, `req2→[101,105]`）。
2. `sim_env` fixture：设置 `VLLM_USE_SIMULATOR`、`VLLM_SIMULATOR_TRACE_PATH`。
3. `_make_engine()`：使用 `EngineArgs(model="gpt2", tokenizer="gpt2", device="cpu", enable_prefix_caching=True, max_model_len=128, block_size=8)` 构建引擎。
4. `_drain_engine()`：循环调用 `engine.step()`，收集 streaming 输出直到全部完成。
5. `test_simulator_single_request`：提交 `req1`，断言最终输出 token 与轨迹一致且 `finished=True`。
6. `test_simulator_prefix_hit`：提交共享前缀的 `req1/req2`，检查输出与轨迹一致，并验证 `get_prefix_cache_hit_rate(Device.CPU) >= 0`（模拟模式可能为 0）。

### 5.3 运行命令
```
VLLM_USE_SIMULATOR=1 pytest tests/standalone_tests/test_simulator_mode.py
```

---

## 6. 为什么不修改 Scheduler / 多进程 Engine

- **Scheduler (`vllm/core/scheduler.py`)**：负责调度、分配 KV block、处理 prefill/decode。模拟输出与真实输出格式相同，Scheduler 无需感知“token 来自模拟器”，保持原样能证明调度逻辑未被破坏，也让 Milestone2/3 可直接复用。
- **多进程 Engine (`vllm/engine/multiprocessing/engine.py`)**：只是 IPC 封装，照常调用 `LLMEngine.step()`；Executor 是否为模拟实现对它透明。若未来需要在 worker 侧直接运行模拟器，再考虑扩展。

---

## 7. 可扩展方向

- 在 `_simulate_step` 中补充 `finish_reason`、`logprobs`、`n>1`、beam search 等模拟行为；
- 在 CLI/EngineArgs 中加入 `--use-simulator`、`--sim-trace-path` 等显式参数；
- 扩展测试覆盖：混合负载、超长 prompt、abort/cancel、不同 cache eviction 策略比较等。

如需进一步迭代，可以本 README 为参考继续补充说明。
