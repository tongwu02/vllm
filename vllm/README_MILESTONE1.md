# Milestone 1

本里程碑目标：在 **不跑真实模型** 的情况下，让引擎按照调度节奏（prefill / decode）**一步一 token** 地输出结果，用于作业与性能/节奏验证。

------

## 1) 环境准备（WSL Ubuntu 22.04）

> 目标：创建 `vllm-env` 虚拟环境，并基于 **CPU** 依赖安装（不需要 GPU/CUDA）。

```bash
# 1) 创建并启用 venv
python3 -m venv vllm-env
source vllm-env/bin/activate

# 2) 进入项目根目录
cd path_to_project/vllm

# 3) 升级基础工具
python -m pip install --upgrade pip setuptools wheel

# 4) 安装 CPU 依赖
pip install -r requirement-cpu.txt
```

> 按照project.pdf中的命令构建
>
> 但是uv似乎对于版本管理十分严格。如果构建失败，尝试删去命令中的uv
>
> ```bash
> VLLM_USE_PRECOMPILED=1 uv pip install --editable .
> ```

------

## 2) 本里程碑的代码改动

### 2.1 新增 / 修改的文件

- `vllm/engine/arg_utils.py`
   新增 **AsyncEngineArgs** 两个参数（及 CLI）：

  - `use_simulator: bool = False`  → `--use-simulator`
  - `sim_trace_path: Optional[str] = None`  → `--sim-trace /path/to/trace.jsonl`

- `vllm/engine/multiprocessing/engine.py`
   在 `MQLLMEngine.from_engine_args(...)` 读取上述参数，并在 `MQLLMEngine.__init__` **透传**给 `LLMEngine`（通过 `kwargs` 注入）。

- `vllm/engine/llm_engine.py`

  - `__init__`：接收 `use_simulator` / `sim_trace_path`，按需创建 `self.simulator`。

  - `add_request(...)`：在加入请求后调用 `self.simulator.on_add_request(...)` 建立游标/状态。

  - `step()`：**基于本轮 `scheduler_outputs` 拆分**

    ```text
    prompt_run = scheduled[:num_prefill]
    decode_run = scheduled[num_prefill:]
    ```

    若 `use_simulator=True`：

    - `simulate_prefill(prompt_run)` 只做记账
    - `simulate_decode(decode_run)` 每个请求吐 1 token
    - 触发与真实路径一致的回调（若存在），将结果写回 `ctx.request_outputs`，并 `return`

    > 注意：不要在模拟器分支里**再次调用** `self.scheduler.schedule(...)`，只用**本轮**调度结果。

- `vllm/sim/simulator.py`（新增）

  - 读取 JSONL trace：`{"prompt": "...", "response": "..."}`
  - 记录每个请求的游标、已生成 token 数、结束状态
  - `simulate_prefill(...)` 只标记可进入 decode
  - `simulate_decode(...)` 返回 **List[RequestOutput]**，并根据 `max_tokens` / `stop` / trace 末尾设置 `finish_reason` 为 `length` 或 `stop`

> 代码风格：`RequestOutput/SequenceOutput` 的构造参数做了签名过滤，兼容不同 vLLM 版本。

------

## 3) 测试方法

### 3.1 单元级（不加载整个 vLLM 包，零 GPU 依赖）✅

> 我们提供了一个**独立测试脚本**不会触发 CUDA 依赖（否则因为vllm/\_\_init\_\_.py的关系，会触发CUDA调用从而报错）：
>  `vllm/vllm/sim/test_simulator_unit_standalone.py`

1. 准备 trace（脚本会自动生成，无需手动创建）

   ```json
   {"prompt":"Hello","response":" world!"}
   {"prompt":"Hi","response":" there"}
   {"prompt":"StopCase","response":" hello STOP and more"}
   
   ```

2. 运行脚本

```bash
python vllm/vllm/sim/test_simulator_unit_standalone.py
```

**预期输出（示意）**

- 请求 `r1`（Hello）会一路生成到 `" world!"`，`finish_reason="stop"`
- 请求 `r2`（Hi）受 `max_tokens=3` 限制，`finish_reason="length"`
- 请求 `r3`（StopCase）命中 `stop=["STOP"]` 后以 `stop` 结束（测试脚本已将 `max_tokens` 调大）

> 若你把脚本移动了目录：当前脚本已按“同目录定位 `simulator.py` 与 `trace.jsonl`”，不依赖工作目录，直接可跑。