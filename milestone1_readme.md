Milestone 1 变更说明
====================

本文档总结 `m1-tongwu` 分支在 Milestone 1 中完成的所有改动。目标是让
CPU-only 的 Simulator 也能稳定地触发 prefix caching，从而与 GPU 执行器保持
相同的指标统计行为。

平台/设备检测相关
----------------

* commit `5f49e7fd6`（skip device checks on macOS）
  * `vllm/config.py`：`DeviceConfig` 在无法自动探测设备类型时直接返回，避免
    在 macOS（尤其是无独显机型）上因为探测失败而抛错。
  * `vllm/core/scheduler.py`：在 mac 环境下强制使用一个超小的 KV cache
    （GPU/CPU block 数各 16），防止因显存不可用导致调度器初始化失败。
  * `vllm/platforms/interface.py`：`supports_async_output` 提前返回，跳过对平台
    能力的进一步校验，避免 macOS 环境下的兼容性问题。

API Server 行为调整
------------------

* commit `fab395ce3`（disable frontend multiprocessing in API server startup）
  * `vllm/entrypoints/openai/api_server.py`
    * 默认开启 `disable_frontend_multiprocessing`，避免前端多进程在 macOS 上
      引起 UVLoop/Ray 初始化竞态。
    * 暂时注释掉自定义监听 socket 的创建/关闭逻辑，交由 Uvicorn 默认行为处理，
      以规避 macOS 环境下端口被占用造成的启动失败。

调度器与缓存配置
----------------

* `vllm/core/scheduler.py`
  * 在创建 `Scheduler` 时，直接将 GPU/CPU block 数限制为 16，确保 Simulator
    与 Scheduler 对 cache 大小的认知一致，不再出现一个组件认为有 1w blocks、
    另一个组件只有几十个的情况。

Simulator
---------

* `vllm/sim/simulator.py`
  * `determine_num_available_blocks()` 直接返回 `cache_config` 中的配置，而不是
    强行改写成 8192/0，使 block manager 的记录与调度端一致。

前缀缓存相关测试
----------------

* `tests/standalone_tests/test_simulator_mode.py`
  * trace 中的 prompt 改为长文本，方便构造共享前缀。
  * 构建引擎时把 `block_size` 降到 2，让短 prompt 也能生成多个 KV block。
  * 新增 8 个带相同前缀的大 prompt，逼迫调度器复用缓存。
  * 命中率断言改为查询 `Device.GPU`，因为 prefix caching 逻辑只在 GPU allocator
    上实现；即使在 CPU-only 模式下也如此。

运行前准备
----------

1. 先准备好包含 prompt/response 的 trace 文件，并设置
   `VLLM_SIM_TRACE_PATH=/path/to/trace.jsonl`。Simulator 启动时会读取该环境变量，
   没有的话会直接抛异常。
2. 其余 vLLM 配置与常规启动一致（例如模型、tokenizer、CacheConfig 等），本分支中
   的修改会自动复用这些配置，不需要额外 flag。

测试方法
--------

运行 `python3 -m pytest tests/standalone_tests/test_simulator_mode.py`
即可覆盖上述改动。（目前镜像里尚未安装 `pytest`，会报 “No module named pytest”，
需要先安装依赖再执行。）

补充说明（面向 Milestone 2）
----------------------------

Milestone 2 如果需要调整 GPU / CPU block 数量，只需在 `vllm/core/scheduler.py`
中创建 `Scheduler` 时修改那几行 `self.cache_config.num_gpu_blocks` /
`num_cpu_blocks` 的赋值即可，其余逻辑都会自动继承新的缓存规模。
