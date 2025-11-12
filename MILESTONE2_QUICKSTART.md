# Milestone 2 快速开始指南

## 🎯 核心文件

**推荐使用**: `vllm/sim/run_milestone2_correct_approach.py`

这是**唯一正确的实现**，使用 vLLM 真实 block manager。

## 📋 快速运行

### 1. 激活环境

```bash
source .venv/bin/activate
```

### 2. 运行 Multi-turn 实验

```bash
python vllm/sim/run_milestone2_correct_approach.py \
    --trace-file milestone2_results/multi_turn_trace.jsonl \
    --model facebook/opt-125m \
    --block-size 16
```

### 3. 运行 Single-turn 对比

```bash
python vllm/sim/run_milestone2_correct_approach.py \
    --trace-file milestone2_results/single_turn_trace.jsonl \
    --model facebook/opt-125m \
    --block-size 16
```

## 📊 预期结果

### Multi-turn (最重要)

```
Overall Sharing Fraction: 64.96%
Cache Hit Rate: 64.31%
Blocks Reused: 3,300
```

**解释**: 65% 的 blocks 被重用，证明 prefix caching 极其有效！

### Single-turn (对比组)

```
Overall Sharing Fraction: 4.22%
Cache Hit Rate: 0.05%
Blocks Reused: 92
```

**解释**: 几乎没有重用，因为每个请求都是独立的。

## 🔧 参数说明

- `--trace-file`: Trace 文件路径（必需）
- `--model`: 模型名称（默认 facebook/opt-125m）
- `--block-size`: Block 大小（默认 16）
- `--output-file`: 保存 JSON 统计（可选）
- `--no-prefix-caching`: 禁用 prefix caching（对照实验）

## ✅ 正确性验证

### 使用 vLLM 真实组件

- `SelfAttnBlockSpaceManager` - 真实 block manager
- `PrefixCachingBlockAllocator` - 真实 prefix caching
- Cache hit rate 来自 vLLM 内部 API

### 独立加载技术

- 避免 GPU/平台检测问题
- Mock CPU 平台
- 使用 10000 blocks 避免 eviction 边界情况

## ⚠️ 注意事项

1. **必须在 `.venv` 环境中运行**
   ```bash
   source .venv/bin/activate
   ```

2. **使用 `python` 不是 `python3`**
   ```bash
   python vllm/sim/run_milestone2_correct_approach.py ...
   ```

3. **需要 trace 文件**

   如果没有，先生成：
   ```bash
   python vllm/sim/run_milestone2_task2.py \
       --data-path vllm/ShareGPTData.jsonl \
       --max-conversations 100 \
       --output-dir milestone2_results \
       --skip-visualization
   ```

## 📈 对比实验

### 完整对比（推荐）

```bash
# 1. Multi-turn with caching
python vllm/sim/run_milestone2_correct_approach.py \
    --trace-file milestone2_results/multi_turn_trace.jsonl \
    --model facebook/opt-125m \
    --output-file multi_cache.json

# 2. Single-turn with caching
python vllm/sim/run_milestone2_correct_approach.py \
    --trace-file milestone2_results/single_turn_trace.jsonl \
    --model facebook/opt-125m \
    --output-file single_cache.json

# 3. Multi-turn without caching (baseline)
python vllm/sim/run_milestone2_correct_approach.py \
    --trace-file milestone2_results/multi_turn_trace.jsonl \
    --model facebook/opt-125m \
    --no-prefix-caching \
    --output-file multi_no_cache.json
```

## 📝 其他文件

### 文档

- `MILESTONE2_CORRECT_IMPLEMENTATION.md` - 详细实现说明
- `MILESTONE2_RESULTS.md` - 完整实验结果和分析
- `MILESTONE2_QUICKSTART.md` - 本文档

### 代码

- `vllm/sim/run_milestone2_correct_approach.py` - ✅ **正确实现**
- `vllm/sim/client_simulator.py` - Task 1（生成 traces）
- `vllm/sim/run_milestone2_task2.py` - 旧版本（已过时）

### 数据

- `milestone2_results/multi_turn_trace.jsonl` - Multi-turn traces
- `milestone2_results/single_turn_trace.jsonl` - Single-turn traces
- `milestone2_multi_stats.json` - Multi-turn 统计
- `milestone2_single_stats.json` - Single-turn 统计

## 🎉 总结

**一行命令运行实验**:

```bash
source .venv/bin/activate && \
python vllm/sim/run_milestone2_correct_approach.py \
    --trace-file milestone2_results/multi_turn_trace.jsonl \
    --model facebook/opt-125m \
    --block-size 16
```

**预期结果**: 65% sharing fraction，证明 prefix caching 非常有效！
