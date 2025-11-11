# Milestone 2: Prefix Sharing for ShareGPT Workload

本里程碑实现了对 ShareGPT 数据集的回放，并测量 prefix sharing 的有效性。

## 文件说明

- `client_simulator.py`: 客户端模拟器，负责加载和处理 ShareGPT 数据
- `prefix_sharing_metrics.py`: Prefix sharing 指标收集器
- `run_milestone2_task2.py`: 主运行脚本

## Task 1: Client Simulator

客户端模拟器实现了以下功能：

### 1. ShareGPT 数据加载
```python
from client_simulator import ShareGPTLoader

loader = ShareGPTLoader("ShareGPTData.jsonl", max_conversations=1000)
conversations = loader.get_conversations()
```

### 2. Chat Template 格式化
支持使用模型的 chat template 格式化对话：
```python
from client_simulator import ChatTemplateFormatter

formatter = ChatTemplateFormatter(tokenizer)  # 可传入 HF tokenizer
formatted_prompt = formatter.format_conversation(conversation_turns)
```

### 3. 请求生成（支持 Poisson 分布）
```python
from client_simulator import RequestGenerator

generator = RequestGenerator(
    conversations,
    arrival_rate=2.0,      # 每秒 2 个请求
    use_poisson=True,      # 使用 Poisson 分布
    seed=42
)

# 单轮对话
single_turn_traces = generator.generate_single_turn_traces(formatter)

# 多轮对话
multi_turn_traces = generator.generate_multi_turn_traces(formatter, turn_delay=1.0)
```

## Task 2: 运行实验和收集指标

### 快速运行

```bash
cd /Users/thea/Documents/GitHub/vllm

# 运行完整实验（使用 1000 个对话）
python vllm/sim/run_milestone2_task2.py \
    --data-path vllm/ShareGPTData.jsonl \
    --max-conversations 1000 \
    --output-dir milestone2_results \
    --block-size 16 \
    --arrival-rate 2.0
```

### 参数说明

- `--data-path`: ShareGPT 数据文件路径
- `--max-conversations`: 处理的最大对话数（用于快速测试）
- `--output-dir`: 结果输出目录
- `--block-size`: KV cache 的 block 大小（tokens per block）
- `--arrival-rate`: 请求到达率（requests/second）
- `--skip-visualization`: 跳过可视化（如果没有 matplotlib）

### 输出文件

运行后会在 `milestone2_results/` 目录下生成：

#### 1. 统计数据（JSON）
- `single_turn_stats.json`: 单轮对话的统计数据
- `multi_turn_stats.json`: 多轮对话的统计数据

包含的指标：
- 总请求数
- 总分配的 blocks
- 总重用的 blocks
- Sharing fraction（每个请求从 prefix sharing 受益的比例）
- Block hit counts（每个 block 被访问的次数）
- Reuse gaps（block 被重用之间的时间间隔）

#### 2. 详细数据（CSV）
- `single_turn_request_metrics.csv`: 每个请求的详细指标
- `single_turn_block_metrics.csv`: 每个 block 的详细指标
- `multi_turn_request_metrics.csv`: 多轮对话的请求指标
- `multi_turn_block_metrics.csv`: 多轮对话的 block 指标

#### 3. 可视化图表（PNG）
- `sharing_fraction_cdf.png`: Sharing fraction 的 CDF
- `block_hit_distribution.png`: Block hit count 分布
- `reuse_gap_cdf.png`: Reuse gap 的 CDF
- `comparison.png`: 单轮 vs 多轮对比

#### 4. Trace 文件（用于 vLLM simulator）
- `single_turn_trace.jsonl`: 单轮对话的 trace
- `multi_turn_trace.jsonl`: 多轮对话的 trace

这些文件可以直接用于 Milestone 1 的 simulator。

## 收集的指标说明

### 1. Sharing Fraction
每个请求中有多少比例的 tokens 受益于 prefix sharing（从 cache 中重用）。

- **高 sharing fraction**: 说明大部分 prompt 都能重用已有的 KV cache
- **低 sharing fraction**: 说明大部分 prompt 是新的，需要重新计算

### 2. Block Hit Count
每个 cache block 被访问的次数。

- **高 hit count**: 说明这个 block 被频繁重用，prefix sharing 很有效
- **Hit count = 1**: 说明这个 block 只被使用了一次，没有被重用

### 3. Reuse Gap
两次访问同一个 block 之间的时间间隔。

- **短 reuse gap**: 说明 block 很快被再次访问，cache 命中率高
- **长 reuse gap**: 说明 block 重用间隔长，可能需要不同的 eviction policy

## 预期结果分析

### Single-turn vs Multi-turn

**Single-turn（单轮对话）：**
- 每个对话只使用第一个 user prompt 和对应的回复
- Prefix sharing 效果可能较弱，因为不同对话的开头通常不同
- 主要的 sharing 来自于：相似的问题开头、系统 prompt（如果有）

**Multi-turn（多轮对话）：**
- 包含完整的对话历史
- Prefix sharing 效果应该更强，因为：
  - 同一对话的后续轮次会重用之前的所有历史
  - 对话历史随着轮次增长而增长
  - 每一轮都完全重用之前的 KV cache

**预期观察：**
1. Multi-turn 的 sharing fraction 应该 **明显高于** single-turn
2. Multi-turn 的 block hit count 应该更高
3. Multi-turn 的 reuse gap 应该更短（因为同一对话的连续请求）

## 安装依赖

```bash
# 基础依赖
pip install numpy

# 可视化依赖（可选）
pip install matplotlib
```

## 示例输出

```
============================================================
PREFIX SHARING METRICS SUMMARY
============================================================

Total Requests: 1000
Total Blocks Allocated: 5243
Total Blocks Reused: 3891
Unique Blocks: 5243
Block Reuse Rate: 42.61%

Sharing Fraction (per request):
  Mean: 38.50%
  Median: 42.13%
  Min: 0.00%
  Max: 95.23%

Block Hit Counts:
  Mean: 2.74
  Median: 2
  Min: 1
  Max: 156

Reuse Gaps (seconds):
  Mean: 12.45s
  Median: 8.32s
  Min: 0.15s
  Max: 485.23s
============================================================
```

## 扩展实验

你可以通过调整参数来探索不同的场景：

### 1. 调整 Block Size
```bash
# 小 block size (8 tokens)
python vllm/sim/run_milestone2_task2.py --block-size 8

# 大 block size (32 tokens)
python vllm/sim/run_milestone2_task2.py --block-size 32
```

**预期：**
- 小 block size: 更细粒度的 sharing，hit count 可能更高
- 大 block size: 更粗粒度的 sharing，hit count 可能更低

### 2. 调整到达率
```bash
# 低到达率（更多时间间隔）
python vllm/sim/run_milestone2_task2.py --arrival-rate 0.5

# 高到达率（请求密集）
python vllm/sim/run_milestone2_task2.py --arrival-rate 10.0
```

**预期：**
- 高到达率: reuse gap 更短，cache 更有效
- 低到达率: reuse gap 更长，可能需要 eviction

### 3. 使用不同数据量
```bash
# 小规模测试
python vllm/sim/run_milestone2_task2.py --max-conversations 100

# 大规模实验
python vllm/sim/run_milestone2_task2.py --max-conversations 10000
```

## 与 vLLM 集成（可选）

生成的 trace 文件可以直接用于 vLLM simulator：

```bash
# 使用生成的 trace 运行 vLLM
python -m vllm.entrypoints.openai.api_server \
    --model facebook/opt-125m \
    --use-simulator \
    --sim-trace-path milestone2_results/multi_turn_trace.jsonl \
    --enable-prefix-caching \
    --port 8000
```
