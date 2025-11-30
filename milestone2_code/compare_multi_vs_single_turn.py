#!/usr/bin/env python3
"""
比较Multi-turn vs Single-turn的prefix cache hit rate
目标：证明multi-turn hit rate明显大于single-turn hit rate
"""
import sys
import os
import json
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from vllm import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from transformers import AutoTokenizer

model_path = str(Path(__file__).parent.parent / "exported_models" / "Llama-3.2-1B-Instruct")
multi_turn_trace = str(Path(__file__).parent / "traces" / "sharegpt_multi_turn.jsonl")
single_turn_trace = str(Path(__file__).parent / "traces" / "sharegpt_single_turn.jsonl")

tokenizer = AutoTokenizer.from_pretrained(model_path)

print("=" * 80)
print("Multi-Turn vs Single-Turn Prefix Cache Hit Rate Comparison")
print("=" * 80)

# 过滤multi-turn trace，只保留短prompts的完整conversations
print("\n【Step 1】Filtering multi-turn conversations...")
MAX_TOKENS = 2048  # 增大token限制以处理更多数据

conversations = defaultdict(list)
with open(multi_turn_trace, 'r') as f:
    for line in f:
        entry = json.loads(line.strip())
        conv_id = entry.get('conversation_id', 'unknown')
        prompt_tokens = tokenizer.encode(entry['prompt'], add_special_tokens=False)
        entry['token_count'] = len(prompt_tokens)
        conversations[conv_id].append(entry)

# 选择符合条件的conversations
filtered_multi_convs = {}
for conv_id, turns in conversations.items():
    if len(turns) >= 2 and all(turn['token_count'] <= MAX_TOKENS for turn in turns):
        filtered_multi_convs[conv_id] = turns

print(f"Total conversations: {len(conversations)}")
print(f"Filtered conversations (>= 2 turns, all <=800 tokens): {len(filtered_multi_convs)}")

if not filtered_multi_convs:
    print("❌ No suitable multi-turn conversations found!")
    sys.exit(1)

# NUM_CONVS_TO_TEST = min(50, len(filtered_multi_convs))
# 选择所有数据进行测试
NUM_CONVS_TO_TEST = len(filtered_multi_convs)
selected_convs = dict(list(filtered_multi_convs.items())[:NUM_CONVS_TO_TEST])

# 统计信息
total_multi_requests = sum(len(turns) for turns in selected_convs.values())
print(f"\nSelected {NUM_CONVS_TO_TEST} conversations for testing")
print(f"Total multi-turn requests: {total_multi_requests}")

# 创建filtered multi-turn trace
import tempfile
fd_multi, filtered_multi_trace = tempfile.mkstemp(suffix='_multi.jsonl')
with open(filtered_multi_trace, 'w') as f:
    for conv_id in sorted(selected_convs.keys()):
        for turn in selected_convs[conv_id]:
            f.write(json.dumps(turn) + '\n')
os.close(fd_multi)

print(f"✓ Created filtered multi-turn trace: {filtered_multi_trace}")

# 从single-turn trace中选择相同数量的requests
print("\n【Step 2】Selecting single-turn requests...")
single_requests = []
with open(single_turn_trace, 'r') as f:
    for line in f:
        entry = json.loads(line.strip())
        prompt_tokens = tokenizer.encode(entry['prompt'], add_special_tokens=False)
        if len(prompt_tokens) <= MAX_TOKENS:
            single_requests.append(entry)
        if len(single_requests) >= total_multi_requests:
            break

fd_single, filtered_single_trace = tempfile.mkstemp(suffix='_single.jsonl')
with open(filtered_single_trace, 'w') as f:
    for entry in single_requests:
        f.write(json.dumps(entry) + '\n')
os.close(fd_single)

print(f"Selected {len(single_requests)} single-turn requests")
print(f"✓ Created filtered single-turn trace: {filtered_single_trace}")

# 函数：运行实验并收集统计
def run_experiment(trace_path, experiment_name, use_conversation_mode=False):
    print(f"\n{'=' * 80}")
    print(f"【{experiment_name}】")
    print(f"{'=' * 80}")

    os.environ["VLLM_SIM_TRACE_PATH"] = trace_path

    # Reset trackers
    from correct_hit_rate_tracker import global_hit_rate_tracker
    from cache_block_tracker import global_cache_block_tracker
    global_hit_rate_tracker.reset()
    global_cache_block_tracker.reset()

    # 创建engine (增大max_model_len以处理更多数据)
    args = EngineArgs(
        model=model_path,
        tokenizer=model_path,
        device="cpu",
        max_model_len=2048,  # 增大到2048
        max_num_seqs=1,
        block_size=128,  # 增大到128
        enable_prefix_caching=True,
    )
    engine = LLMEngine.from_engine_args(args)

    # 创建simulator
    from milestone2_code.client_simulator import ClientSimulator
    simulator = ClientSimulator(
        trace_path=trace_path,
        tokenizer=tokenizer,
        arrival_rate=1.0,
    )

    # 运行实验
    if use_conversation_mode:
        print("Using conversation-by-conversation processing...")
        simulator.send_requests_conversation_by_conversation(engine, max_steps_per_turn=2000)
    else:
        print("Using standard all-at-once processing...")
        simulator.send_requests_to_engine(engine)
        simulator.run_engine_until_complete(engine, max_steps=10000)

    # 收集统计
    stats = global_hit_rate_tracker.get_stats()
    cache_stats = global_cache_block_tracker.get_stats()

    # 获取 vLLM GPU/CPU hit rate
    from vllm.utils import Device
    gpu_hit_rate = engine.scheduler[0].get_prefix_cache_hit_rate(Device.GPU)
    cpu_hit_rate = engine.scheduler[0].get_prefix_cache_hit_rate(Device.CPU)

    print(f"\n【Results】")
    print(f"  Total requests: {stats['total_requests']}")
    print(f"  Total blocks: {stats['total_blocks']}")
    print(f"  Hit blocks: {stats['hit_blocks']}")
    print(f"  Correct hit rate (first prefill only): {stats['overall_hit_rate']:.2%}")
    print(f"  vLLM GPU hit rate: {gpu_hit_rate:.2%}" if gpu_hit_rate >= 0 else "  vLLM GPU hit rate: N/A (CPU mode)")
    print(f"  vLLM CPU hit rate: {cpu_hit_rate:.2%}" if cpu_hit_rate >= 0 else "  vLLM CPU hit rate: N/A")

    print(f"\n【Task 2 Additional Metrics】")
    print(f"  Cache blocks used: {cache_stats['total_cached_blocks']}")
    print(f"  Total cache block accesses: {cache_stats['total_block_accesses']}")
    print(f"  Avg hits per block: {cache_stats['avg_hits_per_block']:.2f}")
    print(f"  Max hits per block: {cache_stats['max_hits_per_block']}")
    if cache_stats['total_reuses'] > 0:
        print(f"  Block reuses with time gap: {cache_stats['total_reuses']}")
        print(f"  Avg reuse gap: {cache_stats['avg_reuse_gap_seconds']:.4f}s")
        print(f"  Min reuse gap: {cache_stats['min_reuse_gap_seconds']:.4f}s")
        print(f"  Max reuse gap: {cache_stats['max_reuse_gap_seconds']:.4f}s")
    else:
        print(f"  Block reuses with time gap: 0 (no repeated accesses)")

    stats['gpu_hit_rate'] = gpu_hit_rate
    stats['cpu_hit_rate'] = cpu_hit_rate
    stats['cache_stats'] = cache_stats

    return stats

# 运行single-turn实验
single_stats = run_experiment(filtered_single_trace, "Single-Turn Experiment", use_conversation_mode=False)

# 运行multi-turn实验（conversation-by-conversation mode）
multi_stats = run_experiment(filtered_multi_trace, "Multi-Turn Experiment (Conversation-by-Conversation)", use_conversation_mode=True)

# 比较结果
print("\n" + "=" * 80)
print("【Comparison】")
print("=" * 80)

# Correct hit rate (first prefill only)
single_hit_rate = single_stats['overall_hit_rate']
multi_hit_rate = multi_stats['overall_hit_rate']

print(f"\n【Correct Hit Rate (First Prefill Only)】")
print(f"  Single-turn: {single_hit_rate:.2%}")
print(f"  Multi-turn:  {multi_hit_rate:.2%}")

if multi_hit_rate > single_hit_rate:
    improvement = multi_hit_rate - single_hit_rate
    relative_improvement = (multi_hit_rate / single_hit_rate - 1) * 100 if single_hit_rate > 0 else float('inf')
    print(f"  ✅ Multi-turn is HIGHER! (+{improvement:.2%})")
else:
    print(f"  ❌ Multi-turn is not higher ({multi_hit_rate - single_hit_rate:+.2%})")

# vLLM GPU hit rate
single_gpu = single_stats.get('gpu_hit_rate', -1)
multi_gpu = multi_stats.get('gpu_hit_rate', -1)

print(f"\n【vLLM GPU Hit Rate】")
if single_gpu >= 0 and multi_gpu >= 0:
    print(f"  Single-turn: {single_gpu:.2%}")
    print(f"  Multi-turn:  {multi_gpu:.2%}")
    if multi_gpu > single_gpu:
        print(f"  ✅ Multi-turn is HIGHER! (+{multi_gpu - single_gpu:.2%})")
    else:
        print(f"  ❌ Multi-turn is not higher ({multi_gpu - single_gpu:+.2%})")
else:
    print(f"  N/A (running in CPU mode)")

# vLLM CPU hit rate
single_cpu = single_stats.get('cpu_hit_rate', -1)
multi_cpu = multi_stats.get('cpu_hit_rate', -1)

print(f"\n【vLLM CPU Hit Rate】")
if single_cpu >= 0 and multi_cpu >= 0:
    print(f"  Single-turn: {single_cpu:.2%}")
    print(f"  Multi-turn:  {multi_cpu:.2%}")
    if multi_cpu > single_cpu:
        print(f"  ✅ Multi-turn is HIGHER! (+{multi_cpu - single_cpu:.2%})")
    else:
        print(f"  ❌ Multi-turn is not higher ({multi_cpu - single_cpu:+.2%})")
else:
    print(f"  N/A")

# Task 2: Cache Block Usage Comparison
single_cache = single_stats.get('cache_stats', {})
multi_cache = multi_stats.get('cache_stats', {})

print(f"\n【Task 2: Cache Block Usage Statistics】")
print(f"  Metric                        | Single-turn | Multi-turn")
print(f"  " + "-" * 60)
print(f"  Unique cache blocks           | {single_cache.get('total_cached_blocks', 0):11d} | {multi_cache.get('total_cached_blocks', 0):10d}")
print(f"  Total block accesses          | {single_cache.get('total_block_accesses', 0):11d} | {multi_cache.get('total_block_accesses', 0):10d}")
print(f"  Avg accesses per block        | {single_cache.get('avg_hits_per_block', 0):11.2f} | {multi_cache.get('avg_hits_per_block', 0):10.2f}")
print(f"  Max accesses per block        | {single_cache.get('max_hits_per_block', 0):11d} | {multi_cache.get('max_hits_per_block', 0):10d}")
print(f"  Repeated accesses (>=2 times) | {single_cache.get('total_reuses', 0):11d} | {multi_cache.get('total_reuses', 0):10d}")

print(f"\n【Task 2: Cache Block Reuse Time Gaps】")
print(f"  Metric                        | Single-turn | Multi-turn")
print(f"  " + "-" * 60)
print(f"  Avg reuse gap (seconds)       | {single_cache.get('avg_reuse_gap_seconds', 0):11.4f} | {multi_cache.get('avg_reuse_gap_seconds', 0):10.4f}")
print(f"  Min reuse gap (seconds)       | {single_cache.get('min_reuse_gap_seconds', 0):11.4f} | {multi_cache.get('min_reuse_gap_seconds', 0):10.4f}")
print(f"  Max reuse gap (seconds)       | {single_cache.get('max_reuse_gap_seconds', 0):11.4f} | {multi_cache.get('max_reuse_gap_seconds', 0):10.4f}")

# 最终结论
print(f"\n{'=' * 80}")
if multi_hit_rate > single_hit_rate:
    print(f"✅ SUCCESS: Multi-turn hit rate is HIGHER than single-turn!")
    print(f"\n  This proves that conversation-by-conversation processing enables")
    print(f"  subsequent turns to reuse previous turns' cached blocks!")
else:
    print(f"❌ FAIL: Multi-turn hit rate is not higher than single-turn")

# 清理
os.unlink(filtered_multi_trace)
os.unlink(filtered_single_trace)

print("\n" + "=" * 80)
print("✓ Done")
print("=" * 80)
