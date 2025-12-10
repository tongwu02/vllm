#!/usr/bin/env python3
"""
Multi-turn vs Single-turn Prefix Cache Hit Rate Experiment
+ Advanced Metrics for Micro-analysis (Shared Fraction, Block Hits, Reuse Gap)
"""
import sys
import os
import json
import tempfile
import time
import numpy as np
from pathlib import Path
from collections import defaultdict, deque
from typing import Dict, Any, List

# =============================================================================
# 1. Path Setup and Imports
# =============================================================================
sys.path.insert(0, str(Path(__file__).parent.parent))

from vllm import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from transformers import AutoTokenizer

try:
    from correct_hit_rate_tracker import global_hit_rate_tracker
    from cache_block_tracker import global_cache_block_tracker
    from milestone2_code.client_simulator import ClientSimulator
except ImportError as e:
    print(f"❌ Import Error: {e}")
    sys.exit(1)

# Model and Trace Paths
model_path = str(Path(__file__).parent.parent / "exported_models" / "Llama-3.2-1B-Instruct")
multi_turn_trace = str(Path(__file__).parent / "traces" / "sharegpt_multi_turn.jsonl")
single_turn_trace = str(Path(__file__).parent / "traces" / "sharegpt_single_turn.jsonl")

# =============================================================================
# 2. Experimental Parameter Configuration
# =============================================================================
BLOCK_SIZES = [16, 128]
BLOCK_NUMBERS = [16, 1024, 16384]
EVICTION_POLICIES = ["LRU", "LFU", "FIFO"]

TEST_CONFIGS = []
for bs in BLOCK_SIZES:
    for bn in BLOCK_NUMBERS:
        for ep in EVICTION_POLICIES:
            TEST_CONFIGS.append({"block_size": bs, "block_number": bn, "eviction_policy": ep})

# =============================================================================
# 3. Data Preparation
# =============================================================================
tokenizer = AutoTokenizer.from_pretrained(model_path)
# [Critical Modification] Unified length limit to avoid Warnings
MAX_TOKENS = 8192
tokenizer.model_max_length = 1_000_000_000  # Suppress warning

print("\n【Step 1】Preparing Trace Files...")

# A. Filter Multi-turn
conversations = defaultdict(list)
with open(multi_turn_trace, 'r') as f:
    for line in f:
        entry = json.loads(line.strip())
        conv_id = entry.get('conversation_id', 'unknown')
        # Prioritize reading length from meta to avoid duplicate tokenization
        if 'meta' in entry and 'token_len' in entry['meta']:
            entry['token_count'] = entry['meta']['token_len']
        else:
            prompt_tokens = tokenizer.encode(entry['prompt'], add_special_tokens=False)
            entry['token_count'] = len(prompt_tokens)
        conversations[conv_id].append(entry)

filtered_multi_convs = {}
for conv_id, turns in conversations.items():
    if len(turns) >= 2 and all(turn['token_count'] <= MAX_TOKENS for turn in turns):
        filtered_multi_convs[conv_id] = turns

NUM_CONVS_TO_TEST = len(filtered_multi_convs)
selected_convs = dict(list(filtered_multi_convs.items())[:NUM_CONVS_TO_TEST])
total_requests = sum(len(turns) for turns in selected_convs.values())

fd_multi, filtered_multi_trace_path = tempfile.mkstemp(suffix='_multi.jsonl')
with open(filtered_multi_trace_path, 'w') as f:
    for conv_id in sorted(selected_convs.keys()):
        for turn in selected_convs[conv_id]:
            f.write(json.dumps(turn) + '\n')
os.close(fd_multi)

# B. Filter Single-turn
single_requests = []
with open(single_turn_trace, 'r') as f:
    for line in f:
        entry = json.loads(line.strip())
        # Simple filtering
        if len(tokenizer.encode(entry['prompt'], add_special_tokens=False)) <= MAX_TOKENS:
            single_requests.append(entry)
        if len(single_requests) >= total_requests:
            break

fd_single, filtered_single_trace_path = tempfile.mkstemp(suffix='_single.jsonl')
with open(filtered_single_trace_path, 'w') as f:
    for entry in single_requests:
        f.write(json.dumps(entry) + '\n')
os.close(fd_single)

print(f"✓ Data ready. Total Requests: {total_requests}")


# =============================================================================
# 4. Macro Experiment Logic
# =============================================================================
def run_single_experiment(trace_path, is_multi_turn, config):
    bs = config['block_size']
    bn = config['block_number']
    ep = config['eviction_policy']
    mode_name = "Multi" if is_multi_turn else "Single"
    print(f"Running {mode_name}: BS={bs}, BN={bn}, Policy={ep}...", end="\r")

    # 1. Set environment variables
    os.environ["VLLM_TEST_BLOCK_NUMBER"] = str(bn)
    os.environ["VLLM_TEST_EVICTION_POLICY"] = ep

    # [Critical Fix] Add this line to tell Tracker where the data is!
    os.environ["VLLM_SIM_TRACE_PATH"] = str(trace_path)

    global_hit_rate_tracker.reset()

    # Engine Args
    engine_args = EngineArgs(
        model=model_path, tokenizer=model_path, device="cpu",
        max_model_len=MAX_TOKENS, max_num_seqs=16, block_size=bs,
        enable_prefix_caching=True, gpu_memory_utilization=0.9, enforce_eager=True
    )

    try:
        engine = LLMEngine.from_engine_args(engine_args)
    except Exception as e:
        print(f"\n❌ Engine Error: {e}")  # Print error on new line for visibility
        return None

    simulator = ClientSimulator(trace_path=trace_path, tokenizer=tokenizer, arrival_rate=1.0)
    start_t = time.time()

    if is_multi_turn:
        simulator.send_requests_conversation_by_conversation(engine, max_steps_per_turn=5000)
    else:
        simulator.send_requests_to_engine(engine)
        simulator.run_engine_until_complete(engine, max_steps=10000)

    duration = time.time() - start_t
    stats = global_hit_rate_tracker.get_stats()

    # Clean up environment
    os.environ.pop("VLLM_TEST_BLOCK_NUMBER", None)
    os.environ.pop("VLLM_TEST_EVICTION_POLICY", None)
    os.environ.pop("VLLM_SIM_TRACE_PATH", None)  # Remember to clean up

    del engine
    import gc
    gc.collect()

    return {"hit_rate": stats['overall_hit_rate'], "duration": duration}


# =============================================================================
# 5. Micro-Metrics Collector
# =============================================================================
def collect_advanced_metrics(trace_path, block_size=16):
    """
    Does not run vLLM, but simulates Block Hash logic to calculate required metrics:
    1. Shared Fraction (CDF)
    2. Hits per Block (Histogram)
    3. Reuse Gap (CDF)
    """
    print(f"\n🔬 Deep Diving into Advanced Metrics (BS={block_size})...")

    # Result container
    metrics = {
        "shared_fractions": [],
        "block_access_counts": defaultdict(int),
        "reuse_gaps": []
    }

    # Helper structures
    block_last_access = {}  # {block_hash: global_request_idx}
    global_request_idx = 0

    # Simulated cache state (Conversation ID -> Last Token IDs)
    history_cache = {}

    with open(trace_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            conv_id = entry.get('conversation_id')
            prompt = entry['prompt']

            # 1. Tokenize
            token_ids = tokenizer.encode(prompt, add_special_tokens=False)
            curr_len = len(token_ids)
            if curr_len == 0: continue

            # --- Metric 1: Shared Fraction ---
            # Calculate overlap between current Prompt and previous history
            last_tokens = history_cache.get(conv_id, [])
            common_len = 0
            min_len = min(len(last_tokens), curr_len)

            # Find common prefix length
            for i in range(min_len):
                if last_tokens[i] == token_ids[i]:
                    common_len += 1
                else:
                    break

            fraction = common_len / curr_len
            metrics["shared_fractions"].append(fraction)

            # Update history
            history_cache[conv_id] = token_ids

            # --- Metric 2 & 3: Block Analysis ---
            # Slice Tokens into Blocks and simulate Hash
            num_blocks = curr_len // block_size

            for b_idx in range(num_blocks):
                # Extract Block content
                block_content = tuple(token_ids[b_idx * block_size: (b_idx + 1) * block_size])
                # Generate pseudo-Hash (content is Hash)
                block_hash = hash(block_content)

                # Metric 2: Access Count
                metrics["block_access_counts"][block_hash] += 1

                # Metric 3: Reuse Gap
                if block_hash in block_last_access:
                    gap = global_request_idx - block_last_access[block_hash]
                    metrics["reuse_gaps"].append(gap)

                # Update access time
                block_last_access[block_hash] = global_request_idx

            global_request_idx += 1

    return metrics


# =============================================================================
# 6. Main Loop
# =============================================================================
results_summary = []
print("\n🚀 Starting Parameter Sweep...")

for i, config in enumerate(TEST_CONFIGS):
    print(f"\n[{i + 1}/{len(TEST_CONFIGS)}] Config: {config}")

    # 1. Run Standard Experiments
    single_res = run_single_experiment(filtered_single_trace_path, False, config)
    multi_res = run_single_experiment(filtered_multi_trace_path, True, config)

    if single_res and multi_res:
        imp = multi_res['hit_rate'] - single_res['hit_rate']

        # Calculate Average Latency
        avg_lat = multi_res['duration'] / total_requests if total_requests > 0 else 0

        print(f"   >>> Hit Rate: {multi_res['hit_rate']:.2%} | Latency: {avg_lat * 1000:.1f}ms")

        results_summary.append({
            "config": config,
            "single_turn": single_res,
            "multi_turn": multi_res,
            "improvement": imp,
            "avg_latency": avg_lat  # Save for plotting
        })

# =============================================================================
# 7. Deep Analysis & Save
# =============================================================================
print("\n" + "=" * 80)
print("📊 Collecting Micro-Metrics for Professor's Requirements...")

# Select one best configuration for deep analysis (BS=16, Multi-turn)
advanced_metrics = collect_advanced_metrics(filtered_multi_trace_path, block_size=16)

# Convert Block Counts to list (store counts only, drop Hash for anonymity and size reduction)
block_hits_distribution = list(advanced_metrics["block_access_counts"].values())

final_output = {
    "sweep_results": results_summary,
    "advanced_metrics": {
        "shared_fractions": advanced_metrics["shared_fractions"],
        "block_hits": block_hits_distribution,
        "reuse_gaps": advanced_metrics["reuse_gaps"]
    }
}

output_file = Path(__file__).parent / "milestone2_task3_results_complete.json"
with open(output_file, 'w') as f:
    json.dump(final_output, f, indent=2)

print(f"\n✓ All Data Saved to: {output_file}")

# Cleanup
if os.path.exists(filtered_multi_trace_path): os.unlink(filtered_multi_trace_path)
if os.path.exists(filtered_single_trace_path): os.unlink(filtered_single_trace_path)

# =============================================================================
# 8. Auto-generate Plotting Script Hint
# =============================================================================
print("\n💡 Now use 'sharegpt_workload_visual.py' to summarize and visualize results:")
print("   1. Hit Rate vs Capacity (Line Chart)")
print("   2. Shared Fraction (CDF)")
print("   3. Block Hits (Histogram)")
print("   4. Reuse Gap (CDF)")