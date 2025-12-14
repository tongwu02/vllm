#!/usr/bin/env python3
"""
Task 1: Baseline Performance Testing (Final Version)
- Automatically handles format conversion from 'messages' to 'prompt'
- Automatically truncates ultra-long traces to prevent OOM
- Automatically prints a formatted final result table after execution
"""
import sys
import os
import json
import time
import gc
from pathlib import Path
from typing import Dict, Any, List

# =============================================================================
# 1. Path Setup
# =============================================================================
# Ensure vllm module can be found
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

# Model Path
model_path = str(Path(__file__).parent.parent / "exported_models" / "Llama-3.2-1B-Instruct")
TRACE_DIR = Path(__file__).parent / "traces"

# Dataset path configuration
DATASETS = {
    "ShareGPT": TRACE_DIR / "sharegpt_multi_turn.jsonl",
    "AgentBank": TRACE_DIR / "agentbank_multi_turn.jsonl",
    "CC": TRACE_DIR / "ccbench_multi_turn.jsonl"
}

# =============================================================================
# 2. Configuration
# =============================================================================

MAX_REQUESTS_PER_DATASET = 500
MAX_TOKENS = 8192

BLOCK_SIZES = [16]

BLOCK_NUMBERS = [64]

EVICTION_POLICIES = ["LRU", "LFU", "FIFO", "PROTECTED_LRU"]

TEST_CONFIGS = []
for bs in BLOCK_SIZES:
    for bn in BLOCK_NUMBERS:
        for ep in EVICTION_POLICIES:
            TEST_CONFIGS.append({
                "block_size": bs,
                "block_number": bn,
                "eviction_policy": ep
            })

print("=" * 80)
print(f"Task 1: Baseline Sweep (Final Auto-Print)")
print(f"Max Requests: {MAX_REQUESTS_PER_DATASET} | Max Tokens: {MAX_TOKENS}")
print("=" * 80)

# Load Tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # Prevent warnings from some special tokenizers
    tokenizer.model_max_length = 1000000000
except Exception as e:
    print(f"Tokenizer error: {e}")
    sys.exit(1)


# =============================================================================
# 3. Helper: Smart Trace Adapter
# =============================================================================
def get_sample_trace(name: str, full_path: Path, limit: int) -> str:
    """
    Read the original file, convert format, and truncate the first 'limit' entries to a temp file.
    """
    temp_path = TRACE_DIR / f"temp_{name}_{limit}.jsonl"
    print(f"   -> Converting & Sampling {name} to {temp_path}...")

    processed_count = 0

    with open(full_path, 'r', encoding='utf-8') as f_in, \
            open(temp_path, 'w', encoding='utf-8') as f_out:

        for line in f_in:
            if processed_count >= limit:
                break

            try:
                data = json.loads(line)
                # Compatibility check: use directly if already in 'prompt' format, convert if 'messages'
                if "messages" in data:
                    messages = data["messages"]
                    conversation_id = data.get("conversation_id", f"conv_{processed_count}")

                    history = []
                    turn_index = 0

                    for msg in messages:
                        if msg['role'] == 'system':
                            history.append(msg)
                            continue

                        if msg['role'] == 'user':
                            current_input = history + [msg]

                            # Convert to String Prompt
                            prompt_str = tokenizer.apply_chat_template(
                                current_input,
                                tokenize=False,
                                add_generation_prompt=True
                            )

                            # Length check
                            token_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
                            if len(token_ids) > MAX_TOKENS:
                                continue

                            entry = {
                                "conversation_id": conversation_id,
                                "turn_index": turn_index,
                                "prompt": prompt_str,
                                "response": ""
                            }
                            f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")

                            history.append(msg)
                            turn_index += 1

                        elif msg['role'] == 'assistant':
                            history.append(msg)

                    processed_count += 1

                else:
                    # If already in old format, write directly
                    f_out.write(line)
                    processed_count += 1

            except Exception as e:
                continue

    return str(temp_path)


# =============================================================================
# 4. Core Experiment
# =============================================================================
def run_single_experiment(dataset_name: str, trace_file: str, config: Dict[str, Any]) -> Dict[str, Any]:
    bs = config['block_size']
    bn = config['block_number']
    ep = config['eviction_policy']

    print(f"Running [{dataset_name}]: BS={bs} | BN={bn} | Policy={ep} ...", end="\r")

    # Set Env Vars
    os.environ["VLLM_TEST_BLOCK_NUMBER"] = str(bn)
    os.environ["VLLM_TEST_EVICTION_POLICY"] = ep
    os.environ["VLLM_SIM_TRACE_PATH"] = str(trace_file)

    global_hit_rate_tracker.reset()
    global_cache_block_tracker.reset()

    # Suppress vLLM logs
    import logging
    logging.getLogger("vllm").setLevel(logging.ERROR)

    engine_args = EngineArgs(
        model=model_path,
        tokenizer=model_path,
        device="cpu",
        max_model_len=MAX_TOKENS,
        max_num_seqs=1,
        block_size=bs,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.9,
        enforce_eager=True
    )

    try:
        engine = LLMEngine.from_engine_args(engine_args)
    except Exception as e:
        print(f"\n❌ Engine Init Failed: {e}")
        return None

    simulator = ClientSimulator(trace_path=str(trace_file), tokenizer=tokenizer, arrival_rate=1.0)

    start_t = time.time()
    simulator.send_requests_conversation_by_conversation(engine, max_steps_per_turn=5000)
    duration = time.time() - start_t

    stats = global_hit_rate_tracker.get_stats()

    # Cleanup
    os.environ.pop("VLLM_TEST_BLOCK_NUMBER", None)
    os.environ.pop("VLLM_TEST_EVICTION_POLICY", None)
    del engine
    gc.collect()

    print(f"Running [{dataset_name}]: BS={bs} | BN={bn} | Policy={ep} -> Hit Rate: {stats['overall_hit_rate']:.2%}")

    return {
        "dataset": dataset_name,
        "hit_rate": stats['overall_hit_rate'],
        "duration": duration,
        "config": config,
    }


# =============================================================================
# 5. Main Loop
# =============================================================================
results_summary = []

for dataset_name, path in DATASETS.items():
    if not os.path.exists(path):
        print(f"⚠️ Skipping {dataset_name}: File not found.")
        continue

    print(f"\n📂 Processing: {dataset_name}")
    temp_trace = get_sample_trace(dataset_name, path, MAX_REQUESTS_PER_DATASET)

    for config in TEST_CONFIGS:
        res = run_single_experiment(dataset_name, temp_trace, config)
        if res:
            results_summary.append(res)

    if os.path.exists(temp_trace):
        os.remove(temp_trace)

# =============================================================================
# 6. Final Print & Save
# =============================================================================

# 1. Save JSON
output_file = Path(__file__).parent / "task1_results_sweep.json"
with open(output_file, 'w') as f:
    json.dump(results_summary, f, indent=2)

# 2. Print final large table
print("\n" + "=" * 80)
print("🏆 FINAL RESULTS SUMMARY")
print("=" * 80)
print(f"{'Dataset':<12} | {'BN':<6} | {'Policy':<15} | {'Hit Rate':<10} | {'Duration':<10}")
print("-" * 80)

# Sort results: Dataset -> BN -> Policy
results_summary.sort(key=lambda x: (x['dataset'], x['config']['block_number'], x['config']['eviction_policy']))

current_dataset = ""
for res in results_summary:
    d = res['dataset']
    c = res['config']

    # Add an empty line between different Datasets for readability
    if current_dataset != "" and d != current_dataset:
        print("-" * 80)
    current_dataset = d

    print(
        f"{d:<12} | {c['block_number']:<6} | {c['eviction_policy']:<15} | {res['hit_rate']:<10.2%} | {res['duration']:<10.2f}s")

print("-" * 80)
print(f"✓ Results saved to: {output_file}")