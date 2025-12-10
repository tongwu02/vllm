#!/usr/bin/env python3
"""
Task 3 Victory: Micro-Scale Double Tap
Shrinks prompt size to guarantee physical fit, ensuring policy logic is the ONLY variable.
"""
import sys
import os
import json
import uuid
from pathlib import Path

# Setup Paths
sys.path.insert(0, str(Path(__file__).parent.parent))

from vllm import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from transformers import AutoTokenizer

try:
    from correct_hit_rate_tracker import global_hit_rate_tracker
    from milestone2_code.client_simulator import ClientSimulator
except:
    pass

# ================= Key Configuration =================
MODEL_PATH = str(Path(__file__).parent.parent / "exported_models" / "Llama-3.2-1B-Instruct")
TRACE_PATH = str(Path(__file__).parent / "traces" / "mixed_workload_standard.jsonl")

# Capacity = 1024 tokens
BLOCK_NUMBERS = [64]
POLICIES = ["LRU", "PROTECTED_LRU"]  # Only compare these two opposing policies


# ================= 1. Generate Micro-Scale Data =================
def generate_micro_workload():
    print("1. Generating Micro-Scale Workload...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    except:
        return

    # [Critical] Only 200 tokens, absolutely safe
    SHARED_SYSTEM_PROMPT = "You are a persistent AI context. " * 30

    mixed_trace = []

    # Generate 20 Clusters (enough to show statistical difference)
    for i in range(20):

        # VIP Prompt
        vip_prompt = tokenizer.apply_chat_template([
            {"role": "system", "content": SHARED_SYSTEM_PROMPT},
            {"role": "user", "content": f"Query_{i}"}
        ], tokenize=False, add_generation_prompt=True)

        # 1. First Tap (Cold Start)
        mixed_trace.append({
            "conversation_id": f"vip_{i}_1",
            "turn_index": 0, "prompt": vip_prompt, "response": ""
        })

        # 2. Second Tap (Promote!)
        mixed_trace.append({
            "conversation_id": f"vip_{i}_2",
            "turn_index": 0, "prompt": vip_prompt, "response": ""
        })

        # 3. Flood (20 Small Noises)
        # 20 * 100 = 2000 tokens > 1024. Must trigger eviction.
        for j in range(20):
            noise_content = f"[NOISE_{uuid.uuid4()}] " + "flood " * 20
            mixed_trace.append({
                "conversation_id": f"noise_{i}_{j}",
                "turn_index": 0, "prompt": noise_content, "response": ""
            })

    os.makedirs(os.path.dirname(TRACE_PATH), exist_ok=True)
    with open(TRACE_PATH, 'w') as f:
        for item in mixed_trace:
            f.write(json.dumps(item) + "\n")
    print(f"   -> Generated {len(mixed_trace)} requests.")


# ================= 2. Run Experiment =================
def run_experiment(policy):
    print(f"Testing {policy:<14} ...", end=" ")

    os.environ["VLLM_TEST_BLOCK_NUMBER"] = "64"
    os.environ["VLLM_TEST_EVICTION_POLICY"] = policy
    os.environ["VLLM_SIM_TRACE_PATH"] = TRACE_PATH

    global_hit_rate_tracker.reset()

    # Serial mode, ensures cleanest logic
    engine_args = EngineArgs(
        model=MODEL_PATH, tokenizer=MODEL_PATH, device="cpu",
        max_model_len=8192, max_num_seqs=1, block_size=16,
        enable_prefix_caching=True, gpu_memory_utilization=0.9, enforce_eager=True
    )

    try:
        # Suppress logs
        import logging
        logging.getLogger("vllm").setLevel(logging.ERROR)
        engine = LLMEngine.from_engine_args(engine_args)
    except:
        return 0.0

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    simulator = ClientSimulator(trace_path=TRACE_PATH, tokenizer=tokenizer, arrival_rate=1.0)
    simulator.send_requests_conversation_by_conversation(engine, max_steps_per_turn=5000)

    stats = global_hit_rate_tracker.get_stats()

    del engine
    import gc;
    gc.collect()

    hr = stats['overall_hit_rate']
    print(f"-> {hr:.2%}")
    return hr


# ================= 3. Main Program =================
if __name__ == "__main__":
    generate_micro_workload()

    print("\n" + "=" * 50)
    print("🔬 MICRO-SCALE VICTORY CHECK")
    print("=" * 50)

    results = {}
    for pol in POLICIES:
        results[pol] = run_experiment(pol)

    print("\n" + "=" * 50)
    print(f"{'Policy':<15} | {'Hit Rate':<10} | {'Outcome':<15}")
    print("-" * 50)

    lru = results["LRU"]
    prot = results["PROTECTED_LRU"]

    # LRU: VIP hits on 2nd tap (1/22=4.5%), then evicted by next cycle
    # Prot: VIP hits on 2nd tap, then hits on 1st tap of next cycle, 2nd tap... (3/22=13.6%)

    # LRU Status
    print(f"{'LRU':<15} | {lru:<10.2%} | {'🔹 BASELINE':<15}")

    # Protected Status
    status = "🔸 TIED"
    if prot > lru + 0.02: status = "🏆 WINNER"
    print(f"{'PROTECTED_LRU':<15} | {prot:<10.2%} | {status:<15}")
    print("-" * 50)

    if prot > lru * 1.5:
        print("\n✅ SUCCESS: Protected LRU hit rate is significantly higher!")
        print("   This proves the 'Probation vs Protected' partition works.")