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
import matplotlib.pyplot as plt

# Setup Paths
sys.path.insert(0, str(Path(__file__).parent.parent))

from vllm import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from transformers import AutoTokenizer

try:
    # Assuming these modules are available in your execution environment
    from correct_hit_rate_tracker import global_hit_rate_tracker
    from milestone2_code.client_simulator import ClientSimulator
except ImportError:
    # Placeholder classes/objects if custom modules are not available
    class GlobalHitRateTracker:
        def reset(self): pass
        def get_stats(self): return {'overall_hit_rate': 0.0}
    global_hit_rate_tracker = GlobalHitRateTracker()
    
    class ClientSimulator:
        def __init__(self, trace_path, tokenizer, arrival_rate): pass
        def send_requests_conversation_by_conversation(self, engine, max_steps_per_turn): pass

# ================= Key Configuration =================
# NOTE: Replace this path with your actual model location if different
MODEL_PATH = str(Path(__file__).parent.parent / "exported_models" / "Llama-3.2-1B-Instruct")
TRACE_PATH = str(Path(__file__).parent / "traces" / "mixed_workload_standard.jsonl")

# Capacity = 1024 tokens
BLOCK_NUMBERS = [64]
POLICIES = ["FIFO", "LFU", "LRU", "PROTECTED_LRU", ]  # Compare all four policies


# ================= 1. Generate Micro-Scale Data =================
def generate_micro_workload():
    print("1. Generating Micro-Scale Workload...")
    try:
        # Check if the model path exists or mock the tokenizer if necessary
        if not Path(MODEL_PATH).exists():
             print(f"   -> WARNING: Model path not found at {MODEL_PATH}. Using a mock setup.")
             # Mock the tokenizer for length calculation
             class MockTokenizer:
                 def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
                     # Simple mock to estimate a consistent length
                     return "A" * 200 # Roughly 200 tokens
                 def from_pretrained(self, *args, **kwargs): return self
             tokenizer = MockTokenizer()
        else:
             tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    except Exception as e:
        print(f"   -> ERROR loading tokenizer: {e}")
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

    # Reset the tracker for each policy
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
        
        # NOTE: This part assumes vLLM and its components are properly set up.
        # It will likely fail without the proper vLLM/LLM model in place.
        engine = LLMEngine.from_engine_args(engine_args)
    except Exception as e:
        print(f"-> FAILED (Engine Init: {e})")
        return 0.0

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    except Exception:
        # Fallback if AutoTokenizer fails (necessary for client simulator)
        class MockTokenizer:
            def __init__(self): self.pad_token_id = 0
            def __call__(self, text, return_tensors): return {'input_ids': [1]*len(text.split())}
        tokenizer = MockTokenizer()
        
    simulator = ClientSimulator(trace_path=TRACE_PATH, tokenizer=tokenizer, arrival_rate=1.0)
    # The simulation runs the workload and updates the global_hit_rate_tracker
    simulator.send_requests_conversation_by_conversation(engine, max_steps_per_turn=5000)

    stats = global_hit_rate_tracker.get_stats()

    # Cleanup resources
    del engine
    import gc
    gc.collect()

    hr = stats.get('overall_hit_rate', 0.0)
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

    # --- Start Plotting Logic ---
    print("\n" + "=" * 50)
    print("📊 GENERATING PLOT")
    print("-" * 50)

    try:
        policies = list(results.keys())
        hit_rates = [results[pol] for pol in policies]

        plt.figure(figsize=(10, 6))
        
        # Color code for better distinction
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] 
        bars = plt.bar(policies, hit_rates, color=colors)

        # Add hit rate labels on top of the bars
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.005, f'{yval:.2%}', 
                     ha='center', va='bottom', fontweight='bold')

        plt.title('Token Caching Hit Rate by Eviction Policy (Micro-Scale Double Tap)', fontsize=14)
        plt.ylabel('Overall Prefix Cache Hit Rate', fontsize=12)
        plt.xlabel('Eviction Policy', fontsize=12)
        plt.ylim(0, max(hit_rates) * 1.25 if hit_rates else 0.5) # Dynamic limit
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plot_filename = "policy_hit_rate_plot.png"
        plt.savefig(plot_filename)
        print(f"✅ Success: Plot generated and saved as '{plot_filename}'")
        print("   (Look for the file in the script's directory)")
        
        # Optional: Display the plot in an interactive environment
        # plt.show() 
        
    except ImportError:
        print("❌ FAILED: matplotlib not found.")
        print("   Please install it using: pip install matplotlib")
    except Exception as e:
        print(f"❌ FAILED: An error occurred during plotting: {e}")

    # --- End Plotting Logic ---

    print("\n" + "=" * 50)
    print(f"{'Policy':<15} | {'Hit Rate':<10} | {'Outcome':<15}")
    print("-" * 50)

    lru = results.get("LRU", 0.0)
    prot = results.get("PROTECTED_LRU", 0.0)

    # Print all results
    for pol in POLICIES:
        hr = results.get(pol, 0.0)
        status = "🔹 BASELINE"
        if pol == "PROTECTED_LRU":
            if prot > lru + 0.02: 
                status = "🏆 WINNER"
            elif prot > lru:
                status = "🔸 IMPROVED"
            else:
                status = "🔹 BASELINE"
        
        print(f"{pol:<15} | {hr:<10.2%} | {status:<15}")
    print("-" * 50)

    if prot > lru * 1.5:
        print("\n✅ SUCCESS: Protected LRU hit rate is significantly higher!")
        print("   This proves the 'Probation vs Protected' partition works.")
    elif prot > lru + 0.02:
        print("\n✅ SUCCESS: Protected LRU shows a notable improvement over LRU.")
        print("   This confirms the policy is functioning as intended.")
    else:
        print("\n⚠️ WARNING: Protected LRU did not show significant improvement or the test failed.")