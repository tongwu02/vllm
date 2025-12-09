#!/usr/bin/env python3
"""
Task 1: Baseline Performance Testing (Fixed Adapter Version)
修复了 KeyError: 'prompt' 问题。
自动将 'messages' 格式转换为 Simulator 需要的 'prompt' 格式。
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

# 指向你刚才生成的 Clean 文件
DATASETS = {
    "ShareGPT": TRACE_DIR / "sharegpt_multi_turn.jsonl",
    "AgentBank": TRACE_DIR / "agentbank_multi_turn.jsonl",
    "CC":        TRACE_DIR / "ccbench_multi_turn.jsonl" 
}

# =============================================================================
# 2. Configuration
# =============================================================================

# 取样数量：500条对话足以测出缓存性能
MAX_REQUESTS_PER_DATASET = 500 
MAX_TOKENS = 8192

BLOCK_SIZES = [16]
BLOCK_NUMBERS = [32, 28] 
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
print(f"Task 1: Baseline Sweep (Adapter Mode)")
print(f"Max Requests per Dataset: {MAX_REQUESTS_PER_DATASET}")
print("=" * 80)

# 加载 Tokenizer (用于把 messages 转成 prompt string)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# =============================================================================
# 3. Helper: Smart Trace Adapter (关键修复)
# =============================================================================
def get_sample_trace(name: str, full_path: Path, limit: int) -> str:
    """
    读取 Clean 格式 (messages list) 的文件，
    将其转换为 Simulator 需要的 Trace 格式 (prompt string)，
    并只保存前 limit 条到临时文件。
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
                # 兼容性检查：如果已经是 prompt 格式就直接用，如果是 messages 就转换
                if "messages" in data:
                    messages = data["messages"]
                    conversation_id = data.get("conversation_id", f"conv_{processed_count}")
                    
                    # --- 核心转换逻辑 ---
                    # 我们需要把整个对话拆解成 Simulator 能看懂的 Single Turn 序列
                    # Simulator 会根据 conversation_id 自动把它们串起来
                    
                    history = []
                    turn_index = 0
                    
                    for msg in messages:
                        if msg['role'] == 'system':
                            history.append(msg)
                            continue
                            
                        if msg['role'] == 'user':
                            # 构造当前轮次的 Prompt：历史 + 当前问题
                            current_input = history + [msg]
                            
                            # 使用 chat_template 变成字符串 
                            prompt_str = tokenizer.apply_chat_template(
                                current_input, 
                                tokenize=False, 
                                add_generation_prompt=True
                            )
                            
                            # === [关键修改] 长度检查 ===
                            # 使用全局常量 MAX_TOKENS
                            token_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
                            if len(token_ids) > MAX_TOKENS:
                                # 如果这一轮太长，跳过这一轮 (或者你可以选择 break 跳过整个对话)
                                print(f"      [Warn] Skipping a turn > {MAX_TOKENS} tokens.")
                                continue 
                            # ============================
                            
                            # 写入 Trace 条目
                            entry = {
                                "conversation_id": conversation_id,
                                "turn_index": turn_index,
                                "prompt": prompt_str,
                                "response": "" # Simulator 运行时不需要真实 response
                            }
                            f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")
                            
                            # 更新历史和轮数
                            history.append(msg)
                            turn_index += 1
                            
                        elif msg['role'] == 'assistant':
                            history.append(msg)
                    
                    processed_count += 1
                    
                else:
                    # 如果已经是旧格式，直接写
                    f_out.write(line)
                    processed_count += 1
                    
            except Exception as e:
                print(f"Warning: failed to convert line: {e}")
                continue
            
    return str(temp_path)

# =============================================================================
# 4. Core Experiment
# =============================================================================
def run_single_experiment(
    dataset_name: str,
    trace_file: str,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    
    bs = config['block_size']
    bn = config['block_number']
    ep = config['eviction_policy']
    capacity_tokens = bs * bn

    print(f"\n--- [{dataset_name}] BS={bs} | BN={bn} | Policy={ep} ---")

    # Env Vars
    os.environ["VLLM_TEST_BLOCK_NUMBER"] = str(bn)
    os.environ["VLLM_TEST_EVICTION_POLICY"] = ep
    os.environ["VLLM_SIM_TRACE_PATH"] = str(trace_file)

    # Reset
    global_hit_rate_tracker.reset()
    global_cache_block_tracker.reset()

    # Engine
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
        print(f"❌ Engine Init Failed: {e}")
        return None

    # Simulator
    simulator = ClientSimulator(
        trace_path=str(trace_file),
        tokenizer=tokenizer,
        arrival_rate=1.0,
    )

    start_t = time.time()
    # 这里的 simulator 现在能读到正确的 prompt 字段了
    simulator.send_requests_conversation_by_conversation(engine, max_steps_per_turn=5000)
    duration = time.time() - start_t

    stats = global_hit_rate_tracker.get_stats()
    
    os.environ.pop("VLLM_TEST_BLOCK_NUMBER", None)
    os.environ.pop("VLLM_TEST_EVICTION_POLICY", None)
    del engine
    gc.collect() 

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
        print(f"⚠️ Skipping {dataset_name}: {path} not found.")
        continue
        
    print(f"\n" + "="*60)
    print(f"📂 Processing: {dataset_name}")
    
    # 这一步现在会自动处理格式转换，生成带 'prompt' 字段的临时文件
    temp_trace = get_sample_trace(dataset_name, path, MAX_REQUESTS_PER_DATASET)
    
    print("="*60)

    for config in TEST_CONFIGS:
        res = run_single_experiment(dataset_name, temp_trace, config)
        if res:
            results_summary.append(res)
            print(f"   >>> Result: Hit Rate={res['hit_rate']:.2%} | Duration={res['duration']:.2f}s")
    
    # 清理临时文件
    if os.path.exists(temp_trace):
        os.remove(temp_trace)

# =============================================================================
# 6. Save
# =============================================================================
print("\n" + "="*100)
print(f"{'Dataset':<12} | {'BN':<5} | {'Policy':<14} | {'Hit Rate':<10}")
print("-" * 100)

for res in results_summary:
    d = res['dataset']
    c = res['config']
    print(f"{d:<12} | {c['block_number']:<5} | {c['eviction_policy']:<14} | {res['hit_rate']:<10.2%}")

output_file = Path(__file__).parent / "task1_results_sweep.json"
with open(output_file, 'w') as f:
    json.dump(results_summary, f, indent=2)

print(f"\n✓ Results saved to: {output_file}")