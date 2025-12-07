#!/usr/bin/env python3
"""
Multi-turn vs Single-turn Prefix Cache Hit Rate Experiment
Parameter Sweep: Block Size, Block Number (Env), Eviction Policy (Env)
"""
import sys
import os
import json
import tempfile
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any

# =============================================================================
# 1. 路径设置与导入
# =============================================================================
# 假设当前脚本位于 milestones/milestone2/ 目录 (根据您之前的路径推断)
sys.path.insert(0, str(Path(__file__).parent.parent))

from vllm import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from transformers import AutoTokenizer

# 导入自定义 Tracker 和 Simulator
try:
    from correct_hit_rate_tracker import global_hit_rate_tracker
    from cache_block_tracker import global_cache_block_tracker
    from milestone2_code.client_simulator import ClientSimulator
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Please ensure 'correct_hit_rate_tracker.py' and 'milestone2_code/' are in the path.")
    sys.exit(1)

# 模型与Trace路径
model_path = str(Path(__file__).parent.parent / "exported_models" / "Llama-3.2-1B-Instruct")
multi_turn_trace = str(Path(__file__).parent / "traces" / "sharegpt_multi_turn.jsonl")
single_turn_trace = str(Path(__file__).parent / "traces" / "sharegpt_single_turn.jsonl")

# =============================================================================
# 2. 实验参数配置 (在此处修改要测试的范围)
# =============================================================================

# 1. Block Sizes (vLLM Engine Argument)
BLOCK_SIZES = [16, 128] 
# 注意：1024作为block size可能过大，会导致碎片化，建议测试 16, 64, 128

# 2. Block Numbers (Passed via ENV: VLLM_TEST_BLOCK_NUMBER)
# 这些值应该根据 block_size 调整，或者设为固定值 (代表 GPU 显存大小)
BLOCK_NUMBERS = [16, 64, 1024, 16384] 

# 3. Eviction Policies (Passed via ENV: VLLM_TEST_EVICTION_POLICY)
EVICTION_POLICIES = ["LRU", "LFU", "FIFO"]

# 生成测试组合
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
print(f"Experimental Configuration Loaded")
print(f"Total Combinations: {len(TEST_CONFIGS)}")
print("=" * 80)

# =============================================================================
# 3. 数据准备 (Data Preparation)
# =============================================================================
tokenizer = AutoTokenizer.from_pretrained(model_path)
MAX_TOKENS = 2048

print("\n【Step 1】Preparing Trace Files...")

# A. 过滤 Multi-turn
conversations = defaultdict(list)
with open(multi_turn_trace, 'r') as f:
    for line in f:
        entry = json.loads(line.strip())
        conv_id = entry.get('conversation_id', 'unknown')
        prompt_tokens = tokenizer.encode(entry['prompt'], add_special_tokens=False)
        entry['token_count'] = len(prompt_tokens)
        conversations[conv_id].append(entry)

filtered_multi_convs = {}
for conv_id, turns in conversations.items():
    if len(turns) >= 2 and all(turn['token_count'] <= MAX_TOKENS for turn in turns):
        filtered_multi_convs[conv_id] = turns

# 选择 Conversation 数量 (全量或部分)
NUM_CONVS_TO_TEST = len(filtered_multi_convs) # run all
# NUM_CONVS_TO_TEST = 20 # fast debug
selected_convs = dict(list(filtered_multi_convs.items())[:NUM_CONVS_TO_TEST])
total_requests = sum(len(turns) for turns in selected_convs.values())

fd_multi, filtered_multi_trace_path = tempfile.mkstemp(suffix='_multi.jsonl')
with open(filtered_multi_trace_path, 'w') as f:
    for conv_id in sorted(selected_convs.keys()):
        for turn in selected_convs[conv_id]:
            f.write(json.dumps(turn) + '\n')
os.close(fd_multi)

# B. 过滤 Single-turn (匹配 Request 数量)
single_requests = []
with open(single_turn_trace, 'r') as f:
    for line in f:
        entry = json.loads(line.strip())
        if len(tokenizer.encode(entry['prompt'], add_special_tokens=False)) <= MAX_TOKENS:
            single_requests.append(entry)
        if len(single_requests) >= total_requests:
            break

fd_single, filtered_single_trace_path = tempfile.mkstemp(suffix='_single.jsonl')
with open(filtered_single_trace_path, 'w') as f:
    for entry in single_requests:
        f.write(json.dumps(entry) + '\n')
os.close(fd_single)

print(f"✓ Data ready.")
print(f"  Multi-turn path: {filtered_multi_trace_path}")
print(f"  Single-turn path: {filtered_single_trace_path}")
print(f"  Total Requests per run: {total_requests}")


# =============================================================================
# 4. 实验核心逻辑
# =============================================================================
def run_single_experiment(
    trace_path: str,
    is_multi_turn: bool,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    
    bs = config['block_size']
    bn = config['block_number']
    ep = config['eviction_policy']
    mode_name = "Multi-Turn" if is_multi_turn else "Single-Turn"

    print(f"\n--- Running: {mode_name} | BS={bs} | BN={bn} | Policy={ep} ---")

    # 1. 设置环境变量 (Injecting into vLLM)
    os.environ["VLLM_TEST_BLOCK_NUMBER"] = str(bn)
    os.environ["VLLM_TEST_EVICTION_POLICY"] = ep
    
    # 设置 Trace 路径供 Tracker 使用 (如果需要)
    os.environ["VLLM_SIM_TRACE_PATH"] = trace_path

    # 2. 重置 Trackers
    global_hit_rate_tracker.reset()
    global_cache_block_tracker.reset()

    # 3. 初始化 Engine
    # 注意: 我们使用 CPU 模式以避免显存 OOM，并通过 Env Var 强制设置内部的 Block Number
    engine_args = EngineArgs(
        model=model_path,
        tokenizer=model_path,
        device="cpu", 
        max_model_len=2048,
        max_num_seqs=1,
        block_size=bs, # 通过参数传递 Block Size
        enable_prefix_caching=True,
        gpu_memory_utilization=0.9, # CPU模式下这个参数影响较小，但也保留
        enforce_eager=True # 简化图执行
    )
    
    try:
        engine = LLMEngine.from_engine_args(engine_args)
    except Exception as e:
        print(f"❌ Engine Init Failed: {e}")
        return None

    # 4. 初始化 Simulator
    simulator = ClientSimulator(
        trace_path=trace_path,
        tokenizer=tokenizer,
        arrival_rate=1.0,
    )

    # 5. 运行模拟
    start_t = time.time()
    if is_multi_turn:
        # Conversation 模式: 逐个对话处理，允许复用
        simulator.send_requests_conversation_by_conversation(engine, max_steps_per_turn=5000)
    else:
        # Standard 模式: 一次性发送所有
        simulator.send_requests_to_engine(engine)
        simulator.run_engine_until_complete(engine, max_steps=10000)
    
    duration = time.time() - start_t

    # 6. 收集结果
    stats = global_hit_rate_tracker.get_stats()
    
    # 获取 vLLM 内部统计 (如果是 GPU 模式才有效，CPU 模式下通常返回 -1 或 0)
    # 我们主要依赖 global_hit_rate_tracker 的逻辑统计
    from vllm.utils import Device
    vllm_gpu_hit_rate = engine.scheduler[0].get_prefix_cache_hit_rate(Device.GPU)
    
    # 7. 清理环境变量
    os.environ.pop("VLLM_TEST_BLOCK_NUMBER", None)
    os.environ.pop("VLLM_TEST_EVICTION_POLICY", None)
    
    # 显式释放 engine 资源 (尽量)
    del engine
    import gc
    gc.collect()

    return {
        "hit_rate": stats['overall_hit_rate'],
        "total_requests": stats['total_requests'],
        "vllm_internal_hit_rate": vllm_gpu_hit_rate,
        "duration": duration,
        "detailed_stats": stats
    }

# =============================================================================
# 5. 主循环 (Main Loop)
# =============================================================================
results_summary = []

print("\n🚀 Starting Parameter Sweep...")

for i, config in enumerate(TEST_CONFIGS):
    print(f"\n[{i+1}/{len(TEST_CONFIGS)}] Processing Configuration...")
    
    # Run Single Turn
    single_res = run_single_experiment(filtered_single_trace_path, False, config)
    
    # Run Multi Turn
    multi_res = run_single_experiment(filtered_multi_trace_path, True, config)
    
    if single_res and multi_res:
        # Calculate Improvement
        s_rate = single_res['hit_rate']
        m_rate = multi_res['hit_rate']
        imp = m_rate - s_rate
        
        entry = {
            "config": config,
            "single_turn": {
                "hit_rate": s_rate,
                "duration": single_res['duration']
            },
            "multi_turn": {
                "hit_rate": m_rate,
                "duration": multi_res['duration']
            },
            "improvement": imp
        }
        results_summary.append(entry)
        
        print(f"   >>> Result: Single={s_rate:.2%} | Multi={m_rate:.2%} | Diff={imp:+.2%}")
    else:
        print("   >>> Result: FAILED")

# =============================================================================
# 6. 结果输出与保存
# =============================================================================
print("\n" + "="*100)
print(f"{'BS':<5} | {'BN':<8} | {'Policy':<6} | {'Single Turn':<12} | {'Multi Turn':<12} | {'Improvement':<12}")
print("-" * 100)

for res in results_summary:
    c = res['config']
    print(f"{c['block_size']:<5} | {c['block_number']:<8} | {c['eviction_policy']:<6} | "
          f"{res['single_turn']['hit_rate']:<12.2%} | {res['multi_turn']['hit_rate']:<12.2%} | "
          f"{res['improvement']:<+12.2%}")

print("-" * 100)

output_file = Path(__file__).parent / "task2_results_sweep.json"
with open(output_file, 'w') as f:
    json.dump(results_summary, f, indent=2)

print(f"\n✓ Results saved to: {output_file}")

# Cleanup
if os.path.exists(filtered_multi_trace_path):
    os.unlink(filtered_multi_trace_path)
if os.path.exists(filtered_single_trace_path):
    os.unlink(filtered_single_trace_path)

print("Done.")