#!/usr/bin/env python3
"""
Preprocess CC-Bench: Truncate Mode
对于超长数据，不再跳过，而是强制截断到 MAX_ALLOWED_TOKENS。
这样可以保留足够多的样本用于测试。
"""

import argparse
import json
import sys
import logging
from pathlib import Path
from typing import Dict, Optional
from transformers import AutoTokenizer
from datasets import load_dataset

# 屏蔽 Tokenizer 警告
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

# [配置] 截断长度
# 16384 (16k) 足够代表 "Long Context" 负载，且运行速度尚可接受
MAX_ALLOWED_TOKENS = 8000 
MODEL_PATH = str(ROOT_DIR / "exported_models" / "Llama-3.2-1B-Instruct")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-samples", type=int, default=500, help="处理多少条数据")
    parser.add_argument("--output-dir", type=Path, default=ROOT_DIR / "milestone2_code" / "traces")
    parser.add_argument("--output-filename", type=str, default="ccbench_multi_turn.jsonl")
    return parser.parse_args()

def iter_ccbench_samples(split: str, limit: Optional[int]):
    try:
        dataset = load_dataset("zai-org/CC-Bench-trajectories", split=split, streaming=True)
        count = 0
        for sample in dataset:
            yield sample
            count += 1
            if limit and count >= limit:
                break
    except Exception as e:
        print(f"Error loading dataset: {e}")

def make_entry(sample: Dict, tokenizer) -> Dict:
    sample_id = str(sample.get("id"))
    task_category = sample.get("task_category", "unknown")

    # 1. 提取文本
    traj = sample.get("trajectory", "")
    if isinstance(traj, (dict, list)):
        prompt_text = json.dumps(traj, ensure_ascii=False)
    else:
        prompt_text = str(traj)

    # 2. 编码并截断 (Truncate)
    # add_special_tokens=False 避免重复添加 BOS
    token_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    
    original_len = len(token_ids)
    is_truncated = False
    
    if original_len > MAX_ALLOWED_TOKENS:
        # 强制截取前 MAX_ALLOWED_TOKENS 个 token
        token_ids = token_ids[:MAX_ALLOWED_TOKENS]
        # 解码回字符串，确保截断点是合法的字符边界
        prompt_text = tokenizer.decode(token_ids)
        is_truncated = True

    # 3. 构造 Messages
    messages = [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": prompt_text}
    ]

    entry = {
        "conversation_id": f"ccbench-{sample_id}",
        "workload": f"ccbench/{task_category}",
        "messages": messages,
        "meta": {
            "dataset": "CC-Bench",
            "orig_len": original_len,
            "truncated": is_truncated,
            "final_len": len(token_ids)
        }
    }
    return entry, original_len, is_truncated

def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / args.output_filename

    print(f"Loading tokenizer from: {MODEL_PATH}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        # [关键] 设为超大值，防止 encode 时报 Warning
        tokenizer.model_max_length = 1_000_000_000 
    except:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.model_max_length = 1_000_000_000

    print(f"Processing CC-Bench... (Truncating to {MAX_ALLOWED_TOKENS} tokens)")
    
    processed_count = 0
    truncated_count = 0
    
    with output_path.open("w", encoding="utf-8") as f_out:
        for sample in iter_ccbench_samples(args.split, args.max_samples):
            entry, orig_len, is_truncated = make_entry(sample, tokenizer)
            
            f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
            processed_count += 1
            if is_truncated:
                truncated_count += 1
                
            # 打印进度
            status = f"[Truncated {orig_len} -> {MAX_ALLOWED_TOKENS}]" if is_truncated else f"[Kept {orig_len}]"
            print(f"   Sample {sample.get('id')}: {status}")

    print("="*60)
    print(f"Done. File saved to: {output_path}")
    print(f"Total Processed: {processed_count}")
    print(f"Truncated: {truncated_count} (Raw length > {MAX_ALLOWED_TOKENS})")
    print(f"Kept Original: {processed_count - truncated_count}")
    print("="*60)

if __name__ == "__main__":
    main()