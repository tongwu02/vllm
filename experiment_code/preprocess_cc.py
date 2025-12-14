#!/usr/bin/env python3
"""
Preprocess CC-Bench: Truncate Mode
For ultra-long data, do not skip, but force truncate to MAX_ALLOWED_TOKENS.
This ensures enough samples are retained for testing.
"""

import argparse
import json
import sys
import logging
from pathlib import Path
from typing import Dict, Optional
from transformers import AutoTokenizer
from datasets import load_dataset

# Suppress Tokenizer warnings
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

# [Config] Truncation length
# 16384 (16k) is enough to represent "Long Context" workload, and running speed is acceptable
MAX_ALLOWED_TOKENS = 8000
MODEL_PATH = str(ROOT_DIR / "exported_models" / "Llama-3.2-1B-Instruct")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-samples", type=int, default=500, help="Number of samples to process")
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

    # 1. Extract text
    traj = sample.get("trajectory", "")
    if isinstance(traj, (dict, list)):
        prompt_text = json.dumps(traj, ensure_ascii=False)
    else:
        prompt_text = str(traj)

    # 2. Encode and Truncate
    # add_special_tokens=False to avoid duplicate BOS
    token_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

    original_len = len(token_ids)
    is_truncated = False

    if original_len > MAX_ALLOWED_TOKENS:
        # Force truncate to the first MAX_ALLOWED_TOKENS tokens
        token_ids = token_ids[:MAX_ALLOWED_TOKENS]
        # Decode back to string to ensure the truncation point is a valid character boundary
        prompt_text = tokenizer.decode(token_ids)
        is_truncated = True

    # 3. Construct Messages
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
        # [Critical] Set to a very large value to prevent warnings during encode
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

            # Print progress
            status = f"[Truncated {orig_len} -> {MAX_ALLOWED_TOKENS}]" if is_truncated else f"[Kept {orig_len}]"
            print(f"   Sample {sample.get('id')}: {status}")

    print("=" * 60)
    print(f"Done. File saved to: {output_path}")
    print(f"Total Processed: {processed_count}")
    print(f"Truncated: {truncated_count} (Raw length > {MAX_ALLOWED_TOKENS})")
    print(f"Kept Original: {processed_count - truncated_count}")
    print("=" * 60)


if __name__ == "__main__":
    main()