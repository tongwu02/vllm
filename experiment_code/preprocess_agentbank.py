#!/usr/bin/env python3
"""
Custom AgentBank Preprocessing Script (with length filtering)
Replaces teammate's preprocessing logic to solve token length issues and unify output format.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, List
from transformers import AutoTokenizer
from datasets import load_dataset

# ================= Configuration =================
# Limit max tokens (Task 1 baseline suggests 8192)
MAX_ALLOWED_TOKENS = 8192

ROOT_DIR = Path(__file__).parent.parent
MODEL_PATH = str(ROOT_DIR / "exported_models" / "Llama-3.2-1B-Instruct")
OUTPUT_DIR = Path("traces")
OUTPUT_FILENAME = "agentbank_multi_turn.jsonl"

# AgentBank sub-task configuration
DEFAULT_CONFIGS = ["apps", "gsm8k", "strategyqa"]


# =======================================

def get_tokenizer():
    print(f"⏳ Loading tokenizer from {MODEL_PATH}...")
    try:
        return AutoTokenizer.from_pretrained(MODEL_PATH)
    except Exception as e:
        print(f"⚠️ Load local tokenizer failed: {e}")
        return AutoTokenizer.from_pretrained("gpt2")


def normalize_role(role: str) -> str:
    role = role.lower()
    if role in ['human', 'user']:
        return 'user'
    if role in ['gpt', 'chatgpt', 'assistant', 'model']:
        return 'assistant'
    if role == 'system':
        return 'system'
    return 'user'  # fallback


def iter_agentbank_samples(config: str, split: str, limit: Optional[int]) -> Iterable[Dict]:
    """Stream load HuggingFace dataset"""
    print(f"   Downloading/Loading AgentBank config: '{config}'...")
    try:
        dataset = load_dataset("Solaris99/AgentBank", config, split=split, streaming=True)
        count = 0
        for sample in dataset:
            yield sample
            count += 1
            if limit and count >= limit:
                break
    except Exception as e:
        print(f"   ❌ Error loading config {config}: {e}")


def process_agentbank(configs: List[str], output_path: Path, tokenizer, max_samples: int):
    print(f"🚀 Processing AgentBank -> {output_path}")
    print(f"🛡️  Filtering Threshold: {MAX_ALLOWED_TOKENS} tokens")

    kept_count = 0
    skipped_count = 0
    processed_count = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f_out:

        for config_name in configs:
            print(f"📂 Processing Sub-task: {config_name}")

            for sample in iter_agentbank_samples(config_name, "train", max_samples):
                processed_count += 1

                # 1. Extract Conversations
                # AgentBank structure is usually also conversations: [{'from':..., 'value':...}]
                raw_convs = sample.get('conversations', [])
                if not raw_convs:
                    continue

                # 2. Convert format & Normalize
                messages = []
                # Uniformly add System Prompt
                messages.append({"role": "system", "content": "You are a helpful assistant."})

                for turn in raw_convs:
                    role = normalize_role(turn.get('from', ''))
                    content = turn.get('value', '')
                    if not content.strip():
                        continue
                    messages.append({"role": role, "content": content})

                # Ensure at least one round of user/assistant
                if len(messages) < 2:
                    continue

                # 3. Length Check (Core Step)
                full_text = "".join([m['content'] for m in messages])
                token_ids = tokenizer.encode(full_text, add_special_tokens=False)

                if len(token_ids) > MAX_ALLOWED_TOKENS:
                    skipped_count += 1
                    continue  # Discard too long ones

                # 4. Construct output entry
                sample_id = sample.get("id", processed_count)
                entry = {
                    "conversation_id": f"agentbank-{config_name}-{sample_id}",
                    "workload": f"agentbank/{config_name}",
                    "messages": messages,
                    "meta": {
                        "dataset": "AgentBank",
                        "config": config_name,
                        "token_len": len(token_ids)
                    }
                }

                f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")
                kept_count += 1

                if processed_count % 100 == 0:
                    print(f"   Stats: Kept {kept_count} | Skipped {skipped_count}", end='\r')

            print(f"\n   Finished {config_name}.\n")

    print("=" * 60)
    print(f"✅ Done! File saved to: {output_path}")
    print(f"   Total Processed: {processed_count}")
    print(f"   Kept (<= {MAX_ALLOWED_TOKENS}): {kept_count}")
    print(f"   Skipped (> {MAX_ALLOWED_TOKENS}): {skipped_count}")
    print("=" * 60)


if __name__ == "__main__":
    tokenizer = get_tokenizer()
    output_path = OUTPUT_DIR / OUTPUT_FILENAME

    # Max samples per config, can be set larger because some will be filtered out
    MAX_SAMPLES_PER_CONFIG = 500

    process_agentbank(DEFAULT_CONFIGS, output_path, tokenizer, MAX_SAMPLES_PER_CONFIG)