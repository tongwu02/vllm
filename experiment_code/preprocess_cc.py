#!/usr/bin/env python3
"""
Preprocess CC-Bench: Correct Multi-Turn Mode
Generates sequential requests for prefix caching testing by treating the trajectory
as a growing conversation history.
"""

import argparse
import json
import sys
import logging
from pathlib import Path
from typing import Dict, Optional, List
from transformers import AutoTokenizer
from datasets import load_dataset

# Suppress Tokenizer warnings
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

MAX_ALLOWED_TOKENS = 8000
MODEL_PATH = str(ROOT_DIR / "exported_models" / "Llama-3.2-1B-Instruct")


def parse_args() -> argparse.Namespace:
    # ... (arguments remain the same)
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-samples", type=int, default=500, help="Number of samples to process")
    parser.add_argument("--output-dir", type=Path, default=ROOT_DIR / "experiment_code" / "traces")
    parser.add_argument("--output-filename", type=str, default="ccbench_multi_turn.jsonl")
    return parser.parse_args()


def iter_ccbench_samples(split: str, limit: Optional[int]):
    # ... (dataset loading remains the same)
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


def process_sample_into_turns(sample: Dict, tokenizer) -> List[Dict]:
    """
    Transforms one CC-Bench sample into a list of sequential, multi-turn entries.
    Each entry reuses the messages from the previous entry as its prefix.
    """
    sample_id = str(sample.get("id"))
    task_category = sample.get("task_category", "unknown")
    
    # The 'trajectory' is typically a structured list that needs parsing
    traj_data = sample.get("trajectory", [])
    if isinstance(traj_data, str):
        try:
            traj_data = json.loads(traj_data)
        except json.JSONDecodeError:
            # If it's not JSON, we cannot properly segment it. Skip this sample.
            return []
            
    if not isinstance(traj_data, list):
        # Data format is unexpected, skip.
        return []
    
    entries = []
    # Start the conversation with the system prompt
    current_messages = [
        {"role": "system", "content": "You are a helpful coding assistant."}
    ]
    
    # Iterate through the steps of the trajectory
    for i, step in enumerate(traj_data):
        # 1. Extract the current turn's message
        # Based on your file snippet, the message is nested inside a 'message' key
        message_data = step.get('message', {})
        role = message_data.get('role', 'user')
        content = message_data.get('content', '')
        
        if not content: continue # Skip empty steps
        
        # 2. Add the new message to the history
        new_message = {"role": role, "content": content}
        current_messages.append(new_message)
        
        # 3. Check for truncation before generating the entry
        # Apply the chat template to the current history to get the full prompt text
        full_prompt_text = tokenizer.apply_chat_template(current_messages, tokenize=False)
        token_ids = tokenizer.encode(full_prompt_text, add_special_tokens=False)
        
        if len(token_ids) > MAX_ALLOWED_TOKENS:
            # The current message makes the context too long. Stop the conversation here.
            break
            
        # 4. Create the JSONL entry for the current turn
        # This entry contains the full conversation history up to this point (the prefix)
        entry = {
            # Each turn must have a unique ID for the simulator to treat it as a new request
            "conversation_id": f"ccbench-{sample_id}", 
            "turn_index": i + 1, # Indexing is critical for tracking turns
            "workload": f"ccbench/{task_category}",
            "messages": current_messages.copy(), # MUST copy the list for isolation
            "meta": {
                "dataset": "CC-Bench",
                "final_len": len(token_ids),
                "turn": i + 1
            }
        }
        entries.append(entry)
        
        # 5. After a 'user' message, insert a dummy 'assistant' response to complete the loop
        # This makes the history reusable for the *next* user query.
        if role == 'user' and (i + 1) < len(traj_data):
             # A simple placeholder response that will be part of the next prefix
             current_messages.append({"role": "assistant", "content": "Acknowledged."})
             
    return entries


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / args.output_filename

    print(f"Loading tokenizer from: {MODEL_PATH}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        tokenizer.model_max_length = 1_000_000_000
    except:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.model_max_length = 1_000_000_000

    print(f"Processing CC-Bench... (Max length {MAX_ALLOWED_TOKENS} tokens)")

    processed_conv_count = 0
    generated_turn_count = 0

    with output_path.open("w", encoding="utf-8") as f_out:
        for sample in iter_ccbench_samples(args.split, args.max_samples):
            # NEW: Process one sample into multiple turn entries
            entries = process_sample_into_turns(sample, tokenizer)
            
            if entries:
                processed_conv_count += 1
                
                # Write all turns for this sample to the file
                for entry in entries:
                    f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    generated_turn_count += 1
                
                # Print progress (using the turn count)
                print(f"   Conversation {sample.get('id'):<3}: Generated {len(entries)} turns.")


    print("=" * 60)
    print(f"Done. File saved to: {output_path}")
    print(f"Total Conversations Processed: {processed_conv_count}")
    print(f"Total Turns Generated: {generated_turn_count}")
    print("=" * 60)


if __name__ == "__main__":
    main()