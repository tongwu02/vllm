#!/usr/bin/env python3
"""
Preprocess ShareGPT dataset

Convert ShareGPT_V3_unfiltered_cleaned_split.json to trace format:
- sharegpt_single_turn.jsonl
- sharegpt_multi_turn.jsonl

How to use it:
    python preprocess_sharegpt.py --max-conversations 500
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoTokenizer

from trace_preprocessor import ShareGPTPreprocessor

def parse_args():
    parser = argparse.ArgumentParser(description="ShareGPT pre-process script")
    
    parser.add_argument(
        "--max-conversations", 
        type=int, 
        default=500,
        help="max conversation number (default: 500)"
    )
    
    parser.add_argument(
        "--input-file", 
        type=str, 
        default="ShareGPT_V3_unfiltered_cleaned_split.json",
        help="input file path"
    )

    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="traces",
        help="output file path"
    )

    return parser.parse_args()


def main():
    # parse arguments
    args = parse_args()

    print("=" * 60)
    print("ShareGPT data pre-process")
    print("=" * 60)
    print()

    # Configurations
    input_file = Path(args.input_file)
    output_dir = Path(args.output_dir)
    model_path = str(Path(__file__).parent.parent / 'exported_models' / 'Llama-3.2-1B-Instruct')
    
    # Get maximum number of conversations if provided
    max_conversations = args.max_conversations

    # System prompt Config
    system_prompt = "random" 

    # Check Input Path
    if not input_file.exists():
        print(f"✗ Error: can't find input file {input_file}")
        print("  Please run command: python milestone2_code/download_sharegpt.py --download")
        return

    # Create output Directory
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input File: {input_file}")
    print(f"Output File: {output_dir}")
    print(f"Input File Size: {input_file.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"Process: {max_conversations} conversations")
    print(f"System prompt: {system_prompt}")
    print()

    # Load tokenizer
    print("Step 1: load tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        print(f"  ✓ Tokenizer loaded: {model_path}")
    except Exception as e:
        print(f"  Failed to load local tokenizer: {e}")
        print("  try using 'gpt2' as fallback (仅用于估算长度)")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # used to deprecate tokenizer warning
    tokenizer.model_max_length = 1_000_000_000
    print()

    # create ShareGPT preprocessor
    preprocessor = ShareGPTPreprocessor(tokenizer, system_prompt=system_prompt)

    # process single-turn data
    print("Step 2: process Single-Turn data...")
    print("-" * 60)
    single_turn_output = output_dir / 'sharegpt_single_turn.jsonl'

    preprocessor.process_dataset(
        input_path=input_file,
        output_path=single_turn_output,
        max_conversations=max_conversations,
        single_turn_only=True
    )

    print()

    # reset stats
    preprocessor.stats = {
        'total_conversations': 0,
        'filtered_conversations': 0,
        'valid_conversations': 0,
        'total_turns': 0,
        'conversations_with_system_prompt': 0,
        'conversations_without_system_prompt': 0,
    }

    # process multi-turn data
    print("Step 3: process Multi-Turn data...")
    print("-" * 60)
    multi_turn_output = output_dir / 'sharegpt_multi_turn.jsonl'

    preprocessor.process_dataset(
        input_path=input_file,
        output_path=multi_turn_output,
        max_conversations=max_conversations,
        single_turn_only=False
    )

    print()
    print("=" * 60)
    print("✓ Finished Pre-processing!")
    print("=" * 60)
    print()

    # Show files generated
    print("Files Generated:")
    if single_turn_output.exists():
        num_lines = sum(1 for _ in open(single_turn_output))
        size = single_turn_output.stat().st_size / 1024
        print(f"  ✓ {single_turn_output}")
        print(f"    - {num_lines} 个trace条目")
        print(f"    - {size:.1f} KB")

    if multi_turn_output.exists():
        num_lines = sum(1 for _ in open(multi_turn_output))
        size = multi_turn_output.stat().st_size / 1024
        print(f"  ✓ {multi_turn_output}")
        print(f"    - {num_lines} 个trace条目")
        print(f"    - {size:.1f} KB")


if __name__ == "__main__":
    main()