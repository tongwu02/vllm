#!/usr/bin/env python3
"""
Download and prepare the ShareGPT dataset

Method 1: Download from HuggingFace
Method 2: Load from local file
"""
import json
import logging
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_from_huggingface(
    output_path: Path = Path("ShareGPT_V3_unfiltered_cleaned_split.json"),
    max_samples: Optional[int] = None,
):
    """
    Download ShareGPT dataset from HuggingFace

    Args:
        output_path: Output file path
        max_samples: Maximum number of samples to download (None = all)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("Please install datasets: pip install datasets")
        return False

    logger.info("Downloading ShareGPT dataset from HuggingFace...")

    # Try multiple potential dataset sources
    dataset_sources = [
        "anon8231489123/ShareGPT_Vicuna_unfiltered",
        "shibing624/sharegpt_gpt4",
        "RyokoAI/ShareGPT52K",
    ]

    # cdf, distribution of block hit, sharing rate over time
    dataset = None
    used_source = None

    for source in dataset_sources:
        try:
            logger.info(f"Trying dataset: {source}")
            dataset = load_dataset(source, split="train")
            used_source = source
            logger.info(f"✓ Successfully loaded from: {source}")
            break
        except Exception as e:
            logger.warning(f"Failed to load {source}: {e}")
            continue

    if dataset is None:
        logger.error("All dataset sources failed. Please download manually.")
        logger.info("\nManual download instructions:")
        logger.info("1. Visit: https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered")
        logger.info("2. Download the JSON file")
        logger.info("3. Save as: ShareGPT_V3_unfiltered_cleaned_split.json")
        return False

    try:

        logger.info(f"✓ Downloaded {len(dataset)} conversations")

        # If max_samples is specified, take only a portion
        if max_samples and max_samples < len(dataset):
            dataset = dataset.select(range(max_samples))
            logger.info(f"  Limited to {max_samples} conversations")

        # Save in JSONL format
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for item in dataset:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        logger.info(f"✓ Saved to: {output_path}")
        logger.info(f"  File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

        return True

    except Exception as e:
        logger.error(f"Failed to download: {e}")
        logger.info("\nTroubleshooting:")
        logger.info("1. Check your internet connection")
        logger.info("2. Make sure you have access to HuggingFace datasets")
        logger.info("3. Try: pip install --upgrade datasets")
        return False


def verify_dataset(file_path: Path) -> bool:
    """Verify if the dataset format is correct"""
    logger.info(f"\nVerifying dataset: {file_path}")

    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return False

    try:
        # Read the first few lines to check format
        sample_size = 5
        samples = []

        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= sample_size:
                    break
                samples.append(json.loads(line))

        logger.info(f"✓ File format is valid JSONL")
        logger.info(f"  Sample size checked: {len(samples)}")

        # Check data structure
        if samples:
            first = samples[0]
            logger.info(f"\n  Sample conversation structure:")
            logger.info(f"    Keys: {list(first.keys())}")

            if 'conversations' in first:
                logger.info(f"    First conversation has {len(first['conversations'])} turns")
                if first['conversations']:
                    turn = first['conversations'][0]
                    logger.info(f"    Turn keys: {list(turn.keys())}")
            elif 'messages' in first:
                logger.info(f"    First conversation has {len(first['messages'])} messages")

        return True

    except json.JSONDecodeError as e:
        logger.error(f"✗ Invalid JSON format: {e}")
        return False
    except Exception as e:
        logger.error(f"✗ Verification failed: {e}")
        return False


def show_sample_conversations(file_path: Path, num_samples: int = 3):
    """Show a few sample conversations"""
    logger.info(f"\n{'='*60}")
    logger.info("Sample Conversations")
    logger.info('='*60)

    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break

            conv = json.loads(line)
            conv_id = conv.get('id', f'conversation_{i}')

            print(f"\n[Conversation {i+1}: {conv_id}]")

            messages = conv.get('conversations', conv.get('messages', []))
            for j, msg in enumerate(messages[:6]):  # Show only the first 6 turns
                role = msg.get('from', msg.get('role', 'unknown'))
                content = msg.get('value', msg.get('content', ''))
                content_preview = content[:100] + '...' if len(content) > 100 else content
                print(f"  {j+1}. {role}: {content_preview}")

            if len(messages) > 6:
                print(f"  ... (total {len(messages)} turns)")


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description='Download and prepare ShareGPT dataset')
    parser.add_argument(
        '--download',
        action='store_true',
        help='Download from HuggingFace'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='ShareGPT_V3_unfiltered_cleaned_split.json',
        help='Output file path'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum number of samples to download (default: all)'
    )
    parser.add_argument(
        '--verify',
        type=str,
        help='Verify existing dataset file'
    )
    parser.add_argument(
        '--show-samples',
        type=str,
        help='Show sample conversations from file'
    )

    args = parser.parse_args()

    if args.download:
        logger.info("=" * 60)
        logger.info("Downloading ShareGPT Dataset")
        logger.info("=" * 60)

        output_path = Path(args.output)
        success = download_from_huggingface(output_path, args.max_samples)

        if success:
            verify_dataset(output_path)
            show_sample_conversations(output_path)

    elif args.verify:
        verify_dataset(Path(args.verify))
        show_sample_conversations(Path(args.verify))

    elif args.show_samples:
        show_sample_conversations(Path(args.show_samples), num_samples=5)

    else:
        print("""
ShareGPT Dataset Preparation Tool

Usage:
    1. Download from HuggingFace:
       python milestone2_code/download_sharegpt.py --download --max-samples 100

    2. Verify existing file:
       python milestone2_code/download_sharegpt.py --verify ShareGPT_V3_unfiltered_cleaned_split.json

    3. Show samples:
       python milestone2_code/download_sharegpt.py --show-samples ShareGPT_V3_unfiltered_cleaned_split.json

Options:
    --download              Download from HuggingFace
    --output FILE          Output file path (default: ShareGPT_V3_unfiltered_cleaned_split.json)
    --max-samples N        Download only N samples
    --verify FILE          Verify file format
    --show-samples FILE    Show sample conversations
        """)


if __name__ == "__main__":
    main()