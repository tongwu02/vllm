#!/usr/bin/env python3
"""Preprocess AgentBank dataset into simulator-compatible traces."""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional

from datasets import load_dataset

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from milestone2_code.trace_preprocessor import ShareGPTPreprocessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert AgentBank trajectories into single/multi-turn traces."
    )
    parser.add_argument(
        "--configs",
        nargs="*",
        default=["apps", "gsm8k", "strategyqa"],
        help="AgentBank configs (tasks) to include",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to use (default: train)",
    )
    parser.add_argument(
        "--max-per-config",
        type=int,
        default=500,
        help="Max number of conversations per config",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Optional limit on turns per conversation",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "milestone3_code" / "traces",
        help="Directory to write processed traces",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="System prompt mode (None, string, or 'random')",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=ROOT_DIR / "exported_models" / "Llama-3.2-1B-Instruct",
        help="Tokenizer path for formatting prompts",
    )
    parser.add_argument(
        "--single-output",
        type=str,
        default="agentbank_single_turn.jsonl",
        help="Filename for single-turn trace",
    )
    parser.add_argument(
        "--multi-output",
        type=str,
        default="agentbank_multi_turn.jsonl",
        help="Filename for multi-turn trace",
    )
    return parser.parse_args()


def load_tokenizer(model_path: Path):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    return tokenizer


def iter_agentbank_samples(
    config: str, split: str, limit: Optional[int]
) -> Iterable[Dict]:
    dataset = load_dataset(
        "Solaris99/AgentBank",
        config,
        split=split,
        streaming=True,
    )
    count = 0
    for sample in dataset:
        yield sample
        count += 1
        if limit and count >= limit:
            break


def enrich_entry(entry: Dict, sample: Dict, config: str, workload: str) -> Dict:
    conversation_id = entry.get("conversation_id", sample.get("id", ""))
    turn_index = entry.get("turn_index", 0)
    request_id = f"agentbank-{config}-{conversation_id}-{turn_index}"
    meta = {
        "dataset": "AgentBank",
        "config": config,
        "source_id": sample.get("id"),
        "task": sample.get("task"),
        "skill": sample.get("skill"),
    }
    enriched = {
        "request_id": request_id,
        "workload": workload,
        "prompt": entry["prompt"],
        "response": entry["response"],
        "conversation_id": conversation_id,
        "turn_index": turn_index,
        "meta": meta,
    }
    return enriched


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(args.model_path)
    preprocessor = ShareGPTPreprocessor(tokenizer, system_prompt=args.system_prompt)

    single_path = args.output_dir / args.single_output
    multi_path = args.output_dir / args.multi_output

    total_single = total_multi = 0
    stats = {}

    with single_path.open("w", encoding="utf-8") as single_f, multi_path.open(
        "w", encoding="utf-8"
    ) as multi_f:
        for config in args.configs:
            config_counts = {"single": 0, "multi": 0}
            dataset_iter = iter_agentbank_samples(
                config, args.split, args.max_per_config
            )
            for idx, sample in enumerate(dataset_iter):
                raw_conv = {
                    "id": sample.get("id", f"{config}_{idx:06d}"),
                    "conversations": sample.get("conversations", []),
                }
                conv = preprocessor.clean_conversation(raw_conv, conv_index=idx)
                if conv is None or not preprocessor.is_valid_conversation(conv):
                    continue

                single_entries = preprocessor.conversation_to_trace_entries(
                    conv,
                    max_turns=args.max_turns,
                    single_turn_only=True,
                )
                for entry in single_entries:
                    enriched = enrich_entry(
                        entry, sample, config, f"agentbank/{config}"
                    )
                    single_f.write(json.dumps(enriched, ensure_ascii=False) + "\n")
                    config_counts["single"] += 1
                    total_single += 1

                multi_entries = preprocessor.conversation_to_trace_entries(
                    conv,
                    max_turns=args.max_turns,
                    single_turn_only=False,
                )
                for entry in multi_entries:
                    enriched = enrich_entry(
                        entry, sample, config, f"agentbank/{config}"
                    )
                    multi_f.write(json.dumps(enriched, ensure_ascii=False) + "\n")
                    config_counts["multi"] += 1
                    total_multi += 1

            stats[config] = config_counts

    print("AgentBank preprocessing complete.")
    print(f"Single-turn entries: {total_single}")
    print(f"Multi-turn entries:  {total_multi}")
    for config, cnts in stats.items():
        print(f"  - {config}: single={cnts['single']}, multi={cnts['multi']}")
    print(f"Output files:\n  {single_path}\n  {multi_path}")


if __name__ == "__main__":
    main()
