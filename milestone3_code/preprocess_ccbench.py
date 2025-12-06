#!/usr/bin/env python3
"""
Preprocess CC-Bench trajectories into simulator-ready traces.

Workflow:
1. Stream the dataset "zai-org/CC-Bench-trajectories" via HF datasets.
2. Convert each sample's trajectory into ShareGPT-style conversations:
   [{"from": "human"/"gpt"/"system", "value": "..."}]
3. Reuse ShareGPTPreprocessor to clean the conversations and slice them
   into single-turn / multi-turn trace entries.
4. Emit JSONL files: ccbench_single_turn.jsonl / ccbench_multi_turn.jsonl
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Any, List, Set

from datasets import load_dataset

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from milestone2_code.trace_preprocessor import ShareGPTPreprocessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert CC-Bench trajectories into simulator traces."
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to use (e.g., train/validation/test; default: train)",
    )
    parser.add_argument(
        "--max-conversations",
        type=int,
        default=500,
        help="Maximum CC-Bench trajectories to process.",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Optional cap on number of turns kept per conversation.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=ROOT_DIR / "exported_models" / "Llama-3.2-1B-Instruct",
        help="Tokenizer / chat template path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "milestone3_code" / "traces",
        help="Directory to store output traces.",
    )
    parser.add_argument(
        "--single-output",
        type=str,
        default="ccbench_single_turn.jsonl",
        help="Filename for single-turn trace.",
    )
    parser.add_argument(
        "--multi-output",
        type=str,
        default="ccbench_multi_turn.jsonl",
        help="Filename for multi-turn trace.",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="Optional system prompt handling (None / literal string / 'random').",
    )
    parser.add_argument(
        "--dedup-prompts",
        action="store_true",
        default=True,
        help="If set, only keep the first conversation for each unique user prompt.",
    )
    return parser.parse_args()


# Normalize role names that look like they come from the user
USER_LIKE_ROLES = {
    "user",
    "human",
    "requester",
    "task",
    "instruction",
    "input",
    "goal",
    "description",
}


def normalize_role(raw_role: str) -> str:
    """
    Map raw role labels to ShareGPT style: human / gpt / system.

    - user/human/requester/... -> human
    - system -> system
    - everything else (assistant/agent/tool/...) -> gpt
    """
    if not raw_role:
        return "gpt"

    role = raw_role.lower()

    if role in USER_LIKE_ROLES:
        return "human"
    if role in {"system"}:
        return "system"

    # Covers assistant / assistant1 / assistant_1 / model / agent / tool etc.
    return "gpt"


def parse_turns(raw_turns: Any) -> Optional[List[Any]]:
    """
    Trajectory fields may be JSON strings or already lists.
    Normalize everything to list[dict|str].
    """
    if raw_turns is None:
        return None

    if isinstance(raw_turns, str):
        stripped = raw_turns.strip()
        if not stripped:
            return None
        try:
            decoded = json.loads(stripped)
        except json.JSONDecodeError:
            return None
        if not isinstance(decoded, list):
            return None
        return decoded

    if isinstance(raw_turns, list):
        return raw_turns

    # Other encodings are not supported for now
    return None


def extract_text_from_content(content_field: Any) -> str:
    """
    Handle the different content encodings we see in CC-Bench:
    - string -> return as-is
    - list[dict|str] -> concatenate text/tool_result payloads
    - everything else -> empty string
    """
    if isinstance(content_field, str):
        return content_field.strip()

    if isinstance(content_field, list):
        pieces: List[str] = []
        for item in content_field:
            if isinstance(item, str):
                pieces.append(item)
            elif isinstance(item, dict):
                # Typical structure: {"type": "text", "text": "..."}
                if item.get("type") == "text":
                    pieces.append(item.get("text", ""))
                # We may also see tool_result items
                elif item.get("type") == "tool_result":
                    tool_content = item.get("content")
                    if isinstance(tool_content, str):
                        pieces.append(tool_content)
                    elif isinstance(tool_content, list):
                        for s in tool_content:
                            if isinstance(s, str):
                                pieces.append(s)
        return "\n".join(pieces).strip()

    # Drop everything else
    return ""


def sample_to_conversation(sample: Dict[str, Any], index: int) -> Optional[Dict]:
    """
    Convert one CC-Bench sample into the ShareGPTPreprocessor format:
    {
      "id": <str>,
      "conversations": [
        {"from": "human"/"gpt"/"system", "value": "..."},
        ...
      ],
      "meta": {...}
    }
    """
    conv_id = (
        sample.get("id")
        or sample.get("trajectory_uuid")
        or sample.get("trajectory_id")
        or sample.get("task_id")
        or f"ccbench_{index:06d}"
    )

    # Trajectory field names differ: "trajectory" / "trajectories" / "messages"
    raw_turns_field = (
        sample.get("trajectory")
        or sample.get("trajectories")
        or sample.get("messages")
    )
    raw_turns = parse_turns(raw_turns_field)
    if not raw_turns:
        return None

    conversations: List[Dict[str, str]] = []

    for turn in raw_turns:
        if isinstance(turn, dict) and turn.get("type") == "summary":
            # Skip the summary metadata that CC-Bench prepends
            continue
        # Treat bare strings as assistant outputs
        if isinstance(turn, str):
            role = "gpt"
            content = turn.strip()
        elif isinstance(turn, dict):
            # Some samples wrap the payload inside "message"
            message = turn.get("message") or turn

            raw_role = (
                message.get("role")
                or turn.get("userType")
                or turn.get("role")
                or turn.get("speaker")
                or turn.get("name")
                or ""
            )
            role = normalize_role(raw_role)

            content_field = (
                message.get("content")
                or turn.get("content")
                or turn.get("text")
                or turn.get("response")
            )
            content = extract_text_from_content(content_field)
        else:
            # Unknown shape, skip
            continue

        if not content:
            continue

        # Merge consecutive turns with the same role so the conversation is cleaner
        if conversations and conversations[-1]["from"] == role:
            conversations[-1]["value"] += "\n" + content
        else:
            conversations.append({"from": role, "value": content})

    if not conversations:
        return None

    # Only keep conversations that contain at least one user turn and start from it
    first_human_idx = next(
        (i for i, msg in enumerate(conversations) if msg["from"] == "human"),
        None,
    )
    if first_human_idx is None:
        return None

    conversations = conversations[first_human_idx:]

    meta = {
        "dataset": "CC-Bench",
        "task_id": sample.get("task_id"),
        "task_category": sample.get("task_category"),
        "model_name": sample.get("model_name"),
        "difficulty": sample.get("difficulty"),
        "source_id": sample.get("trajectory_id") or sample.get("id"),
    }

    return {
        "id": conv_id,
        "conversations": conversations,
        "meta": meta,
    }


def iter_ccbench_samples(
    split: str,
    limit: Optional[int],
) -> Iterable[Dict[str, Any]]:
    """Stream CC-Bench samples while honoring the max_conversations limit."""
    dataset = load_dataset(
        "zai-org/CC-Bench-trajectories",
        split=split,
        streaming=True,
    )
    count = 0
    for sample in dataset:
        yield sample
        count += 1
        if limit and count >= limit:
            break


def enrich_entry(entry: Dict, sample: Dict, workload: str) -> Dict:
    """Attach simulator-specific metadata (request_id/workload/etc.) to entries."""
    conversation_id = (
        entry.get("conversation_id")
        or sample.get("id")
        or sample.get("trajectory_uuid")
        or sample.get("trajectory_id")
        or sample.get("task_id")
    )

    turn_index = entry.get("turn_index", 0)
    request_id = entry.get("request_id") or f"ccbench-{conversation_id}-{turn_index}"

    # Prefer entry.meta > sample.meta > synthesized defaults
    meta = (
        entry.get("meta")
        or sample.get("meta")
        or {
            "dataset": "CC-Bench",
            "task_id": sample.get("task_id"),
            "task_category": sample.get("task_category"),
            "model_name": sample.get("model_name"),
            "difficulty": sample.get("difficulty"),
            "source_id": sample.get("trajectory_id") or sample.get("id"),
        }
    )

    return {
        "request_id": request_id,
        "workload": workload,
        "prompt": entry["prompt"],
        "response": entry["response"],
        "conversation_id": conversation_id,
        "turn_index": turn_index,
        "meta": meta,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_path),
        local_files_only=True,
    )
    preprocessor = ShareGPTPreprocessor(
        tokenizer,
        system_prompt=args.system_prompt,
    )

    single_path = args.output_dir / args.single_output
    multi_path = args.output_dir / args.multi_output

    total_single = 0
    total_multi = 0
    seen_prompts: Set[str] = set()

    # Name workload as ccbench/<task_category>; JSON file names cover single/multi
    with single_path.open("w", encoding="utf-8") as single_f, \
            multi_path.open("w", encoding="utf-8") as multi_f:

        for idx, sample in enumerate(
            iter_ccbench_samples(args.split, args.max_conversations)
        ):
            raw_conv = sample_to_conversation(sample, idx)
            if raw_conv is None:
                continue

            conv = preprocessor.clean_conversation(raw_conv, conv_index=idx)
            if conv is None or not preprocessor.is_valid_conversation(conv):
                continue

            if args.dedup_prompts and conv.messages:
                prompt_key = conv.messages[0].content.strip()
                if not prompt_key:
                    continue
                if prompt_key in seen_prompts:
                    continue
                seen_prompts.add(prompt_key)

            # Use task_category to distinguish workloads
            task_cat = sample.get("task_category") or "unknown"
            workload_tag = f"ccbench/{task_cat}"

            # single-turn entries
            single_entries = preprocessor.conversation_to_trace_entries(
                conv,
                max_turns=args.max_turns,
                single_turn_only=True,
            )
            for entry in single_entries:
                enriched = enrich_entry(entry, sample, workload=workload_tag)
                single_f.write(json.dumps(enriched, ensure_ascii=False) + "\n")
                total_single += 1

            # multi-turn entries
            multi_entries = preprocessor.conversation_to_trace_entries(
                conv,
                max_turns=args.max_turns,
                single_turn_only=False,
            )
            for entry in multi_entries:
                enriched = enrich_entry(entry, sample, workload=workload_tag)
                multi_f.write(json.dumps(enriched, ensure_ascii=False) + "\n")
                total_multi += 1

    print("CC-Bench preprocessing complete.")
    print(f"Single-turn entries: {total_single}")
    print(f"Multi-turn entries:  {total_multi}")
    print(f"Output files:\n  {single_path}\n  {multi_path}")


if __name__ == "__main__":
    main()
