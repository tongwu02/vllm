# CC-Bench Trace Preprocessing Guide

This document explains how to convert the CC-Bench trajectories dataset into simulator-ready traces using `preprocess_ccbench.py`.

## Basic Usage

```bash
python milestone3_code/preprocess_ccbench.py --max-conversations 100
```

This command:
1. Streams up to 100 CC-Bench samples from the train split.
2. Converts each trajectory into ShareGPT-style conversations.
3. Produces:
   - `milestone3_code/traces/ccbench_single_turn.jsonl`
   - `milestone3_code/traces/ccbench_multi_turn.jsonl`

Each JSONL line contains `{prompt, response, request_id, workload, meta, …}` ready for the simulator.

## Key Arguments

| Flag | Description |
| --- | --- |
| `--max-conversations N` | Upper bound on raw samples to read. The final number of trace entries can be smaller if a sample is invalid or filtered. |
| `--split SPLIT` | HF dataset split (`train`, `validation`, `test`). |
| `--max-turns K` | Limit the number of turns per conversation before generating trace entries. Useful when you want shorter multi-turn traces. |
| `--system-prompt MODE` | Passed to `ShareGPTPreprocessor` (`None`, a literal string, or `random`). |
| `--model-path PATH` | Tokenizer/chat-template path. Defaults to `exported_models/Llama-3.2-1B-Instruct`. |
| `--single-output FILE` / `--multi-output FILE` | Override default filenames under `milestone3_code/traces`. |
| *(default)* prompt dedup | The script now deduplicates prompts by default—only the first conversation for each unique user prompt is kept. If you truly need every model’s response (duplicates included), comment out the `seen_prompts` block in `preprocess_ccbench.py` or add your own switch. |

## Typical Workflows

### 1. Default (Deduplicated) Trace

Useful when one prompt can appear dozens of times (different models). The default run already keeps only the first encounter:

```bash
python milestone3_code/preprocess_ccbench.py --max-conversations 500
```

The resulting single-turn file contains unique prompts, each tagged with the first model seen in CC-Bench.

### 2. Keep Every Model Variant

If you need to compare all model outputs on the same prompt, disable the dedup block manually:

1. Open `preprocess_ccbench.py`.
2. Comment out (or remove) the `seen_prompts` check in the main loop.
3. Run the script (optionally from a fresh branch so you don’t lose the default behavior).

This produces traces where each prompt repeats for every CC-Bench model.

### 3. Limit Conversation Length

```bash
python milestone3_code/preprocess_ccbench.py \
  --max-conversations 200 \
  --max-turns 3
```

This keeps only the first three turns for each conversation when generating multi-turn traces, which helps reduce long histories.

## Output Structure

Each JSON object contains:

```json
{
  "request_id": "ccbench-<conversation_id>-<turn>",
  "workload": "ccbench/<task_category>",
  "prompt": "<chat template formatted prompt>",
  "response": "assistant answer",
  "conversation_id": "<original conversation id>",
  "turn_index": 0,
  "meta": {
    "dataset": "CC-Bench",
    "task_id": ...,
    "task_category": ...,
    "model_name": ...,
    "difficulty": ...,
    "source_id": ...
  }
}
```

This format matches the simulator’s expectations (same schema used for ShareGPT/AgentBank traces).

## Tips

- **HF streaming warnings**: The `datasets` library may print `Operation not permitted` lines due to sandboxed `ps` calls on macOS; they are harmless.
- **Performance**: Deduplication requires holding a set of seen prompts. For large `--max-conversations`, this increases memory usage slightly but is typically negligible.
- **Re-running**: The script overwrites existing JSONL files. Copy or rename traces if you need to keep multiple versions.

## Troubleshooting

- **Zero entries**: Most likely every sample was filtered out (e.g., never reaches a user message). Increase `--max-conversations` or temporarily disable `--dedup-prompts`.
- **Tokenizer not found**: Ensure the model checkpoint used for the tokenizer exists at `--model-path`. Place the exported model under `exported_models/` or point the flag elsewhere.
- **HF auth issues**: If the dataset requires authentication, set `HF_TOKEN` in your environment. The default CC-Bench split is public, though.

With these instructions, you can quickly regenerate CC-Bench single/multi-turn traces tailored to your simulator experiments.
