# Milestone 3

## AgentBank Preprocessing

This directory hosts tooling for Milestone 3 experiments in which we compare cache eviction
policies across heterogeneous workloads. The `preprocess_agentbank.py` script converts the
`Solaris99/AgentBank` dataset into simulator-friendly traces that follow the same JSONL format
used in Milestone 2.

### Usage

```bash
# Example: process three AgentBank configs with default settings
python milestone3_code/preprocess_agentbank.py
```

Key arguments:

- `--configs`: Which AgentBank configs/tasks to include (default: `apps gsm8k strategyqa`).  
- `--max-per-config`: Limit the number of conversations sampled per config.  
- `--max-turns`: Optional cap on turns kept per conversation.  
- `--system-prompt`: Same behavior as Milestone 2 (`None`, explicit prompt string, or `random`).  
- `--output-dir`: Directory for emitting JSONL traces (defaults to `milestone3_code/traces`).  
- `--single-output` / `--multi-output`: Filenames for single-turn and multi-turn traces.

Outputs:

- `agentbank_single_turn.jsonl`: Each entry contains a prompt/response pair derived from the first turn.  
- `agentbank_multi_turn.jsonl`: Conversation-by-conversation traces where each turn becomes a request.

Each trace entry includes `request_id`, `workload` (e.g., `agentbank/apps`), metadata such as task
and skill dimension, and is ready to be consumed by `ClientSimulator` or other Milestone 2 tooling.
