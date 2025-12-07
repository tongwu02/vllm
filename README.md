# Project: A study of prefix sharing in LLM serving

## 🚀 Overview

This project explores prefix sharing in vLLM through a CPU-based simulator and a series of cache-behavior studies. We first build a simulator that replays pre-recorded responses while preserving vLLM’s scheduling and KV-cache management logic. Using this simulator, we systematically analyze prefix reuse patterns under single-turn and multi-turn chatbot workloads, and evaluate the impact of cache size, block size, and eviction policies. Finally, we extend the study to heterogeneous workloads and design a hybrid eviction policy that improves cache reuse under mixed scenarios.

The technical report (`report.pdf`) provides full design details and results.

## 📁 Repository Structure

```
├── vllm/                          # Modified vLLM source code
│   ├── engine/
│   │   └── llm_engine.py          # Hook to integrate simulator
│   ├── sim/
│   │   └── simulator.py           # Main simulator implementation (new)
│   └── core/
│       └── scheduler.py           # Minor modifications for simulator mode
├── milestone2_code/               # Scripts for M2 evaluation
│   └── ...
├── milestone3_code/               # Scripts for eviction policy experiments
│   └── ...
│
├── README.md                      # (This file)
└── report.pdf                     # Final project report
```

## 🛠️ Installation

This project uses the same environment setup as the course-provided vLLM instructions.  
In addition, `matplotlib` is required for generating the figures used in Milestone 2 and Milestone 3 experiments.

**Model setup**

This project uses a local copy of `Llama-3.2-1B-Instruct`.

Place the exported model under the directory `exported_models/` so that its path becomes:

```
exported_models/Llama-3.2-1B-Instruct/
```

**Tested environment**

This project was tested under the following environment:
- Python 3.12
- vLLM (local modified version)
- macOS and Windows


## ▶️ Running Milestone1

Follow the steps below to start the vLLM server in simulator mode and verify that it correctly replays responses from the trace.

### 1. Enable simulator mode
Enable simulator mode via:

```bash
cd vllm # project root's vllm/, NOT vllm/vllm/
export VLLM_SIM_TRACE_PATH=trace.jsonl
```

### 2. Start the vLLM server

```bash
python -m vllm.entrypoints.openai.api_server \
  --model exported_models/Llama-3.2-1B-Instruct \
  --tokenizer exported_models/Llama-3.2-1B-Instruct \
  --served-model-name meta-llama/Llama-3.2-1B-Instruct \
  --device cpu \
  --max-model-len 4096 \
  --max-num-seqs 4 \
  --block-size 8 \
  --host localhost \
  --port 8000
```

### 3. Send a request

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.2-1B-Instruct",
    "prompt": "hehe",
    "max_tokens": 64
  }'
```

**Expected behavior**

The response returned by the server should exactly match the text stored in the trace for the prompt "hehe". In our trace, the prompt "hehe" maps to the response "hahaha!", so the simulator should return exactly "hahaha!" as the completion.

Example output:

```json
{"id":"cmpl-b64a4c330c8f4d799a71b2eadacb3ca2","object":"text_completion","created":1764989848,"model":"meta-llama/Llama-3.2-1B-Instruct","choices":[{"index":0,"text":"hahaha!","logprobs":null,"finish_reason":"stop","stop_reason":null,"prompt_logprobs":null}],"usage":{"prompt_tokens":3,"total_tokens":7,"completion_tokens":4,"prompt_tokens_details":null}}%
```

### 4. (Optional) Run Unit Tests

This project includes optional unit tests that verify the correctness of the simulator:
- The simulator replays responses exactly as stored in the trace.
- Prefix sharing produces non-zero cache hit rate, confirming that the scheduler and block manager behave normally.

To run the tests:

```bash
python3 -m pytest tests/standalone_tests/test_simulator_mode.py
```


## 📊 Running Milestone 2

```bash
# 2. download data
python download_sharegpt.py

# 3. preprocess data
python preprocess_sharegpt.py

# 4. run experiment
python compare_multi_vs_single_turn.py

# 5. plot results
python visualize_task2.py 
python visualize_task2_advance.py
```

## 🗂️ Running Milestone 3

