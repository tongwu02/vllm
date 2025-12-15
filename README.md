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
├── experiment_code/               # Scripts for M2 & M3
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


## ▶️ Running Milestone 1

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


## 📚 Running Milestone 2 & Milestone 3

This guide provides the exact commands needed to execute the VLLM prefix caching experiments across Milestone 2 and Milestone 3.

**Execution Environment:** All commands are run from the **root VLLM directory**, with the exception of the data preprocessing and experiment commands, which are run from the `experiment_code/` subdirectory.

### 📌 I. Prerequisites and Data Preprocessing (One-Time Setup)

These steps download, clean, and format the external datasets (ShareGPT, CC, AgentBank) into the trace files required for the VLLM simulator.

| Step | Command | Description |
| :--- | :--- | :--- |
| **1. Change Directory** | `cd experiment_code` | Move into the directory containing data scripts. |
| **2. Download ShareGPT** | `python download_sharegpt.py --download` | Downloads the raw ShareGPT conversational dataset. |
| **3. Preprocess ShareGPT** | `python preprocess_sharegpt.py` | Converts raw ShareGPT JSON into VLLM trace format. |
| **4. Preprocess CC** | `python preprocess_cc.py` | Generates a trace file based on the Common Crawl corpus (often used for short, single-turn prompts). |
| **5. Preprocess AgentBank** | `python preprocess_agentbank.py` | Generates a trace file for agent-based, complex workloads. |

### 🧪 II. Milestone 2: Baseline Workloads

This milestone validates the simulation environment and establishes baseline performance metrics using the default settings and workloads.

#### Task 2: Multi-Turn vs. Single-Turn Comparison

* **Objective:** Compare the hit rate achieved when a conversation is processed as multiple turns versus when it is aggregated into a single prompt, testing prefix cache efficiency across different request formats.
* **Location:** `experiment_code/`

| Command | Description |
| :--- | :--- |
| `python compare_multi_vs_single_turn.py` | Executes the simulation and prints a comparison of hit rates for the two workload types. |

#### Task 3: ShareGPT Workload Runner

* **Objective:** Run a standard, multi-turn conversational workload (ShareGPT) using the default eviction policy (typically LRU) to establish a performance baseline for complex inputs.
* **Location:** `experiment_code/`

| Command | Description |
| :--- | :--- |
| `python sharegpt_workload_runner.py` | Executes the VLLM trace runner on the preprocessed ShareGPT data. |

### 🔬 III. Milestone 3: Advanced Eviction Policies

This milestone is dedicated to testing the performance of different cache eviction policies, culminating in the evaluation of the `PROTECTED_LRU` policy.

#### Task 1: Policy Comparison on Standard Datasets

* **Objective:** Test a range of policies (FIFO, LFU, LRU, PROTECTED\_LRU) across the standard, single-type datasets to understand their performance characteristics in isolation.
* **Location:** `experiment_code/`

| Command | Description |
| :--- | :--- |
| `python multiple_workload_runner.py` | Executes the trace runner for all policies against all specified datasets (e.g., ShareGPT, CC, AgentBank). |

#### Task 2: Mixed Workload (The "Double Tap" Test)

* **Objective:** Run the critical test designed to expose the difference between LRU and PROTECTED\_LRU. The workload involves highly reused "VIP" prefixes and a continuous "Flood" of noisy, low-value prefixes, which should force LRU to evict the VIPs.
* **Location:** `experiment_code/`

| Step | Command | Description |
| :--- | :--- | :--- |
| **1. Generate Mixed Workload** | `python generate_mixed_workload.py` | Creates the synthetic trace file (`mixed_workload_standard.jsonl` or similar) containing the VIP double-tap patterns and noise flood requests. |
| **2. Run Mixed Workload** | `python mixed_workload_runner.py` | Executes the trace runner for all policies on the generated mixed trace file. **Expected Result:** A substantial hit rate increase for `PROTECTED_LRU` over `LRU`. |

