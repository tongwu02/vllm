# 📚 VLLM Caching Experiments: Command Line Guide

This guide provides the exact commands needed to execute the VLLM prefix caching experiments across Milestone 2 and Milestone 3.

**Execution Environment:** All commands are run from the **root VLLM directory**, with the exception of the data preprocessing and experiment commands, which are run from the `experiment_code/` subdirectory.

## 📌 I. Prerequisites and Data Preprocessing (One-Time Setup)

These steps download, clean, and format the external datasets (ShareGPT, CC, AgentBank) into the trace files required for the VLLM simulator.

| Step | Command | Description |
| :--- | :--- | :--- |
| **1. Change Directory** | `cd experiment_code` | Move into the directory containing data scripts. |
| **2. Download ShareGPT** | `python download_sharegpt.py --download` | Downloads the raw ShareGPT conversational dataset. |
| **3. Preprocess ShareGPT** | `python preprocess_sharegpt.py` | Converts raw ShareGPT JSON into VLLM trace format. |
| **4. Preprocess CC** | `python preprocess_cc.py` | Generates a trace file based on the Common Crawl corpus (often used for short, single-turn prompts). |
| **5. Preprocess AgentBank** | `python preprocess_agentbank.py` | Generates a trace file for agent-based, complex workloads. |

## 🧪 II. Milestone 2: Baseline Workloads

This milestone validates the simulation environment and establishes baseline performance metrics using the default settings and workloads.

### Task 2: Multi-Turn vs. Single-Turn Comparison

* **Objective:** Compare the hit rate achieved when a conversation is processed as multiple turns versus when it is aggregated into a single prompt, testing prefix cache efficiency across different request formats.
* **Location:** `experiment_code/`

| Command | Description |
| :--- | :--- |
| `python compare_multi_vs_single_turn.py` | Executes the simulation and prints a comparison of hit rates for the two workload types. |

### Task 3: ShareGPT Workload Runner

* **Objective:** Run a standard, multi-turn conversational workload (ShareGPT) using the default eviction policy (typically LRU) to establish a performance baseline for complex inputs.
* **Location:** `experiment_code/`

| Command | Description |
| :--- | :--- |
| `python sharegpt_workload_runner.py` | Executes the VLLM trace runner on the preprocessed ShareGPT data. |

## 🔬 III. Milestone 3: Advanced Eviction Policies

This milestone is dedicated to testing the performance of different cache eviction policies, culminating in the evaluation of the `PROTECTED_LRU` policy.

### Task 1: Policy Comparison on Standard Datasets

* **Objective:** Test a range of policies (FIFO, LFU, LRU, PROTECTED\_LRU) across the standard, single-type datasets to understand their performance characteristics in isolation.
* **Location:** `experiment_code/`

| Command | Description |
| :--- | :--- |
| `python multiple_workload_runner.py` | Executes the trace runner for all policies against all specified datasets (e.g., ShareGPT, CC, AgentBank). |

### Task 2: Mixed Workload (The "Double Tap" Test)

* **Objective:** Run the critical test designed to expose the difference between LRU and PROTECTED\_LRU. The workload involves highly reused "VIP" prefixes and a continuous "Flood" of noisy, low-value prefixes, which should force LRU to evict the VIPs.
* **Location:** `experiment_code/`

| Step | Command | Description |
| :--- | :--- | :--- |
| **1. Generate Mixed Workload** | `python generate_mixed_workload.py` | Creates the synthetic trace file (`mixed_workload_standard.jsonl` or similar) containing the VIP double-tap patterns and noise flood requests. |
| **2. Run Mixed Workload** | `python mixed_workload_runner.py` | Executes the trace runner for all policies on the generated mixed trace file. **Expected Result:** A substantial hit rate increase for `PROTECTED_LRU` over `LRU`. |