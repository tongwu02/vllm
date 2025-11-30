# Task 2: Cache Block Usage and Reuse Pattern Analysis

## Executive Summary

This report presents a comprehensive analysis of prefix cache performance in vLLM, comparing **single-turn** versus **multi-turn** conversation workloads. The experimental results demonstrate that multi-turn conversations achieve dramatically higher cache hit rates through effective conversation history reuse.

---

## 1. Hit Rate Comparison

### Correct Hit Rate (First Prefill Only)
- **Single-turn**: 0.46%
- **Multi-turn**: 55.77%
- **Improvement**: +55.30 percentage points

### vLLM GPU Hit Rate (All Prefills)
- **Single-turn**: 0.50%
- **Multi-turn**: 57.00%
- **Improvement**: +56.49 percentage points

### Analysis
Multi-turn conversations demonstrate **significantly higher hit rates** due to conversation history reuse. Each turn in a multi-turn conversation builds upon previous turns, creating extensive opportunities for prefix sharing. The conversation-by-conversation processing strategy enables the KV cache from earlier turns to be effectively reused in subsequent turns, resulting in over **100x improvement** in cache efficiency.

The "Correct Hit Rate" metric (measuring only the first prefill operation per request) provides a more accurate assessment of prefix caching effectiveness compared to vLLM's GPU hit rate, which includes chunked prefill operations and can inflate the reported values.

---

## 2. Cache Block Usage Statistics

| Metric | Single-turn | Multi-turn | Ratio |
|--------|-------------|------------|-------|
| Unique cache blocks | 4 | 128 | 32.0x |
| Total block accesses | 4 | 1,605 | 401.3x |
| Repeated accesses (≥2 times) | 0 | 1,477 | ∞ |
| Avg accesses per block | 1.00 | 12.54 | 12.54x |
| Max accesses per block | 1 | 19 | 19.0x |

### Analysis
The stark contrast in cache block usage patterns reveals the fundamental difference between single-turn and multi-turn workloads:

- **Single-turn workload**: Each request uses completely unique cache blocks with zero reuse (avg 1.00 access per block). This indicates no prefix sharing across independent requests.

- **Multi-turn workload**: Shows extensive block reuse with an average of **12.54 accesses per block**. The presence of **1,477 repeated accesses** (compared to 0 in single-turn) provides concrete evidence that conversation history blocks are being effectively cached and reused across turns.

- **Peak reuse**: Some frequently-accessed blocks (likely containing common conversation prefixes such as system prompts or early conversation context) are reused up to **19 times**, demonstrating the compounding benefits of prefix caching in multi-turn scenarios.

---

## 3. Block Access Distribution

### Single-turn Distribution
- **4 blocks** accessed **1 time** each
- **Distribution**: Completely uniform, 100% single-access

### Multi-turn Distribution
The multi-turn workload exhibits a rich distribution of access frequencies:

| Access Count | Number of Blocks |
|--------------|------------------|
| 7 times | 3 blocks |
| 8 times | 4 blocks |
| 9 times | 9 blocks |
| 10 times | 12 blocks |
| 11 times | 14 blocks |
| 12 times | 21 blocks |
| 13 times | 21 blocks |
| 14 times | 19 blocks |
| 15 times | 12 blocks |
| 16 times | 4 blocks |
| 17 times | 1 block |
| 18 times | 7 blocks |
| 19 times | 1 block |

### Analysis
The access distribution patterns reveal distinct caching behaviors:

- **Single-turn**: Exhibits a degenerate distribution where all blocks are accessed exactly once, confirming the absence of any prefix sharing opportunities in independent single-turn requests.

- **Multi-turn**: Demonstrates a **normal-like distribution** centered around 11-14 accesses per block. This pattern is characteristic of conversation-based workloads where:
  - Early conversation context (system prompts, initial user queries) resides in blocks that are reused across all subsequent turns
  - Middle sections of conversations receive moderate reuse as turns build incrementally
  - Turn-specific content appears in blocks with lower access counts

The presence of blocks accessed 19 times (peak frequency) indicates highly-reused conversation prefixes across the 26 test conversations (averaging 2.96 turns per conversation).

---

## 4. Reuse Time Gap Analysis

### Multi-turn Reuse Statistics
- **Total reuse events**: 1,477
- **Mean time gap**: 0.0750 seconds
- **Median time gap**: 0.0545 seconds
- **Min time gap**: 0.0037 seconds
- **Max time gap**: 0.4136 seconds

### Analysis
The temporal reuse patterns provide insight into cache block lifecycle and conversation dynamics:

- **Short mean gap (75ms)**: The small average time gap between consecutive accesses of the same block indicates that cache blocks are being reused rapidly within active conversations. This aligns with the expected pattern where each new turn in a conversation immediately references the shared prefix from previous turns.

- **Median < Mean**: The median (54.5ms) being lower than the mean (75.0ms) suggests a right-skewed distribution, where most reuses happen very quickly (within tens of milliseconds), but occasional longer gaps occur.

- **Minimum gap (3.7ms)**: The extremely short minimum gap indicates that some blocks are accessed in rapid succession, likely representing consecutive engine steps within the same conversation turn or immediately sequential turns.

- **Maximum gap (414ms)**: The longer maximum gaps may represent:
  - Delays between conversation turns as the model generates responses
  - Different conversations accessing similar prefix blocks (e.g., shared system prompts)
  - Scheduler latency in conversation-by-conversation processing

The concentration of gaps in the 40-100ms range (as shown in the histogram) confirms that most cache reuse occurs within the natural rhythm of multi-turn conversation processing.

---

## 5. Key Findings and Implications

### Primary Findings

1. **Dramatic Hit Rate Improvement**: Multi-turn hit rate (55.77%) is **119x higher** than single-turn (0.46%), demonstrating that conversation-by-conversation processing successfully enables prefix cache reuse.

2. **Extensive Block Reuse**: Multi-turn workloads show **1,477 repeated block accesses** compared to **0** in single-turn, providing direct evidence of conversation history reuse.

3. **High Cache Efficiency**: Average block reuse of **12.54x** in multi-turn versus **1.00x** in single-turn shows that each cached block serves multiple requests effectively.

4. **Rapid Temporal Reuse**: Mean reuse gap of **75ms** demonstrates that cached blocks are accessed frequently and quickly within active conversations.

### Implications for LLM Serving Systems

1. **Conversation-aware Scheduling**: The results validate that processing conversations sequentially (conversation-by-conversation) rather than interleaving requests enables effective prefix cache reuse. Systems should prioritize keeping conversation contexts alive in cache.

2. **Cache Design**: The wide distribution of block access frequencies (7-19 times) suggests that cache replacement policies should consider conversation-level context, not just recency. Early conversation blocks have high reuse value throughout the conversation lifetime.

3. **Memory Efficiency**: With 55.77% of blocks being cache hits in multi-turn scenarios, systems can potentially reduce KV cache memory requirements by ~56% through effective prefix caching in conversation workloads.

4. **Latency Optimization**: The short reuse gaps (median 54.5ms) indicate that prefix cache hits occur on the critical path of conversation turn processing. Optimizing cache hit latency can directly improve user-perceived latency in chatbot applications.

### Validation of Milestone 2 Implementation

These results **conclusively validate** the Milestone 2 implementation objectives:

✅ **Task 1**: Successfully demonstrated that multi-turn hit rate is substantially higher than single-turn hit rate (55.77% vs 0.46%)

✅ **Task 2**: Comprehensive measurement and visualization of:
- Per-request prefix sharing ratio (55.77% hit rate)
- Hits per cache block (average 12.54, max 19)
- Cache block reuse time gaps (mean 75ms, range 3.7-414ms)

The conversation-by-conversation processing strategy enables subsequent turns to effectively reuse cached blocks from previous turns, achieving the core goal of prefix caching optimization in multi-turn conversation scenarios.

---

## Experimental Methodology

- **Workload**: ShareGPT dataset with 26 conversations (2-6 turns each, 77 total turns)
- **Model**: Llama-3.2-1B-Instruct
- **Processing**: Conversation-by-conversation sequential processing for multi-turn; standard processing for single-turn
- **Measurement**: Custom hit rate tracker recording first prefill only (avoiding chunked prefill inflation)
- **Cache Block Tracking**: Full lifecycle tracking of block creation, access times, and reuse patterns

---

## Conclusion

This analysis provides strong empirical evidence that **multi-turn conversation workloads benefit dramatically from prefix caching**, achieving over 100x improvement in cache hit rates compared to single-turn workloads. The conversation-by-conversation processing strategy successfully enables cache reuse across conversation turns, with each cached block serving an average of 12.54 requests.

The results validate the importance of conversation-aware scheduling and cache management in modern LLM serving systems, particularly for chatbot and conversational AI applications where multi-turn interactions are the primary workload pattern.
