#!/bin/bash
# Milestone 2 Quick Test Script
# 快速测试 Milestone 2 的所有功能

set -e  # Exit on error

echo "========================================"
echo "Milestone 2 Task 1 & 2 Quick Test"
echo "========================================"
echo ""

# Check if ShareGPT data exists
if [ ! -f "vllm/ShareGPTData.jsonl" ]; then
    echo "❌ Error: ShareGPT data not found at vllm/ShareGPTData.jsonl"
    echo "Please download the data first."
    exit 1
fi

echo "✅ Found ShareGPT data file"
echo ""

# Run small test
echo "Running small test with 100 conversations..."
echo ""

python3 vllm/sim/run_milestone2_task2.py \
    --data-path vllm/ShareGPTData.jsonl \
    --max-conversations 100 \
    --output-dir milestone2_test_results \
    --block-size 16 \
    --arrival-rate 2.0 \
    --skip-visualization

echo ""
echo "========================================"
echo "Test completed successfully! ✅"
echo "========================================"
echo ""
echo "Generated files:"
ls -lh milestone2_test_results/
echo ""
echo "Key Results:"
echo ""

# Extract key metrics from JSON
echo "Single-turn results:"
python3 -c "
import json
with open('milestone2_test_results/single_turn_stats.json') as f:
    stats = json.load(f)
print(f\"  Total Requests: {stats['total_requests']}\")
print(f\"  Block Reuse Rate: {stats['total_blocks_reused']/(stats['total_blocks_allocated']+stats['total_blocks_reused'])*100:.2f}%\")
print(f\"  Mean Sharing Fraction: {stats['sharing_fraction']['mean']*100:.2f}%\")
print(f\"  Median Block Hits: {stats['block_hits']['median']}\")
"

echo ""
echo "Multi-turn results:"
python3 -c "
import json
with open('milestone2_test_results/multi_turn_stats.json') as f:
    stats = json.load(f)
print(f\"  Total Requests: {stats['total_requests']}\")
print(f\"  Block Reuse Rate: {stats['total_blocks_reused']/(stats['total_blocks_allocated']+stats['total_blocks_reused'])*100:.2f}%\")
print(f\"  Mean Sharing Fraction: {stats['sharing_fraction']['mean']*100:.2f}%\")
print(f\"  Median Block Hits: {stats['block_hits']['median']}\")
"

echo ""
echo "========================================"
echo "✅ All tests passed!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Run full experiment: python3 vllm/sim/run_milestone2_task2.py --max-conversations 1000"
echo "2. Check MILESTONE2_GUIDE.md for detailed instructions"
echo "3. Analyze results in milestone2_test_results/"
echo ""
