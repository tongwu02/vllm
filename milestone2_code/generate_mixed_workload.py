import json
import random
import uuid
import sys
from pathlib import Path
from transformers import AutoTokenizer

# ================= Configuration =================
MODEL_PATH = str(Path(__file__).parent.parent / "exported_models" / "Llama-3.2-1B-Instruct")
OUTPUT_FILE = Path("traces/mixed_workload_standard.jsonl")

# Dataset paths
DATASET_PATHS = {
    "ShareGPT": Path("traces/sharegpt_multi_turn.jsonl"),
    "CC_Bench": Path("traces/ccbench_multi_turn.jsonl"),
    "AgentBank": Path("traces/agentbank_multi_turn.jsonl")
}

# Total number of requests
TOTAL_REQUESTS = 1000

# Mix ratio (simulating real traffic)
# 60% is daily conversation (ShareGPT, has reuse potential)
# 40% is various one-off tasks (Noise)
RATIOS = {
    "ShareGPT": 0.6,
    "CC_Bench": 0.2,
    "AgentBank": 0.2
}

# Shared System Prompt (Only for VIP use)
SHARED_SYSTEM_PROMPT = (
    "You are a helpful and harmless AI assistant. "
    "Your goal is to provide accurate, concise, and safe responses to user queries."
)


# =======================================

def load_dataset(path):
    data = []
    if not path.exists():
        return []
    with open(path, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except:
                continue
    print(f"   -> Loaded {len(data)} entries from {path.name}")
    return data


def extract_content(entry):
    # Try to extract user input content
    if "messages" in entry:
        for m in entry["messages"]:
            if m["role"] == "user": return m["content"]
    if "prompt" in entry: return entry["prompt"]
    return "Hello"


def main():
    print(f"Generating Standard Mixed Workload...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    except:
        print("❌ Tokenizer not found.")
        return

    pools = {name: load_dataset(path) for name, path in DATASET_PATHS.items()}
    if not pools["ShareGPT"]: return

    mixed_trace = []

    for i in range(TOTAL_REQUESTS):
        # 1. Randomly select task type
        dataset_name = random.choices(list(RATIOS.keys()), weights=list(RATIOS.values()), k=1)[0]
        if not pools[dataset_name]: dataset_name = "ShareGPT"

        sample = random.choice(pools[dataset_name])
        user_text = extract_content(sample)[:2000]  # Moderate truncation to prevent OOM

        entry = {}

        if dataset_name == "ShareGPT":
            # === VIP: Use Shared Prompt ===
            # These requests will share prefixes, Ref Count will increase
            messages = [
                {"role": "system", "content": SHARED_SYSTEM_PROMPT},
                {"role": "user", "content": user_text}
            ]
            full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            entry = {
                "conversation_id": f"vip_{i}",
                "turn_index": 0,
                "prompt": full_prompt,
                "dataset": "ShareGPT"
            }
        else:
            # === Noise: Unique Prefix ===
            # These requests' Ref Count will also increase, but will never be accessed again
            unique_id = str(uuid.uuid4())[:8]
            raw_prompt = f"[{dataset_name}_{unique_id}]\n{user_text}"

            entry = {
                "conversation_id": f"noise_{i}",
                "turn_index": 0,
                "prompt": raw_prompt,
                "dataset": dataset_name
            }

        mixed_trace.append(entry)

    # Save
    with open(OUTPUT_FILE, 'w') as f:
        for item in mixed_trace:
            f.write(json.dumps(item) + "\n")

    print(f"✅ Generated {len(mixed_trace)} requests.")


if __name__ == "__main__":
    main()