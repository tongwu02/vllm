import json
import random
import uuid
from pathlib import Path
from transformers import AutoTokenizer

# ================= 配置 =================
MODEL_PATH = str(Path(__file__).parent.parent / "exported_models" / "Llama-3.2-1B-Instruct")
OUTPUT_FILE = Path("traces/mixed_flood.jsonl")

# 800 tokens Shared
SHARED_SYSTEM_PROMPT = "You are a persistent AI context. " * 150 

def main():
    print(f"Generating Flood Workload...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    mixed_trace = []
    
    # 生成 50 个 Cluster
    # 每个 Cluster: 1 VIP + 20 Noises (足以把 Shared Prompt 挤出 LRU 列表)
    for i in range(50):
        
        # 1. VIP (Refresh Shared Prompt)
        vip_prompt = tokenizer.apply_chat_template([
            {"role": "system", "content": SHARED_SYSTEM_PROMPT},
            {"role": "user", "content": f"Query_{i}"}
        ], tokenize=False, add_generation_prompt=True)

        mixed_trace.append({
            "conversation_id": f"vip_{i}",
            "turn_index": 0,
            "prompt": vip_prompt,
            "response": ""
        })
        
        # 2. Flood of Noises (20 个)
        # 每个 Noise 约 50 tokens。20 * 50 = 1000 tokens。
        # 加上 Shared (800)，总量 1800 > Capacity (1024)。必爆！
        for j in range(20):
            noise_content = f"[NOISE_{uuid.uuid4()}] " + "flood " * 10
            mixed_trace.append({
                "conversation_id": f"noise_{i}_{j}",
                "turn_index": 0,
                "prompt": noise_content,
                "response": ""
            })

    with open(OUTPUT_FILE, 'w') as f:
        for item in mixed_trace:
            f.write(json.dumps(item) + "\n")
            
    print(f"✅ Generated Flood Workload: {len(mixed_trace)} requests")

if __name__ == "__main__":
    main()