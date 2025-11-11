from datasets import load_dataset
import json
import os

# Configuration
dataset_name = "anon8231489123/ShareGPT_Vicuna_unfiltered"
data_file_name = "ShareGPT_V3_unfiltered_cleaned_split.json"
# Note the .jsonl extension
output_filename = "ShareGPTData.jsonl" 

print(f"Loading dataset: {dataset_name} (File: {data_file_name})...")

try:
    dataset = load_dataset(dataset_name, data_files=data_file_name)
    
    if 'train' in dataset:
        data_split = dataset['train']
        
        print("Dataset loaded. Writing as JSON Lines (.jsonl) format...")
        
        count = 0
        with open(output_filename, 'w', encoding='utf-8') as f:
            # Iterate and write one JSON object (as a string) per line
            for item in data_split:
                # Convert the dictionary for this one item to a JSON string
                json_string = json.dumps(item, ensure_ascii=False)
                # Write that string as a single line
                f.write(json_string + '\n')
                count += 1
            
        file_size = os.path.getsize(output_filename) / (1024*1024) # MB
        
        print("\n--- Success ---")
        print(f"Dataset successfully saved to: {output_filename}")
        print(f"File size: {file_size:.2f} MB")
        print(f"Total conversations (lines): {count}")
        print("This file is in the efficient JSON Lines format.")

    else:
        print(f"Error: 'train' split not found. Available splits: {list(dataset.keys())}")

except Exception as e:
    print(f"\n--- Error ---")
    print(f"Details: {e}")