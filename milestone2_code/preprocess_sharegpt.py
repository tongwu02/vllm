#!/usr/bin/env python3
"""
预处理ShareGPT数据集

将ShareGPT_V3_unfiltered_cleaned_split.json转换为trace格式:
- sharegpt_single_turn.jsonl (单轮对话)
- sharegpt_multi_turn.jsonl (多轮对话)

用法:
    python preprocess_sharegpt.py --max-conversations 500
"""
import sys
import argparse
from pathlib import Path

# 确保可以import milestone2_code
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoTokenizer

# 直接import,因为已经在milestone2_code目录下
from trace_preprocessor import ShareGPTPreprocessor


def parse_args():
    parser = argparse.ArgumentParser(description="ShareGPT 数据预处理脚本")
    
    parser.add_argument(
        "--max-conversations", 
        type=int, 
        default=500,
        help="限制处理的最大对话数量 (默认: 500)"
    )
    
    parser.add_argument(
        "--input-file", 
        type=str, 
        default="ShareGPT_V3_unfiltered_cleaned_split.json",
        help="原始输入文件路径"
    )

    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="traces",
        help="输出目录路径"
    )

    return parser.parse_args()


def main():
    # 解析命令行参数
    args = parse_args()

    print("=" * 60)
    print("ShareGPT数据预处理")
    print("=" * 60)
    print()

    # 配置参数
    input_file = Path(args.input_file)
    output_dir = Path(args.output_dir)
    model_path = str(Path(__file__).parent.parent / 'exported_models' / 'Llama-3.2-1B-Instruct')
    
    # 获取用户指定的数量
    max_conversations = args.max_conversations

    # System prompt配置
    system_prompt = "random" 

    # 检查输入文件
    if not input_file.exists():
        print(f"✗ 错误: 找不到输入文件 {input_file}")
        print("  请先运行: python milestone2_code/download_sharegpt.py --download")
        return

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"输入文件: {input_file}")
    print(f"输出目录: {output_dir}")
    print(f"文件大小: {input_file.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"处理数量: {max_conversations} 条对话")
    print(f"System prompt: {system_prompt}")
    print()

    # 加载tokenizer
    print("步骤1: 加载tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        print(f"  ✓ Tokenizer loaded: {model_path}")
    except Exception as e:
        print(f"  ⚠ 加载本地 Tokenizer 失败: {e}")
        print("  尝试使用 'gpt2' 作为 fallback (仅用于估算长度)")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # 屏蔽 Tokenizer 的长度警告
    tokenizer.model_max_length = 1_000_000_000
    print()

    # 创建preprocessor
    preprocessor = ShareGPTPreprocessor(tokenizer, system_prompt=system_prompt)

    # 处理single-turn数据
    print("步骤2: 处理Single-Turn数据...")
    print("-" * 60)
    single_turn_output = output_dir / 'sharegpt_single_turn.jsonl'

    preprocessor.process_dataset(
        input_path=input_file,
        output_path=single_turn_output,
        max_conversations=max_conversations,
        single_turn_only=True
    )

    print()

    # 重置统计
    preprocessor.stats = {
        'total_conversations': 0,
        'filtered_conversations': 0,
        'valid_conversations': 0,
        'total_turns': 0,
        'conversations_with_system_prompt': 0,
        'conversations_without_system_prompt': 0,
    }

    # 处理multi-turn数据
    print("步骤3: 处理Multi-Turn数据...")
    print("-" * 60)
    multi_turn_output = output_dir / 'sharegpt_multi_turn.jsonl'

    preprocessor.process_dataset(
        input_path=input_file,
        output_path=multi_turn_output,
        max_conversations=max_conversations,
        single_turn_only=False
    )

    print()
    print("=" * 60)
    print("✓ 预处理完成!")
    print("=" * 60)
    print()

    # 显示生成的文件
    print("生成的文件:")
    if single_turn_output.exists():
        num_lines = sum(1 for _ in open(single_turn_output))
        size = single_turn_output.stat().st_size / 1024
        print(f"  ✓ {single_turn_output}")
        print(f"    - {num_lines} 个trace条目")
        print(f"    - {size:.1f} KB")

    if multi_turn_output.exists():
        num_lines = sum(1 for _ in open(multi_turn_output))
        size = multi_turn_output.stat().st_size / 1024
        print(f"  ✓ {multi_turn_output}")
        print(f"    - {num_lines} 个trace条目")
        print(f"    - {size:.1f} KB")


if __name__ == "__main__":
    main()