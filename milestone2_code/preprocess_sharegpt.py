#!/usr/bin/env python3
"""
预处理ShareGPT数据集

将ShareGPT_V3_unfiltered_cleaned_split.json转换为trace格式:
- sharegpt_single_turn.jsonl (单轮对话)
- sharegpt_multi_turn.jsonl (多轮对话)
"""
import sys
from pathlib import Path

# 确保可以import milestone2_code
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoTokenizer

# 直接import,因为已经在milestone2_code目录下
from trace_preprocessor import ShareGPTPreprocessor


def main():
    print("=" * 60)
    print("ShareGPT数据预处理")
    print("=" * 60)
    print()

    # 配置参数
    input_file = Path('ShareGPT_V3_unfiltered_cleaned_split.json')
    output_dir = Path('milestone2_code/traces')
    model_path = 'exported_models/Llama-3.2-1B-Instruct'
    max_conversations = 100  # 限制为100条对话用于测试

    # 检查输入文件
    if not input_file.exists():
        print(f"✗ 错误: 找不到输入文件 {input_file}")
        print("  请先运行: python milestone2_code/download_sharegpt.py --download")
        return

    print(f"输入文件: {input_file}")
    print(f"文件大小: {input_file.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"处理数量: {max_conversations} 条对话")
    print()

    # 加载tokenizer
    print("步骤1: 加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(f"  ✓ Tokenizer loaded: {model_path}")
    print()

    # 创建preprocessor
    preprocessor = ShareGPTPreprocessor(tokenizer)

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

    print()
    print("下一步:")
    print("  运行实验: python milestone2_code/run_experiments.py")
    print()


if __name__ == "__main__":
    main()
