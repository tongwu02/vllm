#!/usr/bin/env python3
"""
ShareGPT数据预处理器

将ShareGPT格式的对话数据转换为Simulator可用的trace格式
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Message:
    """单个消息"""
    role: str  # 'user' 或 'assistant'
    content: str


@dataclass
class Conversation:
    """完整对话"""
    id: str
    messages: List[Message]


class ShareGPTPreprocessor:
    """ShareGPT数据预处理器"""

    def __init__(self, tokenizer):
        """
        Args:
            tokenizer: HuggingFace tokenizer (用于应用chat template)
        """
        self.tokenizer = tokenizer
        self.stats = {
            'total_conversations': 0,
            'filtered_conversations': 0,
            'valid_conversations': 0,
            'total_turns': 0,
        }

    def normalize_role(self, role: str) -> str:
        """
        标准化role名称

        ShareGPT使用 'human'/'gpt'
        我们需要转换为 'user'/'assistant'
        """
        role_mapping = {
            'human': 'user',
            'user': 'user',
            'gpt': 'assistant',
            'assistant': 'assistant',
            'system': 'system',
        }
        normalized = role_mapping.get(role.lower(), role.lower())
        return normalized

    def clean_conversation(self, raw_conv: Dict, conv_index: int = 0) -> Optional[Conversation]:
        """
        清理和标准化对话

        执行以下操作:
        1. 标准化role labels
        2. 移除system消息
        3. 移除leading assistant消息
        4. 验证对话有效性

        Args:
            raw_conv: 原始对话数据
            conv_index: 对话索引(用于生成ID)

        Returns:
            Conversation对象,如果对话无效则返回None
        """
        # 提取对话ID,如果没有则自动生成
        conv_id = raw_conv.get('id', f'conversation_{conv_index:05d}')

        # 获取原始消息列表
        raw_messages = raw_conv.get('conversations', [])
        if not raw_messages:
            return None

        # 标准化roles并移除system消息
        messages = []
        for msg in raw_messages:
            role = self.normalize_role(msg.get('from', ''))
            content = msg.get('value', '').strip()

            # 跳过system消息
            if role == 'system':
                continue

            # 跳过空消息
            if not content:
                continue

            messages.append(Message(role=role, content=content))

        if not messages:
            return None

        # 移除leading assistant消息
        # 对话必须从user开始
        while messages and messages[0].role == 'assistant':
            messages.pop(0)

        if not messages:
            return None

        return Conversation(id=conv_id, messages=messages)

    def is_valid_conversation(self, conv: Conversation) -> bool:
        """
        验证对话是否有效

        有效对话必须满足:
        1. 至少有2个消息(1轮对话)
        2. 只包含user和assistant
        3. 严格交替(user -> assistant -> user -> ...)
        """
        if len(conv.messages) < 2:
            return False

        # 检查是否只有user/assistant
        for msg in conv.messages:
            if msg.role not in ['user', 'assistant']:
                return False

        # 检查是否严格交替
        expected_role = 'user'
        for msg in conv.messages:
            if msg.role != expected_role:
                return False
            # 切换expected role
            expected_role = 'assistant' if expected_role == 'user' else 'user'

        return True

    def conversation_to_trace_entries(
        self,
        conv: Conversation,
        max_turns: Optional[int] = None,
        single_turn_only: bool = False
    ) -> List[Dict[str, str]]:
        """
        将对话转换为trace条目

        Args:
            conv: 对话对象
            max_turns: 最多保留多少轮对话(None = 全部)
            single_turn_only: 是否只生成单轮trace

        Returns:
            trace条目列表,每个条目格式为:
            {"prompt": "...", "response": "..."}
        """
        if not self.is_valid_conversation(conv):
            return []

        trace_entries = []

        # 计算总轮数
        num_turns = len(conv.messages) // 2
        if max_turns:
            num_turns = min(num_turns, max_turns)

        if single_turn_only:
            # Single-turn模式:只取第一轮
            user_msg = conv.messages[0]
            assistant_msg = conv.messages[1]

            # 使用chat template格式化
            prompt = self._format_prompt([user_msg])
            response = assistant_msg.content

            trace_entries.append({
                "prompt": prompt,
                "response": response,
                "conversation_id": conv.id,
                "turn_index": 0
            })

        else:
            # Multi-turn模式:每轮都生成一个trace条目
            conversation_history = []

            for turn_idx in range(num_turns):
                msg_idx = turn_idx * 2

                user_msg = conv.messages[msg_idx]
                assistant_msg = conv.messages[msg_idx + 1]

                # 添加user消息到历史
                conversation_history.append(user_msg)

                # 格式化prompt(包含完整历史)
                prompt = self._format_prompt(conversation_history)
                response = assistant_msg.content

                trace_entries.append({
                    "prompt": prompt,
                    "response": response,
                    "conversation_id": conv.id,
                    "turn_index": turn_idx
                })

                # 添加assistant消息到历史
                conversation_history.append(assistant_msg)

        return trace_entries

    def _format_prompt(self, messages: List[Message]) -> str:
        """
        手动格式化prompt,遵循Llama格式但不包含system message

        重要: 根据25TheFutureCloud.pdf第6页:
        - Remove all 'system' messages
        - 只包含user和assistant的对话
        - Llama的tokenizer会自动添加system message,所以我们手动构建

        Args:
            messages: 消息列表(已经移除了system messages)

        Returns:
            格式化后的prompt字符串
        """
        # 手动构建Llama格式的prompt,不包含system message
        prompt_parts = ["<|begin_of_text|>"]

        for msg in messages:
            prompt_parts.append(f"<|start_header_id|>{msg.role}<|end_header_id|>\n\n")
            prompt_parts.append(f"{msg.content}<|eot_id|>")

        # 添加assistant的起始标记(准备生成)
        prompt_parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")

        return "".join(prompt_parts)

    def process_dataset(
        self,
        input_path: Path,
        output_path: Path,
        max_conversations: Optional[int] = None,
        max_turns_per_conversation: Optional[int] = None,
        single_turn_only: bool = False,
    ):
        """
        处理完整数据集

        Args:
            input_path: 输入JSONL文件路径
            output_path: 输出trace文件路径
            max_conversations: 最多处理多少个对话
            max_turns_per_conversation: 每个对话最多保留多少轮
            single_turn_only: 是否只生成单轮trace
        """
        logger.info(f"Processing dataset: {input_path}")
        logger.info(f"Output: {output_path}")
        logger.info(f"Single-turn only: {single_turn_only}")

        # 创建输出目录
        output_path.parent.mkdir(parents=True, exist_ok=True)

        all_trace_entries = []

        with open(input_path, 'r', encoding='utf-8') as f:
            # 计数器
            collected_conversations = 0
            for line_idx, line in enumerate(f):
                
                # 限制处理数量
                if max_conversations and collected_conversations >= max_conversations:
                    break

                try:
                    raw_conv = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON at line {line_idx}")
                    continue

                self.stats['total_conversations'] += 1

                # 清理对话(传入line_idx作为对话索引)
                conv = self.clean_conversation(raw_conv, conv_index=line_idx)
                if conv is None:
                    self.stats['filtered_conversations'] += 1
                    continue

                # 验证对话
                if not self.is_valid_conversation(conv):
                    self.stats['filtered_conversations'] += 1
                    continue

                self.stats['valid_conversations'] += 1

                collected_conversations += 1

                # 转换为trace条目
                trace_entries = self.conversation_to_trace_entries(
                    conv,
                    max_turns=max_turns_per_conversation,
                    single_turn_only=single_turn_only
                )

                all_trace_entries.extend(trace_entries)
                self.stats['total_turns'] += len(trace_entries)

        # 写入输出文件
        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in all_trace_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

        logger.info("✓ Processing complete!")
        logger.info(f"  Total conversations: {self.stats['total_conversations']}")
        logger.info(f"  Filtered out: {self.stats['filtered_conversations']}")
        logger.info(f"  Valid conversations: {self.stats['valid_conversations']}")
        logger.info(f"  Generated trace entries: {len(all_trace_entries)}")
