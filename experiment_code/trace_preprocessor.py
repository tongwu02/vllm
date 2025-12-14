#!/usr/bin/env python3
"""
ShareGPT Data Preprocessor

Convert ShareGPT format conversation data into trace format usable by Simulator
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
    """Single message"""
    role: str  # 'user' or 'assistant'
    content: str


@dataclass
class Conversation:
    """Complete conversation"""
    id: str
    messages: List[Message]


class ShareGPTPreprocessor:
    """ShareGPT Data Preprocessor"""

    def __init__(self, tokenizer, system_prompt: Optional[str] = None):
        """
        Args:
            tokenizer: HuggingFace tokenizer (used for applying chat template)
            system_prompt: Optional system prompt configuration:
                          - None: Do not add system prompt
                          - String: Add this system prompt to all conversations
                          - "random": Randomly add default system prompt to 50% of conversations
        """
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.default_system_prompt = "You are a helpful assistant."
        self.stats = {
            'total_conversations': 0,
            'filtered_conversations': 0,
            'valid_conversations': 0,
            'total_turns': 0,
            'conversations_with_system_prompt': 0,
            'conversations_without_system_prompt': 0,
        }

    def normalize_role(self, role: str) -> str:
        """
        Normalize role labels

        ShareGPT uses 'human'/'gpt'
        We need to convert to 'user'/'assistant'
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
        Clean and normalize conversation

        Perform the following operations:
        1. Normalize role labels
        2. Remove metadata-like system messages (e.g., "This is from xxx.com")
        3. Retain role instruction system prompts to be added during formatting
        4. Remove leading assistant messages
        5. Verify conversation validity

        Args:
            raw_conv: Raw conversation data
            conv_index: Conversation index (used for generating ID)

        Returns:
            Conversation object, or None if conversation is invalid
        """
        # Extract conversation ID, generate automatically if missing
        conv_id = raw_conv.get('id', f'conversation_{conv_index:05d}')

        # Get original message list
        raw_messages = raw_conv.get('conversations', [])
        if not raw_messages:
            return None

        # Normalize roles and remove metadata-like system messages
        messages = []
        for msg in raw_messages:
            role = self.normalize_role(msg.get('from', ''))
            content = msg.get('value', '').strip()

            # Skip metadata-like system messages
            # These are usually website info, not role instructions
            if role == 'system':
                # Simple heuristic: if it contains website-related keywords, consider it metadata
                metadata_keywords = ['sharegpt', '.com', 'website', 'conversation from']
                if any(keyword in content.lower() for keyword in metadata_keywords):
                    continue
                # Otherwise retain as potential role instruction
                # But in our implementation, role instructions are added uniformly via self.system_prompt

            # Skip empty messages
            if not content:
                continue

            messages.append(Message(role=role, content=content))

        if not messages:
            return None

        # Remove leading assistant messages
        # Conversation must start with user
        while messages and messages[0].role == 'assistant':
            messages.pop(0)

        if not messages:
            return None

        return Conversation(id=conv_id, messages=messages)

    def is_valid_conversation(self, conv: Conversation) -> bool:
        """
        Verify if conversation is valid

        Valid conversations must satisfy:
        1. At least 2 messages (1 turn)
        2. Contain only user and assistant
        3. Strict alternation (user -> assistant -> user -> ...)
        """
        if len(conv.messages) < 2:
            return False

        # Check if only user/assistant present
        for msg in conv.messages:
            if msg.role not in ['user', 'assistant']:
                return False

        # Check for strict alternation
        expected_role = 'user'
        for msg in conv.messages:
            if msg.role != expected_role:
                return False
            # Switch expected role
            expected_role = 'assistant' if expected_role == 'user' else 'user'

        return True

    def conversation_to_trace_entries(
            self,
            conv: Conversation,
            max_turns: Optional[int] = None,
            single_turn_only: bool = False
    ) -> List[Dict[str, str]]:
        """
        Convert conversation to trace entries

        Args:
            conv: Conversation object
            max_turns: Maximum number of turns to keep (None = all)
            single_turn_only: Whether to generate only single-turn traces

        Returns:
            List of trace entries, each formatted as:
            {"prompt": "...", "response": "..."}
        """
        if not self.is_valid_conversation(conv):
            return []

        # Decide if this conversation should have a system prompt
        use_system_prompt_for_this_conv = self._should_use_system_prompt(conv.id)

        trace_entries = []

        # Calculate total turns
        num_turns = len(conv.messages) // 2
        if max_turns:
            num_turns = min(num_turns, max_turns)

        if single_turn_only:
            # Single-turn mode: take only the first turn
            user_msg = conv.messages[0]
            assistant_msg = conv.messages[1]

            # Format using chat template
            prompt = self._format_prompt([user_msg], use_system_prompt_for_this_conv)
            response = assistant_msg.content

            trace_entries.append({
                "prompt": prompt,
                "response": response,
                "conversation_id": conv.id,
                "turn_index": 0
            })

        else:
            # Multi-turn mode: generate a trace entry for each turn
            conversation_history = []

            for turn_idx in range(num_turns):
                msg_idx = turn_idx * 2

                user_msg = conv.messages[msg_idx]
                assistant_msg = conv.messages[msg_idx + 1]

                # Add user message to history
                conversation_history.append(user_msg)

                # Format prompt (including full history)
                prompt = self._format_prompt(conversation_history, use_system_prompt_for_this_conv)
                response = assistant_msg.content

                trace_entries.append({
                    "prompt": prompt,
                    "response": response,
                    "conversation_id": conv.id,
                    "turn_index": turn_idx
                })

                # Add assistant message to history
                conversation_history.append(assistant_msg)

        return trace_entries

    def _should_use_system_prompt(self, conversation_id: str) -> bool:
        """
        Decide whether a specific conversation should have a system prompt

        Args:
            conversation_id: Conversation ID

        Returns:
            Whether to use system prompt
        """
        if self.system_prompt is None:
            return False
        elif self.system_prompt == "random":
            # Use hash of conversation_id to decide, ensuring consistency for the same conversation
            import hashlib
            hash_value = int(hashlib.md5(conversation_id.encode()).hexdigest(), 16)
            return hash_value % 2 == 0  # 50% probability
        else:
            # Fixed system prompt, used for all conversations
            return True

    def _format_prompt(self, messages: List[Message], use_system_prompt: bool = False) -> str:
        """
        Manually format prompt, following Llama format

        Strategy:
        - Remove metadata-like system messages (e.g., "This is from xxx.com")
        - Optionally add role instruction system prompt (e.g., "You are a helpful assistant")
        - Contain only user and assistant conversations

        Args:
            messages: List of messages (metadata system messages already removed)
            use_system_prompt: Whether to add a system prompt for this prompt

        Returns:
            Formatted prompt string
        """
        # Manually construct Llama format prompt
        prompt_parts = ["<|begin_of_text|>"]

        # If system prompt should be used, add it
        if use_system_prompt:
            actual_prompt = self.system_prompt if self.system_prompt != "random" else self.default_system_prompt
            prompt_parts.append("<|start_header_id|>system<|end_header_id|>\n\n")
            prompt_parts.append(f"{actual_prompt}<|eot_id|>")

        # Add conversation messages
        for msg in messages:
            prompt_parts.append(f"<|start_header_id|>{msg.role}<|end_header_id|>\n\n")
            prompt_parts.append(f"{msg.content}<|eot_id|>")

        # Add assistant start token (ready for generation)
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
        Process complete dataset

        Args:
            input_path: Input JSONL file path
            output_path: Output trace file path
            max_conversations: Maximum number of conversations to process
            max_turns_per_conversation: Maximum number of turns per conversation
            single_turn_only: Whether to generate only single-turn traces
        """
        logger.info(f"Processing dataset: {input_path}")
        logger.info(f"Output: {output_path}")
        logger.info(f"Single-turn only: {single_turn_only}")

        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)

        all_trace_entries = []

        with open(input_path, 'r', encoding='utf-8') as f:
            # Counter
            collected_conversations = 0
            for line_idx, line in enumerate(f):

                # Limit processing count
                if max_conversations and collected_conversations >= max_conversations:
                    break

                try:
                    raw_conv = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON at line {line_idx}")
                    continue

                self.stats['total_conversations'] += 1

                # Clean conversation (pass line_idx as conversation index)
                conv = self.clean_conversation(raw_conv, conv_index=line_idx)
                if conv is None:
                    self.stats['filtered_conversations'] += 1
                    continue

                # Validate conversation
                if not self.is_valid_conversation(conv):
                    self.stats['filtered_conversations'] += 1
                    continue

                self.stats['valid_conversations'] += 1

                collected_conversations += 1

                # Convert to trace entries
                trace_entries = self.conversation_to_trace_entries(
                    conv,
                    max_turns=max_turns_per_conversation,
                    single_turn_only=single_turn_only
                )

                # Count if system prompt was used
                if self._should_use_system_prompt(conv.id):
                    self.stats['conversations_with_system_prompt'] += 1
                else:
                    self.stats['conversations_without_system_prompt'] += 1

                all_trace_entries.extend(trace_entries)
                self.stats['total_turns'] += len(trace_entries)

        # Write to output file
        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in all_trace_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

        logger.info("✓ Processing complete!")
        logger.info(f"  Total conversations: {self.stats['total_conversations']}")
        logger.info(f"  Filtered out: {self.stats['filtered_conversations']}")
        logger.info(f"  Valid conversations: {self.stats['valid_conversations']}")
        logger.info(f"  Conversations with system prompt: {self.stats['conversations_with_system_prompt']}")
        logger.info(f"  Conversations without system prompt: {self.stats['conversations_without_system_prompt']}")
        logger.info(f"  Generated trace entries: {len(all_trace_entries)}")