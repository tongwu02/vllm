"""
Client Simulator for replaying ShareGPT traces.

This module provides functionality to:
1. Load and parse ShareGPT conversation traces
2. Format prompts using model-specific chat templates
3. Generate request traces with timing information (Poisson distribution)
4. Support both single-turn and multi-turn conversation modes
"""

import json
import logging
import random
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """Represents a single turn in a conversation."""
    role: str  # 'human' or 'gpt'
    content: str


@dataclass
class Conversation:
    """Represents a complete conversation with multiple turns."""
    conversation_id: str
    turns: List[ConversationTurn]


@dataclass
class RequestTrace:
    """Represents a request to be sent to the vLLM server."""
    request_id: str
    conversation_id: str
    turn_index: int  # Which turn in the conversation
    prompt: str  # The formatted prompt (with chat template)
    response: str  # Expected response from trace
    arrival_time: float  # When to send this request (in seconds)


class ShareGPTLoader:
    """Loads and parses ShareGPT conversation traces."""

    def __init__(self, trace_path: str, max_conversations: Optional[int] = None):
        """
        Args:
            trace_path: Path to ShareGPT JSONL file
            max_conversations: Maximum number of conversations to load (None = all)
        """
        self.trace_path = trace_path
        self.max_conversations = max_conversations
        self.conversations: List[Conversation] = []
        self._load_traces()

    def _load_traces(self):
        """Load conversations from ShareGPT JSONL file."""
        count = 0
        with open(self.trace_path, 'r', encoding='utf-8') as f:
            for line in f:
                if self.max_conversations and count >= self.max_conversations:
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    conv = self._parse_conversation(data)
                    if conv and len(conv.turns) > 0:
                        self.conversations.append(conv)
                        count += 1
                except Exception as e:
                    logger.warning(f"Failed to parse conversation: {e}")
                    continue

        logger.info(f"Loaded {len(self.conversations)} conversations from {self.trace_path}")

    def _parse_conversation(self, data: Dict[str, Any]) -> Optional[Conversation]:
        """Parse a single conversation from JSON data."""
        conv_id = data.get('id', 'unknown')
        conversations = data.get('conversations', [])

        turns = []
        for msg in conversations:
            role = msg.get('from', '')
            # Try multiple fields for content
            content = msg.get('value') or msg.get('text') or msg.get('markdown') or ''

            if not content or not isinstance(content, str):
                continue

            turns.append(ConversationTurn(role=role, content=content.strip()))

        if len(turns) == 0:
            return None

        return Conversation(conversation_id=conv_id, turns=turns)

    def get_conversations(self) -> List[Conversation]:
        """Get all loaded conversations."""
        return self.conversations


class ChatTemplateFormatter:
    """Formats prompts using model-specific chat templates."""

    def __init__(self, tokenizer=None, model_name: Optional[str] = None):
        """
        Args:
            tokenizer: HuggingFace tokenizer with chat template (optional)
            model_name: Model name for template selection (optional)
        """
        self.tokenizer = tokenizer
        self.model_name = model_name or ""

    def format_conversation(self, turns: List[ConversationTurn]) -> str:
        """
        Format a conversation history into a prompt.

        Args:
            turns: List of conversation turns (human/gpt alternating)

        Returns:
            Formatted prompt string
        """
        # Try to use tokenizer's chat template if available
        if self.tokenizer and hasattr(self.tokenizer, 'apply_chat_template'):
            try:
                messages = []
                for turn in turns:
                    role = 'user' if turn.role == 'human' else 'assistant'
                    messages.append({'role': role, 'content': turn.content})

                # apply_chat_template with tokenize=False returns the formatted string
                formatted = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                return formatted
            except Exception as e:
                logger.warning(f"Failed to apply chat template: {e}, falling back to simple format")

        # Fallback: simple format
        return self._simple_format(turns)

    def _simple_format(self, turns: List[ConversationTurn]) -> str:
        """Simple chat format as fallback."""
        lines = []
        for turn in turns:
            if turn.role == 'human':
                lines.append(f"User: {turn.content}")
            else:
                lines.append(f"Assistant: {turn.content}")

        # Add prompt for next assistant response
        lines.append("Assistant:")
        return "\n".join(lines)


class RequestGenerator:
    """Generates request traces with timing information."""

    def __init__(
        self,
        conversations: List[Conversation],
        arrival_rate: float = 1.0,  # requests per second
        use_poisson: bool = True,
        seed: Optional[int] = 42
    ):
        """
        Args:
            conversations: List of conversations to generate requests from
            arrival_rate: Average request arrival rate (requests/second)
            use_poisson: Whether to use Poisson distribution for arrival times
            seed: Random seed for reproducibility
        """
        self.conversations = conversations
        self.arrival_rate = arrival_rate
        self.use_poisson = use_poisson

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def generate_single_turn_traces(
        self,
        formatter: ChatTemplateFormatter
    ) -> List[RequestTrace]:
        """
        Generate request traces for single-turn conversations.
        Only the first user prompt and expected response.

        Returns:
            List of RequestTrace objects
        """
        traces = []
        current_time = 0.0

        for conv in self.conversations:
            # Find the first human turn
            human_turn = None
            gpt_turn = None

            for i, turn in enumerate(conv.turns):
                if turn.role == 'human' and human_turn is None:
                    human_turn = turn
                elif turn.role == 'gpt' and human_turn is not None and gpt_turn is None:
                    gpt_turn = turn
                    break

            if human_turn is None or gpt_turn is None:
                continue

            # Format the prompt
            prompt = formatter.format_conversation([human_turn])

            # Generate arrival time
            if self.use_poisson:
                inter_arrival = np.random.exponential(1.0 / self.arrival_rate)
            else:
                inter_arrival = 1.0 / self.arrival_rate

            current_time += inter_arrival

            request_id = f"{conv.conversation_id}_turn0"
            traces.append(RequestTrace(
                request_id=request_id,
                conversation_id=conv.conversation_id,
                turn_index=0,
                prompt=prompt,
                response=gpt_turn.content,
                arrival_time=current_time
            ))

        logger.info(f"Generated {len(traces)} single-turn request traces")
        return traces

    def generate_multi_turn_traces(
        self,
        formatter: ChatTemplateFormatter,
        turn_delay: float = 0.0  # Additional delay between turns in same conversation
    ) -> List[RequestTrace]:
        """
        Generate request traces for multi-turn conversations.
        All turns in each conversation.

        Args:
            formatter: Chat template formatter
            turn_delay: Additional delay between turns in the same conversation (seconds)

        Returns:
            List of RequestTrace objects
        """
        traces = []
        current_time = 0.0

        for conv in self.conversations:
            # Process all human-gpt pairs
            conversation_history = []
            turn_index = 0

            i = 0
            while i < len(conv.turns):
                turn = conv.turns[i]

                if turn.role == 'human':
                    # Add human turn to history
                    conversation_history.append(turn)

                    # Look for corresponding GPT response
                    if i + 1 < len(conv.turns) and conv.turns[i + 1].role == 'gpt':
                        gpt_turn = conv.turns[i + 1]

                        # Format prompt with conversation history
                        prompt = formatter.format_conversation(conversation_history)

                        # Generate arrival time
                        if turn_index == 0:
                            # First turn: use inter-conversation delay
                            if self.use_poisson:
                                inter_arrival = np.random.exponential(1.0 / self.arrival_rate)
                            else:
                                inter_arrival = 1.0 / self.arrival_rate
                        else:
                            # Subsequent turns: use turn delay
                            inter_arrival = turn_delay

                        current_time += inter_arrival

                        request_id = f"{conv.conversation_id}_turn{turn_index}"
                        traces.append(RequestTrace(
                            request_id=request_id,
                            conversation_id=conv.conversation_id,
                            turn_index=turn_index,
                            prompt=prompt,
                            response=gpt_turn.content,
                            arrival_time=current_time
                        ))

                        # Add GPT response to history for next turn
                        conversation_history.append(gpt_turn)
                        turn_index += 1
                        i += 2  # Skip the GPT turn we just processed
                    else:
                        i += 1
                else:
                    i += 1

        logger.info(f"Generated {len(traces)} multi-turn request traces")
        return traces


def create_trace_file_for_simulator(
    request_traces: List[RequestTrace],
    output_path: str
):
    """
    Create a trace file compatible with the Milestone 1 simulator.
    Format: {"prompt": "...", "response": "..."}

    Args:
        request_traces: List of request traces
        output_path: Output JSONL file path
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for trace in request_traces:
            record = {
                'prompt': trace.prompt,
                'response': trace.response
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    logger.info(f"Created trace file with {len(request_traces)} entries: {output_path}")


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Load ShareGPT data
    loader = ShareGPTLoader("ShareGPTData.jsonl", max_conversations=100)
    conversations = loader.get_conversations()

    # Create formatter
    formatter = ChatTemplateFormatter()

    # Generate traces
    generator = RequestGenerator(conversations, arrival_rate=2.0, use_poisson=True)

    # Single-turn traces
    single_turn_traces = generator.generate_single_turn_traces(formatter)
    create_trace_file_for_simulator(single_turn_traces, "single_turn_trace.jsonl")

    # Multi-turn traces
    multi_turn_traces = generator.generate_multi_turn_traces(formatter, turn_delay=1.0)
    create_trace_file_for_simulator(multi_turn_traces, "multi_turn_trace.jsonl")

    print(f"Generated {len(single_turn_traces)} single-turn traces")
    print(f"Generated {len(multi_turn_traces)} multi-turn traces")
