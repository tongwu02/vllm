#!/usr/bin/env python3
"""
Client Simulator for Milestone 2 Task 1
Simulates a client sending requests to the vLLM engine
"""
import json
import logging
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Request:
    """Single request"""
    request_id: str
    prompt: str
    response: str  # Ground truth from trace
    arrival_time: float
    conversation_id: str
    turn_index: int
    prompt_token_ids: List[int] = field(default_factory=list)
    max_tokens: int = 64


@dataclass
class RequestResult:
    """Result of a request"""
    request_id: str
    conversation_id: str
    turn_index: int
    prompt_length: int  # Number of prompt tokens
    response_length: int  # Number of actually generated tokens
    hit_blocks: int = 0  # Number of hit cache blocks
    total_blocks: int = 0  # Total number of blocks
    hit_rate: float = 0.0  # Cache hit rate for this request
    start_time: float = 0.0
    end_time: float = 0.0
    latency: float = 0.0  # Seconds


class ClientSimulator:
    """
    Client Simulator

    According to project.pdf Task 1 requirements:
    - Timing: Use Poisson distribution to simulate request arrival times
    - Chat templates: Correctly format the prompt
    """

    def __init__(
        self,
        trace_path: Path,
        tokenizer,
        arrival_rate: float = 1.0,  # Average requests per second
        use_trace_timing: bool = False,  # Whether to use time from trace
    ):
        """
        Args:
            trace_path: Path to the preprocessed trace file
            tokenizer: HuggingFace tokenizer
            arrival_rate: Lambda parameter for Poisson distribution (requests/second)
            use_trace_timing: If True and trace has timestamps, use trace time
        """
        self.trace_path = trace_path
        self.tokenizer = tokenizer
        self.arrival_rate = arrival_rate
        self.use_trace_timing = use_trace_timing
        self.requests: List[Request] = []
        self.results: Dict[str, RequestResult] = {}

        self._load_trace()

    def _load_trace(self):
        """Load trace file"""
        logger.info(f"Loading trace from {self.trace_path}")

        with open(self.trace_path, 'r', encoding='utf-8') as f:
            entries = [json.loads(line) for line in f if line.strip()]

        # Generate request arrival times
        current_time = 0.0
        for i, entry in enumerate(entries):
            # Use Poisson process to generate arrival time intervals
            if self.use_trace_timing and 'timestamp' in entry:
                arrival_time = entry['timestamp']
            elif 'arrival_time' in entry:
                # If trace contains arrival_time, use it directly
                arrival_time = entry['arrival_time']
            else:
                # Poisson distribution: intervals follow exponential distribution
                interval = np.random.exponential(1.0 / self.arrival_rate)
                current_time += interval
                arrival_time = current_time

            prompt = entry['prompt']
            response = entry['response']

            # Tokenize prompt to calculate length
            try:
                prompt_token_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
            except Exception as e:
                logger.warning(f"Failed to tokenize prompt: {e}")
                prompt_token_ids = []

            request = Request(
                request_id=f"req-{i}",
                prompt=prompt,
                response=response,
                arrival_time=arrival_time,
                conversation_id=entry.get('conversation_id', 'unknown'),
                turn_index=entry.get('turn_index', 0),
                prompt_token_ids=prompt_token_ids,
                max_tokens=entry.get('max_tokens', 64),
            )
            self.requests.append(request)

        logger.info(f"Loaded {len(self.requests)} requests")
        if self.requests:
            logger.info(f"  Time span: {self.requests[-1].arrival_time:.2f} seconds")
            logger.info(f"  Avg arrival rate: {len(self.requests) / self.requests[-1].arrival_time:.2f} req/s")

    def send_requests_to_engine(self, engine, params_override: Optional[Dict] = None):
        """
        Send requests to the vLLM engine

        Args:
            engine: LLMEngine instance
            params_override: Override default SamplingParams
        """
        from vllm import SamplingParams

        logger.info(f"Sending {len(self.requests)} requests to engine...")

        start_time = time.time()

        for req in self.requests:
            # Construct SamplingParams
            params_dict = {
                'max_tokens': req.max_tokens,
                'temperature': 0.0,  # Deterministic generation
                'top_p': 1.0,
            }
            if params_override:
                params_dict.update(params_override)

            params = SamplingParams(**params_dict)

            # Add request to engine
            engine.add_request(
                request_id=req.request_id,
                prompt=req.prompt,
                params=params,
            )

            # Record start time
            self.results[req.request_id] = RequestResult(
                request_id=req.request_id,
                conversation_id=req.conversation_id,
                turn_index=req.turn_index,
                prompt_length=len(req.prompt_token_ids),
                response_length=0,
                start_time=time.time(),
            )

        logger.info(f"All requests submitted in {time.time() - start_time:.2f}s")

    def send_requests_conversation_by_conversation(self, engine, params_override: Optional[Dict] = None, max_steps_per_turn: int = 10000):
        """
        Conversation-by-conversation processing (Improved Option A)
        Group by conversation_id, process sequentially by turn_index within each conversation

        This ensures subsequent turns in the same conversation can reuse cached blocks from previous turns

        Args:
            engine: LLMEngine instance
            params_override: Override default SamplingParams
            max_steps_per_turn: Maximum steps per turn
        """
        from vllm import SamplingParams

        # Group by conversation_id
        conversations = defaultdict(list)
        for req in self.requests:
            conversations[req.conversation_id].append(req)

        # Sort by turn_index within each conversation
        for conv_id in conversations:
            conversations[conv_id].sort(key=lambda r: r.turn_index)

        logger.info("=" * 80)
        logger.info("Conversation-by-Conversation Sequential Processing")
        logger.info("=" * 80)
        logger.info(f"Total conversations: {len(conversations)}")

        # Statistics on turn distribution
        turn_distribution = defaultdict(int)
        for conv_id, conv_requests in conversations.items():
            num_turns = len(conv_requests)
            turn_distribution[num_turns] += 1

        logger.info(f"Turn distribution:")
        for num_turns in sorted(turn_distribution.keys()):
            logger.info(f"  {num_turns} turns: {turn_distribution[num_turns]} conversations")

        # Process conversations sequentially
        conv_count = 0
        for conv_id in sorted(conversations.keys()):
            conv_requests = conversations[conv_id]
            conv_count += 1

            logger.info("")
            logger.info(f"【Conversation {conv_count}/{len(conversations)}】{conv_id} - {len(conv_requests)} turns")

            # Process each turn of this conversation
            for turn_req in conv_requests:
                logger.info(f"  Turn {turn_req.turn_index}: Processing request {turn_req.request_id}...")

                # Construct SamplingParams
                params_dict = {
                    'max_tokens': turn_req.max_tokens,
                    'temperature': 0.0,
                    'top_p': 1.0,
                }
                if params_override:
                    params_dict.update(params_override)

                params = SamplingParams(**params_dict)

                # Add request to engine
                start_time = time.time()
                engine.add_request(
                    request_id=turn_req.request_id,
                    prompt=turn_req.prompt,
                    params=params,
                )

                # Record start time
                self.results[turn_req.request_id] = RequestResult(
                    request_id=turn_req.request_id,
                    conversation_id=turn_req.conversation_id,
                    turn_index=turn_req.turn_index,
                    prompt_length=len(turn_req.prompt_token_ids),
                    response_length=0,
                    start_time=time.time(),
                )

                # Run engine until this turn completes
                step = 0
                while step < max_steps_per_turn:
                    outputs = engine.step()

                    # Update results
                    for output in outputs:
                        if output.request_id == turn_req.request_id:
                            result = self.results[turn_req.request_id]
                            result.response_length = len(output.outputs[0].token_ids)

                            if output.finished:
                                result.end_time = time.time()
                                result.latency = result.end_time - result.start_time
                                logger.info(f"    ✓ Turn {turn_req.turn_index} completed in {step + 1} steps ({result.latency:.2f}s)")
                                break

                    # Check if this turn is complete
                    if self.results[turn_req.request_id].end_time > 0:
                        break

                    step += 1

                if step >= max_steps_per_turn:
                    logger.warning(f"    ⚠️  Turn {turn_req.turn_index} reached max steps ({max_steps_per_turn})")

        logger.info("")
        logger.info("=" * 80)
        logger.info("✓ All conversations completed sequentially")
        logger.info("=" * 80)

    def send_requests_turn_by_turn(self, engine, params_override: Optional[Dict] = None, max_steps_per_turn: int = 10000):
        """
        Turn-by-turn sequential processing (Original Option A - Group by turn_index)

        Note: This method is suitable for testing a single conversation; for multiple conversations use
        send_requests_conversation_by_conversation()

        Args:
            engine: LLMEngine instance
            params_override: Override default SamplingParams
            max_steps_per_turn: Maximum steps per turn
        """
        from vllm import SamplingParams

        # Group by turn_index
        turns_groups = defaultdict(list)
        for req in self.requests:
            turns_groups[req.turn_index].append(req)

        logger.info("=" * 80)
        logger.info("Turn-by-Turn Sequential Processing (by turn_index)")
        logger.info("=" * 80)
        logger.info(f"Total turns: {len(turns_groups)}")
        for turn_idx in sorted(turns_groups.keys()):
            logger.info(f"  Turn {turn_idx}: {len(turns_groups[turn_idx])} requests")

        # Process sequentially by turn_index
        for turn_idx in sorted(turns_groups.keys()):
            turn_requests = turns_groups[turn_idx]

            logger.info("")
            logger.info(f"【Turn {turn_idx}】Processing {len(turn_requests)} requests...")

            # Send all requests for this turn
            start_time = time.time()
            for req in turn_requests:
                # Construct SamplingParams
                params_dict = {
                    'max_tokens': req.max_tokens,
                    'temperature': 0.0,
                    'top_p': 1.0,
                }
                if params_override:
                    params_dict.update(params_override)

                params = SamplingParams(**params_dict)

                # Add request to engine
                engine.add_request(
                    request_id=req.request_id,
                    prompt=req.prompt,
                    params=params,
                )

                # Record start time
                self.results[req.request_id] = RequestResult(
                    request_id=req.request_id,
                    conversation_id=req.conversation_id,
                    turn_index=req.turn_index,
                    prompt_length=len(req.prompt_token_ids),
                    response_length=0,
                    start_time=time.time(),
                )

            logger.info(f"  Submitted {len(turn_requests)} requests in {time.time() - start_time:.2f}s")

            # Run engine until all requests for this turn complete
            logger.info(f"  Running engine until Turn {turn_idx} completes...")
            step = 0
            turn_request_ids = set(req.request_id for req in turn_requests)

            while step < max_steps_per_turn:
                outputs = engine.step()

                # Update results
                for output in outputs:
                    if output.request_id in self.results:
                        result = self.results[output.request_id]
                        result.response_length = len(output.outputs[0].token_ids)

                        if output.finished:
                            result.end_time = time.time()
                            result.latency = result.end_time - result.start_time

                # Check if all requests for this turn are complete
                turn_completed = all(
                    self.results[req_id].end_time > 0
                    for req_id in turn_request_ids
                )

                if turn_completed:
                    logger.info(f"  ✓ Turn {turn_idx} completed in {step + 1} steps")
                    break

                step += 1

                # Output progress every 100 steps
                if step % 100 == 0:
                    pending = len([req_id for req_id in turn_request_ids if self.results[req_id].end_time == 0])
                    logger.info(f"    Step {step}: {pending}/{len(turn_request_ids)} requests pending")

            if step >= max_steps_per_turn:
                logger.warning(f"  ⚠️  Turn {turn_idx} reached max steps ({max_steps_per_turn})")

        logger.info("")
        logger.info("=" * 80)
        logger.info("✓ All turns completed sequentially")
        logger.info("=" * 80)

    def run_engine_until_complete(self, engine, max_steps: int = 10000):
        """
        Run engine until all requests are complete

        Args:
            engine: LLMEngine instance
            max_steps: Maximum steps
        """
        logger.info("Running engine until all requests complete...")

        step = 0
        while step < max_steps:
            outputs = engine.step()

            # Update results
            for output in outputs:
                if output.request_id in self.results:
                    result = self.results[output.request_id]
                    result.response_length = len(output.outputs[0].token_ids)

                    if output.finished:
                        result.end_time = time.time()
                        result.latency = result.end_time - result.start_time

            if not engine.has_unfinished_requests():
                logger.info(f"All requests completed in {step + 1} steps")
                break

            step += 1

            # Output progress every 100 steps
            if step % 100 == 0:
                pending = len([r for r in self.results.values() if r.end_time == 0])
                logger.info(f"  Step {step}: {pending} requests pending")

        if step >= max_steps:
            logger.warning(f"Reached max steps ({max_steps}), some requests may not be finished")

    def collect_prefix_cache_stats(self, engine):
        """
        Collect prefix cache statistics

        Collect according to project.pdf Task 2 requirements:
        - The fraction of each request that benefits from prefix sharing
        - The number of hits per cache block
        - The time gap between reuses of each cache block
        """
        from vllm.utils import Device

        logger.info("Collecting prefix cache statistics...")

        # Use our corrected hit rate tracker
        try:
            from correct_hit_rate_tracker import global_hit_rate_tracker
            tracker_stats = global_hit_rate_tracker.get_stats()
            overall_hit_rate = tracker_stats['overall_hit_rate']
            logger.info(f"✓ Correct prefix cache hit rate (first prefill only): {overall_hit_rate:.2%}")
            logger.info(f"  Total requests: {tracker_stats['total_requests']}")
            logger.info(f"  Total blocks: {tracker_stats['total_blocks']}")
            logger.info(f"  Hit blocks: {tracker_stats['hit_blocks']}")
            logger.info(f"  Requests with hits: {tracker_stats['requests_with_hits']}")
            logger.info(f"  Requests with zero hits: {tracker_stats['requests_with_zero_hits']}")
        except Exception as e:
            logger.warning(f"Failed to get correct hit rate: {e}")
            # Fallback to old method
            try:
                overall_hit_rate = engine.scheduler[0].get_prefix_cache_hit_rate(Device.GPU)
                logger.info(f"Overall prefix cache hit rate (old method): {overall_hit_rate:.2%}")
                tracker_stats = {}
            except Exception as e2:
                logger.warning(f"Failed to get overall hit rate: {e2}")
                overall_hit_rate = 0.0
                tracker_stats = {}

        # Collect cache block statistics (Task 2 requirement)
        block_stats = {}
        try:
            from cache_block_tracker import global_cache_block_tracker
            block_stats = global_cache_block_tracker.get_stats()
            logger.info(f"✓ Cache block statistics:")
            logger.info(f"  Total cached blocks: {block_stats.get('total_cached_blocks', 0)}")
            logger.info(f"  Average hits per block: {block_stats.get('avg_hits_per_block', 0):.2f}")
            logger.info(f"  Average reuse gap: {block_stats.get('avg_reuse_gap_seconds', 0):.4f}s")
        except Exception as e:
            logger.warning(f"Failed to get block statistics: {e}")

        result = {
            'overall_hit_rate': overall_hit_rate,
            'total_requests': len(self.results),
            'completed_requests': len([r for r in self.results.values() if r.end_time > 0]),
            'hit_rate_stats': tracker_stats,
            'block_stats': block_stats,
        }
        return result

    def generate_report(self, output_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Generate experiment report

        Returns:
            Dictionary of statistics
        """
        logger.info("Generating report...")

        completed_results = [r for r in self.results.values() if r.end_time > 0]

        if not completed_results:
            logger.warning("No completed requests to report")
            return {}

        # Basic statistics
        stats = {
            'total_requests': len(self.requests),
            'completed_requests': len(completed_results),
            'avg_prompt_length': np.mean([r.prompt_length for r in completed_results]),
            'avg_response_length': np.mean([r.response_length for r in completed_results]),
            'avg_latency': np.mean([r.latency for r in completed_results]),
            'p50_latency': np.percentile([r.latency for r in completed_results], 50),
            'p95_latency': np.percentile([r.latency for r in completed_results], 95),
            'p99_latency': np.percentile([r.latency for r in completed_results], 99),
        }

        # Group statistics by conversation
        conv_groups = defaultdict(list)
        for result in completed_results:
            conv_groups[result.conversation_id].append(result)

        stats['num_conversations'] = len(conv_groups)
        stats['avg_turns_per_conversation'] = np.mean([len(turns) for turns in conv_groups.values()])

        # Output report
        logger.info("=" * 60)
        logger.info("Experiment Report")
        logger.info("=" * 60)
        logger.info(f"Total requests: {stats['total_requests']}")
        logger.info(f"Completed requests: {stats['completed_requests']}")
        logger.info(f"Conversations: {stats['num_conversations']}")
        logger.info(f"Avg turns/conversation: {stats['avg_turns_per_conversation']:.2f}")
        logger.info(f"Avg prompt length: {stats['avg_prompt_length']:.1f} tokens")
        logger.info(f"Avg response length: {stats['avg_response_length']:.1f} tokens")
        logger.info(f"Avg latency: {stats['avg_latency']:.3f}s")
        logger.info(f"P50 latency: {stats['p50_latency']:.3f}s")
        logger.info(f"P95 latency: {stats['p95_latency']:.3f}s")
        logger.info(f"P99 latency: {stats['p99_latency']:.3f}s")
        logger.info("=" * 60)

        # Save to file
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(stats, f, indent=2)
            logger.info(f"Report saved to {output_path}")

        return stats


def main():
    """Example usage"""
    print("""
Client Simulator for Milestone 2

Usage:
    from milestone2_code.client_simulator import ClientSimulator
    from vllm import EngineArgs
    from vllm.engine.llm_engine import LLMEngine
    from transformers import AutoTokenizer

    # 1. Prepare trace
    # Use trace_preprocessor.py to preprocess ShareGPT data

    # 2. Create simulator
    tokenizer = AutoTokenizer.from_pretrained("exported_models/Llama-3.2-1B-Instruct")
    simulator = ClientSimulator(
        trace_path=Path("milestone2_code/sharegpt_trace.jsonl"),
        tokenizer=tokenizer,
        arrival_rate=1.0,
    )

    # 3. Create engine
    args = EngineArgs(
        model="exported_models/Llama-3.2-1B-Instruct",
        tokenizer="exported_models/Llama-3.2-1B-Instruct",
        device="cpu",
        enable_prefix_caching=True,
    )
    engine = LLMEngine.from_engine_args(args)

    # 4. Run experiment
    simulator.send_requests_to_engine(engine)
    simulator.run_engine_until_complete(engine)

    # 5. Collect statistics
    simulator.collect_prefix_cache_stats(engine)
    stats = simulator.generate_report()
    """)


if __name__ == "__main__":
    main()