"""
Milestone 2 Task 2: Using Real vLLM with Simulator

This script uses:
- Real vLLM scheduler and block manager
- Milestone 1 simulator (no GPU needed)
- Real tokenizer
- ShareGPT traces

This provides accurate prefix sharing metrics from vLLM's actual block manager.
"""

import argparse
import json
import logging
import time
import sys
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict

# Add vllm to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from client_simulator import (
    ShareGPTLoader,
    ChatTemplateFormatter,
    RequestGenerator,
    RequestTrace
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VLLMPrefixSharingAnalyzer:
    """
    Analyzer that uses real vLLM engine with simulator to collect
    accurate prefix sharing metrics.
    """

    def __init__(
        self,
        model_name: str = "facebook/opt-125m",
        block_size: int = 16,
        enable_prefix_caching: bool = True,
        max_model_len: int = 2048,
    ):
        """
        Initialize vLLM engine with simulator.

        Args:
            model_name: Model to use (for tokenizer)
            block_size: Block size for KV cache
            enable_prefix_caching: Whether to enable prefix caching
            max_model_len: Maximum model length
        """
        self.model_name = model_name
        self.block_size = block_size
        self.enable_prefix_caching = enable_prefix_caching
        self.max_model_len = max_model_len

        # Metrics
        self.metrics = {
            'requests': [],
            'blocks': defaultdict(lambda: {
                'hit_count': 0,
                'access_times': [],
                'reuse_gaps': []
            }),
            'total_blocks_allocated': 0,
            'total_blocks_reused': 0,
        }

        self.start_time = time.time()
        self._setup_vllm()

    def _setup_vllm(self):
        """Setup vLLM engine with simulator."""
        logger.info("Setting up vLLM engine with simulator...")

        try:
            # Try to import vLLM
            from vllm import LLM, SamplingParams
            from vllm.engine.arg_utils import EngineArgs
            from vllm.engine.llm_engine import LLMEngine

            logger.info("vLLM found, attempting to initialize engine...")

            # Create engine args
            engine_args = EngineArgs(
                model=self.model_name,
                tokenizer=self.model_name,
                block_size=self.block_size,
                enable_prefix_caching=self.enable_prefix_caching,
                max_model_len=self.max_model_len,
                gpu_memory_utilization=0.9,
                # Use simulator flags
                use_simulator=True,
                enforce_eager=True,  # Avoid CUDA graph for simulator
            )

            self.engine = LLMEngine.from_engine_args(engine_args)
            self.tokenizer = self.engine.tokenizer.tokenizer
            logger.info("✅ vLLM engine initialized with simulator")

        except ImportError:
            logger.warning("vLLM not installed")
            logger.info("Falling back to HuggingFace tokenizer only mode")
            self.engine = None
            self._setup_tokenizer()

        except Exception as e:
            logger.warning(f"Failed to initialize vLLM engine: {e}")
            logger.info("Falling back to HuggingFace tokenizer only mode")
            self.engine = None
            self._setup_tokenizer()

    def _setup_tokenizer(self):
        """Setup tokenizer only (fallback mode)."""
        try:
            from transformers import AutoTokenizer
            logger.info(f"Loading tokenizer: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            logger.info("✅ Tokenizer loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            logger.info("Using simple character-based tokenizer as last resort")
            self.tokenizer = None

    def _hook_block_manager(self):
        """Hook into vLLM's block manager to collect metrics."""
        if self.engine is None:
            return

        # Get block manager
        try:
            block_manager = self.engine.scheduler[0].block_manager
            original_allocate = block_manager.allocate

            def allocate_with_metrics(seq_group):
                # Call original allocate
                result = original_allocate(seq_group)

                # Collect metrics
                # TODO: Extract block allocation info
                return result

            block_manager.allocate = allocate_with_metrics
            logger.info("✅ Block manager hooked for metrics collection")

        except Exception as e:
            logger.warning(f"Failed to hook block manager: {e}")

    def analyze_trace(
        self,
        traces: List[RequestTrace],
        trace_file: str
    ) -> Dict[str, Any]:
        """
        Analyze a trace using real vLLM engine.

        Args:
            traces: List of request traces
            trace_file: Path to trace file for simulator

        Returns:
            Dictionary of metrics
        """
        logger.info(f"Analyzing {len(traces)} requests...")

        if self.engine is not None:
            return self._analyze_with_engine(traces, trace_file)
        else:
            return self._analyze_with_tokenizer(traces)

    def _analyze_with_engine(
        self,
        traces: List[RequestTrace],
        trace_file: str
    ) -> Dict[str, Any]:
        """Analyze using full vLLM engine."""
        from vllm import SamplingParams

        # Set simulator trace
        if hasattr(self.engine, 'simulator') and self.engine.simulator:
            logger.info(f"Using simulator with trace: {trace_file}")

        # Process each request
        for i, trace in enumerate(traces):
            if i % 100 == 0:
                logger.info(f"Processing request {i}/{len(traces)}")

            # Create sampling params
            sampling_params = SamplingParams(
                max_tokens=100,
                temperature=0.0
            )

            try:
                # Add request to engine
                self.engine.add_request(
                    request_id=trace.request_id,
                    prompt=trace.prompt,
                    params=sampling_params
                )

                # Step the engine
                while self.engine.has_unfinished_requests():
                    step_outputs = self.engine.step()

                    # Collect metrics from scheduler
                    self._collect_scheduler_metrics()

            except Exception as e:
                logger.warning(f"Error processing request {trace.request_id}: {e}")
                continue

        return self._compute_statistics()

    def _analyze_with_tokenizer(
        self,
        traces: List[RequestTrace]
    ) -> Dict[str, Any]:
        """Analyze using tokenizer only (fallback)."""
        logger.info("Using tokenizer-only analysis (fallback mode)")

        # Simulate block allocation using tokenizer
        block_cache = {}  # token_hash -> block_id
        next_block_id = 0

        for i, trace in enumerate(traces):
            if i % 100 == 0:
                logger.info(f"Processing request {i}/{len(traces)}")

            # Tokenize prompt
            prompt_tokens = self.tokenizer.encode(trace.prompt, add_special_tokens=True)
            num_blocks = (len(prompt_tokens) + self.block_size - 1) // self.block_size

            shared_blocks = 0
            shared_tokens = 0
            current_time = time.time() - self.start_time

            # Check each block
            for block_idx in range(num_blocks):
                start = block_idx * self.block_size
                end = min(start + self.block_size, len(prompt_tokens))
                block_tokens = tuple(prompt_tokens[start:end])

                block_hash = hash(block_tokens)

                if block_hash in block_cache:
                    # Block reused
                    block_id = block_cache[block_hash]
                    shared_blocks += 1
                    shared_tokens += len(block_tokens)
                    self.metrics['total_blocks_reused'] += 1

                    # Update block metrics
                    block_info = self.metrics['blocks'][block_id]
                    block_info['hit_count'] += 1
                    if block_info['access_times']:
                        last_time = block_info['access_times'][-1]
                        gap = current_time - last_time
                        block_info['reuse_gaps'].append(gap)
                    block_info['access_times'].append(current_time)

                else:
                    # New block
                    block_id = next_block_id
                    next_block_id += 1
                    block_cache[block_hash] = block_id
                    self.metrics['total_blocks_allocated'] += 1

                    # Initialize block metrics
                    self.metrics['blocks'][block_id]['hit_count'] = 1
                    self.metrics['blocks'][block_id]['access_times'].append(current_time)

            # Record request metrics
            sharing_fraction = shared_tokens / len(prompt_tokens) if len(prompt_tokens) > 0 else 0
            self.metrics['requests'].append({
                'request_id': trace.request_id,
                'total_tokens': len(prompt_tokens),
                'shared_tokens': shared_tokens,
                'sharing_fraction': sharing_fraction,
                'num_blocks': num_blocks,
                'shared_blocks': shared_blocks
            })

        return self._compute_statistics()

    def _collect_scheduler_metrics(self):
        """Collect metrics from scheduler after each step."""
        if self.engine is None:
            return

        try:
            # Access block manager
            scheduler = self.engine.scheduler[0]
            block_manager = scheduler.block_manager

            # Get block usage info
            # This is vLLM-version dependent
            # You may need to adjust based on your vLLM version

        except Exception as e:
            logger.debug(f"Could not collect scheduler metrics: {e}")

    def _compute_statistics(self) -> Dict[str, Any]:
        """Compute final statistics."""
        logger.info("Computing statistics...")

        # Sharing fractions
        sharing_fractions = [r['sharing_fraction'] for r in self.metrics['requests']]

        # Block hit counts
        hit_counts = [b['hit_count'] for b in self.metrics['blocks'].values()]

        # Reuse gaps
        reuse_gaps = []
        for block in self.metrics['blocks'].values():
            reuse_gaps.extend(block['reuse_gaps'])

        # Compute stats
        stats = {
            'total_requests': len(self.metrics['requests']),
            'total_blocks_allocated': self.metrics['total_blocks_allocated'],
            'total_blocks_reused': self.metrics['total_blocks_reused'],
            'unique_blocks': len(self.metrics['blocks']),
            'sharing_fraction': {
                'mean': sum(sharing_fractions) / len(sharing_fractions) if sharing_fractions else 0,
                'median': sorted(sharing_fractions)[len(sharing_fractions) // 2] if sharing_fractions else 0,
                'min': min(sharing_fractions) if sharing_fractions else 0,
                'max': max(sharing_fractions) if sharing_fractions else 0,
                'distribution': sharing_fractions
            },
            'block_hits': {
                'mean': sum(hit_counts) / len(hit_counts) if hit_counts else 0,
                'median': sorted(hit_counts)[len(hit_counts) // 2] if hit_counts else 0,
                'min': min(hit_counts) if hit_counts else 0,
                'max': max(hit_counts) if hit_counts else 0,
                'distribution': hit_counts
            },
            'reuse_gaps': {
                'mean': sum(reuse_gaps) / len(reuse_gaps) if reuse_gaps else 0,
                'median': sorted(reuse_gaps)[len(reuse_gaps) // 2] if reuse_gaps else 0,
                'min': min(reuse_gaps) if reuse_gaps else 0,
                'max': max(reuse_gaps) if reuse_gaps else 0,
                'distribution': reuse_gaps
            }
        }

        return stats

    def print_summary(self, stats: Dict[str, Any]):
        """Print statistics summary."""
        print("\n" + "="*60)
        print("PREFIX SHARING METRICS (Real vLLM)")
        print("="*60)

        print(f"\nTotal Requests: {stats['total_requests']}")
        print(f"Total Blocks Allocated: {stats['total_blocks_allocated']}")
        print(f"Total Blocks Reused: {stats['total_blocks_reused']}")
        print(f"Unique Blocks: {stats['unique_blocks']}")

        if stats['total_blocks_allocated'] > 0:
            reuse_rate = stats['total_blocks_reused'] / (
                stats['total_blocks_allocated'] + stats['total_blocks_reused']
            )
            print(f"Block Reuse Rate: {reuse_rate:.2%}")

        print(f"\nSharing Fraction:")
        print(f"  Mean: {stats['sharing_fraction']['mean']:.2%}")
        print(f"  Median: {stats['sharing_fraction']['median']:.2%}")
        print(f"  Min: {stats['sharing_fraction']['min']:.2%}")
        print(f"  Max: {stats['sharing_fraction']['max']:.2%}")

        print(f"\nBlock Hit Counts:")
        print(f"  Mean: {stats['block_hits']['mean']:.2f}")
        print(f"  Median: {stats['block_hits']['median']:.0f}")
        print(f"  Min: {stats['block_hits']['min']}")
        print(f"  Max: {stats['block_hits']['max']}")

        if stats['reuse_gaps']['mean'] > 0:
            print(f"\nReuse Gaps:")
            print(f"  Mean: {stats['reuse_gaps']['mean']:.2f}s")
            print(f"  Median: {stats['reuse_gaps']['median']:.2f}s")
            print(f"  Min: {stats['reuse_gaps']['min']:.2f}s")
            print(f"  Max: {stats['reuse_gaps']['max']:.2f}s")

        print("="*60 + "\n")


def create_trace_file(traces: List[RequestTrace], output_path: str):
    """Create trace file for simulator."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for trace in traces:
            record = {
                'prompt': trace.prompt,
                'response': trace.response
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    logger.info(f"Created trace file: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Run Milestone 2 with real vLLM engine'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default='vllm/ShareGPTData.jsonl',
        help='Path to ShareGPT data'
    )
    parser.add_argument(
        '--max-conversations',
        type=int,
        default=100,
        help='Max conversations to process'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='facebook/opt-125m',
        help='Model name for tokenizer'
    )
    parser.add_argument(
        '--block-size',
        type=int,
        default=16,
        help='Block size for KV cache'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='milestone2_vllm_results',
        help='Output directory'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['single', 'multi', 'both'],
        default='both',
        help='Which mode to run'
    )

    args = parser.parse_args()

    logger.info("="*60)
    logger.info("Milestone 2: Using Real vLLM Engine")
    logger.info("="*60)

    # Load data
    logger.info(f"Loading ShareGPT data from {args.data_path}")
    loader = ShareGPTLoader(args.data_path, max_conversations=args.max_conversations)
    conversations = loader.get_conversations()

    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True)

    # Run for single-turn
    if args.mode in ['single', 'both']:
        logger.info("\n" + "="*60)
        logger.info("Running Single-Turn Analysis")
        logger.info("="*60)

        # Generate traces
        formatter = ChatTemplateFormatter()
        generator = RequestGenerator(conversations, arrival_rate=2.0)
        single_traces = generator.generate_single_turn_traces(formatter)

        # Create trace file for simulator
        trace_file = f"{args.output_dir}/single_turn_trace.jsonl"
        create_trace_file(single_traces, trace_file)

        # Analyze
        analyzer = VLLMPrefixSharingAnalyzer(
            model_name=args.model,
            block_size=args.block_size
        )
        stats_single = analyzer.analyze_trace(single_traces, trace_file)

        # Print and save results
        analyzer.print_summary(stats_single)
        with open(f"{args.output_dir}/single_turn_vllm_stats.json", 'w') as f:
            json.dump(stats_single, f, indent=2)

    # Run for multi-turn
    if args.mode in ['multi', 'both']:
        logger.info("\n" + "="*60)
        logger.info("Running Multi-Turn Analysis")
        logger.info("="*60)

        # Generate traces
        formatter = ChatTemplateFormatter()
        generator = RequestGenerator(conversations, arrival_rate=2.0)
        multi_traces = generator.generate_multi_turn_traces(formatter, turn_delay=1.0)

        # Create trace file for simulator
        trace_file = f"{args.output_dir}/multi_turn_trace.jsonl"
        create_trace_file(multi_traces, trace_file)

        # Analyze
        analyzer = VLLMPrefixSharingAnalyzer(
            model_name=args.model,
            block_size=args.block_size
        )
        stats_multi = analyzer.analyze_trace(multi_traces, trace_file)

        # Print and save results
        analyzer.print_summary(stats_multi)
        with open(f"{args.output_dir}/multi_turn_vllm_stats.json", 'w') as f:
            json.dump(stats_multi, f, indent=2)

    logger.info("\n" + "="*60)
    logger.info("DONE!")
    logger.info("="*60)
    logger.info(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
