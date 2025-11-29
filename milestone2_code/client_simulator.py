#!/usr/bin/env python3
"""
Client Simulator for Milestone 2 Task 1
模拟客户端发送请求到vLLM引擎
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
    """单个请求"""
    request_id: str
    prompt: str
    response: str  # 来自trace的ground truth
    arrival_time: float
    conversation_id: str
    turn_index: int
    prompt_token_ids: List[int] = field(default_factory=list)
    max_tokens: int = 64


@dataclass
class RequestResult:
    """请求的结果"""
    request_id: str
    conversation_id: str
    turn_index: int
    prompt_length: int  # prompt token数
    response_length: int  # 实际生成的token数
    hit_blocks: int = 0  # 命中的cache blocks数
    total_blocks: int = 0  # 总的blocks数
    hit_rate: float = 0.0  # 该请求的cache命中率
    start_time: float = 0.0
    end_time: float = 0.0
    latency: float = 0.0  # 秒


class ClientSimulator:
    """
    客户端模拟器

    根据project.pdf Task 1要求：
    - Timing: 使用泊松分布模拟请求到达时间
    - Chat templates: 正确格式化prompt
    """

    def __init__(
        self,
        trace_path: Path,
        tokenizer,
        arrival_rate: float = 1.0,  # 平均每秒请求数
        use_trace_timing: bool = False,  # 是否使用trace中的时间
    ):
        """
        Args:
            trace_path: 预处理后的trace文件路径
            tokenizer: HuggingFace tokenizer
            arrival_rate: 泊松分布的lambda参数（请求/秒）
            use_trace_timing: 如果True且trace有时间戳，使用trace时间
        """
        self.trace_path = trace_path
        self.tokenizer = tokenizer
        self.arrival_rate = arrival_rate
        self.use_trace_timing = use_trace_timing
        self.requests: List[Request] = []
        self.results: Dict[str, RequestResult] = {}

        self._load_trace()

    def _load_trace(self):
        """加载trace文件"""
        logger.info(f"Loading trace from {self.trace_path}")

        with open(self.trace_path, 'r', encoding='utf-8') as f:
            entries = [json.loads(line) for line in f if line.strip()]

        # 生成请求到达时间
        current_time = 0.0
        for i, entry in enumerate(entries):
            # 使用泊松过程生成到达时间间隔
            if self.use_trace_timing and 'timestamp' in entry:
                arrival_time = entry['timestamp']
            elif 'arrival_time' in entry:
                # 如果trace中包含arrival_time，直接使用
                arrival_time = entry['arrival_time']
            else:
                # 泊松分布：间隔时间服从指数分布
                interval = np.random.exponential(1.0 / self.arrival_rate)
                current_time += interval
                arrival_time = current_time

            prompt = entry['prompt']
            response = entry['response']

            # Tokenize prompt来计算长度
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
        将请求发送到vLLM引擎

        Args:
            engine: LLMEngine实例
            params_override: 覆盖默认的SamplingParams
        """
        from vllm import SamplingParams

        logger.info(f"Sending {len(self.requests)} requests to engine...")

        start_time = time.time()

        for req in self.requests:
            # 构造SamplingParams
            params_dict = {
                'max_tokens': req.max_tokens,
                'temperature': 0.0,  # 确定性生成
                'top_p': 1.0,
            }
            if params_override:
                params_dict.update(params_override)

            params = SamplingParams(**params_dict)

            # 添加请求到引擎
            engine.add_request(
                request_id=req.request_id,
                prompt=req.prompt,
                params=params,
            )

            # 记录开始时间
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
        Conversation-by-conversation processing (改进的选项A)
        按conversation_id分组，每个conversation内部按turn_index顺序处理

        这样确保同一个conversation的后续turns能够复用前面turns的cached blocks

        Args:
            engine: LLMEngine实例
            params_override: 覆盖默认的SamplingParams
            max_steps_per_turn: 每个turn的最大步数
        """
        from vllm import SamplingParams

        # 按conversation_id分组，每个conversation内部按turn_index排序
        conversations = defaultdict(list)
        for req in self.requests:
            conversations[req.conversation_id].append(req)

        # 对每个conversation内部按turn_index排序
        for conv_id in conversations:
            conversations[conv_id].sort(key=lambda r: r.turn_index)

        logger.info("=" * 80)
        logger.info("Conversation-by-Conversation Sequential Processing")
        logger.info("=" * 80)
        logger.info(f"Total conversations: {len(conversations)}")

        # 统计turn分布
        turn_distribution = defaultdict(int)
        for conv_id, conv_requests in conversations.items():
            num_turns = len(conv_requests)
            turn_distribution[num_turns] += 1

        logger.info(f"Turn distribution:")
        for num_turns in sorted(turn_distribution.keys()):
            logger.info(f"  {num_turns} turns: {turn_distribution[num_turns]} conversations")

        # 按conversation顺序处理
        conv_count = 0
        for conv_id in sorted(conversations.keys()):
            conv_requests = conversations[conv_id]
            conv_count += 1

            logger.info("")
            logger.info(f"【Conversation {conv_count}/{len(conversations)}】{conv_id} - {len(conv_requests)} turns")

            # 处理这个conversation的每个turn
            for turn_req in conv_requests:
                logger.info(f"  Turn {turn_req.turn_index}: Processing request {turn_req.request_id}...")

                # 构造SamplingParams
                params_dict = {
                    'max_tokens': turn_req.max_tokens,
                    'temperature': 0.0,
                    'top_p': 1.0,
                }
                if params_override:
                    params_dict.update(params_override)

                params = SamplingParams(**params_dict)

                # 添加请求到引擎
                start_time = time.time()
                engine.add_request(
                    request_id=turn_req.request_id,
                    prompt=turn_req.prompt,
                    params=params,
                )

                # 记录开始时间
                self.results[turn_req.request_id] = RequestResult(
                    request_id=turn_req.request_id,
                    conversation_id=turn_req.conversation_id,
                    turn_index=turn_req.turn_index,
                    prompt_length=len(turn_req.prompt_token_ids),
                    response_length=0,
                    start_time=time.time(),
                )

                # 运行引擎直到这个turn完成
                step = 0
                while step < max_steps_per_turn:
                    outputs = engine.step()

                    # 更新结果
                    for output in outputs:
                        if output.request_id == turn_req.request_id:
                            result = self.results[turn_req.request_id]
                            result.response_length = len(output.outputs[0].token_ids)

                            if output.finished:
                                result.end_time = time.time()
                                result.latency = result.end_time - result.start_time
                                logger.info(f"    ✓ Turn {turn_req.turn_index} completed in {step + 1} steps ({result.latency:.2f}s)")
                                break

                    # 检查这个turn是否完成
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
        Turn-by-turn sequential processing (原始选项A - 按turn_index分组)

        注意：这个方法适合测试单个conversation，对于多个conversations应使用
        send_requests_conversation_by_conversation()

        Args:
            engine: LLMEngine实例
            params_override: 覆盖默认的SamplingParams
            max_steps_per_turn: 每个turn的最大步数
        """
        from vllm import SamplingParams

        # 按turn_index分组
        turns_groups = defaultdict(list)
        for req in self.requests:
            turns_groups[req.turn_index].append(req)

        logger.info("=" * 80)
        logger.info("Turn-by-Turn Sequential Processing (by turn_index)")
        logger.info("=" * 80)
        logger.info(f"Total turns: {len(turns_groups)}")
        for turn_idx in sorted(turns_groups.keys()):
            logger.info(f"  Turn {turn_idx}: {len(turns_groups[turn_idx])} requests")

        # 按turn_index顺序处理
        for turn_idx in sorted(turns_groups.keys()):
            turn_requests = turns_groups[turn_idx]

            logger.info("")
            logger.info(f"【Turn {turn_idx}】Processing {len(turn_requests)} requests...")

            # 发送这个turn的所有requests
            start_time = time.time()
            for req in turn_requests:
                # 构造SamplingParams
                params_dict = {
                    'max_tokens': req.max_tokens,
                    'temperature': 0.0,
                    'top_p': 1.0,
                }
                if params_override:
                    params_dict.update(params_override)

                params = SamplingParams(**params_dict)

                # 添加请求到引擎
                engine.add_request(
                    request_id=req.request_id,
                    prompt=req.prompt,
                    params=params,
                )

                # 记录开始时间
                self.results[req.request_id] = RequestResult(
                    request_id=req.request_id,
                    conversation_id=req.conversation_id,
                    turn_index=req.turn_index,
                    prompt_length=len(req.prompt_token_ids),
                    response_length=0,
                    start_time=time.time(),
                )

            logger.info(f"  Submitted {len(turn_requests)} requests in {time.time() - start_time:.2f}s")

            # 运行引擎直到这个turn的所有请求完成
            logger.info(f"  Running engine until Turn {turn_idx} completes...")
            step = 0
            turn_request_ids = set(req.request_id for req in turn_requests)

            while step < max_steps_per_turn:
                outputs = engine.step()

                # 更新结果
                for output in outputs:
                    if output.request_id in self.results:
                        result = self.results[output.request_id]
                        result.response_length = len(output.outputs[0].token_ids)

                        if output.finished:
                            result.end_time = time.time()
                            result.latency = result.end_time - result.start_time

                # 检查这个turn的所有请求是否完成
                turn_completed = all(
                    self.results[req_id].end_time > 0
                    for req_id in turn_request_ids
                )

                if turn_completed:
                    logger.info(f"  ✓ Turn {turn_idx} completed in {step + 1} steps")
                    break

                step += 1

                # 每100步输出进度
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
        运行引擎直到所有请求完成

        Args:
            engine: LLMEngine实例
            max_steps: 最大步数
        """
        logger.info("Running engine until all requests complete...")

        step = 0
        while step < max_steps:
            outputs = engine.step()

            # 更新结果
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

            # 每100步输出进度
            if step % 100 == 0:
                pending = len([r for r in self.results.values() if r.end_time == 0])
                logger.info(f"  Step {step}: {pending} requests pending")

        if step >= max_steps:
            logger.warning(f"Reached max steps ({max_steps}), some requests may not be finished")

    def collect_prefix_cache_stats(self, engine):
        """
        收集prefix cache统计信息

        根据project.pdf Task 2要求收集：
        - The fraction of each request that benefits from prefix sharing
        - The number of hits per cache block
        - The time gap between reuses of each cache block
        """
        from vllm.utils import Device

        logger.info("Collecting prefix cache statistics...")

        # 使用我们修正后的hit rate tracker
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

        # 收集cache block统计（Task 2要求）
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
        生成实验报告

        Returns:
            统计信息字典
        """
        logger.info("Generating report...")

        completed_results = [r for r in self.results.values() if r.end_time > 0]

        if not completed_results:
            logger.warning("No completed requests to report")
            return {}

        # 基本统计
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

        # 按conversation分组统计
        conv_groups = defaultdict(list)
        for result in completed_results:
            conv_groups[result.conversation_id].append(result)

        stats['num_conversations'] = len(conv_groups)
        stats['avg_turns_per_conversation'] = np.mean([len(turns) for turns in conv_groups.values()])

        # 输出报告
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

        # 保存到文件
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(stats, f, indent=2)
            logger.info(f"Report saved to {output_path}")

        return stats


def main():
    """示例用法"""
    print("""
Client Simulator for Milestone 2

Usage:
    from milestone2_code.client_simulator import ClientSimulator
    from vllm import EngineArgs
    from vllm.engine.llm_engine import LLMEngine
    from transformers import AutoTokenizer

    # 1. 准备trace
    # 使用trace_preprocessor.py预处理ShareGPT数据

    # 2. 创建simulator
    tokenizer = AutoTokenizer.from_pretrained("exported_models/Llama-3.2-1B-Instruct")
    simulator = ClientSimulator(
        trace_path=Path("milestone2_code/sharegpt_trace.jsonl"),
        tokenizer=tokenizer,
        arrival_rate=1.0,
    )

    # 3. 创建引擎
    args = EngineArgs(
        model="exported_models/Llama-3.2-1B-Instruct",
        tokenizer="exported_models/Llama-3.2-1B-Instruct",
        device="cpu",
        enable_prefix_caching=True,
    )
    engine = LLMEngine.from_engine_args(args)

    # 4. 运行实验
    simulator.send_requests_to_engine(engine)
    simulator.run_engine_until_complete(engine)

    # 5. 收集统计
    simulator.collect_prefix_cache_stats(engine)
    stats = simulator.generate_report()
    """)


if __name__ == "__main__":
    main()
