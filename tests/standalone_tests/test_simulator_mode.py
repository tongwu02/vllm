import json
from pathlib import Path
from typing import List

import pytest

from vllm import EngineArgs, SamplingParams
from vllm.engine.llm_engine import LLMEngine
from vllm.utils import Device


@pytest.fixture()
def trace_path(tmp_path: Path) -> Path:
    trace_file = tmp_path / "trace.jsonl"
    records = [{
        "prompt": "Hello",
        "response": "Hi there!",
    }, {
        "prompt": "What is vLLM?",
        "response": "vLLM is a fast inference engine.",
    }, {
        "prompt": "Prefix cache demo",
        "response": "alpha beta",
    }, {
        "prompt": "Prefix cache demo extended",
        "response": "alpha beta gamma",
    }]
    trace_file.write_text("\n".join(json.dumps(r) for r in records),
                          encoding="utf-8")
    return trace_file


@pytest.fixture()
def simulator_env(monkeypatch: pytest.MonkeyPatch, trace_path: Path):
    monkeypatch.setenv("VLLM_SIM_TRACE_PATH", str(trace_path))
    yield
    monkeypatch.delenv("VLLM_SIM_TRACE_PATH", raising=False)


def _make_engine() -> LLMEngine:
    args = EngineArgs(
        model="meta-llama/Llama-3.2-1B-Instruct",
        tokenizer="meta-llama/Llama-3.2-1B-Instruct",
        device="cpu",
        max_model_len=128,
        max_num_seqs=4,
        block_size=8,
        enable_prefix_caching=True,
    )
    return LLMEngine.from_engine_args(args)


def _drain_engine(engine: LLMEngine, max_steps: int = 50) -> List:
    outputs: List = []
    for _ in range(max_steps):
        outputs.extend(engine.step())
        if not engine.has_unfinished_requests():
            break
    return outputs


def _final_text(outputs: List, request_id: str) -> str:
    matches = [o for o in outputs if o.request_id == request_id]
    assert matches, f"No outputs produced for {request_id}"
    return matches[-1].outputs[0].text


def test_simulator_replays_trace_entry(simulator_env):
    engine = _make_engine()
    engine.add_request("hello",
                       prompt="Hello",
                       params=SamplingParams(max_tokens=6))

    produced = _drain_engine(engine)
    assert produced

    reply = _final_text(produced, "hello").strip()
    assert reply == "Hi there!"


def test_simulator_handles_multiple_prompts(simulator_env):
    engine = _make_engine()
    engine.add_request("hello",
                       prompt="Hello",
                       params=SamplingParams(max_tokens=6))
    engine.add_request("vllm",
                       prompt="What is vLLM?",
                       params=SamplingParams(max_tokens=10))

    produced = _drain_engine(engine)
    assert len(produced) >= 2

    hello_text = _final_text(produced, "hello").strip()
    vllm_text = _final_text(produced, "vllm").strip()
    assert hello_text == "Hi there!"
    assert vllm_text == "vLLM is a fast inference engine."


def test_simulator_reports_prefix_cache_hits(simulator_env):
    engine = _make_engine()
    engine.add_request("prefix-base",
                       prompt="Prefix cache demo",
                       params=SamplingParams(max_tokens=4))
    engine.add_request("prefix-extended",
                       prompt="Prefix cache demo extended",
                       params=SamplingParams(max_tokens=5))

    produced = _drain_engine(engine)
    assert len(produced) >= 2

    assert _final_text(produced, "prefix-base").strip() == "alpha beta"
    assert (_final_text(produced,
                        "prefix-extended").strip() == "alpha beta gamma")

    hit_rate = engine.scheduler[0].get_prefix_cache_hit_rate(Device.CPU)
    assert hit_rate is not None
    assert 0.0 < hit_rate <= 1.0, f"unexpected hit rate {hit_rate}"
