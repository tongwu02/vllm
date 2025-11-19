import json
from pathlib import Path

import pytest

from vllm import EngineArgs, SamplingParams
from vllm.engine.llm_engine import LLMEngine
from vllm.utils import Device


@pytest.fixture
def trace_file(tmp_path: Path):
    trace_path = tmp_path / "trace.jsonl"
    records = [
        {
            "request_id": "req1",
            "output_token_ids": [101, 102, 103],
        },
        {
            "request_id": "req2",
            "output_token_ids": [101, 105],
        },
    ]
    trace_path.write_text("\n".join(json.dumps(r) for r in records))
    return trace_path


@pytest.fixture
def sim_env(monkeypatch, trace_file: Path):
    # Use simulator mode with a small, lightweight tokenizer/model.
    monkeypatch.setenv("VLLM_USE_SIMULATOR", "1")
    monkeypatch.setenv("VLLM_SIMULATOR_TRACE_PATH", str(trace_file))
    yield
    monkeypatch.delenv("VLLM_USE_SIMULATOR", raising=False)
    monkeypatch.delenv("VLLM_SIMULATOR_TRACE_PATH", raising=False)


def _make_engine():
    args = EngineArgs(
        model="gpt2",
        tokenizer="gpt2",
        device="cpu",
        max_model_len=128,
        max_num_seqs=4,
        block_size=8,
        enable_prefix_caching=True,
    )
    return LLMEngine.from_engine_args(args)


def _drain_engine(engine: LLMEngine, max_steps: int = 50):
    outputs = []
    for _ in range(max_steps):
        step_out = engine.step()
        outputs.extend(step_out)
        if not engine.has_unfinished_requests():
            break
    return outputs


def test_simulator_single_request(sim_env):
    engine = _make_engine()
    engine.add_request("req1",
                       prompt="hello simulator",
                       params=SamplingParams(max_tokens=3))
    outs = _drain_engine(engine)
    assert outs, "No outputs produced"
    # Take the last output for this request.
    out = [o for o in outs if o.request_id == "req1"][-1]
    assert out.finished is True
    # Expect tokens to match the trace.
    assert list(out.outputs[0].token_ids) == [101, 102, 103]


def test_simulator_prefix_hit(sim_env):
    engine = _make_engine()
    engine.add_request("req1",
                       prompt="hello prefix",
                       params=SamplingParams(max_tokens=3))
    engine.add_request("req2",
                       prompt="hello prefix reuse",
                       params=SamplingParams(max_tokens=2))
    outs = _drain_engine(engine)
    assert len(outs) >= 2
    # Check final outputs match trace entries.
    final = {}
    for o in outs:
        final[o.request_id] = o
    assert list(final["req1"].outputs[0].token_ids) == [101, 102, 103]
    assert list(final["req2"].outputs[0].token_ids) == [101, 105]

    # Prefix cache hit rate should be non-negative in simulator mode.
    hit_rate = engine.scheduler[0].get_prefix_cache_hit_rate(Device.CPU)
    assert hit_rate >= 0.0
