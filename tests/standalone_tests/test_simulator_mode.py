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
        "prompt": "You are a helpful assistant in recognizes the content of tables in markdown format.",
        "response": "alpha beta",
    }, {
        "prompt": "You are a helpful assistant in recognizes the content of tables in markdown format. This is an extended prompt.",
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
        model="exported_models/Llama-3.2-1B-Instruct",
        tokenizer="exported_models/Llama-3.2-1B-Instruct",
        device="cpu",
        max_model_len=128,
        max_num_seqs=4,
        block_size=2,
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
    common_prefix = (
        "You are a helpful assistant that analyzes long structured data. "
        "Below is a detailed list of 20 fruits and vegetables that appear in the dataset: "
        "apples, bananas, oranges, grapes, melons, peaches, pears, plums, cherries, "
        "strawberries, blueberries, raspberries, pineapples, kiwis, mangoes, papayas, "
        "figs, dates, apricots, watermelons. "
        "Please read all items carefully and answer the following query: "
    )
    prompt_big1 = common_prefix + "Summarize the nutritional benefits."
    prompt_big2 = common_prefix + "List three items that are high in fiber."
    prompt_big3 = common_prefix + "Which items contain the most vitamin C?"
    prompt_big4 = common_prefix + "Group the items into fruits and non-fruits."
    prompt_big5 = common_prefix + "Identify which items are tropical."
    prompt_big6 = common_prefix + "Provide a short story using at least 5 items from the list."
    prompt_big7 = common_prefix + "Explain which items would be suitable for a smoothie recipe."
    prompt_big8 = common_prefix + "Rank the items by estimated sweetness level."
    engine.add_request("prefix-base",
                       prompt="You are a helpful assistant in recognizes the content of tables in markdown format.",
                       params=SamplingParams(max_tokens=4))
    engine.add_request("prefix-extended",
                       prompt="You are a helpful assistant in recognizes the content of tables in markdown format. This is an extended prompt.",
                       params=SamplingParams(max_tokens=5))
    engine.add_request("big1", prompt=prompt_big1, params=SamplingParams(max_tokens=5))
    engine.add_request("big2", prompt=prompt_big2, params=SamplingParams(max_tokens=5))
    engine.add_request("big3", prompt=prompt_big3, params=SamplingParams(max_tokens=5))
    engine.add_request("big4", prompt=prompt_big4, params=SamplingParams(max_tokens=5))
    engine.add_request("big5", prompt=prompt_big5, params=SamplingParams(max_tokens=5))
    engine.add_request("big6", prompt=prompt_big6, params=SamplingParams(max_tokens=5))
    engine.add_request("big7", prompt=prompt_big7, params=SamplingParams(max_tokens=5))
    engine.add_request("big8", prompt=prompt_big8, params=SamplingParams(max_tokens=5))

    produced = _drain_engine(engine)
    assert len(produced) >= 2

    # assert _final_text(produced, "prefix-base").strip() == "alpha beta"
    # assert (_final_text(produced,
    #                     "prefix-extended").strip() == "alpha beta gamma")

    hit_rate = engine.scheduler[0].get_prefix_cache_hit_rate(Device.GPU)
    print("hit_rate: ", hit_rate)
    assert hit_rate is not None
    assert 0.0 < hit_rate <= 1.0, f"unexpected hit rate {hit_rate}"
