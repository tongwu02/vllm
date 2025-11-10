# vllm/vllm/sim/test_simulator_unit_standalone.py
import os, sys, types, importlib.util
from pathlib import Path
from typing import Any

# ---- 1) 注入 vllm.outputs 的最小替身（保持不变） ----
class SequenceOutput:
    def __init__(self, index=0, text="", token_ids=None, finish_reason=None,
                 cumulative_logprob=None, logprobs=None):
        self.index = index
        self.text = text
        self.token_ids = token_ids or []
        self.finish_reason = finish_reason

class RequestOutput:
    def __init__(self, request_id, prompt, prompt_token_ids, outputs, finished,
                 metrics=None, usage=None):
        self.request_id = request_id
        self.prompt = prompt
        self.prompt_token_ids = prompt_token_ids or []
        self.outputs = outputs or []
        self.finished = finished
        self.metrics = metrics
        self.usage = usage

outputs_mod = types.ModuleType("vllm.outputs")
outputs_mod.SequenceOutput = SequenceOutput
outputs_mod.RequestOutput = RequestOutput
sys.modules["vllm.outputs"] = outputs_mod

# ---- 2) 直接按“同目录”加载 simulator.py，避免触发 vllm/__init__.py ----
HERE = Path(__file__).resolve()
SIM_PATH = HERE.with_name("simulator.py")  # ← 关键修改：同目录的 simulator.py
spec = importlib.util.spec_from_file_location("simulator", str(SIM_PATH))
sim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sim)
Simulator = sim.Simulator

# ---- 3) 轻量 tokenizer 与伪调度条目（保持不变） ----
class ToyTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False):
        return [ord(c) for c in text]
    def decode(self, ids, skip_special_tokens: bool = True):
        return "".join(chr(i) for i in ids)

class FakeRunItem:
    def __init__(self, request_id: str):
        self.request_id = request_id

class Params:
    def __init__(self, max_tokens: int = 16, stop=None):
        self.max_tokens = max_tokens
        self.stop = stop or []

def print_step(tag: str, outs):
    print(f"\n[{tag}] outputs={len(outs)}")
    for o in outs:
        rid = getattr(o, "request_id", None)
        finished = getattr(o, "finished", None)
        outputs = getattr(o, "outputs", [])
        text = outputs[0].text if outputs else ""
        fr = outputs[0].finish_reason if outputs else None
        print(f"- id={rid} text='{text}' finished={finished} reason={fr}")

def ensure_trace(path: Path):
    if not path.exists():
        path.write_text(
            '{"prompt":"Hello","response":" world!"}\n'
            '{"prompt":"Hi","response":" there"}\n'
            '{"prompt":"StopCase","response":" hello STOP and more"}\n',
            encoding="utf-8"
        )
        return
    content = path.read_text(encoding="utf-8")
    if '"prompt":"StopCase"' not in content and '"prompt": "StopCase"' not in content:
        with path.open("a", encoding="utf-8") as f:
            f.write('{"prompt":"StopCase","response":" hello STOP and more"}\n')

def main():
    # 准备最小 trace（包含 StopCase）
    trace_path = HERE.with_name("trace.jsonl")
    ensure_trace(trace_path)

    sim = Simulator(trace_path=str(trace_path), tokenizer=ToyTokenizer())
    
    # 新增三个请求
    sim.on_add_request("r1", "Hello",     Params(max_tokens=10))
    sim.on_add_request("r2", "Hi",        Params(max_tokens=3))
    sim.on_add_request("r3", "StopCase",  Params(max_tokens=50, stop=["STOP"]))

    # prefill
    sim.simulate_prefill([FakeRunItem("r1"), FakeRunItem("r2"), FakeRunItem("r3")])
    print("[prefill] done")

    # 多轮 decode：三条请求都参与
    step = 1
    while step < 50:
        outs = sim.simulate_decode([FakeRunItem("r1"), FakeRunItem("r2"), FakeRunItem("r3")])
        print_step(f"decode step {step}", outs)

        # 如果这三条都已完成（outs 里可能逐步变少，给个稳妥判断）
        # 简单做法：当 r1/r2/r3 都不再出现在输出里，说明都 finished
        seen_ids = {getattr(o, "request_id", "") for o in outs}
        if not seen_ids:  # 全都 finished 后 simulate_decode 不再返回对应输出
            break

        step += 1

if __name__ == "__main__":
    main()
