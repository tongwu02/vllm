import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from vllm.sequence import Sequence, SequenceStatus


@dataclass
class _SimState:
    output_tokens: List[int]
    cursor: int = 0
    eos_token_id: int = 2


class Simulator:
    """A lightweight simulator that replays pre-canned outputs per request.

    This is a minimal implementation intended for Milestone 1. It avoids
    running any real model compute while letting the rest of the engine
    (scheduler, block manager, request lifecycle) behave normally.
    """

    def __init__(self, trace_path: Optional[str] = None,
                 default_output_len: int = 8) -> None:
        self.default_output_len = default_output_len
        self._states: Dict[str, _SimState] = {}
        self._trace: Dict[str, List[int]] = {}
        if trace_path:
            try:
                with open(trace_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except Exception:
                            continue
                        req_id = obj.get("request_id")
                        out = obj.get("output_token_ids")
                        if req_id and isinstance(out, list):
                            self._trace[req_id] = [int(x) for x in out]
            except FileNotFoundError:
                # Best-effort loading; proceed without a trace.
                pass

    def start_request(self, request_id: str, prompt: Sequence,
                      max_new_tokens: Optional[int] = None) -> None:
        """Prepare simulation state for a new request."""
        if request_id in self._states:
            return

        traced = self._trace.get(request_id)
        if traced:
            output_tokens = traced
        else:
            # Deterministic dummy outputs based on prompt length.
            base = (len(prompt.prompt_token_ids) or 1) * 10
            length = max_new_tokens or self.default_output_len
            output_tokens = [base + i for i in range(length)]

        eos_token_id = prompt.eos_token_id
        self._states[request_id] = _SimState(output_tokens=output_tokens,
                                             cursor=0,
                                             eos_token_id=eos_token_id)

    def next_token(self,
                   request_id: str) -> Tuple[int, bool]:
        """Return next token and whether the sequence is finished."""
        state = self._states.get(request_id)
        if state is None:
            # Unknown request: immediately finish.
            return (2, True)

        if state.cursor >= len(state.output_tokens):
            return (state.eos_token_id, True)

        token = state.output_tokens[state.cursor]
        state.cursor += 1
        finished = state.cursor >= len(state.output_tokens)
        return (token, finished)

    def finish_request(self, request_id: str, seq: Sequence) -> None:
        seq.status = SequenceStatus.FINISHED_STOPPED
        self._states.pop(request_id, None)
