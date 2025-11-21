import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from vllm.executor.executor_base import ExecutorBase
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import (CompletionSequenceGroupOutput, ExecuteModelRequest,
                           Logprob, SequenceGroupMetadata, SequenceOutput)
from vllm.transformers_utils.tokenizer import AnyTokenizer

logger = logging.getLogger(__name__)


@dataclass
class _RequestState:
    prompt_key: Tuple[int, ...]
    response_ids: List[int]
    cursor: int = 0
    generated_tokens: int = 0
    prefill_done: bool = False
    finished: bool = False


class Simulator(ExecutorBase):
    """Executor that replays pre-recorded responses from a JSONL trace file."""

    uses_ray: bool = False

    def __init__(self,
                 vllm_config,
                 tokenizer: AnyTokenizer,
                 trace_path: str) -> None:
        self._tokenizer = tokenizer
        self._trace_path = trace_path
        self._trace_map: Dict[Tuple[int, ...], List[int]] = {}
        self._states: Dict[str, _RequestState] = {}
        self._eos_token_id: Optional[int] = getattr(tokenizer, "eos_token_id",
                                                    None)
        self._fallback_token_id: int = (self._eos_token_id
                                        if self._eos_token_id is not None else
                                        0)
        super().__init__(vllm_config=vllm_config)

    def _init_executor(self) -> None:
        if not self._trace_path:
            raise ValueError("Trace path for simulator is not provided.")
        self._load_trace(self._trace_path)
        logger.info("[Simulator] Loaded %d prompt-response pairs.",
                    len(self._trace_map))

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        # Provide generous defaults so that scheduler can make progress even
        # though no actual KV cache exists.
        num_gpu_blocks = self.cache_config.num_gpu_blocks or 8192
        num_cpu_blocks = self.cache_config.num_cpu_blocks or 0
        return num_gpu_blocks, num_cpu_blocks

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        # Nothing to initialize for the simulator.
        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

    def execute_model(
            self, execute_model_req: ExecuteModelRequest
    ) -> Optional[List[SamplerOutput]]:
        metadata_list = execute_model_req.seq_group_metadata_list or []
        if not metadata_list:
            return []

        group_outputs: List[CompletionSequenceGroupOutput] = []
        for metadata in metadata_list:
            group_outputs.append(self._simulate_seq_group(metadata))

        return [SamplerOutput(outputs=group_outputs)]

    def add_lora(self, lora_request) -> bool:  # type: ignore[override]
        raise NotImplementedError("LoRA is not supported in simulator mode.")

    def remove_lora(self, lora_id: int) -> bool:
        raise NotImplementedError("LoRA is not supported in simulator mode.")

    def pin_lora(self, lora_id: int) -> bool:
        raise NotImplementedError("LoRA is not supported in simulator mode.")

    def list_loras(self):
        return set()

    def add_prompt_adapter(self, prompt_adapter_request) -> bool:  # type: ignore[override]  # noqa: E501
        raise NotImplementedError(
            "Prompt adapters are not supported in simulator mode.")

    def remove_prompt_adapter(self, prompt_adapter_id: int) -> bool:
        raise NotImplementedError(
            "Prompt adapters are not supported in simulator mode.")

    def pin_prompt_adapter(self, prompt_adapter_id: int) -> bool:
        raise NotImplementedError(
            "Prompt adapters are not supported in simulator mode.")

    def list_prompt_adapters(self):
        return set()

    def check_health(self) -> None:
        return

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #

    def _load_trace(self, trace_path: str) -> None:
        with open(trace_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                prompt = str(record.get("prompt", ""))
                response = str(record.get("response", ""))

                prompt_ids = tuple(self._encode(prompt, include_bos=True))
                response_ids = self._encode(response, include_bos=False)
                if self._eos_token_id is not None and (
                        not response_ids
                        or response_ids[-1] != self._eos_token_id):
                    response_ids.append(self._eos_token_id)

                if prompt_ids not in self._trace_map:
                    self._trace_map[prompt_ids] = response_ids
                else:
                    logger.debug("[Simulator] Duplicate prompt skipped.")

    def _encode(self, text: str, include_bos: bool) -> List[int]:
        if hasattr(self._tokenizer, "encode"):
            try:
                return self._tokenizer.encode(
                    text, add_special_tokens=include_bos)
            except TypeError:
                return self._tokenizer.encode(text)
        return []

    def _decode_token(self, token_id: int) -> Optional[str]:
        if hasattr(self._tokenizer, "decode"):
            try:
                return self._tokenizer.decode([token_id])
            except Exception:  # pragma: no cover - best effort decode
                return None
        return None

    def _simulate_seq_group(
            self,
            metadata: SequenceGroupMetadata) -> CompletionSequenceGroupOutput:
        state = self._ensure_state(metadata)

        token_id = self._next_token(metadata, state)
        seq_id = metadata.get_first_seq_id()
        decoded = self._decode_token(token_id)
        logprob = Logprob(logprob=0.0, rank=0, decoded_token=decoded)
        seq_output = SequenceOutput(parent_seq_id=seq_id,
                                    output_token=token_id,
                                    logprobs={token_id: logprob})
        return CompletionSequenceGroupOutput(samples=[seq_output],
                                             prompt_logprobs=None)

    def _ensure_state(self, metadata: SequenceGroupMetadata) -> _RequestState:
        request_id = metadata.request_id
        if request_id in self._states:
            return self._states[request_id]

        prompt_ids = self._extract_prompt_ids(metadata)
        prompt_key = tuple(prompt_ids)
        response_ids = list(self._trace_map.get(prompt_key, []))
        if not response_ids:
            logger.warning(
                "[Simulator] Prompt not found in trace for request_id=%s. "
                "Falling back to EOS token only.", request_id)
            response_ids = (
                [self._fallback_token_id]
                if self._fallback_token_id is not None else [])

        state = _RequestState(prompt_key=prompt_key, response_ids=response_ids)
        self._states[request_id] = state
        return state

    def _extract_prompt_ids(self, metadata: SequenceGroupMetadata) -> List[int]:
        seq_data = metadata.seq_data or {}
        if not seq_data:
            return []

        first_seq_data = next(iter(seq_data.values()))
        return list(first_seq_data.prompt_token_ids)

    def _next_token(self, metadata: SequenceGroupMetadata,
                    state: _RequestState) -> int:
        if state.finished:
            return self._fallback_token_id

        sampling_params = metadata.sampling_params
        max_tokens = getattr(sampling_params, "max_tokens",
                             None) if sampling_params else None
        if max_tokens is not None and state.generated_tokens >= max_tokens:
            state.finished = True
            return self._fallback_token_id

        if state.cursor >= len(state.response_ids):
            state.finished = True
            return self._fallback_token_id

        token_id = state.response_ids[state.cursor]
        state.cursor += 1
        state.generated_tokens += 1

        if state.cursor >= len(state.response_ids):
            state.finished = True

        return token_id
