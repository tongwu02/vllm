# vllm/sim/simulator.py
import json
import logging
import hashlib
import inspect
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class _ReqState:
    request_id: str
    prompt: str
    params: Any                        # SamplingParams (vLLM)
    prompt_token_ids: List[int]
    resp_token_ids: List[int]
    cursor: int = 0                    # 已经生成的新token数量
    prefill_done: bool = False
    finished: bool = False
    finish_reason: Optional[str] = None


class Simulator:
    """
    一个最小可用的“trace 驱动”模拟器：
    - 从 JSONL trace 加载 (prompt, response)
    - 用 tokenizer 预编码 response -> token_ids
    - prefill: 仅做记账（标记可开始decode）
    - decode: 每轮给每个请求吐 1 个 token；命中边界条件时 finished
    - 返回 List[RequestOutput]（与真实路径一致）
    """

    def __init__(self, trace_path: str, tokenizer):
        """
        Args:
            trace_path: JSONL 文件，每行: {"prompt": "...", "response": "..."}
            tokenizer: 兼容 HF tokenizer 接口 (encode/decode)
        """
        self.tokenizer = tokenizer
        self._trace_map: Dict[str, List[int]] = {}
        self._states: Dict[str, _ReqState] = {}

        self._load_trace(trace_path)

        # 延迟导入 vLLM 输出类，并做签名自适配
        from vllm.outputs import RequestOutput
        try:
            from vllm.outputs import SequenceOutput  # vLLM 0.x
        except Exception:
            SequenceOutput = None  # 兼容性占位
        self._RequestOutput = RequestOutput
        self._SequenceOutput = SequenceOutput
        self._req_sig = inspect.signature(RequestOutput)
        self._seq_sig = inspect.signature(SequenceOutput) if SequenceOutput else None

    # -----------------------------
    # 公共接口（供 LLMEngine 调用）
    # -----------------------------

    def on_add_request(self, request_id: str, prompt: str, params):
        """
        注册新请求：建立游标/状态。
        - 如果 prompt 未在 trace 中，给一个空响应并立即可完成（finished=True）
        """
        norm_key = self._normalize_prompt(prompt)
        if norm_key in self._trace_map:
            resp_ids = list(self._trace_map[norm_key])
        else:
            logger.warning(
                "[Simulator] Prompt not found in trace; request_id=%s (will end quickly).",
                request_id
            )
            resp_ids = []  # 没有可生成的token

        # 计算 prompt token_ids（用于构造 RequestOutput）
        try:
            prompt_ids = self._encode(prompt)
        except Exception:
            prompt_ids = []

        self._states[request_id] = _ReqState(
            request_id=request_id,
            prompt=prompt,
            params=params,
            prompt_token_ids=prompt_ids,
            resp_token_ids=resp_ids,
            cursor=0,
            prefill_done=False,
            finished=(len(resp_ids) == 0),  # 无trace时立即完成
            finish_reason="stop" if len(resp_ids) == 0 else None,
        )

    def simulate_prefill(self, prompt_runs):
        """
        对本轮 prefill 的请求仅做记账：
        - 标记 prefill_done，表示下一轮可进入 decode
        prompt_runs: List[ScheduledGroup/Any]（由调度器产物传入）
        """
        for rid in self._iter_request_ids(prompt_runs):
            state = self._states.get(rid)
            if state and not state.prefill_done:
                state.prefill_done = True

    def simulate_decode(self, decode_runs):
        """
        对本轮 decode 的请求：
        - 每个请求吐出 1 个 token
        - 处理 max_tokens / stop 条件
        - 返回 List[RequestOutput]
        """
        outputs: List[Any] = []
        for rid in self._iter_request_ids(decode_runs):
            st = self._states.get(rid)
            if not st:
                logger.warning("[Simulator] Missing state for request_id=%s", rid)
                continue
            if st.finished:
                # 已完成的请求不再输出（真实路径通常不会调到这里）
                continue
            if not st.prefill_done:
                # 理论上 decode 前应完成 prefill
                st.prefill_done = True

            # 计算本轮应否结束（max_tokens/trace上限/stop）
            max_new_tokens = self._get_max_new_tokens(st.params)

            # 1) 如果 max_tokens 已经达到（无需再吐）
            if max_new_tokens is not None and st.cursor >= max_new_tokens:
                st.finished = True
                st.finish_reason = "length"
                outputs.append(self._build_request_output(st))
                continue

            # 2) 如果 trace 已经用尽（无需再吐）
            if st.cursor >= len(st.resp_token_ids):
                st.finished = True
                st.finish_reason = "stop"
                outputs.append(self._build_request_output(st))
                continue

            # 3) 正常吐 1 个 token
            st.cursor += 1

            # 4) 吐完后再次检查结束条件（max_tokens / trace 末尾 / stop 序列）
            #    - trace 末尾
            if st.cursor >= len(st.resp_token_ids):
                st.finished = True
                st.finish_reason = "stop"
            #    - max_tokens
            if (not st.finished) and (max_new_tokens is not None) and (st.cursor >= max_new_tokens):
                st.finished = True
                st.finish_reason = "length"
            #    - stop 序列（简单实现：基于当前生成文本后缀匹配）
            if not st.finished and self._hit_stop(st):
                st.finished = True
                st.finish_reason = "stop"

            outputs.append(self._build_request_output(st))

        return outputs

    # -----------------------------
    # 内部工具
    # -----------------------------

    def _load_trace(self, trace_path: str):
        """
        JSONL: {"prompt": "...", "response": "..."}
        建立 {normalized_prompt: response_token_ids}
        """
        count = 0
        with open(trace_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                prompt = str(rec.get("prompt", ""))
                response = str(rec.get("response", ""))

                # 为了“每轮吐 1 token”的行为稳定，建议禁用特殊token
                resp_ids = self._encode(response)
                key = self._normalize_prompt(prompt)
                self._trace_map[key] = resp_ids
                count += 1
        logger.info("[Simulator] Loaded %d prompt-response pairs from %s",
                    count, trace_path)

    def _normalize_prompt(self, prompt: str) -> str:
        # 简单归一化：去掉两端空白
        return prompt.strip()

    def _encode(self, text: str) -> List[int]:
        # 统一关闭 add_special_tokens，保持“每步1token”的节奏和 trace 一致
        if hasattr(self.tokenizer, "encode"):
            try:
                return self.tokenizer.encode(text, add_special_tokens=False)
            except TypeError:
                # 某些tokenizer不支持该参数
                return self.tokenizer.encode(text)
        # 兜底：无 tokenizer 时给空
        return []

    def _decode_tokens(self, token_ids: List[int]) -> str:
        if hasattr(self.tokenizer, "decode"):
            return self.tokenizer.decode(token_ids, skip_special_tokens=True)
        return ""

    def _iter_request_ids(self, run_items) -> List[str]:
        """
        从调度产物中提取 request_id：
        - 兼容不同版本对象结构（尽量鲁棒）
        - 若传入的本身就是 request_id（str），直接返回
        """
        result = []
        if not run_items:
            return result

        for item in run_items:
            rid = None
            if isinstance(item, str):
                rid = item
            else:
                # 常见路径尝试
                for path in (
                    "request_id",
                    "seq_group.request_id",
                    "seq_group.group_id",
                    "group_id",
                    "seq_data.request_id",
                ):
                    rid = self._dig_attr(item, path)
                    if rid is not None:
                        break
            if rid is None:
                logger.warning("[Simulator] Cannot extract request_id from %r", item)
                continue
            result.append(str(rid))
        return result

    def _dig_attr(self, obj: Any, dotted: str):
        cur = obj
        try:
            for name in dotted.split("."):
                cur = getattr(cur, name)
            return cur
        except Exception:
            return None

    def _get_max_new_tokens(self, params) -> Optional[int]:
        # vLLM 的采样参数一般叫 max_tokens；也兼容 max_new_tokens
        for name in ("max_tokens", "max_new_tokens"):
            if hasattr(params, name):
                val = getattr(params, name)
                if isinstance(val, int):
                    return val
        return None

    def _current_generated_ids(self, st: _ReqState) -> List[int]:
        # 已生成的新 token 序列
        return st.resp_token_ids[: st.cursor]

    def _hit_stop(self, st: _ReqState) -> bool:
        """
        简单的 stop 序列判定：基于“当前生成文本”的后缀匹配。
        注意：不同 tokenizer 会导致字符/空白对齐细节差异，这里做最小实现即可。
        """
        stops = []
        for name in ("stop", "stop_sequences", "stop_words"):
            if hasattr(st.params, name) and getattr(st.params, name):
                stops = list(getattr(st.params, name) or [])
                break
        if not stops:
            return False

        text = self._decode_tokens(self._current_generated_ids(st))
        for s in stops:
            try:
                if s and text.endswith(s):
                    return True
            except Exception:
                # 容错
                continue
        return False

    # -----------------------------
    # 输出对象构造（自适配签名）
    # -----------------------------

    def _build_request_output(self, st: _ReqState):
        """
        构造 vLLM 的 RequestOutput：
        - outputs: [SequenceOutput]（单路采样 index=0）
        - token_ids: 目前累计生成的新 token 序列
        - text: 对这些 token 的 decode
        """
        gen_ids = self._current_generated_ids(st)
        text = self._decode_tokens(gen_ids)

        seq_kwargs = dict(
            index=0,
            text=text,
            token_ids=gen_ids,
            finish_reason=st.finish_reason,
            # 其它字段让签名过滤器决定是否传入
            cumulative_logprob=None,
            logprobs=None,
        )
        seq_obj = self._make_sequence_output(seq_kwargs)

        req_kwargs = dict(
            request_id=st.request_id,
            prompt=st.prompt,
            prompt_token_ids=st.prompt_token_ids,
            outputs=[seq_obj],
            finished=bool(st.finished),
            # usage/metrics等字段由签名过滤器判断是否可传
            metrics=None,
            usage=None,
        )
        req_obj = self._make_request_output(req_kwargs)
        return req_obj

    def _make_sequence_output(self, kwargs: Dict[str, Any]):
        if self._SequenceOutput is None:
            raise RuntimeError("vllm.outputs.SequenceOutput not available.")
        # 过滤出签名可接受的参数，提升跨版本兼容
        params = self._seq_sig.parameters if self._seq_sig else {}
        fkwargs = {k: v for k, v in kwargs.items() if k in params}
        return self._SequenceOutput(**fkwargs)

    def _make_request_output(self, kwargs: Dict[str, Any]):
        params = self._req_sig.parameters
        fkwargs = {k: v for k, v in kwargs.items() if k in params}
        return self._RequestOutput(**fkwargs)
