"""Unified OpenAI-compatible model pool with DeepSeek/Bailian/Ark support and FakeModelPool."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from openai import APIConnectionError, APIError, APITimeoutError, AuthenticationError, OpenAI, RateLimitError

from .billing_meter import (
    BillingMeter,
    BudgetExceededError,
    compute_cost_cny,
    estimate_worst_case_cost_cny,
    now_cn,
)
from .config import ModelsCatalog, get_api_key, get_base_url


@dataclass
class ToolCallSpec:
    """Normalized tool call extracted from model response."""

    id: str
    name: str
    arguments_raw: str
    arguments_dict: dict[str, Any] | None
    parse_error: str | None = None


@dataclass
class ModelResponse:
    """Normalized response across all providers and tiers."""

    provider: str
    tier: str
    model_requested: str
    model_actual: str
    content: str
    reasoning_content: str
    tool_calls: list[ToolCallSpec]
    prompt_tokens: int
    cache_hit_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    latency_ms: int
    cost_cny: float
    pricing_period: str
    finish_reason: str = "stop"
    raw_assistant_message: dict[str, Any] = field(default_factory=dict)


def _estimate_prompt_tokens(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> int:
    payload = json.dumps({"messages": messages, "tools": tools or []}, ensure_ascii=False)
    return max(64, len(payload) // 2 + 128)


def _request_hash(provider: str, tier: str, model_id: str, messages: list[dict[str, Any]]) -> str:
    encoded = json.dumps(
        {"provider": provider, "tier": tier, "model": model_id, "messages": messages},
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def _extract_usage_fields(usage: Any) -> tuple[int, int, int, int]:
    if usage is None:
        return 0, 0, 0, 0
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)

    cache_hit = int(getattr(usage, "prompt_cache_hit_tokens", 0) or 0)
    p_details = getattr(usage, "prompt_tokens_details", None)
    if p_details is not None and cache_hit == 0:
        cache_hit = int(getattr(p_details, "cached_tokens", 0) or 0)

    reasoning_tokens = 0
    c_details = getattr(usage, "completion_tokens_details", None)
    if c_details is not None:
        reasoning_tokens = int(getattr(c_details, "reasoning_tokens", 0) or 0)

    return prompt_tokens, cache_hit, completion_tokens, reasoning_tokens


def _parse_tool_call_fields(tc_id: str, name: str, args_raw: str) -> ToolCallSpec:
    args_dict: dict[str, Any] | None = None
    err: str | None = None
    try:
        obj = json.loads(args_raw)
        if isinstance(obj, dict):
            args_dict = obj
        else:
            err = f"Tool arguments JSON must be object, got {type(obj).__name__}"
    except Exception as exc:
        err = f"Invalid JSON tool arguments: {exc}"
    return ToolCallSpec(
        id=tc_id,
        name=name,
        arguments_raw=args_raw,
        arguments_dict=args_dict,
        parse_error=err,
    )


def _parse_tool_calls(raw_calls: Any) -> list[ToolCallSpec]:
    if not raw_calls:
        return []
    parsed: list[ToolCallSpec] = []
    for tc in raw_calls:
        fn = getattr(tc, "function", None)
        name = str(getattr(fn, "name", "") or "")
        args_raw = str(getattr(fn, "arguments", "") or "{}")
        parsed.append(_parse_tool_call_fields(str(getattr(tc, "id", "") or ""), name, args_raw))
    return parsed


def _merge_stream_tool_calls(acc: dict[int, dict[str, str]], delta_calls: list[Any]) -> None:
    for tc in delta_calls:
        idx = int(getattr(tc, "index", 0) or 0)
        entry = acc.setdefault(idx, {"id": "", "name": "", "arguments": ""})
        if getattr(tc, "id", None):
            entry["id"] = str(tc.id)
        fn = getattr(tc, "function", None)
        if fn is not None:
            if getattr(fn, "name", None):
                entry["name"] += str(fn.name)
            if getattr(fn, "arguments", None):
                entry["arguments"] += str(fn.arguments)


def _execute_completion(
    client: OpenAI,
    kwargs: dict[str, Any],
    fallback_model: str,
) -> tuple[str, str, str, list[ToolCallSpec], int, int, int, int, str]:
    """Run completion, automatically retrying with streaming when a provider requires it."""
    try:
        resp = client.chat.completions.create(**kwargs)
        choice0 = resp.choices[0]
        finish_reason = str(getattr(choice0, "finish_reason", None) or "stop")
        msg = choice0.message
        msg_dump = msg.model_dump() if hasattr(msg, "model_dump") else {}
        content = str(getattr(msg, "content", None) or msg_dump.get("content") or "")
        reasoning_content = str(
            getattr(msg, "reasoning_content", None)
            or (getattr(msg, "model_extra", None) or {}).get("reasoning_content")
            or msg_dump.get("reasoning_content")
            or ""
        )
        tool_calls = _parse_tool_calls(getattr(msg, "tool_calls", None))
        p_tok, hit_tok, c_tok, r_tok = _extract_usage_fields(getattr(resp, "usage", None))
        actual_model = str(getattr(resp, "model", None) or fallback_model)
        return finish_reason, content, reasoning_content, tool_calls, p_tok, hit_tok, c_tok, r_tok, actual_model
    except APIError as exc:
        if "stream" not in str(exc).lower():
            raise
        stream_kwargs = {**kwargs, "stream": True, "stream_options": {"include_usage": True}}
        stream = client.chat.completions.create(**stream_kwargs)
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tc_acc: dict[int, dict[str, str]] = {}
        finish_reason = "stop"
        actual_model = fallback_model
        usage_obj = None
        for chunk in stream:
            if getattr(chunk, "model", None):
                actual_model = str(chunk.model)
            if getattr(chunk, "usage", None) is not None:
                usage_obj = chunk.usage
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            c0 = choices[0]
            if getattr(c0, "finish_reason", None):
                finish_reason = str(c0.finish_reason)
            delta = getattr(c0, "delta", None)
            if delta is None:
                continue
            delta_dump = delta.model_dump() if hasattr(delta, "model_dump") else {}
            if d_content := (getattr(delta, "content", None) or delta_dump.get("content")):
                content_parts.append(str(d_content))
            if d_reason := (
                getattr(delta, "reasoning_content", None)
                or (getattr(delta, "model_extra", None) or {}).get("reasoning_content")
                or delta_dump.get("reasoning_content")
            ):
                reasoning_parts.append(str(d_reason))
            if d_tcs := getattr(delta, "tool_calls", None):
                _merge_stream_tool_calls(tc_acc, d_tcs)
        tool_calls = [
            _parse_tool_call_fields(
                tc_acc[idx]["id"] or f"call_{idx}",
                tc_acc[idx]["name"],
                tc_acc[idx]["arguments"] or "{}",
            )
            for idx in sorted(tc_acc)
        ]
        p_tok, hit_tok, c_tok, r_tok = _extract_usage_fields(usage_obj)
        return (
            finish_reason,
            "".join(content_parts),
            "".join(reasoning_parts),
            tool_calls,
            p_tok,
            hit_tok,
            c_tok,
            r_tok,
            actual_model,
        )


class LiveModelPool:
    """Live API pool calling DeepSeek, Bailian (DashScope), or Volcengine Ark."""

    def __init__(
        self,
        catalog: ModelsCatalog,
        meter: BillingMeter,
        *,
        timeout_sec: float = 60.0,
        max_retries: int = 2,
    ) -> None:
        self.catalog = catalog
        self.meter = meter
        self.timeout_sec = float(timeout_sec)
        self.max_retries = int(max_retries)
        self._clients: dict[str, OpenAI] = {}
        self._client_lock = threading.Lock()
        self._holidays = set(catalog.holidays_2026)

    def _get_client(self, provider: str) -> OpenAI:
        with self._client_lock:
            if provider in self._clients:
                return self._clients[provider]
            if provider not in self.catalog.providers:
                raise ValueError(f"Unknown provider: {provider}")
            cfg = self.catalog.providers[provider]
            api_key = get_api_key(cfg)
            if not api_key:
                raise RuntimeError(f"API key not configured for provider '{provider}' ({cfg.api_key_env})")
            base_url = get_base_url(cfg)
            client = OpenAI(api_key=api_key, base_url=base_url, timeout=self.timeout_sec, max_retries=0)
            self._clients[provider] = client
            return client

    def call_with_tools(
        self,
        *,
        run_id: str,
        episode_id: str,
        task_id: str,
        step_idx: int,
        provider: str,
        tier: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens_override: int | None = None,
        extra_body_override: dict[str, Any] | None = None,
        reasoning_effort_override: str | None = None,
    ) -> ModelResponse:
        provider_cfg = self.catalog.providers[provider]
        if tier not in provider_cfg.tiers:
            raise ValueError(f"Provider '{provider}' does not define tier '{tier}'")
        tier_cfg = provider_cfg.tiers[tier]
        model_id = tier_cfg.resolve_model_id()
        max_tokens = int(max_tokens_override or tier_cfg.max_tokens)
        est_prompt = _estimate_prompt_tokens(messages, tools)
        est_max_cost = estimate_worst_case_cost_cny(tier_cfg, est_prompt, max_tokens)
        req_hash = _request_hash(provider, tier, model_id, messages)
        client = self._get_client(provider)

        merged_extra = {**tier_cfg.extra_body, **(extra_body_override or {})}
        effort = reasoning_effort_override if reasoning_effort_override is not None else tier_cfg.reasoning_effort

        last_exc: Exception | None = None
        for retry_idx in range(self.max_retries + 1):
            attempt_id = self.meter.begin_attempt(
                run_id=run_id,
                episode_id=episode_id,
                task_id=task_id,
                step_idx=step_idx,
                provider=provider,
                tier=tier,
                model_requested=model_id,
                estimated_max_cost_cny=est_max_cost,
                request_hash=req_hash,
            )

            t0 = time.perf_counter()
            kwargs: dict[str, Any] = {
                "model": model_id,
                "messages": messages,
                "max_tokens": max_tokens,
            }
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            if tier_cfg.temperature is not None:
                kwargs["temperature"] = tier_cfg.temperature
            if effort:
                kwargs["reasoning_effort"] = effort
            if merged_extra:
                kwargs["extra_body"] = merged_extra

            try:
                (
                    finish_reason,
                    content,
                    reasoning_content,
                    tool_calls,
                    p_tok,
                    hit_tok,
                    c_tok,
                    r_tok,
                    actual_model,
                ) = _execute_completion(client, kwargs, model_id)
                latency_ms = int((time.perf_counter() - t0) * 1000)
                cost_cny, period = compute_cost_cny(
                    provider_cfg,
                    tier_cfg,
                    prompt_tokens=p_tok,
                    cache_hit_tokens=hit_tok,
                    completion_tokens=c_tok,
                    dt_cn=now_cn(),
                    holidays=self._holidays,
                )

                assistant_msg: dict[str, Any] = {"role": "assistant", "content": content}
                if reasoning_content:
                    assistant_msg["reasoning_content"] = reasoning_content
                if tool_calls:
                    assistant_msg["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": tc.arguments_raw},
                        }
                        for tc in tool_calls
                    ]

                self.meter.finish_attempt_ok(
                    attempt_id=attempt_id,
                    model_actual=actual_model,
                    pricing_period=period,
                    prompt_tokens=p_tok,
                    cache_hit_tokens=hit_tok,
                    completion_tokens=c_tok,
                    reasoning_tokens=r_tok,
                    actual_cost_cny=cost_cny,
                    latency_ms=latency_ms,
                    response_payload={
                        "model": actual_model,
                        "finish_reason": finish_reason,
                        "content_preview": content[:500],
                        "reasoning_len": len(reasoning_content),
                        "tool_calls": [
                            {"name": tc.name, "arguments": tc.arguments_raw, "parse_error": tc.parse_error}
                            for tc in tool_calls
                        ],
                    },
                )

                return ModelResponse(
                    provider=provider,
                    tier=tier,
                    model_requested=model_id,
                    model_actual=actual_model,
                    content=content,
                    reasoning_content=reasoning_content,
                    tool_calls=tool_calls,
                    prompt_tokens=p_tok,
                    cache_hit_tokens=hit_tok,
                    completion_tokens=c_tok,
                    reasoning_tokens=r_tok,
                    latency_ms=latency_ms,
                    cost_cny=cost_cny,
                    pricing_period=period,
                    finish_reason=finish_reason,
                    raw_assistant_message=assistant_msg,
                )

            except BudgetExceededError:
                raise
            except AuthenticationError as exc:
                latency_ms = int((time.perf_counter() - t0) * 1000)
                self.meter.finish_attempt_error(
                    attempt_id=attempt_id,
                    error_msg=f"AuthenticationError: {type(exc).__name__}",
                    latency_ms=latency_ms,
                    billing_unknown=False,
                )
                raise
            except (APITimeoutError, APIConnectionError, RateLimitError, APIError) as exc:
                latency_ms = int((time.perf_counter() - t0) * 1000)
                unknown = isinstance(exc, APITimeoutError)
                self.meter.finish_attempt_error(
                    attempt_id=attempt_id,
                    error_msg=f"{type(exc).__name__}: {exc}",
                    latency_ms=latency_ms,
                    billing_unknown=unknown,
                )
                last_exc = exc
                if retry_idx < self.max_retries:
                    time.sleep(1.0 * (retry_idx + 1))
                    continue
                raise

        assert last_exc is not None
        raise last_exc


FakeHandler = Callable[[str, str, int, list[dict[str, Any]], list[dict[str, Any]] | None], dict[str, Any]]


class FakeModelPool:
    """Deterministic zero-cost fake model pool for offline tests and CI."""

    def __init__(
        self,
        catalog: ModelsCatalog,
        meter: BillingMeter,
        handler: FakeHandler,
    ) -> None:
        self.catalog = catalog
        self.meter = meter
        self.handler = handler
        self._holidays = set(catalog.holidays_2026)

    def call_with_tools(
        self,
        *,
        run_id: str,
        episode_id: str,
        task_id: str,
        step_idx: int,
        provider: str,
        tier: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens_override: int | None = None,
        extra_body_override: dict[str, Any] | None = None,
        reasoning_effort_override: str | None = None,
    ) -> ModelResponse:
        _ = (extra_body_override, reasoning_effort_override)
        provider_cfg = self.catalog.providers[provider]
        tier_cfg = provider_cfg.tiers[tier]
        model_id = tier_cfg.model_id
        max_tokens = int(max_tokens_override or tier_cfg.max_tokens)
        est_prompt = _estimate_prompt_tokens(messages, tools)
        est_max_cost = estimate_worst_case_cost_cny(tier_cfg, est_prompt, max_tokens)

        attempt_id = self.meter.begin_attempt(
            run_id=run_id,
            episode_id=episode_id,
            task_id=task_id,
            step_idx=step_idx,
            provider=provider,
            tier=tier,
            model_requested=model_id,
            estimated_max_cost_cny=est_max_cost,
            request_hash=_request_hash(provider, tier, model_id, messages),
        )

        spec = self.handler(task_id, tier, step_idx, messages, tools)
        content = str(spec.get("content", ""))
        reasoning_content = str(spec.get("reasoning_content", ""))
        raw_tcs = spec.get("tool_calls", [])
        tool_calls: list[ToolCallSpec] = []
        for idx, tc in enumerate(raw_tcs):
            args_raw = tc["arguments"] if isinstance(tc["arguments"], str) else json.dumps(tc["arguments"])
            args_dict: dict[str, Any] | None = None
            parse_err: str | None = None
            try:
                loaded = json.loads(args_raw)
                if isinstance(loaded, dict):
                    args_dict = loaded
                else:
                    parse_err = "Tool arguments JSON must be object"
            except Exception as exc:
                parse_err = f"Invalid JSON: {exc}"
            tool_calls.append(
                ToolCallSpec(
                    id=tc.get("id", f"fake_call_{step_idx}_{idx}"),
                    name=tc["name"],
                    arguments_raw=args_raw,
                    arguments_dict=args_dict,
                    parse_error=parse_err,
                )
            )

        p_tok = int(spec.get("prompt_tokens", est_prompt))
        hit_tok = int(spec.get("cache_hit_tokens", 0))
        c_tok = int(spec.get("completion_tokens", 280 if tier in ("think", "pro") else 85))
        r_tok = int(spec.get("reasoning_tokens", 200 if tier in ("think", "pro") else 0))
        cost_cny, period = compute_cost_cny(
            provider_cfg,
            tier_cfg,
            prompt_tokens=p_tok,
            cache_hit_tokens=hit_tok,
            completion_tokens=c_tok,
            dt_cn=now_cn(),
            holidays=self._holidays,
        )

        assistant_msg: dict[str, Any] = {"role": "assistant", "content": content}
        if reasoning_content:
            assistant_msg["reasoning_content"] = reasoning_content
        if tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments_raw},
                }
                for tc in tool_calls
            ]

        finish_reason = str(spec.get("finish_reason", "stop"))

        self.meter.finish_attempt_ok(
            attempt_id=attempt_id,
            model_actual=f"{model_id}-fake",
            pricing_period=period,
            prompt_tokens=p_tok,
            cache_hit_tokens=hit_tok,
            completion_tokens=c_tok,
            reasoning_tokens=r_tok,
            actual_cost_cny=cost_cny,
            latency_ms=5,
            response_payload={
                "fake": True,
                "finish_reason": finish_reason,
                "tool_calls": [tc.name for tc in tool_calls],
            },
        )

        return ModelResponse(
            provider=provider,
            tier=tier,
            model_requested=model_id,
            model_actual=f"{model_id}-fake",
            content=content,
            reasoning_content=reasoning_content,
            tool_calls=tool_calls,
            prompt_tokens=p_tok,
            cache_hit_tokens=hit_tok,
            completion_tokens=c_tok,
            reasoning_tokens=r_tok,
            latency_ms=5,
            cost_cny=cost_cny,
            pricing_period=period,
            finish_reason=finish_reason,
            raw_assistant_message=assistant_msg,
        )
