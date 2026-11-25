"""Episode and split execution runner with FSM guardrails, ledger compression, and SQLite auditing."""

from __future__ import annotations

import concurrent.futures
import json
import re
import time
import uuid
from pathlib import Path
from typing import Any

from openai import APIConnectionError, APIStatusError, AuthenticationError, RateLimitError

from aroute.envs import create_env
from aroute.envs.base import ToolResult
from aroute.orchestrator.fsm_guard import FSMGuard, canonical_call_hash, canonical_snapshot_hash
from aroute.orchestrator.memory_ledger import StateLedger
from aroute.orchestrator.prewrite import PrewriteControl
from aroute.orchestrator.research_router import ResearchStepRouter
from aroute.orchestrator.router import RouterParams, StepRouter, load_router_params

from .billing_meter import BillingMeter, BudgetExceededError, get_active_tariff_rates
from .config import PROJECT_ROOT, load_models_catalog
from .failure_tagger import EpisodeTraceStats, classify_failure
from .model_pool import FakeHandler, FakeModelPool, LiveModelPool
from .prewrite_execution import ProposalRule, assistant_action, contract_debit, resolve_proposal
from .research_telemetry import episode_timing, request_settings
from .verifier import verify_episode

SYSTEM_PROMPT = (
    "You are an exact multi-step enterprise tool agent. "
    "Use the provided tools carefully, obey domain preconditions (check status/dependencies/frozen flags/budgets before mutating state), "
    "and stop with a concise summary only after all task goals are verified."
)

STEP0_CLASSIFIER_TIERS = ("fast", "think_med", "think", "pro")
_STEP0_TIER_PATTERN = re.compile(r"\b(fast|think_med|think|pro)\b")
RESEARCH_CONSTRAINED_TOOLS = {
    "supply_chain": frozenset({"create_restock_po"}),
    "db_transaction": frozenset({"post_adjustment"}),
    "ops_troubleshoot": frozenset({"rollback_config", "restart_service"}),
}


def _classify_step0_tier(
    pool: LiveModelPool | FakeModelPool,
    *,
    run_id: str,
    episode_id: str,
    task_id: str,
    provider: str,
    task_prompt: str,
) -> tuple[str, float]:
    """Prompt-only Step-0 router rubric simulating commercial black-box Harness auto mode."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a pre-execution task complexity router. Given only the initial user request, "
                "select one execution compute tier strictly as a JSON object "
                '{"tier": "fast"|"think_med"|"think"|"pro"}. Rubric: '
                "fast = single simple entity or straightforward update; "
                "think_med = moderate constraints or 2-step workflow; "
                "think = multi-entity coupled constraints, fees/discounts, or tight quotas; "
                "pro = complex multi-stage recovery or cascading dependencies."
            ),
        },
        {"role": "user", "content": task_prompt},
    ]
    resp = pool.call_with_tools(
        run_id=run_id,
        episode_id=episode_id,
        task_id=task_id,
        step_idx=-1,
        provider=provider,
        tier="fast",
        messages=messages,
        tools=None,
        max_tokens_override=64,
    )
    chosen = "think_med"
    raw = resp.content.strip()
    try:
        m_json = re.search(r"\{.*?\}", raw, re.DOTALL)
        obj = json.loads(m_json.group(0) if m_json else raw)
        if isinstance(obj, dict) and str(obj.get("tier", "")) in STEP0_CLASSIFIER_TIERS:
            chosen = str(obj["tier"])
    except Exception:
        m_tok = _STEP0_TIER_PATTERN.search(raw)
        if m_tok:
            chosen = m_tok.group(1)

    if provider in pool.catalog.providers and chosen not in pool.catalog.providers[provider].tiers:
        chosen = "think" if "think" in pool.catalog.providers[provider].tiers else "fast"
    return chosen, resp.cost_cny


def load_split_tasks(
    split: str,
    *,
    allow_locked_test: bool = False,
    data_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Load task specifications for a split with blind-test leakage protection."""
    base = data_dir or (PROJECT_ROOT / "data")
    if split == "dev":
        return json.loads((base / "tasks_dev.json").read_text(encoding="utf-8"))
    if split == "val":
        return json.loads((base / "tasks_val.json").read_text(encoding="utf-8"))
    if split == "cal":
        # Deterministic 24-task stratified subset from dev: 8 per domain (2 Easy, 3 Medium, 3 Hard)
        dev_tasks = json.loads((base / "tasks_dev.json").read_text(encoding="utf-8"))
        quotas = {"Easy": 2, "Medium": 3, "Hard": 3}
        selected: list[dict[str, Any]] = []
        for domain in ("supply_chain", "db_transaction", "ops_troubleshoot"):
            for diff, need in quotas.items():
                matches = [t for t in dev_tasks if t["domain"] == domain and t["difficulty"] == diff]
                selected.extend(matches[:need])
        return selected
    if split == "test_locked":
        if not allow_locked_test:
            raise PermissionError(
                "Refusing to load locked test set without explicit allow_locked_test=True"
            )
        return json.loads((base / "tasks_test_locked.json").read_text(encoding="utf-8"))
    raise ValueError(f"Unknown split '{split}', choose from dev|val|cal|test_locked")


def run_episode(
    *,
    task: dict[str, Any],
    run_id: str,
    provider: str,
    router: StepRouter,
    pool: LiveModelPool | FakeModelPool,
    meter: BillingMeter,
    tariff_regime_override: str | None = None,
    pause_on_transport_error: bool = False,
    prewrite: PrewriteControl | None = None,
    proposal_rule: ProposalRule | None = None,
) -> dict[str, Any]:
    """Execute one complete multi-step episode against a fresh deterministic SQLite sandbox."""
    started = time.perf_counter()
    task_id = str(task["task_id"])
    domain = str(task["domain"])
    if prewrite is not None and (not isinstance(router, ResearchStepRouter) or prewrite.domain != domain):
        raise ValueError("Prewrite requires a matching research environment and router.")
    if proposal_rule is not None and prewrite is None:
        raise ValueError("The public proposal rule requires a prewrite controller.")
    difficulty = str(task["difficulty"])
    max_steps = int(task.get("max_steps", 20))
    episode_id = f"{run_id}_{task_id}"
    ep_cap = float(router.params.episode_budget_cap_cny)

    meter.start_episode(
        episode_id=episode_id,
        run_id=run_id,
        task_id=task_id,
        domain=domain,
        difficulty=difficulty,
    )

    env = create_env(domain)
    env.reset(task)
    ledger = StateLedger(window_k=router.params.ledger_window_k)
    ledger.init_raw_conversation(SYSTEM_PROMPT, task["prompt"])
    fsm_guard = FSMGuard(max_repeat_same_call=router.params.max_repeat_same_call)
    use_fsm = router.fsm_enabled()
    research = isinstance(router, ResearchStepRouter)
    use_ledger = router.ledger_enabled() and prewrite is None
    tools = env.tool_schemas()

    ep_cost = 0.0
    steps_taken = 0
    step0_selected_tier: str | None = None
    budget_exhausted = False
    loop_aborted = False
    schema_errors = 0
    precondition_blocks = 0
    duplicate_successful_actions = 0
    total_tool_calls = 0
    write_calls = 0
    successful_write_hashes: set[str] = set()
    prev_state_sig = None if research else json.dumps(env.snapshot_state(), sort_keys=True)
    unchanged_streak = 0
    truncation_streak = 0
    truncation_abort = False
    thinking_truncations = 0
    research_trace: dict[str, Any] | None = None
    replan_fast = False

    def record_prewrite(event: str, data: dict[str, Any]) -> None:
        meter.record_step_event(
            run_id=run_id, episode_id=episode_id, task_id=task_id, step_idx=step_idx,
            tier=tier, tool_name="__V3_PREWRITE__", tool_args={"event": event, **data}, fsm_blocked=False,
            fsm_reason=None, tool_ok=None, error_code=None, state_changed=False, ledger_summary="",
        )

    def supplement(call) -> ToolResult:
        args = call.arguments_dict
        if env.is_write_tool(call.name):
            raise ValueError("A prewrite observation cannot dispatch a write.")
        result = env.execute_tool(call.name, args)
        if result.is_write or result.state_changed:
            raise ValueError("Prewrite observation violated its read-only contract.")
        prewrite.observe(call.name, args, result)
        ledger.record_assistant_message(assistant_action(call))
        ledger.record_tool_message(call.id, result.to_tool_message())
        record_prewrite("supplement", {"id": call.id, "tool": call.name, "arguments": args,
                                       "sequence": prewrite.sequence, "observation": json.loads(result.to_tool_message())})
        return result

    def review_candidate(instruction: str):
        prefix = ledger.build_messages(system_prompt=SYSTEM_PROMPT, task_prompt=task["prompt"], active_tier="fast",
                                       use_state_ledger=False, provider=step_provider)
        return pool.call_with_tools(
            run_id=run_id, episode_id=episode_id, task_id=task_id, step_idx=step_idx, provider=step_provider,
            tier="think", messages=[*prefix, {"role": "user", "content": instruction}], tools=None,
        )

    active_period = "offpeak"
    holidays_set = set(pool.catalog.holidays_2026)
    if provider in pool.catalog.providers:
        prov_cfg = pool.catalog.providers[provider]
        if "think" in prov_cfg.tiers:
            active_period = get_active_tariff_rates(
                prov_cfg,
                prov_cfg.tiers["think"],
                holidays=holidays_set,
                regime_override=tariff_regime_override,
            )[3]
    elif tariff_regime_override in ("peak", "offpeak"):
        active_period = tariff_regime_override

    provider_think_rates: dict[str, tuple[float, float, float]] = {}
    for pname, pcfg in pool.catalog.providers.items():
        if "think" in pcfg.tiers:
            provider_think_rates[pname] = get_active_tariff_rates(
                pcfg,
                pcfg.tiers["think"],
                holidays=holidays_set,
                regime_override=tariff_regime_override,
            )[:3]

    try:
        if router.policy == "step0_auto":
            try:
                step0_selected_tier, cls_cost = _classify_step0_tier(
                    pool,
                    run_id=run_id,
                    episode_id=episode_id,
                    task_id=task_id,
                    provider=provider,
                    task_prompt=task["prompt"],
                )
            except (BudgetExceededError, AuthenticationError):
                raise
            except Exception as exc:
                meter.finish_episode(
                    episode_id=episode_id,
                    passed=False,
                    failure_tag="API_ERROR",
                    steps_taken=0,
                    total_cost_cny=round(ep_cost, 8),
                    loop_abort=False,
                    summary={
                        "api_error": f"{type(exc).__name__}: {exc}",
                        "step_idx": -1,
                        **episode_timing(meter.db_path, episode_id, started),
                    },
                )
                return {
                    "task_id": task_id,
                    "passed": False,
                    "failure_tag": "API_ERROR",
                    "steps_taken": 0,
                    "cost_cny": round(ep_cost, 8),
                    "loop_abort": False,
                }
            ep_cost += cls_cost
            router.locked_step0_tier = step0_selected_tier

        for step_idx in range(max_steps):
            if ep_cost >= ep_cap:
                budget_exhausted = True
                break

            rem_ratio = max(0.0, (ep_cap - ep_cost) / ep_cap) if ep_cap > 0 else 0.0
            step_provider, tier = router.select_route(
                default_provider=provider,
                ledger=ledger,
                remaining_budget_ratio=rem_ratio,
                step_idx=step_idx,
                difficulty="" if research else difficulty,
                domain=domain,
                snapshot=None if research else env.snapshot_state(),
                active_pricing_period=active_period,
                provider_think_rates=provider_think_rates,
            )
            if replan_fast:
                tier = "fast"
                router.last_route_event = {**router.last_route_event, "reason": "prewrite_reproposal"}
                replan_fast = False
            if router.last_route_event is not None:
                if research_trace is None:
                    research_trace = {
                        "first_write_step": None,
                        "first_successful_write_step": None,
                        "write_attempts": 0,
                        "failed_write_attempts": 0,
                        "blocked_write_attempts": 0,
                        "first_constrained_write_step": None,
                        "constrained_write_attempts": 0,
                        "failed_constrained_write_attempts": 0,
                        "first_proactive_step": None,
                        "first_restart_step": None,
                        "lead_steps": None,
                        "proactive_think_calls": 0,
                        "reactive_think_calls": 0,
                        "orchestrator_guards": use_fsm,
                    }
                meter.record_step_event(
                    run_id=run_id,
                    episode_id=episode_id,
                    task_id=task_id,
                    step_idx=step_idx,
                    tier=tier,
                    tool_name="__ROUTE__",
                    tool_args={
                        **router.last_route_event, "provider": step_provider,
                        "request_settings": request_settings(
                            pool.catalog.providers[step_provider].tiers[tier], simulated=isinstance(pool, FakeModelPool),
                        ),
                    },
                    fsm_blocked=False,
                    fsm_reason=None,
                    tool_ok=None,
                    error_code=None,
                    state_changed=False,
                    ledger_summary="",
                )
            ledger.just_rejected_stop = False
            messages = ledger.build_messages(
                system_prompt=SYSTEM_PROMPT,
                task_prompt=task["prompt"],
                active_tier=tier,
                use_state_ledger=use_ledger,
                provider=step_provider,
            )

            try:
                resp = pool.call_with_tools(
                    run_id=run_id,
                    episode_id=episode_id,
                    task_id=task_id,
                    step_idx=step_idx,
                    provider=step_provider,
                    tier=tier,
                    messages=messages,
                    tools=tools,
                )
            except (BudgetExceededError, AuthenticationError):
                raise
            except Exception as exc:
                if pause_on_transport_error and (
                    isinstance(exc, (APIConnectionError, RateLimitError))
                    or (isinstance(exc, APIStatusError) and exc.status_code >= 500)
                ):
                    raise
                meter.finish_episode(
                    episode_id=episode_id,
                    passed=False,
                    failure_tag="API_ERROR",
                    steps_taken=steps_taken,
                    total_cost_cny=round(ep_cost, 8),
                    loop_abort=False,
                    summary={
                        "api_error": f"{type(exc).__name__}: {exc}",
                        "step_idx": step_idx,
                        **(research_trace or {}),
                        **episode_timing(meter.db_path, episode_id, started),
                    },
                )
                return {
                    "task_id": task_id,
                    "passed": False,
                    "failure_tag": "API_ERROR",
                    "steps_taken": steps_taken,
                    "cost_cny": round(ep_cost, 8),
                    "loop_abort": False,
                }

            ep_cost += resp.cost_cny
            steps_taken += 1
            ledger.last_prompt_tokens = resp.prompt_tokens
            if research_trace is not None and tier == "think":
                reason = router.last_route_event["reason"]
                if reason in ("proactive_state", "proactive_fixed"):
                    research_trace["proactive_think_calls"] += 1
                    if research_trace["first_proactive_step"] is None:
                        research_trace["first_proactive_step"] = step_idx
                        constrained = research_trace["first_constrained_write_step"]
                        if constrained is not None:
                            research_trace["lead_steps"] = constrained - step_idx
                elif reason.startswith("reactive_"):
                    research_trace["reactive_think_calls"] += 1

            if not resp.tool_calls:
                if resp.finish_reason == "length":
                    truncation_streak += 1
                    thinking_truncations += 1
                    recovery_msg = (
                        "Previous response hit max_tokens before emitting a tool call. "
                        "Emit a concise direct tool call now."
                    )
                    ledger.open_errors = [f"s{step_idx} OUTPUT_TRUNCATION: {recovery_msg}"]
                    ledger.error_streak += 1
                    ledger.record_user_feedback(recovery_msg)
                    meter.record_step_event(
                        run_id=run_id,
                        episode_id=episode_id,
                        task_id=task_id,
                        step_idx=step_idx,
                        tier=tier,
                        tool_name="__TRUNCATED__",
                        tool_args=None,
                        fsm_blocked=False,
                        fsm_reason=None,
                        tool_ok=False,
                        error_code="OUTPUT_TRUNCATION",
                        state_changed=False,
                        ledger_summary=ledger.compact_summary(),
                    )
                    if truncation_streak >= 2:
                        truncation_abort = True
                        break
                    continue

                truncation_streak = 0
                if use_fsm and router.params.use_stop_guard and ledger.premature_stop_count < router.params.max_stop_rejections:
                    stop_dec = fsm_guard.check_stop(
                        domain=domain,
                        snapshot=env.snapshot_state(),
                        ledger=ledger,
                    )
                    if not stop_dec.allowed:
                        ledger.premature_stop_count += 1
                        ledger.just_rejected_stop = True
                        ledger.error_streak += 1
                        precondition_blocks += 1
                        ledger.open_errors = [
                            f"s{step_idx} [STOP_GUARD] {stop_dec.reason}. {stop_dec.suggestion}"
                        ]
                        ledger.record_user_feedback(
                            f"Task NOT complete yet: {stop_dec.reason}. {stop_dec.suggestion}",
                            assistant_text=resp.content,
                        )
                        meter.record_step_event(
                            run_id=run_id,
                            episode_id=episode_id,
                            task_id=task_id,
                            step_idx=step_idx,
                            tier=tier,
                            tool_name="__STOP_REJECTED__",
                            tool_args=None,
                            fsm_blocked=True,
                            fsm_reason=stop_dec.reason,
                            tool_ok=False,
                            error_code="INCOMPLETE_STOP",
                            state_changed=False,
                            ledger_summary=ledger.compact_summary(),
                        )
                        continue

                meter.record_step_event(
                    run_id=run_id,
                    episode_id=episode_id,
                    task_id=task_id,
                    step_idx=step_idx,
                    tier=tier,
                    tool_name="__STOP__",
                    tool_args=None,
                    fsm_blocked=False,
                    fsm_reason=None,
                    tool_ok=True,
                    error_code=None,
                    state_changed=False,
                    ledger_summary=ledger.compact_summary(),
                )
                break

            truncation_streak = 0
            if prewrite is None:
                ledger.record_assistant_message(resp.raw_assistant_message)
            else:
                record_prewrite("batch", {"proposals": [{"id": call.id, "tool": call.name,
                                                        "arguments": call.arguments_dict, "parse_error": call.parse_error}
                                                       for call in resp.tool_calls]})

            for tc in resp.tool_calls:
                changed = False
                if prewrite is not None:
                    try:
                        resolved = resolve_proposal(
                            prewrite, tc, tier=tier, schemas=tools, read=supplement, review=review_candidate,
                            log=record_prewrite, can_review=ep_cost < ep_cap, rule=proposal_rule,
                        )
                    except (BudgetExceededError, AuthenticationError):
                        raise
                    except (APIConnectionError, APIStatusError) as exc:
                        if pause_on_transport_error and (isinstance(exc, (APIConnectionError, RateLimitError))
                                                         or exc.status_code >= 500):
                            raise
                        meter.finish_episode(
                            episode_id=episode_id, passed=False, failure_tag="API_ERROR", steps_taken=steps_taken,
                            total_cost_cny=round(ep_cost, 8), loop_abort=False,
                            summary={"review_api_error": type(exc).__name__, **episode_timing(meter.db_path, episode_id, started)},
                        )
                        return {"task_id": task_id, "passed": False, "failure_tag": "API_ERROR",
                                "steps_taken": steps_taken, "cost_cny": round(ep_cost, 8), "loop_abort": False}
                    ep_cost += resolved.cost
                    if resolved.call is None:
                        replan_fast = True
                        notice = "Pending proposal and remaining batch were not dispatched. "
                        ledger.record_user_feedback(notice + (resolved.feedback or "Replan using current public observations and propose the next action."))
                        break
                    tc = resolved.call
                    changed = resolved.changed
                    ledger.record_assistant_message(assistant_action(tc))
                total_tool_calls += 1
                fsm_blocked = False
                fsm_reason: str | None = None

                if tc.parse_error or tc.arguments_dict is None:
                    schema_errors += 1
                    args: dict[str, Any] = {}
                    res = ToolResult(
                        ok=False,
                        error_code="SCHEMA_PARSE_ERROR",
                        error_msg=tc.parse_error or "Invalid tool arguments",
                        is_write=False,
                    )
                else:
                    args = tc.arguments_dict
                    if use_fsm:
                        # Enabled research guards explicitly use the mock backend read replica.
                        pre_snapshot = env.snapshot_state()
                        decision = fsm_guard.check(
                            domain=domain,
                            tool_name=tc.name,
                            args=args,
                            snapshot=pre_snapshot,
                            ledger=ledger,
                        )
                        if not decision.allowed:
                            fsm_blocked = True
                            fsm_reason = decision.reason
                            precondition_blocks += 1
                            if decision.loop_triggered:
                                loop_aborted = True
                            res = ToolResult(
                                ok=False,
                                error_code="NO_PROGRESS_LOOP" if decision.loop_triggered else "FSM_BLOCKED",
                                error_msg=f"{decision.reason}. Hint: {decision.suggestion}",
                                is_write=env.is_write_tool(tc.name),
                            )
                            fsm_guard.record_outcome(tc.name, args, ok=False, snapshot=pre_snapshot)
                        else:
                            res = env.execute_tool(tc.name, args)
                            fsm_guard.record_outcome(tc.name, args, ok=res.ok, snapshot=pre_snapshot)
                    elif not research:
                        pre_snapshot = env.snapshot_state()
                        res = env.execute_tool(tc.name, args)
                        # Track loop repeats passively within unchanged state snapshots
                        skey = f"{canonical_snapshot_hash(pre_snapshot)}:{canonical_call_hash(tc.name, args)}"
                        fsm_guard.record_outcome(tc.name, args, ok=res.ok, snapshot=pre_snapshot)
                        if fsm_guard.fail_counts.get(skey, 0) >= router.params.max_repeat_same_call + 1:
                            loop_aborted = True
                    else:
                        res = env.execute_tool(tc.name, args)

                    if not fsm_blocked:
                        if not res.ok:
                            if res.error_code in ("UNKNOWN_TOOL", "INVALID_ARG"):
                                schema_errors += 1
                            else:
                                precondition_blocks += 1
                        if res.is_write:
                            write_calls += 1
                            if res.ok:
                                w_hash = canonical_call_hash(tc.name, args)
                                if w_hash in successful_write_hashes:
                                    duplicate_successful_actions += 1
                                successful_write_hashes.add(w_hash)

                if research_trace is not None and env.is_write_tool(tc.name):
                    research_trace["write_attempts"] += 1
                    research_trace["blocked_write_attempts"] += int(fsm_blocked)
                    research_trace["failed_write_attempts"] += int(not fsm_blocked and not res.ok)
                    if research_trace["first_write_step"] is None:
                        research_trace["first_write_step"] = step_idx
                    if res.ok and research_trace["first_successful_write_step"] is None:
                        research_trace["first_successful_write_step"] = step_idx
                    if tc.name == "restart_service" and research_trace["first_restart_step"] is None:
                        research_trace["first_restart_step"] = step_idx
                    if tc.name in RESEARCH_CONSTRAINED_TOOLS[domain]:
                        if research_trace["first_constrained_write_step"] is None:
                            research_trace["first_constrained_write_step"] = step_idx
                            first_proactive = research_trace["first_proactive_step"]
                            if first_proactive is not None:
                                research_trace["lead_steps"] = step_idx - first_proactive
                        research_trace["constrained_write_attempts"] += 1
                        research_trace["failed_constrained_write_attempts"] += int(not res.ok)

                if prewrite is not None and not fsm_blocked and tc.arguments_dict is not None and not tc.parse_error:
                    facts = prewrite.assess(tc.name, args)
                    if facts is not None:
                        debit = contract_debit(tc.name, res)
                        record_prewrite("dispatch", {"id": tc.id, "tool": tc.name, "target": facts.target,
                                                     "pools": facts.pools, "sources": facts.sources,
                                                     "facts_complete": not facts.missing,
                                                     "before": facts.before, "debited_count": debit,
                                                     "used_after_by_contract": facts.before["used"] + debit
                                                     if "used" in facts.before else None,
                                                     "ok": res.ok, "error": res.error_code})
                    receipt = prewrite.observe(tc.name, args, res)
                    if receipt is not None:
                        record_prewrite("receipt", {"id": tc.id, "tool": tc.name, "arguments": args,
                                                   "sequence": receipt.sequence, "observation": receipt.data})
                ledger.record_tool_message(tc.id, res.to_tool_message())

                ledger.update(
                    step_idx=step_idx,
                    tier=tier,
                    tool_name=tc.name,
                    args=args,
                    result=res,
                    fsm_blocked=fsm_blocked,
                )

                if research:
                    unchanged_streak = 0 if res.state_changed else unchanged_streak + 1
                else:
                    cur_state_sig = json.dumps(env.snapshot_state(), sort_keys=True)
                    unchanged_streak = unchanged_streak + 1 if cur_state_sig == prev_state_sig else 0
                    prev_state_sig = cur_state_sig

                meter.record_step_event(
                    run_id=run_id,
                    episode_id=episode_id,
                    task_id=task_id,
                    step_idx=step_idx,
                    tier=tier,
                    tool_name=tc.name,
                    tool_args=args,
                    fsm_blocked=fsm_blocked,
                    fsm_reason=fsm_reason,
                    tool_ok=res.ok,
                    error_code=res.error_code,
                    state_changed=res.state_changed,
                    ledger_summary=ledger.compact_summary(),
                )

                if changed:
                    # A different action invalidates any suffix proposed with the original action.
                    break
                if loop_aborted:
                    break

            if loop_aborted:
                break

        vres = verify_episode(env, task)
        stats = EpisodeTraceStats(
            passed=vres.passed,
            steps_taken=steps_taken,
            max_steps=max_steps,
            budget_exhausted=budget_exhausted,
            loop_aborted=loop_aborted,
            schema_errors=schema_errors,
            precondition_blocks=precondition_blocks,
            duplicate_successful_actions=duplicate_successful_actions,
            total_tool_calls=total_tool_calls,
            write_calls=write_calls,
            repeated_final_state_streak=unchanged_streak,
            truncation_abort=truncation_abort,
        )
        failure_tag = classify_failure(stats, vres.violations)
        meter.finish_episode(
            episode_id=episode_id,
            passed=vres.passed,
            failure_tag=failure_tag,
            steps_taken=steps_taken,
            total_cost_cny=round(ep_cost, 8),
            loop_abort=loop_aborted,
            summary={
                "violations": vres.violations,
                "checks": vres.checks_summary,
                "schema_errors": schema_errors,
                "precondition_blocks": precondition_blocks,
                "budget_exhausted": budget_exhausted,
                "thinking_truncations": thinking_truncations,
                "stop_rejections": ledger.premature_stop_count,
                "proactive_think_steps": ledger.proactive_think_steps,
                "step0_selected_tier": step0_selected_tier,
                **({"v3_supplement_reads": prewrite.reads_started, "v3_logical_review_started": prewrite.review_used}
                   if prewrite is not None else {}),
                **(research_trace or {}),
                **episode_timing(meter.db_path, episode_id, started),
            },
        )
        return {
            "task_id": task_id,
            "passed": vres.passed,
            "failure_tag": failure_tag,
            "steps_taken": steps_taken,
            "cost_cny": round(ep_cost, 8),
            "loop_abort": loop_aborted,
        }
    finally:
        env.close()


def run_split(
    *,
    split: str,
    policy: str,
    provider: str,
    stage: str = "dev",
    db_path: Path,
    budget_cap_cny: float,
    stage_cap_cny: float | None = None,
    run_id: str | None = None,
    resume: bool = False,
    use_fake: bool = False,
    fake_handler: FakeHandler | None = None,
    router_params: RouterParams | None = None,
    allow_locked_test: bool = False,
    data_dir: Path | None = None,
    limit: int | None = None,
    max_workers: int = 1,
    timeout_sec: float = 60.0,
    max_retries: int = 2,
    tariff_regime_override: str | None = None,
) -> dict[str, Any]:
    """Run an entire task split under strict billing, resume, and guardrail controls."""
    catalog = load_models_catalog(project_root=PROJECT_ROOT)
    params = router_params or load_router_params()
    tasks = load_split_tasks(split, allow_locked_test=allow_locked_test, data_dir=data_dir)
    if limit is not None:
        tasks = tasks[:limit]

    meter = BillingMeter(
        db_path,
        project_cap_cny=catalog.project_budget_cap_cny,
        holidays=catalog.holidays_2026,
    )
    rid = run_id or f"{stage}_{split}_{policy}_{uuid.uuid4().hex[:6]}"
    meter.create_or_resume_run(
        run_id=rid,
        stage=stage,
        split=split,
        policy=policy,
        provider=provider,
        round_cap_cny=budget_cap_cny,
        stage_cap_cny=stage_cap_cny if stage_cap_cny is not None else budget_cap_cny,
        metadata={"use_fake": use_fake, "params": params.model_dump()},
    )

    if use_fake:
        if fake_handler is None:
            raise ValueError("fake_handler must be provided when use_fake=True")
        pool: LiveModelPool | FakeModelPool = FakeModelPool(catalog, meter, fake_handler)
    else:
        pool = LiveModelPool(catalog, meter, timeout_sec=timeout_sec, max_retries=max_retries)

    completed = meter.get_completed_task_ids(rid) if resume else set()
    pending = [(idx, task) for idx, task in enumerate(tasks) if task["task_id"] not in completed]

    def _run_one(item: tuple[int, dict[str, Any]]) -> dict[str, Any]:
        idx, task = item
        task_params = params.model_copy(update={"random_seed": params.random_seed + idx})
        task_router = StepRouter(policy=policy, params=task_params)
        return run_episode(
            task=task,
            run_id=rid,
            provider=provider,
            router=task_router,
            pool=pool,
            meter=meter,
            tariff_regime_override=tariff_regime_override,
        )

    workers = max(1, int(max_workers))
    try:
        if workers == 1 or len(pending) <= 1:
            for item in pending:
                _run_one(item)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(_run_one, item) for item in pending]
                for fut in concurrent.futures.as_completed(futures):
                    try:
                        fut.result()
                    except Exception:
                        executor.shutdown(wait=False, cancel_futures=True)
                        raise
    except (BudgetExceededError, AuthenticationError):
        meter.finish_run(rid, status="ABORTED")
        raise

    meter.finish_run(rid, status="COMPLETED")
    return meter.summarize_run(rid)
