"""Apply the research contracts at the existing runner's dispatch boundary."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Callable

from aroute.envs.base import ToolResult
from aroute.orchestrator.prewrite import READ_ONLY, Assessment, PrewriteControl

from .model_pool import ModelResponse, ToolCallSpec


ProposalRule = Callable[[PrewriteControl, ToolCallSpec], tuple[ToolCallSpec | None, dict | None]]


@dataclass(frozen=True)
class Dispatch:
    call: ToolCallSpec | None
    cost: float = 0.0
    changed: bool = False
    feedback: str | None = None


def assistant_action(call: ToolCallSpec) -> dict:
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": call.id, "type": "function", "function": {"name": call.name, "arguments": call.arguments_raw}},
    ]}


def review_instruction(call: ToolCallSpec, schemas: list[dict]) -> str:
    data = {"candidate": {"name": call.name, "arguments": call.arguments_dict},
            "tools": [s["function"] for s in schemas]}
    return (
        "Review this one pending proposal using only the preceding public observations. "
        "You have no tools in this review. Return one JSON object and no other text. "
        'Use {"decision":"keep"}, {"decision":"cancel"}, or '
        '{"decision":"replace","name":"tool_name","arguments":{...}}. '
        "A replacement can be a public query. The application will check it again before dispatch.\n"
        + json.dumps(data, ensure_ascii=False, sort_keys=True)
    )


def review_decision(content: str, call: ToolCallSpec, schemas: list[dict]) -> tuple[str, ToolCallSpec | None]:
    try:
        reply = json.loads(content)
    except (TypeError, ValueError):
        return "invalid_review", None
    if reply == {"decision": "keep"}:
        return "keep", call
    if reply == {"decision": "cancel"}:
        return "cancel", None
    if not isinstance(reply, dict) or set(reply) != {"decision", "name", "arguments"} or reply["decision"] != "replace":
        return "invalid_review", None
    name, args = reply["name"], reply["arguments"]
    schema = next((item["function"]["parameters"] for item in schemas if item["function"]["name"] == name), None)
    if schema is None or not isinstance(args, dict) or not set(schema.get("required", [])) <= args.keys() <= schema["properties"].keys():
        return "invalid_review", None
    if name == call.name and args == call.arguments_dict:
        return "keep", call
    return "replace", ToolCallSpec(call.id, name, json.dumps(args, ensure_ascii=False), args)


def resolve_proposal(
    control: PrewriteControl, call: ToolCallSpec, *, tier: str, schemas: list[dict],
    read: Callable[[ToolCallSpec], ToolResult], review: Callable[[str], ModelResponse],
    log: Callable[[str, dict], None], can_review: bool, rule: ProposalRule | None = None,
) -> Dispatch:
    """Resolve one candidate without firing a business write or consulting a backend snapshot."""
    if call.parse_error or call.arguments_dict is None:
        return Dispatch(call)

    def assess(candidate: ToolCallSpec, event: str) -> Assessment | None:
        result = control.assess(candidate.name, candidate.arguments_dict)
        if result is not None:
            log(event, {"id": candidate.id, "tool": candidate.name, **asdict(result)})
        return result

    def defer(result: Assessment | None, candidate: ToolCallSpec) -> bool:
        if not (control.supplement and result and result.missing):
            return False
        for index, requirement in enumerate(control.pending(result)):
            if requirement.tool not in READ_ONLY:
                raise ValueError("Prewrite can only issue an explicitly allowlisted public read.")
            control.reads_started += 1
            args = requirement.arguments
            read(ToolCallSpec(f"{candidate.id}_prewrite_{index}", requirement.tool,
                              json.dumps(args, ensure_ascii=False), args))
            if requirement not in control.receipts:
                control.failed_reads.add(requirement)
        log("defer_for_facts", {"id": candidate.id, "unresolved": sorted(r.tool for r in result.missing
                                                                        if r not in control.receipts)})
        return True

    def apply_rule(candidate: ToolCallSpec, event: str) -> tuple[ToolCallSpec | None, str | None]:
        if rule is None:
            return candidate, None
        revised, decision = rule(control, candidate)
        if decision is None:
            return revised, None
        log(event, {"id": candidate.id, "tool": candidate.name, "original": candidate.arguments_dict,
                    "effective": revised.arguments_dict if revised else None, **decision})
        return revised, decision.get("feedback")

    original = assess(call, "assessment")
    if defer(original, call):
        return Dispatch(None)
    revised, feedback = apply_rule(call, "rule_before_review")
    if revised is None:
        return Dispatch(None, feedback=feedback)
    changed = revised != call
    if changed:
        call = revised
        original = assess(call, "rule_assessment")
    if original is None or not control.wants_review(original, tier):
        return Dispatch(call, changed=changed)
    if not can_review:
        log("review_budget_unavailable", {"id": call.id})
        return Dispatch(call, changed=changed)

    control.review_used = True
    evidence = {"remaining": original.before.get("remaining"), "arguments": call.arguments_dict, "tool": call.name} if rule is not None else {}
    log("review_started", {"id": call.id, "target": original.target, "pools": original.pools, **evidence})
    response = review(review_instruction(call, schemas))
    disposition, replacement = review_decision(response.content, call, schemas)
    if response.tool_calls or response.finish_reason == "length":
        disposition, replacement = "invalid_review", None
    log("review_result", {"id": call.id, "disposition": disposition, "replacement_tool": replacement.name if replacement else None,
                          "replacement_args": replacement.arguments_dict if replacement else None})
    if replacement is not None and disposition == "replace":
        if defer(assess(replacement, "replacement_assessment"), replacement):
            replacement = None
        else:
            replacement, feedback = apply_rule(replacement, "rule_after_review")
    return Dispatch(replacement, response.cost_cny, changed=changed or disposition != "keep", feedback=feedback)


def contract_debit(tool: str, outcome: ToolResult) -> int:
    if tool == "restart_service":
        # The frozen mock backend increments before checking these startup errors.
        return int(outcome.ok or outcome.error_code in ("CORRUPT_ACTIVE_CONFIG", "POLICY_VIOLATION", "UPSTREAM_NOT_HEALTHY",
                                                       "UPSTREAM_INCOMPATIBLE", "CLUSTER_MEM_EXCEEDED"))
    if tool == "post_adjustment":
        return int(outcome.ok)
    return 0
