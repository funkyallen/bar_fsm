"""Exact finite search for the declared v4.0 experimental domains.

The transition search never evaluates the controller's lower bound. Amounts are
integer cents and every authorized amount is enumerated, including splits.
"""

from collections import deque
from functools import lru_cache


@lru_cache(maxsize=4096)
def accounting_reachable(residual, quotas, headroom, duplicate_amounts, ticket_for, cap):
    start = residual, quotas, headroom, duplicate_amounts
    pending = deque([start])
    seen = {start}
    while pending:
        delta, remaining, balance, duplicates = pending.popleft()
        if not any(delta) and not any(duplicates):
            return True
        for i, need in enumerate(delta):
            new_delta, new_quota, new_duplicates = list(delta), list(remaining), list(duplicates)
            next_steps = []
            if duplicates[i]:
                change = -duplicates[i]
                new_duplicates[i] = 0
                next_steps.append(action_accounting(delta, remaining, balance, new_duplicates, i, change, None))
            elif need and remaining[ticket_for[i]]:
                direction = 1 if need > 0 else -1
                for amount in range(1, min(abs(need), cap) + 1):
                    change = direction * amount
                    if balance + change >= 0:
                        next_steps.append(action_accounting(delta, new_quota, balance, duplicates, i, change, ticket_for[i]))
            for state in next_steps:
                if state not in seen:
                    seen.add(state)
                    pending.append(state)
    return False


def action_accounting(residual, quotas, headroom, duplicates, invoice, amount, ticket):
    after = list(residual)
    after[invoice] -= amount
    quotas = list(quotas)
    if ticket is not None:
        quotas[ticket] -= 1
    return tuple(after), tuple(quotas), headroom + amount, tuple(duplicates)


def service_restart(state, sid, revision, deps):
    running, healthy, quotas = state
    if quotas[sid] <= 0 or any(not healthy[d] or running[d] < minimum for d, minimum in deps[sid]):
        return None
    new_revisions, new_health, new_quota = list(running), list(healthy), list(quotas)
    new_revisions[sid], new_health[sid], new_quota[sid] = revision, True, quotas[sid] - 1
    return tuple(new_revisions), tuple(new_health), tuple(new_quota)


@lru_cache(maxsize=4096)
def service_reachable(state, required, deps):
    """Collapse free staging into restart choices; probes are free final actions."""
    pending = deque([state])
    seen = {state}
    while pending:
        current = pending.popleft()
        running, healthy, _ = current
        if (running == required and all(healthy)
                and all(healthy[d] and running[d] >= minimum for row in deps for d, minimum in row)):
            return True
        for i in range(len(required)):
            for revision in (0, 1):
                if healthy[i] and running[i] == revision:
                    continue
                after = service_restart(current, i, revision, deps)
                if after is not None and after not in seen:
                    seen.add(after)
                    pending.append(after)
    return False


def candidate_truth(case):
    model = case["truth"]
    if case["task"]["domain"] == "db_transaction":
        residual, quotas, cash, dupe, ticket_for, cap = (
            tuple(model["residual"]), tuple(model["quotas"]), model["headroom"], tuple(model["duplicates"]),
            tuple(model["ticket_for"]), model["cap"])
        before = accounting_reachable(residual, quotas, cash, dupe, ticket_for, cap)
        i = model["candidate_index"]
        value = case["candidate"]["arguments"]["delta_cents"]
        ticket = ticket_for[i]
        valid = (not dupe[i] and quotas[ticket] > 0 and cash + value >= 0 and value * residual[i] > 0
                 and abs(value) <= min(cap, abs(residual[i])))
        if not valid:
            return before, None
        state = action_accounting(residual, quotas, cash, dupe, i, value, ticket)
        return before, accounting_reachable(*state, ticket_for, cap)
    deps = tuple(tuple(tuple(edge) for edge in d) for d in model["deps"])
    required = tuple(model["required"])
    state = tuple(model["running"]), tuple(model["healthy"]), tuple(model["quotas"])
    before = service_reachable(state, required, deps)
    i = model["candidate_index"]
    after = service_restart(state, i, model["staged"][i], deps)
    return before, None if after is None else service_reachable(after, required, deps)
