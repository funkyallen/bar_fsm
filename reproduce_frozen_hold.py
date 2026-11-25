"""Recompute the registered first-hold comparison from archived runs."""

import argparse
from collections import Counter, defaultdict
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import sys


ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "runtime"), str(ROOT / "runtime" / "src")]

from aroute.core.immutable import sha256_file
from aroute.core.model_pool import _request_hash
from scripts.research_complete_frozen_trial import bill
from scripts.research_feedback_and_transfer import replace_obligations
from scripts.research_v33_checkpoint_trial import _record, _spec
from scripts.research_v40_control import Assessment, feedback
from scripts.research_v40_trial import SYSTEM, _independent_state, _json_digest, _read_json, env_for
from scripts.research_v49_frozen_hold import ARMS, PROVIDER, summary


def replay(row, checkpoint, case, attempts):
    task = case["task"]
    prefix = checkpoint["trace"]
    assert row["checkpoint_state_sha256"] == checkpoint["state_sha256"]
    assert _json_digest(row["trace"][:len(prefix)]) == _json_digest(prefix)
    assert row["remaining_decisions"] == checkpoint["remaining_decisions"]
    assert len(attempts) <= row["remaining_decisions"]
    assert bill(attempts) == {key: row[key] for key in bill(attempts)}
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task["prompt"]}]
    with closing(env_for(task)) as env:
        for event in prefix:
            if event["event"] == "future_block":
                assert event == prefix[-1]
                assert _json_digest(env.snapshot_state()) == checkpoint["state_sha256"]
                break
            candidate = event["candidate"]
            result = env.execute_tool(candidate["name"], candidate["arguments"])
            assert json.loads(result.to_tool_message()) == event["reply"]
            step = event["step"]
            # Each registered first hold follows one topology call and one automatic read.
            call_id = {"dispatch": f"checkpoint_{step}", "auto_read": f"r{step}_0_0"}[event["event"]]
            _record(messages, _spec(candidate, call_id), result)
        assert _json_digest(messages) == row["shared_prompt_sha256"]
        text = feedback(Assessment(**replace_obligations(prefix[-1]["assessment"])), row["arm"])
        assert text == row["feedback_text"]
        messages.append({"role": "user", "content": text})
        first = attempts[0]
        assert first["step_idx"] == prefix[-1]["step"] + 1
        assert first["request_hash"] == _request_hash(PROVIDER, "fast", first["model_requested"], messages)

        for event in row["trace"][len(prefix):]:
            if event["event"] == "schema_hold":
                continue
            candidate = event["candidate"]
            if "oracle" in event:
                before, after = _independent_state(env, case, candidate)
                assert {"before": before, "after": after, "new_loss": before is True and after is False} == event["oracle"]
            if event["event"] in ("dispatch", "auto_read"):
                reply = env.execute_tool(candidate["name"], candidate["arguments"])
                assert json.loads(reply.to_tool_message()) == event["reply"]
        if row["terminal"] is not None:
            assert env.verify(task).passed == row["terminal"]
    later_holds = sum(e["event"] == "future_block" for e in row["trace"][len(prefix):])
    first_reply = json.loads(first["response_json"]) if first["status"] == "OK" else None
    stopped_before_call = bool(first_reply and first_reply["finish_reason"] == "stop"
                               and not first_reply["tool_calls"] and row["stop"] == "model_stop"
                               and len(attempts) == 1)
    return stopped_before_call, later_holds


def reproduce():
    paths = ROOT / "live/frozen_hold"
    protocol = _read_json(paths / "registration.json")
    result = _read_json(paths / "result.json")
    assert result["scope"] == "live" and result["registration_sha256"] == _json_digest(protocol)
    for path, digest in protocol["source_sha256"].items():
        assert sha256_file(ROOT / path) == digest
    scripts = ROOT / "runtime/scripts"
    assert sha256_file(scripts / "research_v49_frozen_hold.py") == protocol["runner_sha256"]
    assert sha256_file(scripts / "research_feedback_and_transfer.py") == protocol["prefix_pool_sha256"]
    groups = {p["id"]: p for p in protocol["checkpoints"]}
    tasks = {c["task"]["task_id"]: c for c in protocol["cases"]}
    rows = result["rows"]
    assert len(rows) <= len(protocol["allocation"])
    attempts = defaultdict(list)
    ledger_path = (paths / "meter.sqlite").as_uri() + "?mode=ro&immutable=1"
    with closing(sqlite3.connect(ledger_path, uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        records = list(conn.execute("SELECT * FROM api_attempts ORDER BY run_id, step_idx"))
        windows = list(conn.execute("SELECT MIN(created_at), MAX(updated_at) FROM api_attempts"))[0]
    for record in records:
        assert record["provider"] == PROVIDER and record["model_requested"] == protocol["models"][PROVIDER]
        attempts[record["run_id"]].append(record)
    hashes = defaultdict(set)
    early_stops, later_holds = Counter(), Counter()
    for slot, row in zip(protocol["allocation"][:len(rows)], rows, strict=True):
        assert slot == {key: row[key] for key in slot}
        stopped, holds = replay(row, groups[row["prefix"]], tasks[row["task"]], attempts.pop(row["run_id"]))
        early_stops[row["arm"]] += stopped
        later_holds[row["arm"]] += holds
        hashes[row["prefix"]].add(row["shared_prompt_sha256"])
    assert not attempts and all(len(values) == 1 for values in hashes.values())
    payload = summary(protocol, rows)
    assert payload == {key: result[key] for key in payload}
    ledger = bill(records)
    assert ledger == result["ledger"]
    assert all(not model.endswith("-fake") for model in ledger["actual_model_ids"])
    return {"status": result["scope"], "source_holds": sum(len(p["source_slots"]) for p in groups.values()),
            "later_holds_excluded": protocol["later_holds_excluded"],
            "model_window": {"first_request": windows[0], "last_attempt_update": windows[1]},
            "post_hold_diagnostics": {
                "stopped_before_next_tool_call": {arm: early_stops[arm] for arm in ARMS},
                "later_certificate_holds": {arm: later_holds[arm] for arm in ARMS},
            },
            "ledger": ledger, **payload}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report = reproduce()
    (args.out / "frozen_hold_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
