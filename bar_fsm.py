"""Run BAR-FSM's synthetic checks or reproduce the paper with its data archive."""
from argparse import ArgumentParser
from hashlib import file_digest
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile
import json
import os
import shutil
import socket
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
ARCHIVE_NAME = "BAR-FSM-reproduction-v5.2.zip"
ARCHIVE_SHA256 = "f552625db9ae9aa739e0a31e9271a3e55a3b8574cb1b9e049d1f708461d33bd2"
LEGACY_ARCHIVE_SHA256 = "c37ed473c6a363904486dcb431d9fcd3ec9ff56ddbb13547abc3ef6c840e4703"
REFERENCE_OUTPUTS = (
    "table1.csv", "family_effects.csv", "accounting_contract_audit.json", "trace_forensics.json",
    "frozen_hold_audit.json", "control_versions_v50.json", "oracle_v50.json", "protocol_v50.json",
    "request_call_map_v50.jsonl", "evidence_v50.json", "formal_episode_exposure.csv",
    "formal_candidate_exposure.csv", "natural_accounting_exposure.csv", "qualification_stages_v51.json",
)


def deny_network(*_args, **_kwargs):
    raise RuntimeError("Model and network access are disabled for offline checks.")


def synthetic(check, output):
    # These checks use the native sandbox and have no reason to open a connection.
    socket.socket.connect = deny_network
    socket.socket.connect_ex = deny_network
    socket.create_connection = deny_network
    sys.path[:0] = [str(ROOT / "runtime"), str(ROOT / "runtime/src")]
    report, summary = check()
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def demo_report():
    from scripts.research_accounting_void_contract import qualify

    report = qualify()
    # A free void clears the residual that the archived guard counted against the allowance.
    example = report["hidden_voidability_counterexample"]
    installments = report["installment_witness"]
    return report, {
        "scope": "deterministic synthetic controls",
        "archived_guard_on_hidden_free_void": example["old_guard"]["result"],
        "hidden_free_void_native_completion": example["native_completion_on_continuation"],
        "known_no_void_deficit": report["known_no_void_positive_control"]["authoritative_receipt"]["result"],
        "one_cent_posting_preserves_reachability": installments["candidate_plus_one"]["native_completion_on_continuation"],
        "two_cent_posting_preserves_reachability": installments["candidate_plus_two"]["native_completion_on_continuation"],
    }


def qualification_report():
    from scripts.research_voidability_qualification import run

    report = run()
    summary = report["summary"]["conservative_with_public_receipt"]["all"]
    expected = {
        "ineligible": 366, "initially_unreachable": 62, "true_block": 21,
        "true_pass": 131, "unknown_loss": 15, "unknown_preserving": 125,
    }
    if len(report["rows"]) != 720 or summary != expected:
        raise ValueError(f"Qualification differs from the paper: {summary}")
    return report, summary


def checksum(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def reproduce(archive, output):
    # Verify the archive before extraction, including the upstream source it contains.
    if checksum(archive) not in (ARCHIVE_SHA256, LEGACY_ARCHIVE_SHA256):
        raise ValueError(f"Use the frozen {ARCHIVE_NAME}. The supplied archive has a different SHA-256.")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Use an empty directory for the reproduced outputs.")

    provenance = json.loads((ROOT / "frozen_sources.sha256.json").read_text(encoding="utf-8"))
    with TemporaryDirectory(prefix="bar-fsm-reproduction-") as temporary:
        with ZipFile(archive) as zipped:
            zipped.extractall(temporary)
        # Both verified archives contain a single top-level directory.
        inputs = next(Path(temporary).iterdir())
        # Pair the archived data with the source from this checkout.
        for relative, digest in provenance.items():
            source = ROOT / relative
            if checksum(source) != digest:
                raise ValueError(f"The release source changed: {relative}")
            shutil.copyfile(source, inputs / relative)

        environment = {key: value for key, value in os.environ.items()
                       if not any(marker in key.upper() for marker in ("API_KEY", "TOKEN", "SECRET", "PYTHONPATH"))}
        subprocess.run([sys.executable, "-B", "-E", "reproduce_paper.py", "--out", str(output)],
                       cwd=inputs, env=environment, check=True)
        for name in REFERENCE_OUTPUTS:
            actual = (output / name).read_bytes().replace(b"\r\n", b"\n")
            expected = (inputs / "source_data" / name).read_bytes().replace(b"\r\n", b"\n")
            if actual != expected:
                raise ValueError(f"Reproduced output differs from the archived evidence: {name}")
    print(f"Matched {len(REFERENCE_OUTPUTS)} archived outputs. Tables and plots are in {output}.")


def main():
    parser = ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    for command, check, purpose in (
        ("demo", demo_report, "Run four synthetic accounting controls."),
        ("qualify", qualification_report, "Recheck the 720 synthetic state–action pairs."),
    ):
        child = subcommands.add_parser(command, help=purpose)
        child.add_argument("--out", type=Path, help="Optional JSON output path.")
        child.set_defaults(check=check)
    full = subcommands.add_parser("reproduce", help="Regenerate the paper's tables and figures from the v5.2 data archive.")
    full.add_argument("--archive", required=True, type=Path)
    full.add_argument("--out", required=True, type=Path, help="Empty directory for the reproduced outputs.")
    args = parser.parse_args()
    try:
        if args.command == "reproduce":
            reproduce(args.archive, args.out.resolve())
        else:
            synthetic(args.check, args.out)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"BAR-FSM: {exc}\n")


if __name__ == "__main__":
    main()
