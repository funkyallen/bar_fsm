"""SHA-256 manifest freezing, tamper detection, and AST anti-cheating audit."""

from __future__ import annotations

import ast
import hashlib
import re
from pathlib import Path

from .config import PROJECT_ROOT

TASK_ID_PATTERN = re.compile(r"\b(?:sc|db|ops)_\d{3}\b")
FORBIDDEN_STRINGS = ("golden_plans_offline_only", "golden_plans", "tasks_test_locked")
FORBIDDEN_IMPORTS = {"sqlite3", "openai"}


def _immutable_targets(root: Path) -> list[Path]:
    paths: list[Path] = []
    envs_dir = root / "src" / "aroute" / "envs"
    if envs_dir.exists():
        paths.extend(sorted(envs_dir.glob("*.py")))

    data_dir = root / "data"
    if data_dir.exists():
        paths.extend(sorted(data_dir.glob("*.json")))

    for rel in (
        "src/aroute/core/verifier.py",
        "src/aroute/core/billing_meter.py",
        "src/aroute/core/model_pool.py",
        "src/aroute/core/failure_tagger.py",
        "configs/models.yaml",
        "configs/val_anchors.yaml",
    ):
        p = root / rel
        if p.exists():
            paths.append(p)
    return sorted({p.resolve() for p in paths})


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(root: Path | None = None) -> Path:
    base = (root or PROJECT_ROOT).resolve()
    targets = _immutable_targets(base)
    manifest_path = base / "experiments" / "manifest.sha256"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{sha256_file(p)}  {p.relative_to(base).as_posix()}" for p in targets]
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest_path


def verify_manifest(root: Path | None = None) -> list[str]:
    """Return list of integrity violation messages (empty if valid)."""
    base = (root or PROJECT_ROOT).resolve()
    manifest_path = base / "experiments" / "manifest.sha256"
    if not manifest_path.exists():
        return [f"Manifest missing: {manifest_path}"]

    errors: list[str] = []
    seen_rel: set[str] = set()
    for line_no, raw in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split("  ", 1)
        if len(parts) != 2:
            errors.append(f"Malformed manifest line {line_no}: {raw}")
            continue
        expected_hash, rel = parts
        seen_rel.add(rel)
        target = base / rel
        if not target.exists():
            errors.append(f"Missing immutable file: {rel}")
            continue
        actual_hash = sha256_file(target)
        if actual_hash != expected_hash:
            errors.append(f"Hash mismatch for {rel}: expected {expected_hash[:12]}, got {actual_hash[:12]}")

    if not seen_rel:
        errors.append("Manifest contains zero entries")
    return errors


def audit_orchestrator_ast(root: Path | None = None) -> list[str]:
    """Audit src/aroute/orchestrator/*.py for hardcoded task IDs or prohibited imports."""
    base = (root or PROJECT_ROOT).resolve()
    orch_dir = base / "src" / "aroute" / "orchestrator"
    if not orch_dir.exists():
        return [f"Missing orchestrator directory: {orch_dir}"]

    violations: list[str] = []
    for py_file in sorted(orch_dir.glob("*.py")):
        src = py_file.read_text(encoding="utf-8")
        try:
            tree = ast.parse(src, filename=str(py_file))
        except SyntaxError as exc:
            violations.append(f"SyntaxError in {py_file.name}: {exc}")
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    if top in FORBIDDEN_IMPORTS:
                        violations.append(f"{py_file.name}:{node.lineno}: forbidden import '{alias.name}'")
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    top = node.module.split(".")[0]
                    if top in FORBIDDEN_IMPORTS:
                        violations.append(f"{py_file.name}:{node.lineno}: forbidden from-import '{node.module}'")
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                text = node.value
                if TASK_ID_PATTERN.search(text):
                    violations.append(f"{py_file.name}:{node.lineno}: hardcoded task ID literal '{text}'")
                for kw in FORBIDDEN_STRINGS:
                    if kw in text:
                        violations.append(f"{py_file.name}:{node.lineno}: forbidden reference '{kw}'")
    return violations
