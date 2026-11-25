"""Unified deterministic terminal state verifier."""

from __future__ import annotations

from typing import Any

from aroute.envs.base import BaseSandboxEnv, VerifyResult


def verify_episode(env: BaseSandboxEnv, task_spec: dict[str, Any]) -> VerifyResult:
    """Verify final sandbox state deterministically against task assertions."""
    return env.verify(task_spec)
