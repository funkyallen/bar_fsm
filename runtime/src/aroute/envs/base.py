"""Base interface and structured data types for deterministic SQLite sandboxes."""

from __future__ import annotations

import json
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    """Structured tool execution result returned by a sandbox."""

    ok: bool
    data: dict[str, Any] = field(default_factory=dict)
    error_code: str | None = None
    error_msg: str | None = None
    state_changed: bool = False
    is_write: bool = False

    def to_tool_message(self) -> str:
        """Serialize as compact deterministic JSON for LLM tool observation."""
        payload: dict[str, Any] = {"ok": self.ok}
        if self.ok:
            payload["data"] = self.data
        else:
            payload["error_code"] = self.error_code or "TOOL_ERROR"
            payload["error_msg"] = self.error_msg or "Operation failed"
            if self.data:
                payload["data"] = self.data
        return json.dumps(payload, ensure_ascii=False)


@dataclass
class VerifyResult:
    """Deterministic final-state verification outcome."""

    passed: bool
    violations: list[str] = field(default_factory=list)
    checks_summary: dict[str, Any] = field(default_factory=dict)


class BaseSandboxEnv(ABC):
    """Abstract base class for in-memory SQLite deterministic tool sandboxes."""

    domain: str = "base"

    def __init__(self) -> None:
        self.conn: sqlite3.Connection | None = None
        self.task_spec: dict[str, Any] = {}
        self.precondition_violations: int = 0
        self.write_count: int = 0
        self.read_count: int = 0

    def _new_memory_conn(self) -> sqlite3.Connection:
        if self.conn is not None:
            self.conn.close()
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        self.conn = conn
        return conn

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    @abstractmethod
    def reset(self, task_spec: dict[str, Any]) -> None:
        """Reset SQLite tables cleanly from task specification."""

    @abstractmethod
    def tool_schemas(self) -> list[dict[str, Any]]:
        """Return OpenAI tool schemas for this environment."""

    @abstractmethod
    def is_write_tool(self, name: str) -> bool:
        """Return True if tool mutates sandbox state."""

    @abstractmethod
    def execute_tool(self, name: str, args: dict[str, Any]) -> ToolResult:
        """Execute a tool deterministically against the SQLite state."""

    @abstractmethod
    def snapshot_state(self) -> dict[str, Any]:
        """Return compact observable state snapshot for ledger/loop detection."""

    @abstractmethod
    def verify(self, task_spec: dict[str, Any]) -> VerifyResult:
        """Deterministically verify terminal state against task assertions."""
