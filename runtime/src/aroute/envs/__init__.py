"""Deterministic local SQLite sandbox environments for multi-step tool agents."""

from .base import BaseSandboxEnv, ToolResult, VerifyResult
from .db_transaction_env import DbTransactionEnv
from .ops_troubleshoot_env import OpsTroubleshootEnv
from .supply_chain_env import SupplyChainEnv


def create_env(domain: str) -> BaseSandboxEnv:
    """Instantiate a sandbox environment by domain name."""
    registry: dict[str, type[BaseSandboxEnv]] = {
        "supply_chain": SupplyChainEnv,
        "db_transaction": DbTransactionEnv,
        "ops_troubleshoot": OpsTroubleshootEnv,
    }
    if domain not in registry:
        raise ValueError(f"Unknown sandbox domain: {domain}")
    return registry[domain]()


__all__ = [
    "BaseSandboxEnv",
    "ToolResult",
    "VerifyResult",
    "SupplyChainEnv",
    "DbTransactionEnv",
    "OpsTroubleshootEnv",
    "create_env",
]
