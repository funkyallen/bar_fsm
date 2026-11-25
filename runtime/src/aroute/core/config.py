"""Configuration loaders for providers, pricing, and orchestrator parameters."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class PricingConfig(BaseModel):
    """Per-1M-token prices in CNY."""

    peak_input_hit: float
    peak_input_miss: float
    peak_output: float
    offpeak_input_hit: float
    offpeak_input_miss: float
    offpeak_output: float


class TierConfig(BaseModel):
    """Configuration for a model tier (fast / think / pro)."""

    model_id: str
    model_id_env: str | None = None
    extra_body: dict[str, Any] = Field(default_factory=dict)
    max_tokens: int = 1024
    temperature: float | None = None
    reasoning_effort: str | None = None
    pricing: PricingConfig
    verified_snapshot_id: str | None = None
    probe_verified_at: str | None = None

    def resolve_model_id(self) -> str:
        """Resolve endpoint ID override from env if configured and valid."""
        if self.model_id_env:
            override = os.environ.get(self.model_id_env, "").strip()
            if override and not override.startswith("ep-replace"):
                return override
        return self.model_id


class ProviderConfig(BaseModel):
    """Configuration for a model API provider."""

    display_name: str
    api_key_env: str
    base_url_env: str
    default_base_url: str
    peak_schedule: str = "flat"
    tiers: dict[str, TierConfig]


class ModelsCatalog(BaseModel):
    """Root model catalog loaded from configs/models.yaml."""

    project_budget_cap_cny: float = 350.0
    holidays_2026: list[str] = Field(default_factory=list)
    providers: dict[str, ProviderConfig]


def load_env(project_root: Path | None = None) -> None:
    """Load .env from project root without overriding existing process env."""
    root = project_root or PROJECT_ROOT
    env_path = root / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)


def load_models_catalog(
    config_path: Path | None = None,
    project_root: Path | None = None,
) -> ModelsCatalog:
    """Load provider and tier definitions from YAML."""
    root = project_root or PROJECT_ROOT
    load_env(root)
    path = config_path or (root / "configs" / "models.yaml")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return ModelsCatalog.model_validate(raw)


def get_api_key(provider_cfg: ProviderConfig) -> str | None:
    """Return configured non-placeholder API key for a provider, or None."""
    val = os.environ.get(provider_cfg.api_key_env, "").strip()
    if not val or "replace-me" in val.lower():
        return None
    return val


def get_base_url(provider_cfg: ProviderConfig) -> str:
    """Resolve base URL from environment or fallback default."""
    val = os.environ.get(provider_cfg.base_url_env, "").strip()
    return val or provider_cfg.default_base_url


def available_providers(catalog: ModelsCatalog) -> list[str]:
    """Return provider names that currently have a configured API key."""
    return [name for name, cfg in catalog.providers.items() if get_api_key(cfg) is not None]
