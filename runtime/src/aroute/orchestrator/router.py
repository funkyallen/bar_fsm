"""Budget-aware adaptive thinking router V3 and ablation policy definitions."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from aroute.core.config import PROJECT_ROOT
from .memory_ledger import StateLedger


class RouterParams(BaseModel):
    """Mutable routing and guardrail hyperparameters."""

    ledger_window_k: int = 6
    rho_min: float = 0.20
    tau_err: int = 1
    tau_loop: int = 2
    max_repeat_same_call: int = 2
    episode_budget_cap_cny: float = 0.50
    use_fsm_guard: bool = True
    use_state_ledger: bool = False
    use_budget_throttle: bool = True
    enable_initial_think_on_hard: bool = False
    random_think_prob: float = 0.20
    random_seed: int = 20261120

    # V3 BAR-FSM structural risk & Stop-Guard parameters
    w_quota: float = 0.35
    w_couple: float = 0.35
    w_math: float = 0.30
    tau_risk: float = 0.70
    max_proactive_think_steps: int = 3
    max_stop_rejections: int = 2
    rho_pro_min: float = 0.35
    use_proactive_risk: bool = True
    use_stop_guard: bool = True

    # Stage 5C 2D BAR-FSM-Auto & Peak-aware SLA guardrail parameters
    auto_tau_risk_low: float = 0.35
    auto_max_proactive_steps: int = 4
    auto_hold_steps: int = 2
    auto_peak_rho_pro_min: float = 0.50

    # Stage 5D Cross-provider Tariff x KV-Cache hysteresis parameters
    hyst_alpha_cache: float = 0.88
    hyst_expected_think_out_tok: int = 1200
    hyst_delta_new_in_tok: int = 300
    hyst_margin_cny: float = 0.0005


def compute_cross_provider_think_surplus_cny(
    prompt_tokens: int,
    ds_rates: tuple[float, float, float],
    qw_rates: tuple[float, float, float],
    params: RouterParams,
) -> float:
    """Return expected single-step net CNY surplus of switching a Think call from DeepSeek to DashScope.

    Positive surplus indicates switching to uncached DashScope saves more on output/new tokens
    than it loses from DeepSeek implicit prefix cache invalidation.
    """
    ds_hit, ds_miss, ds_out = ds_rates
    _, qw_miss, qw_out = qw_rates
    alpha = max(0.0, min(1.0, float(params.hyst_alpha_cache)))
    prefix = max(0, int(prompt_tokens))
    delta_in = max(0, int(params.hyst_delta_new_in_tok))
    out_tok = max(0, int(params.hyst_expected_think_out_tok))

    cost_sticky_ds = (
        alpha * prefix * ds_hit
        + ((1.0 - alpha) * prefix + delta_in) * ds_miss
        + out_tok * ds_out
    ) / 1_000_000.0
    cost_switch_qw = ((prefix + delta_in) * qw_miss + out_tok * qw_out) / 1_000_000.0
    return cost_sticky_ds - cost_switch_qw


def load_router_params(config_path: Path | None = None) -> RouterParams:
    path = config_path or (PROJECT_ROOT / "configs" / "router_params.yaml")
    if not path.exists():
        return RouterParams()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return RouterParams.model_validate(raw)


class StepRouter:
    """Step-level thinking tier router supporting BAR-FSM and baseline/ablation policies."""

    VALID_POLICIES = (
        "all_fast",
        "all_think",
        "all_pro",
        "random_router",
        "fsm_only",
        "reactive_only",
        "bar_fsm_no_ledger",
        "bar_fsm_rewrite",
        "adaptive",
        "bar_fsm",
        "bar_fsm_pro",
        "bar_fsm_3tier",
        "step0_auto",
        "bar_fsm_auto",
        "hetero_greedy_oblivious",
        "hetero_sticky_offpeak",
        "hetero_sticky_peak",
    )

    def __init__(self, policy: str, params: RouterParams | None = None) -> None:
        if policy not in self.VALID_POLICIES:
            raise ValueError(f"Unsupported policy '{policy}', choose from {self.VALID_POLICIES}")
        self.policy = policy
        self.params = params or load_router_params()
        self._rng = random.Random(self.params.random_seed)
        self.locked_step0_tier: str = "think_med"
        self.last_route_event: dict[str, Any] | None = None

    def fsm_enabled(self) -> bool:
        if self.policy in ("all_fast", "all_think", "all_pro", "random_router", "step0_auto"):
            return False
        if self.policy in ("fsm_only", "reactive_only", "bar_fsm_no_ledger", "bar_fsm_rewrite"):
            return True
        return bool(self.params.use_fsm_guard)

    def ledger_enabled(self) -> bool:
        if self.policy == "bar_fsm_rewrite":
            return True
        if self.policy == "adaptive":
            return bool(self.params.use_state_ledger)
        return False

    def _select_bar_fsm_auto_tier(
        self,
        *,
        ledger: StateLedger,
        remaining_budget_ratio: float,
        domain: str,
        snapshot: dict[str, Any] | None,
        active_pricing_period: str,
    ) -> str:
        if self.params.use_budget_throttle and remaining_budget_ratio <= self.params.rho_min:
            ledger.escalation_hold_steps = 0
            ledger.escalation_hold_tier = None
            return "fast"

        eff_rho_pro = (
            max(self.params.rho_pro_min, self.params.auto_peak_rho_pro_min)
            if active_pricing_period == "peak"
            else self.params.rho_pro_min
        )

        # Reactive crisis ladder with sticky-until-write holds to prevent read-reset thrashing
        if (
            ledger.error_streak >= self.params.tau_err
            or ledger.repeat_count >= self.params.tau_loop
            or ledger.just_rejected_stop
        ):
            ledger.crisis_episodes += 1
            if (
                ledger.error_streak >= 3
                or (ledger.premature_stop_count >= 2 and ledger.error_streak >= 1)
                or ledger.crisis_episodes >= 4
            ):
                if remaining_budget_ratio > eff_rho_pro:
                    ledger.arm_escalation_hold("pro", self.params.auto_hold_steps)
                    return "pro"
                ledger.arm_escalation_hold("think", self.params.auto_hold_steps)
                return "think"
            if (
                ledger.error_streak >= 2
                or ledger.premature_stop_count >= 2
                or ledger.crisis_episodes >= 3
            ):
                ledger.arm_escalation_hold("think", self.params.auto_hold_steps)
                return "think"
            if ledger.error_streak >= 1 or ledger.just_rejected_stop:
                ledger.arm_escalation_hold("think_med", self.params.auto_hold_steps)
                return "think_med"
            ledger.arm_escalation_hold("think_low", 1)
            return "think_low"

        # Carry active hold across preparatory read turns until corrective write succeeds
        if ledger.escalation_hold_steps > 0 and ledger.escalation_hold_tier:
            return ledger.escalation_hold_tier

        # Proactive high-risk planning (full capacity before irreversible writes)
        if (
            self.params.use_proactive_risk
            and domain
            and snapshot is not None
            and ledger.proactive_think_steps < self.params.max_proactive_think_steps
        ):
            risk, _ = ledger.compute_structural_risk(
                domain=domain,
                snapshot=snapshot,
                w_quota=self.params.w_quota,
                w_couple=self.params.w_couple,
                w_math=self.params.w_math,
            )
            if risk >= self.params.tau_risk:
                ledger.proactive_think_steps += 1
                return "think"

        return "fast"

    def select_tier(
        self,
        *,
        ledger: StateLedger,
        remaining_budget_ratio: float,
        step_idx: int,
        difficulty: str = "",
        domain: str = "",
        snapshot: dict[str, Any] | None = None,
        active_pricing_period: str = "offpeak",
    ) -> str:
        _ = (step_idx, difficulty)  # difficulty is intentionally unused (zero label leakage)

        if self.policy == "step0_auto":
            return self.locked_step0_tier
        if self.policy == "bar_fsm_auto":
            return self._select_bar_fsm_auto_tier(
                ledger=ledger,
                remaining_budget_ratio=remaining_budget_ratio,
                domain=domain,
                snapshot=snapshot,
                active_pricing_period=active_pricing_period,
            )

        if self.policy in ("all_fast", "fsm_only"):
            return "fast"
        if self.policy == "all_think":
            return "think"
        if self.policy == "all_pro":
            return "pro"
        if self.policy == "random_router":
            return "think" if self._rng.random() < self.params.random_think_prob else "fast"

        # Adaptive / BAR-FSM policy family
        if self.params.use_budget_throttle and remaining_budget_ratio <= self.params.rho_min:
            return "fast"

        esc_tier = "pro" if self.policy == "bar_fsm_pro" else "think"

        # Reactive crisis trigger (error streak, repeat loop, or Stop-Guard rejection)
        if ledger.error_streak >= self.params.tau_err or ledger.repeat_count >= self.params.tau_loop or ledger.just_rejected_stop:
            if (
                self.policy == "bar_fsm_3tier"
                and (ledger.error_streak >= 2 or ledger.premature_stop_count >= 2)
                and remaining_budget_ratio > self.params.rho_pro_min
            ):
                return "pro"
            return esc_tier

        # Proactive structural risk trigger (skipped on reactive_only ablation)
        if self.policy != "reactive_only" and self.params.use_proactive_risk:
            if domain and snapshot is not None and ledger.proactive_think_steps < self.params.max_proactive_think_steps:
                risk, _ = ledger.compute_structural_risk(
                    domain=domain,
                    snapshot=snapshot,
                    w_quota=self.params.w_quota,
                    w_couple=self.params.w_couple,
                    w_math=self.params.w_math,
                )
                if risk >= self.params.tau_risk:
                    ledger.proactive_think_steps += 1
                    return esc_tier

        return "fast"

    def _canonical_bar_fsm_3tier(
        self,
        *,
        ledger: StateLedger,
        remaining_budget_ratio: float,
        domain: str,
        snapshot: dict[str, Any] | None,
    ) -> str:
        if self.params.use_budget_throttle and remaining_budget_ratio <= self.params.rho_min:
            return "fast"
        if (
            ledger.error_streak >= self.params.tau_err
            or ledger.repeat_count >= self.params.tau_loop
            or ledger.just_rejected_stop
        ):
            if (
                (ledger.error_streak >= 2 or ledger.premature_stop_count >= 2)
                and remaining_budget_ratio > self.params.rho_pro_min
            ):
                return "pro"
            return "think"
        if self.params.use_proactive_risk and domain and snapshot is not None:
            if ledger.proactive_think_steps < self.params.max_proactive_think_steps:
                risk, _ = ledger.compute_structural_risk(
                    domain=domain,
                    snapshot=snapshot,
                    w_quota=self.params.w_quota,
                    w_couple=self.params.w_couple,
                    w_math=self.params.w_math,
                )
                if risk >= self.params.tau_risk:
                    ledger.proactive_think_steps += 1
                    return "think"
        return "fast"

    def select_route(
        self,
        *,
        default_provider: str,
        ledger: StateLedger,
        remaining_budget_ratio: float,
        step_idx: int,
        difficulty: str = "",
        domain: str = "",
        snapshot: dict[str, Any] | None = None,
        active_pricing_period: str = "offpeak",
        provider_think_rates: dict[str, tuple[float, float, float]] | None = None,
    ) -> tuple[str, str]:
        """Select (provider, tier) for the next step across homogeneous or heterogeneous pools."""
        if self.policy not in ("hetero_greedy_oblivious", "hetero_sticky_offpeak", "hetero_sticky_peak"):
            tier = self.select_tier(
                ledger=ledger,
                remaining_budget_ratio=remaining_budget_ratio,
                step_idx=step_idx,
                difficulty=difficulty,
                domain=domain,
                snapshot=snapshot,
                active_pricing_period=active_pricing_period,
            )
            return default_provider, tier

        _ = difficulty  # zero oracle label leakage
        tier = self._canonical_bar_fsm_3tier(
            ledger=ledger,
            remaining_budget_ratio=remaining_budget_ratio,
            domain=domain,
            snapshot=snapshot,
        )

        if self.policy == "hetero_greedy_oblivious":
            if tier == "fast":
                return "dashscope", "fast"
            if tier == "pro":
                return "dashscope", "pro"
            return ("dashscope", "think") if active_pricing_period == "peak" else ("deepseek", "think")

        if self.policy == "hetero_sticky_offpeak":
            if tier == "pro":
                return "dashscope", "pro"
            return "deepseek", tier

        # hetero_sticky_peak: anchor fast on DeepSeek, offload Pro crises, and use TOU x KV-cache hysteresis for Think
        if tier == "fast":
            return "deepseek", "fast"
        if tier == "pro":
            return "dashscope", "pro"

        rates = provider_think_rates or {}
        ds_rates = rates.get("deepseek", (0.04, 2.0, 8.0))
        qw_rates = rates.get("dashscope", (0.16, 0.8, 4.8))
        surplus = compute_cross_provider_think_surplus_cny(
            ledger.last_prompt_tokens,
            ds_rates,
            qw_rates,
            self.params,
        )
        if surplus > self.params.hyst_margin_cny:
            return "dashscope", "think"
        return "deepseek", "think"
