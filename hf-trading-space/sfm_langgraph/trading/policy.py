from __future__ import annotations

"""Deterministic risk policy for trading-agent proposals.

This module deliberately does not try to predict markets.  Its job is to make
unsafe proposals impossible to execute accidentally.
"""

from dataclasses import replace
import os
from typing import Any, Dict, Iterable, List, Mapping

from .types import GateDecision, TradeProposal, TradingGateResult, TradingRiskPolicy

PROTECTED_SIDE_EFFECTS = [
    "capital_loss",
    "overtrading",
    "unauthorized_asset",
    "oversized_trade",
    "missing_stop_loss",
    "live_trading_without_ack",
    "margin_or_futures_exposure",
    "llm_news_high_risk",
    "llm_recommends_no_trade",
    "conflicting_market_signal",
]


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if out == out else default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def policy_from_mapping(data: Mapping[str, Any] | None) -> TradingRiskPolicy:
    """Create a policy from a mapping, ignoring unknown keys."""
    if not data:
        return TradingRiskPolicy()
    base = TradingRiskPolicy()
    allowed = data.get("allowed_symbols", base.allowed_symbols)
    if isinstance(allowed, str):
        allowed = [allowed]
    return replace(
        base,
        mode=str(data.get("mode", base.mode)),  # type: ignore[arg-type]
        allowed_symbols=list(allowed),
        max_notional_quote=_as_float(data.get("max_notional_quote"), base.max_notional_quote),
        max_daily_notional_quote=_as_float(data.get("max_daily_notional_quote"), base.max_daily_notional_quote),
        max_open_position_quote=_as_float(data.get("max_open_position_quote"), base.max_open_position_quote),
        max_trades_per_day=_as_int(data.get("max_trades_per_day"), base.max_trades_per_day),
        min_confidence=_as_float(data.get("min_confidence"), base.min_confidence),
        require_stop_loss=bool(data.get("require_stop_loss", base.require_stop_loss)),
        min_stop_loss_pct=_as_float(data.get("min_stop_loss_pct"), base.min_stop_loss_pct),
        max_stop_loss_pct=_as_float(data.get("max_stop_loss_pct"), base.max_stop_loss_pct),
        max_take_profit_pct=_as_float(data.get("max_take_profit_pct"), base.max_take_profit_pct),
        allow_margin=bool(data.get("allow_margin", base.allow_margin)),
        allow_futures=bool(data.get("allow_futures", base.allow_futures)),
        allow_live_trading=bool(data.get("allow_live_trading", base.allow_live_trading)),
        require_live_ack=bool(data.get("require_live_ack", base.require_live_ack)),
        live_ack_value=str(data.get("live_ack_value", base.live_ack_value)),
    )


def proposal_from_mapping(data: Mapping[str, Any] | None, *, default_symbol: str = "BTC/USDT") -> TradeProposal:
    """Create a proposal from a mapping, with safe hold defaults."""
    if not data:
        return TradeProposal(symbol=default_symbol, side="hold", rationale="no proposal supplied")
    side = str(data.get("side", "hold")).lower()
    if side not in {"buy", "sell", "hold"}:
        side = "hold"
    qty = data.get("quantity_base")
    return TradeProposal(
        symbol=str(data.get("symbol", default_symbol)),
        side=side,  # type: ignore[arg-type]
        notional_quote=_as_float(data.get("notional_quote"), 0.0),
        quantity_base=None if qty in {None, ""} else _as_float(qty),
        order_type=str(data.get("order_type", "market")),
        stop_loss_pct=None if data.get("stop_loss_pct") in {None, ""} else _as_float(data.get("stop_loss_pct")),
        take_profit_pct=None if data.get("take_profit_pct") in {None, ""} else _as_float(data.get("take_profit_pct")),
        confidence=_as_float(data.get("confidence"), 0.0),
        rationale=str(data.get("rationale", "")),
        source=str(data.get("source", "sfm_langgraph.trading")),
    )


def _daily_notional(prior_trades_today: Iterable[Mapping[str, Any]]) -> float:
    total = 0.0
    for item in prior_trades_today:
        if bool(item.get("executed", True)):
            total += abs(_as_float(item.get("notional_quote"), 0.0))
    return total



def _market_view_recommendation(violations: List[str], warnings: List[str]) -> Dict[str, Any]:
    """Return safe next-step recommendation for block/review outcomes."""
    all_flags = set(violations + warnings)
    required_changes: List[str] = []
    if "oversized_trade" in all_flags:
        required_changes.append("reduce_notional")
    if "missing_stop_loss" in all_flags or "stop_loss_outside_policy" in all_flags:
        required_changes.append("add_policy_compliant_stop_loss")
    if "overtrading" in all_flags or "daily_notional_limit_exceeded" in all_flags:
        required_changes.append("wait_cooldown_or_stop_for_today")
    if "unauthorized_asset" in all_flags:
        required_changes.append("use_allowed_symbol_only")
    if "llm_news_high_risk" in all_flags or "llm_recommends_no_trade" in all_flags:
        required_changes.append("wait_for_news_risk_to_clear_or_request_human_review")
    if "live_trading_without_ack" in all_flags or "live_trading_disabled_by_policy" in all_flags:
        required_changes.append("switch_to_paper_or_testnet")
    if not required_changes:
        required_changes.append("keep_order_in_review_until_risk_context_is_clear")
    action = "no_trade" if violations else "human_review"
    return {
        "recommended_action": action,
        "required_changes": sorted(set(required_changes)),
        "fallback_action": "no_trade",
        "message": "SFM/risk gate recommends the safest next step, not a profitable trade prediction.",
    }

def evaluate_trade_policy(
    proposal: TradeProposal,
    policy: TradingRiskPolicy,
    *,
    prior_trades_today: Iterable[Mapping[str, Any]] | None = None,
    open_position_quote: float = 0.0,
    sfm_analysis: Mapping[str, Any] | None = None,
    live_ack: str | None = None,
    market_view: Mapping[str, Any] | None = None,
) -> TradingGateResult:
    """Return allow/review/block for a proposed trade.

    Blocking checks are intentionally simple and conservative.  A downstream
    broker should only receive orders when this function returns ``allow``.
    """
    prior = list(prior_trades_today or [])
    violations: List[str] = []
    warnings: List[str] = []

    if proposal.side == "hold":
        return TradingGateResult(
            decision="block",
            reason="No trade: strategy produced hold/no-action.",
            violations=["no_trade_signal"],
            risk_score=0.0,
            protected_side_effects=[],
            safety_recommendation={
                "recommended_action": "no_trade",
                "required_changes": ["wait_for_actionable_signal"],
                "fallback_action": "no_trade",
                "message": "No order is submitted when the strategy emits hold/no-action.",
            },
            sfm_claim_level=str((sfm_analysis or {}).get("final_cause_claim_level", "diagnostic_only")),
            sfm_primary_intent=str((sfm_analysis or {}).get("primary_intent", "")),
        )

    if proposal.symbol not in policy.allowed_symbols:
        violations.append("unauthorized_asset")
    if proposal.notional_quote <= 0:
        violations.append("non_positive_notional")
    if proposal.notional_quote > policy.max_notional_quote:
        violations.append("oversized_trade")
    if len(prior) >= policy.max_trades_per_day:
        violations.append("overtrading")
    if _daily_notional(prior) + proposal.notional_quote > policy.max_daily_notional_quote:
        violations.append("daily_notional_limit_exceeded")
    if open_position_quote + proposal.notional_quote > policy.max_open_position_quote and proposal.side == "buy":
        violations.append("open_position_limit_exceeded")
    if proposal.confidence < policy.min_confidence:
        warnings.append("low_strategy_confidence")

    view = market_view or {}
    llm_risk_level = str(view.get("risk_level", "unknown")).lower()
    llm_trade_bias = str(view.get("trade_bias", "hold_bias")).lower()
    llm_confidence = _as_float(view.get("confidence"), 0.0)
    if llm_risk_level == "high":
        violations.append("llm_news_high_risk")
    elif llm_risk_level == "medium" and llm_confidence >= 0.55:
        warnings.append("llm_news_medium_risk")
    if llm_trade_bias == "no_trade" and llm_confidence >= 0.5:
        violations.append("llm_recommends_no_trade")
    if proposal.side == "buy" and llm_trade_bias == "sell_bias" and llm_confidence >= 0.6:
        warnings.append("conflicting_market_signal")
    if proposal.side == "sell" and llm_trade_bias == "buy_bias" and llm_confidence >= 0.6:
        warnings.append("conflicting_market_signal")

    if policy.require_stop_loss and proposal.side == "buy":
        if proposal.stop_loss_pct is None:
            violations.append("missing_stop_loss")
        elif proposal.stop_loss_pct < policy.min_stop_loss_pct or proposal.stop_loss_pct > policy.max_stop_loss_pct:
            violations.append("stop_loss_outside_policy")

    if proposal.take_profit_pct is not None and proposal.take_profit_pct > policy.max_take_profit_pct:
        warnings.append("take_profit_outside_policy")

    if policy.allow_margin or policy.allow_futures:
        violations.append("margin_or_futures_exposure")

    if policy.mode == "live":
        ack = live_ack or os.getenv("SFM_TRADING_LIVE_ACK", "")
        if not policy.allow_live_trading:
            violations.append("live_trading_disabled_by_policy")
        if policy.require_live_ack and ack != policy.live_ack_value:
            violations.append("live_trading_without_ack")

    analysis = sfm_analysis or {}
    claim_level = str(analysis.get("final_cause_claim_level", "diagnostic_only"))
    primary_intent = str(analysis.get("primary_intent", ""))
    sfm_gate = str(analysis.get("gate_status", "review")).lower()
    side_effect_risk = str(analysis.get("side_effect_risk", "unknown")).lower()
    deception_risk = str(analysis.get("deception_risk", "unknown")).lower()

    if side_effect_risk == "high":
        violations.append("sfm_high_side_effect_risk")
    if deception_risk == "high":
        violations.append("sfm_high_deception_risk")
    if sfm_gate == "block":
        violations.append("sfm_gate_block")
    elif sfm_gate == "review":
        warnings.append("sfm_gate_review")

    protected = [item for item in PROTECTED_SIDE_EFFECTS if item in set(violations + warnings)]
    risk_score = min(1.0, 0.18 * len(violations) + 0.08 * len(warnings))

    unique_warnings = sorted(set(warnings))
    non_blocking_sfm_review = unique_warnings == ["sfm_gate_review"] and policy.mode in {"paper", "testnet"}

    if violations:
        decision: GateDecision = "block"
        reason = "Blocked by hard trading risk policy: " + ", ".join(sorted(set(violations)))
    elif warnings and not non_blocking_sfm_review:
        decision = "review"
        reason = "Human review recommended before trade: " + ", ".join(unique_warnings)
    elif non_blocking_sfm_review:
        decision = "allow"
        reason = "Allowed for paper/testnet: hard risk policy passed; SFM claim remains diagnostic/review-only."
    else:
        decision = "allow"
        reason = "Allowed by hard trading risk policy and SFM gate."

    return TradingGateResult(
        decision=decision,
        reason=reason,
        violations=sorted(set(violations + warnings)),
        risk_score=round(risk_score, 6),
        protected_side_effects=protected,
        safety_recommendation=_market_view_recommendation(violations, warnings) if decision in {"block", "review"} else {
            "recommended_action": "execute_with_limits",
            "required_changes": [],
            "fallback_action": "no_trade",
            "message": "Order may proceed only through the configured paper/testnet/live execution layer.",
        },
        sfm_claim_level=claim_level,
        sfm_primary_intent=primary_intent,
    )
