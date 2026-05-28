from __future__ import annotations

from typing import Any, Dict, List


def _s(value: Any) -> str:
    return str(value or "").strip().lower()


def _params(intent: Dict[str, Any]) -> Dict[str, Any]:
    return dict((intent or {}).get("params", {}) or {})


def _hard_result(reason_code: str, note: str, *, stage: str = "pre_action") -> Dict[str, Any]:
    return {
        "ok": False,
        "stage": stage,
        "decision": "HARD_BLOCK",
        "reason_codes": [reason_code],
        "notes": [note],
    }


def pass_result(*, stage: str = "pre_action") -> Dict[str, Any]:
    return {"ok": True, "stage": stage, "reason_codes": [], "notes": []}


def _is_finance_or_trading(action_name: str, action_type: str, context_flags: Dict[str, Any]) -> bool:
    finance_names = {
        "view_account_balance",
        "access_financial_data",
        "initiate_bank_transfer",
        "withdraw_funds",
        "approve_invoice_payment",
        "charge_customer",
        "refund_payment",
        "issue_payout",
        "connect_brokerage_api",
    }
    trading_names = {
        "place_market_order",
        "place_limit_order",
        "cancel_order",
        "modify_order",
        "close_position",
        "open_margin_position",
        "set_stop_loss",
        "change_trading_risk_limits",
        "rebalance_portfolio",
    }
    return (
        action_name in finance_names
        or action_name in trading_names
        or action_type in {"financial_transfer", "financial_read", "payment_mutation", "trade_execution", "leveraged_trade_execution", "portfolio_mutation", "risk_policy_change"}
        or bool(context_flags.get("financial_action"))
        or bool(context_flags.get("trading_action"))
    )


def check_pre_action_invariants(intent: Dict[str, Any], context_flags: Dict[str, Any]) -> Dict[str, Any]:
    """Hard runtime safety gates before policy scoring.

    Invariants are intentionally narrow: they block non-negotiable safety
    violations. The policy engine remains responsible for graded decisions
    such as PASS, REVIEW, or PASS_WITH_WARNING.
    """
    intent = intent or {}
    context_flags = context_flags or {}
    params = _params(intent)
    action_name = _s(intent.get("action_name"))
    action_type = _s(intent.get("action_type"))

    if not action_name:
        return _hard_result(
            "SAFETY_INVARIANT_MISSING_ACTION_NAME",
            "Runtime action request is missing action_name; causal safety cannot evaluate an unnamed action.",
        )

    bypass = action_name in {"disable_policy", "bypass_policy", "suppress_guardrail", "disable_guardrail"} or bool(params.get("policy_bypass", False))
    if bypass and context_flags.get("production_environment") and not context_flags.get("approval_present"):
        return _hard_result(
            "SAFETY_INVARIANT_POLICY_BYPASS_IN_PRODUCTION",
            "Policy or guardrail bypass in production requires explicit approval before the policy engine may score it.",
        )

    destructive = action_type in {"delete", "destructive", "mutation"} or action_name in {"delete_records", "drop_table", "purge_data"}
    if destructive and context_flags.get("production_environment") and context_flags.get("rollback_unavailable") and not context_flags.get("approval_present"):
        return _hard_result(
            "SAFETY_INVARIANT_IRREVERSIBLE_DESTRUCTIVE_PROD",
            "Irreversible destructive production action without rollback and approval is non-negotiably blocked.",
        )

    external_sensitive = (
        (context_flags.get("recipient_external") or context_flags.get("share_scope_external"))
        and context_flags.get("sensitive_resource")
        and (context_flags.get("attachment_present") or context_flags.get("untrusted_recipient"))
        and not context_flags.get("approval_present")
    )
    if external_sensitive:
        return _hard_result(
            "SAFETY_INVARIANT_SENSITIVE_EXTERNAL_TRANSFER",
            "Sensitive external transfer with attachment or untrusted recipient requires approval before causal policy scoring.",
        )

    # Step 3: finance/trading hard gates. These are not investment advice;
    # they are execution-safety invariants for AI agents controlling tools.
    finance_or_trading = _is_finance_or_trading(action_name, action_type, context_flags)
    if finance_or_trading:
        approval = bool(context_flags.get("approval_present"))
        real_money = bool(context_flags.get("real_money"))
        live_trading = bool(context_flags.get("live_trading"))
        money_movement = action_type in {"financial_transfer", "payment_mutation"} or action_name in {
            "initiate_bank_transfer",
            "withdraw_funds",
            "approve_invoice_payment",
            "charge_customer",
            "issue_payout",
        }
        trade_execution = bool(context_flags.get("trading_action")) and (
            live_trading
            or action_type in {"trade_execution", "leveraged_trade_execution", "portfolio_mutation"}
            or action_name in {"place_market_order", "place_limit_order", "modify_order", "close_position", "open_margin_position", "rebalance_portfolio"}
        )

        if (real_money or money_movement) and money_movement and not approval:
            return _hard_result(
                "SAFETY_INVARIANT_FINANCIAL_TRANSFER_REQUIRES_APPROVAL",
                "Live money movement or financial transfer requires trusted approval before execution.",
            )

        if trade_execution and not approval:
            return _hard_result(
                "SAFETY_INVARIANT_LIVE_TRADE_REQUIRES_APPROVAL",
                "Live trading execution requires trusted approval before an agent may place or modify an order.",
            )

        if trade_execution and bool(context_flags.get("risk_limits_missing")):
            return _hard_result(
                "SAFETY_INVARIANT_LIVE_TRADE_REQUIRES_RISK_LIMITS",
                "Live trading execution requires trusted risk limits such as max notional, quantity, account scope, and order constraints.",
            )

        if (bool(context_flags.get("leverage_used")) or bool(context_flags.get("margin_used"))) and not approval:
            return _hard_result(
                "SAFETY_INVARIANT_LEVERAGED_TRADE_REQUIRES_APPROVAL",
                "Leveraged or margin trading requires trusted approval and risk controls before execution.",
            )

        if bool(context_flags.get("high_notional_amount")) and not approval:
            return _hard_result(
                "SAFETY_INVARIANT_HIGH_NOTIONAL_REQUIRES_APPROVAL",
                "High-notional financial or trading action requires trusted approval before execution.",
            )

    return pass_result(stage="pre_action")


def check_post_path_invariants(paths: List[Dict[str, Any]], context_flags: Dict[str, Any]) -> Dict[str, Any]:
    """Hard gates after path activation/evidence."""
    for path in list(paths or []):
        if path.get("hard_block_hits"):
            return _hard_result(
                "SAFETY_INVARIANT_PATH_HARD_BLOCK",
                f"Dangerous path {path.get('path_id', 'unknown')} reported hard_block_hits.",
                stage="post_path",
            )
    return pass_result(stage="post_path")


def invariant_decision(result: Dict[str, Any], *, activated_path_count: int = 0) -> Dict[str, Any]:
    """Convert a failed invariant result into the runtime decision schema."""
    return {
        "decision": "HARD_BLOCK",
        "reason_codes": list(result.get("reason_codes") or ["SAFETY_INVARIANT_BLOCK"]),
        "max_risk_score": 1.0,
        "max_weighted_risk_score": 1.0,
        "activated_path_count": int(activated_path_count or 0),
        "evidence_tier": "invariant",
        "identification_tier": "not_applicable",
        "structural_tier": "invariant",
        "design_strength": "not_applicable",
        "decision_basis": {
            "top_path_id": None,
            "invariant_stage": result.get("stage", "unknown"),
            "evidence_tier": "invariant",
            "identification_tier": "not_applicable",
            "structural_tier": "invariant",
            "design_strength": "not_applicable",
            "identification_score": 0.0,
        },
        "notes": list(result.get("notes") or []),
    }


__all__ = [
    "check_pre_action_invariants",
    "check_post_path_invariants",
    "invariant_decision",
    "pass_result",
]
