from __future__ import annotations
from typing import Any, Dict


def _truthy(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        out = float(value)
        if out != out or out in (float("inf"), float("-inf")):
            return default
        return out
    except (TypeError, ValueError):
        return default


def _param(params: Dict[str, Any], defaults: Dict[str, Any], key: str, default: Any = None) -> Any:
    return params.get(key, defaults.get(key, default))


def extract_runtime_context(intent: Dict[str, Any], action_spec: Dict[str, Any] | None = None) -> Dict[str, Any]:
    action_spec = action_spec or {}
    defaults = dict(action_spec.get("default_params", {}) or {})
    params = dict(intent.get("params", {}) or {})
    env = str(intent.get("environment", "unknown") or "unknown").lower()
    action_name = str(intent.get("action_name", "") or "").strip().lower()
    action_type = str(intent.get("action_type", action_spec.get("action_type", "unknown")) or "unknown").strip().lower()
    domain = str(action_spec.get("domain", "") or "").strip().lower()

    recipient_scope = str(_param(params, defaults, "recipient_scope", "") or "").lower()
    share_scope = str(_param(params, defaults, "share_scope", "") or "").lower()
    sensitivity = str(_param(params, defaults, "resource_sensitivity", "unknown") or "unknown").lower()
    trust = str(_param(params, defaults, "recipient_trust_level", "unknown") or "unknown").lower()
    rollback = _truthy(_param(params, defaults, "rollback_available", False))
    approval = _truthy(_param(params, defaults, "approval_present", False))
    novelty = _truthy(_param(params, defaults, "novel_action", False))
    blast = str(_param(params, defaults, "blast_radius", "low") or "low").lower()
    attachment_present = _truthy(_param(params, defaults, "attachment_present", False))

    # Finance/trading flags. Defaults from action_registry.yaml are treated as
    # action semantics; caller-provided runtime facts must still come from the
    # trusted_runtime_context path in ActionPackage.
    financial_action = _truthy(_param(params, defaults, "financial_action", False)) or domain in {"finance", "trading"} or action_type.startswith("financial")
    trading_action = _truthy(_param(params, defaults, "trading_action", False)) or domain == "trading" or "trade" in action_type or "order" in action_name
    real_money = _truthy(_param(params, defaults, "real_money", False))
    live_trading = _truthy(_param(params, defaults, "live_trading", False)) or (trading_action and real_money)
    leverage_used = _truthy(_param(params, defaults, "leverage_used", False))
    margin_used = _truthy(_param(params, defaults, "margin_used", False))
    risk_limits_present = _truthy(_param(params, defaults, "risk_limits_present", False))
    notional_known = _truthy(_param(params, defaults, "notional_amount_known", False))
    notional_amount = _num(_param(params, defaults, "notional_amount", 0.0), 0.0)
    max_notional = _num(_param(params, defaults, "max_notional_without_review", 0.0), 0.0)
    high_notional = _truthy(_param(params, defaults, "high_notional_amount", False)) or (
        notional_amount > 0 and max_notional > 0 and notional_amount > max_notional
    )
    external_counterparty = _truthy(_param(params, defaults, "external_counterparty", False)) or recipient_scope == "external"

    return {
        "production_environment": env in {"prod", "production"},
        "recipient_external": recipient_scope == "external",
        "share_scope_external": share_scope == "external",
        "sensitive_resource": sensitivity in {"high", "critical", "sensitive", "financial", "secret"},
        "untrusted_recipient": trust in {"unknown", "low", "untrusted"},
        "rollback_unavailable": not rollback,
        "approval_present": approval,
        "approval_missing": not approval,
        "novel_action": novelty,
        "high_blast_radius": blast in {"high", "very_high", "critical"},
        "attachment_present": attachment_present,
        "financial_action": financial_action,
        "trading_action": trading_action,
        "real_money": real_money,
        "live_trading": live_trading,
        "leverage_used": leverage_used,
        "margin_used": margin_used,
        "risk_limits_present": risk_limits_present,
        "risk_limits_missing": not risk_limits_present,
        "notional_amount_known": notional_known,
        "notional_amount_missing": not notional_known,
        "high_notional_amount": high_notional,
        "external_counterparty": external_counterparty,
    }
