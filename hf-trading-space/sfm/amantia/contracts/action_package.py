from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set


_TRUE = {"1", "true", "yes", "y", "on"}
_FALSE = {"0", "false", "no", "n", "off"}

# These fields must be injected by the trusted runtime/tool wrapper, not by
# untrusted LLM text.  They influence whether a tool call is allowed to execute.
SENSITIVE_RUNTIME_KEYS: Set[str] = {
    "environment",
    "actor",
    "actor_permissions",
    "approval_present",
    "approval_state",
    "rollback_available",
    "rollback_plan",
    "recipient_external",
    "share_scope_external",
    "sensitive_resource",
    "attachment_present",
    "untrusted_recipient",
    "policy_bypass",
    "blast_radius",
    "resource_sensitivity",
    "risk_level",
    "ambiguity",
    "reversibility",
    "requires_tool",
    "requires_user_confirmation",
    "evidence_available",
    "data_classification",
    "target_resource",
    # Finance/trading runtime facts must be trusted, not LLM-asserted.
    "financial_action",
    "trading_action",
    "real_money",
    "live_trading",
    "leverage_used",
    "margin_used",
    "risk_limits_present",
    "notional_amount_known",
    "notional_amount",
    "max_notional_without_review",
    "high_notional_amount",
    "external_counterparty",
    "account_scope",
    "instrument_symbol",
    "order_type",
    "trade_side",
    "quantity",
    "currency",
}

RUNTIME_PARAM_KEYS: Set[str] = {
    "approval_present",
    "approval_state",
    "rollback_available",
    "rollback_plan",
    "recipient_external",
    "share_scope_external",
    "sensitive_resource",
    "attachment_present",
    "untrusted_recipient",
    "policy_bypass",
    "blast_radius",
    "resource_sensitivity",
    "actor_permissions",
    "data_classification",
    "financial_action",
    "trading_action",
    "real_money",
    "live_trading",
    "leverage_used",
    "margin_used",
    "risk_limits_present",
    "notional_amount_known",
    "notional_amount",
    "max_notional_without_review",
    "high_notional_amount",
    "external_counterparty",
    "account_scope",
    "instrument_symbol",
    "order_type",
    "trade_side",
    "quantity",
    "currency",
}


def _clean_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value).strip()


def _clean_lower(value: Any, default: str = "unknown") -> str:
    text = _clean_str(value, default=default).lower()
    return text or default


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    return default


def _as_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _as_str_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray, Mapping)):
        return [str(v).strip() for v in value if str(v).strip()]
    return []


def _pick_context(payload: Mapping[str, Any], legacy_context: Mapping[str, Any], trusted_context: Mapping[str, Any], key: str, default: Any = None) -> Any:
    """Resolve a runtime-sensitive value.

    In split-context mode, sensitive values are read only from
    trusted_runtime_context.  Top-level fields and untrusted_llm_context are
    ignored for these values.  Without explicit split context, the old behavior
    is preserved for backward compatibility.
    """

    if trusted_context or _as_dict(payload.get("untrusted_llm_context") or payload.get("llm_context")):
        return trusted_context.get(key, default)
    return payload.get(key, legacy_context.get(key, default))


def _ignored_untrusted_runtime_fields(untrusted_context: Mapping[str, Any]) -> List[str]:
    return sorted(str(k) for k in untrusted_context.keys() if str(k) in SENSITIVE_RUNTIME_KEYS)


@dataclass
class ActionPackage:
    """Canonical online action contract for Amantia.

    This object is the boundary between an LLM/agent and the causal safety
    runtime.  It is richer than the legacy runtime ActionIntent, but can still
    be converted back into that legacy payload through ``to_runtime_payload``.

    Step 2 security rule:
    - ``untrusted_llm_context`` may contain the LLM's explanation, requested
      action, or assumptions, but must not authorize execution.
    - ``trusted_runtime_context`` is the only new-schema source for execution
      facts such as environment, approval, rollback, blast radius, and resource
      sensitivity.
    - The legacy ``context`` field remains supported for old tests and existing
      callers, but new agent integrations should use the split fields.
    """

    user_message: str = ""
    candidate_action: str = ""
    candidate_actions: List[str] = field(default_factory=list)

    action_name: str = ""
    action_type: str = "unknown"
    target_resource: str = ""

    intended_outcome: str = "task_success"
    protected_outcome: str = "user_or_system_harm"

    # Optional causal query fields. These let the online DecisionGate call the
    # SCM-ID adapter when a caller supplies a graph, without making ID mandatory
    # for every chat turn.
    treatment: str = ""
    outcome: str = ""
    adjustment_set: List[str] = field(default_factory=list)
    scm_graph: Dict[str, Any] = field(default_factory=dict)
    causal_query: Dict[str, Any] = field(default_factory=dict)
    estimation_query: Dict[str, Any] = field(default_factory=dict)
    counterfactual_query: Dict[str, Any] = field(default_factory=dict)

    # ``context`` is legacy compatibility. New agent integrations should supply
    # trusted_runtime_context + untrusted_llm_context.
    context: Dict[str, Any] = field(default_factory=dict)
    trusted_runtime_context: Dict[str, Any] = field(default_factory=dict)
    untrusted_llm_context: Dict[str, Any] = field(default_factory=dict)
    ignored_untrusted_runtime_fields: List[str] = field(default_factory=list)
    context_trust_mode: str = "legacy_context"
    params: Dict[str, Any] = field(default_factory=dict)

    environment: str = "unknown"
    actor: str = "agent"

    risk_level: str = "unknown"
    ambiguity: str = "unknown"
    reversibility: str = "unknown"
    requires_tool: bool = False
    requires_user_confirmation: bool = False
    evidence_available: str = "unknown"

    request_id: str = ""
    source: str = "llm_or_agent"

    @classmethod
    def from_dict(cls, payload: Optional[Mapping[str, Any]]) -> "ActionPackage":
        payload = dict(payload or {})
        context = _as_dict(payload.get("context"))
        params = _as_dict(payload.get("params"))
        trusted_context = _as_dict(payload.get("trusted_runtime_context") or payload.get("trusted_context"))
        untrusted_context = _as_dict(payload.get("untrusted_llm_context") or payload.get("llm_context"))
        split_context_mode = bool(trusted_context or untrusted_context)
        context_trust_mode = "split_trusted_runtime" if split_context_mode else "legacy_context"

        action_name = _clean_str(
            payload.get("action_name")
            or payload.get("candidate_action")
            or payload.get("selected_action")
            or ""
        )
        candidate_action = _clean_str(payload.get("candidate_action") or action_name)

        # Allow legacy runtime JSON to be used directly as an ActionPackage.
        runtime_context = trusted_context if split_context_mode else context
        environment = _clean_lower(_pick_context(payload, context, trusted_context, "environment"), "unknown")
        actor = _clean_str(_pick_context(payload, context, trusted_context, "actor", "agent"), "agent")

        risk_level = _clean_lower(_pick_context(payload, context, trusted_context, "risk_level"), "unknown")
        ambiguity = _clean_lower(_pick_context(payload, context, trusted_context, "ambiguity"), "unknown")
        reversibility = _clean_lower(_pick_context(payload, context, trusted_context, "reversibility"), "unknown")

        requires_tool = _as_bool(_pick_context(payload, context, trusted_context, "requires_tool", False))
        requires_confirmation = _as_bool(
            _pick_context(payload, context, trusted_context, "requires_user_confirmation", False)
        )

        # Causal/estimation/counterfactual query payloads may be supplied
        # explicitly at top-level by the SDK/runtime.  The split context only
        # protects execution-authorizing fields.
        causal_query = _as_dict(payload.get("causal_query") or runtime_context.get("causal_query"))
        estimation_query = _as_dict(payload.get("estimation_query") or runtime_context.get("estimation_query"))
        counterfactual_query = _as_dict(payload.get("counterfactual_query") or runtime_context.get("counterfactual_query"))
        scm_graph = _as_dict(
            payload.get("scm_graph")
            or payload.get("graph")
            or runtime_context.get("scm_graph")
            or runtime_context.get("graph")
            or causal_query.get("scm_graph")
            or causal_query.get("graph")
        )
        treatment = _clean_str(payload.get("treatment") or runtime_context.get("treatment") or causal_query.get("treatment"))
        outcome = _clean_str(payload.get("outcome") or runtime_context.get("outcome") or causal_query.get("outcome"))
        adjustment_set = _as_str_list(
            payload.get("adjustment_set")
            or runtime_context.get("adjustment_set")
            or causal_query.get("adjustment_set")
        )

        target_resource = _clean_str(
            trusted_context.get("target_resource") if split_context_mode else payload.get("target_resource")
        )
        if not target_resource and not split_context_mode:
            target_resource = _clean_str(context.get("target_resource"))
        if not target_resource:
            target_resource = _clean_str(payload.get("target_resource"))

        return cls(
            user_message=_clean_str(payload.get("user_message")),
            candidate_action=candidate_action,
            candidate_actions=_as_str_list(payload.get("candidate_actions")),
            action_name=action_name,
            action_type=_clean_lower(payload.get("action_type"), "unknown"),
            target_resource=target_resource,
            intended_outcome=_clean_str(payload.get("intended_outcome") or "task_success"),
            protected_outcome=_clean_str(payload.get("protected_outcome") or "user_or_system_harm"),
            treatment=treatment,
            outcome=outcome,
            adjustment_set=adjustment_set,
            scm_graph=scm_graph,
            causal_query=causal_query,
            estimation_query=estimation_query,
            counterfactual_query=counterfactual_query,
            context=context,
            trusted_runtime_context=trusted_context,
            untrusted_llm_context=untrusted_context,
            ignored_untrusted_runtime_fields=_ignored_untrusted_runtime_fields(untrusted_context),
            context_trust_mode=context_trust_mode,
            params=params,
            environment=environment,
            actor=actor,
            risk_level=risk_level,
            ambiguity=ambiguity,
            reversibility=reversibility,
            requires_tool=requires_tool,
            requires_user_confirmation=requires_confirmation,
            evidence_available=_clean_lower(
                _pick_context(payload, context, trusted_context, "evidence_available"),
                "unknown",
            ),
            request_id=_clean_str(payload.get("request_id")),
            source=_clean_str(payload.get("source") or "llm_or_agent"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def has_split_context(self) -> bool:
        return self.context_trust_mode == "split_trusted_runtime"

    def to_runtime_payload(self) -> Dict[str, Any]:
        """Convert the rich agentic package to the legacy runtime veto payload."""
        params = dict(self.params or {})
        context = dict(self.trusted_runtime_context or self.context or {})

        # Preserve explicit runtime params, while filling common agentic fields.
        if self.risk_level and self.risk_level != "unknown":
            params.setdefault("risk_level", self.risk_level)
            if self.risk_level in {"high", "critical"}:
                params.setdefault("resource_sensitivity", "high")

        if self.ambiguity and self.ambiguity != "unknown":
            params.setdefault("ambiguity", self.ambiguity)

        if self.requires_user_confirmation:
            params.setdefault("requires_user_confirmation", True)
            params.setdefault("approval_present", _as_bool(context.get("approval_present"), False))

        if self.reversibility in {"irreversible", "not_reversible", "none"}:
            params.setdefault("rollback_available", False)
        elif self.reversibility in {"reversible", "rollback", "rollback_available"}:
            params.setdefault("rollback_available", True)

        # Pass through frequent context flags expected by runtime_context.py.
        # In split-context mode this reads only trusted_runtime_context.
        for key in RUNTIME_PARAM_KEYS:
            if key in context and key not in params:
                params[key] = context[key]

        if self.ignored_untrusted_runtime_fields:
            params.setdefault("_ignored_untrusted_runtime_fields", list(self.ignored_untrusted_runtime_fields))

        return {
            "action_name": self.action_name or self.candidate_action,
            "action_type": self.action_type,
            "target_resource": self.target_resource,
            "params": params,
            "environment": self.environment,
            "actor": self.actor,
            "context_trust_mode": self.context_trust_mode,
        }


def normalize_action_package(payload: Any) -> ActionPackage:
    if isinstance(payload, ActionPackage):
        return payload
    if isinstance(payload, Mapping):
        return ActionPackage.from_dict(payload)
    return ActionPackage.from_dict({})


__all__ = [
    "ActionPackage",
    "RUNTIME_PARAM_KEYS",
    "SENSITIVE_RUNTIME_KEYS",
    "normalize_action_package",
]
