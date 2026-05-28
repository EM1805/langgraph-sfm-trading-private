from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Set

from amantia.contracts import ActionPackage, DecisionPackage, normalize_action_package

from .requirements import (
    CODE_OR_OPS_ACTIONS,
    DESTRUCTIVE_ACTIONS,
    EXTERNAL_COMM_ACTIONS,
    FINANCE_ACTIONS,
    TRADING_ACTIONS,
    VALUE_TRANSFER_ACTIONS,
)
from .risk_levels import CRITICAL, HIGH, LOW, MEDIUM, UNKNOWN, max_risk, normalize_risk_level

_IDENTIFIED_TIERS = {"identified", "identified_graphical", "identified_recursive", "backdoor", "frontdoor"}
_IRREVERSIBLE = {"irreversible", "not_reversible", "none"}
_APPROVED_STATES = {"approved", "present", "granted", "true", "yes", "confirmed"}


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray, Mapping)):
        return list(value)
    return [value]


def _clean_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _clean_lower(value: Any, default: str = "unknown") -> str:
    return _clean_str(value, default).lower() or default


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on", "approved", "present", "granted"}:
        return True
    if text in {"0", "false", "no", "n", "off", "missing", "denied", "none"}:
        return False
    return default


def _dedupe(items: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for item in items:
        text = _clean_str(item)
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


@dataclass
class PolicyEvaluation:
    """Result of the causal-evidence-by-risk policy.

    The policy is intentionally conservative: it may strengthen a decision
    from allow -> warn/abstain/veto, but it never softens a veto.
    """

    policy_name: str = "causal_evidence_by_risk"
    policy_version: str = "step4b.v1"
    risk_level: str = UNKNOWN
    inferred_domain: str = "general"
    policy_decision: str = "abstain"
    original_decision: str = "abstain"
    evidence_required: List[str] = field(default_factory=list)
    evidence_present: List[str] = field(default_factory=list)
    evidence_missing: List[str] = field(default_factory=list)
    reason: str = ""
    reason_codes: List[str] = field(default_factory=list)
    hard_block: bool = False
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CausalEvidenceByRiskPolicy:
    """Apply escalating evidence requirements according to action risk.

    Core rule for agentic use:
    - low risk: runtime gate is enough.
    - medium risk: missing causal/approval evidence turns allow into warn.
    - high risk: trusted approval or causal support is required; otherwise abstain.
    - critical risk: missing hard runtime evidence such as trusted approval,
      trading risk limits, known notional, or rollback turns execution into veto.
    """

    def evaluate(self, action_payload: Any, decision_package: Any) -> PolicyEvaluation:
        action = normalize_action_package(action_payload)
        decision = self._decision_to_dict(decision_package)
        original_decision = _clean_lower(decision.get("decision"), "abstain")
        name = _clean_lower(decision.get("selected_action") or action.action_name or action.candidate_action, "")
        ctx = self._trusted_context(action)
        params = _as_dict(action.params)
        facts = self._facts(action, decision, name, ctx, params)
        risk = self._infer_risk(action, decision, name, facts)
        domain = self._infer_domain(name, facts)

        required = self._requirements_for(risk, name, facts)
        present = self._present_evidence(facts)
        missing = self._missing_evidence(required, present, facts)
        policy_decision, hard_block, codes, reason = self._policy_decision(
            original_decision=original_decision,
            risk=risk,
            name=name,
            facts=facts,
            missing=missing,
        )

        return PolicyEvaluation(
            risk_level=risk,
            inferred_domain=domain,
            policy_decision=policy_decision,
            original_decision=original_decision,
            evidence_required=required,
            evidence_present=present,
            evidence_missing=missing,
            reason=reason,
            reason_codes=codes,
            hard_block=hard_block,
            details={
                "action_name": name,
                "context_trust_mode": getattr(action, "context_trust_mode", "legacy_context"),
                "finance_or_trading": facts["finance_or_trading"],
                "live_or_real_money": facts["live_or_real_money"],
                "margin_or_leverage": facts["margin_or_leverage"],
                "production_environment": facts["production_environment"],
                "destructive_action": facts["destructive_action"],
                "code_or_ops_action": facts["code_or_ops_action"],
                "value_transfer_action": facts["value_transfer_action"],
                "causal_query_supplied": facts["causal_query_supplied"],
            },
        )

    def _decision_to_dict(self, decision: Any) -> Dict[str, Any]:
        if isinstance(decision, DecisionPackage):
            return decision.to_dict()
        if isinstance(decision, Mapping):
            return dict(decision)
        return {}

    def _trusted_context(self, action: ActionPackage) -> Dict[str, Any]:
        # Split-context mode is strict: only trusted_runtime_context can affect
        # execution evidence. Legacy context remains supported for old callers.
        if getattr(action, "context_trust_mode", "legacy_context") == "split_trusted_runtime":
            return _as_dict(action.trusted_runtime_context)
        return _as_dict(action.trusted_runtime_context or action.context)

    def _facts(
        self,
        action: ActionPackage,
        decision: Mapping[str, Any],
        name: str,
        ctx: Mapping[str, Any],
        params: Mapping[str, Any],
    ) -> Dict[str, Any]:
        def get(key: str, default: Any = None) -> Any:
            if key in ctx:
                return ctx[key]
            if key in params:
                return params[key]
            return default

        approval_state = _clean_lower(get("approval_state"), "")
        approval_present = _as_bool(get("approval_present"), False) or approval_state in _APPROVED_STATES
        risk_limits_present = _as_bool(get("risk_limits_present"), False)
        notional_known = _as_bool(get("notional_amount_known"), False) or get("notional_amount") not in (None, "")
        rollback_available = _as_bool(get("rollback_available"), False) or bool(_clean_str(get("rollback_plan")))

        identification = _as_dict(decision.get("causal_identification"))
        estimation = _as_dict(decision.get("causal_estimation"))
        counterfactual = _as_dict(decision.get("causal_counterfactual"))
        reason_codes = {str(c) for c in _as_list(decision.get("reason_codes"))}
        id_tier = _clean_lower(identification.get("identification_tier") or decision.get("identification_tier"), "unknown")
        scm_identified = bool(identification.get("identified")) or id_tier in _IDENTIFIED_TIERS or "SCM_ID_IDENTIFIED" in reason_codes
        scm_unidentified = bool(identification) and not scm_identified or "SCM_ID_UNIDENTIFIED" in reason_codes
        estimation_available = bool(estimation.get("estimated")) or "ESTIMATION_AVAILABLE" in reason_codes
        counterfactual_available = bool(counterfactual.get("compared")) or "COUNTERFACTUAL_COMPARED" in reason_codes
        safer_alternative = bool(counterfactual.get("recommended_action")) or "COUNTERFACTUAL_ALTERNATIVE_RECOMMENDED" in reason_codes
        recommendation_available = bool(decision.get("recommended_actions")) or "ACTION_RECOMMENDATION_AVAILABLE" in reason_codes
        causal_query_supplied = bool(action.causal_query or action.scm_graph or action.estimation_query or action.counterfactual_query)

        is_trading = name in TRADING_ACTIONS or _as_bool(get("trading_action"), False)
        is_finance = name in FINANCE_ACTIONS or is_trading or _as_bool(get("financial_action"), False)
        is_value_transfer = name in VALUE_TRANSFER_ACTIONS
        is_destructive = name in DESTRUCTIVE_ACTIONS or _clean_lower(action.reversibility) in _IRREVERSIBLE
        is_ops = name in CODE_OR_OPS_ACTIONS
        is_external = name in EXTERNAL_COMM_ACTIONS or _as_bool(get("external_counterparty"), False)
        live_or_real_money = _as_bool(get("live_trading"), False) or _as_bool(get("real_money"), False)
        margin_or_leverage = _as_bool(get("margin_used"), False) or _as_bool(get("leverage_used"), False) or name == "open_margin_position"
        high_notional = _as_bool(get("high_notional_amount"), False)
        production = _clean_lower(action.environment or get("environment"), "unknown") in {"prod", "production", "live"}
        policy_bypass = _as_bool(get("policy_bypass"), False)

        return {
            "approval_present": approval_present,
            "risk_limits_present": risk_limits_present,
            "notional_amount_known": notional_known,
            "rollback_available": rollback_available,
            "scm_identified": scm_identified,
            "scm_unidentified": scm_unidentified,
            "estimation_available": estimation_available,
            "counterfactual_available": counterfactual_available,
            "safer_alternative": safer_alternative,
            "recommendation_available": recommendation_available,
            "causal_query_supplied": causal_query_supplied,
            "trading_action": is_trading,
            "financial_action": is_finance,
            "finance_or_trading": is_finance or is_trading,
            "value_transfer_action": is_value_transfer,
            "destructive_action": is_destructive,
            "code_or_ops_action": is_ops,
            "external_action": is_external,
            "live_or_real_money": live_or_real_money,
            "margin_or_leverage": margin_or_leverage,
            "high_notional_amount": high_notional,
            "production_environment": production,
            "policy_bypass": policy_bypass,
            "causal_support_present": scm_identified or estimation_available or counterfactual_available,
            "mitigation_present": rollback_available or safer_alternative or recommendation_available,
        }

    def _infer_risk(self, action: ActionPackage, decision: Mapping[str, Any], name: str, facts: Mapping[str, Any]) -> str:
        explicit = normalize_risk_level(decision.get("risk_level") or action.risk_level)
        inferred = LOW
        if facts["external_action"] or name in {"write_file", "call_api", "write_memory", "create_calendar_event"}:
            inferred = max_risk(inferred, MEDIUM)
        if facts["destructive_action"] or facts["code_or_ops_action"] or facts["finance_or_trading"]:
            inferred = max_risk(inferred, HIGH)
        if (
            facts["policy_bypass"]
            or facts["value_transfer_action"]
            or (facts["finance_or_trading"] and facts["live_or_real_money"])
            or (facts["trading_action"] and facts["margin_or_leverage"])
            or facts["high_notional_amount"]
            or (facts["destructive_action"] and facts["production_environment"])
            or (facts["code_or_ops_action"] and facts["production_environment"])
        ):
            inferred = max_risk(inferred, CRITICAL)
        return max_risk(explicit, inferred)

    def _infer_domain(self, name: str, facts: Mapping[str, Any]) -> str:
        if facts["trading_action"]:
            return "trading"
        if facts["financial_action"]:
            return "finance"
        if facts["destructive_action"]:
            return "destructive_data"
        if facts["code_or_ops_action"]:
            return "code_or_ops"
        if facts["external_action"]:
            return "external_communication"
        return "general"

    def _requirements_for(self, risk: str, name: str, facts: Mapping[str, Any]) -> List[str]:
        required: List[str] = ["runtime_gate"]
        if risk == LOW:
            return required
        if risk == MEDIUM:
            required.append("causal_support_or_trusted_approval_or_safe_mitigation")
            return required

        required.append("causal_support_or_trusted_approval")
        if facts["destructive_action"] or facts["code_or_ops_action"]:
            required.append("rollback_or_sandbox_mitigation")
        if facts["external_action"]:
            required.append("recipient_or_counterparty_verified")
        if risk == CRITICAL:
            required.append("trusted_approval")
            if facts["finance_or_trading"]:
                required.append("notional_amount_known")
            if facts["trading_action"] or facts["live_or_real_money"] or facts["margin_or_leverage"]:
                required.append("risk_limits_present")
            if facts["destructive_action"] or facts["code_or_ops_action"]:
                required.append("rollback_available")
        return _dedupe(required)

    def _present_evidence(self, facts: Mapping[str, Any]) -> List[str]:
        present = ["runtime_gate"]
        if facts["approval_present"]:
            present.append("trusted_approval")
        if facts["scm_identified"] or facts["estimation_available"] or facts["counterfactual_available"]:
            present.append("causal_support")
        if facts["scm_identified"]:
            present.append("scm_identified")
        if facts["estimation_available"]:
            present.append("estimation_available")
        if facts["counterfactual_available"]:
            present.append("counterfactual_available")
        if facts["risk_limits_present"]:
            present.append("risk_limits_present")
        if facts["notional_amount_known"]:
            present.append("notional_amount_known")
        if facts["rollback_available"]:
            present.append("rollback_available")
        if facts["mitigation_present"]:
            present.append("safe_mitigation")
        if facts["safer_alternative"]:
            present.append("counterfactual_safer_alternative")
        if facts["recommendation_available"]:
            present.append("action_recommendation_available")
        return _dedupe(present)

    def _missing_evidence(self, required: List[str], present: List[str], facts: Mapping[str, Any]) -> List[str]:
        p = set(present)
        missing: List[str] = []
        for item in required:
            if item == "runtime_gate":
                continue
            if item == "trusted_approval" and "trusted_approval" not in p:
                missing.append(item)
            elif item == "risk_limits_present" and "risk_limits_present" not in p:
                missing.append(item)
            elif item == "notional_amount_known" and "notional_amount_known" not in p:
                missing.append(item)
            elif item == "rollback_available" and "rollback_available" not in p:
                missing.append(item)
            elif item == "rollback_or_sandbox_mitigation" and not (facts["rollback_available"] or facts["mitigation_present"]):
                missing.append(item)
            elif item == "recipient_or_counterparty_verified":
                # No hard field yet; approval or mitigation is enough for Step 4B.
                if not (facts["approval_present"] or facts["mitigation_present"]):
                    missing.append(item)
            elif item == "causal_support_or_trusted_approval":
                if not (facts["causal_support_present"] or facts["approval_present"]):
                    missing.append(item)
            elif item == "causal_support_or_trusted_approval_or_safe_mitigation":
                if not (facts["causal_support_present"] or facts["approval_present"] or facts["mitigation_present"]):
                    missing.append(item)
        return _dedupe(missing)

    def _policy_decision(
        self,
        *,
        original_decision: str,
        risk: str,
        name: str,
        facts: Mapping[str, Any],
        missing: List[str],
    ) -> tuple[str, bool, List[str], str]:
        codes: List[str] = ["RISK_POLICY_EVALUATED"]
        if not missing:
            codes.append("RISK_POLICY_REQUIREMENTS_SATISFIED")
            return original_decision, original_decision == "veto", codes, "Risk policy requirements are satisfied for this action."

        codes.append("RISK_POLICY_EVIDENCE_MISSING")
        codes.extend(f"MISSING_{m.upper()}" for m in missing)

        # Existing veto remains veto. Never soften a hard block.
        if original_decision == "veto":
            return "veto", True, codes, "Risk policy confirms the existing veto because required evidence is missing."

        hard_missing = {"trusted_approval", "risk_limits_present", "notional_amount_known", "rollback_available"}
        critical_hard_failure = risk == CRITICAL and bool(set(missing) & hard_missing)
        if critical_hard_failure:
            codes.append("RISK_POLICY_CRITICAL_HARD_BLOCK")
            return (
                "veto",
                True,
                codes,
                "Critical-risk action is blocked because trusted approval, limits, notional, or rollback evidence is missing.",
            )

        if risk == CRITICAL and facts["causal_query_supplied"] and facts["scm_unidentified"]:
            codes.append("RISK_POLICY_CRITICAL_UNIDENTIFIED")
            return "abstain", False, codes, "Critical-risk causal query was supplied but SCM-ID did not identify the effect."

        if risk == HIGH:
            codes.append("RISK_POLICY_HIGH_RISK_ABSTAIN")
            return "abstain", False, codes, "High-risk action lacks required causal support, trusted approval, or mitigation evidence."

        if risk == MEDIUM and original_decision == "allow":
            codes.append("RISK_POLICY_MEDIUM_RISK_WARN")
            return "warn", False, codes, "Medium-risk action is allowed only with warning because supporting evidence is incomplete."

        # Low risk, existing warning, or ask_clarification: do not over-block.
        return original_decision, False, codes, "Risk policy records missing evidence but does not strengthen this decision further."


def evaluate_evidence_by_risk(action_payload: Any, decision_package: Any) -> Dict[str, Any]:
    return CausalEvidenceByRiskPolicy().evaluate(action_payload, decision_package).to_dict()
