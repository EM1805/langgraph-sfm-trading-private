from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from .action_package import ActionPackage, normalize_action_package


RUNTIME_TO_AGENTIC = {
    "PASS": "allow",
    "PASS_WITH_WARNING": "warn",
    "REVIEW": "ask_clarification",
    "HARD_BLOCK": "veto",
}

VALID_AGENTIC_DECISIONS = {"allow", "warn", "ask_clarification", "abstain", "veto"}


def _reason_from_runtime(runtime_decision: Mapping[str, Any]) -> str:
    notes = [str(n).strip() for n in runtime_decision.get("notes", []) or [] if str(n).strip()]
    codes = [str(c).strip() for c in runtime_decision.get("reason_codes", []) or [] if str(c).strip()]
    if notes:
        return notes[0]
    if codes:
        return "; ".join(codes)
    return "No runtime reason supplied."


def _instruction_for(decision: str, action: ActionPackage, reason: str) -> str:
    name = action.action_name or action.candidate_action or "the candidate action"
    if decision == "allow":
        return f"Proceed with {name}; no dangerous causal path was activated."
    if decision == "warn":
        return f"Proceed carefully with {name}; include a short warning or mitigation because Amantia detected elevated risk."
    if decision == "ask_clarification":
        return f"Do not execute {name} yet; ask the user for confirmation or missing information first. Reason: {reason}"
    if decision == "veto":
        return f"Do not execute {name}; Amantia vetoed it. Explain the safety reason and offer a safer alternative."
    return f"Abstain from {name}; causal/safety support is insufficient. Ask for more evidence or route to review."


@dataclass
class DecisionPackage:
    """Canonical agent-facing output of Amantia's Decision Gate."""

    decision: str = "abstain"
    selected_action: str = ""
    candidate_action: str = ""
    reason: str = ""
    reason_codes: List[str] = field(default_factory=list)

    runtime_decision: str = "UNKNOWN"
    risk_level: str = "unknown"
    evidence_tier: str = "unknown"
    identification_tier: str = "unknown"
    structural_tier: str = "unknown"
    confidence: str = "unknown"
    causal_identification: Dict[str, Any] = field(default_factory=dict)
    causal_estimation: Dict[str, Any] = field(default_factory=dict)
    causal_counterfactual: Dict[str, Any] = field(default_factory=dict)

    # Step 4B: causal-evidence-by-risk policy. Higher-risk actions require
    # stronger evidence before an agent may execute tools.
    risk_policy: Dict[str, Any] = field(default_factory=dict)
    policy_decision: str = ""
    evidence_required: List[str] = field(default_factory=list)
    evidence_present: List[str] = field(default_factory=list)
    evidence_missing: List[str] = field(default_factory=list)

    # Step 4A: bounded action recommendations for AI agents. These are
    # proposal-only and must never bypass DecisionGate/ToolGuard.
    recommended_action: Dict[str, Any] = field(default_factory=dict)
    recommended_actions: List[Dict[str, Any]] = field(default_factory=list)
    recommendation_summary: str = ""
    recommendation_status: str = "no_recommendation"
    short_for_llm: Dict[str, Any] = field(default_factory=dict)

    llm_instruction: str = ""
    audit_payload: Dict[str, Any] = field(default_factory=dict)
    raw_runtime_result: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.short_for_llm:
            self.short_for_llm = build_short_for_llm(asdict(self))

    def to_dict(self) -> Dict[str, Any]:
        if not self.short_for_llm:
            self.short_for_llm = build_short_for_llm(asdict(self))
        return asdict(self)


def _confidence_from_runtime(runtime_decision: Mapping[str, Any], agentic_decision: str) -> str:
    evidence = str(runtime_decision.get("evidence_tier", "unknown"))
    ident = str(runtime_decision.get("identification_tier", "unknown"))
    if agentic_decision == "veto" and evidence in {"invariant", "high"}:
        return "high"
    if evidence == "high" and ident in {"medium", "high", "not_applicable"}:
        return "high"
    if evidence in {"medium", "invariant"} or ident in {"medium", "high"}:
        return "medium"
    if agentic_decision == "allow":
        return "medium"
    return "low"




def build_short_for_llm(decision: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the compact safe output that an LLM/agent planner should see."""

    d = dict(decision or {})
    return {
        "decision": d.get("decision", "abstain"),
        "selected_action": d.get("selected_action", ""),
        "reason": d.get("reason", ""),
        "instruction": d.get("llm_instruction", ""),
        "recommended_action": d.get("recommended_action", {}),
        "recommendation_summary": d.get("recommendation_summary", ""),
        "policy_decision": d.get("policy_decision", d.get("decision", "abstain")),
        "evidence_missing": list(d.get("evidence_missing", []) or []),
        "execution_rule": "Any recommended action is proposal-only and must pass ToolGuard before execution.",
    }

def decision_package_from_runtime(
    action_package: Any,
    runtime_result: Optional[Mapping[str, Any]],
) -> DecisionPackage:
    action = normalize_action_package(action_package)
    runtime_result = dict(runtime_result or {})
    runtime_decision = dict(runtime_result.get("decision", {}) or {})

    legacy_decision = str(runtime_decision.get("decision", "UNKNOWN") or "UNKNOWN").strip().upper()
    reason_codes = [str(c) for c in runtime_decision.get("reason_codes", []) or []]

    # Agentic correction: missing action name should ask for clarification, not become a final product veto.
    if "SAFETY_INVARIANT_MISSING_ACTION_NAME" in reason_codes:
        agentic_decision = "ask_clarification"
    else:
        agentic_decision = RUNTIME_TO_AGENTIC.get(legacy_decision, "abstain")

    reason = _reason_from_runtime(runtime_decision)
    selected = action.action_name or action.candidate_action
    confidence = _confidence_from_runtime(runtime_decision, agentic_decision)

    audit_payload = {
        "request_id": action.request_id,
        "source": action.source,
        "user_message": action.user_message,
        "candidate_actions": list(action.candidate_actions or []),
        "selected_action": selected,
        "gate_decision": agentic_decision,
        "legacy_runtime_decision": legacy_decision,
        "reason_codes": reason_codes,
        "risk_level": action.risk_level,
        "ambiguity": action.ambiguity,
        "context_trust_mode": getattr(action, "context_trust_mode", "legacy_context"),
        "ignored_untrusted_runtime_fields": list(getattr(action, "ignored_untrusted_runtime_fields", []) or []),
        "evidence_tier": runtime_decision.get("evidence_tier", "unknown"),
        "identification_tier": runtime_decision.get("identification_tier", "unknown"),
    }

    return DecisionPackage(
        decision=agentic_decision,
        selected_action=selected,
        candidate_action=action.candidate_action,
        reason=reason,
        reason_codes=reason_codes,
        runtime_decision=legacy_decision,
        risk_level=action.risk_level,
        evidence_tier=str(runtime_decision.get("evidence_tier", "unknown")),
        identification_tier=str(runtime_decision.get("identification_tier", "unknown")),
        structural_tier=str(runtime_decision.get("structural_tier", "unknown")),
        confidence=confidence,
        llm_instruction=_instruction_for(agentic_decision, action, reason),
        audit_payload=audit_payload,
        raw_runtime_result=runtime_result,
    )


__all__ = [
    "DecisionPackage",
    "RUNTIME_TO_AGENTIC",
    "VALID_AGENTIC_DECISIONS",
    "build_short_for_llm",
    "decision_package_from_runtime",
]
