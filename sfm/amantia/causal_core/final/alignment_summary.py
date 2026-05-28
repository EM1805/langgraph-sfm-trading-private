from __future__ import annotations

"""Governance-facing summary for Structural Final Model diagnostics.

The SFM stack intentionally keeps every diagnostic layer separate.  That is good
for auditability, but external gates need a compact contract.  This module folds
final-cause evidence, falsification, constraints, normative status,
identifiability, and recommendation support into one conservative verdict.

It does not erase the underlying evidence.  In particular, a goal can be
teleologically plausible and still prohibited, constraint-blocked, or only
weakly identifiable.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .schema import FinalCauseQuery, GoalSpec


@dataclass
class SFMAlignmentSummary:
    """Single governance verdict over the layered SFM diagnostics."""

    assessed: bool = False
    verdict: str = "unassessed"
    gate_status: str = "review"
    confidence_level: str = "none"
    authority_status: str = "diagnostic_only"
    goal_variable: str = ""
    observed_action: str = ""
    intent_supported: bool = False
    intent_score: float = 0.0
    support_level: str = "none"
    falsification_passed: bool = True
    constraints_satisfied: bool = True
    normatively_aligned: bool = False
    prohibited: bool = False
    requires_escalation: bool = False
    side_effects_excluded: bool = False
    recommendation_status: str = "unassessed"
    recommended_action: str = ""
    recommendation_matches_observed: bool = False
    recommended_action_allowed: bool = False
    robustness_status: str = "unassessed"
    robust_to_uncertainty: bool = False
    uncertainty_review_required: bool = False
    pessimistic_intent_score: float = 0.0
    allow_execution: bool = False
    reason_codes: List[str] = field(default_factory=list)
    blocking_reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _clean_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    return [value]


def _unique(items: Iterable[Any]) -> List[str]:
    out: List[str] = []
    for item in items:
        text = _clean_str(item)
        if text and text not in out:
            out.append(text)
    return out


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        low = value.strip().lower()
        if low in {"true", "1", "yes", "y", "pass", "passed"}:
            return True
        if low in {"false", "0", "no", "n", "fail", "failed"}:
            return False
    return bool(value)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _confidence_from(score: float, authority_status: str, *, intent_supported: bool) -> str:
    authority = _clean_str(authority_status, "diagnostic_only")
    strong_authority = authority in {"partial_sfm_identification", "strong_diagnostic_sfm_support"}
    falsifiable_authority = authority in {"falsifiable_diagnostic_only", "partial_sfm_identification", "strong_diagnostic_sfm_support"}
    if not intent_supported:
        if score >= 0.55 and falsifiable_authority:
            return "low"
        return "none"
    if score >= 0.85 and strong_authority:
        return "high"
    if score >= 0.65 and falsifiable_authority:
        return "moderate"
    return "low"


class SFMAlignmentSummarizer:
    """Build a compact gate/governance summary from SFM layer outputs."""

    def summarize(
        self,
        query: FinalCauseQuery,
        goal: GoalSpec,
        *,
        intent_supported: bool,
        intent_score: float,
        support_level: str,
        authority_status: str,
        falsification_passed: bool,
        side_effects_excluded: bool,
        constraint_support: Optional[Mapping[str, Any]] = None,
        normative_support: Optional[Mapping[str, Any]] = None,
        sfm_identifiability_support: Optional[Mapping[str, Any]] = None,
        action_recommendation_support: Optional[Mapping[str, Any]] = None,
        robustness_support: Optional[Mapping[str, Any]] = None,
        reason_codes: Optional[Iterable[Any]] = None,
        limits: Optional[Iterable[Any]] = None,
    ) -> SFMAlignmentSummary:
        constraint = _as_dict(constraint_support)
        normative = _as_dict(normative_support)
        ident = _as_dict(sfm_identifiability_support)
        recommendation = _as_dict(action_recommendation_support)
        robustness = _as_dict(robustness_support)

        constraint_assessed = _safe_bool(constraint.get("assessed"), False)
        constraints_satisfied = _safe_bool(constraint.get("observed_feasible"), True) if constraint_assessed else True
        normative_assessed = _safe_bool(normative.get("assessed"), False)
        normatively_aligned = _safe_bool(normative.get("normatively_aligned"), False) if normative_assessed else False
        prohibited = _safe_bool(normative.get("prohibited"), False)
        requires_escalation = _safe_bool(normative.get("requires_escalation"), False)

        rec_assessed = _safe_bool(recommendation.get("assessed"), False)
        recommendation_status = _clean_str(recommendation.get("recommendation_status"), "unassessed") if rec_assessed else "unassessed"
        recommended_action = _clean_str(recommendation.get("recommended_action") or recommendation.get("selected_action"))
        recommendation_matches_observed = _safe_bool(recommendation.get("recommendation_matches_observed"), False)
        recommended_action_allowed = rec_assessed and recommendation_status in {"recommended", "observed_matches_recommendation"}
        robustness_assessed = _safe_bool(robustness.get("assessed"), False)
        robustness_status = _clean_str(robustness.get("robustness_status"), "unassessed") if robustness_assessed else "unassessed"
        robust_to_uncertainty = _safe_bool(robustness.get("robust_to_uncertainty"), False) if robustness_assessed else False
        uncertainty_review_required = _safe_bool(robustness.get("uncertainty_review_required"), False) if robustness_assessed else False
        pessimistic_intent_score = _safe_float(robustness.get("pessimistic_intent_score"), 0.0) if robustness_assessed else 0.0
        robustness_reason_codes = set(_as_list(robustness.get("reason_codes")))
        high_uncertainty_review = "SFM_ROBUSTNESS_HIGH_UNCERTAINTY_REVIEW_REQUIRED" in robustness_reason_codes
        fragile_uncertainty = robustness_status in {"fragile_support", "not_robust", "not_robust_due_to_hard_blocks"}

        codes = _unique(reason_codes or [])
        blocks: List[str] = []
        warnings: List[str] = []
        if not intent_supported:
            warnings.append("intent_not_claimable_under_current_sfm_evidence")
        if not falsification_passed:
            blocks.append("falsification_failed")
        if not side_effects_excluded:
            blocks.append("candidate_goal_overlaps_protected_or_side_effect")
        if not constraints_satisfied:
            blocks.append("observed_action_violates_constraints")
        if prohibited:
            blocks.append("goal_or_action_prohibited_by_normative_policy")
        if requires_escalation or recommendation_status == "requires_escalation":
            blocks.append("escalation_required")
        if rec_assessed and not recommendation_matches_observed and recommended_action:
            warnings.append("sfm_recommendation_differs_from_observed_action")
        if ident and not _safe_bool(ident.get("can_claim_intent"), False):
            warnings.append("sfm_identifiability_does_not_authorize_intent_claim")
        if robustness_assessed and uncertainty_review_required:
            warnings.append("sfm_robustness_audit_requires_uncertainty_review")
        if robustness_assessed and fragile_uncertainty:
            warnings.append("sfm_claim_fragile_under_pessimistic_uncertainty")
        for item in limits or []:
            text = _clean_str(item)
            if text and text not in warnings:
                warnings.append(text)

        confidence = _confidence_from(intent_score, authority_status, intent_supported=intent_supported)

        if not query.observed_action:
            verdict = "insufficient_evidence"
            gate_status = "review"
        elif not falsification_passed:
            verdict = "falsification_failed"
            gate_status = "block"
        elif prohibited:
            # Governance block regardless of whether the telic claim is fully
            # identifiable.  The summary preserves intent_supported separately.
            verdict = "supported_but_prohibited"
            gate_status = "block"
        elif not constraints_satisfied:
            # Constraint violations are operational blockers even when the
            # final-cause evidence remains only diagnostic.
            verdict = "supported_but_constraint_blocked"
            gate_status = "block"
        elif not intent_supported and intent_score <= 0.0:
            verdict = "unsupported"
            gate_status = "observe"
        elif not intent_supported:
            verdict = "plausible_but_unidentified" if intent_score >= query.min_intent_score else "insufficient_evidence"
            gate_status = "review"
        elif requires_escalation or recommendation_status == "requires_escalation":
            verdict = "requires_escalation"
            gate_status = "escalate"
        elif robustness_assessed and (fragile_uncertainty or high_uncertainty_review):
            verdict = "supported_but_uncertain"
            gate_status = "review"
        elif authority_status in {"diagnostic_only", "falsifiable_diagnostic_only"}:
            verdict = "diagnostic_only" if intent_supported else "plausible_but_unidentified"
            gate_status = "review"
        elif normative_assessed and not normatively_aligned:
            verdict = "supported_but_normatively_unspecified"
            gate_status = "review"
        elif rec_assessed and recommended_action and not recommendation_matches_observed:
            verdict = "supported_but_recommendation_differs"
            gate_status = "review"
        else:
            verdict = "aligned_supported"
            gate_status = "allow"

        allow_execution = (
            gate_status == "allow"
            and intent_supported
            and falsification_passed
            and constraints_satisfied
            and not prohibited
            and not requires_escalation
            and (not robustness_assessed or robust_to_uncertainty or robustness_status == "supported_but_uncertain")
        )

        summary_codes = list(codes)
        summary_codes.append(f"SFM_ALIGNMENT_VERDICT_{verdict.upper()}")
        summary_codes.append(f"SFM_GATE_STATUS_{gate_status.upper()}")
        if blocks:
            summary_codes.append("SFM_ALIGNMENT_HAS_BLOCKING_REASONS")
        if warnings:
            summary_codes.append("SFM_ALIGNMENT_HAS_WARNINGS")
        if robustness_assessed:
            summary_codes.append(f"SFM_ALIGNMENT_ROBUSTNESS_{robustness_status.upper()}")

        return SFMAlignmentSummary(
            assessed=True,
            verdict=verdict,
            gate_status=gate_status,
            confidence_level=confidence,
            authority_status=_clean_str(authority_status, "diagnostic_only"),
            goal_variable=goal.goal_variable,
            observed_action=query.observed_action,
            intent_supported=bool(intent_supported),
            intent_score=round(_safe_float(intent_score), 6),
            support_level=_clean_str(support_level, "none"),
            falsification_passed=bool(falsification_passed),
            constraints_satisfied=bool(constraints_satisfied),
            normatively_aligned=bool(normatively_aligned),
            prohibited=bool(prohibited),
            requires_escalation=bool(requires_escalation or recommendation_status == "requires_escalation"),
            side_effects_excluded=bool(side_effects_excluded),
            recommendation_status=recommendation_status,
            recommended_action=recommended_action,
            recommendation_matches_observed=bool(recommendation_matches_observed),
            recommended_action_allowed=bool(recommended_action_allowed),
            robustness_status=robustness_status,
            robust_to_uncertainty=bool(robust_to_uncertainty),
            uncertainty_review_required=bool(uncertainty_review_required),
            pessimistic_intent_score=round(_safe_float(pessimistic_intent_score), 6),
            allow_execution=bool(allow_execution),
            reason_codes=_unique(summary_codes),
            blocking_reasons=_unique(blocks),
            warnings=_unique(warnings),
            raw={
                "normative_support": normative,
                "constraint_support": constraint,
                "sfm_identifiability_support": ident,
                "action_recommendation_support": recommendation,
                "robustness_support": robustness,
            },
        )


def summarize_sfm_alignment(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Convenience helper for callers that already have layer outputs."""

    query = FinalCauseQuery.from_payload(payload.get("query") or payload)
    goal = GoalSpec.from_payload(payload.get("goal") or payload.get("candidate_goal") or payload.get("most_likely_goal") or "")
    return SFMAlignmentSummarizer().summarize(
        query,
        goal,
        intent_supported=_safe_bool(payload.get("intent_supported") or payload.get("inferred"), False),
        intent_score=_safe_float(payload.get("intent_score"), 0.0),
        support_level=_clean_str(payload.get("support_level"), "none"),
        authority_status=_clean_str(payload.get("authority_status"), "diagnostic_only"),
        falsification_passed=_safe_bool(payload.get("falsification_passed"), True),
        side_effects_excluded=_safe_bool(payload.get("side_effects_excluded"), False),
        constraint_support=_as_dict(payload.get("constraint_support")),
        normative_support=_as_dict(payload.get("normative_support")),
        sfm_identifiability_support=_as_dict(payload.get("sfm_identifiability_support")),
        action_recommendation_support=_as_dict(payload.get("action_recommendation_support")),
        robustness_support=_as_dict(payload.get("robustness_support") or payload.get("uncertainty_support")),
        reason_codes=_as_list(payload.get("reason_codes")),
        limits=_as_list(payload.get("limits")),
    ).to_dict()
