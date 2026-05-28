from __future__ import annotations

"""Uncertainty-aware robustness audit for SFM diagnostics.

This module is deliberately conservative.  It does not introduce a new causal
estimator.  It stress-tests an already-computed final-cause hypothesis by
combining common uncertainty signals:

- candidate-action uncertainty / CI width / standard errors;
- evidence-quality labels;
- weak recommendation top margins;
- weak SFM identifiability authority;
- known hard blocks from falsification, constraints, and normative policy.

The output is a diagnostic robustness verdict that a governance gate can use to
route fragile claims to review rather than treating a high point score as a
strong claim.
"""

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .schema import FinalCauseQuery, GoalSpec


@dataclass
class RobustnessScenario:
    """One score stress-test scenario for a candidate final-cause claim."""

    name: str = "baseline"
    intent_score: float = 0.0
    penalty: float = 0.0
    pessimistic_score: float = 0.0
    passes_threshold: bool = False
    reason_codes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RobustSFMAudit:
    """Conservative robustness assessment for an SFM inference result."""

    assessed: bool = False
    goal_variable: str = ""
    observed_action: str = ""
    robustness_status: str = "unassessed"
    robust_to_uncertainty: bool = False
    uncertainty_review_required: bool = False
    robustness_score: float = 0.0
    baseline_intent_score: float = 0.0
    pessimistic_intent_score: float = 0.0
    min_intent_score: float = 0.6
    total_uncertainty_penalty: float = 0.0
    observed_action_uncertainty: Optional[float] = None
    max_candidate_uncertainty: Optional[float] = None
    authority_penalty: float = 0.0
    margin_penalty: float = 0.0
    hard_block_penalty: float = 0.0
    top_margin: Optional[float] = None
    uncertainty_policy: Dict[str, Any] = field(default_factory=dict)
    scenarios: List[Dict[str, Any]] = field(default_factory=list)
    reason: str = ""
    reason_codes: List[str] = field(default_factory=list)
    limits: List[str] = field(default_factory=list)
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


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _clip01(value: Any, default: float = 0.0) -> float:
    numeric = _safe_float(value, default)
    if numeric is None:
        numeric = default
    return max(0.0, min(1.0, float(numeric)))


def _unique(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    for item in items:
        text = _clean_str(item)
        if text and text not in out:
            out.append(text)
    return out


def _action_name(option: Any) -> str:
    if isinstance(option, Mapping):
        return _clean_str(
            option.get("action")
            or option.get("action_name")
            or option.get("candidate_action")
            or option.get("selected_action")
            or option.get("name")
        )
    return _clean_str(option)


def _ci_half_width(value: Any) -> Optional[float]:
    if isinstance(value, Mapping):
        lo = _safe_float(value.get("lower") or value.get("lo") or value.get("low"))
        hi = _safe_float(value.get("upper") or value.get("hi") or value.get("high"))
    elif isinstance(value, (list, tuple)) and len(value) >= 2:
        lo = _safe_float(value[0])
        hi = _safe_float(value[1])
    else:
        return None
    if lo is None or hi is None:
        return None
    return abs(float(hi) - float(lo)) / 2.0


def _option_uncertainty(option: Mapping[str, Any]) -> Tuple[Optional[float], List[str]]:
    """Extract a bounded uncertainty penalty from common option fields."""

    values: List[float] = []
    codes: List[str] = []
    for key in [
        "uncertainty",
        "outcome_uncertainty",
        "causal_uncertainty",
        "model_uncertainty",
        "standard_error",
        "se",
        "ci_width",
    ]:
        value = _safe_float(option.get(key))
        if value is not None:
            values.append(abs(float(value)))

    for key in ["effect_ci", "confidence_interval", "ci", "outcome_ci"]:
        half_width = _ci_half_width(option.get(key))
        if half_width is not None:
            values.append(float(half_width))

    evidence_quality = _clean_str(option.get("evidence_quality") or option.get("causal_evidence_quality")).lower()
    quality_penalty = {
        "very_high": 0.00,
        "high": 0.03,
        "medium": 0.10,
        "moderate": 0.10,
        "low": 0.22,
        "very_low": 0.35,
        "weak": 0.30,
        "unknown": 0.18,
    }.get(evidence_quality)
    if quality_penalty is not None:
        values.append(quality_penalty)
        codes.append(f"SFM_ROBUSTNESS_EVIDENCE_QUALITY_{evidence_quality.upper()}")

    if not values:
        return None, ["SFM_ROBUSTNESS_OPTION_UNCERTAINTY_NOT_SUPPLIED"]
    penalty = min(0.45, max(0.0, max(values)))
    if penalty >= 0.20:
        codes.append("SFM_ROBUSTNESS_HIGH_OPTION_UNCERTAINTY")
    else:
        codes.append("SFM_ROBUSTNESS_OPTION_UNCERTAINTY_BOUNDED")
    return round(float(penalty), 6), codes


def _uncertainty_policy(query: FinalCauseQuery) -> Dict[str, Any]:
    raw = _as_dict(query.raw)
    policy = _as_dict(raw.get("uncertainty_policy") or raw.get("robustness_policy") or raw.get("robust_sfm_policy"))
    return {
        "default_unknown_uncertainty_penalty": _clip01(policy.get("default_unknown_uncertainty_penalty", 0.08)),
        "high_uncertainty_threshold": _clip01(policy.get("high_uncertainty_threshold", 0.20), 0.20),
        "min_margin": max(0.0, float(_safe_float(policy.get("min_margin"), 0.05) or 0.05)),
        "fragility_buffer": _clip01(policy.get("fragility_buffer", 0.05), 0.05),
        "pessimistic_multiplier": max(0.0, float(_safe_float(policy.get("pessimistic_multiplier"), 1.0) or 1.0)),
        "review_on_unknown_uncertainty": bool(policy.get("review_on_unknown_uncertainty", True)),
    }


def _authority_penalty(sfm_identifiability: Mapping[str, Any]) -> Tuple[float, List[str]]:
    status = _clean_str(sfm_identifiability.get("authority_status") or sfm_identifiability.get("identifiability_status")).lower()
    if not status:
        return 0.12, ["SFM_ROBUSTNESS_AUTHORITY_UNKNOWN"]
    if status in {"strong_diagnostic_sfm_support", "strong_sfm_support"}:
        return 0.00, ["SFM_ROBUSTNESS_AUTHORITY_STRONG"]
    if status in {"partial_sfm_identification", "partially_identified"}:
        return 0.04, ["SFM_ROBUSTNESS_AUTHORITY_PARTIAL"]
    if status in {"falsifiable_diagnostic_only"}:
        return 0.10, ["SFM_ROBUSTNESS_AUTHORITY_FALSIFIABLE_DIAGNOSTIC"]
    if status in {"diagnostic_only", "unidentified", "unsupported"}:
        return 0.18, ["SFM_ROBUSTNESS_AUTHORITY_DIAGNOSTIC_ONLY"]
    return 0.12, ["SFM_ROBUSTNESS_AUTHORITY_WEAK_OR_UNKNOWN"]


def _margin_penalty(action_recommendation: Mapping[str, Any], min_margin: float) -> Tuple[float, Optional[float], List[str]]:
    margin = _safe_float(action_recommendation.get("top_margin"))
    if margin is None:
        return 0.05, None, ["SFM_ROBUSTNESS_RECOMMENDATION_MARGIN_UNKNOWN"]
    if margin < min_margin:
        return round(min(0.16, (min_margin - margin) + 0.06), 6), round(float(margin), 6), ["SFM_ROBUSTNESS_RECOMMENDATION_MARGIN_FRAGILE"]
    return 0.0, round(float(margin), 6), ["SFM_ROBUSTNESS_RECOMMENDATION_MARGIN_SUFFICIENT"]


def _hard_block_penalty(
    *,
    falsification_passed: bool,
    constraint_support: Mapping[str, Any],
    normative_support: Mapping[str, Any],
    action_recommendation: Mapping[str, Any],
    observed_action: str,
) -> Tuple[float, List[str]]:
    penalty = 0.0
    codes: List[str] = []
    if not falsification_passed:
        penalty += 0.45
        codes.append("SFM_ROBUSTNESS_FALSIFICATION_FAILED")
    if bool(constraint_support.get("assessed")) and not bool(constraint_support.get("observed_feasible")):
        penalty += 0.35
        codes.append("SFM_ROBUSTNESS_OBSERVED_ACTION_CONSTRAINT_BLOCKED")
    if bool(normative_support.get("assessed")) and bool(normative_support.get("prohibited")):
        penalty += 0.35
        codes.append("SFM_ROBUSTNESS_NORMATIVE_PROHIBITION_PRESENT")
    if bool(action_recommendation.get("assessed")) and observed_action in set(action_recommendation.get("blocked_actions") or []):
        penalty += 0.35
        codes.append("SFM_ROBUSTNESS_RECOMMENDER_BLOCKS_OBSERVED_ACTION")
    if not codes:
        codes.append("SFM_ROBUSTNESS_NO_HARD_BLOCKS")
    return round(min(1.0, penalty), 6), codes


class RobustSFMEvaluator:
    """Stress-test a final-cause claim under uncertainty and weak evidence."""

    def evaluate(
        self,
        query: Any,
        goal: Any,
        *,
        intent_score: float = 0.0,
        intent_supported: bool = False,
        falsification_passed: bool = True,
        sfm_identifiability_support: Optional[Mapping[str, Any]] = None,
        constraint_support: Optional[Mapping[str, Any]] = None,
        normative_support: Optional[Mapping[str, Any]] = None,
        action_recommendation_support: Optional[Mapping[str, Any]] = None,
    ) -> RobustSFMAudit:
        query = FinalCauseQuery.from_payload(query)
        goal = GoalSpec.from_payload(goal)
        identifiability = _as_dict(sfm_identifiability_support)
        constraints = _as_dict(constraint_support)
        normative = _as_dict(normative_support)
        recommendation = _as_dict(action_recommendation_support)
        policy = _uncertainty_policy(query)

        if not goal.goal_variable:
            return RobustSFMAudit(
                assessed=False,
                observed_action=query.observed_action,
                robustness_status="no_goal",
                reason="SFM robustness audit requires a candidate goal.",
                reason_codes=["SFM_ROBUSTNESS_REQUIRES_GOAL"],
                limits=["candidate_goal_required"],
                raw={"query": query.to_dict()},
            )

        option_uncertainties: Dict[str, float] = {}
        unknown_uncertainty_actions: List[str] = []
        option_codes: List[str] = []
        for option in query.candidate_actions:
            option = _as_dict(option)
            action = _action_name(option)
            value, codes = _option_uncertainty(option)
            option_codes.extend(codes)
            if not action:
                continue
            if value is None:
                unknown_uncertainty_actions.append(action)
            else:
                option_uncertainties[action] = value

        default_unknown = float(policy["default_unknown_uncertainty_penalty"])
        observed_uncertainty = option_uncertainties.get(query.observed_action)
        if observed_uncertainty is None and query.observed_action:
            observed_uncertainty = default_unknown
        max_candidate_uncertainty: Optional[float]
        if option_uncertainties:
            max_candidate_uncertainty = max(option_uncertainties.values())
        elif query.candidate_actions:
            max_candidate_uncertainty = default_unknown
        else:
            max_candidate_uncertainty = None

        authority_penalty, authority_codes = _authority_penalty(identifiability)
        margin_penalty, top_margin, margin_codes = _margin_penalty(recommendation, float(policy["min_margin"]))
        hard_block_penalty, block_codes = _hard_block_penalty(
            falsification_passed=falsification_passed,
            constraint_support=constraints,
            normative_support=normative,
            action_recommendation=recommendation,
            observed_action=query.observed_action,
        )

        unknown_penalty = default_unknown if unknown_uncertainty_actions and policy["review_on_unknown_uncertainty"] else 0.0
        option_penalty = max(float(observed_uncertainty or 0.0), float(max_candidate_uncertainty or 0.0), unknown_penalty)
        total_penalty = (
            option_penalty
            + authority_penalty
            + margin_penalty
            + hard_block_penalty
        ) * float(policy["pessimistic_multiplier"])
        total_penalty = round(min(1.0, max(0.0, total_penalty)), 6)
        baseline = _clip01(intent_score)
        pessimistic = round(max(0.0, baseline - total_penalty), 6)
        min_score = float(query.min_intent_score or 0.6)
        buffer = float(policy["fragility_buffer"])
        passes = pessimistic >= min_score

        scenarios = [
            RobustnessScenario(
                name="baseline",
                intent_score=baseline,
                penalty=0.0,
                pessimistic_score=baseline,
                passes_threshold=baseline >= min_score,
                reason_codes=["SFM_ROBUSTNESS_BASELINE_SCORE"],
            ).to_dict(),
            RobustnessScenario(
                name="pessimistic_uncertainty",
                intent_score=baseline,
                penalty=total_penalty,
                pessimistic_score=pessimistic,
                passes_threshold=passes,
                reason_codes=["SFM_ROBUSTNESS_PESSIMISTIC_STRESS_TEST"],
            ).to_dict(),
        ]

        high_uncertainty = bool(
            (max_candidate_uncertainty is not None and max_candidate_uncertainty >= float(policy["high_uncertainty_threshold"]))
            or (observed_uncertainty is not None and observed_uncertainty >= float(policy["high_uncertainty_threshold"]))
        )
        near_threshold = bool(0 <= baseline - min_score < buffer)
        unknown_uncertainty = bool(unknown_uncertainty_actions)

        if not intent_supported and baseline < min_score:
            status = "not_applicable_intent_not_supported"
        elif hard_block_penalty >= 0.35:
            status = "not_robust_due_to_hard_blocks"
        elif passes and not high_uncertainty and not unknown_uncertainty:
            status = "robust_supported"
        elif passes:
            status = "supported_but_uncertain"
        elif baseline >= min_score:
            status = "fragile_support"
        else:
            status = "not_robust"

        review_required = status in {
            "supported_but_uncertain",
            "fragile_support",
            "not_robust",
            "not_robust_due_to_hard_blocks",
        } or high_uncertainty or near_threshold or (unknown_uncertainty and policy["review_on_unknown_uncertainty"])
        robust = status == "robust_supported"
        robustness_score = round(max(0.0, min(1.0, pessimistic / max(min_score, 1e-9))), 6)

        reason_codes: List[str] = ["SFM_ROBUSTNESS_ASSESSED"]
        reason_codes.extend(option_codes)
        reason_codes.extend(authority_codes)
        reason_codes.extend(margin_codes)
        reason_codes.extend(block_codes)
        reason_codes.append(f"SFM_ROBUSTNESS_STATUS_{status.upper()}")
        if robust:
            reason_codes.append("SFM_ROBUSTNESS_PASSES_PESSIMISTIC_STRESS_TEST")
        if high_uncertainty:
            reason_codes.append("SFM_ROBUSTNESS_HIGH_UNCERTAINTY_REVIEW_REQUIRED")
        if near_threshold:
            reason_codes.append("SFM_ROBUSTNESS_BASELINE_SCORE_NEAR_THRESHOLD")
        if unknown_uncertainty:
            reason_codes.append("SFM_ROBUSTNESS_UNKNOWN_ACTION_UNCERTAINTY")

        limits: List[str] = ["robustness_audit_is_diagnostic_not_formal_sensitivity_analysis"]
        if unknown_uncertainty:
            limits.append("some_candidate_actions_lack_uncertainty_metadata")
        if high_uncertainty:
            limits.append("candidate_action_uncertainty_is_high")
        if margin_penalty > 0:
            limits.append("recommendation_top_margin_is_weak_or_unknown")
        if authority_penalty >= 0.10:
            limits.append("sfm_identifiability_authority_is_weak")
        if hard_block_penalty > 0:
            limits.append("hard_blocks_reduce_robustness")

        return RobustSFMAudit(
            assessed=True,
            goal_variable=goal.goal_variable,
            observed_action=query.observed_action,
            robustness_status=status,
            robust_to_uncertainty=robust,
            uncertainty_review_required=bool(review_required),
            robustness_score=robustness_score,
            baseline_intent_score=baseline,
            pessimistic_intent_score=pessimistic,
            min_intent_score=min_score,
            total_uncertainty_penalty=total_penalty,
            observed_action_uncertainty=round(float(observed_uncertainty), 6) if observed_uncertainty is not None else None,
            max_candidate_uncertainty=round(float(max_candidate_uncertainty), 6) if max_candidate_uncertainty is not None else None,
            authority_penalty=round(float(authority_penalty), 6),
            margin_penalty=round(float(margin_penalty), 6),
            hard_block_penalty=round(float(hard_block_penalty), 6),
            top_margin=top_margin,
            uncertainty_policy=policy,
            scenarios=scenarios,
            reason=(
                "SFM robustness audit stress-tested the final-cause claim under candidate-action uncertainty, "
                "identifiability authority, recommendation margin, and hard-block penalties."
            ),
            reason_codes=_unique(reason_codes),
            limits=_unique(limits),
            raw={
                "query": query.to_dict(),
                "unknown_uncertainty_actions": unknown_uncertainty_actions,
                "option_uncertainties": option_uncertainties,
                "sfm_identifiability_support": identifiability,
                "action_recommendation_support": recommendation,
            },
        )


def evaluate_sfm_robustness(payload: Any) -> Dict[str, Any]:
    """Convenience helper for standalone robustness audits.

    The payload can either be a full inference-style query plus optional layer
    outputs, or a dict with explicit keys: ``query``, ``goal``, ``intent_score``
    and diagnostic supports.
    """

    data = _as_dict(payload)
    query = FinalCauseQuery.from_payload(data.get("query") or data)
    goal_payload = data.get("goal") or data.get("candidate_goal") or (query.candidate_goals[0] if query.candidate_goals else {})
    return RobustSFMEvaluator().evaluate(
        query,
        goal_payload,
        intent_score=float(data.get("intent_score", data.get("baseline_intent_score", 0.0)) or 0.0),
        intent_supported=bool(data.get("intent_supported", data.get("inferred", False))),
        falsification_passed=bool(data.get("falsification_passed", True)),
        sfm_identifiability_support=_as_dict(data.get("sfm_identifiability_support") or data.get("identifiability_support")),
        constraint_support=_as_dict(data.get("constraint_support")),
        normative_support=_as_dict(data.get("normative_support")),
        action_recommendation_support=_as_dict(data.get("action_recommendation_support") or data.get("recommendation_support")),
    ).to_dict()
