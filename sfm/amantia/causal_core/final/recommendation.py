from __future__ import annotations

"""SFM action recommendation under goals, constraints, norms, and uncertainty.

This module turns the diagnostic SFM layers into an operational recommendation
surface.  It does not claim that a recommended action is *morally* or
*causally* proven best.  It ranks candidate interventions under:

- a final-goal bundle G;
- hard/protected/soft constraints;
- normative/value policy rules;
- risk and causal/outcome uncertainty penalties;
- the existing action/outcome forecasts carried by candidate_actions.

The output is deliberately auditable: every score is decomposed and every block
or escalation is surfaced as a reason code.
"""

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .constraint import ConstraintAwareEvaluator
from .do_star import _format_do_star_expression
from .normative import NormalizedNormativePolicy, normalize_normative_policy
from .schema import FinalCauseQuery, GoalSpec
from .utility import _directional_utility, _outcome_value, _risk_penalty


@dataclass
class RecommendedInterventionAction:
    """Score decomposition for one candidate action recommendation."""

    action: str = ""
    rank: Optional[int] = None
    recommendation_score: float = 0.0
    goal_score: float = 0.0
    auxiliary_score: float = 0.0
    constraint_penalty: float = 0.0
    risk_penalty: float = 0.0
    uncertainty_penalty: float = 0.0
    normative_penalty: float = 0.0
    total_penalty: float = 0.0
    constraint_feasible: bool = True
    normatively_allowed: bool = True
    requires_escalation: bool = False
    hard_blocked: bool = False
    recommendable: bool = True
    goal_contributions: List[Dict[str, Any]] = field(default_factory=list)
    constraint_evaluations: List[Dict[str, Any]] = field(default_factory=list)
    normative_status: str = "unspecified"
    violated_constraints: List[str] = field(default_factory=list)
    reason_codes: List[str] = field(default_factory=list)
    raw_option: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SFMActionRecommendationAudit:
    """Aggregate SFM intervention recommendation."""

    assessed: bool = False
    action_variable: str = "agent_action"
    observed_action: str = ""
    recommended_action: str = ""
    selected_action: str = ""
    recommendation_matches_observed: bool = False
    recommendation_status: str = "unassessed"
    support_strength: float = 0.0
    top_margin: Optional[float] = None
    goal_bundle: List[str] = field(default_factory=list)
    feasible_actions: List[str] = field(default_factory=list)
    blocked_actions: List[str] = field(default_factory=list)
    escalation_actions: List[str] = field(default_factory=list)
    rankings: List[Dict[str, Any]] = field(default_factory=list)
    recommended_intervention: Dict[str, Any] = field(default_factory=dict)
    authority_status: str = "recommendation_diagnostic"
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


def _clean_lower(value: Any, default: str = "") -> str:
    return _clean_str(value, default).lower()


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


def _clip01(value: Optional[float], default: float = 0.0) -> float:
    if value is None:
        value = default
    return max(0.0, min(1.0, float(value)))


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


def _explicit_goal_bundle(query: FinalCauseQuery) -> List[GoalSpec]:
    raw = _as_dict(query.raw)
    explicit = raw.get("recommendation_goal_bundle") or raw.get("goal_bundle") or raw.get("multi_goal_bundle") or raw.get("multi_objective_goals")
    if explicit:
        if isinstance(explicit, list):
            return [GoalSpec.from_payload(item) for item in explicit]
        return [GoalSpec.from_payload(explicit)]
    return list(query.candidate_goals or [])


def _policy(query: FinalCauseQuery) -> Dict[str, Any]:
    raw = _as_dict(query.raw)
    return _as_dict(raw.get("normative_policy") or raw.get("value_policy") or raw.get("alignment_policy") or raw.get("policy"))


def _normalized_policy(query_or_policy: Any) -> NormalizedNormativePolicy:
    if isinstance(query_or_policy, FinalCauseQuery):
        return normalize_normative_policy(_policy(query_or_policy))
    return normalize_normative_policy(query_or_policy)


def _goal_policy_status(policy: Any, goals: Sequence[GoalSpec]) -> Tuple[bool, bool, List[str], List[str]]:
    names = [goal.goal_variable for goal in goals if goal.goal_variable]
    normalized = normalize_normative_policy(policy)
    prohibited_hits: List[str] = []
    escalation_hits: List[str] = []
    for name in names:
        status = normalized.status_for_target(name, "goal", allowlist_mode=normalized.strict_goal_allowlist)
        if status in {"prohibited", "not_on_allowlist"}:
            prohibited_hits.append(name)
        if status == "escalation_required":
            escalation_hits.append(name)
    goal_allowed = not prohibited_hits
    goal_requires_escalation = bool(escalation_hits)
    return goal_allowed, goal_requires_escalation, prohibited_hits, escalation_hits


def _action_policy_status(policy: Any, action: str) -> Tuple[str, bool, bool, bool]:
    normalized = normalize_normative_policy(policy)
    status = normalized.status_for_target(action, "action", allowlist_mode=normalized.strict_action_allowlist)
    if status == "prohibited":
        return "prohibited", False, False, True
    if status == "not_on_allowlist":
        return "not_on_allowlist", False, False, True
    if status == "escalation_required":
        return "escalation_required", True, True, False
    if status == "discouraged":
        return "discouraged", False, False, False
    if status == "allowed":
        return "allowed", True, False, False
    return "unspecified", True, False, False


def _option_uncertainty_penalty(option: Mapping[str, Any]) -> Tuple[float, List[str]]:
    """Conservative uncertainty penalty from common uncertainty/evidence fields."""

    codes: List[str] = []
    candidates: List[float] = []
    for key in ["uncertainty", "outcome_uncertainty", "causal_uncertainty", "standard_error", "se", "ci_width"]:
        value = _safe_float(option.get(key))
        if value is not None:
            candidates.append(abs(float(value)))

    ci = option.get("effect_ci") or option.get("confidence_interval") or option.get("ci")
    if isinstance(ci, (list, tuple)) and len(ci) >= 2:
        lo = _safe_float(ci[0])
        hi = _safe_float(ci[1])
        if lo is not None and hi is not None:
            candidates.append(abs(float(hi) - float(lo)) / 2.0)

    evidence_quality = _clean_lower(option.get("evidence_quality") or option.get("causal_evidence_quality"))
    quality_penalty = {
        "very_high": 0.00,
        "high": 0.03,
        "medium": 0.10,
        "moderate": 0.10,
        "low": 0.22,
        "very_low": 0.35,
        "weak": 0.30,
    }.get(evidence_quality)
    if quality_penalty is not None:
        candidates.append(quality_penalty)
        codes.append(f"SFM_RECOMMENDATION_EVIDENCE_QUALITY_{evidence_quality.upper()}")

    if not candidates:
        return 0.05, ["SFM_RECOMMENDATION_UNCERTAINTY_DEFAULT_PENALTY"]
    penalty = min(0.45, max(0.0, max(candidates)))
    if penalty >= 0.20:
        codes.append("SFM_RECOMMENDATION_HIGH_UNCERTAINTY_PENALTY")
    else:
        codes.append("SFM_RECOMMENDATION_UNCERTAINTY_PENALTY_APPLIED")
    return round(float(penalty), 6), codes


def _goal_score(option: Mapping[str, Any], goals: Sequence[GoalSpec]) -> Tuple[float, List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    total_weight = sum(abs(float(goal.utility_weight or 0.0)) for goal in goals if goal.goal_variable) or 1.0
    score = 0.0
    for goal in goals:
        if not goal.goal_variable:
            continue
        value, source = _outcome_value(option, goal.goal_variable)
        utility = _directional_utility(value, goal.desired_direction, goal.metadata)
        contribution = abs(float(goal.utility_weight or 0.0)) * utility
        score += contribution
        rows.append(
            {
                "outcome": goal.goal_variable,
                "direction": goal.desired_direction,
                "weight": round(abs(float(goal.utility_weight or 0.0)), 6),
                "value": round(float(value), 6),
                "normalized_utility": round(float(utility), 6),
                "contribution": round(float(contribution), 6),
                "value_source": source,
            }
        )
    return round(float(score / total_weight), 6), rows


def _auxiliary_score(option: Mapping[str, Any], query: FinalCauseQuery, goals: Sequence[GoalSpec]) -> float:
    excluded = {goal.goal_variable for goal in goals}
    excluded.add(query.protected_outcome)
    score = 0.0
    total_weight = 0.0
    for outcome, weight in (query.agent.utility_model or {}).items():
        outcome = _clean_str(outcome)
        numeric = _safe_float(weight)
        if not outcome or numeric is None or outcome in excluded:
            continue
        direction = "decrease" if numeric < 0 or any(token in outcome.lower() for token in ["harm", "risk", "damage", "loss"]) else "increase"
        value, _source = _outcome_value(option, outcome)
        score += abs(float(numeric)) * _directional_utility(value, direction, {})
        total_weight += abs(float(numeric))
    if total_weight <= 1e-12:
        return 0.0
    return round(float(score / total_weight), 6)


def _constraint_rows(query: FinalCauseQuery) -> Tuple[Dict[str, Dict[str, Any]], List[str], List[str], Dict[str, Any]]:
    audit = ConstraintAwareEvaluator().evaluate(query).to_dict()
    rows = {str(row.get("action") or ""): dict(row) for row in audit.get("rankings") or []}
    feasible = [str(x) for x in audit.get("feasible_actions") or []]
    infeasible = [str(x) for x in audit.get("infeasible_actions") or []]
    return rows, feasible, infeasible, audit


def _normalise_constraint_penalty(row: Mapping[str, Any]) -> float:
    soft = _safe_float(row.get("soft_penalty"), 0.0) or 0.0
    hard_count = int(row.get("hard_violation_count") or 0)
    # Hard violations are primarily exposed as hard_blocked; retain a bounded
    # numeric penalty so ranking remains stable even if all actions violate.
    return round(min(1.0, float(soft) + 0.60 * hard_count), 6)


def _support_strength(matches: bool, top_margin: Optional[float], status: str) -> float:
    if status == "no_recommendable_action":
        return 0.0
    base = 0.72 if matches else 0.48
    if status == "requires_escalation":
        base = min(base, 0.58)
    if top_margin is not None:
        base += min(max(float(top_margin), 0.0), 0.18)
    return round(max(0.0, min(0.95, base)), 6)


def _top_margin(rankings: Sequence[Mapping[str, Any]]) -> Optional[float]:
    recommendable = [row for row in rankings if row.get("recommendable")]
    if len(recommendable) < 2:
        return None
    top = _safe_float(recommendable[0].get("recommendation_score"))
    second = _safe_float(recommendable[1].get("recommendation_score"))
    if top is None or second is None:
        return None
    return round(float(top - second), 6)


class SFMActionRecommender:
    """Rank candidate interventions under SFM goals, constraints, norms, and uncertainty."""

    def recommend(self, payload: Any) -> SFMActionRecommendationAudit:
        query = FinalCauseQuery.from_payload(payload)
        goals = _explicit_goal_bundle(query)
        if goals and not query.candidate_goals:
            # Reuse downstream constraint scoring, which reads query.candidate_goals.
            query.candidate_goals = list(goals)
        goal_bundle = [goal.goal_variable for goal in goals if goal.goal_variable]
        observed = query.observed_action

        if not query.candidate_actions:
            return SFMActionRecommendationAudit(
                assessed=False,
                action_variable=query.action_variable,
                observed_action=observed,
                goal_bundle=goal_bundle,
                recommendation_status="no_candidate_actions",
                reason="SFM recommendation requires candidate_actions with expected outcomes.",
                reason_codes=["SFM_RECOMMENDATION_REQUIRES_CANDIDATE_ACTIONS"],
                limits=["candidate_action_set_required"],
                raw=query.to_dict(),
            )
        if not goals:
            return SFMActionRecommendationAudit(
                assessed=False,
                action_variable=query.action_variable,
                observed_action=observed,
                goal_bundle=goal_bundle,
                recommendation_status="no_goal_bundle",
                reason="SFM recommendation requires at least one candidate goal or goal_bundle.",
                reason_codes=["SFM_RECOMMENDATION_REQUIRES_GOAL_BUNDLE"],
                limits=["goal_bundle_required"],
                raw=query.to_dict(),
            )

        policy = _policy(query)
        goal_allowed, goal_requires_escalation, prohibited_goals, escalation_goals = _goal_policy_status(policy, goals)
        constraint_by_action, feasible_from_constraints, blocked_from_constraints, constraint_audit = _constraint_rows(query)
        rows: List[RecommendedInterventionAction] = []

        for option in query.candidate_actions:
            option = _as_dict(option)
            action = _action_name(option)
            goal_score, goal_contributions = _goal_score(option, goals)
            auxiliary_score = _auxiliary_score(option, query, goals)
            constraint_row = constraint_by_action.get(action, {})
            constraint_feasible = bool(constraint_row.get("feasible", True))
            constraint_penalty = _normalise_constraint_penalty(constraint_row) if constraint_row else 0.0
            risk_penalty = _risk_penalty(option)
            uncertainty_penalty, uncertainty_codes = _option_uncertainty_penalty(option)
            action_status, action_allowed, action_requires_escalation, action_blocked = _action_policy_status(policy, action)

            requires_escalation = bool(goal_requires_escalation or action_requires_escalation)
            hard_blocked = bool((not constraint_feasible) or (not goal_allowed) or action_blocked)
            normatively_allowed = bool(goal_allowed and action_allowed and not action_blocked)
            normative_penalty = 0.0
            reason_codes: List[str] = ["SFM_RECOMMENDATION_ACTION_SCORED"]
            reason_codes.extend(uncertainty_codes)

            if not constraint_feasible:
                reason_codes.append("SFM_RECOMMENDATION_ACTION_BLOCKED_BY_HARD_CONSTRAINT")
            else:
                reason_codes.append("SFM_RECOMMENDATION_ACTION_SATISFIES_HARD_CONSTRAINTS")
            if not goal_allowed:
                reason_codes.append("SFM_RECOMMENDATION_GOAL_BUNDLE_NORMATIVELY_BLOCKED")
                normative_penalty += 1.0
            if action_blocked:
                reason_codes.append("SFM_RECOMMENDATION_ACTION_NORMATIVELY_BLOCKED")
                normative_penalty += 1.0
            elif action_status == "allowed":
                reason_codes.append("SFM_RECOMMENDATION_ACTION_NORMATIVELY_ALLOWED")
            elif action_status == "discouraged":
                reason_codes.append("SFM_RECOMMENDATION_ACTION_NORMATIVELY_DISCOURAGED")
                normative_penalty += 0.15
            if requires_escalation:
                reason_codes.append("SFM_RECOMMENDATION_ESCALATION_REQUIRED")
                normative_penalty += 0.20

            total_penalty = constraint_penalty + risk_penalty + uncertainty_penalty + normative_penalty
            score = goal_score + 0.20 * auxiliary_score - total_penalty
            if hard_blocked:
                score -= 10.0
            recommendable = not hard_blocked

            rows.append(
                RecommendedInterventionAction(
                    action=action,
                    recommendation_score=round(float(score), 6),
                    goal_score=round(float(goal_score), 6),
                    auxiliary_score=round(float(auxiliary_score), 6),
                    constraint_penalty=round(float(constraint_penalty), 6),
                    risk_penalty=round(float(risk_penalty), 6),
                    uncertainty_penalty=round(float(uncertainty_penalty), 6),
                    normative_penalty=round(float(normative_penalty), 6),
                    total_penalty=round(float(total_penalty), 6),
                    constraint_feasible=constraint_feasible,
                    normatively_allowed=normatively_allowed,
                    requires_escalation=requires_escalation,
                    hard_blocked=hard_blocked,
                    recommendable=recommendable,
                    goal_contributions=goal_contributions,
                    constraint_evaluations=list(constraint_row.get("constraint_evaluations") or []),
                    normative_status=action_status,
                    violated_constraints=list(constraint_row.get("violated_constraints") or []),
                    reason_codes=reason_codes,
                    raw_option=dict(option),
                )
            )

        rankings = [row.to_dict() for row in rows]
        rankings.sort(
            key=lambda item: (
                bool(item.get("recommendable")),
                float(item.get("recommendation_score", float("-inf"))),
                not bool(item.get("requires_escalation")),
                str(item.get("action", "")),
            ),
            reverse=True,
        )
        for idx, row in enumerate(rankings, start=1):
            row["rank"] = idx

        recommendable = [row for row in rankings if row.get("recommendable")]
        selected_row = recommendable[0] if recommendable else (rankings[0] if rankings else {})
        recommended_action = str(selected_row.get("action") or "")
        matches = bool(observed and recommended_action and observed == recommended_action)
        feasible_actions = [str(row.get("action") or "") for row in rankings if row.get("recommendable") and not row.get("requires_escalation")]
        escalation_actions = [str(row.get("action") or "") for row in rankings if row.get("recommendable") and row.get("requires_escalation")]
        blocked_actions = [str(row.get("action") or "") for row in rankings if not row.get("recommendable")]
        margin = _top_margin(rankings)

        if not recommendable:
            status = "no_recommendable_action"
        elif bool(selected_row.get("requires_escalation")):
            status = "requires_escalation"
        else:
            status = "recommended"

        expression = _format_do_star_expression(
            action_variable=query.action_variable,
            selected_action=recommended_action or "<no-recommendable-action>",
            policy_name=_clean_str(_as_dict(query.raw).get("recommendation_policy_name") or _as_dict(query.raw).get("policy_name"), "sfm_recommendation_policy"),
            agent_id=query.agent.agent_id,
            goals=goals,
        )
        recommended_intervention = {
            "operator": "do_star",
            "expression": expression,
            "policy_signature": "recommendation_policy(S, B, G_bundle, U, C, N, uncertainty)",
            "selected_action": recommended_action,
            "goal_bundle": goal_bundle,
        }

        reason_codes: List[str] = ["SFM_ACTION_RECOMMENDATION_ASSESSED"]
        if status == "recommended":
            reason_codes.append("SFM_ACTION_RECOMMENDATION_SELECTED_ACTION")
        elif status == "requires_escalation":
            reason_codes.append("SFM_ACTION_RECOMMENDATION_SELECTED_ESCALATION_ACTION")
        else:
            reason_codes.append("SFM_ACTION_RECOMMENDATION_NO_RECOMMENDABLE_ACTION")
        if matches:
            reason_codes.append("SFM_ACTION_RECOMMENDATION_MATCHES_OBSERVED_ACTION")
        elif observed:
            reason_codes.append("SFM_ACTION_RECOMMENDATION_DIFFERS_FROM_OBSERVED_ACTION")
        if prohibited_goals:
            reason_codes.append("SFM_ACTION_RECOMMENDATION_GOAL_BUNDLE_HAS_PROHIBITED_GOALS")
        if escalation_goals:
            reason_codes.append("SFM_ACTION_RECOMMENDATION_GOAL_BUNDLE_REQUIRES_ESCALATION")
        if blocked_actions:
            reason_codes.append("SFM_ACTION_RECOMMENDATION_FILTERED_BLOCKED_ACTIONS")
        if constraint_audit.get("assessed"):
            reason_codes.extend(constraint_audit.get("reason_codes") or [])

        limits: List[str] = ["recommendation_is_diagnostic_not_policy_proof"]
        if not query.scm_graph:
            limits.append("scm_graph_not_supplied_for_recommendation")
        if any(row.get("uncertainty_penalty", 0.0) >= 0.20 for row in rankings):
            limits.append("some_candidate_actions_have_high_uncertainty")
        if status == "requires_escalation":
            limits.append("recommended_action_requires_escalation_before_execution")
        if prohibited_goals:
            limits.append("goal_bundle_contains_normatively_blocked_goal")
        if blocked_actions:
            limits.append("some_candidate_actions_were_blocked_by_constraints_or_normative_policy")
        limits.extend(constraint_audit.get("limits") or [])

        return SFMActionRecommendationAudit(
            assessed=True,
            action_variable=query.action_variable,
            observed_action=observed,
            recommended_action=recommended_action,
            selected_action=recommended_action,
            recommendation_matches_observed=matches,
            recommendation_status=status,
            support_strength=_support_strength(matches, margin, status),
            top_margin=margin,
            goal_bundle=goal_bundle,
            feasible_actions=feasible_actions,
            blocked_actions=blocked_actions,
            escalation_actions=escalation_actions,
            rankings=rankings,
            recommended_intervention=recommended_intervention,
            reason=(
                "SFM action recommendation ranked candidate interventions under final goals, constraints, "
                "normative/value policy, risk, and uncertainty."
            ),
            reason_codes=reason_codes,
            limits=_unique(limits),
            raw={
                "query": query.to_dict(),
                "constraint_audit": constraint_audit,
                "prohibited_goals": prohibited_goals,
                "escalation_goals": escalation_goals,
            },
        )


def recommend_sfm_action(payload: Any) -> Dict[str, Any]:
    """Convenience API for SFM intervention recommendation."""

    return SFMActionRecommender().recommend(payload).to_dict()
