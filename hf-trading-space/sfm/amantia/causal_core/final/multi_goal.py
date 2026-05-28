from __future__ import annotations

"""Multi-goal / multi-objective diagnostics for Structural Final Models.

Earlier SFM development steps tested one candidate final cause at a time.  Real
agents often act under a *bundle* of ends: maximize task success, preserve
safety, maintain user satisfaction, reduce latency, and avoid side effects.  A
single-goal diagnostic can therefore miss the case where the observed action is
not optimal for any one goal alone, but is optimal for the weighted bundle.

This module adds a conservative multi-objective audit.  It does not prove a
final cause; it asks whether a supplied goal bundle, protected constraints, and
agent utility model select the observed action.
"""

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .schema import FinalCauseQuery, GoalSpec
from .utility import _outcome_value, _directional_utility, _risk_penalty


@dataclass
class MultiGoalContribution:
    """Contribution of one goal or constraint to one action score."""

    outcome: str = ""
    role: str = "primary_goal"
    direction: str = "increase"
    weight: float = 1.0
    value: Optional[float] = None
    normalized_utility: Optional[float] = None
    contribution: float = 0.0
    value_source: str = ""
    active: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MultiGoalActionScore:
    """Multi-objective score decomposition for one candidate action."""

    action: str = ""
    total_score: float = 0.0
    primary_goal_score: float = 0.0
    auxiliary_score: float = 0.0
    protected_penalty: float = 0.0
    risk_penalty: float = 0.0
    goal_coverage_ratio: float = 0.0
    satisfied_goals: List[str] = field(default_factory=list)
    constraint_violations: List[str] = field(default_factory=list)
    contributions: List[Dict[str, Any]] = field(default_factory=list)
    raw_option: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MultiGoalUtilityAudit:
    """Aggregate multi-goal SFM audit."""

    assessed: bool = False
    goal_bundle: List[str] = field(default_factory=list)
    observed_action: str = ""
    selected_action: str = ""
    selected_action_matches_observed: bool = False
    observed_rank: Optional[int] = None
    support_strength: float = 0.0
    top_margin: Optional[float] = None
    tradeoff_detected: bool = False
    bundle_score_exceeds_best_single_goal_score: bool = False
    best_single_goal_actions: Dict[str, str] = field(default_factory=dict)
    dominant_goals: List[str] = field(default_factory=list)
    rankings: List[Dict[str, Any]] = field(default_factory=list)
    authority_status: str = "diagnostic_only"
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


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _clip01(value: Optional[float], default: float = 0.5) -> float:
    if value is None:
        value = default
    return max(0.0, min(1.0, float(value)))


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


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


def _unique(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    for item in items:
        text = _clean_str(item)
        if text and text not in out:
            out.append(text)
    return out


def _protected_outcomes(query: FinalCauseQuery, goals: Sequence[GoalSpec]) -> List[str]:
    values: List[str] = [query.protected_outcome]
    for goal in goals:
        values.extend(goal.protected_outcomes)
    return _unique(values)


def _bundle_goals(query: FinalCauseQuery) -> List[GoalSpec]:
    raw = _as_dict(query.raw)
    explicit = raw.get("multi_goal_bundle") or raw.get("goal_bundle") or raw.get("multi_objective_goals")
    if explicit:
        if isinstance(explicit, list):
            return [GoalSpec.from_payload(item) for item in explicit]
        return [GoalSpec.from_payload(explicit)]
    return list(query.candidate_goals or [])


def _goal_contribution(option: Mapping[str, Any], goal: GoalSpec) -> MultiGoalContribution:
    value, source = _outcome_value(option, goal.goal_variable)
    utility = _directional_utility(value, goal.desired_direction, goal.metadata)
    contribution = float(goal.utility_weight) * float(utility)
    return MultiGoalContribution(
        outcome=goal.goal_variable,
        role="primary_goal",
        direction=goal.desired_direction,
        weight=round(float(goal.utility_weight), 6),
        value=round(float(value), 6),
        normalized_utility=round(float(utility), 6),
        contribution=round(float(contribution), 6),
        value_source=source,
        active=True,
    )


def _protected_contribution(option: Mapping[str, Any], outcome: str) -> MultiGoalContribution:
    value, source = _outcome_value(option, outcome, default=0.0)
    utility = _directional_utility(value, "decrease", {})
    # The contribution is positive when the protected outcome is low.  The audit
    # exposes the corresponding penalty separately.
    return MultiGoalContribution(
        outcome=outcome,
        role="protected_constraint",
        direction="decrease",
        weight=1.0,
        value=round(float(value), 6),
        normalized_utility=round(float(utility), 6),
        contribution=round(float(utility), 6),
        value_source=source,
        active=True,
    )


def _auxiliary_contributions(
    option: Mapping[str, Any],
    query: FinalCauseQuery,
    excluded: Iterable[str],
) -> List[MultiGoalContribution]:
    out: List[MultiGoalContribution] = []
    excluded_set = set(excluded)
    for outcome, weight in (query.agent.utility_model or {}).items():
        outcome = _clean_str(outcome)
        numeric = _safe_float(weight)
        if not outcome or outcome in excluded_set or numeric is None:
            continue
        direction = "decrease" if numeric < 0 or "harm" in outcome.lower() or "risk" in outcome.lower() else "increase"
        value, source = _outcome_value(option, outcome)
        utility = _directional_utility(value, direction, {})
        contribution = abs(float(numeric)) * float(utility)
        out.append(
            MultiGoalContribution(
                outcome=outcome,
                role="auxiliary",
                direction=direction,
                weight=round(abs(float(numeric)), 6),
                value=round(float(value), 6),
                normalized_utility=round(float(utility), 6),
                contribution=round(float(contribution), 6),
                value_source=source,
                active=True,
            )
        )
    return out


def _evaluate_action(
    option: Mapping[str, Any],
    *,
    query: FinalCauseQuery,
    goals: Sequence[GoalSpec],
    protected: Sequence[str],
) -> MultiGoalActionScore:
    action = _action_name(option)
    goal_rows = [_goal_contribution(option, goal) for goal in goals]
    protected_rows = [_protected_contribution(option, outcome) for outcome in protected if outcome]
    excluded = [*(goal.goal_variable for goal in goals), *protected]
    auxiliary_rows = _auxiliary_contributions(option, query, excluded)

    primary_score = sum(row.contribution for row in goal_rows)
    auxiliary_score = sum(row.contribution for row in auxiliary_rows)
    protected_penalty = sum(max(0.0, float(row.weight) - float(row.contribution)) for row in protected_rows)
    risk_penalty = _risk_penalty(option)
    total = primary_score + auxiliary_score - protected_penalty - risk_penalty

    satisfied = [row.outcome for row in goal_rows if (row.normalized_utility or 0.0) >= 0.6]
    coverage = len(satisfied) / max(len(goal_rows), 1)
    violations = [row.outcome for row in protected_rows if (row.value or 0.0) >= 0.5]

    return MultiGoalActionScore(
        action=action,
        total_score=round(float(total), 6),
        primary_goal_score=round(float(primary_score), 6),
        auxiliary_score=round(float(auxiliary_score), 6),
        protected_penalty=round(float(protected_penalty), 6),
        risk_penalty=round(float(risk_penalty), 6),
        goal_coverage_ratio=round(float(coverage), 6),
        satisfied_goals=satisfied,
        constraint_violations=violations,
        contributions=[row.to_dict() for row in [*goal_rows, *protected_rows, *auxiliary_rows]],
        raw_option=dict(option),
    )


def _rank(rows: List[MultiGoalActionScore]) -> List[Dict[str, Any]]:
    ranked = [row.to_dict() for row in rows]
    ranked.sort(key=lambda item: (item.get("total_score", float("-inf")), item.get("action", "")), reverse=True)
    for idx, row in enumerate(ranked, start=1):
        row["rank"] = idx
    return ranked


def _margin(rankings: Sequence[Mapping[str, Any]]) -> Optional[float]:
    if len(rankings) < 2:
        return None
    top = _safe_float(rankings[0].get("total_score"))
    second = _safe_float(rankings[1].get("total_score"))
    if top is None or second is None:
        return None
    return round(float(top - second), 6)


def _observed_rank(rankings: Sequence[Mapping[str, Any]], observed_action: str) -> Optional[int]:
    for row in rankings:
        if row.get("action") == observed_action:
            value = row.get("rank")
            return int(value) if value is not None else None
    return None


def _support_strength(matches: bool, observed_rank: Optional[int], margin: Optional[float]) -> float:
    if matches:
        return round(min(1.0, 0.65 + max(float(margin or 0.0), 0.0) / 2.0), 6)
    if observed_rank is None:
        return 0.0
    return round(max(0.0, 0.45 - 0.10 * max(observed_rank - 1, 0)), 6)


def _best_single_goal_actions(query: FinalCauseQuery, goals: Sequence[GoalSpec]) -> Dict[str, str]:
    best: Dict[str, str] = {}
    for goal in goals:
        rows = []
        for option in query.candidate_actions:
            contribution = _goal_contribution(option, goal)
            rows.append({"action": _action_name(option), "score": contribution.contribution})
        rows.sort(key=lambda item: (item.get("score", float("-inf")), item.get("action", "")), reverse=True)
        best[goal.goal_variable] = rows[0]["action"] if rows else ""
    return best


def _best_single_goal_score(query: FinalCauseQuery, goals: Sequence[GoalSpec], action: str) -> float:
    scores: List[float] = []
    for goal in goals:
        for option in query.candidate_actions:
            if _action_name(option) == action:
                scores.append(_goal_contribution(option, goal).contribution)
    return max(scores) if scores else 0.0


def _dominant_goals(rankings: Sequence[Mapping[str, Any]]) -> List[str]:
    if not rankings:
        return []
    top = rankings[0]
    contribs = [c for c in top.get("contributions", []) if c.get("role") == "primary_goal"]
    contribs.sort(key=lambda item: (item.get("contribution", 0.0), item.get("outcome", "")), reverse=True)
    return [str(item.get("outcome")) for item in contribs[:3] if item.get("outcome")]


class MultiGoalUtilityEvaluator:
    """Evaluate whether a goal bundle selects the observed action."""

    def evaluate(self, payload: Any) -> MultiGoalUtilityAudit:
        query = FinalCauseQuery.from_payload(payload)
        goals = _bundle_goals(query)
        goal_bundle = [goal.goal_variable for goal in goals]

        if len(goals) < 2:
            return MultiGoalUtilityAudit(
                assessed=False,
                goal_bundle=goal_bundle,
                observed_action=query.observed_action,
                reason="At least two goals are required for multi-goal SFM evaluation.",
                reason_codes=["SFM_MULTI_GOAL_REQUIRES_AT_LEAST_TWO_GOALS"],
                limits=["multi_goal_bundle_requires_two_or_more_goals"],
                raw=query.to_dict(),
            )
        if len(query.candidate_actions) < 2:
            return MultiGoalUtilityAudit(
                assessed=False,
                goal_bundle=goal_bundle,
                observed_action=query.observed_action,
                reason="At least two candidate actions are required for multi-goal SFM evaluation.",
                reason_codes=["SFM_MULTI_GOAL_INSUFFICIENT_ACTION_ALTERNATIVES"],
                limits=["candidate_action_set_required"],
                raw=query.to_dict(),
            )

        protected = _protected_outcomes(query, goals)
        rankings = _rank([
            _evaluate_action(option, query=query, goals=goals, protected=protected)
            for option in query.candidate_actions
        ])
        selected = str(rankings[0].get("action") or "") if rankings else ""
        observed = query.observed_action or selected
        matches = bool(selected and selected == observed)
        observed_rank = _observed_rank(rankings, observed)
        margin = _margin(rankings)
        best_single = _best_single_goal_actions(query, goals)
        selected_by_any_single_goal = selected in set(best_single.values())
        tradeoff = not selected_by_any_single_goal or any(
            item.get("action") != selected
            and float(item.get("primary_goal_score", 0.0)) > float(rankings[0].get("primary_goal_score", 0.0))
            and float(item.get("total_score", 0.0)) < float(rankings[0].get("total_score", 0.0))
            for item in rankings[1:]
        )
        best_single_score = _best_single_goal_score(query, goals, selected)
        bundle_beats_single = float(rankings[0].get("primary_goal_score", 0.0)) > best_single_score

        reason_codes: List[str] = ["SFM_MULTI_GOAL_UTILITY_ASSESSED"]
        if matches:
            reason_codes.append("SFM_MULTI_GOAL_SUPPORTS_OBSERVED_ACTION")
        else:
            reason_codes.append("SFM_MULTI_GOAL_DOES_NOT_SELECT_OBSERVED_ACTION")
        if tradeoff:
            reason_codes.append("SFM_MULTI_GOAL_TRADEOFF_DETECTED")
        if protected:
            reason_codes.append("SFM_MULTI_GOAL_PROTECTED_CONSTRAINTS_INCLUDED")
        if bundle_beats_single:
            reason_codes.append("SFM_MULTI_GOAL_BUNDLE_EXPLAINS_MORE_THAN_SINGLE_GOAL")

        return MultiGoalUtilityAudit(
            assessed=True,
            goal_bundle=goal_bundle,
            observed_action=observed,
            selected_action=selected,
            selected_action_matches_observed=matches,
            observed_rank=observed_rank,
            support_strength=_support_strength(matches, observed_rank, margin),
            top_margin=margin,
            tradeoff_detected=tradeoff,
            bundle_score_exceeds_best_single_goal_score=bundle_beats_single,
            best_single_goal_actions=best_single,
            dominant_goals=_dominant_goals(rankings),
            rankings=rankings,
            reason=(
                "Multi-goal SFM audit ranked candidate actions using the weighted goal bundle, "
                "protected constraints, auxiliary utilities, and risk penalties."
            ),
            reason_codes=reason_codes,
            limits=["diagnostic_only_not_full_multi_objective_sfm_identification"],
            raw={"query": query.to_dict(), "goal_bundle": [goal.to_dict() for goal in goals]},
        )


def evaluate_multi_goal_utility(payload: Any) -> Dict[str, Any]:
    """Convenience API for multi-goal / multi-objective SFM diagnostics."""

    return MultiGoalUtilityEvaluator().evaluate(payload).to_dict()
