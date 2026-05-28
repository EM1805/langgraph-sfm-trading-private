from __future__ import annotations

"""Goal-removal twin-policy diagnostics for SFM development.

This module implements a conservative first approximation of the SFM twin
comparison:

    policy_with_goal(G)     -> which action would a goal-directed policy select?
    policy_without_goal(G)  -> which action would the same policy select when G
                               is removed from the utility model?

It is intentionally diagnostic.  It does not claim complete Pearl-style
structural counterfactual identification; it gives Amantia a stable, auditable
surface for testing whether an observed action depends on a candidate final
cause.
"""

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .schema import AgentModel, FinalCauseQuery, GoalSpec


_RISK_PENALTY = {
    "none": 0.0,
    "low": 0.05,
    "medium": 0.18,
    "moderate": 0.18,
    "high": 0.38,
    "critical": 0.65,
    "unknown": 0.12,
}

_PROTECTED_DEFAULTS = {
    "harm",
    "risk",
    "damage",
    "loss",
    "injury",
    "toxicity",
    "unsafe",
    "user_or_system_harm",
}


@dataclass
class TwinPolicyComparison:
    compared: bool = False
    goal_variable: str = ""
    observed_action: str = ""
    selected_with_goal: str = ""
    selected_without_goal: str = ""
    observed_selected_with_goal: bool = False
    action_changes_when_goal_removed: bool = False
    goal_dependence_score: float = 0.0
    with_goal_margin: Optional[float] = None
    without_goal_margin: Optional[float] = None
    observed_with_goal_score: Optional[float] = None
    observed_without_goal_score: Optional[float] = None
    with_goal_rankings: List[Dict[str, Any]] = field(default_factory=list)
    without_goal_rankings: List[Dict[str, Any]] = field(default_factory=list)
    authority_status: str = "diagnostic_only"
    reason: str = ""
    reason_codes: List[str] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _clean_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _clean_lower(value: Any, default: str = "unknown") -> str:
    return _clean_str(value, default).lower()


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


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


def _first(payload: Mapping[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    for key in keys:
        if key in payload and payload.get(key) not in (None, ""):
            return payload.get(key)
    return default


def _mapping_value(option: Mapping[str, Any], containers: Sequence[str], key: str) -> Tuple[Optional[float], str]:
    for container_name in containers:
        container = _as_dict(option.get(container_name))
        if key in container:
            return _safe_float(container.get(key)), container_name
    return None, ""


def _outcome_value(option: Mapping[str, Any], outcome: str, *, default: float = 0.5) -> Tuple[float, str]:
    """Extract an expected outcome value for an action option.

    Supported compact shapes include:
    - expected_outcomes={"task_success": 0.8}
    - outcome_scores={"task_success": 0.8}
    - goal_scores={"task_success": 0.8}
    - effect_estimates={"task_success": 0.2}  # interpreted around 0.5
    - direct fields such as expected_success, harm_probability, p_harm
    """

    outcome = _clean_str(outcome)
    direct_value = _safe_float(option.get(outcome)) if outcome and outcome in option else None
    if direct_value is not None:
        return _clip01(direct_value), f"direct:{outcome}"

    value, source = _mapping_value(
        option,
        ["expected_outcomes", "outcome_scores", "goal_scores", "utility_scores", "utilities"],
        outcome,
    )
    if value is not None:
        return _clip01(value), source

    effect, source = _mapping_value(option, ["effect_estimates", "effects", "effect_scores"], outcome)
    if effect is not None:
        return _clip01(0.5 + float(effect)), source

    outcome_l = outcome.lower()
    if outcome_l in {"task_success", "success", "expected_success", "p_success"}:
        value = _safe_float(_first(option, ["expected_success", "success_probability", "p_success"]))
        if value is not None:
            return _clip01(value), "expected_success"

    if outcome_l in {"user_or_system_harm", "harm", "expected_harm", "p_harm"} or "harm" in outcome_l:
        value = _safe_float(_first(option, ["harm_probability", "expected_harm", "p_harm"]))
        if value is not None:
            return _clip01(value), "harm_probability"

    effect = _safe_float(_first(option, ["effect_estimate", "effect", "ate", "expected_effect"]))
    if effect is not None:
        return _clip01(0.5 + float(effect)), "effect_estimate"

    return _clip01(default), "default"


def _directional_utility(value: float, goal: GoalSpec) -> float:
    direction = (goal.desired_direction or "increase").strip().lower()
    value = _clip01(value)
    if direction in {"decrease", "minimize", "minimise", "reduce", "lower"}:
        return 1.0 - value
    if direction in {"maintain", "stabilize", "stabilise", "keep"}:
        target = _safe_float(goal.metadata.get("target"), 0.5) if isinstance(goal.metadata, Mapping) else 0.5
        target = _clip01(target, 0.5)
        return _clip01(1.0 - abs(value - target))
    # increase / achieve / maximize are treated as monotone-improving.
    return value


def _risk_penalty(option: Mapping[str, Any]) -> float:
    risk = _clean_lower(_first(option, ["risk", "risk_level"], "unknown"), "unknown")
    return float(_RISK_PENALTY.get(risk, _RISK_PENALTY["unknown"]))


def _harm_penalty(option: Mapping[str, Any], protected_outcomes: Iterable[str]) -> float:
    penalties: List[float] = []
    direct_harm = _safe_float(_first(option, ["harm_probability", "expected_harm", "p_harm"]))
    if direct_harm is not None:
        penalties.append(_clip01(direct_harm, 0.0))

    for outcome in protected_outcomes:
        if not outcome:
            continue
        value, source = _outcome_value(option, outcome, default=0.0)
        # Avoid counting the same direct harm value twice unless the caller gave
        # a distinct protected outcome score.
        if source == "harm_probability" and direct_harm is not None:
            continue
        outcome_l = outcome.lower()
        if outcome_l in _PROTECTED_DEFAULTS or any(token in outcome_l for token in ["harm", "risk", "damage", "loss"]):
            penalties.append(value)
    return max(penalties) if penalties else 0.0


def _other_utility(agent: AgentModel, option: Mapping[str, Any], excluded_goal: str, protected_outcomes: Iterable[str]) -> float:
    score = 0.0
    protected = set(protected_outcomes)
    for outcome, weight in (agent.utility_model or {}).items():
        outcome = str(outcome)
        if not outcome or outcome == excluded_goal:
            continue
        value, _source = _outcome_value(option, outcome)
        direction = "decrease" if outcome in protected or "harm" in outcome.lower() or "risk" in outcome.lower() else "increase"
        utility = _directional_utility(value, GoalSpec(goal_variable=outcome, desired_direction=direction))
        score += float(weight) * utility
    return score


def _evaluate_action(
    option: Mapping[str, Any],
    *,
    goal: GoalSpec,
    agent: AgentModel,
    protected_outcome: str,
    include_goal: bool,
) -> Dict[str, Any]:
    name = _action_name(option)
    protected = [protected_outcome, *goal.protected_outcomes]
    goal_value, goal_source = _outcome_value(option, goal.goal_variable)
    goal_utility = _directional_utility(goal_value, goal)
    risk_penalty = _risk_penalty(option)
    harm_penalty = _harm_penalty(option, protected)
    other_utility = _other_utility(agent, option, goal.goal_variable, protected)

    # The no-goal twin keeps safety and any other explicit agent utilities, but
    # removes the candidate final cause from the policy.  A neutral 0.5 baseline
    # prevents all no-goal scores from collapsing to pure negatives.
    base_score = other_utility if abs(other_utility) > 0 else 0.5
    goal_component = float(goal.utility_weight) * goal_utility if include_goal else 0.0
    score = base_score + goal_component - risk_penalty - harm_penalty

    return {
        "action": name,
        "score": round(float(score), 6),
        "goal_value": round(float(goal_value), 6),
        "goal_utility": round(float(goal_utility), 6),
        "goal_source": goal_source,
        "goal_component": round(float(goal_component), 6),
        "other_utility": round(float(other_utility), 6),
        "risk_penalty": round(float(risk_penalty), 6),
        "harm_penalty": round(float(harm_penalty), 6),
        "raw_option": dict(option),
    }


def _rank_actions(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = [dict(row) for row in rows]
    rows.sort(key=lambda item: (item.get("score", float("-inf")), item.get("action", "")), reverse=True)
    for idx, row in enumerate(rows, start=1):
        row["rank"] = idx
    return rows


def _margin(rankings: List[Dict[str, Any]]) -> Optional[float]:
    if len(rankings) < 2:
        return None
    top = _safe_float(rankings[0].get("score"))
    second = _safe_float(rankings[1].get("score"))
    if top is None or second is None:
        return None
    return round(float(top - second), 6)


def _score_for(rankings: List[Dict[str, Any]], action: str) -> Optional[float]:
    for row in rankings:
        if row.get("action") == action:
            value = _safe_float(row.get("score"))
            return round(float(value), 6) if value is not None else None
    return None


class TwinPolicyComparator:
    """Compare action policies with and without a candidate goal."""

    def compare(self, query: FinalCauseQuery, goal: GoalSpec) -> TwinPolicyComparison:
        if len(query.candidate_actions) < 2:
            return TwinPolicyComparison(
                compared=False,
                goal_variable=goal.goal_variable,
                observed_action=query.observed_action,
                reason="At least two candidate actions are required for a twin-policy comparison.",
                reason_codes=["SFM_TWIN_INSUFFICIENT_ACTION_ALTERNATIVES"],
                raw=query.to_dict(),
            )

        with_goal = _rank_actions([
            _evaluate_action(
                option,
                goal=goal,
                agent=query.agent,
                protected_outcome=query.protected_outcome,
                include_goal=True,
            )
            for option in query.candidate_actions
        ])
        without_goal = _rank_actions([
            _evaluate_action(
                option,
                goal=goal,
                agent=query.agent,
                protected_outcome=query.protected_outcome,
                include_goal=False,
            )
            for option in query.candidate_actions
        ])

        selected_with = with_goal[0]["action"] if with_goal else ""
        selected_without = without_goal[0]["action"] if without_goal else ""
        observed = query.observed_action or selected_with
        observed_selected_with = selected_with == observed
        changed = bool(selected_with and selected_without and selected_with != selected_without)

        if observed_selected_with and changed:
            dependence_score = 1.0
        elif observed_selected_with:
            dependence_score = 0.55
        elif changed:
            dependence_score = 0.35
        else:
            dependence_score = 0.0

        reason_codes: List[str] = []
        if observed_selected_with:
            reason_codes.append("SFM_TWIN_WITH_GOAL_SELECTS_OBSERVED")
        else:
            reason_codes.append("SFM_TWIN_WITH_GOAL_DOES_NOT_SELECT_OBSERVED")
        if changed:
            reason_codes.append("SFM_TWIN_REMOVING_GOAL_CHANGES_ACTION")
        else:
            reason_codes.append("SFM_TWIN_GOAL_REMOVAL_DOES_NOT_CHANGE_ACTION")

        return TwinPolicyComparison(
            compared=True,
            goal_variable=goal.goal_variable,
            observed_action=observed,
            selected_with_goal=selected_with,
            selected_without_goal=selected_without,
            observed_selected_with_goal=observed_selected_with,
            action_changes_when_goal_removed=changed,
            goal_dependence_score=round(float(dependence_score), 6),
            with_goal_margin=_margin(with_goal),
            without_goal_margin=_margin(without_goal),
            observed_with_goal_score=_score_for(with_goal, observed),
            observed_without_goal_score=_score_for(without_goal, observed),
            with_goal_rankings=with_goal,
            without_goal_rankings=without_goal,
            reason=(
                "Twin-policy diagnostic compared a policy that includes the candidate goal "
                "with a policy that removes that goal while retaining safety and other supplied utilities."
            ),
            reason_codes=reason_codes,
            raw={"query": query.to_dict(), "goal": goal.to_dict()},
        )


def compare_twin_policies(payload: Any, goal: GoalSpec | str | Mapping[str, Any] | None = None) -> Dict[str, Any]:
    query = FinalCauseQuery.from_payload(payload)
    goal_obj = GoalSpec.from_payload(goal) if goal is not None else query.candidate_goals[0]
    return TwinPolicyComparator().compare(query, goal_obj).to_dict()
