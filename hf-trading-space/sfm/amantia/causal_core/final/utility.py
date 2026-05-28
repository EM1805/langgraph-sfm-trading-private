from __future__ import annotations

"""Explicit utility-function diagnostics for Structural Final Model development.

SFM needs a policy-level account of *why* an action was selected.  Earlier
steps used an implicit utility inside the twin-policy comparator.  This module
makes the utility function auditable by decomposing action choice into:

- primary goal utility: the candidate final cause the model is testing;
- protected constraints: harms/risks the agent should avoid;
- side-effect monitoring: outcomes that may occur but should not automatically
  be promoted to intended ends;
- auxiliary utilities: any other explicit preferences supplied for the agent.

The evaluator is deliberately diagnostic.  It does not prove intent; it exposes
which terms of the utility function make the observed action look selected.
"""

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .schema import FinalCauseQuery, GoalSpec


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
class UtilityComponent:
    """One term in the agent utility function."""

    outcome: str = ""
    weight: float = 1.0
    direction: str = "increase"
    role: str = "auxiliary"
    source: str = "derived"
    active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ActionUtilityBreakdown:
    """Utility decomposition for one candidate action."""

    action: str = ""
    total_utility: float = 0.0
    primary_goal_utility: float = 0.0
    protected_penalty: float = 0.0
    side_effect_utility: float = 0.0
    auxiliary_utility: float = 0.0
    risk_penalty: float = 0.0
    constraint_violations: List[str] = field(default_factory=list)
    components: List[Dict[str, Any]] = field(default_factory=list)
    raw_option: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class UtilityFunctionAudit:
    """Aggregate utility audit for an SFM final-cause query."""

    assessed: bool = False
    goal_variable: str = ""
    observed_action: str = ""
    selected_action: str = ""
    selected_action_matches_observed: bool = False
    observed_rank: Optional[int] = None
    top_margin: Optional[float] = None
    tradeoff_detected: bool = False
    primary_goal_weight: float = 0.0
    protected_constraint_weight: float = 0.0
    side_effect_weight: float = 0.0
    utility_components: List[Dict[str, Any]] = field(default_factory=list)
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


def _clean_lower(value: Any, default: str = "unknown") -> str:
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
    return [value]


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


def _first(payload: Mapping[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    for key in keys:
        if key in payload and payload.get(key) not in (None, ""):
            return payload.get(key)
    return default


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


def _mapping_value(option: Mapping[str, Any], containers: Sequence[str], key: str) -> Tuple[Optional[float], str]:
    for container_name in containers:
        container = _as_dict(option.get(container_name))
        if key in container:
            return _safe_float(container.get(key)), container_name
    return None, ""


def _outcome_value(option: Mapping[str, Any], outcome: str, *, default: float = 0.5) -> Tuple[float, str]:
    """Extract a normalized expected value for an outcome under an action."""

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


def _directional_utility(value: float, direction: str, metadata: Optional[Mapping[str, Any]] = None) -> float:
    direction_l = (direction or "increase").strip().lower()
    value = _clip01(value)
    if direction_l in {"decrease", "minimize", "minimise", "reduce", "lower", "avoid"}:
        return 1.0 - value
    if direction_l in {"maintain", "stabilize", "stabilise", "keep"}:
        target = _safe_float((metadata or {}).get("target"), 0.5)
        target = _clip01(target, 0.5)
        return _clip01(1.0 - abs(value - target))
    return value


def _risk_penalty(option: Mapping[str, Any]) -> float:
    risk = _clean_lower(_first(option, ["risk", "risk_level"], "unknown"), "unknown")
    return float(_RISK_PENALTY.get(risk, _RISK_PENALTY["unknown"]))


def _is_protected_like(outcome: str) -> bool:
    outcome_l = outcome.lower()
    return outcome_l in _PROTECTED_DEFAULTS or any(token in outcome_l for token in ["harm", "risk", "damage", "loss"])


def _role_for_weight(outcome: str, weight: float, protected: Iterable[str], side_effects: Iterable[str], goal: str) -> str:
    if outcome == goal:
        return "primary_goal"
    if outcome in set(protected) or _is_protected_like(outcome):
        return "protected_constraint"
    if outcome in set(side_effects):
        return "side_effect"
    return "auxiliary"


def _default_direction(outcome: str, role: str, weight: float) -> str:
    if role == "protected_constraint" or _is_protected_like(outcome):
        return "decrease"
    if weight < 0:
        return "decrease"
    return "increase"


def _component_from_payload(payload: Any, *, source: str = "explicit") -> Optional[UtilityComponent]:
    if isinstance(payload, UtilityComponent):
        return payload
    if not isinstance(payload, Mapping):
        return None
    outcome = _clean_str(payload.get("outcome") or payload.get("variable") or payload.get("name") or payload.get("goal"))
    if not outcome:
        return None
    weight = _safe_float(payload.get("weight"), 1.0)
    if weight is None:
        weight = 1.0
    role = _clean_str(payload.get("role"), "auxiliary")
    direction = _clean_str(payload.get("direction") or payload.get("desired_direction"), "increase")
    active = bool(payload.get("active", True))
    return UtilityComponent(
        outcome=outcome,
        weight=float(abs(weight)),
        direction=direction,
        role=role,
        source=source,
        active=active,
        metadata=_as_dict(payload.get("metadata")),
    )


def _explicit_components(query: FinalCauseQuery) -> List[UtilityComponent]:
    raw = _as_dict(query.raw)
    explicit = raw.get("utility_function") or raw.get("utility_components") or raw.get("utility_model")
    components: List[UtilityComponent] = []
    if isinstance(explicit, Mapping):
        if isinstance(explicit.get("components"), list):
            for item in explicit.get("components") or []:
                component = _component_from_payload(item)
                if component is not None:
                    components.append(component)
        else:
            for outcome, weight in explicit.items():
                if outcome == "components":
                    continue
                numeric = _safe_float(weight)
                if numeric is None:
                    continue
                role = "protected_constraint" if numeric < 0 or _is_protected_like(str(outcome)) else "auxiliary"
                components.append(
                    UtilityComponent(
                        outcome=str(outcome),
                        weight=abs(float(numeric)),
                        direction=_default_direction(str(outcome), role, float(numeric)),
                        role=role,
                        source="explicit_mapping",
                    )
                )
    elif isinstance(explicit, list):
        for item in explicit:
            component = _component_from_payload(item)
            if component is not None:
                components.append(component)
    return components


def _derive_components(query: FinalCauseQuery, goal: GoalSpec) -> List[UtilityComponent]:
    protected = [query.protected_outcome, *goal.protected_outcomes]
    side_effects = [g.goal_variable for g in query.side_effect_goals]
    side_effects.extend(goal.side_effect_outcomes)

    components: List[UtilityComponent] = [
        UtilityComponent(
            outcome=goal.goal_variable,
            weight=float(goal.utility_weight),
            direction=goal.desired_direction,
            role="primary_goal",
            source="candidate_goal",
            metadata=dict(goal.metadata),
        )
    ]

    seen_protected = set()
    for outcome in protected:
        outcome = _clean_str(outcome)
        if outcome and outcome not in seen_protected:
            seen_protected.add(outcome)
            components.append(
                UtilityComponent(
                    outcome=outcome,
                    weight=1.0,
                    direction="decrease",
                    role="protected_constraint",
                    source="protected_outcome",
                )
            )

    seen_side = set()
    for outcome in side_effects:
        outcome = _clean_str(outcome)
        if outcome and outcome not in seen_side and outcome != goal.goal_variable:
            seen_side.add(outcome)
            components.append(
                UtilityComponent(
                    outcome=outcome,
                    weight=0.0,
                    direction="increase",
                    role="side_effect",
                    source="side_effect_monitor",
                    active=False,
                )
            )

    excluded = {goal.goal_variable, *seen_protected, *seen_side}
    for outcome, weight in (query.agent.utility_model or {}).items():
        outcome = _clean_str(outcome)
        if not outcome or outcome in excluded:
            continue
        numeric = float(weight)
        role = _role_for_weight(outcome, numeric, seen_protected, seen_side, goal.goal_variable)
        components.append(
            UtilityComponent(
                outcome=outcome,
                weight=abs(numeric),
                direction=_default_direction(outcome, role, numeric),
                role=role,
                source="agent.utility_model",
            )
        )
    return components


def _merge_components(derived: List[UtilityComponent], explicit: List[UtilityComponent]) -> List[UtilityComponent]:
    """Merge derived and explicit components, with explicit entries overriding matching roles/outcomes."""

    merged: Dict[Tuple[str, str], UtilityComponent] = {}
    order: List[Tuple[str, str]] = []
    for component in [*derived, *explicit]:
        key = (component.role, component.outcome)
        if key not in merged:
            order.append(key)
        merged[key] = component
    return [merged[key] for key in order]


def _component_contribution(option: Mapping[str, Any], component: UtilityComponent) -> Tuple[float, float, str]:
    value, value_source = _outcome_value(option, component.outcome)
    utility = _directional_utility(value, component.direction, component.metadata)
    contribution = float(component.weight) * utility if component.active else 0.0
    return round(float(contribution), 6), round(float(utility), 6), value_source


def _evaluate_action(option: Mapping[str, Any], components: List[UtilityComponent]) -> ActionUtilityBreakdown:
    action = _action_name(option)
    primary = 0.0
    protected_penalty = 0.0
    side_effect = 0.0
    auxiliary = 0.0
    component_rows: List[Dict[str, Any]] = []
    violations: List[str] = []

    for component in components:
        contribution, normalized_utility, value_source = _component_contribution(option, component)
        row = component.to_dict()
        row.update(
            {
                "normalized_utility": normalized_utility,
                "contribution": contribution,
                "value_source": value_source,
            }
        )
        component_rows.append(row)
        if component.role == "primary_goal":
            primary += contribution
        elif component.role == "protected_constraint":
            # The contribution is positive when the protected outcome is low.
            # We convert the shortfall from the maximum into an explicit penalty.
            penalty = max(0.0, float(component.weight) - contribution)
            protected_penalty += penalty
            if penalty >= 0.5 * max(float(component.weight), 1e-9):
                violations.append(component.outcome)
        elif component.role == "side_effect":
            side_effect += contribution
        else:
            auxiliary += contribution

    risk_penalty = _risk_penalty(option)
    total = primary + auxiliary + side_effect - protected_penalty - risk_penalty
    return ActionUtilityBreakdown(
        action=action,
        total_utility=round(float(total), 6),
        primary_goal_utility=round(float(primary), 6),
        protected_penalty=round(float(protected_penalty), 6),
        side_effect_utility=round(float(side_effect), 6),
        auxiliary_utility=round(float(auxiliary), 6),
        risk_penalty=round(float(risk_penalty), 6),
        constraint_violations=violations,
        components=component_rows,
        raw_option=dict(option),
    )


def _rank(rows: List[ActionUtilityBreakdown]) -> List[Dict[str, Any]]:
    output = [row.to_dict() for row in rows]
    output.sort(key=lambda item: (item.get("total_utility", float("-inf")), item.get("action", "")), reverse=True)
    for idx, row in enumerate(output, start=1):
        row["rank"] = idx
    return output


def _margin(rankings: List[Mapping[str, Any]]) -> Optional[float]:
    if len(rankings) < 2:
        return None
    top = _safe_float(rankings[0].get("total_utility"))
    second = _safe_float(rankings[1].get("total_utility"))
    if top is None or second is None:
        return None
    return round(float(top - second), 6)


def _observed_rank(rankings: List[Mapping[str, Any]], observed_action: str) -> Optional[int]:
    for row in rankings:
        if row.get("action") == observed_action:
            value = row.get("rank")
            return int(value) if value is not None else None
    return None


def _tradeoff_detected(rankings: List[Mapping[str, Any]]) -> bool:
    if len(rankings) < 2:
        return False
    top = rankings[0]
    return any(
        float(row.get("primary_goal_utility", 0.0)) > float(top.get("primary_goal_utility", 0.0))
        and float(row.get("total_utility", 0.0)) < float(top.get("total_utility", 0.0))
        for row in rankings[1:]
    )


def _component_weight(components: Iterable[UtilityComponent], role: str) -> float:
    return round(float(sum(c.weight for c in components if c.role == role and c.active)), 6)


class UtilityFunctionEvaluator:
    """Evaluate whether an explicit utility function selects the observed action."""

    def components_for(self, query: FinalCauseQuery, goal: GoalSpec) -> List[UtilityComponent]:
        return _merge_components(_derive_components(query, goal), _explicit_components(query))

    def evaluate(self, query: FinalCauseQuery, goal: GoalSpec) -> UtilityFunctionAudit:
        if len(query.candidate_actions) < 2:
            return UtilityFunctionAudit(
                assessed=False,
                goal_variable=goal.goal_variable,
                observed_action=query.observed_action,
                reason="At least two candidate actions are required for utility-function evaluation.",
                reason_codes=["SFM_UTILITY_INSUFFICIENT_ACTION_ALTERNATIVES"],
                limits=["candidate_action_set_required"],
                raw=query.to_dict(),
            )

        components = self.components_for(query, goal)
        rankings = _rank([_evaluate_action(option, components) for option in query.candidate_actions])
        selected = str(rankings[0].get("action") or "") if rankings else ""
        observed = query.observed_action or selected
        matches = bool(selected and selected == observed)
        observed_rank = _observed_rank(rankings, observed)
        tradeoff = _tradeoff_detected(rankings)

        reason_codes: List[str] = ["SFM_UTILITY_FUNCTION_ASSESSED"]
        if matches:
            reason_codes.append("SFM_UTILITY_SUPPORTS_OBSERVED_ACTION")
        else:
            reason_codes.append("SFM_UTILITY_DOES_NOT_SELECT_OBSERVED_ACTION")
        if tradeoff:
            reason_codes.append("SFM_UTILITY_TRADEOFF_DETECTED")
        if any(c.role == "side_effect" for c in components):
            reason_codes.append("SFM_UTILITY_SIDE_EFFECTS_MONITORED_NOT_AUTOMATIC_GOALS")
        if any(c.role == "protected_constraint" for c in components):
            reason_codes.append("SFM_UTILITY_PROTECTED_CONSTRAINTS_INCLUDED")

        limits: List[str] = []
        if not _explicit_components(query) and not query.agent.utility_model:
            limits.append("explicit_agent_utility_function_not_supplied_derived_default_used")

        reason = (
            "Explicit utility audit ranked candidate actions from primary-goal utility, protected constraints, "
            "side-effect monitors, auxiliary preferences, and risk penalties."
        )
        return UtilityFunctionAudit(
            assessed=True,
            goal_variable=goal.goal_variable,
            observed_action=observed,
            selected_action=selected,
            selected_action_matches_observed=matches,
            observed_rank=observed_rank,
            top_margin=_margin(rankings),
            tradeoff_detected=tradeoff,
            primary_goal_weight=_component_weight(components, "primary_goal"),
            protected_constraint_weight=_component_weight(components, "protected_constraint"),
            side_effect_weight=_component_weight(components, "side_effect"),
            utility_components=[component.to_dict() for component in components],
            rankings=rankings,
            reason=reason,
            reason_codes=reason_codes,
            limits=limits,
            raw={"query": query.to_dict(), "goal": goal.to_dict()},
        )


def evaluate_utility_function(payload: Any, goal: GoalSpec | str | Mapping[str, Any] | None = None) -> Dict[str, Any]:
    """Convenience API for explicit utility-function diagnostics."""

    query = FinalCauseQuery.from_payload(payload)
    goal_obj = GoalSpec.from_payload(goal) if goal is not None else query.candidate_goals[0]
    return UtilityFunctionEvaluator().evaluate(query, goal_obj).to_dict()
