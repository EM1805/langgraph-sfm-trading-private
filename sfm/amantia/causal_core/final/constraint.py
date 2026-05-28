from __future__ import annotations

"""Constraint-aware diagnostics for Structural Final Models.

A teleological explanation should not confuse what an agent *optimizes* with
what the agent must *respect*.  This module separates final goals from hard
constraints, soft constraints, protected outcomes, and monitored side effects.

The evaluator is deliberately conservative: it does not prove intent.  It asks
whether the observed action is the action selected by a policy that maximizes
candidate goals only after enforcing declared constraints.
"""

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .schema import FinalCauseQuery, GoalSpec
from .protection import normalize_protection_policy
from .utility import _directional_utility, _is_protected_like, _outcome_value, _risk_penalty


@dataclass
class SFMConstraintSpec:
    """One SFM constraint or monitored side effect.

    constraint_type is intentionally small:
    - hard: violation makes an action infeasible;
    - protected: hard-like safety/protected outcome;
    - soft: violation creates a finite penalty;
    - side_effect: monitored but not optimized by default.
    """

    outcome: str = ""
    constraint_type: str = "soft"
    direction: str = "decrease"
    threshold: Optional[float] = None
    weight: float = 1.0
    tolerance: float = 0.05
    active: bool = True
    source: str = "derived"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConstraintEvaluation:
    """Evaluation of one constraint for one candidate action."""

    outcome: str = ""
    constraint_type: str = "soft"
    direction: str = "decrease"
    threshold: Optional[float] = None
    value: Optional[float] = None
    satisfied: bool = True
    violation_amount: float = 0.0
    penalty: float = 0.0
    blocks_action: bool = False
    value_source: str = ""
    source: str = "derived"
    active: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConstraintActionAssessment:
    """Constraint-aware score decomposition for one action."""

    action: str = ""
    feasible: bool = True
    total_score: float = 0.0
    goal_score: float = 0.0
    auxiliary_score: float = 0.0
    soft_penalty: float = 0.0
    hard_penalty: float = 0.0
    risk_penalty: float = 0.0
    hard_violation_count: int = 0
    violated_constraints: List[str] = field(default_factory=list)
    monitored_side_effects: List[str] = field(default_factory=list)
    constraint_evaluations: List[Dict[str, Any]] = field(default_factory=list)
    raw_option: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConstraintAwareAudit:
    """Aggregate constraint-aware SFM audit."""

    assessed: bool = False
    observed_action: str = ""
    selected_action: str = ""
    selected_action_matches_observed: bool = False
    observed_feasible: bool = False
    observed_rank: Optional[int] = None
    support_strength: float = 0.0
    final_goals: List[str] = field(default_factory=list)
    hard_constraints: List[str] = field(default_factory=list)
    soft_constraints: List[str] = field(default_factory=list)
    protected_constraints: List[str] = field(default_factory=list)
    side_effect_outcomes: List[str] = field(default_factory=list)
    constraint_like_candidate_goals: List[str] = field(default_factory=list)
    feasible_actions: List[str] = field(default_factory=list)
    infeasible_actions: List[str] = field(default_factory=list)
    tradeoff_under_constraints: bool = False
    rankings: List[Dict[str, Any]] = field(default_factory=list)
    normalized_protection_policy: Dict[str, Any] = field(default_factory=dict)
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


def _constraint_outcome(payload: Mapping[str, Any]) -> str:
    return _clean_str(
        payload.get("outcome")
        or payload.get("variable")
        or payload.get("name")
        or payload.get("goal")
        or payload.get("constraint")
    )


def _default_constraint_type(outcome: str, explicit_type: str = "") -> str:
    ctype = _clean_lower(explicit_type)
    if ctype in {"hard", "soft", "protected", "protected_constraint", "side_effect", "monitor"}:
        return "protected" if ctype == "protected_constraint" else "side_effect" if ctype == "monitor" else ctype
    if _is_protected_like(outcome):
        return "protected"
    return "soft"


def _default_direction(outcome: str, constraint_type: str, weight: float) -> str:
    if constraint_type in {"hard", "protected"} or weight < 0 or _is_protected_like(outcome):
        return "decrease"
    return "increase"


def _threshold_from_payload(payload: Mapping[str, Any], direction: str, constraint_type: str) -> Optional[float]:
    for key in ["threshold", "max_value", "max", "limit", "min_value", "min", "target"]:
        if key in payload and payload.get(key) not in (None, ""):
            return _clip01(_safe_float(payload.get(key), None), 0.0)
    # Derived side-effect monitors should not become penalties unless the user
    # supplied a threshold explicitly.
    if constraint_type == "side_effect":
        return None
    if direction in {"decrease", "avoid", "reduce", "minimize", "minimise", "lower"}:
        return 0.10 if constraint_type in {"hard", "protected"} else 0.25
    if direction in {"maintain", "stabilize", "stabilise", "keep"}:
        return 0.50
    return 0.60


def _constraint_from_payload(payload: Any, *, constraint_type: str = "soft", source: str = "explicit") -> Optional[SFMConstraintSpec]:
    if isinstance(payload, SFMConstraintSpec):
        return payload
    if isinstance(payload, str):
        outcome = _clean_str(payload)
        if not outcome:
            return None
        ctype = _default_constraint_type(outcome, constraint_type)
        direction = _default_direction(outcome, ctype, 1.0)
        return SFMConstraintSpec(
            outcome=outcome,
            constraint_type=ctype,
            direction=direction,
            threshold=_threshold_from_payload({}, direction, ctype),
            source=source,
        )
    if not isinstance(payload, Mapping):
        return None
    outcome = _constraint_outcome(payload)
    if not outcome:
        return None
    weight = _safe_float(payload.get("weight"), 1.0)
    if weight is None:
        weight = 1.0
    ctype = _default_constraint_type(outcome, _clean_str(payload.get("constraint_type") or payload.get("type") or payload.get("role") or constraint_type))
    direction = _clean_lower(payload.get("direction") or payload.get("desired_direction"), _default_direction(outcome, ctype, float(weight)))
    return SFMConstraintSpec(
        outcome=outcome,
        constraint_type=ctype,
        direction=direction,
        threshold=_threshold_from_payload(payload, direction, ctype),
        weight=abs(float(weight)),
        tolerance=float(_safe_float(payload.get("tolerance"), 0.05) or 0.05),
        active=bool(payload.get("active", True)),
        source=source,
        metadata=_as_dict(payload.get("metadata")),
    )


def _constraints_from_mapping(mapping: Mapping[str, Any], *, constraint_type: str, source: str) -> List[SFMConstraintSpec]:
    out: List[SFMConstraintSpec] = []
    for key, value in mapping.items():
        if key in {"hard", "soft", "protected", "side_effects", "side_effect", "components"}:
            continue
        if isinstance(value, Mapping):
            payload = {"outcome": key, **dict(value)}
        else:
            payload = {"outcome": key, "threshold": value}
        spec = _constraint_from_payload(payload, constraint_type=constraint_type, source=source)
        if spec is not None:
            out.append(spec)
    return out


def _collect_explicit_constraints(query: FinalCauseQuery) -> List[SFMConstraintSpec]:
    raw = _as_dict(query.raw)
    out: List[SFMConstraintSpec] = []

    model = raw.get("constraint_model") or raw.get("constraints") or raw.get("constraint_function")
    if isinstance(model, Mapping):
        for key, ctype in [
            ("hard", "hard"),
            ("hard_constraints", "hard"),
            ("soft", "soft"),
            ("soft_constraints", "soft"),
            ("protected", "protected"),
            ("protected_constraints", "protected"),
            ("side_effects", "side_effect"),
            ("side_effect", "side_effect"),
            ("side_effect_constraints", "side_effect"),
        ]:
            value = model.get(key)
            if isinstance(value, Mapping):
                out.extend(_constraints_from_mapping(value, constraint_type=ctype, source=f"constraint_model.{key}"))
            else:
                for item in _as_list(value):
                    spec = _constraint_from_payload(item, constraint_type=ctype, source=f"constraint_model.{key}")
                    if spec is not None:
                        out.append(spec)
        if isinstance(model.get("components"), list):
            for item in model.get("components") or []:
                spec = _constraint_from_payload(item, constraint_type="soft", source="constraint_model.components")
                if spec is not None:
                    out.append(spec)
        # If the mapping itself is outcome -> threshold, parse it as soft/default.
        out.extend(_constraints_from_mapping(model, constraint_type="soft", source="constraints.mapping"))
    elif isinstance(model, list):
        for item in model:
            spec = _constraint_from_payload(item, constraint_type="soft", source="constraints.list")
            if spec is not None:
                out.append(spec)

    top_level_groups = [
        ("hard_constraints", "hard"),
        ("soft_constraints", "soft"),
        ("protected_constraints", "protected"),
        ("side_effect_constraints", "side_effect"),
    ]
    for key, ctype in top_level_groups:
        value = raw.get(key)
        if isinstance(value, Mapping):
            out.extend(_constraints_from_mapping(value, constraint_type=ctype, source=key))
        else:
            for item in _as_list(value):
                spec = _constraint_from_payload(item, constraint_type=ctype, source=key)
                if spec is not None:
                    out.append(spec)
    return out


def _derived_constraints(query: FinalCauseQuery) -> List[SFMConstraintSpec]:
    out: List[SFMConstraintSpec] = []
    if query.protected_outcome:
        out.append(
            SFMConstraintSpec(
                outcome=query.protected_outcome,
                constraint_type="protected",
                direction="decrease",
                threshold=0.10,
                weight=1.0,
                source="query.protected_outcome",
            )
        )
    for goal in query.candidate_goals:
        for outcome in goal.protected_outcomes:
            if outcome:
                out.append(
                    SFMConstraintSpec(
                        outcome=outcome,
                        constraint_type="protected",
                        direction="decrease",
                        threshold=0.10,
                        weight=1.0,
                        source=f"candidate_goal.{goal.goal_variable}.protected_outcomes",
                    )
                )
        for outcome in goal.side_effect_outcomes:
            if outcome:
                out.append(
                    SFMConstraintSpec(
                        outcome=outcome,
                        constraint_type="side_effect",
                        direction="increase",
                        threshold=None,
                        weight=0.0,
                        active=True,
                        source=f"candidate_goal.{goal.goal_variable}.side_effect_outcomes",
                    )
                )
    for goal in query.side_effect_goals:
        if goal.goal_variable:
            out.append(
                SFMConstraintSpec(
                    outcome=goal.goal_variable,
                    constraint_type="side_effect",
                    direction=goal.desired_direction,
                    threshold=None,
                    weight=0.0,
                    active=True,
                    source="query.side_effect_goals",
                )
            )
    return out


def _merge_constraints(specs: Sequence[SFMConstraintSpec]) -> List[SFMConstraintSpec]:
    priority = {"side_effect": 0, "soft": 1, "protected": 2, "hard": 3}
    merged: Dict[Tuple[str, str], SFMConstraintSpec] = {}
    outcome_best_type: Dict[str, str] = {}
    order: List[Tuple[str, str]] = []
    for spec in specs:
        outcome = _clean_str(spec.outcome)
        if not outcome:
            continue
        ctype = _clean_lower(spec.constraint_type, "soft")
        spec.constraint_type = ctype
        key = (outcome, ctype)
        if key not in merged:
            order.append(key)
        merged[key] = spec
        if priority.get(ctype, 0) >= priority.get(outcome_best_type.get(outcome, ""), -1):
            outcome_best_type[outcome] = ctype
    # Drop weaker duplicate roles for the same outcome, except side-effect monitors
    # can coexist for reporting.
    output: List[SFMConstraintSpec] = []
    for key in order:
        spec = merged[key]
        strongest = outcome_best_type.get(spec.outcome)
        if spec.constraint_type == strongest or spec.constraint_type == "side_effect":
            output.append(spec)
    return output


def _constraints(query: FinalCauseQuery) -> List[SFMConstraintSpec]:
    return _merge_constraints([*_derived_constraints(query), *_collect_explicit_constraints(query)])


def _constraint_sets(specs: Sequence[SFMConstraintSpec]) -> Tuple[List[str], List[str], List[str], List[str]]:
    hard = _unique(spec.outcome for spec in specs if spec.constraint_type == "hard")
    soft = _unique(spec.outcome for spec in specs if spec.constraint_type == "soft")
    protected = _unique(spec.outcome for spec in specs if spec.constraint_type == "protected" or (spec.constraint_type == "hard" and _is_protected_like(spec.outcome)))
    side = _unique(spec.outcome for spec in specs if spec.constraint_type == "side_effect")
    return hard, soft, protected, side


def _final_goals(query: FinalCauseQuery, constraint_outcomes: Iterable[str]) -> List[str]:
    blocked = set(constraint_outcomes)
    return [goal.goal_variable for goal in query.candidate_goals if goal.goal_variable and goal.goal_variable not in blocked]


def _constraint_like_goals(query: FinalCauseQuery, constraint_outcomes: Iterable[str]) -> List[str]:
    blocked = set(constraint_outcomes)
    return [goal.goal_variable for goal in query.candidate_goals if goal.goal_variable in blocked]


def _violation(value: float, spec: SFMConstraintSpec) -> Tuple[bool, float]:
    direction = _clean_lower(spec.direction, "decrease")
    threshold = spec.threshold
    if threshold is None:
        return True, 0.0
    threshold = _clip01(threshold)
    value = _clip01(value)
    if direction in {"decrease", "avoid", "reduce", "minimize", "minimise", "lower"}:
        violation = max(0.0, value - threshold)
    elif direction in {"maintain", "stabilize", "stabilise", "keep"}:
        violation = max(0.0, abs(value - threshold) - abs(float(spec.tolerance)))
    else:
        violation = max(0.0, threshold - value)
    return violation <= 1e-9, round(float(violation), 6)


def _evaluate_constraint(option: Mapping[str, Any], spec: SFMConstraintSpec) -> ConstraintEvaluation:
    default = 0.0 if spec.constraint_type in {"protected", "hard"} or _is_protected_like(spec.outcome) else 0.5
    value, source = _outcome_value(option, spec.outcome, default=default)
    satisfied, violation = _violation(value, spec)
    blocks = bool(spec.active and spec.constraint_type in {"hard", "protected"} and not satisfied)
    penalty = 0.0
    if spec.active and spec.constraint_type == "soft":
        penalty = abs(float(spec.weight)) * violation
    elif spec.active and spec.constraint_type == "side_effect" and spec.threshold is not None:
        # Explicitly thresholded side effects are still not goals, but can add a
        # small diagnostic penalty when they exceed a user-supplied bound.
        penalty = abs(float(spec.weight)) * violation
    return ConstraintEvaluation(
        outcome=spec.outcome,
        constraint_type=spec.constraint_type,
        direction=spec.direction,
        threshold=spec.threshold,
        value=round(float(value), 6),
        satisfied=satisfied,
        violation_amount=violation,
        penalty=round(float(penalty), 6),
        blocks_action=blocks,
        value_source=source,
        source=spec.source,
        active=spec.active,
    )


def _goal_score(option: Mapping[str, Any], goals: Sequence[GoalSpec], excluded: Iterable[str]) -> float:
    excluded_set = set(excluded)
    score = 0.0
    for goal in goals:
        if goal.goal_variable in excluded_set:
            continue
        value, _source = _outcome_value(option, goal.goal_variable)
        score += float(goal.utility_weight) * _directional_utility(value, goal.desired_direction, goal.metadata)
    return score


def _auxiliary_score(option: Mapping[str, Any], query: FinalCauseQuery, excluded: Iterable[str]) -> float:
    excluded_set = set(excluded)
    score = 0.0
    for outcome, weight in (query.agent.utility_model or {}).items():
        outcome = _clean_str(outcome)
        numeric = _safe_float(weight)
        if not outcome or numeric is None or outcome in excluded_set:
            continue
        direction = "decrease" if numeric < 0 or _is_protected_like(outcome) else "increase"
        value, _source = _outcome_value(option, outcome)
        score += abs(float(numeric)) * _directional_utility(value, direction, {})
    return score


def _evaluate_action(option: Mapping[str, Any], *, query: FinalCauseQuery, constraints: Sequence[SFMConstraintSpec]) -> ConstraintActionAssessment:
    action = _action_name(option)
    evals = [_evaluate_constraint(option, spec) for spec in constraints if spec.active]
    hard_violations = [row.outcome for row in evals if row.blocks_action]
    monitored_side_effects = [row.outcome for row in evals if row.constraint_type == "side_effect"]
    constraint_outcomes = [spec.outcome for spec in constraints]
    goal = _goal_score(option, query.candidate_goals, constraint_outcomes)
    auxiliary = _auxiliary_score(option, query, [*constraint_outcomes, *(g.goal_variable for g in query.candidate_goals)])
    soft_penalty = sum(row.penalty for row in evals if row.constraint_type in {"soft", "side_effect"})
    hard_penalty = 1000.0 * len(hard_violations)
    risk_penalty = _risk_penalty(option)
    total = goal + auxiliary - soft_penalty - hard_penalty - risk_penalty
    return ConstraintActionAssessment(
        action=action,
        feasible=not hard_violations,
        total_score=round(float(total), 6),
        goal_score=round(float(goal), 6),
        auxiliary_score=round(float(auxiliary), 6),
        soft_penalty=round(float(soft_penalty), 6),
        hard_penalty=round(float(hard_penalty), 6),
        risk_penalty=round(float(risk_penalty), 6),
        hard_violation_count=len(hard_violations),
        violated_constraints=hard_violations,
        monitored_side_effects=monitored_side_effects,
        constraint_evaluations=[row.to_dict() for row in evals],
        raw_option=dict(option),
    )


def _rank(rows: Sequence[ConstraintActionAssessment]) -> List[Dict[str, Any]]:
    output = [row.to_dict() for row in rows]
    output.sort(
        key=lambda item: (
            bool(item.get("feasible")),
            -int(item.get("hard_violation_count", 9999)),
            float(item.get("total_score", float("-inf"))),
            str(item.get("action", "")),
        ),
        reverse=True,
    )
    for idx, row in enumerate(output, start=1):
        row["rank"] = idx
    return output


def _observed_rank(rankings: Sequence[Mapping[str, Any]], observed_action: str) -> Optional[int]:
    for row in rankings:
        if row.get("action") == observed_action:
            rank = row.get("rank")
            return int(rank) if rank is not None else None
    return None


def _margin(rankings: Sequence[Mapping[str, Any]]) -> Optional[float]:
    feasible = [row for row in rankings if row.get("feasible")]
    if len(feasible) < 2:
        return None
    top = _safe_float(feasible[0].get("total_score"))
    second = _safe_float(feasible[1].get("total_score"))
    if top is None or second is None:
        return None
    return round(float(top - second), 6)


def _support_strength(matches: bool, observed_feasible: bool, observed_rank: Optional[int], margin: Optional[float]) -> float:
    if not observed_feasible:
        return 0.0
    if matches:
        base = 0.78
        if margin is not None:
            base += min(max(float(margin), 0.0), 0.20)
        return round(min(base, 0.95), 6)
    if observed_rank is not None:
        return round(max(0.05, 0.45 - 0.10 * max(observed_rank - 1, 0)), 6)
    return 0.0


class ConstraintAwareEvaluator:
    """Evaluate whether a constrained telic policy selects the observed action."""

    def evaluate(self, payload: Any) -> ConstraintAwareAudit:
        query = FinalCauseQuery.from_payload(payload)
        constraints = _constraints(query)
        protection_policy = normalize_protection_policy(query)
        hard, soft, protected, side = _constraint_sets(constraints)
        constraint_outcomes = _unique([*hard, *soft, *protected, *side])
        final_goals = _final_goals(query, constraint_outcomes)
        constraint_like_goals = _constraint_like_goals(query, constraint_outcomes)

        if len(query.candidate_actions) < 2:
            return ConstraintAwareAudit(
                assessed=False,
                observed_action=query.observed_action,
                final_goals=final_goals,
                hard_constraints=hard,
                soft_constraints=soft,
                protected_constraints=protected,
                side_effect_outcomes=side,
                constraint_like_candidate_goals=constraint_like_goals,
                normalized_protection_policy=protection_policy.to_dict(),
                reason="At least two candidate actions are required for constraint-aware SFM evaluation.",
                reason_codes=["SFM_CONSTRAINT_AWARE_INSUFFICIENT_ACTION_ALTERNATIVES"],
                limits=["candidate_action_set_required"],
                raw=query.to_dict(),
            )
        if not constraints:
            return ConstraintAwareAudit(
                assessed=False,
                observed_action=query.observed_action,
                final_goals=final_goals,
                normalized_protection_policy=protection_policy.to_dict(),
                reason="No constraints or protected outcomes were supplied for constraint-aware SFM evaluation.",
                reason_codes=["SFM_CONSTRAINT_AWARE_NO_CONSTRAINTS"],
                limits=["constraint_set_required"],
                raw=query.to_dict(),
            )

        rankings = _rank([_evaluate_action(option, query=query, constraints=constraints) for option in query.candidate_actions])
        selected = str(rankings[0].get("action") or "") if rankings else ""
        observed = query.observed_action or selected
        observed_row = next((row for row in rankings if row.get("action") == observed), {})
        observed_feasible = bool(observed_row.get("feasible"))
        matches = bool(selected and selected == observed)
        observed_rank = _observed_rank(rankings, observed)
        margin = _margin(rankings)
        feasible_actions = [str(row.get("action")) for row in rankings if row.get("feasible")]
        infeasible_actions = [str(row.get("action")) for row in rankings if not row.get("feasible")]
        unconstrained_best = ""
        unconstrained_rows = sorted(
            rankings,
            key=lambda row: (float(row.get("goal_score", 0.0)) + float(row.get("auxiliary_score", 0.0)), str(row.get("action", ""))),
            reverse=True,
        )
        if unconstrained_rows:
            unconstrained_best = str(unconstrained_rows[0].get("action") or "")
        tradeoff = bool(unconstrained_best and selected and unconstrained_best != selected)

        reason_codes: List[str] = ["SFM_CONSTRAINT_AWARE_ASSESSED"]
        if matches:
            reason_codes.append("SFM_CONSTRAINT_AWARE_POLICY_SUPPORTS_OBSERVED_ACTION")
        else:
            reason_codes.append("SFM_CONSTRAINT_AWARE_POLICY_DOES_NOT_SELECT_OBSERVED_ACTION")
        if observed_feasible:
            reason_codes.append("SFM_OBSERVED_ACTION_SATISFIES_HARD_CONSTRAINTS")
        else:
            reason_codes.append("SFM_OBSERVED_ACTION_VIOLATES_HARD_CONSTRAINTS")
        if tradeoff:
            reason_codes.append("SFM_CONSTRAINTS_CHANGE_UNCONSTRAINED_ACTION_CHOICE")
        if protected:
            reason_codes.append("SFM_PROTECTED_OUTCOMES_TREATED_AS_CONSTRAINTS")
        if constraint_like_goals:
            reason_codes.append("SFM_CANDIDATE_GOAL_OVERLAPS_CONSTRAINT")
        if side:
            reason_codes.append("SFM_SIDE_EFFECTS_MONITORED_NOT_OPTIMIZED")
        reason_codes.extend(protection_policy.reason_codes)

        limits = ["diagnostic_only_not_full_constraint_aware_sfm_identification"]
        if constraint_like_goals:
            limits.append("candidate_goal_overlaps_constraint_or_side_effect")
        if not feasible_actions:
            limits.append("no_candidate_action_satisfies_hard_constraints")

        return ConstraintAwareAudit(
            assessed=True,
            observed_action=observed,
            selected_action=selected,
            selected_action_matches_observed=matches,
            observed_feasible=observed_feasible,
            observed_rank=observed_rank,
            support_strength=_support_strength(matches, observed_feasible, observed_rank, margin),
            final_goals=final_goals,
            hard_constraints=hard,
            soft_constraints=soft,
            protected_constraints=protected,
            side_effect_outcomes=side,
            constraint_like_candidate_goals=constraint_like_goals,
            feasible_actions=feasible_actions,
            infeasible_actions=infeasible_actions,
            tradeoff_under_constraints=tradeoff,
            rankings=rankings,
            normalized_protection_policy=protection_policy.to_dict(),
            reason=(
                "Constraint-aware SFM audit ranked candidate actions by final-goal utility after enforcing hard/protected "
                "constraints, applying soft penalties, and monitoring side effects."
            ),
            reason_codes=reason_codes,
            limits=limits,
            raw={"query": query.to_dict(), "constraints": [spec.to_dict() for spec in constraints]},
        )


def evaluate_constraint_aware_sfm(payload: Any) -> Dict[str, Any]:
    """Convenience API for constraint-aware SFM diagnostics."""

    return ConstraintAwareEvaluator().evaluate(payload).to_dict()
