from __future__ import annotations

"""Empirical utility learning for Structural Final Model development.

This module connects the SFM diagnostic layer to Amantia's outcome tracker.  It
estimates a small, conservative *implicit utility* signal from observed
``action -> outcome`` records.  The output is not causal identification.  It is
an empirical consistency check: does the observed action match the action that
has historically produced the candidate goal, after basic safety penalties?
"""

from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .schema import FinalCauseQuery, GoalSpec


@dataclass
class EmpiricalActionEvidence:
    """Observed outcome summary for one action."""

    action: str = ""
    n: int = 0
    goal_observed_n: int = 0
    goal_mean: Optional[float] = None
    goal_utility: Optional[float] = None
    harm_observed_n: int = 0
    harm_rate: Optional[float] = None
    success_observed_n: int = 0
    success_rate: Optional[float] = None
    satisfaction_n: int = 0
    avg_user_satisfaction: Optional[float] = None
    latency_n: int = 0
    avg_latency_ms: Optional[float] = None
    empirical_utility: Optional[float] = None
    rank: Optional[int] = None
    evidence_status: str = "unassessed"
    reason_codes: List[str] = field(default_factory=list)
    raw_records: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EmpiricalUtilityAudit:
    """Aggregate implicit-utility audit for an SFM final-cause query."""

    assessed: bool = False
    goal_variable: str = ""
    observed_action: str = ""
    selected_action: str = ""
    selected_action_matches_observed: bool = False
    observed_rank: Optional[int] = None
    support_strength: float = 0.0
    total_records: int = 0
    usable_records: int = 0
    min_records_per_action: int = 2
    action_evidence: List[Dict[str, Any]] = field(default_factory=list)
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
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _clip01(value: Optional[float], default: float = 0.5) -> float:
    if value is None:
        value = default
    return max(0.0, min(1.0, float(value)))


def _bool_value(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on", "success", "passed"}:
        return True
    if text in {"0", "false", "no", "n", "off", "failure", "failed"}:
        return False
    return None


def _action_name(value: Any) -> str:
    if isinstance(value, Mapping):
        return _clean_str(
            value.get("action")
            or value.get("action_name")
            or value.get("candidate_action")
            or value.get("selected_action")
            or value.get("name")
        )
    return _clean_str(value)


def _candidate_action_names(query: FinalCauseQuery) -> List[str]:
    names: List[str] = []
    for option in query.candidate_actions:
        name = _action_name(option)
        if name and name not in names:
            names.append(name)
    if query.observed_action and query.observed_action not in names:
        names.insert(0, query.observed_action)
    return names


def _nested_value(record: Mapping[str, Any], keys: Sequence[str], outcome: str) -> Optional[Any]:
    for key in keys:
        container = _as_dict(record.get(key))
        if outcome in container:
            return container.get(outcome)
    metadata = _as_dict(record.get("metadata"))
    for key in keys:
        container = _as_dict(metadata.get(key))
        if outcome in container:
            return container.get(outcome)
    outcome_metadata = _as_dict(metadata.get("outcome_metadata"))
    for key in keys:
        container = _as_dict(outcome_metadata.get(key))
        if outcome in container:
            return container.get(outcome)
    return None


def _normalize_satisfaction(value: Any) -> Optional[float]:
    number = _safe_float(value)
    if number is None:
        return None
    if 0.0 <= number <= 1.0:
        return number
    if 0.0 <= number <= 5.0:
        return number / 5.0
    if 0.0 <= number <= 10.0:
        return number / 10.0
    return _clip01(number)


def _normalize_latency_ms(value: Any) -> Optional[float]:
    number = _safe_float(value)
    if number is None:
        return None
    # Generic bounded latency proxy.  Callers can pass a direct normalized
    # latency outcome when they need domain-specific calibration.
    return _clip01(number / 10000.0, 0.5)


def _record_goal_value(record: Mapping[str, Any], goal: GoalSpec) -> Tuple[Optional[float], str]:
    outcome = _clean_str(goal.goal_variable)
    if not outcome:
        return None, "missing_goal"

    if outcome in record:
        value = _safe_float(record.get(outcome))
        if value is not None:
            return _clip01(value), f"direct:{outcome}"

    nested = _nested_value(
        record,
        ["observed_outcomes", "outcomes", "metrics", "outcome_scores", "goal_scores"],
        outcome,
    )
    value = _safe_float(nested)
    if value is not None:
        return _clip01(value), f"nested:{outcome}"

    outcome_l = outcome.lower()
    success = _bool_value(record.get("success"))
    harm = _bool_value(record.get("harm"))

    if outcome_l in {"task_success", "success", "resolved", "helpful", "target_success"}:
        if success is not None:
            return (1.0 if success else 0.0), "record.success"
        label = _clean_lower(record.get("outcome"), "unknown")
        if label in {"task_success", "resolved", "success", "helpful", "completed"}:
            return 1.0, "record.outcome_label"
        if label in {"task_failure", "failed", "failure", "not_resolved", "unhelpful", "abandoned"}:
            return 0.0, "record.outcome_label"

    if outcome_l in {"user_or_system_harm", "harm", "unsafe_outcome", "target_harm"} or "harm" in outcome_l:
        if harm is not None:
            return (1.0 if harm else 0.0), "record.harm"
        label = _clean_lower(record.get("outcome"), "unknown")
        if label in {"harm_event", "unsafe_outcome", "data_loss", "wrong_action", "policy_violation"}:
            return 1.0, "record.outcome_label"
        if success is not None:
            return 0.0, "record.success_inverse_harm_proxy"

    if outcome_l in {"user_satisfaction", "satisfaction", "avg_user_satisfaction"}:
        sat = _normalize_satisfaction(record.get("user_satisfaction"))
        if sat is not None:
            return sat, "record.user_satisfaction"

    if outcome_l in {"latency", "latency_ms", "response_latency"}:
        latency = _normalize_latency_ms(record.get("latency_ms"))
        if latency is not None:
            return latency, "record.latency_ms"

    return None, "goal_unobserved"


def _directional_utility(value: float, goal: GoalSpec) -> float:
    direction = (goal.desired_direction or "increase").strip().lower()
    value = _clip01(value)
    if direction in {"decrease", "minimize", "minimise", "reduce", "lower", "avoid"}:
        return 1.0 - value
    if direction in {"maintain", "stabilize", "stabilise", "keep"}:
        target = _safe_float(goal.metadata.get("target"), 0.5) if isinstance(goal.metadata, Mapping) else 0.5
        target = _clip01(target, 0.5)
        return _clip01(1.0 - abs(value - target))
    return value


def _mean(values: Iterable[float]) -> Optional[float]:
    vals = [float(v) for v in values]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _empirical_records_from_payload(query: FinalCauseQuery) -> List[Dict[str, Any]]:
    raw = _as_dict(query.raw)
    supplied = (
        raw.get("empirical_outcome_records")
        or raw.get("outcome_records")
        or raw.get("learning_records")
        or query.outcome_records
    )
    records = [dict(item) for item in _as_list(supplied) if isinstance(item, Mapping)]
    if records:
        return records

    path = _clean_str(raw.get("outcome_log_path") or raw.get("learning_log_path") or query.outcome_log_path)
    if not path:
        return []
    try:
        from amantia.learning.outcome_tracker import OutcomeTracker

        return [record.to_dict() for record in OutcomeTracker(Path(path)).build_records()]
    except Exception:
        return []


def _summarize_action(action: str, records: List[Dict[str, Any]], goal: GoalSpec) -> EmpiricalActionEvidence:
    goal_values: List[float] = []
    success_values: List[float] = []
    harm_values: List[float] = []
    satisfaction_values: List[float] = []
    latency_values: List[float] = []
    reason_codes: List[str] = []

    for record in records:
        value, _source = _record_goal_value(record, goal)
        if value is not None:
            goal_values.append(value)
        success = _bool_value(record.get("success"))
        if success is not None:
            success_values.append(1.0 if success else 0.0)
        harm = _bool_value(record.get("harm"))
        if harm is not None:
            harm_values.append(1.0 if harm else 0.0)
        sat = _normalize_satisfaction(record.get("user_satisfaction"))
        if sat is not None:
            satisfaction_values.append(sat)
        latency = _safe_float(record.get("latency_ms"))
        if latency is not None:
            latency_values.append(latency)

    goal_mean = _mean(goal_values)
    goal_utility = _directional_utility(goal_mean, goal) if goal_mean is not None else None
    harm_rate = _mean(harm_values)
    success_rate = _mean(success_values)
    avg_satisfaction = _mean(satisfaction_values)
    avg_latency_ms = _mean(latency_values)

    empirical_utility: Optional[float] = None
    if goal_utility is not None:
        empirical_utility = float(goal.utility_weight) * float(goal_utility)
        if harm_rate is not None:
            empirical_utility -= 0.35 * float(harm_rate)
        # Satisfaction is treated as weak auxiliary evidence, not as a goal by default.
        if avg_satisfaction is not None and goal.goal_variable.lower() not in {"user_satisfaction", "satisfaction"}:
            empirical_utility += 0.10 * float(avg_satisfaction)
        empirical_utility = round(float(empirical_utility), 6)

    if goal_values:
        reason_codes.append("SFM_EMPIRICAL_GOAL_OBSERVED_FOR_ACTION")
    else:
        reason_codes.append("SFM_EMPIRICAL_GOAL_NOT_OBSERVED_FOR_ACTION")

    return EmpiricalActionEvidence(
        action=action,
        n=len(records),
        goal_observed_n=len(goal_values),
        goal_mean=round(float(goal_mean), 6) if goal_mean is not None else None,
        goal_utility=round(float(goal_utility), 6) if goal_utility is not None else None,
        harm_observed_n=len(harm_values),
        harm_rate=round(float(harm_rate), 6) if harm_rate is not None else None,
        success_observed_n=len(success_values),
        success_rate=round(float(success_rate), 6) if success_rate is not None else None,
        satisfaction_n=len(satisfaction_values),
        avg_user_satisfaction=round(float(avg_satisfaction), 6) if avg_satisfaction is not None else None,
        latency_n=len(latency_values),
        avg_latency_ms=round(float(avg_latency_ms), 6) if avg_latency_ms is not None else None,
        empirical_utility=empirical_utility,
        evidence_status="usable" if empirical_utility is not None else "goal_unobserved",
        reason_codes=reason_codes,
        raw_records=records,
    )


def _min_records(query: FinalCauseQuery) -> int:
    raw = _as_dict(query.raw)
    value = raw.get("min_empirical_records_per_action", raw.get("min_empirical_records", query.min_empirical_records_per_action))
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return 2


class EmpiricalUtilityLearner:
    """Infer weak implicit-utility evidence from tracked outcomes."""

    def evaluate(self, query: FinalCauseQuery, goal: GoalSpec) -> EmpiricalUtilityAudit:
        records = _empirical_records_from_payload(query)
        min_records = _min_records(query)
        action_names = _candidate_action_names(query)
        observed = query.observed_action or (action_names[0] if action_names else "")

        if not records:
            return EmpiricalUtilityAudit(
                assessed=False,
                goal_variable=goal.goal_variable,
                observed_action=observed,
                min_records_per_action=min_records,
                reason="No outcome records or outcome log path were supplied for empirical utility learning.",
                reason_codes=["SFM_EMPIRICAL_UTILITY_NO_OUTCOME_RECORDS"],
                limits=["outcome_records_or_outcome_log_path_required"],
                raw={"query": query.to_dict(), "goal": goal.to_dict()},
            )

        by_action: Dict[str, List[Dict[str, Any]]] = {name: [] for name in action_names if name}
        for record in records:
            action = _clean_str(record.get("selected_action") or record.get("action") or record.get("candidate_action"))
            if not action:
                continue
            if action_names and action not in by_action:
                # Preserve records for candidate actions only; this keeps the
                # audit aligned with the counterfactual choice set.
                continue
            by_action.setdefault(action, []).append(dict(record))

        evidence = [_summarize_action(action, rows, goal) for action, rows in by_action.items()]
        usable = [row for row in evidence if row.empirical_utility is not None and row.n >= min_records]
        rankings = sorted(usable, key=lambda row: (row.empirical_utility if row.empirical_utility is not None else -999, row.action), reverse=True)
        for idx, row in enumerate(rankings, start=1):
            row.rank = idx

        total_records = len(records)
        usable_records = sum(row.n for row in usable)
        evidence_dicts = [row.to_dict() for row in sorted(evidence, key=lambda row: (row.rank or 9999, row.action))]

        if len(usable) < 2:
            limits = ["at_least_two_actions_with_empirical_goal_evidence_required"]
            if any(row.n < min_records for row in evidence):
                limits.append("some_actions_below_min_empirical_records")
            return EmpiricalUtilityAudit(
                assessed=False,
                goal_variable=goal.goal_variable,
                observed_action=observed,
                total_records=total_records,
                usable_records=usable_records,
                min_records_per_action=min_records,
                action_evidence=evidence_dicts,
                reason="Not enough actions have sufficient empirical goal evidence for implicit-utility ranking.",
                reason_codes=["SFM_EMPIRICAL_UTILITY_INSUFFICIENT_ACTION_EVIDENCE"],
                limits=limits,
                raw={"query": query.to_dict(), "goal": goal.to_dict()},
            )

        selected = rankings[0].action
        observed_rank: Optional[int] = None
        observed_utility: Optional[float] = None
        for row in rankings:
            if row.action == observed:
                observed_rank = row.rank
                observed_utility = row.empirical_utility
                break
        matches = bool(selected and selected == observed)
        top = rankings[0].empirical_utility or 0.0
        second = rankings[1].empirical_utility or 0.0
        margin = max(0.0, float(top - second))
        support_strength = 1.0 if matches else 0.0
        if matches:
            # Keep the score bounded and conservative; a tiny margin is still
            # useful but weaker than a clear empirical separation.
            support_strength = min(1.0, 0.55 + margin)
        elif observed_utility is not None and top > 0:
            support_strength = max(0.0, 0.45 * float(observed_utility / max(top, 1e-9)))
        support_strength = round(float(support_strength), 6)

        reason_codes = ["SFM_EMPIRICAL_UTILITY_ASSESSED"]
        if matches:
            reason_codes.append("SFM_EMPIRICAL_UTILITY_SUPPORTS_OBSERVED_ACTION")
        else:
            reason_codes.append("SFM_EMPIRICAL_UTILITY_DOES_NOT_SELECT_OBSERVED_ACTION")
        if margin < 0.05:
            reason_codes.append("SFM_EMPIRICAL_UTILITY_LOW_MARGIN")

        limits: List[str] = []
        if any(row.n < min_records for row in evidence):
            limits.append("some_actions_below_min_empirical_records")
        if any(row.goal_observed_n == 0 for row in evidence):
            limits.append("some_actions_lacked_goal_measurements")

        return EmpiricalUtilityAudit(
            assessed=True,
            goal_variable=goal.goal_variable,
            observed_action=observed,
            selected_action=selected,
            selected_action_matches_observed=matches,
            observed_rank=observed_rank,
            support_strength=support_strength,
            total_records=total_records,
            usable_records=usable_records,
            min_records_per_action=min_records,
            action_evidence=evidence_dicts,
            reason=(
                "Empirical utility audit ranked candidate actions from observed action-outcome records. "
                "This is historical consistency evidence, not causal identification."
            ),
            reason_codes=reason_codes,
            limits=limits,
            raw={"query": query.to_dict(), "goal": goal.to_dict()},
        )


def evaluate_empirical_utility(payload: Any, goal: GoalSpec | str | Mapping[str, Any] | None = None) -> Dict[str, Any]:
    """Convenience API for implicit utility learned from outcome records."""

    query = FinalCauseQuery.from_payload(payload)
    goal_obj = GoalSpec.from_payload(goal) if goal is not None else query.candidate_goals[0]
    return EmpiricalUtilityLearner().evaluate(query, goal_obj).to_dict()
