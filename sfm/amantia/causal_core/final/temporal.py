from __future__ import annotations

"""Temporal SFM diagnostics / goal drift detection.

Step 12 adds a conservative time-aware layer on top of policy learning.  Given
an ordered sequence of decisions, it evaluates inverse-goal evidence in adjacent
time windows and asks whether the dominant inferred goal changes over time.

This is not proof that an agent's telos changed.  It is a diagnostic for goal
stability, drift, or instability that can be combined with SCM/SFM evidence.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .policy_learning import PolicyLearningEngine
from .schema import FinalCauseQuery


@dataclass
class TemporalGoalWindow:
    """Inverse-goal evidence for one contiguous temporal window."""

    window_index: int
    start_index: int
    end_index: int
    records: int
    start_time: str = ""
    end_time: str = ""
    most_likely_goal: str = ""
    support_strength: float = 0.0
    second_goal: str = ""
    second_goal_support: float = 0.0
    support_margin: float = 0.0
    assessed: bool = False
    reason_codes: List[str] = field(default_factory=list)
    limits: List[str] = field(default_factory=list)
    goal_evidence: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GoalDriftEvent:
    """A change in dominant goal between adjacent windows."""

    from_window: int
    to_window: int
    from_goal: str
    to_goal: str
    from_support: float
    to_support: float
    drift_strength: float
    event_type: str = "goal_change"
    reason_codes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TemporalGoalDriftAudit:
    """Goal stability / drift audit over an ordered decision sequence."""

    assessed: bool = False
    mode: str = "temporal_goal_drift_detection"
    total_records: int = 0
    usable_records: int = 0
    window_size: int = 0
    min_window_records: int = 3
    min_goal_support: float = 0.5
    window_count: int = 0
    drift_detected: bool = False
    stability_status: str = "unassessed"
    initial_goal: str = ""
    final_goal: str = ""
    dominant_goal: str = ""
    drift_count: int = 0
    drift_score: float = 0.0
    support_strength: float = 0.0
    time_order_source: str = "input_order"
    window_evidence: List[Dict[str, Any]] = field(default_factory=list)
    drift_events: List[Dict[str, Any]] = field(default_factory=list)
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
    return [value]


def _dedupe(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    for item in items:
        text = _clean_str(item)
        if text and text not in out:
            out.append(text)
    return out


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _record_time(record: Mapping[str, Any], fallback_index: int) -> Tuple[Any, str]:
    for key in ["timestamp", "time", "date", "datetime", "t", "step", "index"]:
        value = record.get(key)
        if value not in (None, ""):
            return value, key
    return fallback_index, "input_order"


def _sort_records(records: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], str]:
    decorated: List[Tuple[Any, int, str, Dict[str, Any]]] = []
    source_counts: Dict[str, int] = {}
    for idx, record in enumerate(records):
        time_value, source = _record_time(record, idx)
        source_counts[source] = source_counts.get(source, 0) + 1
        decorated.append((time_value, idx, source, dict(record)))

    # Sorting by ISO-like timestamps works lexicographically; numeric steps work
    # directly.  If mixed/unorderable values appear, fall back to input order.
    try:
        decorated.sort(key=lambda item: (item[0], item[1]))
    except TypeError:
        decorated.sort(key=lambda item: item[1])
        return [item[3] for item in decorated], "input_order"

    time_source = max(source_counts.items(), key=lambda item: item[1])[0] if source_counts else "input_order"
    return [item[3] for item in decorated], time_source


def _temporal_records(query: FinalCauseQuery) -> List[Dict[str, Any]]:
    raw = _as_dict(query.raw)
    for key in [
        "temporal_records",
        "goal_drift_records",
        "policy_records",
        "decision_records",
        "action_records",
        "action_history",
        "decision_history",
        "trajectory_records",
        "trajectory",
        "trajectories",
    ]:
        records = raw.get(key)
        if isinstance(records, list) and records:
            return [_as_dict(record) for record in records if isinstance(record, Mapping)]
    return [dict(record) for record in query.outcome_records if isinstance(record, Mapping)]


def _make_windows(records: Sequence[Mapping[str, Any]], window_size: int) -> List[List[Dict[str, Any]]]:
    windows: List[List[Dict[str, Any]]] = []
    for start in range(0, len(records), window_size):
        chunk = [dict(record) for record in records[start : start + window_size]]
        if chunk:
            windows.append(chunk)
    return windows


def _dominant_goal_counts(windows: Sequence[TemporalGoalWindow]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for window in windows:
        goal = _clean_str(window.most_likely_goal)
        if goal:
            counts[goal] = counts.get(goal, 0) + 1
    return counts


class TemporalGoalDriftDetector:
    """Detect goal stability and drift across temporal policy windows."""

    def __init__(self, *, policy_learning_engine: Optional[PolicyLearningEngine] = None) -> None:
        self.policy_learning_engine = policy_learning_engine or PolicyLearningEngine()

    def evaluate(self, payload: Any) -> TemporalGoalDriftAudit:
        query = FinalCauseQuery.from_payload(payload)
        raw = _as_dict(query.raw)
        records = _temporal_records(query)
        min_window_records = max(
            1,
            _safe_int(raw.get("min_temporal_window_records") or raw.get("min_window_records"), query.min_policy_records),
        )
        configured_window_size = _safe_int(raw.get("temporal_window_size") or raw.get("drift_window_size"), 0)
        if configured_window_size <= 0:
            configured_window_size = min_window_records
        window_size = max(configured_window_size, min_window_records)
        min_goal_support = max(0.0, min(1.0, _safe_float(raw.get("min_temporal_goal_support"), 0.5)))
        reason_codes: List[str] = []
        limits: List[str] = []

        if not query.candidate_goals:
            return TemporalGoalDriftAudit(
                assessed=False,
                total_records=len(records),
                window_size=window_size,
                min_window_records=min_window_records,
                min_goal_support=min_goal_support,
                reason="Temporal drift detection requires candidate goals; run goal discovery first.",
                reason_codes=["SFM_TEMPORAL_REQUIRES_CANDIDATE_GOALS"],
                limits=["candidate_goal_set_required"],
                raw=query.to_dict(),
            )
        if len(records) < 2 * min_window_records:
            return TemporalGoalDriftAudit(
                assessed=False,
                total_records=len(records),
                usable_records=len(records),
                window_size=window_size,
                min_window_records=min_window_records,
                min_goal_support=min_goal_support,
                reason="Not enough temporal records to compare at least two goal windows.",
                reason_codes=["SFM_TEMPORAL_INSUFFICIENT_RECORDS"],
                limits=["at_least_two_temporal_windows_required"],
                raw=query.to_dict(),
            )

        ordered_records, time_order_source = _sort_records(records)
        raw_windows = _make_windows(ordered_records, window_size)
        # Drop trailing fragments that cannot support a policy-learning window.
        raw_windows = [window for window in raw_windows if len(window) >= min_window_records]
        if len(raw_windows) < 2:
            return TemporalGoalDriftAudit(
                assessed=False,
                total_records=len(records),
                usable_records=sum(len(window) for window in raw_windows),
                window_size=window_size,
                min_window_records=min_window_records,
                min_goal_support=min_goal_support,
                time_order_source=time_order_source,
                reason="Temporal records produced fewer than two usable windows.",
                reason_codes=["SFM_TEMPORAL_FEWER_THAN_TWO_USABLE_WINDOWS"],
                limits=["at_least_two_temporal_windows_required"],
                raw=query.to_dict(),
            )

        temporal_windows: List[TemporalGoalWindow] = []
        cursor = 0
        for idx, window_records in enumerate(raw_windows):
            start_index = cursor
            end_index = cursor + len(window_records) - 1
            cursor += len(window_records)
            window_payload = dict(raw)
            # Use the same candidate goals and agent context, but restrict the
            # learning sequence to this temporal slice.
            window_payload["policy_records"] = window_records
            window_payload["candidate_goals"] = [goal.to_dict() for goal in query.candidate_goals]
            window_payload["observed_action"] = query.observed_action
            window_payload["action_variable"] = query.action_variable
            window_payload["candidate_actions"] = query.candidate_actions
            window_payload["agent"] = query.agent.to_dict()
            window_payload["scm_graph"] = query.scm_graph
            window_payload["protected_outcome"] = query.protected_outcome
            window_payload["min_policy_records"] = min_window_records
            report = self.policy_learning_engine.evaluate(window_payload).to_dict()
            evidence = report.get("goal_evidence") or []
            top = evidence[0] if evidence else {}
            second = evidence[1] if len(evidence) > 1 else {}
            top_support = float(top.get("support_strength", report.get("support_strength", 0.0)) or 0.0)
            second_support = float(second.get("support_strength", 0.0) or 0.0)
            start_time, _ = _record_time(window_records[0], start_index)
            end_time, _ = _record_time(window_records[-1], end_index)
            temporal_windows.append(
                TemporalGoalWindow(
                    window_index=idx,
                    start_index=start_index,
                    end_index=end_index,
                    records=len(window_records),
                    start_time=_clean_str(start_time),
                    end_time=_clean_str(end_time),
                    most_likely_goal=_clean_str(report.get("most_likely_goal") or top.get("goal_variable")),
                    support_strength=round(top_support, 6),
                    second_goal=_clean_str(second.get("goal_variable")),
                    second_goal_support=round(second_support, 6),
                    support_margin=round(top_support - second_support, 6),
                    assessed=bool(report.get("assessed")),
                    reason_codes=_dedupe(report.get("reason_codes") or []),
                    limits=_dedupe(report.get("limits") or []),
                    goal_evidence=evidence,
                )
            )

        drift_events: List[GoalDriftEvent] = []
        strong_windows = [
            window
            for window in temporal_windows
            if window.assessed and window.most_likely_goal and window.support_strength >= min_goal_support
        ]
        for previous, current in zip(temporal_windows, temporal_windows[1:]):
            if not (previous.assessed and current.assessed and previous.most_likely_goal and current.most_likely_goal):
                continue
            both_strong = previous.support_strength >= min_goal_support and current.support_strength >= min_goal_support
            if previous.most_likely_goal != current.most_likely_goal and both_strong:
                strength = min(previous.support_strength, current.support_strength)
                drift_events.append(
                    GoalDriftEvent(
                        from_window=previous.window_index,
                        to_window=current.window_index,
                        from_goal=previous.most_likely_goal,
                        to_goal=current.most_likely_goal,
                        from_support=previous.support_strength,
                        to_support=current.support_strength,
                        drift_strength=round(strength, 6),
                        reason_codes=["SFM_TEMPORAL_GOAL_CHANGE_BETWEEN_STRONG_WINDOWS"],
                    )
                )
            elif previous.most_likely_goal == current.most_likely_goal and abs(current.support_strength - previous.support_strength) >= 0.25:
                # Confidence shifts are tracked as events, but do not count as
                # telic drift unless the dominant goal changes.
                drift_events.append(
                    GoalDriftEvent(
                        from_window=previous.window_index,
                        to_window=current.window_index,
                        from_goal=previous.most_likely_goal,
                        to_goal=current.most_likely_goal,
                        from_support=previous.support_strength,
                        to_support=current.support_strength,
                        drift_strength=round(abs(current.support_strength - previous.support_strength), 6),
                        event_type="confidence_shift",
                        reason_codes=["SFM_TEMPORAL_GOAL_CONFIDENCE_SHIFT"],
                    )
                )

        goal_change_events = [event for event in drift_events if event.event_type == "goal_change"]
        drift_detected = bool(goal_change_events)
        dominant_counts = _dominant_goal_counts(strong_windows or temporal_windows)
        dominant_goal = max(dominant_counts.items(), key=lambda item: (item[1], item[0]))[0] if dominant_counts else ""
        initial_goal = temporal_windows[0].most_likely_goal if temporal_windows else ""
        final_goal = temporal_windows[-1].most_likely_goal if temporal_windows else ""
        usable_records = sum(window.records for window in temporal_windows if window.assessed)
        assessed = len(temporal_windows) >= 2 and usable_records > 0
        goal_change_denominator = max(1, len(temporal_windows) - 1)
        avg_goal_change_strength = (
            sum(event.drift_strength for event in goal_change_events) / len(goal_change_events)
            if goal_change_events
            else 0.0
        )
        drift_score = round((len(goal_change_events) / goal_change_denominator) * avg_goal_change_strength, 6)
        support_strength = round(
            max([window.support_strength for window in temporal_windows] + [event.drift_strength for event in goal_change_events] + [0.0]),
            6,
        )

        if drift_detected:
            stability_status = "goal_drift_detected"
            reason_codes.append("SFM_TEMPORAL_GOAL_DRIFT_DETECTED")
            if initial_goal and final_goal and initial_goal != final_goal:
                reason_codes.append("SFM_TEMPORAL_INITIAL_FINAL_GOAL_DIFFER")
        elif len(strong_windows) >= 2 and len({window.most_likely_goal for window in strong_windows}) == 1:
            stability_status = "stable_goal_pattern"
            reason_codes.append("SFM_TEMPORAL_STABLE_GOAL_PATTERN")
        else:
            stability_status = "weak_or_unstable_goal_pattern"
            reason_codes.append("SFM_TEMPORAL_WEAK_OR_UNSTABLE_GOAL_PATTERN")
            limits.append("dominant_goal_not_stable_or_not_strong_enough")

        if any(window.limits for window in temporal_windows):
            limits.extend(limit for window in temporal_windows for limit in window.limits)
        if not strong_windows:
            limits.append("no_temporal_window_met_min_goal_support")

        return TemporalGoalDriftAudit(
            assessed=assessed,
            total_records=len(records),
            usable_records=usable_records,
            window_size=window_size,
            min_window_records=min_window_records,
            min_goal_support=round(min_goal_support, 6),
            window_count=len(temporal_windows),
            drift_detected=drift_detected,
            stability_status=stability_status,
            initial_goal=initial_goal,
            final_goal=final_goal,
            dominant_goal=dominant_goal,
            drift_count=len(goal_change_events),
            drift_score=drift_score,
            support_strength=support_strength,
            time_order_source=time_order_source,
            window_evidence=[window.to_dict() for window in temporal_windows],
            drift_events=[event.to_dict() for event in drift_events],
            reason=(
                "Temporal SFM audit compared inverse-goal evidence across ordered policy windows."
                if assessed
                else "Temporal SFM audit could not assess goal drift."
            ),
            reason_codes=_dedupe([*reason_codes, *(code for event in drift_events for code in event.reason_codes)]),
            limits=_dedupe(limits),
            raw=query.to_dict(),
        )


def evaluate_temporal_goal_drift(payload: Any) -> Dict[str, Any]:
    """Convenience wrapper for temporal goal-drift detection."""

    return TemporalGoalDriftDetector().evaluate(payload).to_dict()
