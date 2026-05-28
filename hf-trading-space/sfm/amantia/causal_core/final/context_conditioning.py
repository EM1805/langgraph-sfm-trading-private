from __future__ import annotations

"""Context-conditioned SFM diagnostics.

Step 13 distinguishes a genuine temporal change of telos from a stable policy
that chooses different ends under different contexts.  For example, an agent may
appear to drift from ``task_success`` to ``latency`` over time, but the better
explanation may be: when ambiguity is high it optimizes task success; when
ambiguity is low it optimizes speed.

This module is diagnostic.  It groups historical decision records by a context
feature, runs inverse-goal/policy learning inside each context bucket, and
reports whether different contexts support different dominant goals.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .policy_learning import PolicyLearningEngine
from .schema import FinalCauseQuery


@dataclass
class ContextGoalProfile:
    """Inverse-goal evidence for one context value."""

    context_key: str
    context_value: str
    records: int
    assessed: bool = False
    most_likely_goal: str = ""
    support_strength: float = 0.0
    second_goal: str = ""
    second_goal_support: float = 0.0
    support_margin: float = 0.0
    selected_action_frequency: Dict[str, int] = field(default_factory=dict)
    goal_evidence: List[Dict[str, Any]] = field(default_factory=list)
    reason_codes: List[str] = field(default_factory=list)
    limits: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ContextConditioningAudit:
    """Diagnostic report for context-conditioned final-cause policies."""

    assessed: bool = False
    mode: str = "context_conditioned_sfm"
    total_records: int = 0
    usable_records: int = 0
    context_key: str = ""
    context_value_count: int = 0
    min_context_records: int = 3
    min_context_goal_support: float = 0.5
    context_conditioning_detected: bool = False
    policy_type: str = "unassessed"
    stable_context_policy: bool = False
    current_context_value: str = ""
    current_context_goal: str = ""
    current_context_support: float = 0.0
    dominant_goal_by_context: Dict[str, str] = field(default_factory=dict)
    context_profiles: List[Dict[str, Any]] = field(default_factory=list)
    alternative_context_keys: List[str] = field(default_factory=list)
    support_strength: float = 0.0
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


def _dedupe(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    for item in items:
        text = _clean_str(item)
        if text and text not in out:
            out.append(text)
    return out


def _policy_records(query: FinalCauseQuery) -> List[Dict[str, Any]]:
    raw = _as_dict(query.raw)
    for key in [
        "context_records",
        "contextual_records",
        "policy_records",
        "decision_records",
        "action_records",
        "action_history",
        "decision_history",
        "temporal_records",
        "trajectory_records",
        "trajectory",
        "trajectories",
    ]:
        records = raw.get(key)
        if isinstance(records, list) and records:
            return [_as_dict(record) for record in records if isinstance(record, Mapping)]
    return [dict(record) for record in query.outcome_records if isinstance(record, Mapping)]


def _context_mapping(record: Mapping[str, Any]) -> Dict[str, Any]:
    context: Dict[str, Any] = {}
    for key in ["context", "state", "environment", "conditions", "features"]:
        value = record.get(key)
        if isinstance(value, Mapping):
            context.update(dict(value))
    # Also allow common top-level context fields.
    for key, value in record.items():
        if key in {
            "ambiguity",
            "urgency",
            "risk_level",
            "user_expertise",
            "channel",
            "task_type",
            "environment",
            "phase",
            "segment",
            "cohort",
            "scenario",
        }:
            context.setdefault(key, value)
    return context


def _configured_context_keys(raw: Mapping[str, Any]) -> List[str]:
    keys: List[str] = []
    for key in ["context_key", "context_dimension", "context_variable"]:
        if raw.get(key) not in (None, ""):
            keys.append(_clean_str(raw.get(key)))
    for key in ["context_keys", "context_dimensions", "context_variables"]:
        for value in _as_list(raw.get(key)):
            text = _clean_str(value)
            if text:
                keys.append(text)
    return _dedupe(keys)


def _candidate_context_keys(records: Sequence[Mapping[str, Any]], configured: Sequence[str], min_records: int) -> Tuple[List[str], Dict[str, Dict[str, List[Dict[str, Any]]]]]:
    grouped_by_key: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    keys = list(configured)
    if not keys:
        seen: Dict[str, int] = {}
        for record in records:
            for key, value in _context_mapping(record).items():
                if value not in (None, ""):
                    seen[_clean_str(key)] = seen.get(_clean_str(key), 0) + 1
        keys = [key for key, count in sorted(seen.items(), key=lambda item: (-item[1], item[0])) if count >= min_records]

    for key in keys:
        buckets: Dict[str, List[Dict[str, Any]]] = {}
        for record in records:
            context = _context_mapping(record)
            if key not in context or context.get(key) in (None, ""):
                continue
            value = _clean_str(context.get(key))
            buckets.setdefault(value, []).append(dict(record))
        usable_buckets = {value: bucket for value, bucket in buckets.items() if len(bucket) >= min_records}
        if len(usable_buckets) >= 2:
            grouped_by_key[key] = usable_buckets
    # Prefer configured keys, otherwise keys with more usable buckets/records.
    ranked = sorted(
        grouped_by_key,
        key=lambda key: (
            0 if key in configured else 1,
            -len(grouped_by_key[key]),
            -sum(len(bucket) for bucket in grouped_by_key[key].values()),
            key,
        ),
    )
    return ranked, grouped_by_key


def _selected_action(record: Mapping[str, Any]) -> str:
    return _clean_str(
        record.get("selected_action")
        or record.get("observed_action")
        or record.get("action")
        or record.get("action_name")
        or record.get("decision")
        or record.get("candidate_action")
    )


def _action_frequencies(records: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for record in records:
        action = _selected_action(record)
        if action:
            counts[action] = counts.get(action, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _current_context_value(query: FinalCauseQuery, context_key: str) -> str:
    raw = _as_dict(query.raw)
    for container_key in ["current_context", "context", "current_state", "state"]:
        container = raw.get(container_key)
        if isinstance(container, Mapping) and context_key in container and container.get(context_key) not in (None, ""):
            return _clean_str(container.get(context_key))
    if context_key in query.state and query.state.get(context_key) not in (None, ""):
        return _clean_str(query.state.get(context_key))
    if context_key in raw and raw.get(context_key) not in (None, ""):
        return _clean_str(raw.get(context_key))
    return ""


class ContextConditioningEvaluator:
    """Evaluate whether apparent goal changes are explained by context."""

    def __init__(self, *, policy_learning_engine: Optional[PolicyLearningEngine] = None) -> None:
        self.policy_learning_engine = policy_learning_engine or PolicyLearningEngine()

    def evaluate(self, payload: Any) -> ContextConditioningAudit:
        query = FinalCauseQuery.from_payload(payload)
        raw = _as_dict(query.raw)
        records = _policy_records(query)
        min_context_records = max(1, _safe_int(raw.get("min_context_records"), query.min_policy_records))
        min_context_goal_support = max(0.0, min(1.0, _safe_float(raw.get("min_context_goal_support"), 0.5)))
        reason_codes: List[str] = []
        limits: List[str] = []

        if not query.candidate_goals:
            return ContextConditioningAudit(
                assessed=False,
                total_records=len(records),
                min_context_records=min_context_records,
                min_context_goal_support=min_context_goal_support,
                reason="Context-conditioned SFM requires candidate goals; run goal discovery first.",
                reason_codes=["SFM_CONTEXT_REQUIRES_CANDIDATE_GOALS"],
                limits=["candidate_goal_set_required"],
                raw=query.to_dict(),
            )
        if len(records) < 2 * min_context_records:
            return ContextConditioningAudit(
                assessed=False,
                total_records=len(records),
                usable_records=len(records),
                min_context_records=min_context_records,
                min_context_goal_support=min_context_goal_support,
                reason="Not enough records to compare at least two context buckets.",
                reason_codes=["SFM_CONTEXT_INSUFFICIENT_RECORDS"],
                limits=["at_least_two_context_buckets_required"],
                raw=query.to_dict(),
            )

        configured = _configured_context_keys(raw)
        candidate_keys, grouped_by_key = _candidate_context_keys(records, configured, min_context_records)
        if not candidate_keys:
            return ContextConditioningAudit(
                assessed=False,
                total_records=len(records),
                usable_records=0,
                min_context_records=min_context_records,
                min_context_goal_support=min_context_goal_support,
                reason="No context key produced two usable buckets.",
                reason_codes=["SFM_CONTEXT_NO_USABLE_CONTEXT_KEY"],
                limits=["context_key_with_two_buckets_required"],
                raw=query.to_dict(),
            )

        context_key = candidate_keys[0]
        buckets = grouped_by_key[context_key]
        profiles: List[ContextGoalProfile] = []
        for value, bucket_records in sorted(buckets.items(), key=lambda item: item[0]):
            bucket_payload = dict(raw)
            bucket_payload["policy_records"] = bucket_records
            bucket_payload["candidate_goals"] = [goal.to_dict() for goal in query.candidate_goals]
            bucket_payload["observed_action"] = query.observed_action
            bucket_payload["action_variable"] = query.action_variable
            bucket_payload["candidate_actions"] = query.candidate_actions
            bucket_payload["agent"] = query.agent.to_dict()
            bucket_payload["scm_graph"] = query.scm_graph
            bucket_payload["protected_outcome"] = query.protected_outcome
            bucket_payload["min_policy_records"] = min_context_records
            report = self.policy_learning_engine.evaluate(bucket_payload).to_dict()
            evidence = report.get("goal_evidence") or []
            top = evidence[0] if evidence else {}
            second = evidence[1] if len(evidence) > 1 else {}
            top_support = float(top.get("support_strength", report.get("support_strength", 0.0)) or 0.0)
            second_support = float(second.get("support_strength", 0.0) or 0.0)
            profiles.append(
                ContextGoalProfile(
                    context_key=context_key,
                    context_value=value,
                    records=len(bucket_records),
                    assessed=bool(report.get("assessed")),
                    most_likely_goal=_clean_str(report.get("most_likely_goal") or top.get("goal_variable")),
                    support_strength=round(top_support, 6),
                    second_goal=_clean_str(second.get("goal_variable")),
                    second_goal_support=round(second_support, 6),
                    support_margin=round(top_support - second_support, 6),
                    selected_action_frequency=_action_frequencies(bucket_records),
                    goal_evidence=evidence,
                    reason_codes=_dedupe(report.get("reason_codes") or []),
                    limits=_dedupe(report.get("limits") or []),
                )
            )

        strong_profiles = [
            profile
            for profile in profiles
            if profile.assessed and profile.most_likely_goal and profile.support_strength >= min_context_goal_support
        ]
        distinct_goals = sorted({profile.most_likely_goal for profile in strong_profiles if profile.most_likely_goal})
        assessed = len(strong_profiles) >= 2
        context_conditioning_detected = assessed and len(distinct_goals) >= 2
        if context_conditioning_detected:
            policy_type = "context_conditioned_policy"
            reason_codes.append("SFM_CONTEXT_CONDITIONED_POLICY_DETECTED")
        elif assessed and len(distinct_goals) == 1:
            policy_type = "context_invariant_policy"
            reason_codes.append("SFM_CONTEXT_INVARIANT_GOAL_PATTERN")
        elif profiles:
            policy_type = "weak_context_pattern"
            reason_codes.append("SFM_CONTEXT_WEAK_OR_UNSTABLE_PATTERN")
            limits.append("context_goal_support_below_threshold")
        else:
            policy_type = "unassessed"

        dominant_goal_by_context = {
            profile.context_value: profile.most_likely_goal
            for profile in strong_profiles
            if profile.context_value and profile.most_likely_goal
        }
        current_value = _current_context_value(query, context_key)
        current_goal = dominant_goal_by_context.get(current_value, "") if current_value else ""
        current_support = 0.0
        for profile in strong_profiles:
            if profile.context_value == current_value:
                current_support = profile.support_strength
                break
        if current_value and not current_goal:
            limits.append("current_context_value_not_in_strong_context_profiles")
        if not current_value:
            limits.append("current_context_value_not_supplied")

        support_strength = 0.0
        if strong_profiles:
            avg_support = sum(profile.support_strength for profile in strong_profiles) / len(strong_profiles)
            coverage = sum(profile.records for profile in strong_profiles) / max(len(records), 1)
            diversity_bonus = 1.0 if context_conditioning_detected else 0.75
            support_strength = round(min(1.0, avg_support * coverage * diversity_bonus), 6)

        if assessed and not context_conditioning_detected:
            reason_codes.append("SFM_CONTEXT_DOES_NOT_EXPLAIN_GOAL_CHANGE")
        if context_conditioning_detected and current_goal:
            reason_codes.append("SFM_CURRENT_CONTEXT_GOAL_IDENTIFIED")

        return ContextConditioningAudit(
            assessed=assessed,
            total_records=len(records),
            usable_records=sum(profile.records for profile in profiles),
            context_key=context_key,
            context_value_count=len(profiles),
            min_context_records=min_context_records,
            min_context_goal_support=min_context_goal_support,
            context_conditioning_detected=context_conditioning_detected,
            policy_type=policy_type,
            stable_context_policy=context_conditioning_detected,
            current_context_value=current_value,
            current_context_goal=current_goal,
            current_context_support=round(current_support, 6),
            dominant_goal_by_context=dominant_goal_by_context,
            context_profiles=[profile.to_dict() for profile in profiles],
            alternative_context_keys=[key for key in candidate_keys[1:]],
            support_strength=support_strength,
            authority_status="falsifiable_diagnostic_only" if assessed else "diagnostic_only",
            reason=(
                "Context-conditioned SFM grouped historical decisions by context and ran inverse-goal learning per bucket."
                if assessed
                else "Context-conditioned SFM could not obtain two strong context buckets."
            ),
            reason_codes=_dedupe([*reason_codes, *(code for profile in profiles for code in profile.reason_codes)]),
            limits=_dedupe([*limits, *(limit for profile in profiles for limit in profile.limits)]),
            raw=query.to_dict(),
        )


def evaluate_context_conditioning(payload: Any) -> Dict[str, Any]:
    """Convenience wrapper for context-conditioned SFM diagnostics."""

    return ContextConditioningEvaluator().evaluate(payload).to_dict()
