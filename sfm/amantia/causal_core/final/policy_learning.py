from __future__ import annotations

"""Policy learning / inverse goal inference for Structural Final Models.

Earlier SFM steps evaluated a single observed decision.  This module adds a
conservative sequence-level diagnostic: given a history of decisions, candidate
actions, and observed or expected outcomes, which candidate goal best explains
what the agent repeatedly selected?

This is not causal identification.  It is an inverse-policy consistency check
that can strengthen or weaken a final-cause hypothesis when enough historical
records are available.
"""

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .schema import FinalCauseQuery, GoalSpec
from .utility import _directional_utility, _outcome_value


@dataclass
class PolicyGoalEvidence:
    """Sequence-level evidence for one candidate goal."""

    goal_variable: str = ""
    desired_direction: str = "increase"
    utility_weight: float = 1.0
    records_considered: int = 0
    option_records: int = 0
    selected_utility_n: int = 0
    optimal_selected_n: int = 0
    selected_optimal_rate: Optional[float] = None
    avg_selected_utility: Optional[float] = None
    avg_best_available_utility: Optional[float] = None
    avg_goal_margin: Optional[float] = None
    likelihood_score: float = 0.0
    support_strength: float = 0.0
    rank: Optional[int] = None
    evidence_status: str = "unassessed"
    role: str = "candidate_goal"
    reason_codes: List[str] = field(default_factory=list)
    limits: List[str] = field(default_factory=list)
    examples: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PolicyLearningAudit:
    """Aggregate inverse goal-inference audit over a decision sequence."""

    assessed: bool = False
    mode: str = "sequence_policy_learning"
    observed_action: str = ""
    total_records: int = 0
    usable_records: int = 0
    min_policy_records: int = 3
    candidate_goal_count: int = 0
    most_likely_goal: str = ""
    most_likely_goal_score: float = 0.0
    inferred_goal_bundle: List[Dict[str, Any]] = field(default_factory=list)
    observed_action_frequency: Dict[str, int] = field(default_factory=dict)
    current_action_seen_in_sequence: bool = False
    support_strength: float = 0.0
    authority_status: str = "diagnostic_only"
    goal_evidence: List[Dict[str, Any]] = field(default_factory=list)
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


def _mean(values: Iterable[float]) -> Optional[float]:
    vals = [float(value) for value in values]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _action_name(value: Any) -> str:
    if isinstance(value, Mapping):
        return _clean_str(
            value.get("action")
            or value.get("action_name")
            or value.get("candidate_action")
            or value.get("selected_action")
            or value.get("observed_action")
            or value.get("decision")
            or value.get("name")
        )
    return _clean_str(value)


def _selected_action(record: Mapping[str, Any]) -> str:
    return _clean_str(
        record.get("selected_action")
        or record.get("observed_action")
        or record.get("action")
        or record.get("action_name")
        or record.get("decision")
        or record.get("candidate_action")
    )


def _record_options(record: Mapping[str, Any]) -> List[Dict[str, Any]]:
    for key in ["candidate_actions", "action_options", "alternatives", "options", "available_actions"]:
        raw_options = record.get(key)
        if isinstance(raw_options, list):
            return [_as_dict(option) if isinstance(option, Mapping) else {"action": _clean_str(option)} for option in raw_options]
    return []


def _policy_records(query: FinalCauseQuery) -> List[Dict[str, Any]]:
    raw = _as_dict(query.raw)
    for key in [
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


def _min_policy_records(query: FinalCauseQuery) -> int:
    raw = _as_dict(query.raw)
    value = raw.get("min_policy_records") or raw.get("min_inverse_goal_records") or getattr(query, "min_policy_records", None)
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = 3
    return max(parsed, 1)


def _is_protected_like(query: FinalCauseQuery, goal: GoalSpec) -> bool:
    outcome = _clean_str(goal.goal_variable).lower()
    protected = {_clean_str(query.protected_outcome).lower()}
    protected.update(_clean_str(item).lower() for item in goal.protected_outcomes)
    protected.update(_clean_str(item).lower() for item in goal.side_effect_outcomes)
    return bool(outcome and (outcome in protected or any(token in outcome for token in ["harm", "risk", "damage", "unsafe"])))


def _score_option_for_goal(option: Mapping[str, Any], goal: GoalSpec) -> Tuple[float, str]:
    value, source = _outcome_value(option, goal.goal_variable)
    utility = _directional_utility(value, goal.desired_direction, goal.metadata)
    return _clip01(utility), source


def _selected_option(record: Mapping[str, Any], options: Sequence[Mapping[str, Any]], selected: str) -> Dict[str, Any]:
    selected_l = selected.lower()
    for option in options:
        if _action_name(option).lower() == selected_l:
            return dict(option)
    fallback = dict(record)
    if selected and not _action_name(fallback):
        fallback["action"] = selected
    return fallback


def _action_frequencies(records: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for record in records:
        action = _selected_action(record)
        if action:
            counts[action] = counts.get(action, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _dedupe_reason_codes(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    for item in items:
        text = _clean_str(item)
        if text and text not in out:
            out.append(text)
    return out


class PolicyLearningEngine:
    """Infer candidate goals from a sequence of observed decisions."""

    def evaluate(self, payload: Any) -> PolicyLearningAudit:
        query = FinalCauseQuery.from_payload(payload)
        records = _policy_records(query)
        min_records = _min_policy_records(query)
        reason_codes: List[str] = []
        limits: List[str] = []

        if not query.candidate_goals:
            return PolicyLearningAudit(
                assessed=False,
                observed_action=query.observed_action,
                total_records=len(records),
                min_policy_records=min_records,
                reason="Policy learning requires candidate goals; run goal discovery before this audit.",
                reason_codes=["SFM_POLICY_LEARNING_REQUIRES_CANDIDATE_GOALS"],
                limits=["candidate_goal_set_required"],
                raw=query.to_dict(),
            )
        if not records:
            return PolicyLearningAudit(
                assessed=False,
                observed_action=query.observed_action,
                candidate_goal_count=len(query.candidate_goals),
                total_records=0,
                min_policy_records=min_records,
                reason="No policy, decision, action, trajectory, or outcome records were supplied.",
                reason_codes=["SFM_POLICY_LEARNING_NO_SEQUENCE_RECORDS"],
                limits=["sequence_records_required"],
                raw=query.to_dict(),
            )

        action_frequency = _action_frequencies(records)
        evidence_rows: List[PolicyGoalEvidence] = []

        for goal in query.candidate_goals:
            selected_utilities: List[float] = []
            best_utilities: List[float] = []
            margins: List[float] = []
            optimal_n = 0
            option_n = 0
            considered = 0
            examples: List[Dict[str, Any]] = []
            role = "protected_or_side_effect" if _is_protected_like(query, goal) else "candidate_goal"
            goal_reason_codes: List[str] = []
            goal_limits: List[str] = []

            for index, record in enumerate(records):
                selected = _selected_action(record)
                if not selected:
                    continue
                options = _record_options(record)
                selected_option = _selected_option(record, options, selected)
                selected_score, selected_source = _score_option_for_goal(selected_option, goal)
                selected_utilities.append(selected_score)
                considered += 1

                example: Dict[str, Any] = {
                    "index": index,
                    "selected_action": selected,
                    "selected_goal_utility": round(selected_score, 6),
                    "selected_source": selected_source,
                }

                if len(options) >= 2:
                    scored_options: List[Tuple[str, float]] = []
                    for option in options:
                        option_action = _action_name(option)
                        option_score, _ = _score_option_for_goal(option, goal)
                        scored_options.append((option_action, option_score))
                    best_score = max(score for _, score in scored_options)
                    best_utilities.append(best_score)
                    option_n += 1
                    if selected_score >= best_score - 1e-9:
                        optimal_n += 1
                    best_other = max((score for action, score in scored_options if action.lower() != selected.lower()), default=best_score)
                    margin = selected_score - best_other
                    margins.append(margin)
                    example.update({
                        "best_available_utility": round(best_score, 6),
                        "selected_is_goal_optimal": selected_score >= best_score - 1e-9,
                        "goal_margin_vs_best_other": round(margin, 6),
                    })
                if len(examples) < 4:
                    examples.append(example)

            avg_selected = _mean(selected_utilities)
            avg_best = _mean(best_utilities)
            avg_margin = _mean(margins)
            optimal_rate = (optimal_n / option_n) if option_n else None

            if considered == 0:
                likelihood = 0.0
                status = "no_usable_records"
                goal_reason_codes.append("SFM_POLICY_LEARNING_GOAL_NO_USABLE_RECORDS")
            elif optimal_rate is not None:
                margin_norm = _clip01(0.5 + float(avg_margin or 0.0) / 2.0)
                likelihood = (0.55 * optimal_rate) + (0.30 * float(avg_selected or 0.0)) + (0.15 * margin_norm)
                status = "sequence_with_action_alternatives"
                goal_reason_codes.append("SFM_POLICY_LEARNING_USED_ACTION_ALTERNATIVES")
            else:
                likelihood = float(avg_selected or 0.0)
                status = "sequence_outcome_only"
                goal_reason_codes.append("SFM_POLICY_LEARNING_USED_SELECTED_ACTION_OUTCOMES_ONLY")
                goal_limits.append("no_per_decision_action_alternatives")

            record_factor = min(1.0, considered / float(min_records))
            support = _clip01(likelihood * record_factor, 0.0)
            if role == "protected_or_side_effect":
                # A sequence can show that an agent avoids harm, but this should
                # be treated as a protected constraint unless explicitly modeled
                # as a final cause with stronger evidence.
                support = min(support, 0.49)
                goal_reason_codes.append("SFM_POLICY_LEARNING_PROTECTED_LIKE_GOAL_CAPPED")
                goal_limits.append("protected_or_side_effect_goal_not_promoted_as_final_cause")

            if considered < min_records:
                goal_reason_codes.append("SFM_POLICY_LEARNING_BELOW_MIN_RECORDS")
                goal_limits.append("insufficient_sequence_length")

            evidence_rows.append(
                PolicyGoalEvidence(
                    goal_variable=goal.goal_variable,
                    desired_direction=goal.desired_direction,
                    utility_weight=round(float(goal.utility_weight), 6),
                    records_considered=considered,
                    option_records=option_n,
                    selected_utility_n=len(selected_utilities),
                    optimal_selected_n=optimal_n,
                    selected_optimal_rate=round(optimal_rate, 6) if optimal_rate is not None else None,
                    avg_selected_utility=round(avg_selected, 6) if avg_selected is not None else None,
                    avg_best_available_utility=round(avg_best, 6) if avg_best is not None else None,
                    avg_goal_margin=round(avg_margin, 6) if avg_margin is not None else None,
                    likelihood_score=round(_clip01(likelihood, 0.0), 6),
                    support_strength=round(support, 6),
                    evidence_status=status,
                    role=role,
                    reason_codes=_dedupe_reason_codes(goal_reason_codes),
                    limits=_dedupe_reason_codes(goal_limits),
                    examples=examples,
                )
            )

        ranked = sorted(evidence_rows, key=lambda row: (row.support_strength, row.likelihood_score, row.goal_variable), reverse=True)
        for idx, row in enumerate(ranked, start=1):
            row.rank = idx

        usable_records = max((row.records_considered for row in evidence_rows), default=0)
        assessed = bool(evidence_rows) and usable_records > 0
        top = ranked[0] if ranked else PolicyGoalEvidence()
        bundle_threshold = max(0.5, float(top.support_strength) - 0.12)
        bundle = [row.to_dict() for row in ranked if row.support_strength >= bundle_threshold and row.support_strength > 0.0][:4]

        if assessed:
            reason_codes.append("SFM_POLICY_LEARNING_ASSESSED_SEQUENCE")
            if top.support_strength >= 0.65 and usable_records >= min_records:
                reason_codes.append("SFM_POLICY_LEARNING_FOUND_PLAUSIBLE_GOAL")
            elif usable_records < min_records:
                reason_codes.append("SFM_POLICY_LEARNING_SEQUENCE_TOO_SHORT")
                limits.append("insufficient_sequence_length")
            else:
                reason_codes.append("SFM_POLICY_LEARNING_WEAK_GOAL_EVIDENCE")
        else:
            reason_codes.append("SFM_POLICY_LEARNING_NO_USABLE_SEQUENCE_RECORDS")
            limits.append("usable_sequence_records_required")

        goal_evidence = [row.to_dict() for row in ranked]
        return PolicyLearningAudit(
            assessed=assessed,
            observed_action=query.observed_action,
            total_records=len(records),
            usable_records=usable_records,
            min_policy_records=min_records,
            candidate_goal_count=len(query.candidate_goals),
            most_likely_goal=top.goal_variable if assessed else "",
            most_likely_goal_score=round(float(top.support_strength), 6) if assessed else 0.0,
            inferred_goal_bundle=bundle,
            observed_action_frequency=action_frequency,
            current_action_seen_in_sequence=bool(query.observed_action and query.observed_action in action_frequency),
            support_strength=round(float(top.support_strength), 6) if assessed else 0.0,
            goal_evidence=goal_evidence,
            reason=(
                "Sequence-level inverse goal inference computed from historical selected actions, "
                "per-decision alternatives when available, and candidate-goal utilities."
                if assessed
                else "Policy learning could not assess the sequence."
            ),
            reason_codes=_dedupe_reason_codes([*reason_codes, *(code for row in ranked for code in row.reason_codes)]),
            limits=_dedupe_reason_codes([*limits, *(limit for row in ranked for limit in row.limits)]),
            raw=query.to_dict(),
        )


def evaluate_policy_learning(payload: Any) -> Dict[str, Any]:
    """Convenience wrapper for sequence-level inverse goal inference."""

    return PolicyLearningEngine().evaluate(payload).to_dict()
