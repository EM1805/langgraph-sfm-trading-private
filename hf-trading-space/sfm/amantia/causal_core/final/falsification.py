from __future__ import annotations

"""Falsification diagnostics for Structural Final Model development.

This module adds conservative checks before Amantia promotes an action-goal
relation to a diagnostic final-cause claim.  It asks whether the same observed
action also looks "intentional" for goals that should *not* explain it:

- negative-control goals: outcomes the analyst expects not to be targeted;
- placebo goals: fake or irrelevant outcomes used to probe over-attribution;
- side-effect goals: effects that may occur but should not be treated as ends
  without stronger evidence.

The checks are deliberately diagnostic.  They do not prove absence of intent;
they reduce confidence when the model is too eager to explain the same action
by many implausible goals.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .schema import FinalCauseQuery, GoalSpec
from .twin_model import TwinPolicyComparator


@dataclass
class FalsificationGoalAudit:
    """Audit result for one control/placebo/side-effect goal."""

    goal_variable: str = ""
    check_type: str = ""
    assessed: bool = False
    failed: bool = False
    severity: str = "none"
    reason: str = ""
    reason_codes: List[str] = field(default_factory=list)
    twin_support: Dict[str, Any] = field(default_factory=dict)
    observed_goal_source: str = ""
    observed_goal_value: Optional[float] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FalsificationReport:
    """Aggregate SFM falsification report for a candidate final cause."""

    assessed: bool = False
    candidate_goal: str = ""
    falsified: bool = False
    passed: bool = True
    intent_score_multiplier: float = 1.0
    failed_checks: List[Dict[str, Any]] = field(default_factory=list)
    passed_checks: List[Dict[str, Any]] = field(default_factory=list)
    unassessable_checks: List[Dict[str, Any]] = field(default_factory=list)
    negative_control_audits: List[Dict[str, Any]] = field(default_factory=list)
    placebo_audits: List[Dict[str, Any]] = field(default_factory=list)
    side_effect_audits: List[Dict[str, Any]] = field(default_factory=list)
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


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _observed_row(twin: Mapping[str, Any], observed_action: str) -> Dict[str, Any]:
    for row in _as_list(twin.get("with_goal_rankings")):
        if isinstance(row, Mapping) and row.get("action") == observed_action:
            return dict(row)
    return {}


def _has_observed_goal_measurement(row: Mapping[str, Any]) -> bool:
    source = _clean_str(row.get("goal_source"))
    return bool(source and source != "default")


def _float_or_none(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _twin_false_positive_signal(twin: Mapping[str, Any], observed_action: str) -> tuple[bool, str, Optional[float], List[str]]:
    """Return whether a control goal suspiciously explains the observed action.

    A control/placebo check fails only when the goal has an actual observed
    measurement for the action and the twin-policy comparison says the observed
    action is selected with the goal and changes when that goal is removed.
    This avoids treating missing/default outcome fields as falsification.
    """

    if not twin.get("compared"):
        return False, "", None, ["SFM_FALSIFICATION_TWIN_NOT_COMPARED"]

    row = _observed_row(twin, observed_action)
    source = _clean_str(row.get("goal_source"))
    value = _float_or_none(row.get("goal_value"))
    if not _has_observed_goal_measurement(row):
        return False, source or "default", value, ["SFM_FALSIFICATION_CONTROL_GOAL_UNMEASURED"]

    selected = bool(twin.get("observed_selected_with_goal"))
    dependent = bool(twin.get("action_changes_when_goal_removed"))
    if selected and dependent:
        return True, source, value, ["SFM_FALSIFICATION_CONTROL_GOAL_EXPLAINS_OBSERVED_ACTION"]
    return False, source, value, ["SFM_FALSIFICATION_CONTROL_GOAL_DOES_NOT_EXPLAIN_ACTION"]


class SFMFalsificationAuditor:
    """Run negative-control, placebo, and side-effect goal audits."""

    def __init__(self, *, twin_policy_comparator: Optional[TwinPolicyComparator] = None) -> None:
        self.twin_policy_comparator = twin_policy_comparator or TwinPolicyComparator()

    def _audit_goal(self, query: FinalCauseQuery, goal: GoalSpec, check_type: str) -> FalsificationGoalAudit:
        twin = self.twin_policy_comparator.compare(query, goal).to_dict()
        failed, source, value, signal_codes = _twin_false_positive_signal(twin, query.observed_action)
        assessed = bool(twin.get("compared")) and source != "default"

        severity = "none"
        reason_codes = ["SFM_FALSIFICATION_GOAL_AUDIT", f"SFM_{check_type.upper()}_GOAL_AUDIT"]
        reason_codes.extend(signal_codes)
        if failed:
            if check_type in {"negative_control", "placebo"}:
                severity = "high"
                reason_codes.append(f"SFM_{check_type.upper()}_GOAL_FAILED")
            else:
                severity = "medium"
                reason_codes.append("SFM_SIDE_EFFECT_GOAL_AMBIGUITY")
        elif assessed:
            reason_codes.append(f"SFM_{check_type.upper()}_GOAL_PASSED")
        else:
            reason_codes.append(f"SFM_{check_type.upper()}_GOAL_NOT_ASSESSABLE")

        if failed:
            reason = "The observed action also appears goal-dependent for this control/side-effect goal."
        elif assessed:
            reason = "The control/side-effect goal did not independently explain the observed action."
        else:
            reason = "The control/side-effect goal was not assessable with available action-outcome measurements."

        return FalsificationGoalAudit(
            goal_variable=goal.goal_variable,
            check_type=check_type,
            assessed=assessed,
            failed=failed,
            severity=severity,
            reason=reason,
            reason_codes=reason_codes,
            twin_support=twin,
            observed_goal_source=source,
            observed_goal_value=value,
            raw={"query": query.to_dict(), "goal": goal.to_dict()},
        )

    def audit(self, query: FinalCauseQuery, candidate_goal: GoalSpec) -> FalsificationReport:
        negative_controls = list(query.negative_control_goals)
        placebo_goals = list(query.placebo_goals)
        side_effect_goals = list(query.side_effect_goals)
        side_effect_goals.extend(GoalSpec.from_payload(x) for x in candidate_goal.side_effect_outcomes)

        # Avoid self-falsification: the candidate goal should not also be treated
        # as a placebo/control unless the caller explicitly marks it as a side
        # effect in the same GoalSpec.  Side-effect overlap is already handled by
        # FinalCauseEngine, so duplicate side-effect audits add no value.
        def without_candidate(goals: Iterable[GoalSpec]) -> List[GoalSpec]:
            return [g for g in goals if g.goal_variable and g.goal_variable != candidate_goal.goal_variable]

        negative_controls = without_candidate(negative_controls)
        placebo_goals = without_candidate(placebo_goals)
        side_effect_goals = without_candidate(side_effect_goals)

        neg_audits = [self._audit_goal(query, goal, "negative_control") for goal in negative_controls]
        placebo_audits = [self._audit_goal(query, goal, "placebo") for goal in placebo_goals]
        side_audits = [self._audit_goal(query, goal, "side_effect") for goal in side_effect_goals]
        all_audits = [*neg_audits, *placebo_audits, *side_audits]

        failed = [audit for audit in all_audits if audit.failed]
        passed = [audit for audit in all_audits if audit.assessed and not audit.failed]
        unassessable = [audit for audit in all_audits if not audit.assessed]

        high_fail = any(audit.failed and audit.severity == "high" for audit in failed)
        medium_fail = any(audit.failed and audit.severity == "medium" for audit in failed)
        multiplier = 1.0
        if high_fail:
            multiplier = 0.45
        elif medium_fail:
            multiplier = 0.70

        reason_codes: List[str] = ["SFM_FALSIFICATION_ASSESSED"]
        if not all_audits:
            reason_codes.append("SFM_FALSIFICATION_NO_CONTROL_GOALS_SUPPLIED")
        if failed:
            reason_codes.append("SFM_FALSIFICATION_FAILED")
        else:
            reason_codes.append("SFM_FALSIFICATION_PASSED")
        if unassessable:
            reason_codes.append("SFM_FALSIFICATION_HAS_UNASSESSABLE_CONTROLS")

        limits: List[str] = []
        if not all_audits:
            limits.append("negative_control_or_placebo_goals_not_supplied")
        if unassessable:
            limits.append("some_falsification_goals_lacked_action_outcome_measurements")

        if failed:
            reason = "At least one control/placebo/side-effect goal also explained the observed action; intent score was penalized."
        elif all_audits:
            reason = "Available control/placebo/side-effect goals did not falsify the candidate final-cause interpretation."
        else:
            reason = "No negative-control, placebo, or side-effect goals were supplied for SFM falsification."

        return FalsificationReport(
            assessed=True,
            candidate_goal=candidate_goal.goal_variable,
            falsified=bool(failed),
            passed=not bool(failed),
            intent_score_multiplier=multiplier,
            failed_checks=[audit.to_dict() for audit in failed],
            passed_checks=[audit.to_dict() for audit in passed],
            unassessable_checks=[audit.to_dict() for audit in unassessable],
            negative_control_audits=[audit.to_dict() for audit in neg_audits],
            placebo_audits=[audit.to_dict() for audit in placebo_audits],
            side_effect_audits=[audit.to_dict() for audit in side_audits],
            reason=reason,
            reason_codes=reason_codes,
            limits=limits,
            raw={"query": query.to_dict(), "candidate_goal": candidate_goal.to_dict()},
        )


def audit_sfm_falsification(payload: Any, goal: GoalSpec | str | Mapping[str, Any] | None = None) -> Dict[str, Any]:
    query = FinalCauseQuery.from_payload(payload)
    candidate_goal = GoalSpec.from_payload(goal) if goal is not None else query.candidate_goals[0]
    return SFMFalsificationAuditor().audit(query, candidate_goal).to_dict()
