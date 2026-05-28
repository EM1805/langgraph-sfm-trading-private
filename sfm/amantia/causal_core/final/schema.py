from __future__ import annotations

"""Minimal SFM-facing schemas for future Structural Final Model work.

This module is intentionally conservative: it does not claim to implement the
full Structural Final Model formalism.  It provides typed contracts for the
next layer: goals, agent belief/context, intentional interventions, and final
cause inference results.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional


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


@dataclass
class GoalSpec:
    """A candidate final cause / telic target.

    desired_direction is deliberately small: increase, decrease, maintain, or
    achieve.  protected_outcomes and side_effect_outcomes help separate what the
    agent appears to optimize from effects that merely occur.
    """

    goal_variable: str
    desired_direction: str = "increase"
    utility_weight: float = 1.0
    protected_outcomes: List[str] = field(default_factory=list)
    side_effect_outcomes: List[str] = field(default_factory=list)
    label: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Any) -> "GoalSpec":
        if isinstance(payload, GoalSpec):
            return payload
        if isinstance(payload, str):
            return cls(goal_variable=payload)
        data = _as_dict(payload)
        return cls(
            goal_variable=_clean_str(data.get("goal_variable") or data.get("outcome") or data.get("name")),
            desired_direction=_clean_str(data.get("desired_direction") or data.get("direction"), "increase"),
            utility_weight=float(data.get("utility_weight", data.get("weight", 1.0)) or 1.0),
            protected_outcomes=[_clean_str(x) for x in _as_list(data.get("protected_outcomes")) if _clean_str(x)],
            side_effect_outcomes=[_clean_str(x) for x in _as_list(data.get("side_effect_outcomes")) if _clean_str(x)],
            label=_clean_str(data.get("label")),
            metadata=_as_dict(data.get("metadata")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgentModel:
    """Lightweight agent model for intentional-intervention development."""

    agent_id: str = "agent"
    information_set: Dict[str, Any] = field(default_factory=dict)
    belief_graph: Dict[str, Any] = field(default_factory=dict)
    available_actions: List[str] = field(default_factory=list)
    utility_model: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Any) -> "AgentModel":
        if isinstance(payload, AgentModel):
            return payload
        data = _as_dict(payload)
        return cls(
            agent_id=_clean_str(data.get("agent_id") or data.get("id"), "agent"),
            information_set=_as_dict(data.get("information_set") or data.get("state")),
            belief_graph=_as_dict(data.get("belief_graph") or data.get("scm_graph")),
            available_actions=[_clean_str(x) for x in _as_list(data.get("available_actions")) if _clean_str(x)],
            utility_model={str(k): float(v) for k, v in _as_dict(data.get("utility_model")).items()},
            metadata=_as_dict(data.get("metadata")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class IntentionalIntervention:
    """A do* placeholder: action chosen as a policy of state, goal, and beliefs."""

    action_variable: str = "action"
    selected_action: str = ""
    goal: GoalSpec = field(default_factory=lambda: GoalSpec(goal_variable="task_success"))
    agent: AgentModel = field(default_factory=AgentModel)
    policy_name: str = "goal_directed_policy"
    expression: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FinalCauseQuery:
    """Input contract for goal/intent inference over existing Amantia components."""

    observed_action: str = ""
    action_variable: str = "agent_action"
    candidate_actions: List[Dict[str, Any]] = field(default_factory=list)
    candidate_goals: List[GoalSpec] = field(default_factory=list)
    negative_control_goals: List[GoalSpec] = field(default_factory=list)
    placebo_goals: List[GoalSpec] = field(default_factory=list)
    side_effect_goals: List[GoalSpec] = field(default_factory=list)
    agent: AgentModel = field(default_factory=AgentModel)
    scm_graph: Dict[str, Any] = field(default_factory=dict)
    state: Dict[str, Any] = field(default_factory=dict)
    outcome_records: List[Dict[str, Any]] = field(default_factory=list)
    outcome_log_path: str = ""
    normative_policy: Dict[str, Any] = field(default_factory=dict)
    protected_outcome: str = "user_or_system_harm"
    min_intent_score: float = 0.6
    min_empirical_records_per_action: int = 2
    min_policy_records: int = 3
    execution_profile: str = "full"
    enabled_layers: List[str] = field(default_factory=list)
    disabled_layers: List[str] = field(default_factory=list)
    query_id: str = ""
    source: str = "amantia.causal_core.final"
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Any) -> "FinalCauseQuery":
        if isinstance(payload, FinalCauseQuery):
            return payload
        data = _as_dict(payload)
        candidate_goals = [GoalSpec.from_payload(x) for x in _as_list(data.get("candidate_goals") or data.get("goals"))]
        negative_control_goals = [
            GoalSpec.from_payload(x)
            for x in _as_list(data.get("negative_control_goals") or data.get("negative_controls"))
        ]
        placebo_goals = [GoalSpec.from_payload(x) for x in _as_list(data.get("placebo_goals") or data.get("placebos"))]
        side_effect_goals = [
            GoalSpec.from_payload(x)
            for x in _as_list(data.get("side_effect_goals") or data.get("side_effect_outcomes"))
        ]
        if not candidate_goals and any(data.get(key) not in (None, "", [], {}) for key in ["goal", "intended_outcome", "outcome"]):
            goal = data.get("goal") or data.get("intended_outcome") or data.get("outcome")
            candidate_goals = [GoalSpec.from_payload(goal)]
        return cls(
            observed_action=_clean_str(data.get("observed_action") or data.get("selected_action") or data.get("action")),
            action_variable=_clean_str(data.get("action_variable") or data.get("treatment"), "agent_action"),
            candidate_actions=[_as_dict(x) if isinstance(x, Mapping) else {"action": _clean_str(x)} for x in _as_list(data.get("candidate_actions") or data.get("action_options"))],
            candidate_goals=candidate_goals,
            negative_control_goals=negative_control_goals,
            placebo_goals=placebo_goals,
            side_effect_goals=side_effect_goals,
            agent=AgentModel.from_payload(data.get("agent") or {}),
            scm_graph=_as_dict(data.get("scm_graph") or data.get("graph")),
            state=_as_dict(data.get("state")),
            outcome_records=[
                _as_dict(x)
                for x in _as_list(data.get("outcome_records") or data.get("empirical_outcome_records") or data.get("learning_records"))
                if isinstance(x, Mapping)
            ],
            outcome_log_path=_clean_str(data.get("outcome_log_path") or data.get("learning_log_path")),
            normative_policy=_as_dict(data.get("normative_policy") or data.get("value_policy") or data.get("alignment_policy")),
            protected_outcome=_clean_str(data.get("protected_outcome"), "user_or_system_harm"),
            min_intent_score=float(data.get("min_intent_score", 0.6) or 0.6),
            min_empirical_records_per_action=int(data.get("min_empirical_records_per_action", data.get("min_empirical_records", 2)) or 2),
            min_policy_records=int(data.get("min_policy_records", data.get("min_inverse_goal_records", 3)) or 3),
            execution_profile=_clean_str(data.get("execution_profile") or data.get("sfm_execution_profile") or data.get("profile"), "full"),
            enabled_layers=[_clean_str(x) for x in _as_list(data.get("enabled_layers") or data.get("sfm_enabled_layers")) if _clean_str(x)],
            disabled_layers=[_clean_str(x) for x in _as_list(data.get("disabled_layers") or data.get("sfm_disabled_layers")) if _clean_str(x)],
            query_id=_clean_str(data.get("query_id") or data.get("request_id")),
            source=_clean_str(data.get("source"), "amantia.causal_core.final"),
            raw=dict(data),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FinalCauseResult:
    """Diagnostic, not proof: candidate final-cause assessment."""

    inferred: bool = False
    # Step 23 epistemic split:
    # - intent_hypothesis_supported: diagnostic evidence passes threshold.
    # - intent_claim_authorized: identifiability layer authorizes reporting it as an SFM intent claim.
    # - governance_execution_allowed: alignment summary allows operational execution.
    intent_hypothesis_supported: bool = False
    intent_claim_authorized: bool = False
    governance_execution_allowed: bool = False
    most_likely_goal: str = ""
    observed_action: str = ""
    intent_score: float = 0.0
    intentional_intervention: Dict[str, Any] = field(default_factory=dict)
    causal_support: Dict[str, Any] = field(default_factory=dict)
    counterfactual_support: Dict[str, Any] = field(default_factory=dict)
    twin_support: Dict[str, Any] = field(default_factory=dict)
    belief_support: Dict[str, Any] = field(default_factory=dict)
    falsification_support: Dict[str, Any] = field(default_factory=dict)
    utility_support: Dict[str, Any] = field(default_factory=dict)
    empirical_utility_support: Dict[str, Any] = field(default_factory=dict)
    multi_goal_support: Dict[str, Any] = field(default_factory=dict)
    do_star_support: Dict[str, Any] = field(default_factory=dict)
    sfm_identifiability_support: Dict[str, Any] = field(default_factory=dict)
    goal_discovery_support: Dict[str, Any] = field(default_factory=dict)
    policy_learning_support: Dict[str, Any] = field(default_factory=dict)
    temporal_goal_drift_support: Dict[str, Any] = field(default_factory=dict)
    context_conditioning_support: Dict[str, Any] = field(default_factory=dict)
    hierarchical_goal_support: Dict[str, Any] = field(default_factory=dict)
    constraint_support: Dict[str, Any] = field(default_factory=dict)
    normative_support: Dict[str, Any] = field(default_factory=dict)
    action_recommendation_support: Dict[str, Any] = field(default_factory=dict)
    robustness_support: Dict[str, Any] = field(default_factory=dict)
    alignment_summary: Dict[str, Any] = field(default_factory=dict)
    audit_report: Dict[str, Any] = field(default_factory=dict)
    execution_profile_support: Dict[str, Any] = field(default_factory=dict)
    falsification_passed: bool = True
    side_effects_excluded: bool = False
    support_level: str = "none"
    authority_status: str = "diagnostic_only"
    reason: str = ""
    reason_codes: List[str] = field(default_factory=list)
    limits: List[str] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
