from __future__ import annotations

"""Formal do* operator API for Structural Final Model development.

The earlier SFM scaffold exposed a compact string such as
``do*(A=a | agent=i, goal=g)``.  This module adds a richer, auditable operator:

    do*(A = pi(S, B, G_bundle, U))

where:
- S is the observed state / information set;
- B is the agent belief graph;
- G_bundle is one or more final-cause targets;
- U is the utility specification used by the policy.

This remains a diagnostic API.  It serializes the intentional intervention and
checks whether the induced policy selects the observed action; it does not claim
full SFM identification.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from .multi_goal import MultiGoalUtilityEvaluator
from .schema import FinalCauseQuery, GoalSpec
from .utility import UtilityFunctionEvaluator


@dataclass
class DoStarPolicyInputAudit:
    """Which ingredients were available for a formal do* policy."""

    state_supplied: bool = False
    belief_graph_supplied: bool = False
    goal_bundle_supplied: bool = False
    utility_model_supplied: bool = False
    candidate_actions_supplied: bool = False
    protected_outcome_supplied: bool = False
    available_action_count: int = 0
    goal_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DoStarOperatorResult:
    """Audit result for ``do*(A = policy(S, B, G_bundle, U))``."""

    evaluated: bool = False
    operator: str = "do_star"
    action_variable: str = "agent_action"
    policy_name: str = "goal_directed_policy"
    policy_signature: str = "policy(S, B, G_bundle, U)"
    expression: str = ""
    observed_action: str = ""
    selected_action: str = ""
    selected_action_matches_observed: bool = False
    support_strength: float = 0.0
    goal_bundle: List[str] = field(default_factory=list)
    policy_inputs: Dict[str, Any] = field(default_factory=dict)
    policy_audit: Dict[str, Any] = field(default_factory=dict)
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


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _policy_name(query: FinalCauseQuery) -> str:
    raw = _as_dict(query.raw)
    return _clean_str(
        raw.get("policy_name")
        or raw.get("do_star_policy")
        or raw.get("intentional_policy")
        or query.agent.metadata.get("policy_name"),
        "goal_directed_policy",
    )


def _explicit_goal_bundle(query: FinalCauseQuery) -> List[GoalSpec]:
    raw = _as_dict(query.raw)
    explicit = raw.get("goal_bundle") or raw.get("multi_goal_bundle") or raw.get("multi_objective_goals")
    if explicit:
        if isinstance(explicit, list):
            return [GoalSpec.from_payload(item) for item in explicit]
        return [GoalSpec.from_payload(explicit)]
    return list(query.candidate_goals or [])


def _utility_model_supplied(query: FinalCauseQuery) -> bool:
    raw = _as_dict(query.raw)
    return bool(
        query.agent.utility_model
        or raw.get("utility_function")
        or raw.get("utility_components")
        or raw.get("utility_model")
    )


def _make_input_audit(query: FinalCauseQuery, goals: List[GoalSpec]) -> DoStarPolicyInputAudit:
    return DoStarPolicyInputAudit(
        state_supplied=bool(query.state or query.agent.information_set),
        belief_graph_supplied=bool(query.agent.belief_graph),
        goal_bundle_supplied=bool(goals),
        utility_model_supplied=_utility_model_supplied(query),
        candidate_actions_supplied=bool(query.candidate_actions),
        protected_outcome_supplied=bool(query.protected_outcome),
        available_action_count=len(query.candidate_actions),
        goal_count=len(goals),
    )


def _goal_list_text(goals: List[GoalSpec]) -> str:
    names = [goal.goal_variable for goal in goals if goal.goal_variable]
    return "[" + ",".join(names) + "]" if names else "[]"


def _format_do_star_expression(
    *,
    action_variable: str,
    selected_action: str,
    policy_name: str,
    agent_id: str,
    goals: List[GoalSpec],
) -> str:
    action_variable = _clean_str(action_variable, "agent_action")
    selected_action = _clean_str(selected_action, "<policy-selected-action>")
    policy_name = _clean_str(policy_name, "goal_directed_policy")
    agent_id = _clean_str(agent_id, "agent")
    goal_text = _goal_list_text(goals)
    return (
        f"do*({action_variable}=pi_{policy_name}(S,B,G_bundle,U)->{selected_action} "
        f"| agent={agent_id}, goals={goal_text})"
    )


def _observed_rank(rankings: List[Mapping[str, Any]], observed: str) -> Optional[int]:
    for row in rankings:
        if str(row.get("action") or "") == observed:
            value = row.get("rank")
            return int(value) if value is not None else None
    return None


def _support_from_audit(audit: Mapping[str, Any], matches: bool, rankings: List[Mapping[str, Any]], observed: str) -> float:
    if audit.get("support_strength") is not None:
        try:
            return max(0.0, min(1.0, float(audit.get("support_strength") or 0.0)))
        except (TypeError, ValueError):
            pass
    if matches:
        return 0.75
    rank = _observed_rank(rankings, observed)
    if rank is None:
        return 0.0
    return max(0.0, 0.45 - 0.1 * max(rank - 1, 0))


class DoStarOperator:
    """Build and evaluate a formal intentional-intervention operator.

    The operator delegates action selection to existing utility diagnostics:
    multi-goal evaluation for bundles and single-goal utility evaluation for a
    one-goal policy.  Its job is to standardize the SFM surface:
    expression, policy inputs, selected action, rankings, and reason codes.
    """

    def __init__(
        self,
        *,
        multi_goal_evaluator: Optional[MultiGoalUtilityEvaluator] = None,
        utility_evaluator: Optional[UtilityFunctionEvaluator] = None,
    ) -> None:
        self.multi_goal_evaluator = multi_goal_evaluator or MultiGoalUtilityEvaluator()
        self.utility_evaluator = utility_evaluator or UtilityFunctionEvaluator()

    def evaluate(self, payload: Any) -> DoStarOperatorResult:
        query = FinalCauseQuery.from_payload(payload)
        goals = _explicit_goal_bundle(query)
        input_audit = _make_input_audit(query, goals)
        policy_name = _policy_name(query)
        goal_bundle = [goal.goal_variable for goal in goals if goal.goal_variable]

        if not query.candidate_actions:
            expression = _format_do_star_expression(
                action_variable=query.action_variable,
                selected_action=query.observed_action,
                policy_name=policy_name,
                agent_id=query.agent.agent_id,
                goals=goals,
            )
            return DoStarOperatorResult(
                evaluated=False,
                action_variable=query.action_variable,
                policy_name=policy_name,
                expression=expression,
                observed_action=query.observed_action,
                goal_bundle=goal_bundle,
                policy_inputs=input_audit.to_dict(),
                reason="Formal do* operator requires candidate actions to evaluate a policy selection.",
                reason_codes=["SFM_DO_STAR_REQUIRES_CANDIDATE_ACTIONS"],
                limits=["candidate_action_set_required"],
                raw=query.to_dict(),
            )

        if not goals:
            expression = _format_do_star_expression(
                action_variable=query.action_variable,
                selected_action=query.observed_action,
                policy_name=policy_name,
                agent_id=query.agent.agent_id,
                goals=goals,
            )
            return DoStarOperatorResult(
                evaluated=False,
                action_variable=query.action_variable,
                policy_name=policy_name,
                expression=expression,
                observed_action=query.observed_action,
                goal_bundle=goal_bundle,
                policy_inputs=input_audit.to_dict(),
                reason="Formal do* operator requires at least one goal in G_bundle.",
                reason_codes=["SFM_DO_STAR_REQUIRES_GOAL_BUNDLE"],
                limits=["goal_bundle_required"],
                raw=query.to_dict(),
            )

        if len(goals) >= 2:
            audit = self.multi_goal_evaluator.evaluate(query).to_dict()
            audit_kind = "multi_goal_utility"
        else:
            audit = self.utility_evaluator.evaluate(query, goals[0]).to_dict()
            audit_kind = "single_goal_utility"

        selected = _clean_str(audit.get("selected_action") or query.observed_action)
        observed = query.observed_action or selected
        rankings = list(audit.get("rankings") or [])
        matches = bool(selected and observed and selected == observed)
        expression = _format_do_star_expression(
            action_variable=query.action_variable,
            selected_action=selected,
            policy_name=policy_name,
            agent_id=query.agent.agent_id,
            goals=goals,
        )
        support = round(_support_from_audit(audit, matches, rankings, observed), 6)

        reason_codes: List[str] = ["SFM_DO_STAR_OPERATOR_EVALUATED"]
        if len(goals) >= 2:
            reason_codes.append("SFM_DO_STAR_POLICY_USES_GOAL_BUNDLE")
        else:
            reason_codes.append("SFM_DO_STAR_POLICY_USES_SINGLE_GOAL")
        if input_audit.belief_graph_supplied:
            reason_codes.append("SFM_DO_STAR_POLICY_HAS_BELIEF_GRAPH")
        else:
            reason_codes.append("SFM_DO_STAR_POLICY_MISSING_BELIEF_GRAPH")
        if input_audit.utility_model_supplied:
            reason_codes.append("SFM_DO_STAR_POLICY_HAS_EXPLICIT_UTILITY")
        else:
            reason_codes.append("SFM_DO_STAR_POLICY_USES_DERIVED_UTILITY")
        if matches:
            reason_codes.append("SFM_DO_STAR_SELECTS_OBSERVED_ACTION")
        else:
            reason_codes.append("SFM_DO_STAR_DOES_NOT_SELECT_OBSERVED_ACTION")
        reason_codes.extend(audit.get("reason_codes") or [])

        limits: List[str] = ["diagnostic_do_star_operator_not_full_structural_final_identification"]
        if not input_audit.belief_graph_supplied:
            limits.append("agent_belief_graph_not_supplied")
        if not input_audit.utility_model_supplied:
            limits.append("utility_function_derived_not_explicit")
        limits.extend(audit.get("limits") or [])

        return DoStarOperatorResult(
            evaluated=bool(audit.get("assessed", True)),
            action_variable=query.action_variable,
            policy_name=policy_name,
            expression=expression,
            observed_action=observed,
            selected_action=selected,
            selected_action_matches_observed=matches,
            support_strength=support,
            goal_bundle=goal_bundle,
            policy_inputs=input_audit.to_dict(),
            policy_audit={"kind": audit_kind, **audit},
            rankings=rankings,
            reason=(
                "Formal do* diagnostic evaluated A = policy(S, B, G_bundle, U) using existing "
                "utility and multi-goal SFM components."
            ),
            reason_codes=reason_codes,
            limits=limits,
            raw=query.to_dict(),
        )


def evaluate_do_star_intervention(payload: Any) -> Dict[str, Any]:
    """Convenience API for the formal intentional-intervention operator."""

    return DoStarOperator().evaluate(payload).to_dict()
