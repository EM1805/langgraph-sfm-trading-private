from __future__ import annotations

"""Utilities for representing intentional interventions, do*.

The expression produced here is a compact audit string.  It is not a parser for
SFM equations yet; it gives future code a single place to standardize do*.
"""

from typing import Any

from .schema import AgentModel, GoalSpec, IntentionalIntervention


def do_star_expression(action_variable: str, selected_action: str, goal: GoalSpec, agent: AgentModel) -> str:
    action_variable = action_variable or "action"
    selected_action = selected_action or "<policy-selected-action>"
    goal_name = goal.goal_variable or "<goal>"
    agent_id = agent.agent_id or "agent"
    return f"do*({action_variable}={selected_action} | agent={agent_id}, goal={goal_name})"


def build_intentional_intervention(
    *,
    action_variable: str = "action",
    selected_action: str = "",
    goal: GoalSpec | str | dict[str, Any] = "task_success",
    agent: AgentModel | dict[str, Any] | None = None,
    policy_name: str = "goal_directed_policy",
) -> IntentionalIntervention:
    goal_obj = GoalSpec.from_payload(goal)
    agent_obj = AgentModel.from_payload(agent or {})
    expression = do_star_expression(action_variable, selected_action, goal_obj, agent_obj)
    return IntentionalIntervention(
        action_variable=action_variable,
        selected_action=selected_action,
        goal=goal_obj,
        agent=agent_obj,
        policy_name=policy_name,
        expression=expression,
    )
