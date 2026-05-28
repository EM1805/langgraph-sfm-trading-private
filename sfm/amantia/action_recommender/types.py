from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class RecommendedAction:
    """One action proposed by Amantia after causal/safety evaluation.

    A recommendation is only a proposal. It must be converted into a new
    ActionPackage and pass the DecisionGate/ToolGuard before any tool executes.
    """

    action_name: str
    recommendation_type: str = "safer_alternative"
    rationale: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    safety_constraints: List[str] = field(default_factory=list)
    execution_status: str = "proposal_only_requires_gate_review"
    priority: int = 50
    gate_required: bool = True
    may_execute_without_guard: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RecommendedActionPackage:
    """Agent-facing recommendation bundle attached to a DecisionPackage."""

    original_action: str = ""
    decision: str = "abstain"
    recommended_action: Dict[str, Any] = field(default_factory=dict)
    recommended_actions: List[Dict[str, Any]] = field(default_factory=list)
    recommendation_summary: str = ""
    safety_constraints: List[str] = field(default_factory=list)
    execution_status: str = "no_recommendation"
    generated_by: str = "amantia.action_recommender"
    notes: List[str] = field(default_factory=list)
    causal_inputs: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
