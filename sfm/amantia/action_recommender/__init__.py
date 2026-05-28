"""Causal action recommendation layer for AI agents.

Recommendations are proposal-only. They must pass the DecisionGate/ToolGuard
before execution.
"""

from .recommender import CausalActionRecommender, recommend_actions
from .types import RecommendedAction, RecommendedActionPackage

__all__ = [
    "CausalActionRecommender",
    "RecommendedAction",
    "RecommendedActionPackage",
    "recommend_actions",
]
