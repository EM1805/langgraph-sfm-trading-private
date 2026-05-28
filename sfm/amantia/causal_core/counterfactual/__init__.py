"""Counterfactual comparison adapter for Amantia."""

from .engine import (
    CounterfactualComparisonEngine,
    CounterfactualEngine,
    CounterfactualQuery,
    CounterfactualResult,
    compare_actions,
    compare_many,
)

__all__ = [
    "CounterfactualComparisonEngine",
    "CounterfactualEngine",
    "CounterfactualQuery",
    "CounterfactualResult",
    "compare_actions",
    "compare_many",
]
