"""Deprecated compatibility facade for OperationalCausalGraph.

Canonical implementation: ``runtime.causal_graph_runtime``.
This wrapper is kept only for backward-compatible imports.
"""
from __future__ import annotations

from .causal_graph_runtime import OperationalCausalGraph

__all__ = ["OperationalCausalGraph"]
