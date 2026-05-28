"""Runtime operational graph loader.

The operational causal graph belongs to the runtime/veto boundary, not to
PCMCI Discovery. Discovery should not import this module; runtime code may use
it as a stable loader facade over ``runtime.causal_graph_runtime``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

from .causal_graph_runtime import OperationalCausalGraph


def load_operational_causal_graph(path: Union[str, Path] = "operational_causal_graph.yaml") -> OperationalCausalGraph:
    return OperationalCausalGraph.load(path)


__all__ = ["OperationalCausalGraph", "load_operational_causal_graph"]
