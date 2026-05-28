from __future__ import annotations

"""Execution runner for the SFM diagnostic stack.

Step 24 moves layer gating, disabled reports and optional result coercion out of
``inference.py``.  The public API remains unchanged, but the orchestration point
is now reusable by future platform code and tests.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict

from .execution import SFMExecutionPlan
from .layer_protocol import layer_result_to_dict


@dataclass
class SFMExecutionRunner:
    """Small helper that executes enabled SFM layers conservatively."""

    plan: SFMExecutionPlan
    layer_outputs: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def is_enabled(self, layer: str) -> bool:
        return self.plan.is_enabled(layer)

    def disabled(self, layer: str) -> Dict[str, Any]:
        report = self.plan.disabled_report(layer)
        self.layer_outputs[layer] = report
        return report

    def run(self, layer: str, fn: Callable[[], Any]) -> Dict[str, Any]:
        """Run ``fn`` only when ``layer`` is enabled, returning a dict report."""

        if not self.plan.is_enabled(layer):
            return self.disabled(layer)
        result = layer_result_to_dict(fn())
        self.layer_outputs[layer] = result
        return result

    def to_dict(self) -> Dict[str, Any]:
        return {
            "execution_plan": self.plan.to_dict(),
            "executed_layers": sorted(k for k, v in self.layer_outputs.items() if not v.get("disabled")),
            "disabled_layers": sorted(k for k, v in self.layer_outputs.items() if v.get("disabled")),
        }
