from __future__ import annotations

"""Common protocol types for SFM diagnostic layers.

The SFM stack is intentionally multi-layered.  This module gives future layers a
small shared contract without forcing every evaluator to inherit from a base
class.  A layer may return either a dataclass-like object exposing ``to_dict`` or
an ordinary mapping.
"""

from typing import Any, Dict, Mapping, Protocol, runtime_checkable


@runtime_checkable
class SFMLayerResult(Protocol):
    """Protocol for dataclass-style layer results."""

    def to_dict(self) -> Dict[str, Any]:
        ...


@runtime_checkable
class SFMLayerEvaluator(Protocol):
    """Protocol for diagnostic evaluators used by the execution runner."""

    def evaluate(self, *args: Any, **kwargs: Any) -> Any:
        ...


def layer_result_to_dict(value: Any) -> Dict[str, Any]:
    """Coerce a layer result into a plain dictionary.

    This keeps orchestration code small and avoids repeating ``hasattr`` checks
    across ``inference.py`` and future runner implementations.
    """

    if hasattr(value, "to_dict"):
        return value.to_dict()  # type: ignore[no-any-return]
    if isinstance(value, Mapping):
        return dict(value)
    if value is None:
        return {}
    return {"value": value}
