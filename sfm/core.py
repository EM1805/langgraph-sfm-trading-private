from __future__ import annotations

"""Stable public entry points for the bundled SFM core."""

from typing import Any, Dict, Mapping

from .bootstrap import install_legacy_aliases

install_legacy_aliases()

def infer_final_cause_compact(query: Mapping[str, Any]) -> Dict[str, Any]:
    """Run the full Structural Final Model compact final-cause inference.

    This delegates to the bundled Amantia/SFM research core relocated under
    ``sfm/``.  The return value is intentionally a plain dict so it can be
    consumed directly by LangGraph nodes and monitors.
    """
    from amantia.causal_core.final import infer_final_cause_compact as _impl
    return dict(_impl(dict(query)))

def run_sfm_validation_benchmark() -> Dict[str, Any]:
    """Run the bundled synthetic SFM validation benchmark."""
    from amantia.causal_core.final.validation_benchmark import run_sfm_validation_benchmark as _impl
    result = _impl()
    return result if isinstance(result, dict) else dict(result)

def run_sfm_external_panel_benchmark(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Run the bundled panel-backed SFM benchmark when the data is packaged."""
    from amantia.causal_core.final.external_validation import run_sfm_external_panel_benchmark as _impl
    result = _impl(*args, **kwargs)
    return result if isinstance(result, dict) else dict(result)
