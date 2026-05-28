"""Bundled Structural Final Models core for ``langgraph-sfm``.

The public LangGraph adapter imports from this package so users get the full
SFM backend without needing the old multi-package research repo layout.
"""

from .bootstrap import install_legacy_aliases
from .core import (
    infer_final_cause_compact,
    run_sfm_external_panel_benchmark,
    run_sfm_validation_benchmark,
)

__all__ = [
    "install_legacy_aliases",
    "infer_final_cause_compact",
    "run_sfm_external_panel_benchmark",
    "run_sfm_validation_benchmark",
]
