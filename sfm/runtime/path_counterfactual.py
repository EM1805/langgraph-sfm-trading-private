"""Legacy runtime facade for path-level counterfactual diagnostics.

Canonical path reasoning now lives under ``path_parts``. This module remains
for runtime compatibility and is diagnostic-only: it does not grant SCM causal
authority. SCM/ID-gated authority belongs to ``scm_parts.scm_counterfactual``.
"""
from __future__ import annotations

from .cf_types import PathTreatmentSpec
from path_parts.path_counterfactual_core import (
    CounterfactualEstimator,
    _build_matched_sets,
    _weighted_harm_rate,
    enrich_paths_with_validation_guided_counterfactual,
)

PATH_COUNTERFACTUAL_AUTHORITY = "diagnostic_path_level_only"


def mark_path_counterfactual_diagnostic(row: dict) -> dict:
    """Add explicit non-authority metadata to a path counterfactual row."""
    out = dict(row or {})
    out.setdefault("path_counterfactual_authority", PATH_COUNTERFACTUAL_AUTHORITY)
    out.setdefault("scm_counterfactual_authority", "not_evaluated_use_scm_counterfactual")
    out.setdefault("reason_codes", "PATH_DIAGNOSTIC_NOT_SCM_AUTHORITY")
    return out


__all__ = [
    "CounterfactualEstimator",
    "PathTreatmentSpec",
    "enrich_paths_with_validation_guided_counterfactual",
    "_build_matched_sets",
    "_weighted_harm_rate",
    "PATH_COUNTERFACTUAL_AUTHORITY",
    "mark_path_counterfactual_diagnostic",
]
