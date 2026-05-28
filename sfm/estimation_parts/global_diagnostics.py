"""Top-level diagnostics helpers for the causal decision engine.

This module provides a stable, repo-level diagnostics API that can be imported by
future identification/policy layers without depending directly on internal
Level 3.2 implementation details.

It is intentionally non-destructive: adding this file does not change the
existing pipeline behaviour unless callers choose to import and use it.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional
import math

@dataclass
class DiagnosticsResult:
    overlap_ok: Optional[bool]
    balance_ok: Optional[bool]
    sample_size_ok: bool
    leakage_ok: Optional[bool]
    drift_ok: Optional[bool]
    temporal_order_ok: Optional[bool]
    sensitivity_level: str
    diagnostic_grade: str
    notes: List[str]
    overlap_gap: float = float("nan")
    balance_smd: float = float("nan")
    treated_propensity: float = float("nan")
    control_propensity_min: float = float("nan")
    control_propensity_max: float = float("nan")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _bool_or_none(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return bool(value)


def _sensitivity_level(effect: Any, se_proxy: Any) -> str:
    try:
        effect = float(effect)
        se_proxy = float(se_proxy)
    except (TypeError, ValueError, OverflowError):
        return "unknown"
    if not math.isfinite(effect) or not math.isfinite(se_proxy) or se_proxy <= 1e-9:
        return "unknown"
    ratio = abs(effect) / se_proxy
    if ratio >= 2.5:
        return "low"
    if ratio >= 1.5:
        return "medium"
    return "high"


def _diagnostic_grade(checks: Iterable[Optional[bool]], sensitivity_level: str) -> str:
    vals = [c for c in checks if c is not None]
    if any(v is False for v in vals):
        return "fail"
    passed = sum(1 for v in vals if v is True)
    if passed >= 5 and sensitivity_level in {"low", "medium"}:
        return "strong"
    if passed >= 3:
        return "moderate"
    if passed >= 1:
        return "weak"
    return "weak"


def _load_estimation_diagnostics():
    """Load heavy estimation diagnostics only when overlap/balance checks are requested."""
    try:
        from estimation_parts.diagnostics import _covariate_balance, _overlap_check
    except (ImportError, ModuleNotFoundError):  # pragma: no cover - optional internals may move
        return None, None
    return _covariate_balance, _overlap_check


def recommended_temporal_order(
    treatment_col: str,
    outcome_col: str,
    post_treatment_columns: Optional[Iterable[str]] = None,
) -> Optional[bool]:
    """Cheap temporal guardrail.

    Returns False when outcome clearly appears post-treatment or when the chosen
    treatment is listed among explicitly post-treatment fields.
    """
    t = (treatment_col or "").lower()
    y = (outcome_col or "").lower()
    if not t or not y:
        return None
    if t == y:
        return False
    post = {str(c).lower() for c in (post_treatment_columns or [])}
    if t in post:
        return False
    return True


def build_diagnostics(
    df: Any,
    treated_index: Optional[int] = None,
    controls: Optional[Iterable[int]] = None,
    covariates: Optional[Iterable[str]] = None,
    *,
    min_rows: int = 30,
    leakage_flag: Optional[bool] = None,
    drift_flag: Optional[bool] = None,
    treatment_col: str = "",
    outcome_col: str = "",
    post_treatment_columns: Optional[Iterable[str]] = None,
    effect: Any = float("nan"),
    se_proxy: Any = float("nan"),
) -> DiagnosticsResult:
    """Build a compact diagnostics summary for one treatment/outcome candidate.

    Parameters are intentionally permissive so this can be adopted incrementally
    by the current pipeline.
    """
    notes: List[str] = []
    overlap_ok: Optional[bool] = None
    balance_ok: Optional[bool] = None
    overlap_gap = float("nan")
    balance_smd = float("nan")
    p_t = p_min = p_max = float("nan")

    controls_list = list(controls or [])
    covs_list = list(covariates or [])
    needs_heavy_diagnostics = treated_index is not None and len(controls_list) > 0
    _covariate_balance, _overlap_check = _load_estimation_diagnostics() if needs_heavy_diagnostics else (None, None)
    sample_size_ok = len(df) >= int(min_rows)
    if not sample_size_ok:
        notes.append("LOW_SAMPLE_SIZE")

    if (
        treated_index is not None
        and _overlap_check is not None
        and len(controls_list) > 0
        and 0 <= int(treated_index) < len(df)
    ):
        ok, overlap_gap, p_t, p_min, p_max = _overlap_check(df, int(treated_index), controls_list)
        overlap_ok = bool(ok)
        if overlap_ok is False:
            notes.append("OVERLAP_FAIL")

    if (
        treated_index is not None
        and _covariate_balance is not None
        and len(controls_list) > 0
        and len(covs_list) > 0
        and 0 <= int(treated_index) < len(df)
    ):
        smd, ok = _covariate_balance(df, int(treated_index), controls_list, covs_list)
        balance_smd = float(smd) if math.isfinite(smd) else float("nan")
        balance_ok = bool(ok) if math.isfinite(balance_smd) else None
        if balance_ok is False:
            notes.append("BALANCE_FAIL")

    leakage_ok = None if leakage_flag is None else (not bool(leakage_flag))
    drift_ok = None if drift_flag is None else (not bool(drift_flag))
    temporal_order_ok = recommended_temporal_order(
        treatment_col=treatment_col,
        outcome_col=outcome_col,
        post_treatment_columns=post_treatment_columns,
    )

    if leakage_ok is False:
        notes.append("LEAKAGE_RISK")
    if drift_ok is False:
        notes.append("DRIFT_RISK")
    if temporal_order_ok is False:
        notes.append("TEMPORAL_ORDER_FAIL")

    sensitivity_level = _sensitivity_level(effect=effect, se_proxy=se_proxy)
    grade = _diagnostic_grade(
        [overlap_ok, balance_ok, sample_size_ok, leakage_ok, drift_ok, temporal_order_ok],
        sensitivity_level=sensitivity_level,
    )

    return DiagnosticsResult(
        overlap_ok=_bool_or_none(overlap_ok),
        balance_ok=_bool_or_none(balance_ok),
        sample_size_ok=bool(sample_size_ok),
        leakage_ok=_bool_or_none(leakage_ok),
        drift_ok=_bool_or_none(drift_ok),
        temporal_order_ok=_bool_or_none(temporal_order_ok),
        sensitivity_level=sensitivity_level,
        diagnostic_grade=grade,
        notes=notes,
        overlap_gap=float(overlap_gap) if math.isfinite(overlap_gap) else float("nan"),
        balance_smd=float(balance_smd) if math.isfinite(balance_smd) else float("nan"),
        treated_propensity=float(p_t) if math.isfinite(p_t) else float("nan"),
        control_propensity_min=float(p_min) if math.isfinite(p_min) else float("nan"),
        control_propensity_max=float(p_max) if math.isfinite(p_max) else float("nan"),
    )


__all__ = [
    "DiagnosticsResult",
    "build_diagnostics",
    "recommended_temporal_order",
]
