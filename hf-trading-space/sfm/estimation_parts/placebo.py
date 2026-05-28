"""Estimation-owned placebo diagnostics.

This module currently implements a conservative future-treatment placebo: a
candidate is suspicious when future treatment variation predicts the current
outcome nearly as strongly as the lagged treatment used for the main estimate.
The artifact is diagnostic only and never grants causal authority.
"""
from __future__ import annotations

from runtime_env import configure_scientific_runtime
configure_scientific_runtime()

from typing import Iterable

import numpy as np
import pandas as pd

from . import _utils as U
from . import stat_core as SC

PLACEBO_COLUMNS = [
    "placebo_id", "effect_id", "insight_id", "source", "target", "treatment_col", "outcome_col", "lag",
    "placebo_type", "placebo_status", "placebo_pass", "placebo_effect_estimate",
    "placebo_standard_error", "placebo_t_stat", "placebo_p_value_approx",
    "placebo_abs_ratio_to_main", "support_n", "reason",
]


def _as_str(value) -> str:
    text = U.as_str(value).strip()
    return "" if text.lower() in {"nan", "none", "null", "nat"} else text


def _safe_float(value, default=np.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _future_series(df: pd.DataFrame, col: str, lag: int) -> pd.Series:
    s = pd.to_numeric(df[col], errors="coerce").astype(float)
    lead = max(1, int(lag) if int(lag) > 0 else 1)
    return s.shift(-lead)


def evaluate_future_placebo(df: pd.DataFrame, row: pd.Series, *, main_effect: float, used_covariates: Iterable[str]) -> dict:
    source = _as_str(row.get("source", row.get("treatment_col", "")))
    target = _as_str(row.get("target", row.get("outcome_col", "")))
    treatment = _as_str(row.get("treatment_col", source))
    outcome = _as_str(row.get("outcome_col", target))
    lag = int(_safe_float(row.get("lag", 0), 0) or 0)
    iid = _as_str(row.get("insight_id", "")) or f"{source}->{target}@{lag}"
    effect_id = _as_str(row.get("effect_id", "")) or f"effect::{iid}"
    base = {c: "" for c in PLACEBO_COLUMNS}
    base.update({
        "placebo_id": f"future_placebo::{iid}",
        "effect_id": effect_id,
        "insight_id": iid,
        "source": source,
        "target": target,
        "treatment_col": treatment,
        "outcome_col": outcome,
        "lag": lag,
        "placebo_type": "future_treatment_lead",
    })
    if treatment not in df.columns or outcome not in df.columns:
        base.update({"placebo_status": "not_evaluated_missing_columns", "placebo_pass": "", "reason": "MISSING_TREATMENT_OR_OUTCOME"})
        return base
    covs = [c for c in used_covariates if c not in {treatment, outcome}]
    Z, used = SC.numeric_matrix(df, covs, exclude={treatment, outcome}, standardize=True, min_non_null=8)
    a_s = _future_series(df, treatment, lag)
    y_s = pd.to_numeric(df[outcome], errors="coerce").astype(float)
    mask = np.isfinite(a_s.to_numpy(dtype=float)) & np.isfinite(y_s.to_numpy(dtype=float))
    if Z.size:
        mask = mask & np.all(np.isfinite(Z), axis=1)
    a = a_s.to_numpy(dtype=float)[mask]
    y = y_s.to_numpy(dtype=float)[mask]
    Zm = Z[mask] if Z.size else np.empty((int(np.sum(mask)), 0), dtype=float)
    n = len(y)
    base["support_n"] = n
    if n < max(20, 5 + len(used)):
        base.update({"placebo_status": "not_evaluated_weak_support", "placebo_pass": "", "reason": "INSUFFICIENT_SUPPORT"})
        return base
    if np.nanstd(a) < 1e-12 or np.nanstd(y) < 1e-12:
        base.update({"placebo_status": "not_evaluated_no_variation", "placebo_pass": "", "reason": "NO_VARIATION"})
        return base
    res = SC.linear_treatment_effect(a, y, Zm)
    eff, se, t_stat = res.effect, res.se, res.t_stat
    p_value = SC.normal_approx_p_from_t(t_stat)
    denom = max(abs(float(main_effect)) if np.isfinite(main_effect) else 0.0, 1e-9)
    ratio = abs(float(eff)) / denom if np.isfinite(eff) else np.nan
    passed = bool(np.isfinite(ratio) and ratio <= 0.70)
    base.update({
        "placebo_effect_estimate": eff,
        "placebo_standard_error": se,
        "placebo_t_stat": t_stat,
        "placebo_p_value_approx": p_value,
        "placebo_abs_ratio_to_main": ratio,
        "placebo_status": "pass" if passed else "fail_future_signal_too_large",
        "placebo_pass": int(passed),
        "reason": "" if passed else "FUTURE_PLACEBO_EFFECT_TOO_LARGE",
    })
    return base


def build_placebo_checks_from_effects(effects: pd.DataFrame | None) -> pd.DataFrame:
    if effects is None or len(effects) == 0:
        return pd.DataFrame(columns=PLACEBO_COLUMNS)
    rows = []
    for _, r in effects.iterrows():
        row = {c: r.get(c, "") for c in PLACEBO_COLUMNS}
        row["placebo_id"] = row.get("placebo_id") or f"future_placebo::{_as_str(r.get('insight_id', ''))}"
        row["placebo_type"] = row.get("placebo_type") or "future_treatment_lead"
        rows.append(row)
    return pd.DataFrame(rows, columns=PLACEBO_COLUMNS)


__all__ = ["PLACEBO_COLUMNS", "evaluate_future_placebo", "build_placebo_checks_from_effects"]
