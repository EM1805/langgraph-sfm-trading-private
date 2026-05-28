"""Estimation-owned negative-control diagnostics.

Discovery may suggest negative-control columns, but this module is the owner of
quantitative negative-control checks.  The checks are diagnostic review
artifacts only: they never upgrade a candidate into causal authority.
"""
from __future__ import annotations

from runtime_env import configure_scientific_runtime
configure_scientific_runtime()

from typing import Iterable, List

import numpy as np
import pandas as pd

from . import _utils as U
from . import common as C
from . import stat_core as SC

NEGATIVE_CONTROL_COLUMNS = [
    "check_id", "effect_id", "insight_id", "source", "target", "treatment_col", "outcome_col", "lag",
    "negative_control_col", "negative_control_status", "negative_control_pass",
    "negative_control_effect_estimate", "negative_control_standard_error", "negative_control_t_stat",
    "negative_control_p_value_approx", "negative_control_abs_ratio_to_main", "support_n", "reason",
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


def _split_cols(value) -> List[str]:
    text = _as_str(value)
    if not text:
        return []
    for sep in ["|", ",", ";"]:
        text = text.replace(sep, "|")
    out: List[str] = []
    for item in text.split("|"):
        c = item.strip()
        if c and c not in out:
            out.append(c)
    return out


def _lagged_series(df: pd.DataFrame, col: str, lag: int) -> pd.Series:
    s = pd.to_numeric(df[col], errors="coerce").astype(float)
    return s.shift(lag) if lag > 0 else s


def select_negative_control_col(df: pd.DataFrame, row: pd.Series) -> str:
    """Pick the first available negative-control outcome for this effect row."""
    treatment = _as_str(row.get("treatment_col", row.get("source", "")))
    outcome = _as_str(row.get("outcome_col", row.get("target", "")))
    candidates: List[str] = []
    candidates.extend(_split_cols(row.get("negative_controls", "")))
    candidates.extend(_split_cols(row.get("negative_control_hint", "")))
    configured = _as_str(getattr(C, "NEGCTRL_OUTCOME_COL", "negative_control_outcome"))
    if configured:
        candidates.append(configured)
    candidates.extend([c for c in df.columns if "negative" in c.lower() or "placebo" in c.lower()])
    for c in candidates:
        if c in df.columns and c not in {treatment, outcome}:
            return c
    return ""


def evaluate_negative_control(df: pd.DataFrame, row: pd.Series, *, main_effect: float, used_covariates: Iterable[str]) -> dict:
    """Evaluate treatment -> negative-control-outcome using the same lag/covariate frame."""
    source = _as_str(row.get("source", row.get("treatment_col", "")))
    target = _as_str(row.get("target", row.get("outcome_col", "")))
    treatment = _as_str(row.get("treatment_col", source))
    outcome = _as_str(row.get("outcome_col", target))
    lag = int(_safe_float(row.get("lag", 0), 0) or 0)
    iid = _as_str(row.get("insight_id", "")) or f"{source}->{target}@{lag}"
    effect_id = _as_str(row.get("effect_id", "")) or f"effect::{iid}"
    base = {c: "" for c in NEGATIVE_CONTROL_COLUMNS}
    base.update({
        "check_id": f"negative_control::{iid}",
        "effect_id": effect_id,
        "insight_id": iid,
        "source": source,
        "target": target,
        "treatment_col": treatment,
        "outcome_col": outcome,
        "lag": lag,
    })
    nc = select_negative_control_col(df, row)
    if not nc:
        base.update({"negative_control_status": "not_evaluated_no_negative_control", "negative_control_pass": "", "reason": "NO_NEGATIVE_CONTROL_COLUMN"})
        return base
    base["negative_control_col"] = nc
    if treatment not in df.columns or nc not in df.columns:
        base.update({"negative_control_status": "not_evaluated_missing_columns", "negative_control_pass": "", "reason": "MISSING_TREATMENT_OR_NEGATIVE_CONTROL"})
        return base
    covs = [c for c in used_covariates if c not in {treatment, outcome, nc}]
    Z, used = SC.numeric_matrix(df, covs, exclude={treatment, outcome, nc}, standardize=True, min_non_null=8)
    a_s = _lagged_series(df, treatment, lag)
    y_s = pd.to_numeric(df[nc], errors="coerce").astype(float)
    mask = np.isfinite(a_s.to_numpy(dtype=float)) & np.isfinite(y_s.to_numpy(dtype=float))
    if Z.size:
        mask = mask & np.all(np.isfinite(Z), axis=1)
    a = a_s.to_numpy(dtype=float)[mask]
    y = y_s.to_numpy(dtype=float)[mask]
    Zm = Z[mask] if Z.size else np.empty((int(np.sum(mask)), 0), dtype=float)
    n = len(y)
    base["support_n"] = n
    if n < max(20, 5 + len(used)):
        base.update({"negative_control_status": "not_evaluated_weak_support", "negative_control_pass": "", "reason": "INSUFFICIENT_SUPPORT"})
        return base
    if np.nanstd(a) < 1e-12 or np.nanstd(y) < 1e-12:
        base.update({"negative_control_status": "not_evaluated_no_variation", "negative_control_pass": "", "reason": "NO_VARIATION"})
        return base
    res = SC.linear_treatment_effect(a, y, Zm)
    eff, se, t_stat = res.effect, res.se, res.t_stat
    p_value = SC.normal_approx_p_from_t(t_stat)
    denom = max(abs(float(main_effect)) if np.isfinite(main_effect) else 0.0, 1e-9)
    ratio = abs(float(eff)) / denom if np.isfinite(eff) else np.nan
    passed = bool(np.isfinite(ratio) and ratio <= 0.50)
    base.update({
        "negative_control_effect_estimate": eff,
        "negative_control_standard_error": se,
        "negative_control_t_stat": t_stat,
        "negative_control_p_value_approx": p_value,
        "negative_control_abs_ratio_to_main": ratio,
        "negative_control_status": "pass" if passed else "fail_possible_spurious_effect",
        "negative_control_pass": int(passed),
        "reason": "" if passed else "NEGATIVE_CONTROL_EFFECT_TOO_LARGE",
    })
    return base


def build_negative_control_checks_from_effects(effects: pd.DataFrame | None) -> pd.DataFrame:
    if effects is None or len(effects) == 0:
        return pd.DataFrame(columns=NEGATIVE_CONTROL_COLUMNS)
    rows = []
    for _, r in effects.iterrows():
        rows.append({c: r.get(c, "") for c in NEGATIVE_CONTROL_COLUMNS})
        rows[-1]["check_id"] = rows[-1].get("check_id") or f"negative_control::{_as_str(r.get('insight_id', ''))}"
    return pd.DataFrame(rows, columns=NEGATIVE_CONTROL_COLUMNS)


__all__ = ["NEGATIVE_CONTROL_COLUMNS", "select_negative_control_col", "evaluate_negative_control", "build_negative_control_checks_from_effects"]
