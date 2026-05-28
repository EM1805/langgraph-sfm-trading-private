"""Conservative effect estimation for the Amantia estimation handoff.

This module is deliberately modest: it estimates diagnostic effects only for
rows that the contract/estimation plan marks as estimable or near-estimable.
It does not upgrade a structural prior into causal authority.  The output is a
review artifact consumed by the compact causal report.
"""
from __future__ import annotations

from runtime_env import configure_scientific_runtime
configure_scientific_runtime()


import math
import os
from typing import Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from . import common as C
from . import _utils as U
from . import stat_core as SC
from . import handoff_reader as HR
from . import negative_controls as NC
from . import placebo as PB
from . import estimator_registry as ER

OUT_DIR = C.OUT_DIR
EFFECT_ESTIMATES_CSV = os.path.join(OUT_DIR, "estimation", "effect_estimates.csv")
ROBUSTNESS_DIAGNOSTICS_CSV = os.path.join(OUT_DIR, "estimation", "robustness_diagnostics.csv")
SENSITIVITY_QUANT_CSV = os.path.join(OUT_DIR, "estimation", "sensitivity_quantitative.csv")

EFFECT_ESTIMATE_COLUMNS = [
    "effect_id", "plan_id", "insight_id", "source", "target", "treatment_col", "outcome_col", "lag",
    "estimand_type", "estimator_used", "effect_claim_status", "effect_estimate", "ci_low", "ci_high",
    "ci_level", "standard_error", "t_stat", "p_value_approx", "support_n", "treated_n", "control_n",
    "adjustment_set", "adjustment_set_size", "used_adjustment_set", "dropped_adjustment_set",
    "naive_effect_estimate", "adjusted_vs_naive_delta", "robustness_status", "drop_one_min_effect",
    "drop_one_max_effect", "drop_one_sign_stability", "partial_r2_treatment", "partial_r2_needed_to_explain_away",
    "sensitivity_quant_status", "negative_control_col", "negative_control_status", "negative_control_pass",
    "negative_control_effect_estimate", "negative_control_abs_ratio_to_main",
    "placebo_type", "placebo_status", "placebo_pass", "placebo_effect_estimate", "placebo_abs_ratio_to_main",
    "minimum_report_before_effect_claim", "authority_level", "identification_status",
    "estimation_status", "sensitivity_status", "reason_codes",
]

ROBUSTNESS_COLUMNS = [
    "effect_id", "insight_id", "source", "target", "lag", "robustness_status", "base_effect",
    "naive_effect", "drop_one_min_effect", "drop_one_max_effect", "drop_one_sign_stability",
    "ci_crosses_zero", "used_adjustment_set", "reason",
]

SENSITIVITY_QUANT_COLUMNS = [
    "effect_id", "insight_id", "source", "target", "lag", "partial_r2_treatment",
    "partial_r2_needed_to_explain_away", "unobserved_confounder_risk_band", "sensitivity_quant_status",
    "interpretation", "method",
]

_ESTIMABLE_STATUSES = {"can_estimate_now"}
_ESTIMABLE_AUTHORITIES = {"identified_estimable"}


def _as_str(value) -> str:
    text = U.as_str(value).strip()
    return "" if text.lower() in {"nan", "none", "null", "nat"} else text


def _boolish(value) -> bool:
    return _as_str(value).lower() in {"1", "true", "yes", "y", "on"}


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
    if lag > 0:
        return s.shift(lag)
    return s


def _numeric_covariates(df: pd.DataFrame, covs: Sequence[str], *, exclude: Set[str]) -> Tuple[pd.DataFrame, List[str], List[str]]:
    used: List[str] = []
    dropped: List[str] = []
    cols = []
    for c in covs:
        if c in exclude or c in used:
            dropped.append(c)
            continue
        if c not in df.columns:
            dropped.append(c)
            continue
        s = pd.to_numeric(df[c], errors="coerce").astype(float)
        if s.notna().sum() < 8 or float(np.nanstd(s.to_numpy(dtype=float))) < 1e-12:
            dropped.append(c)
            continue
        used.append(c)
        cols.append(s.rename(c))
    if not cols:
        return pd.DataFrame(index=df.index), used, dropped
    return pd.concat(cols, axis=1), used, dropped



def _standardize_matrix(Z: pd.DataFrame) -> np.ndarray:
    """Compatibility wrapper around estimation_parts.stat_core.standardize_matrix."""
    return SC.standardize_matrix(Z)


def _fit_ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compatibility wrapper around estimation_parts.stat_core.fit_ols."""
    return SC.fit_ols(X, y, ridge=SC.DEFAULT_RIDGE)


def _regression_effect(a: np.ndarray, y: np.ndarray, Z: np.ndarray) -> Tuple[float, float, float, float, int]:
    """Estimate y ~ 1 + treatment + covariates using the shared stat core.

    Contract/ID gates remain outside stat_core; this wrapper preserves the
    historical return shape used by effect_estimates.py.
    """
    result = SC.linear_treatment_effect(a, y, Z, ridge=SC.DEFAULT_RIDGE)
    p = float(math.erfc(abs(result.t_stat) / math.sqrt(2.0))) if np.isfinite(result.t_stat) else np.nan
    return result.effect, result.se, result.t_stat, p, result.n


def _bootstrap_ci(a: np.ndarray, y: np.ndarray, Z: np.ndarray, *, b: int = 160, seed: int = SC.DEFAULT_BOOTSTRAP_SEED) -> Tuple[float, float]:
    """Bootstrap CI wrapper using the shared stat core."""
    n = len(y)
    if n < 20:
        return np.nan, np.nan
    return SC.bootstrap_treatment_effect_ci(
        a,
        y,
        Z,
        b=max(80, int(b)),
        seed=seed,
        ridge=SC.DEFAULT_RIDGE,
        alpha=0.05,
    )

def _support_counts(a: np.ndarray) -> Tuple[int, int]:
    finite = a[np.isfinite(a)]
    if len(finite) == 0:
        return 0, 0
    if np.nanmin(finite) >= 0 and np.nanmax(finite) <= 1 and len(np.unique(finite)) <= 3:
        return int(np.sum(finite > 0.5)), int(np.sum(finite <= 0.5))
    med = float(np.nanmedian(finite))
    return int(np.sum(finite > med)), int(np.sum(finite <= med))


def _drop_one_effects(a: np.ndarray, y: np.ndarray, Z: np.ndarray) -> List[float]:
    if Z.size == 0 or Z.shape[1] == 0:
        return []
    vals: List[float] = []
    for j in range(Z.shape[1]):
        keep = [k for k in range(Z.shape[1]) if k != j]
        Zj = Z[:, keep] if keep else np.empty((len(y), 0), dtype=float)
        eff, _, _, _, _ = _regression_effect(a, y, Zj)
        if np.isfinite(eff):
            vals.append(float(eff))
    return vals



def _partial_r2_from_t(t_stat: float, df: int) -> float:
    """Compatibility wrapper around estimation_parts.stat_core.partial_r2_from_t."""
    return SC.partial_r2_from_t(t_stat, df)

def _sensitivity_band(r2: float) -> str:
    if not np.isfinite(r2):
        return "not_evaluated"
    if r2 >= 0.10:
        return "higher_resilience"
    if r2 >= 0.03:
        return "moderate_resilience"
    return "low_resilience"


def _row_estimable(row: pd.Series) -> bool:
    status = _as_str(row.get("estimation_status", "")).lower()
    authority = _as_str(row.get("authority_level", "")).lower()
    enabled = _boolish(row.get("estimation_enabled", ""))
    allowed_text = _as_str(row.get("allowed_for_estimation", ""))
    allowed = _boolish(allowed_text) if allowed_text else (authority == "identified_estimable" and enabled)
    identified_text = _as_str(row.get("identified", ""))
    identified = _boolish(identified_text) if identified_text else authority == "identified_estimable"
    id_status = _as_str(row.get("identification_status", "")).lower()
    if id_status in {"not_identified", "unidentified", "blocked", "blocked_id_algorithm", "simulable_not_identified"}:
        return False
    return bool(
        status in _ESTIMABLE_STATUSES
        and authority in _ESTIMABLE_AUTHORITIES
        and enabled
        and allowed
        and identified
    )


def _gate_block_reason(row: pd.Series) -> str:
    authority = _as_str(row.get("authority_level", "")).lower()
    id_status = _as_str(row.get("identification_status", "")).lower()
    identified_text = _as_str(row.get("identified", ""))
    identified = _boolish(identified_text) if identified_text else authority == "identified_estimable"
    enabled = _boolish(row.get("estimation_enabled", ""))
    allowed_text = _as_str(row.get("allowed_for_estimation", ""))
    allowed = _boolish(allowed_text) if allowed_text else (authority == "identified_estimable" and enabled)
    if id_status in {"not_identified", "unidentified", "blocked", "blocked_id_algorithm", "simulable_not_identified"} or not identified:
        return "NO_ESTIMATE_ID_NOT_IDENTIFIED"
    if authority != "identified_estimable":
        return "NO_ESTIMATE_NOT_IDENTIFIED_ESTIMABLE"
    if not enabled or not allowed:
        return "NO_ESTIMATE_CONTRACT_GATE_DISABLED"
    return "NOT_AUTHORIZED_FOR_EFFECT_ESTIMATION"


def _empty_effect_row(row: pd.Series, reason: str) -> dict:
    source = _as_str(row.get("source", row.get("treatment_col", "")))
    target = _as_str(row.get("target", row.get("outcome_col", "")))
    lag = _as_str(row.get("lag", ""))
    iid = _as_str(row.get("insight_id", "")) or f"{source}->{target}@{lag or '0'}"
    base = {c: "" for c in EFFECT_ESTIMATE_COLUMNS}
    base.update({
        "effect_id": f"effect::{iid}",
        "plan_id": _as_str(row.get("plan_id", "")),
        "insight_id": iid,
        "source": source,
        "target": target,
        "treatment_col": _as_str(row.get("treatment_col", source)),
        "outcome_col": _as_str(row.get("outcome_col", target)),
        "lag": lag,
        "estimand_type": _as_str(row.get("estimand_type", "")),
        "estimator_used": "none",
        "effect_claim_status": "not_estimated",
        "authority_level": _as_str(row.get("authority_level", "")),
        "identification_status": _as_str(row.get("identification_status", "")),
        "estimation_status": _as_str(row.get("estimation_status", "")),
        "sensitivity_status": _as_str(row.get("sensitivity_status", "")),
        "reason_codes": reason,
    })
    return base


def estimate_plan_row(df: pd.DataFrame, row: pd.Series, *, bootstrap_b: int = 160) -> dict:
    base = _empty_effect_row(row, "")
    if not _row_estimable(row):
        base["reason_codes"] = _gate_block_reason(row)
        base["effect_claim_status"] = "not_estimated_contract_gate"
        return base

    estimator_used, estimator_reason = ER.resolve_effect_estimator_for_row(
        row.to_dict() if hasattr(row, "to_dict") else row
    )
    if estimator_used in {"", "none", "skip"}:
        base["estimator_used"] = estimator_used or "none"
        base["effect_claim_status"] = "identified_but_unestimated"
        base["reason_codes"] = estimator_reason or "ESTIMATOR_NOT_RUNNABLE"
        return base

    treatment = _as_str(row.get("treatment_col", row.get("source", "")))
    outcome = _as_str(row.get("outcome_col", row.get("target", "")))
    lag = int(_safe_float(row.get("lag", 0), 0) or 0)
    if not treatment or treatment not in df.columns:
        base["reason_codes"] = "MISSING_TREATMENT_COLUMN"
        base["effect_claim_status"] = "identified_but_unestimated"
        return base
    if not outcome or outcome not in df.columns:
        base["reason_codes"] = "MISSING_OUTCOME_COLUMN"
        base["effect_claim_status"] = "identified_but_unestimated"
        return base

    covs = _split_cols(row.get("candidate_adjustment_set", "")) or _split_cols(row.get("adjustment_set", ""))
    covs = [c for c in covs if c not in {treatment, outcome}]
    Z_df, used_covs, dropped_covs = _numeric_covariates(df, covs, exclude={treatment, outcome})
    a_s = _lagged_series(df, treatment, lag)
    y_s = pd.to_numeric(df[outcome], errors="coerce").astype(float)
    Z = _standardize_matrix(Z_df)
    mask = np.isfinite(a_s.to_numpy(dtype=float)) & np.isfinite(y_s.to_numpy(dtype=float))
    if Z.size:
        mask = mask & np.all(np.isfinite(Z), axis=1)
    a = a_s.to_numpy(dtype=float)[mask]
    y = y_s.to_numpy(dtype=float)[mask]
    Zm = Z[mask] if Z.size else np.empty((int(np.sum(mask)), 0), dtype=float)
    n = len(y)
    treated_n, control_n = _support_counts(a)
    base.update({
        "estimator_used": estimator_used,
        "support_n": n,
        "treated_n": treated_n,
        "control_n": control_n,
        "adjustment_set": "|".join(covs),
        "adjustment_set_size": len(used_covs),
        "used_adjustment_set": "|".join(used_covs),
        "dropped_adjustment_set": "|".join(dropped_covs),
        "ci_level": "0.95",
    })
    if n < max(20, 5 + len(used_covs)):
        base["effect_claim_status"] = "identified_but_weak_support"
        base["reason_codes"] = "INSUFFICIENT_SUPPORT_FOR_EFFECT_ESTIMATE"
        return base
    if np.nanstd(a) < 1e-12 or np.nanstd(y) < 1e-12:
        base["effect_claim_status"] = "identified_but_unestimated"
        base["reason_codes"] = "NO_VARIATION_IN_TREATMENT_OR_OUTCOME"
        return base

    effect, se, t_stat, p_value, _ = _regression_effect(a, y, Zm)
    naive, _, _, _, _ = _regression_effect(a, y, np.empty((n, 0), dtype=float))
    ci_low, ci_high = _bootstrap_ci(a, y, Zm, b=bootstrap_b)
    drop_vals = _drop_one_effects(a, y, Zm)
    r2 = _partial_r2_from_t(t_stat, max(1, n - (2 + len(used_covs))))
    ci_crosses = bool(np.isfinite(ci_low) and np.isfinite(ci_high) and ci_low <= 0 <= ci_high)
    sign_stability = "not_applicable_no_covariates"
    robustness_status = "no_adjustment_covariates"
    if drop_vals:
        same_sign = [np.sign(v) == np.sign(effect) for v in drop_vals if np.isfinite(v) and np.isfinite(effect) and effect != 0]
        share = float(np.mean(same_sign)) if same_sign else np.nan
        sign_stability = "stable" if np.isfinite(share) and share >= 0.80 else "fragile"
        robustness_status = "robust_direction" if sign_stability == "stable" and not ci_crosses else "fragile_or_uncertain"
    elif ci_crosses:
        robustness_status = "uncertain_ci_crosses_zero"

    sensitivity_band = _sensitivity_band(r2)
    sens_status = "quantitative_sensitivity_required" if sensitivity_band in {"low_resilience", "moderate_resilience"} else "quantitative_sensitivity_recommended"
    claim = "diagnostic_effect_estimate"
    if _as_str(row.get("sensitivity_status", "")).startswith("required"):
        claim = "estimated_but_sensitivity_required"
    if ci_crosses:
        claim = "estimated_but_uncertain_ci_crosses_zero"

    base.update({
        "effect_claim_status": claim,
        "effect_estimate": effect,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "standard_error": se,
        "t_stat": t_stat,
        "p_value_approx": p_value,
        "naive_effect_estimate": naive,
        "adjusted_vs_naive_delta": effect - naive if np.isfinite(effect) and np.isfinite(naive) else np.nan,
        "robustness_status": robustness_status,
        "drop_one_min_effect": float(np.nanmin(drop_vals)) if drop_vals else "",
        "drop_one_max_effect": float(np.nanmax(drop_vals)) if drop_vals else "",
        "drop_one_sign_stability": sign_stability,
        "partial_r2_treatment": r2,
        "partial_r2_needed_to_explain_away": r2,
        "sensitivity_quant_status": sens_status,
        "minimum_report_before_effect_claim": "required" if "required" in sens_status or "required" in claim else "recommended",
        "reason_codes": "CI_CROSSES_ZERO" if ci_crosses else "",
    })

    # Step: falsification checks are now owned by Estimation, not Discovery.
    # They are diagnostic gates only and do not grant causal authority.
    diagnostic_row = pd.Series({**row.to_dict(), **base})
    try:
        nc = NC.evaluate_negative_control(df, diagnostic_row, main_effect=effect, used_covariates=used_covs)
        base.update({
            "negative_control_col": _as_str(nc.get("negative_control_col", "")),
            "negative_control_status": _as_str(nc.get("negative_control_status", "")),
            "negative_control_pass": _as_str(nc.get("negative_control_pass", "")),
            "negative_control_effect_estimate": nc.get("negative_control_effect_estimate", ""),
            "negative_control_abs_ratio_to_main": nc.get("negative_control_abs_ratio_to_main", ""),
        })
    except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError) as exc:
        base.update({"negative_control_status": "not_evaluated_error", "negative_control_pass": "", "reason_codes": (base.get("reason_codes") or "") + ("|" if base.get("reason_codes") else "") + f"NEGATIVE_CONTROL_ERROR:{type(exc).__name__}"})

    try:
        pl = PB.evaluate_future_placebo(df, diagnostic_row, main_effect=effect, used_covariates=used_covs)
        base.update({
            "placebo_type": _as_str(pl.get("placebo_type", "")),
            "placebo_status": _as_str(pl.get("placebo_status", "")),
            "placebo_pass": _as_str(pl.get("placebo_pass", "")),
            "placebo_effect_estimate": pl.get("placebo_effect_estimate", ""),
            "placebo_abs_ratio_to_main": pl.get("placebo_abs_ratio_to_main", ""),
        })
    except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError) as exc:
        base.update({"placebo_status": "not_evaluated_error", "placebo_pass": "", "reason_codes": (base.get("reason_codes") or "") + ("|" if base.get("reason_codes") else "") + f"PLACEBO_ERROR:{type(exc).__name__}"})
    return base


def build_effect_estimates(data: Optional[pd.DataFrame], plan: Optional[pd.DataFrame], *, bootstrap_b: int = 160) -> pd.DataFrame:
    if data is None or plan is None or len(plan) == 0:
        return pd.DataFrame(columns=EFFECT_ESTIMATE_COLUMNS)
    rows = [estimate_plan_row(data, row, bootstrap_b=bootstrap_b) for _, row in plan.iterrows()]
    out = pd.DataFrame(rows)
    for c in EFFECT_ESTIMATE_COLUMNS:
        if c not in out.columns:
            out[c] = ""
    return out[EFFECT_ESTIMATE_COLUMNS].copy()


def build_robustness_diagnostics(effects: Optional[pd.DataFrame]) -> pd.DataFrame:
    if effects is None or len(effects) == 0:
        return pd.DataFrame(columns=ROBUSTNESS_COLUMNS)
    rows = []
    for _, r in effects.iterrows():
        rows.append({
            "effect_id": _as_str(r.get("effect_id", "")),
            "insight_id": _as_str(r.get("insight_id", "")),
            "source": _as_str(r.get("source", "")),
            "target": _as_str(r.get("target", "")),
            "lag": _as_str(r.get("lag", "")),
            "robustness_status": _as_str(r.get("robustness_status", "")),
            "base_effect": _as_str(r.get("effect_estimate", "")),
            "naive_effect": _as_str(r.get("naive_effect_estimate", "")),
            "drop_one_min_effect": _as_str(r.get("drop_one_min_effect", "")),
            "drop_one_max_effect": _as_str(r.get("drop_one_max_effect", "")),
            "drop_one_sign_stability": _as_str(r.get("drop_one_sign_stability", "")),
            "ci_crosses_zero": "1" if _as_str(r.get("reason_codes", "")) == "CI_CROSSES_ZERO" else "0",
            "used_adjustment_set": _as_str(r.get("used_adjustment_set", "")),
            "reason": _as_str(r.get("reason_codes", "")),
        })
    return pd.DataFrame(rows, columns=ROBUSTNESS_COLUMNS)


def build_quantitative_sensitivity(effects: Optional[pd.DataFrame]) -> pd.DataFrame:
    if effects is None or len(effects) == 0:
        return pd.DataFrame(columns=SENSITIVITY_QUANT_COLUMNS)
    rows = []
    for _, r in effects.iterrows():
        r2 = _safe_float(r.get("partial_r2_treatment", np.nan))
        band = _sensitivity_band(r2)
        rows.append({
            "effect_id": _as_str(r.get("effect_id", "")),
            "insight_id": _as_str(r.get("insight_id", "")),
            "source": _as_str(r.get("source", "")),
            "target": _as_str(r.get("target", "")),
            "lag": _as_str(r.get("lag", "")),
            "partial_r2_treatment": _as_str(r.get("partial_r2_treatment", "")),
            "partial_r2_needed_to_explain_away": _as_str(r.get("partial_r2_needed_to_explain_away", "")),
            "unobserved_confounder_risk_band": band,
            "sensitivity_quant_status": _as_str(r.get("sensitivity_quant_status", "")),
            "interpretation": "An unobserved confounder with comparable partial-R2 to the treatment could materially change this diagnostic estimate." if band else "",
            "method": "partial_r2_from_t_stat_diagnostic",
        })
    return pd.DataFrame(rows, columns=SENSITIVITY_QUANT_COLUMNS)


def write_effect_estimates_from_frames(data: Optional[pd.DataFrame], plan: Optional[pd.DataFrame], out_dir: str = OUT_DIR, *, bootstrap_b: int = 160) -> dict:
    os.makedirs(os.path.join(out_dir, "estimation"), exist_ok=True)
    effects = build_effect_estimates(data, plan, bootstrap_b=bootstrap_b)
    robustness = build_robustness_diagnostics(effects)
    quant = build_quantitative_sensitivity(effects)
    negative_checks = NC.build_negative_control_checks_from_effects(effects)
    placebo_checks = PB.build_placebo_checks_from_effects(effects)
    effect_path = os.path.join(out_dir, "estimation", "effect_estimates.csv")
    robustness_path = os.path.join(out_dir, "estimation", "robustness_diagnostics.csv")
    quant_path = os.path.join(out_dir, "estimation", "sensitivity_quantitative.csv")
    negative_path = os.path.join(out_dir, "estimation", "negative_control_checks.csv")
    placebo_path = os.path.join(out_dir, "estimation", "placebo_checks.csv")
    effects.to_csv(effect_path, index=False)
    robustness.to_csv(robustness_path, index=False)
    quant.to_csv(quant_path, index=False)
    negative_checks.to_csv(negative_path, index=False)
    placebo_checks.to_csv(placebo_path, index=False)
    return {
        "effect_estimates_csv": effect_path,
        "robustness_diagnostics_csv": robustness_path,
        "sensitivity_quantitative_csv": quant_path,
        "negative_control_checks_csv": negative_path,
        "placebo_checks_csv": placebo_path,
        "effect_estimate_rows": int(len(effects)),
    }


def write_effect_estimates(data_path: Optional[str] = None, out_dir: str = OUT_DIR, *, plan: Optional[pd.DataFrame] = None, bootstrap_b: int = 160) -> dict:
    os.makedirs(os.path.join(out_dir, "estimation"), exist_ok=True)
    plan_path = os.path.join(out_dir, "estimation", "estimation_plan.csv")
    if plan is None:
        if os.path.exists(plan_path):
            try:
                plan = pd.read_csv(plan_path)
            except (OSError, ValueError, TypeError, pd.errors.ParserError):
                plan = pd.DataFrame(columns=HR.PLAN_COLUMNS)
        else:
            plan = pd.DataFrame(columns=HR.PLAN_COLUMNS)
    data = None
    if data_path and os.path.exists(data_path):
        try:
            data = pd.read_csv(data_path)
        except (OSError, ValueError, TypeError, pd.errors.ParserError):
            data = None
    effects = build_effect_estimates(data, plan, bootstrap_b=bootstrap_b)
    robustness = build_robustness_diagnostics(effects)
    quant = build_quantitative_sensitivity(effects)
    negative_checks = NC.build_negative_control_checks_from_effects(effects)
    placebo_checks = PB.build_placebo_checks_from_effects(effects)
    effect_path = os.path.join(out_dir, "estimation", "effect_estimates.csv")
    robustness_path = os.path.join(out_dir, "estimation", "robustness_diagnostics.csv")
    quant_path = os.path.join(out_dir, "estimation", "sensitivity_quantitative.csv")
    negative_path = os.path.join(out_dir, "estimation", "negative_control_checks.csv")
    placebo_path = os.path.join(out_dir, "estimation", "placebo_checks.csv")
    effects.to_csv(effect_path, index=False)
    robustness.to_csv(robustness_path, index=False)
    quant.to_csv(quant_path, index=False)
    negative_checks.to_csv(negative_path, index=False)
    placebo_checks.to_csv(placebo_path, index=False)
    return {
        "effect_estimates_csv": effect_path,
        "robustness_diagnostics_csv": robustness_path,
        "sensitivity_quantitative_csv": quant_path,
        "negative_control_checks_csv": negative_path,
        "placebo_checks_csv": placebo_path,
        "effect_estimate_rows": int(len(effects)),
    }


__all__ = [
    "EFFECT_ESTIMATE_COLUMNS", "ROBUSTNESS_COLUMNS", "SENSITIVITY_QUANT_COLUMNS", "NC", "PB",
    "build_effect_estimates", "write_effect_estimates", "write_effect_estimates_from_frames",
]
