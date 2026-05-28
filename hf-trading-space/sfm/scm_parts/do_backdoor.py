from __future__ import annotations

from runtime_env import configure_scientific_runtime
configure_scientific_runtime()


"""Backdoor-adjusted strong do-estimator.

This is the first conservative ``do`` estimator in Amantia. It estimates
``E[Y | do(X=x)]`` only when :mod:`scm_parts.do_contract` authorizes the
query from ``out/causal_contract.csv``.

The estimator uses a simple outcome regression plus standardization over the
empirical distribution of the adjustment set. It intentionally remains small
and dependency-light (numpy/pandas only).
"""

import csv
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .do_contract import DoAuthorization, authorize_all_backdoor_do, authorize_do, load_causal_contract

SCM_DIRNAME = "scm"
DO_ESTIMATE_COLUMNS = [
    "effect_id",
    "treatment",
    "outcome",
    "do_value_low",
    "do_value_high",
    "estimand",
    "identification_strategy",
    "adjustment_set",
    "adjustment_set_status",
    "do_authorized",
    "do_mode",
    "effect_estimate",
    "mean_do_low",
    "mean_do_high",
    "ci_low",
    "ci_high",
    "n",
    "support_n_low",
    "support_n_high",
    "authority_level",
    "effect_semantics",
    "analysis_policy",
    "diagnostic_estimation_allowed",
    "diagnostic_authority_level",
    "causal_authority_from_diagnostic",
    "reason_codes",
    "bootstrap_draws",
    "bootstrap_success_n",
    "bootstrap_status",
    "ci_width",
    "ci_width_to_effect_ratio",
    "robustness_status",
    "overlap_score",
    "support_min",
    "support_ratio",
    "support_n_mid",
    "treatment_unique_n",
    "extrapolation_risk",
    "sensitivity_warning",
]

DO_DIAGNOSTIC_COLUMNS = [
    "effect_id",
    "treatment",
    "outcome",
    "contract_row_present",
    "estimation_enabled",
    "do_authorized",
    "do_mode",
    "overlap_pass",
    "support_n_low",
    "support_n_high",
    "adjustment_set_status",
    "adjustment_columns_missing",
    "data_columns_missing",
    "analysis_policy",
    "diagnostic_estimation_allowed",
    "diagnostic_authority_level",
    "causal_authority_from_diagnostic",
    "diagnostic_notes",
    "bootstrap_draws",
    "bootstrap_success_n",
    "bootstrap_status",
    "ci_width",
    "ci_width_to_effect_ratio",
    "robustness_status",
    "overlap_score",
    "support_min",
    "support_ratio",
    "support_n_mid",
    "treatment_unique_n",
    "extrapolation_risk",
    "sensitivity_warning",
]


def _norm(value: object) -> str:
    s = str(value or "").strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s


def parse_adjustment_set(value: object) -> List[str]:
    s = _norm(value)
    if not s or s in {"[]", "{}"}:
        return []
    # Handle simple JSON-ish lists without importing ast for safety surprises.
    for ch in "[]{}'\"":
        s = s.replace(ch, "")
    parts: List[str] = []
    for token in s.replace(",", "|").split("|"):
        item = token.strip()
        if item and item not in parts:
            parts.append(item)
    return parts


def _load_data(data_path: Optional[str] = None, out_dir: str = "out") -> pd.DataFrame:
    candidates = [data_path, os.path.join(out_dir, "data_clean.csv"), "data.csv", os.path.join(out_dir, "demo_data.csv")]
    for path in candidates:
        if path and os.path.exists(path):
            try:
                return pd.read_csv(path)
            except (TypeError, ValueError, OverflowError, np.linalg.LinAlgError):
                continue
    return pd.DataFrame()


def _numeric_frame(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in columns:
        out[col] = pd.to_numeric(df[col], errors="coerce")
    return out


def _treatment_values(series: pd.Series) -> Tuple[float, float]:
    x = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if x.empty:
        return (np.nan, np.nan)
    unique = sorted(set(float(v) for v in x.tolist()))
    if len(unique) == 1:
        return (unique[0], unique[0])
    if len(unique) <= 5:
        return (float(unique[0]), float(unique[-1]))
    return (float(x.quantile(0.25)), float(x.quantile(0.75)))


def _fit_linear_outcome(df: pd.DataFrame, treatment: str, outcome: str, covariates: Sequence[str]) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    columns = [treatment] + list(covariates) + [outcome]
    work = _numeric_frame(df, columns).dropna()
    if work.empty:
        return np.array([]), [], work
    feature_cols = [treatment] + list(covariates)
    X = work[feature_cols].to_numpy(dtype=float)
    y = work[outcome].to_numpy(dtype=float)
    X_design = np.column_stack([np.ones(len(X)), X])
    # Tiny ridge term for numerical stability, with intercept unpenalized.
    ridge = np.eye(X_design.shape[1]) * 1e-8
    ridge[0, 0] = 0.0
    try:
        beta = np.linalg.solve(X_design.T @ X_design + ridge, X_design.T @ y)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(X_design) @ y
    return beta, feature_cols, work


def _standardized_mean(beta: np.ndarray, work: pd.DataFrame, feature_cols: Sequence[str], treatment: str, value: float) -> float:
    if beta.size == 0 or work.empty:
        return float("nan")
    features = work[list(feature_cols)].copy()
    features[treatment] = float(value)
    X = features.to_numpy(dtype=float)
    X_design = np.column_stack([np.ones(len(X)), X])
    return float(np.mean(X_design @ beta))


def _bootstrap_ci(df: pd.DataFrame, treatment: str, outcome: str, covariates: Sequence[str], low: float, high: float, draws: int = 80) -> Tuple[float, float]:
    if df.empty or len(df) < 8:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(1805)
    vals: List[float] = []
    n = len(df)
    for _ in range(int(draws)):
        idx = rng.integers(0, n, size=n)
        sample = df.iloc[idx].reset_index(drop=True)
        beta, feature_cols, work = _fit_linear_outcome(sample, treatment, outcome, covariates)
        if beta.size == 0 or work.empty:
            continue
        lo = _standardized_mean(beta, work, feature_cols, treatment, low)
        hi = _standardized_mean(beta, work, feature_cols, treatment, high)
        if np.isfinite(hi - lo):
            vals.append(float(hi - lo))
    if len(vals) < 5:
        return (float("nan"), float("nan"))
    return (float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975)))


def estimate_backdoor_do(treatment: str, outcome: str, data: pd.DataFrame, authorization: DoAuthorization, *, low_value: Optional[float] = None, high_value: Optional[float] = None, bootstrap_draws: int = 80) -> Tuple[Dict[str, object], Dict[str, object]]:
    effect_id = f"do:{treatment}->{outcome}"
    base_diag = {
        "effect_id": effect_id,
        "treatment": treatment,
        "outcome": outcome,
        "contract_row_present": authorization.contract_row_present,
        "estimation_enabled": authorization.estimation_enabled,
        "do_authorized": int(bool(authorization.do_authorized)),
        "do_mode": authorization.do_mode,
        "adjustment_set_status": authorization.adjustment_set_status,
        "analysis_policy": getattr(authorization, "analysis_policy", "balanced"),
        "diagnostic_estimation_allowed": getattr(authorization, "diagnostic_estimation_allowed", 0),
        "diagnostic_authority_level": getattr(authorization, "diagnostic_authority_level", ""),
        "causal_authority_from_diagnostic": getattr(authorization, "causal_authority_from_diagnostic", 0),
    }
    if not authorization.do_authorized and not getattr(authorization, "diagnostic_estimation_allowed", 0):
        row = {
            "effect_id": effect_id,
            "treatment": treatment,
            "outcome": outcome,
            "do_authorized": 0,
            "do_mode": authorization.do_mode,
            "effect_semantics": "blocked_not_contract_authorized",
            "reason_codes": authorization.reason_codes,
            "authority_level": authorization.authority_level,
        }
        diag = dict(base_diag, overlap_pass=0, support_n_low=0, support_n_high=0, adjustment_columns_missing="", data_columns_missing="", diagnostic_notes=authorization.reason_codes)
        return row, diag

    if data is None or data.empty:
        row = {
            "effect_id": effect_id,
            "treatment": treatment,
            "outcome": outcome,
            "do_authorized": 0,
            "do_mode": "blocked",
            "effect_semantics": "blocked_missing_data",
            "reason_codes": "MISSING_DATA",
            "authority_level": authorization.authority_level,
        }
        diag = dict(base_diag, overlap_pass=0, support_n_low=0, support_n_high=0, adjustment_columns_missing="", data_columns_missing="MISSING_DATA", diagnostic_notes="MISSING_DATA")
        return row, diag

    covariates = parse_adjustment_set(authorization.adjustment_set)
    missing_data_cols = [c for c in [treatment, outcome] if c not in data.columns]
    missing_covars = [c for c in covariates if c not in data.columns]
    covariates = [c for c in covariates if c in data.columns and c not in {treatment, outcome}]
    if missing_data_cols:
        row = {
            "effect_id": effect_id,
            "treatment": treatment,
            "outcome": outcome,
            "do_authorized": 0,
            "do_mode": "blocked",
            "effect_semantics": "blocked_missing_required_columns",
            "reason_codes": "MISSING_REQUIRED_COLUMNS",
            "authority_level": authorization.authority_level,
        }
        diag = dict(base_diag, overlap_pass=0, support_n_low=0, support_n_high=0, adjustment_columns_missing="|".join(missing_covars), data_columns_missing="|".join(missing_data_cols), diagnostic_notes="MISSING_REQUIRED_COLUMNS")
        return row, diag

    low, high = _treatment_values(data[treatment])
    if low_value is not None:
        low = float(low_value)
    if high_value is not None:
        high = float(high_value)
    support = pd.to_numeric(data[treatment], errors="coerce").dropna()
    if not np.isfinite(low) or not np.isfinite(high):
        support_low = support_high = 0
    else:
        sd = float(support.std()) if len(support) > 1 else 0.0
        tol = max(sd * 0.15, 1e-9)
        support_low = int((abs(support - low) <= tol).sum()) if len(set(support.round(8))) > 5 else int((support == low).sum())
        support_high = int((abs(support - high) <= tol).sum()) if len(set(support.round(8))) > 5 else int((support == high).sum())

    beta, feature_cols, work = _fit_linear_outcome(data, treatment, outcome, covariates)
    mean_low = _standardized_mean(beta, work, feature_cols, treatment, low)
    mean_high = _standardized_mean(beta, work, feature_cols, treatment, high)
    effect = float(mean_high - mean_low) if np.isfinite(mean_high - mean_low) else float("nan")
    ci_low, ci_high = _bootstrap_ci(work, treatment, outcome, covariates, low, high, draws=bootstrap_draws)
    overlap_pass = int(support_low > 0 and support_high > 0 and len(work) >= max(8, len(covariates) + 3))
    reason_codes = [authorization.reason_codes]
    if missing_covars:
        reason_codes.append("ADJUSTMENT_COLUMNS_DROPPED_NOT_IN_DATA")
    if not overlap_pass:
        reason_codes.append("WEAK_TREATMENT_SUPPORT")

    row = {
        "effect_id": effect_id,
        "treatment": treatment,
        "outcome": outcome,
        "do_value_low": low,
        "do_value_high": high,
        "estimand": "E[Y|do(X=high)] - E[Y|do(X=low)]",
        "identification_strategy": authorization.identification_strategy or "backdoor",
        "adjustment_set": "|".join(covariates),
        "adjustment_set_status": authorization.adjustment_set_status,
        "do_authorized": int(bool(authorization.do_authorized)),
        "do_mode": authorization.do_mode,
        "effect_estimate": effect,
        "mean_do_low": mean_low,
        "mean_do_high": mean_high,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n": int(len(work)),
        "support_n_low": support_low,
        "support_n_high": support_high,
        "authority_level": authorization.authority_level,
        "effect_semantics": ("backdoor_adjusted_do_estimand_contract_authorized" if authorization.do_authorized else "diagnostic_backdoor_estimate_not_causal_authority"),
        "analysis_policy": getattr(authorization, "analysis_policy", "balanced"),
        "diagnostic_estimation_allowed": getattr(authorization, "diagnostic_estimation_allowed", 0),
        "diagnostic_authority_level": getattr(authorization, "diagnostic_authority_level", ""),
        "causal_authority_from_diagnostic": getattr(authorization, "causal_authority_from_diagnostic", 0),
        "reason_codes": "|".join([r for r in reason_codes if r]),
    }
    diag = dict(
        base_diag,
        overlap_pass=overlap_pass,
        support_n_low=support_low,
        support_n_high=support_high,
        adjustment_columns_missing="|".join(missing_covars),
        data_columns_missing="",
        diagnostic_notes=row["reason_codes"],
    )
    return row, diag


def _write_csv(path: str, rows: Iterable[Dict[str, object]], columns: Sequence[str]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})


def estimate_authorized_do_effects(out_dir: str = "out", data_path: Optional[str] = None, contract_path: Optional[str] = None, bootstrap_draws: int = 80, policy: object = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    contract = load_causal_contract(out_dir, contract_path)
    data = _load_data(data_path=data_path, out_dir=out_dir)
    rows: List[Dict[str, object]] = []
    diagnostics: List[Dict[str, object]] = []
    for auth in authorize_all_backdoor_do(contract, policy=policy, include_diagnostic=True):
        row, diag = estimate_backdoor_do(auth.treatment, auth.outcome, data, auth, bootstrap_draws=bootstrap_draws)
        rows.append(row)
        diagnostics.append(diag)
    estimates = pd.DataFrame(rows, columns=DO_ESTIMATE_COLUMNS)
    diags = pd.DataFrame(diagnostics, columns=DO_DIAGNOSTIC_COLUMNS)
    return estimates, diags


def write_do_outputs(out_dir: str = "out", data_path: Optional[str] = None, contract_path: Optional[str] = None, bootstrap_draws: int = 80, policy: object = None) -> Dict[str, str]:
    estimates, diagnostics = estimate_authorized_do_effects(out_dir=out_dir, data_path=data_path, contract_path=contract_path, bootstrap_draws=bootstrap_draws, policy=policy)
    scm_dir = os.path.join(out_dir, SCM_DIRNAME)
    estimates_path = os.path.join(scm_dir, "do_estimates.csv")
    diagnostics_path = os.path.join(scm_dir, "do_diagnostics.csv")
    _write_csv(estimates_path, estimates.to_dict("records"), DO_ESTIMATE_COLUMNS)
    _write_csv(diagnostics_path, diagnostics.to_dict("records"), DO_DIAGNOSTIC_COLUMNS)
    return {"do_estimates": estimates_path, "do_diagnostics": diagnostics_path}


__all__ = [
    "DO_ESTIMATE_COLUMNS",
    "DO_DIAGNOSTIC_COLUMNS",
    "parse_adjustment_set",
    "estimate_backdoor_do",
    "estimate_authorized_do_effects",
    "write_do_outputs",
]
