from __future__ import annotations

from runtime_env import configure_scientific_runtime
configure_scientific_runtime()


"""Limited front-door strong do-estimator.

This module implements a conservative, regression-based estimator for the
classical front-door formula. It is intentionally limited:
- only runs when the causal contract authorizes a ``frontdoor`` route;
- requires observed mediator columns in the dataset;
- estimates a mean contrast, not a general symbolic ID formula;
- remains separate from the removed legacy diagnostic SCM simulation.
"""

import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .do_contract import DoAuthorization, authorize_all_frontdoor_do, load_causal_contract
from .do_backdoor import (
    DO_DIAGNOSTIC_COLUMNS,
    DO_ESTIMATE_COLUMNS,
    _load_data,
    _numeric_frame,
    _treatment_values,
    _write_csv,
    parse_adjustment_set,
)


def parse_mediators(value: object) -> List[str]:
    return parse_adjustment_set(value)


def _fit_linear(df: pd.DataFrame, feature_cols: Sequence[str], outcome: str) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    cols = list(feature_cols) + [outcome]
    work = _numeric_frame(df, cols).dropna()
    if work.empty:
        return np.array([]), list(feature_cols), work
    X = work[list(feature_cols)].to_numpy(dtype=float)
    y = work[outcome].to_numpy(dtype=float)
    X_design = np.column_stack([np.ones(len(X)), X])
    ridge = np.eye(X_design.shape[1]) * 1e-8
    ridge[0, 0] = 0.0
    try:
        beta = np.linalg.solve(X_design.T @ X_design + ridge, X_design.T @ y)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(X_design) @ y
    return beta, list(feature_cols), work


def _predict_linear(beta: np.ndarray, frame: pd.DataFrame, feature_cols: Sequence[str]) -> np.ndarray:
    if beta.size == 0 or frame.empty:
        return np.array([], dtype=float)
    X = frame[list(feature_cols)].to_numpy(dtype=float)
    X_design = np.column_stack([np.ones(len(X)), X])
    return X_design @ beta


def _frontdoor_mean(data: pd.DataFrame, treatment: str, outcome: str, mediators: Sequence[str], x_value: float) -> Tuple[float, int, str]:
    """Approximate E[Y|do(X=x)] using a front-door standardization proxy.

    For each mediator Z, fit Z ~ X and impute Z(x). Then fit Y ~ X + Z and
    average over the empirical distribution of X while holding Z at Z(x).
    This is a compact numeric analogue of sum_z P(z|x) sum_x' E[Y|x',z]P(x').
    """
    needed = [treatment, outcome] + list(mediators)
    work = _numeric_frame(data, needed).dropna()
    if work.empty:
        return float("nan"), 0, "EMPTY_NUMERIC_FRAME"

    z_at_x: Dict[str, float] = {}
    notes: List[str] = []
    for z in mediators:
        beta_z, feat_z, work_z = _fit_linear(work, [treatment], z)
        if beta_z.size == 0 or work_z.empty:
            return float("nan"), 0, f"MEDIATOR_MODEL_FAILED:{z}"
        pred_frame = pd.DataFrame({treatment: [float(x_value)]})
        pred = _predict_linear(beta_z, pred_frame, feat_z)
        if pred.size == 0 or not np.isfinite(pred[0]):
            return float("nan"), 0, f"MEDIATOR_PREDICTION_FAILED:{z}"
        z_at_x[z] = float(pred[0])

    y_features = [treatment] + list(mediators)
    beta_y, feat_y, work_y = _fit_linear(work, y_features, outcome)
    if beta_y.size == 0 or work_y.empty:
        return float("nan"), 0, "OUTCOME_MODEL_FAILED"

    pred_frame = work_y[y_features].copy()
    # Front-door standardization averages over empirical X distribution and
    # inserts mediator values generated under do(X=x_value).
    for z, val in z_at_x.items():
        pred_frame[z] = val
    pred = _predict_linear(beta_y, pred_frame, feat_y)
    if pred.size == 0:
        return float("nan"), 0, "PREDICTION_FAILED"
    return float(np.mean(pred)), int(len(work_y)), "|".join(notes)


def _bootstrap_ci(data: pd.DataFrame, treatment: str, outcome: str, mediators: Sequence[str], low: float, high: float, draws: int = 80) -> Tuple[float, float]:
    work = _numeric_frame(data, [treatment, outcome] + list(mediators)).dropna()
    if work.empty or len(work) < 10:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(1805)
    vals: List[float] = []
    n = len(work)
    for _ in range(int(draws)):
        idx = rng.integers(0, n, size=n)
        sample = work.iloc[idx].reset_index(drop=True)
        mean_low, _, note_low = _frontdoor_mean(sample, treatment, outcome, mediators, low)
        mean_high, _, note_high = _frontdoor_mean(sample, treatment, outcome, mediators, high)
        if not note_low and not note_high and np.isfinite(mean_high - mean_low):
            vals.append(float(mean_high - mean_low))
    if len(vals) < 5:
        return (float("nan"), float("nan"))
    return (float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975)))


def estimate_frontdoor_do(treatment: str, outcome: str, data: pd.DataFrame, authorization: DoAuthorization, *, low_value: Optional[float] = None, high_value: Optional[float] = None, bootstrap_draws: int = 80) -> Tuple[Dict[str, object], Dict[str, object]]:
    effect_id = f"do_frontdoor:{treatment}->{outcome}"
    mediators = parse_mediators(getattr(authorization, "mediators", ""))
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
            "effect_semantics": "blocked_not_frontdoor_contract_authorized",
            "reason_codes": authorization.reason_codes,
            "authority_level": authorization.authority_level,
        }
        diag = dict(base_diag, overlap_pass=0, support_n_low=0, support_n_high=0, adjustment_columns_missing="", data_columns_missing="", diagnostic_notes=authorization.reason_codes)
        return row, diag

    if data is None or data.empty:
        row = {"effect_id": effect_id, "treatment": treatment, "outcome": outcome, "do_authorized": 0, "do_mode": "blocked", "effect_semantics": "blocked_missing_data", "reason_codes": "MISSING_DATA", "authority_level": authorization.authority_level}
        diag = dict(base_diag, overlap_pass=0, support_n_low=0, support_n_high=0, adjustment_columns_missing="", data_columns_missing="MISSING_DATA", diagnostic_notes="MISSING_DATA")
        return row, diag

    missing = [c for c in [treatment, outcome] + mediators if c not in data.columns]
    if missing or not mediators:
        reason = "MISSING_MEDIATORS" if not mediators else "MISSING_REQUIRED_COLUMNS"
        row = {"effect_id": effect_id, "treatment": treatment, "outcome": outcome, "do_authorized": 0, "do_mode": "blocked", "effect_semantics": "blocked_frontdoor_missing_columns", "reason_codes": reason, "authority_level": authorization.authority_level}
        diag = dict(base_diag, overlap_pass=0, support_n_low=0, support_n_high=0, adjustment_columns_missing="", data_columns_missing="|".join(missing), diagnostic_notes=reason)
        return row, diag

    low, high = _treatment_values(data[treatment])
    if low_value is not None:
        low = float(low_value)
    if high_value is not None:
        high = float(high_value)

    mean_low, n_low, note_low = _frontdoor_mean(data, treatment, outcome, mediators, low)
    mean_high, n_high, note_high = _frontdoor_mean(data, treatment, outcome, mediators, high)
    effect = float(mean_high - mean_low) if np.isfinite(mean_high - mean_low) else float("nan")
    ci_low, ci_high = _bootstrap_ci(data, treatment, outcome, mediators, low, high, draws=bootstrap_draws)
    support = pd.to_numeric(data[treatment], errors="coerce").dropna()
    support_low = int((support <= low).sum()) if len(support) else 0
    support_high = int((support >= high).sum()) if len(support) else 0
    overlap_pass = int(n_low >= max(10, len(mediators) + 4) and n_high >= max(10, len(mediators) + 4) and np.isfinite(effect))
    notes = [authorization.reason_codes]
    for note in [note_low, note_high]:
        if note:
            notes.append(note)
    if not overlap_pass:
        notes.append("WEAK_FRONTDOOR_SUPPORT")

    row = {
        "effect_id": effect_id,
        "treatment": treatment,
        "outcome": outcome,
        "do_value_low": low,
        "do_value_high": high,
        "estimand": "frontdoor: E[Y|do(X=high)] - E[Y|do(X=low)]",
        "identification_strategy": authorization.identification_strategy or "frontdoor",
        "adjustment_set": "frontdoor_mediators=" + "|".join(mediators),
        "adjustment_set_status": authorization.adjustment_set_status,
        "do_authorized": int(bool(authorization.do_authorized)),
        "do_mode": authorization.do_mode,
        "effect_estimate": effect,
        "mean_do_low": mean_low,
        "mean_do_high": mean_high,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n": min(n_low, n_high),
        "support_n_low": support_low,
        "support_n_high": support_high,
        "authority_level": authorization.authority_level,
        "effect_semantics": ("frontdoor_adjusted_do_estimand_contract_authorized_limited" if authorization.do_authorized else "diagnostic_frontdoor_estimate_not_causal_authority"),
        "analysis_policy": getattr(authorization, "analysis_policy", "balanced"),
        "diagnostic_estimation_allowed": getattr(authorization, "diagnostic_estimation_allowed", 0),
        "diagnostic_authority_level": getattr(authorization, "diagnostic_authority_level", ""),
        "causal_authority_from_diagnostic": getattr(authorization, "causal_authority_from_diagnostic", 0),
        "reason_codes": "|".join([x for x in notes if x]),
    }
    diag = dict(base_diag, overlap_pass=overlap_pass, support_n_low=support_low, support_n_high=support_high, adjustment_columns_missing="", data_columns_missing="", diagnostic_notes=row["reason_codes"])
    return row, diag


def estimate_authorized_frontdoor_do_effects(out_dir: str = "out", data_path: Optional[str] = None, contract_path: Optional[str] = None, bootstrap_draws: int = 80, policy: object = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    contract = load_causal_contract(out_dir, contract_path)
    data = _load_data(data_path=data_path, out_dir=out_dir)
    rows: List[Dict[str, object]] = []
    diagnostics: List[Dict[str, object]] = []
    for auth in authorize_all_frontdoor_do(contract, policy=policy, include_diagnostic=True):
        row, diag = estimate_frontdoor_do(auth.treatment, auth.outcome, data, auth, bootstrap_draws=bootstrap_draws)
        rows.append(row)
        diagnostics.append(diag)
    return pd.DataFrame(rows, columns=DO_ESTIMATE_COLUMNS), pd.DataFrame(diagnostics, columns=DO_DIAGNOSTIC_COLUMNS)


__all__ = [
    "parse_mediators",
    "estimate_frontdoor_do",
    "estimate_authorized_frontdoor_do_effects",
]
