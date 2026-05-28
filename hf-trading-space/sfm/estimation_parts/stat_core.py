"""Shared statistical primitives for Amantia estimation.

This module is intentionally small, deterministic, and dependency-light.
It centralizes repeated OLS/ridge/bootstrap/standardization helpers that were
previously duplicated across effect_estimates, placebo, negative_controls,
pearl_backdoor, matching, and effects.

Design rules:
- Conservative by default: invalid/undersupported inputs return NaN/empty
  results rather than raising in the middle of a pipeline run.
- No causal authority is created here. This file only performs numerical
  routines after contract/ID gates have already authorized an estimate.
- Deterministic bootstraps: every bootstrap helper accepts a seed.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable, List, Optional, Sequence, Tuple

try:
    from runtime_env import configure_scientific_runtime
    configure_scientific_runtime()
except Exception:
    pass

import numpy as np
import pandas as pd


EPS = 1e-12
DEFAULT_RIDGE = 1e-6
DEFAULT_BOOTSTRAP_SEED = 1729


@dataclass(frozen=True)
class LinearEffectResult:
    """Result for a treatment coefficient from a linear adjustment model."""

    effect: float
    se: float
    t_stat: float
    n: int
    df: int
    residual_sd: float
    partial_r2: float
    status: str

    def to_dict(self) -> dict:
        return asdict(self)


def safe_float(value, default: float = np.nan) -> float:
    """Convert value to finite float; return default on failure."""

    try:
        out = float(value)
    except Exception:
        return default
    if not np.isfinite(out):
        return default
    return out


def finite_array(values: Sequence[object] | np.ndarray) -> np.ndarray:
    """Return a 1D float array with non-finite entries removed."""

    arr = np.asarray(values, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def split_cols(value) -> List[str]:
    """Parse comma/semicolon/list-style column specs into a clean list."""

    if value is None:
        return []
    if isinstance(value, float) and np.isnan(value):
        return []
    if isinstance(value, (list, tuple, set)):
        out: List[str] = []
        for item in value:
            out.extend(split_cols(item))
        return list(dict.fromkeys(out))
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "[]"}:
        return []
    for ch in "[](){}'\"":
        text = text.replace(ch, "")
    parts: List[str] = []
    for piece in text.replace(";", ",").split(","):
        piece = piece.strip()
        if piece:
            parts.append(piece)
    return list(dict.fromkeys(parts))


def numeric_matrix(
    df: pd.DataFrame,
    cols: Iterable[str],
    *,
    exclude: Optional[set[str]] = None,
    standardize: bool = True,
    min_non_null: int = 3,
) -> Tuple[np.ndarray, List[str]]:
    """Build a numeric matrix from available columns.

    Returns (matrix, used_columns). Columns that are absent, constant, or have
    too little support are dropped. Missing values are median-imputed before
    optional standardization.
    """

    exclude = exclude or set()
    used: List[str] = []
    arrays: List[np.ndarray] = []
    for col in split_cols(list(cols)):
        if col in exclude or col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        if int(s.notna().sum()) < min_non_null:
            continue
        med = float(s.median()) if np.isfinite(float(s.median())) else 0.0
        arr = s.fillna(med).to_numpy(dtype=float)
        if not np.isfinite(arr).all():
            arr = np.nan_to_num(arr, nan=med, posinf=med, neginf=med)
        sd = float(np.std(arr))
        if sd <= EPS:
            continue
        if standardize:
            arr = (arr - float(np.mean(arr))) / (sd + EPS)
        arrays.append(arr)
        used.append(col)
    if not arrays:
        return np.zeros((len(df), 0), dtype=float), []
    return np.column_stack(arrays).astype(float), used


def standardize_matrix(Z: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Standardize each non-constant column of a DataFrame/array.

    DataFrame columns with no usable variance are dropped. Array columns are
    kept only when finite and non-constant. Empty inputs return an (n, 0) array.
    """

    if isinstance(Z, pd.DataFrame):
        X, _ = numeric_matrix(Z, list(Z.columns), standardize=True)
        return X
    arr = np.asarray(Z, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.size == 0:
        n = arr.shape[0] if arr.ndim == 2 else 0
        return np.zeros((n, 0), dtype=float)
    cols: List[np.ndarray] = []
    for j in range(arr.shape[1]):
        x = arr[:, j]
        mask = np.isfinite(x)
        if int(mask.sum()) < 3:
            continue
        med = float(np.nanmedian(x[mask])) if mask.any() else 0.0
        x = np.where(np.isfinite(x), x, med)
        sd = float(np.std(x))
        if sd <= EPS:
            continue
        cols.append((x - float(np.mean(x))) / (sd + EPS))
    if not cols:
        return np.zeros((arr.shape[0], 0), dtype=float)
    return np.column_stack(cols).astype(float)


def add_intercept(X: np.ndarray) -> np.ndarray:
    """Prefix a design matrix with an intercept column."""

    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return np.column_stack([np.ones(X.shape[0], dtype=float), X])


def fit_linear_coef(X: np.ndarray, y: np.ndarray, *, ridge: float = DEFAULT_RIDGE) -> np.ndarray:
    """Fit linear coefficients with tiny ridge fallback.

    X must already include the intercept if one is desired. Returns NaN
    coefficients when dimensions are invalid.
    """

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.shape[0] != y.shape[0] or X.shape[0] == 0:
        return np.full(X.shape[1] if X.ndim == 2 else 0, np.nan)
    mask = np.isfinite(y) & np.isfinite(X).all(axis=1)
    X = X[mask]
    y = y[mask]
    if X.shape[0] < max(2, X.shape[1]):
        return np.full(X.shape[1], np.nan)
    try:
        if ridge and ridge > 0:
            penalty = np.eye(X.shape[1], dtype=float) * float(ridge)
            penalty[0, 0] = 0.0  # do not penalize intercept by convention
            return np.linalg.solve(X.T @ X + penalty, X.T @ y)
        return np.linalg.lstsq(X, y, rcond=None)[0]
    except Exception:
        return np.linalg.pinv(X) @ y


def fit_ols(X: np.ndarray, y: np.ndarray, ridge: float = DEFAULT_RIDGE) -> np.ndarray:
    """Compatibility alias for existing modules."""

    return fit_linear_coef(X, y, ridge=ridge)


def fit_ridge_coef(X: np.ndarray, y: np.ndarray, l2: float = 1.0) -> np.ndarray:
    """Compatibility alias for ridge coefficient estimation."""

    return fit_linear_coef(X, y, ridge=l2)


def predict_linear(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Predict y from a design matrix and coefficients."""

    X = np.asarray(X, dtype=float)
    beta = np.asarray(beta, dtype=float).reshape(-1)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.shape[1] != beta.shape[0]:
        return np.full(X.shape[0], np.nan)
    return X @ beta


def linear_treatment_effect(
    treatment: Sequence[object] | np.ndarray,
    outcome: Sequence[object] | np.ndarray,
    covariates: Optional[np.ndarray] = None,
    *,
    ridge: float = DEFAULT_RIDGE,
) -> LinearEffectResult:
    """Estimate treatment coefficient from y ~ 1 + treatment + covariates.

    This routine is numerical only. The caller is responsible for checking
    identification, authority, temporal ordering, and allowed adjustment sets.
    """

    a = np.asarray(treatment, dtype=float).reshape(-1)
    y = np.asarray(outcome, dtype=float).reshape(-1)
    if covariates is None:
        Z = np.zeros((len(a), 0), dtype=float)
    else:
        Z = np.asarray(covariates, dtype=float)
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)
    if len(a) != len(y) or Z.shape[0] != len(a):
        return LinearEffectResult(np.nan, np.nan, np.nan, 0, 0, np.nan, np.nan, "shape_mismatch")

    X = np.column_stack([np.ones(len(a), dtype=float), a, Z])
    mask = np.isfinite(y) & np.isfinite(X).all(axis=1)
    X = X[mask]
    y = y[mask]
    n = int(len(y))
    p = int(X.shape[1]) if X.ndim == 2 else 0
    df = max(0, n - p)
    if n < max(8, p + 3):
        return LinearEffectResult(np.nan, np.nan, np.nan, n, df, np.nan, np.nan, "insufficient_support")

    beta = fit_linear_coef(X, y, ridge=ridge)
    if beta.shape[0] < 2 or not np.isfinite(beta[1]):
        return LinearEffectResult(np.nan, np.nan, np.nan, n, df, np.nan, np.nan, "fit_failed")

    resid = y - X @ beta
    rss = float(np.sum(resid ** 2))
    residual_sd = float(np.sqrt(rss / max(1, df)))
    try:
        xtx_inv = np.linalg.pinv(X.T @ X)
        se = float(np.sqrt(max(0.0, residual_sd ** 2 * xtx_inv[1, 1])))
    except Exception:
        se = np.nan
    t_stat = float(beta[1] / (se + EPS)) if np.isfinite(se) and se > 0 else np.nan
    pr2 = partial_r2_from_t(t_stat, df)
    return LinearEffectResult(float(beta[1]), se, t_stat, n, df, residual_sd, pr2, "ok")


def percentile_ci(values: Sequence[object] | np.ndarray, *, alpha: float = 0.05) -> Tuple[float, float]:
    """Finite percentile confidence interval."""

    arr = finite_array(values)
    if len(arr) == 0:
        return np.nan, np.nan
    lo = float(np.percentile(arr, 100.0 * alpha / 2.0))
    hi = float(np.percentile(arr, 100.0 * (1.0 - alpha / 2.0)))
    return lo, hi


def bootstrap_coef_ci(
    X: np.ndarray,
    y: np.ndarray,
    *,
    coef_index: int = 1,
    b: int = 160,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    ridge: float = DEFAULT_RIDGE,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """Bootstrap CI for a coefficient in an already-built design matrix."""

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    mask = np.isfinite(y) & np.isfinite(X).all(axis=1)
    X = X[mask]
    y = y[mask]
    n = int(len(y))
    if n < max(8, X.shape[1] + 3) or coef_index >= X.shape[1]:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    vals: List[float] = []
    for _ in range(int(max(1, b))):
        idx = rng.integers(0, n, size=n)
        beta = fit_linear_coef(X[idx], y[idx], ridge=ridge)
        if beta.shape[0] > coef_index and np.isfinite(beta[coef_index]):
            vals.append(float(beta[coef_index]))
    return percentile_ci(vals, alpha=alpha)


def bootstrap_treatment_effect_ci(
    treatment: Sequence[object] | np.ndarray,
    outcome: Sequence[object] | np.ndarray,
    covariates: Optional[np.ndarray] = None,
    *,
    b: int = 160,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    ridge: float = DEFAULT_RIDGE,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """Bootstrap CI for y ~ 1 + treatment + covariates coefficient."""

    a = np.asarray(treatment, dtype=float).reshape(-1)
    y = np.asarray(outcome, dtype=float).reshape(-1)
    if covariates is None:
        Z = np.zeros((len(a), 0), dtype=float)
    else:
        Z = np.asarray(covariates, dtype=float)
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)
    if len(a) != len(y) or Z.shape[0] != len(a):
        return np.nan, np.nan
    X = np.column_stack([np.ones(len(a), dtype=float), a, Z])
    return bootstrap_coef_ci(X, y, coef_index=1, b=b, seed=seed, ridge=ridge, alpha=alpha)


def bootstrap_delta_ci(
    treated_values: Sequence[object] | np.ndarray,
    control_values: Sequence[object] | np.ndarray,
    *,
    b: int = 200,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """Bootstrap CI for mean(treated) - mean(control)."""

    treated = finite_array(treated_values)
    control = finite_array(control_values)
    if len(treated) == 0 or len(control) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    vals: List[float] = []
    for _ in range(int(max(1, b))):
        t = treated[rng.integers(0, len(treated), size=len(treated))]
        c = control[rng.integers(0, len(control), size=len(control))]
        vals.append(float(np.mean(t) - np.mean(c)))
    return percentile_ci(vals, alpha=alpha)


def normal_approx_p_from_t(t_stat: float) -> float:
    """Two-sided normal-approximation p-value from a t/z statistic."""

    import math

    t = safe_float(t_stat)
    if not np.isfinite(t):
        return np.nan
    return float(math.erfc(abs(t) / math.sqrt(2.0)))


def partial_r2_from_t(t_stat: float, df: int) -> float:
    """Partial R^2 implied by a t statistic and degrees of freedom."""

    t = safe_float(t_stat)
    d = int(df) if df is not None else 0
    if not np.isfinite(t) or d <= 0:
        return np.nan
    return float((t * t) / (t * t + d + EPS))


def effect_support_count(treatment: Sequence[object] | np.ndarray, outcome: Sequence[object] | np.ndarray) -> int:
    """Count rows with finite treatment and outcome values."""

    a = np.asarray(treatment, dtype=float).reshape(-1)
    y = np.asarray(outcome, dtype=float).reshape(-1)
    if len(a) != len(y):
        return 0
    return int((np.isfinite(a) & np.isfinite(y)).sum())


def sign_stability(values: Sequence[object] | np.ndarray, expected_sign: Optional[float] = None) -> float:
    """Share of finite values with stable sign.

    If expected_sign is omitted, the sign of the median finite value is used.
    """

    arr = finite_array(values)
    if len(arr) == 0:
        return np.nan
    if expected_sign is None or not np.isfinite(float(expected_sign)) or float(expected_sign) == 0.0:
        expected = float(np.sign(np.median(arr)))
    else:
        expected = float(np.sign(expected_sign))
    if expected == 0.0:
        return 0.0
    return float((np.sign(arr) == expected).mean())


__all__ = [
    "DEFAULT_BOOTSTRAP_SEED",
    "DEFAULT_RIDGE",
    "EPS",
    "LinearEffectResult",
    "add_intercept",
    "bootstrap_coef_ci",
    "bootstrap_delta_ci",
    "bootstrap_treatment_effect_ci",
    "effect_support_count",
    "finite_array",
    "fit_linear_coef",
    "fit_ols",
    "fit_ridge_coef",
    "linear_treatment_effect",
    "normal_approx_p_from_t",
    "numeric_matrix",
    "partial_r2_from_t",
    "percentile_ci",
    "predict_linear",
    "safe_float",
    "sign_stability",
    "split_cols",
    "standardize_matrix",
]
