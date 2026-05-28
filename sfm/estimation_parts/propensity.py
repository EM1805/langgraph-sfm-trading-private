
from runtime_env import configure_scientific_runtime
configure_scientific_runtime()

import os

import numpy as np
from runtime_compat import assert_scientific_stack
assert_scientific_stack()
import pandas as pd

from . import common as C

BOOTSTRAP_SEED = C.BOOTSTRAP_SEED
BOOT_B = C.BOOT_B
DATE_COL = C.DATE_COL
EXP_RESULTS = C.EXP_RESULTS
EXP_SUMMARY_L29 = C.EXP_SUMMARY_L29
LOOKBACK_DAYS = C.LOOKBACK_DAYS
LOOKBACK_ROWS = C.LOOKBACK_ROWS
PRETREND_MAX_DIFF = C.PRETREND_MAX_DIFF
TARGET_COL = C.TARGET_COL

_as_str = C._as_str
_has_date = C._has_date


def _sigmoid(z):
    z = np.asarray(z, dtype=float)
    z = np.clip(z, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def _fit_logistic_proba(X, y, l2=1.0, steps=250, lr=0.1):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, k = X.shape
    mu = np.nanmean(X, axis=0)
    sig = np.nanstd(X, axis=0)
    sig = np.where(np.isfinite(sig) & (sig > 1e-6), sig, 1.0)
    Xs = (X - mu) / sig
    A = np.column_stack([np.ones(n), Xs])
    w = np.zeros(k + 1, dtype=float)
    for _ in range(int(steps)):
        p = _sigmoid(A @ w)
        g = (A.T @ (p - y)) / float(n)
        g[1:] += (float(l2) / float(n)) * w[1:]
        w -= float(lr) * g
    return w, mu, sig


def _predict_logistic_proba(X, w, mu, sig):
    X = np.asarray(X, dtype=float)
    A = np.column_stack([np.ones(len(X)), (X - mu) / sig])
    return _sigmoid(A @ w)


def _compute_propensity(df, action_col, covs):
    if action_col not in df.columns:
        return np.full(len(df), np.nan, dtype=float)
    cols = [c for c in covs if c in df.columns]
    if not cols:
        return np.full(len(df), np.nan, dtype=float)
    X = np.vstack([pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float) for c in cols]).T
    y = pd.to_numeric(df[action_col], errors="coerce").fillna(0).to_numpy(dtype=float)
    m = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    if int(m.sum()) < 25:
        return np.full(len(df), np.nan, dtype=float)
    yy = (y[m] > 0.5).astype(float)
    if yy.sum() < 5 or (len(yy) - yy.sum()) < 5:
        return np.full(len(df), np.nan, dtype=float)
    w, mu, sig = _fit_logistic_proba(X[m], yy, l2=1.0, steps=300, lr=0.15)
    p = np.full(len(df), np.nan, dtype=float)
    p[m] = _predict_logistic_proba(X[m], w, mu, sig)
    return p


def _trend_slope(y):
    y = np.asarray(y, dtype=float)
    m = np.isfinite(y)
    if int(m.sum()) < 3:
        return np.nan
    x = np.arange(len(y), dtype=float)[m]
    yy = y[m]
    try:
        a, _ = np.polyfit(x, yy, 1)
        return float(a)
    except (TypeError, ValueError, np.linalg.LinAlgError):
        return np.nan


def _pretrend_check(df, t_idx, controls, days, outcome_col=None):
    days = int(max(3, days))
    ycol = C.resolve_outcome_col(df, outcome_col or TARGET_COL)
    if ycol not in df.columns:
        return 1, np.nan, np.nan, np.nan, "outcome_missing"
    y = pd.to_numeric(df[ycol], errors="coerce").to_numpy(dtype=float)

    def window(idx):
        a = max(0, int(idx) - days)
        b = int(idx)
        return y[a:b] if b - a >= 3 else None

    yt = window(t_idx)
    if yt is None:
        return 1, np.nan, np.nan, np.nan, "too_early"
    st = _trend_slope(yt)
    if not np.isfinite(st):
        return 1, np.nan, np.nan, np.nan, "trial_slope_nan"

    sc = []
    for ci in controls:
        yc = window(ci)
        if yc is None:
            continue
        s = _trend_slope(yc)
        if np.isfinite(s):
            sc.append(float(s))

    if len(sc) < 3:
        return 1, float(st), np.nan, np.nan, "too_few_control_trends"

    sc_mean = float(np.mean(sc))
    diff = abs(float(st) - sc_mean)
    scale = float(np.nanstd(y))
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0
    thresh = 1.35 * float(PRETREND_MAX_DIFF) * scale / float(days)
    ok = int(diff <= thresh)
    return ok, float(st), sc_mean, diff, ""


def _signed_z(z, expected_direction):
    if not np.isfinite(z):
        return np.nan
    d = _as_str(expected_direction).strip().lower()
    if d in ("increase", "higher", "up", "+"):
        return float(z)
    if d in ("decrease", "lower", "down", "-"):
        return float(-z)
    return float(z)


def _window_end_index(df, t_idx, window_days):
    wd = int(max(1, window_days))
    if _has_date(df):
        d = df[DATE_COL].iloc[t_idx]
        if pd.notna(d):
            end = d + pd.Timedelta(days=wd)
            cand = df.index[df[DATE_COL] <= end].to_numpy(dtype=int)
            cand = cand[cand >= t_idx]
            if len(cand) > 0:
                return int(cand[-1])
    return min(len(df) - 1, int(t_idx) + wd)


def _bootstrap_ci(values, b=None, alpha=0.10, seed=None):
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if len(arr) < 2:
        return np.nan, np.nan
    draws = max(50, int(BOOT_B if b is None else b))
    rng_seed = BOOTSTRAP_SEED if seed is None else int(seed)
    rng = np.random.default_rng(rng_seed)
    means = []
    for _ in range(draws):
        idx = rng.integers(0, len(arr), len(arr))
        means.append(float(np.mean(arr[idx])))
    return float(np.quantile(means, alpha / 2.0)), float(np.quantile(means, 1.0 - alpha / 2.0))


def _load_l29_summary():
    if not os.path.exists(EXP_SUMMARY_L29):
        return pd.DataFrame()
    try:
        return pd.read_csv(EXP_SUMMARY_L29)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame()


def _pick_trials_path():
    candidates = []
    for path in [EXP_RESULTS]:
        if not os.path.exists(path):
            continue
        score = 0
        try:
            df = pd.read_csv(path)
            score += int(len(df)) * 10
            score += 3 if "expected_direction_on_target" in df.columns else 0
            score += 2 if "action_active" in df.columns else 0
            score += 2 if "logged_at_unix" in df.columns else 0
            score += 1 if "raw_t_index" in df.columns else 0
        except (OSError, ValueError, TypeError, pd.errors.ParserError):
            pass
        candidates.append((score, path))
    if not candidates:
        return EXP_RESULTS
    candidates.sort(reverse=True)
    return candidates[0][1]
