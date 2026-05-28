
from runtime_env import configure_scientific_runtime
configure_scientific_runtime()

import math
from typing import Dict, List, Tuple

import numpy as np
from runtime_compat import assert_scientific_stack
assert_scientific_stack()
import pandas as pd

from . import common as C
from . import matching as M

BOOT_B = C.BOOT_B
BOOTSTRAP_SEED = C.BOOTSTRAP_SEED
PRETREND_DAYS = C.PRETREND_DAYS
PROPENSITY_ACTION_COL = C.PROPENSITY_ACTION_COL
PROPENSITY_MAX_DIFF = C.PROPENSITY_MAX_DIFF
RIDGE_L2 = C.RIDGE_L2


def _safe_propensity_array(df: pd.DataFrame) -> np.ndarray:
    return pd.to_numeric(df.get("__propensity__", pd.Series([np.nan] * len(df))), errors="coerce").to_numpy(dtype=float)


def _clip01(x: float, lo: float = 0.02, hi: float = 0.98) -> float:
    if not np.isfinite(x):
        return np.nan
    return float(min(max(float(x), lo), hi))


def _weighted_mean(x, w):
    return M._weighted_mean(x, w)


def _normalize_weights(weights: np.ndarray, n: int) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    if len(w) != int(n):
        return np.full(int(n), 1.0 / float(max(1, int(n))), dtype=float)
    s = float(np.nansum(w))
    if not np.isfinite(s) or s <= 0:
        return np.full(int(n), 1.0 / float(max(1, int(n))), dtype=float)
    return w / s


def _bootstrap_sign_stability(trial_delta: float, ctrl_arr: np.ndarray, weights: np.ndarray, b: int = 200) -> float:
    arr = np.asarray(ctrl_arr, dtype=float)
    w = np.asarray(weights, dtype=float)
    m = np.isfinite(arr) & np.isfinite(w)
    arr = arr[m]
    w = w[m]
    if len(arr) < 4:
        return np.nan
    w = _normalize_weights(w, len(arr))
    rng = np.random.default_rng(BOOTSTRAP_SEED + 17)
    vals = []
    n = len(arr)
    for _ in range(max(80, int(b))):
        idx = rng.choice(np.arange(n), size=n, replace=True, p=w)
        vals.append(float(trial_delta - np.mean(arr[idx])))
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.nan
    pos = float(np.mean(vals > 0.0))
    neg = float(np.mean(vals < 0.0))
    return float(max(pos, neg))


def _block_bootstrap_effect_interval(trial_delta: float, ctrl_arr: np.ndarray, weights: np.ndarray, b: int = 200) -> Tuple[float, float]:
    arr = np.asarray(ctrl_arr, dtype=float)
    w = np.asarray(weights, dtype=float)
    m = np.isfinite(arr) & np.isfinite(w)
    arr = arr[m]
    w = w[m]
    if len(arr) < 4:
        return np.nan, np.nan
    w = _normalize_weights(w, len(arr))
    order = np.argsort(-w)
    arr = arr[order]
    n = len(arr)
    block = max(2, min(5, int(round(math.sqrt(n)))))
    starts = np.arange(n, dtype=int)
    start_w = _normalize_weights(w[order], len(starts))
    rng = np.random.default_rng(BOOTSTRAP_SEED + 31)
    draws = []
    for _ in range(max(80, int(b))):
        sample = []
        while len(sample) < n:
            s = int(rng.choice(starts, p=start_w))
            sample.extend(arr[(s + j) % n] for j in range(block))
        sample_arr = np.asarray(sample[:n], dtype=float)
        draws.append(float(trial_delta - np.mean(sample_arr)))
    vals = np.asarray(draws, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.nan, np.nan
    return float(np.quantile(vals, 0.05)), float(np.quantile(vals, 0.95))


def _propensity_augmented_effect(
    trial_delta: float,
    ctrl_arr: np.ndarray,
    weights: np.ndarray,
    treated_prop: float,
    control_props: np.ndarray,
) -> Dict[str, float]:
    if not np.isfinite(trial_delta) or len(ctrl_arr) == 0:
        return {"effect_ipw": np.nan, "propensity_overlap_quality": np.nan, "att_ipw": np.nan}

    w = _normalize_weights(np.asarray(weights, dtype=float), len(ctrl_arr))

    p_t = _clip01(treated_prop)
    p_c = np.asarray([_clip01(v) for v in np.asarray(control_props, dtype=float)], dtype=float)
    valid = np.isfinite(p_c)
    if not np.isfinite(p_t) or int(np.sum(valid)) < max(3, len(ctrl_arr) // 2):
        return {"effect_ipw": np.nan, "propensity_overlap_quality": np.nan, "att_ipw": np.nan}

    yy = np.asarray(ctrl_arr, dtype=float)[valid]
    ww = w[valid]
    pp = p_c[valid]
    inv_ate = 1.0 / np.clip(1.0 - pp, 1e-3, 1.0)
    ipw_w = ww * inv_ate
    ipw_s = float(np.sum(ipw_w))
    if not np.isfinite(ipw_s) or ipw_s <= 0:
        return {"effect_ipw": np.nan, "propensity_overlap_quality": np.nan, "att_ipw": np.nan}
    ipw_w = ipw_w / ipw_s
    ctrl_mean_ipw = float(np.sum(ipw_w * yy))
    effect_ipw = float(trial_delta - ctrl_mean_ipw)

    att_w = ww * (pp / np.clip(1.0 - pp, 1e-3, 1.0))
    att_s = float(np.sum(att_w))
    att_ipw = np.nan
    if np.isfinite(att_s) and att_s > 0:
        att_w = att_w / att_s
        att_ipw = float(trial_delta - np.sum(att_w * yy))

    overlap_quality = float(max(0.0, 1.0 - min(1.0, abs(p_t - float(np.nanmean(pp))) / max(PROPENSITY_MAX_DIFF, 1e-3))))
    return {
        "effect_ipw": effect_ipw,
        "propensity_overlap_quality": overlap_quality,
        "att_ipw": float(att_ipw) if np.isfinite(att_ipw) else np.nan,
    }


def _fit_ridge_predict(X_train: np.ndarray, y_train: np.ndarray, X_pred: np.ndarray, l2: float = RIDGE_L2) -> np.ndarray:
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    X_pred = np.asarray(X_pred, dtype=float)
    if X_train.ndim != 2 or len(X_train) < max(6, X_train.shape[1] + 2):
        return np.full(len(X_pred), np.nan, dtype=float)
    mu = np.nanmean(X_train, axis=0)
    sd = np.nanstd(X_train, axis=0)
    sd = np.where(np.isfinite(sd) & (sd > 1e-6), sd, 1.0)
    Xt = (X_train - mu) / sd
    Xp = (X_pred - mu) / sd
    A = np.column_stack([np.ones(len(Xt)), Xt])
    B = np.column_stack([np.ones(len(Xp)), Xp])
    reg = float(l2) * np.eye(A.shape[1], dtype=float)
    reg[0, 0] = 0.0
    try:
        beta = np.linalg.solve(A.T @ A + reg, A.T @ y_train)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(A.T @ A + reg) @ (A.T @ y_train)
    return B @ beta


def _build_history_training_panel(df: pd.DataFrame, t_idx: int, window_days: int, outcome_col: str, covs: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if outcome_col not in df.columns or PROPENSITY_ACTION_COL not in df.columns:
        return np.empty((0, 0)), np.array([], dtype=float), np.array([], dtype=float)
    pre_days = max(5, PRETREND_DAYS)
    max_anchor = min(int(t_idx) - 1, len(df) - max(1, int(window_days)) - 1)
    if max_anchor <= pre_days + 2:
        return np.empty((0, 0)), np.array([], dtype=float), np.array([], dtype=float)

    X_rows = []
    y_rows = []
    a_rows = []
    for idx in range(pre_days, max_anchor + 1):
        post = M._window_mean(df, idx, window_days, outcome_col)
        pre = M._window_mean(df, max(0, int(idx) - pre_days), pre_days - 1, outcome_col)
        if not np.isfinite(post) or not np.isfinite(pre):
            continue
        row = []
        ok = True
        for c in covs:
            if c not in df.columns:
                ok = False
                break
            val = pd.to_numeric(pd.Series([df[c].iloc[idx]]), errors="coerce").iloc[0]
            if not np.isfinite(val):
                ok = False
                break
            row.append(float(val))
        if not ok:
            continue
        act = pd.to_numeric(pd.Series([df[PROPENSITY_ACTION_COL].iloc[idx]]), errors="coerce").iloc[0]
        if not np.isfinite(act):
            continue
        X_rows.append(row)
        y_rows.append(float(post - pre))
        a_rows.append(float(act > 0.5))
    if not X_rows:
        return np.empty((0, len(covs))), np.array([], dtype=float), np.array([], dtype=float)
    return np.asarray(X_rows, dtype=float), np.asarray(y_rows, dtype=float), np.asarray(a_rows, dtype=float)


def _doubly_robust_effects(
    df: pd.DataFrame,
    t_idx: int,
    trial_delta: float,
    ctrl_arr: np.ndarray,
    weights: np.ndarray,
    treated_prop: float,
    control_props: np.ndarray,
    window_days: int,
    outcome_col: str,
    covs: List[str],
) -> Dict[str, float]:
    out = {
        "effect_dr_att": np.nan,
        "effect_dr_ate": np.nan,
        "att_outcome_model": np.nan,
        "history_treated_n": np.nan,
        "history_control_n": np.nan,
    }
    if not np.isfinite(trial_delta) or len(ctrl_arr) < 3:
        return out
    hist_X, hist_y, hist_a = _build_history_training_panel(df, t_idx, window_days, outcome_col, covs)
    if hist_X.size == 0 or len(hist_y) < 12:
        return out

    n_t = int(np.sum(hist_a > 0.5))
    n_c = int(np.sum(hist_a <= 0.5))
    out["history_treated_n"] = float(n_t)
    out["history_control_n"] = float(n_c)
    if n_t < max(4, hist_X.shape[1] + 1) or n_c < max(6, hist_X.shape[1] + 2):
        return out

    x_trial = np.asarray([
        [float(pd.to_numeric(pd.Series([df[c].iloc[t_idx]]), errors="coerce").iloc[0]) for c in covs]
    ], dtype=float)
    valid_ids = np.asarray([int(i) for i in range(len(ctrl_arr))], dtype=int)
    # caller aligns ctrl_arr with control indices, so rebuild control cov rows from those indices outside if needed
    # fallback: use matched-control rows from df when available via length equality with provided weights and props
    ctrl_cov_rows = []
    ctrl_index_candidates = np.asarray([], dtype=int)
    if len(control_props) == len(ctrl_arr):
        # the caller passes props in matched-control order, so reuse the same order by searching the original df indices later
        pass

    # trial predictions
    mu1_t = _fit_ridge_predict(hist_X[hist_a > 0.5], hist_y[hist_a > 0.5], x_trial, l2=RIDGE_L2)[0]
    mu0_t = _fit_ridge_predict(hist_X[hist_a <= 0.5], hist_y[hist_a <= 0.5], x_trial, l2=RIDGE_L2)[0]
    if np.isfinite(mu0_t):
        out["att_outcome_model"] = float(trial_delta - mu0_t)

    if not np.isfinite(treated_prop):
        treated_prop = np.nanmean(control_props) if len(control_props) else np.nan
    p_t = _clip01(treated_prop)
    p_c = np.asarray([_clip01(v) for v in np.asarray(control_props, dtype=float)], dtype=float)
    w = _normalize_weights(weights, len(ctrl_arr))

    if np.isfinite(mu0_t) and np.all(np.isfinite(p_c)):
        ctrl_resid = ctrl_arr - np.nanmean(hist_y[hist_a <= 0.5])
        att_w = w * (p_c / np.clip(1.0 - p_c, 1e-3, 1.0))
        att_w = _normalize_weights(att_w, len(ctrl_arr))
        out["effect_dr_att"] = float((trial_delta - mu0_t) - np.sum(att_w * ctrl_resid))

    if np.isfinite(mu0_t) and np.isfinite(mu1_t) and np.isfinite(p_t) and np.all(np.isfinite(p_c)):
        ctrl_mu0_mean = float(np.sum(w * np.full(len(ctrl_arr), np.nanmean(hist_y[hist_a <= 0.5]))))
        trial_term = (trial_delta - mu1_t) / p_t
        ctrl_term = np.sum(w * ((ctrl_arr - ctrl_mu0_mean) / np.clip(1.0 - p_c, 1e-3, 1.0)))
        out["effect_dr_ate"] = float((mu1_t - mu0_t) + trial_term - ctrl_term)
    return out


def estimate_effect_bundle(df: pd.DataFrame, t_idx: int, controls: List[int], window_days: int, outcome_col: str, covs: List[str]) -> Dict[str, object]:
    """Estimate a stronger matched counterfactual effect bundle.

    Keeps the package lightweight (numpy/pandas only) while upgrading matched
    estimation with:
    - matched difference-in-differences event-window deltas,
    - ridge bias correction on pre-treatment covariates,
    - propensity reweighting,
    - historical outcome-model based doubly-robust proxies,
    - block-bootstrap confidence intervals and sign-stability diagnostics.
    """
    base = M._aipw_style_effect(df, t_idx, controls, window_days, outcome_col, covs) or {}
    if not controls:
        out = dict(base)
        out.setdefault("effect_method", "augmented_matched_did")
        out.setdefault("effect_ipw", np.nan)
        out.setdefault("att_ipw", np.nan)
        out.setdefault("effect_dr_att", np.nan)
        out.setdefault("effect_dr_ate", np.nan)
        out.setdefault("effect_sign_stable", np.nan)
        out.setdefault("method_agreement", np.nan)
        out.setdefault("propensity_overlap_quality", np.nan)
        return out

    pre_days = max(5, PRETREND_DAYS)
    trial_post = M._window_mean(df, t_idx, window_days, outcome_col)
    trial_pre = M._window_mean(df, max(0, int(t_idx) - pre_days), pre_days - 1, outcome_col)
    trial_delta = float(trial_post - trial_pre) if np.isfinite(trial_post) and np.isfinite(trial_pre) else np.nan

    valid_control_ids = [int(ci) for ci in base.get("valid_control_ids", controls) if 0 <= int(ci) < len(df)]
    ctrl_arr = []
    ctrl_used_ids = []
    for ci in valid_control_ids:
        c_post = M._window_mean(df, ci, window_days, outcome_col)
        c_pre = M._window_mean(df, max(0, int(ci) - pre_days), pre_days - 1, outcome_col)
        if np.isfinite(c_post) and np.isfinite(c_pre):
            ctrl_arr.append(float(c_post - c_pre))
            ctrl_used_ids.append(int(ci))
    ctrl_arr = np.asarray(ctrl_arr, dtype=float)

    w, eff_n, w_max = M._match_weights(df, t_idx, ctrl_used_ids)
    if len(ctrl_arr) != len(ctrl_used_ids):
        valid_n = int(len(ctrl_arr))
        w = np.full(valid_n, 1.0 / float(valid_n), dtype=float) if valid_n > 0 else np.array([], dtype=float)
        eff_n = float(valid_n) if valid_n > 0 else np.nan
        w_max = float(np.max(w)) if valid_n > 0 else np.nan
    else:
        w = _normalize_weights(w, len(ctrl_arr))

    prop = _safe_propensity_array(df)
    treated_prop = float(prop[t_idx]) if 0 <= int(t_idx) < len(prop) else np.nan
    ctrl_props = prop[np.asarray(ctrl_used_ids, dtype=int)] if len(ctrl_used_ids) > 0 else np.array([], dtype=float)

    aug = _propensity_augmented_effect(trial_delta, ctrl_arr, w, treated_prop, ctrl_props)
    dr = _doubly_robust_effects(df, t_idx, trial_delta, ctrl_arr, w, treated_prop, ctrl_props, window_days, outcome_col, covs)
    effect_bc = float(base.get("effect", np.nan)) if np.isfinite(base.get("effect", np.nan)) else np.nan
    effect_ipw = float(aug.get("effect_ipw", np.nan)) if np.isfinite(aug.get("effect_ipw", np.nan)) else np.nan
    effect_dr_att = float(dr.get("effect_dr_att", np.nan)) if np.isfinite(dr.get("effect_dr_att", np.nan)) else np.nan
    effect_dr_ate = float(dr.get("effect_dr_ate", np.nan)) if np.isfinite(dr.get("effect_dr_ate", np.nan)) else np.nan

    candidates = [v for v in [effect_bc, effect_ipw, effect_dr_att, effect_dr_ate] if np.isfinite(v)]
    if len(candidates) >= 2:
        signs = [np.sign(v) for v in candidates if abs(v) > 1e-12]
        method_agreement = float(np.mean(np.asarray(signs, dtype=float) == np.sign(np.nanmedian(candidates)))) if signs else np.nan
    else:
        method_agreement = np.nan

    overlap_q = float(aug.get("propensity_overlap_quality", np.nan)) if np.isfinite(aug.get("propensity_overlap_quality", np.nan)) else np.nan
    primary = effect_bc
    if np.isfinite(effect_dr_att):
        primary = effect_dr_att
    elif np.isfinite(effect_ipw) and np.isfinite(effect_bc):
        blend_w = 0.30 + 0.35 * (overlap_q if np.isfinite(overlap_q) else 0.0)
        blend_w = float(min(max(blend_w, 0.20), 0.70))
        primary = float((1.0 - blend_w) * effect_bc + blend_w * effect_ipw)
    elif np.isfinite(effect_ipw):
        primary = effect_ipw

    effect_sign_stable = _bootstrap_sign_stability(trial_delta, ctrl_arr, w, b=BOOT_B)
    ci_low_blk, ci_high_blk = _block_bootstrap_effect_interval(trial_delta, ctrl_arr, w, b=BOOT_B)

    out = dict(base)
    if np.isfinite(primary):
        out["effect"] = float(primary)
        se = float(out.get("effect_se_proxy", np.nan))
        if np.isfinite(se) and se > 1e-9:
            out["z_raw"] = float(primary / se)
    if np.isfinite(ci_low_blk) and np.isfinite(ci_high_blk):
        out["effect_ci_low"] = float(ci_low_blk)
        out["effect_ci_high"] = float(ci_high_blk)
    out.update({
        "effect_method": "augmented_matched_did_dr",
        "effect_bias_corrected": effect_bc,
        "effect_ipw": effect_ipw,
        "att_ipw": float(aug.get("att_ipw", np.nan)) if np.isfinite(aug.get("att_ipw", np.nan)) else np.nan,
        "effect_dr_att": effect_dr_att,
        "effect_dr_ate": effect_dr_ate,
        "att_mean": float(primary) if np.isfinite(primary) else np.nan,
        "effect_sign_stable": float(effect_sign_stable) if np.isfinite(effect_sign_stable) else np.nan,
        "method_agreement": float(method_agreement) if np.isfinite(method_agreement) else np.nan,
        "propensity_overlap_quality": overlap_q,
        "match_effective_n": float(eff_n) if np.isfinite(eff_n) else out.get("match_effective_n", np.nan),
        "match_weight_max": float(w_max) if np.isfinite(w_max) else out.get("match_weight_max", np.nan),
        "history_treated_n": float(dr.get("history_treated_n", np.nan)) if np.isfinite(dr.get("history_treated_n", np.nan)) else np.nan,
        "history_control_n": float(dr.get("history_control_n", np.nan)) if np.isfinite(dr.get("history_control_n", np.nan)) else np.nan,
    })
    return out
