
from runtime_env import configure_scientific_runtime
configure_scientific_runtime()

import math
import numpy as np
from runtime_compat import assert_scientific_stack
assert_scientific_stack()
import pandas as pd

from . import common as C
from . import propensity as P

BOOT_B = C.BOOT_B
BOOTSTRAP_SEED = C.BOOTSTRAP_SEED
ENABLE_PROPENSITY = C.ENABLE_PROPENSITY
K_CONTROLS = C.K_CONTROLS
MATCH_DIST_MAX = C.MATCH_DIST_MAX
MIN_MATCHED = C.MIN_MATCHED
PRETREND_DAYS = C.PRETREND_DAYS
PRETREND_MAX_DIFF = C.PRETREND_MAX_DIFF
PROPENSITY_MAX_DIFF = C.PROPENSITY_MAX_DIFF
RIDGE_L2 = C.RIDGE_L2
TARGET_COL = C.TARGET_COL

_past_indices = C._past_indices
_robust_center_scale = C._robust_center_scale
_robust_z = C._robust_z
_window_end_index = P._window_end_index
_trend_slope = P._trend_slope


def _ridge_fit_predict(X_train, y_train, x_target, l2=1.0):
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    x_target = np.asarray(x_target, dtype=float)

    if X_train.ndim != 2 or len(X_train) == 0:
        return np.nan, np.full(len(y_train), np.nan)

    n, k = X_train.shape
    A = np.column_stack([np.ones(n), X_train])
    xt = np.concatenate([[1.0], x_target])
    reg = float(max(1e-9, l2))
    penalty = np.eye(k + 1)
    penalty[0, 0] = 0.0
    try:
        beta = np.linalg.solve(A.T @ A + reg * penalty, A.T @ y_train)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(A.T @ A + reg * penalty) @ (A.T @ y_train)

    pred_t = float(xt @ beta)
    pred_train = A @ beta
    return pred_t, pred_train



def _pre_window_values(df, idx, pre_days, outcome_col):
    y = pd.to_numeric(df[outcome_col], errors="coerce").to_numpy(dtype=float)
    start = max(0, int(idx) - int(pre_days))
    end = max(start, int(idx))
    vals = y[start:end]
    return vals[np.isfinite(vals)]

def _window_values(df, idx, window_days, outcome_col):
    end_idx = _window_end_index(df, idx, window_days)
    y = pd.to_numeric(df[outcome_col], errors="coerce").to_numpy(dtype=float)
    vals = y[int(idx): int(end_idx) + 1]
    return vals[np.isfinite(vals)]


def _window_mean(df, idx, window_days, outcome_col):
    vals = _window_values(df, idx, window_days, outcome_col)
    if len(vals) == 0:
        return np.nan
    return float(np.mean(vals))


def _window_std(df, idx, window_days, outcome_col):
    vals = _window_values(df, idx, window_days, outcome_col)
    if len(vals) < 2:
        return np.nan
    return float(np.std(vals, ddof=1))


def _build_anchor_features(df, idx, covs, pre_days, outcome_col=None):
    feats = []
    names = []
    ycol = C.resolve_outcome_col(df, outcome_col or TARGET_COL)
    outcome_pre = _pre_window_values(df, idx, pre_days, ycol)
    if len(outcome_pre) >= 3:
        feats.extend([float(np.mean(outcome_pre)), float(np.std(outcome_pre, ddof=1)) if len(outcome_pre) >= 2 else 0.0, float(_trend_slope(outcome_pre))])
        names.extend(["outcome_pre_mean", "outcome_pre_sd", "outcome_pre_slope"])
    else:
        feats.extend([np.nan, np.nan, np.nan])
        names.extend(["outcome_pre_mean", "outcome_pre_sd", "outcome_pre_slope"])

    for c in covs:
        if c not in df.columns:
            continue
        col = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        v = float(col[int(idx)]) if 0 <= int(idx) < len(col) and np.isfinite(col[int(idx)]) else np.nan
        feats.append(v)
        names.append(c)

    return np.asarray(feats, dtype=float), names


def _robust_mahalanobis_like(xi, xj, scales):
    d = (xi - xj) / scales
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return np.nan
    return float(np.sqrt(np.mean(np.square(np.clip(d, -6.0, 6.0)))))


def _hybrid_match_controls(df, t_idx, covs, outcome_col=None):
    past = _past_indices(df, t_idx)
    if len(past) == 0:
        return [], {"reason": "no_past"}

    pre_days = max(5, PRETREND_DAYS)
    x_t, names = _build_anchor_features(df, t_idx, covs, pre_days, outcome_col=outcome_col)
    if not np.any(np.isfinite(x_t)):
        return [], {"reason": "trial_cov_missing"}

    Xp = []
    keep = []
    for i in past:
        xi, _ = _build_anchor_features(df, int(i), covs, pre_days, outcome_col=outcome_col)
        Xp.append(xi)
        keep.append(int(i))
    if not Xp:
        return [], {"reason": "no_candidate_features"}

    Xp = np.asarray(Xp, dtype=float)
    scales = []
    for j in range(Xp.shape[1]):
        med, sc = _robust_center_scale(Xp[:, j])
        scales.append(sc)
    scales = np.asarray(scales, dtype=float)
    scales = np.where(np.isfinite(scales) & (scales > 1e-6), scales, 1.0)

    prop = pd.to_numeric(df.get("__propensity__", pd.Series([np.nan] * len(df))), errors="coerce").to_numpy(dtype=float)
    prop_t = float(prop[t_idx]) if 0 <= int(t_idx) < len(prop) and np.isfinite(prop[t_idx]) else np.nan
    finite_past_prop = np.asarray([float(prop[i]) for i in keep if 0 <= int(i) < len(prop) and np.isfinite(prop[i])], dtype=float)
    support_low = np.nan
    support_high = np.nan
    support_margin = np.nan
    support_fail = False
    if ENABLE_PROPENSITY and np.isfinite(prop_t) and len(finite_past_prop) >= max(8, int(MIN_MATCHED) + 1):
        support_low = float(np.quantile(finite_past_prop, 0.05))
        support_high = float(np.quantile(finite_past_prop, 0.95))
        support_margin = max(0.01, float(PROPENSITY_MAX_DIFF) * 0.50)
        if prop_t < support_low - support_margin or prop_t > support_high + support_margin:
            support_fail = True

    if support_fail:
        return [], {
            "reason": "outside_common_support",
            "support_low": support_low,
            "support_high": support_high,
            "support_margin": support_margin,
            "trial_propensity": prop_t,
            "covariates_used": "|".join(names),
        }

    strict_rows = []
    relaxed_rows = []
    support_rejects = 0
    caliper_rejects = 0
    pretrend_rejects = 0
    for i, xi in zip(keep, Xp):
        d = _robust_mahalanobis_like(xi, x_t, scales)
        if not np.isfinite(d):
            continue

        slope_gap = np.nan
        trial_pre = _pre_window_values(df, t_idx, pre_days, C.resolve_outcome_col(df, outcome_col or TARGET_COL))
        ctrl_pre = _pre_window_values(df, i, pre_days, C.resolve_outcome_col(df, outcome_col or TARGET_COL))
        if len(trial_pre) >= 3 and len(ctrl_pre) >= 3:
            trial_slope = _trend_slope(trial_pre)
            ctrl_slope = _trend_slope(ctrl_pre)
            if np.isfinite(trial_slope) and np.isfinite(ctrl_slope):
                slope_gap = abs(float(trial_slope) - float(ctrl_slope))
                if slope_gap <= max(0.20, PRETREND_MAX_DIFF * 2.0):
                    d += 0.30 * slope_gap
                else:
                    pretrend_rejects += 1
                    continue

        prop_diff = np.nan
        caliper_soft = float(PROPENSITY_MAX_DIFF)
        caliper_hard = max(0.01, caliper_soft * 0.80)
        if ENABLE_PROPENSITY and np.isfinite(prop_t) and np.isfinite(prop[i]):
            pi = float(prop[i])
            prop_diff = abs(prop_t - pi)
            if np.isfinite(support_low) and np.isfinite(support_high):
                if pi < support_low - support_margin or pi > support_high + support_margin:
                    support_rejects += 1
                    continue
            if prop_diff <= caliper_hard:
                d += 0.30 * prop_diff
            elif prop_diff <= caliper_soft:
                d += 0.70 * prop_diff
            else:
                caliper_rejects += 1
                continue

        row = (int(i), float(d), prop_diff, slope_gap)
        relaxed_rows.append(row)
        if d <= float(MATCH_DIST_MAX) and (not np.isfinite(prop_diff) or prop_diff <= caliper_hard):
            strict_rows.append(row)

    strict_rows.sort(key=lambda z: z[1])
    relaxed_rows.sort(key=lambda z: z[1])

    chosen = []
    reason = ""
    match_mode = "strict"
    candidate_pool_n = len(strict_rows)
    if len(strict_rows) >= int(MIN_MATCHED):
        chosen = strict_rows[:int(K_CONTROLS)]
    else:
        relaxed_cap = float(MATCH_DIST_MAX) * 1.20
        relaxed_use = [r for r in relaxed_rows if r[1] <= relaxed_cap]
        if len(relaxed_use) >= max(3, int(MIN_MATCHED) - 1):
            chosen = relaxed_use[:int(K_CONTROLS)]
            reason = "soft_match_fallback"
            match_mode = "soft"
            candidate_pool_n = len(relaxed_use)
        else:
            return [], {
                "reason": "too_few_matches",
                "n": int(len(strict_rows)),
                "relaxed_n": int(len(relaxed_use)),
                "support_low": support_low,
                "support_high": support_high,
                "support_margin": support_margin,
                "trial_propensity": prop_t,
                "support_rejects": int(support_rejects),
                "caliper_rejects": int(caliper_rejects),
                "pretrend_rejects": int(pretrend_rejects),
                "covariates_used": "|".join(names),
            }

    return [i for i, _, _, _ in chosen], {
        "reason": reason,
        "match_mode": match_mode,
        "n_candidates": int(candidate_pool_n),
        "avg_dist_top": float(np.mean([d for _, d, _, _ in chosen])),
        "avg_propensity_gap": float(np.nanmean([p for _, _, p, _ in chosen])) if any(np.isfinite(p) for _, _, p, _ in chosen) else np.nan,
        "avg_pretrend_gap": float(np.nanmean([s for _, _, _, s in chosen])) if any(np.isfinite(s) for _, _, _, s in chosen) else np.nan,
        "support_low": support_low,
        "support_high": support_high,
        "support_margin": support_margin,
        "trial_propensity": prop_t,
        "support_rejects": int(support_rejects),
        "caliper_rejects": int(caliper_rejects),
        "pretrend_rejects": int(pretrend_rejects),
        "covariates_used": "|".join(names),
    }


def _match_weights(df, t_idx, controls):
    if len(controls) == 0:
        return np.array([], dtype=float), np.nan, np.nan

    prop = pd.to_numeric(df.get("__propensity__", pd.Series([np.nan] * len(df))), errors="coerce").to_numpy(dtype=float)
    rank = np.arange(len(controls), dtype=float)
    base = np.exp(-0.18 * rank)

    if 0 <= int(t_idx) < len(prop) and np.isfinite(prop[t_idx]):
        gaps = []
        for ci in controls:
            if 0 <= int(ci) < len(prop) and np.isfinite(prop[int(ci)]):
                gaps.append(abs(float(prop[t_idx]) - float(prop[int(ci)])))
            else:
                gaps.append(np.nan)
        gaps = np.asarray(gaps, dtype=float)
        gp = np.where(np.isfinite(gaps), np.exp(-5.0 * gaps), 1.0)
        base = base * gp

    s = float(np.sum(base))
    if not np.isfinite(s) or s <= 0:
        w = np.full(len(controls), 1.0 / float(len(controls)))
    else:
        w = base / s
    eff_n = float(1.0 / np.sum(np.square(w))) if np.sum(np.square(w)) > 0 else np.nan
    return w, eff_n, float(np.max(w))


def _weighted_mean(x, w):
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    m = np.isfinite(x) & np.isfinite(w)
    if int(np.sum(m)) == 0:
        return np.nan
    xx = x[m]
    ww = w[m]
    s = float(np.sum(ww))
    if not np.isfinite(s) or s <= 0:
        return float(np.mean(xx))
    ww = ww / s
    return float(np.sum(ww * xx))


def _weighted_sd(x, w):
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    m = np.isfinite(x) & np.isfinite(w)
    if int(np.sum(m)) < 2:
        return np.nan
    xx = x[m]
    ww = w[m]
    s = float(np.sum(ww))
    if not np.isfinite(s) or s <= 0:
        return float(np.std(xx, ddof=1)) if len(xx) >= 2 else np.nan
    ww = ww / s
    mu = float(np.sum(ww * xx))
    var = float(np.sum(ww * np.square(xx - mu)))
    return float(math.sqrt(max(0.0, var)))


def _placebo_signed_z(deltas, effect, se):
    arr = np.asarray([v for v in deltas if np.isfinite(v)], dtype=float)
    if len(arr) < 4 or not np.isfinite(effect):
        return np.nan, np.nan
    placebo_z = _robust_z(effect, arr)
    p_two_sided = (np.sum(np.abs(arr) >= abs(effect)) + 1.0) / (len(arr) + 1.0)
    return float(placebo_z), float(p_two_sided)


def _bootstrap_effect_interval(trial_delta, ctrl_arr, weights, b=200):
    arr = np.asarray(ctrl_arr, dtype=float)
    w = np.asarray(weights, dtype=float)
    m = np.isfinite(arr) & np.isfinite(w)
    arr = arr[m]
    w = w[m]
    if len(arr) < 4:
        return np.nan, np.nan
    s = float(np.sum(w))
    if not np.isfinite(s) or s <= 0:
        w = np.full(len(arr), 1.0 / float(len(arr)))
    else:
        w = w / s
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    vals = []
    n = len(arr)
    for _ in range(max(80, int(b))):
        idx = rng.choice(np.arange(n), size=n, replace=True, p=w)
        vals.append(float(trial_delta - np.mean(arr[idx])))
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def _aipw_style_effect(df, t_idx, controls, window_days, outcome_col, covs):
    """Estimate a matched, bias-corrected event-study ATT proxy.

    The improved estimator is no longer a simple post-window mean comparison. It uses:
    1) matched historical controls,
    2) a difference-in-differences style delta (post minus pre),
    3) ridge bias correction on pre-treatment features,
    4) placebo distribution over matched controls for a more causal standardization.
    """
    pre_days = max(5, PRETREND_DAYS)
    trial_post = _window_mean(df, t_idx, window_days, outcome_col)
    trial_pre = _window_mean(df, max(0, int(t_idx) - pre_days), pre_days - 1, outcome_col)
    if not np.isfinite(trial_post) or not np.isfinite(trial_pre):
        return {
            "reason": "missing_trial_window",
            "effect": np.nan,
            "trial_mean": np.nan,
            "ctrl_mean_raw": np.nan,
            "ctrl_mean_adj": np.nan,
            "z_raw": np.nan,
            "effect_se_proxy": np.nan,
            "residual_term": np.nan,
            "match_effective_n": np.nan,
            "match_weight_max": np.nan,
            "bias_correction_applied": np.nan,
            "effect_median": np.nan,
            "valid_control_ids": [],
        }

    trial_delta = float(trial_post - trial_pre)

    x_t, used_covs = _build_anchor_features(df, t_idx, covs, pre_days, outcome_col=outcome_col)
    valid_covs = np.isfinite(x_t)
    xt = x_t[valid_covs]

    ctrl_posts = []
    ctrl_deltas = []
    ctrl_pre_means = []
    Xc = []
    valid_control_ids = []
    placebo_pool = []

    for ci in controls:
        c_post = _window_mean(df, ci, window_days, outcome_col)
        c_pre = _window_mean(df, max(0, int(ci) - pre_days), pre_days - 1, outcome_col)
        if not np.isfinite(c_post) or not np.isfinite(c_pre):
            continue
        xi, _ = _build_anchor_features(df, int(ci), covs, pre_days, outcome_col=outcome_col)
        if valid_covs.sum() > 0 and not np.all(np.isfinite(xi[valid_covs])):
            continue
        delta = float(c_post - c_pre)
        ctrl_posts.append(float(c_post))
        ctrl_pre_means.append(float(c_pre))
        ctrl_deltas.append(delta)
        Xc.append(xi[valid_covs])
        valid_control_ids.append(int(ci))
        placebo_pool.append(delta)

    if len(ctrl_deltas) < 3:
        return {
            "reason": "too_few_control_windows",
            "effect": np.nan,
            "trial_mean": float(trial_delta),
            "ctrl_mean_raw": np.nan,
            "ctrl_mean_adj": np.nan,
            "z_raw": np.nan,
            "effect_se_proxy": np.nan,
            "residual_term": np.nan,
            "match_effective_n": np.nan,
            "match_weight_max": np.nan,
            "bias_correction_applied": np.nan,
            "effect_median": np.nan,
            "valid_control_ids": valid_control_ids,
        }

    ctrl_arr = np.asarray(ctrl_deltas, dtype=float)
    Xc = np.asarray(Xc, dtype=float)
    w, eff_n, w_max = _match_weights(df, t_idx, valid_control_ids)
    if len(w) != len(ctrl_arr):
        w = np.full(len(ctrl_arr), 1.0 / float(len(ctrl_arr)))
        eff_n = float(len(ctrl_arr))
        w_max = float(np.max(w))

    ctrl_mean_raw = _weighted_mean(ctrl_arr, w)

    pred_t, pred_ctrl = _ridge_fit_predict(Xc, ctrl_arr, xt, l2=RIDGE_L2) if len(xt) else (np.nan, np.full(len(ctrl_arr), np.nan))
    residual_term = np.nan
    ctrl_mean_adj = ctrl_mean_raw
    bias_correction = np.nan
    if np.isfinite(pred_t) and np.all(np.isfinite(pred_ctrl)):
        residuals = ctrl_arr - pred_ctrl
        residual_term = _weighted_mean(residuals, w)
        ctrl_mean_adj = float(pred_t + residual_term)
        bias_correction = float(ctrl_mean_adj - ctrl_mean_raw)

    effect = float(trial_delta - ctrl_mean_adj)
    sd_ctrl = _weighted_sd(ctrl_arr, w)
    if np.isfinite(sd_ctrl) and np.isfinite(eff_n) and eff_n > 1:
        effect_se = float(sd_ctrl / math.sqrt(eff_n))
    else:
        effect_se = np.nan

    placebo_z, placebo_p = _placebo_signed_z(placebo_pool, effect, effect_se)
    z_raw = placebo_z if np.isfinite(placebo_z) else _robust_z(trial_delta, ctrl_arr)
    ci_low, ci_high = _bootstrap_effect_interval(trial_delta, ctrl_arr, w, b=BOOT_B)

    return {
        "reason": "",
        "effect": float(effect),
        "trial_mean": float(trial_delta),
        "ctrl_mean_raw": float(ctrl_mean_raw) if np.isfinite(ctrl_mean_raw) else np.nan,
        "ctrl_mean_adj": float(ctrl_mean_adj) if np.isfinite(ctrl_mean_adj) else np.nan,
        "z_raw": float(z_raw) if np.isfinite(z_raw) else np.nan,
        "effect_se_proxy": float(effect_se) if np.isfinite(effect_se) else np.nan,
        "residual_term": float(residual_term) if np.isfinite(residual_term) else np.nan,
        "match_effective_n": float(eff_n) if np.isfinite(eff_n) else np.nan,
        "match_weight_max": float(w_max) if np.isfinite(w_max) else np.nan,
        "bias_correction_applied": float(bias_correction) if np.isfinite(bias_correction) else np.nan,
        "effect_median": float(np.nanmedian(ctrl_arr)) if len(ctrl_arr) > 0 else np.nan,
        "valid_control_ids": valid_control_ids,
        "trial_post_mean": float(trial_post),
        "trial_pre_mean": float(trial_pre),
        "placebo_pvalue": float(placebo_p) if np.isfinite(placebo_p) else np.nan,
        "effect_ci_low": float(ci_low) if np.isfinite(ci_low) else np.nan,
        "effect_ci_high": float(ci_high) if np.isfinite(ci_high) else np.nan,
        "effect_sign_stable": int(np.isfinite(ci_low) and np.isfinite(ci_high) and ((ci_low > 0 and ci_high > 0) or (ci_low < 0 and ci_high < 0))),
    }
