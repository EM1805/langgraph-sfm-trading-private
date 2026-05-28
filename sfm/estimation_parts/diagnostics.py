
from runtime_env import configure_scientific_runtime
configure_scientific_runtime()

import math

import numpy as np
from runtime_compat import assert_scientific_stack
assert_scientific_stack()
import pandas as pd

from . import common as C

MIN_CONFIDENCE_CONFIRMED = C.MIN_CONFIDENCE_CONFIRMED
MIN_CONFIDENCE_VALIDATED = C.MIN_CONFIDENCE_VALIDATED
MIN_SUCCESS_LB_CONFIRMED = C.MIN_SUCCESS_LB_CONFIRMED
MIN_SUCCESS_LB_VALIDATED = C.MIN_SUCCESS_LB_VALIDATED
MIN_TRIALS_CONFIRMED = C.MIN_TRIALS_CONFIRMED
MIN_TRIALS_VALIDATED = C.MIN_TRIALS_VALIDATED
OVERLAP_MIN_MARGIN = C.OVERLAP_MIN_MARGIN
ROBUSTNESS_MIN_RATIO = C.ROBUSTNESS_MIN_RATIO
ROSENBAUM_GAMMA_GRID = C.ROSENBAUM_GAMMA_GRID


def _covariate_balance(df, t_idx, controls, covs_trial):
    diffs = []
    for c in covs_trial:
        if c not in df.columns:
            continue
        col = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        vals = col[np.asarray(controls, dtype=int)]
        vals = vals[np.isfinite(vals)]
        if len(vals) < 3 or not np.isfinite(col[t_idx]):
            continue
        mu = float(np.mean(vals))
        sd = float(np.std(vals))
        if not np.isfinite(sd) or sd < 1e-9:
            continue
        diffs.append(abs(float(col[t_idx]) - mu) / sd)
    if len(diffs) == 0:
        return np.nan, 1
    smd = float(np.mean(diffs))
    return smd, int(smd <= 0.85)


def _overlap_check(df, t_idx, controls):
    prop = pd.to_numeric(
        df.get("__propensity__", pd.Series([np.nan] * len(df))),
        errors="coerce"
    ).to_numpy(dtype=float)

    if not np.isfinite(prop[t_idx]):
        return 1, 0.0, np.nan, np.nan, np.nan

    pcs = prop[np.asarray(controls, dtype=int)]
    pcs = pcs[np.isfinite(pcs)]
    if len(pcs) < 2:
        return 1, 0.0, float(prop[t_idx]), np.nan, np.nan

    p_t = float(prop[t_idx])
    p_min = float(np.min(pcs))
    p_max = float(np.max(pcs))

    if p_t < p_min:
        gap = float(p_min - p_t)
    elif p_t > p_max:
        gap = float(p_t - p_max)
    else:
        gap = 0.0

    ok = int(gap <= OVERLAP_MIN_MARGIN)
    return ok, float(gap), p_t, p_min, p_max


def _normal_approx_p_abs_z(z):
    if not np.isfinite(z):
        return np.nan
    zz = abs(float(z))
    cdf = 0.5 * (1.0 + math.erf(zz / math.sqrt(2.0)))
    p = 2.0 * (1.0 - cdf)
    return float(max(0.0, min(1.0, p)))


def _rosenbaum_sensitivity(effect, se_proxy, eff_n):
    """Rosenbaum-style hidden-bias proxy.

    This is *not* a formal Rosenbaum bounds implementation for matched pairs; it is an
    operational approximation based on attenuating the standardized effect under Gamma.
    """
    if not np.isfinite(effect) or not np.isfinite(se_proxy) or se_proxy <= 1e-9 or not np.isfinite(eff_n):
        return {
            "rosenbaum_n": np.nan,
            "rosenbaum_p_gamma_1": np.nan,
            "rosenbaum_gamma_critical": np.nan,
            "rosenbaum_pass": 0,
            "rosenbaum_method": "approx_proxy",
        }

    z = float(abs(effect) / se_proxy)
    p1 = _normal_approx_p_abs_z(z)
    gamma_critical = np.nan

    for gamma in ROSENBAUM_GAMMA_GRID:
        penalty = math.sqrt(float(gamma))
        z_g = z / penalty
        p_g = _normal_approx_p_abs_z(z_g)
        if np.isfinite(p_g) and p_g <= 0.05:
            gamma_critical = float(gamma)

    if not np.isfinite(gamma_critical):
        gamma_critical = 1.0

    passed = int(np.isfinite(gamma_critical) and gamma_critical >= 1.25)
    return {
        "rosenbaum_n": int(max(1, round(float(eff_n)))),
        "rosenbaum_p_gamma_1": float(p1) if np.isfinite(p1) else np.nan,
        "rosenbaum_gamma_critical": float(gamma_critical),
        "rosenbaum_pass": int(passed),
        "rosenbaum_method": "approx_proxy",
    }


def _sensitivity_metrics(effect, se_proxy):
    if not np.isfinite(effect) or not np.isfinite(se_proxy) or se_proxy <= 1e-9:
        return {
            "robustness_ratio": np.nan,
            "robustness_evalue_like": np.nan,
            "robustness_pass": 0,
            "sensitivity_null_shift": np.nan,
            "sensitivity_null_shift_sd": np.nan,
            "sensitivity_ci_low_shift": np.nan,
            "sensitivity_ci_low_shift_sd": np.nan,
        }

    ratio = float(abs(effect) / se_proxy)
    evalue_like = float(max(0.0, math.log1p(max(0.0, ratio))))
    ci_low_shift = float(max(0.0, abs(effect) - 1.96 * se_proxy))

    return {
        "robustness_ratio": ratio,
        "robustness_evalue_like": evalue_like,
        "robustness_pass": int(ratio >= ROBUSTNESS_MIN_RATIO),
        "sensitivity_null_shift": float(abs(effect)),
        "sensitivity_null_shift_sd": float(ratio),
        "sensitivity_ci_low_shift": float(ci_low_shift),
        "sensitivity_ci_low_shift_sd": float(max(0.0, ci_low_shift / se_proxy)),
    }


def _confidence_score(n_trials, success_lb, avg_signed_z, balance_pass, propensity_pass, pretrend_pass, negctrl_pass, l29_conf):
    """More permissive confidence aggregation for veto-layer screening.

    Pretrend and negative-control checks still matter, but they should degrade confidence
    rather than dominate it when evidence is merely sparse.
    """
    n_score = min(float(n_trials) / 4.0, 1.0)
    s_score = min(max(float(success_lb), 0.0), 1.0) if np.isfinite(success_lb) else 0.0
    z_score = min(max(float(avg_signed_z) / 2.5, 0.0), 1.0) if np.isfinite(avg_signed_z) else 0.0
    balance_score = float(np.clip(np.nanmean([float(balance_pass), float(propensity_pass)]), 0.0, 1.0))
    pretrend_score = float(np.clip(pretrend_pass, 0.0, 1.0))
    negctrl_score = float(np.clip(negctrl_pass, 0.0, 1.0))
    gate_score = 0.55 * balance_score + 0.10 * pretrend_score + 0.15 * negctrl_score + 0.20 * float(np.clip(propensity_pass, 0.0, 1.0))
    l29_score = min(max(float(l29_conf), 0.0), 1.0) if np.isfinite(l29_conf) else 0.35
    return float(0.20 * n_score + 0.22 * s_score + 0.16 * z_score + 0.26 * gate_score + 0.16 * l29_score)


def _risk_level(status, negctrl_pass, pretrend_pass, direction_consistent, conf):
    if status == "rejected":
        return "HIGH"
    if negctrl_pass != 1 and pretrend_pass < 0.5 and not direction_consistent:
        return "HIGH"
    if status in ("confirmed", "validated") and conf >= 0.70:
        return "LOW"
    if status == "exploratory":
        return "MEDIUM"
    return "MEDIUM"


def _status_from_metrics(n_trials, success_lb, conf, negctrl_pass, direction_consistency_rate, weak_support_rate):
    """Return softer evidence statuses for a veto-layer workflow.

    Most sparse/uncertain cases should surface as exploratory or candidate rather than
    hard rejections. Keep `rejected` for genuinely hard structural signals only.
    """
    if n_trials <= 0:
        return "exploratory", ["NO_EVALUABLE_TRIALS", "INSUFFICIENT_EVIDENCE"]

    reasons = []
    inconsistent_direction = bool(np.isfinite(direction_consistency_rate) and direction_consistency_rate < 0.5)
    weak_support = bool(np.isfinite(weak_support_rate) and weak_support_rate < 0.5)

    if negctrl_pass != 1:
        reasons.append("NEGCTRL_RISK")
    if inconsistent_direction:
        reasons.append("INCONSISTENT_DIRECTION")
    if weak_support:
        reasons.append("WEAK_SUPPORT")

    hard_structural = (negctrl_pass != 1 and inconsistent_direction and conf < 0.20)
    if hard_structural:
        return "rejected", reasons + ["BLOCKED_BY_SAFETY"]

    if n_trials < MIN_TRIALS_VALIDATED:
        reasons.append("LOW_SAMPLE")
        if conf >= 0.22:
            return "exploratory", reasons + (["INSUFFICIENT_EVIDENCE"] if conf < 0.35 else [])
        return "candidate", reasons + ["INSUFFICIENT_EVIDENCE"]

    if success_lb < 0.20:
        return "exploratory", reasons + ["LOW_SUCCESS", "INSUFFICIENT_EVIDENCE"]
    if success_lb < 0.34 or weak_support:
        return "candidate", reasons + ["LOW_SUCCESS"]

    if n_trials >= MIN_TRIALS_CONFIRMED and success_lb >= MIN_SUCCESS_LB_CONFIRMED and conf >= MIN_CONFIDENCE_CONFIRMED and negctrl_pass == 1 and not inconsistent_direction:
        return "confirmed", reasons or ["STRONG_COUNTERFACTUAL_SUPPORT"]
    if n_trials >= MIN_TRIALS_VALIDATED and success_lb >= MIN_SUCCESS_LB_VALIDATED and conf >= MIN_CONFIDENCE_VALIDATED and negctrl_pass == 1 and not inconsistent_direction:
        return "validated", reasons or ["CONSISTENT_COUNTERFACTUAL_SUPPORT"]

    if conf >= 0.28:
        return "candidate", reasons or ["NEEDS_MORE_EVIDENCE"]
    return "exploratory", reasons + ["LOW_CONFIDENCE", "INSUFFICIENT_EVIDENCE"]
