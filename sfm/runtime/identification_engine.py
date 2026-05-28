
from __future__ import annotations
import math
from typing import Any, Dict

def compute_identification_score(
    overlap_ok: bool,
    balance_ok: bool,
    placebo_pass: bool,
    pretrend_ok: bool,
    valid_strata: int,
    exact_stratum_control: int,
    causal_stratum_control: int,
    prop_gap: float,
    balance_smd: float,
    shared_support_ratio: float = 0.0,
    design_strength: str = "low",
) -> float:
    score = 0.0
    score += 0.18 if overlap_ok else max(0.0, 0.12 - 0.25 * prop_gap)
    score += 0.16 if balance_ok else max(0.0, 0.12 - 0.22 * balance_smd)
    score += 0.12 if placebo_pass else 0.03
    score += 0.10 if pretrend_ok else 0.03
    score += min(0.18, 0.05 * max(0, valid_strata))
    score += min(0.10, 0.04 * max(0, exact_stratum_control))
    score += min(0.08, 0.03 * max(0, causal_stratum_control))
    score += 0.12 * max(0.0, min(1.0, shared_support_ratio))
    if design_strength == "high":
        score += 0.12
    elif design_strength == "medium":
        score += 0.06
    return round(max(0.0, min(1.0, score)), 3)

def classify_identification(score: float) -> str:
    if score >= 0.72:
        return "high"
    if score >= 0.48:
        return "medium"
    if score > 0.0:
        return "low"
    return "none"

def confounding_risk_from_components(
    score: float,
    overlap_ok: bool,
    balance_ok: bool,
    placebo_pass: bool,
    pretrend_ok: bool,
) -> str:
    penalties = sum(1 for x in [overlap_ok, balance_ok, placebo_pass, pretrend_ok] if not x)
    if score >= 0.72 and penalties <= 1:
        return "low"
    if score >= 0.48 and penalties <= 2:
        return "medium"
    return "high"

def sensitivity_ratio(
    effect: float,
    prop_gap: float,
    balance_smd: float,
    n_t: int,
    n_c: int,
    control_q: float,
    placebo_fail_rate: float,
    pretrend_ok: bool,
    stratum_support: float,
    exact_stratum_control: int = 0,
    overlap_ok: bool = False,
) -> float:
    support_term = 1.0 / max(1.0, math.sqrt(max(1, n_t + n_c)))
    denom = 0.08 + prop_gap + balance_smd + support_term + max(0.0, 0.60 - control_q) + 0.30 * placebo_fail_rate + max(0.0, 0.60 - stratum_support)
    if exact_stratum_control < 2:
        denom += 0.10
    if not overlap_ok:
        denom += 0.10
    if not pretrend_ok:
        denom += 0.12
    return round(abs(effect) / max(0.05, denom), 3)

def build_identification_assessment(
    *,
    overlap_ok: bool,
    balance_ok: bool,
    placebo_pass: bool,
    pretrend_ok: bool,
    valid_strata: int,
    exact_stratum_control: int,
    causal_stratum_control: int,
    prop_gap: float,
    balance_smd: float,
    shared_support_ratio: float,
    design_strength: str,
    support_treated: int,
    support_control: int,
    placebo_fail_rate: float,
    control_match_quality: float,
    effect_for_sensitivity: float,
    contrast_key: str,
) -> Dict[str, Any]:
    score = compute_identification_score(
        overlap_ok, balance_ok, placebo_pass, pretrend_ok, valid_strata,
        exact_stratum_control, causal_stratum_control, prop_gap, balance_smd,
        shared_support_ratio, design_strength,
    )
    ident = classify_identification(score)
    sensitivity = sensitivity_ratio(
        effect_for_sensitivity, prop_gap, balance_smd, support_treated, support_control,
        control_match_quality, placebo_fail_rate, pretrend_ok, shared_support_ratio,
        exact_stratum_control, overlap_ok,
    )
    conf_risk = confounding_risk_from_components(score, overlap_ok, balance_ok, placebo_pass, pretrend_ok)
    return {
        "identification_score": score,
        "identification_support": ident,
        "confounding_risk": conf_risk,
        "sensitivity_ratio": sensitivity,
        "overlap_ok": bool(overlap_ok),
        "balance_ok": bool(balance_ok),
        "pretrend_ok": bool(pretrend_ok),
        "placebo_pass": bool(placebo_pass),
        "valid_strata": int(valid_strata),
        "exact_stratum_control": int(exact_stratum_control),
        "causal_stratum_control": int(causal_stratum_control),
        "shared_support_ratio": round(float(shared_support_ratio), 3),
        "design_strength": str(design_strength),
        "contrast_key": str(contrast_key or ""),
        "support": {"treated": int(support_treated), "control": int(support_control)},
    }
