"""Insight-level aggregation helpers for Estimation Level 3.2."""
from __future__ import annotations

from runtime_env import configure_scientific_runtime
configure_scientific_runtime()


from typing import Dict

import numpy as np
import pandas as pd

from . import common as C
from . import propensity as P
from . import diagnostics as D
from . import _utils as U

BOOT_B = C.BOOT_B
DAG_RISK_CONF_PENALTY = C.DAG_RISK_CONF_PENALTY
DAG_RISK_FORCE_WARNING = C.DAG_RISK_FORCE_WARNING
NEGCTRL_ENABLE = C.NEGCTRL_ENABLE
NEGCTRL_MAX_SUCCESS_LB = C.NEGCTRL_MAX_SUCCESS_LB

_as_str = U.as_str
_safe_float = U.safe_float
_mode_or_empty = U.mode_or_empty
_sr_lower_bound = C._sr_lower_bound
_bootstrap_ci = P._bootstrap_ci
_confidence_score = D._confidence_score
_status_from_metrics = D._status_from_metrics
_risk_level = D._risk_level

def _aggregate_insight_identity(iid: str, g: pd.DataFrame, l29_map: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    row0 = g.iloc[0] if len(g) > 0 else pd.Series(dtype=object)
    l29 = l29_map.get(iid, {}) if isinstance(l29_map, dict) else {}
    source = _as_str(row0.get("source", row0.get("dag_action_source", l29.get("source", ""))))
    action_name = _as_str(row0.get("action_name", l29.get("action_name", source)))
    source_family = _as_str(row0.get("source_family", l29.get("source_family", "")))
    action_family = _as_str(row0.get("action_family", l29.get("action_family", "")))
    if not source_family and source:
        source_family = source.split("_")[0].lower()
    if not action_family and action_name:
        action_family = action_name.split("_")[0].lower()
    return {
        "source": source,
        "action_name": action_name,
        "source_family": source_family,
        "action_family": action_family,
    }

def _aggregate_insight(iid, g, l29_map):
    g_eval = g[
        (pd.to_numeric(g["z_cf"], errors="coerce").notna())
        & (pd.to_numeric(g["eligible_flag"], errors="coerce").fillna(0).astype(int) == 1)
    ].copy()

    n = int(len(g_eval))
    wins = int(pd.to_numeric(g_eval.get("success_flag", 0), errors="coerce").fillna(0).astype(int).sum()) if n > 0 else 0
    success_lb = _sr_lower_bound(wins, n) if n > 0 else 0.0
    avg_signed_z = float(np.nanmean(pd.to_numeric(g_eval["z_cf"], errors="coerce").to_numpy(dtype=float))) if n > 0 else np.nan
    median_signed_z = float(np.nanmedian(pd.to_numeric(g_eval["z_cf"], errors="coerce").to_numpy(dtype=float))) if n > 0 else np.nan

    effect_values = pd.to_numeric(g_eval.get("effect_window", np.nan), errors="coerce").to_numpy(dtype=float) if n > 0 else np.array([])
    effect_mean = float(np.nanmean(effect_values)) if n > 0 else np.nan
    ci_lo, ci_hi = _bootstrap_ci(effect_values, b=BOOT_B, alpha=0.10) if n > 0 else (np.nan, np.nan)

    negctrl_lb = np.nan
    if NEGCTRL_ENABLE and "success_flag_negctrl" in g.columns:
        gnc = g[pd.to_numeric(g["success_flag_negctrl"], errors="coerce").notna()].copy()
        nn = int(len(gnc))
        if nn > 0:
            wins_nc = int(pd.to_numeric(gnc["success_flag_negctrl"], errors="coerce").fillna(0).astype(int).sum())
            negctrl_lb = _sr_lower_bound(wins_nc, nn)

    negctrl_pass = int((not np.isfinite(negctrl_lb)) or (float(negctrl_lb) <= float(NEGCTRL_MAX_SUCCESS_LB)))
    balance_pass_rate = float(pd.to_numeric(g.get("balance_pass", 0), errors="coerce").fillna(0).astype(int).mean()) if len(g) else np.nan
    propensity_pass_rate = float(pd.to_numeric(g.get("propensity_pass", 1), errors="coerce").fillna(1).astype(int).mean()) if len(g) else np.nan
    pretrend_pass_rate = float(pd.to_numeric(g.get("pretrend_pass", 1), errors="coerce").fillna(1).astype(int).mean()) if len(g) else np.nan
    overlap_pass_rate = float(pd.to_numeric(g.get("overlap_pass", 1), errors="coerce").fillna(1).astype(int).mean()) if len(g) else np.nan
    direction_match_rate = float(pd.to_numeric(g_eval.get("direction_ok_flag", 0), errors="coerce").fillna(0).astype(int).mean()) if n > 0 else np.nan
    weak_support_rate = float(pd.to_numeric(g_eval.get("success_flag", 0), errors="coerce").fillna(0).astype(int).mean()) if n > 0 else np.nan

    robustness_values = pd.to_numeric(g.get("robustness_ratio", np.nan), errors="coerce").to_numpy(dtype=float) if len(g) else np.array([])
    robustness_values = robustness_values[np.isfinite(robustness_values)]
    robustness_min_ratio = float(np.min(robustness_values)) if robustness_values.size else np.nan
    sign_stable_rate = float(pd.to_numeric(g.get("effect_sign_stable", 0), errors="coerce").fillna(0).astype(int).mean()) if len(g) else np.nan
    placebo_values = pd.to_numeric(g.get("placebo_pvalue", np.nan), errors="coerce").to_numpy(dtype=float) if len(g) else np.array([])
    placebo_values = placebo_values[np.isfinite(placebo_values)]
    placebo_p_median = float(np.median(placebo_values)) if placebo_values.size else np.nan

    l29 = l29_map.get(iid, {})
    l29_conf = _safe_float(l29.get("confidence_score", np.nan), np.nan)
    conf = _confidence_score(
        n,
        success_lb,
        avg_signed_z,
        balance_pass_rate if np.isfinite(balance_pass_rate) else 0,
        propensity_pass_rate if np.isfinite(propensity_pass_rate) else 1,
        pretrend_pass_rate if np.isfinite(pretrend_pass_rate) else 1,
        negctrl_pass,
        l29_conf,
    )
    if np.isfinite(sign_stable_rate):
        conf = float(min(1.0, max(0.0, conf + 0.06 * (sign_stable_rate - 0.5))))
    if np.isfinite(placebo_p_median) and placebo_p_median > 0.25:
        conf = float(max(0.0, conf - 0.05))

    dag_rows = g[[c for c in ["dag_risk_paths", "dag_covariate_violation_flag", "dag_adjustment_set", "dag_forbidden_adjustments", "dag_negative_control", "discovery_adjustment_set", "discovery_forbidden_adjustments", "discovery_negative_control"] if c in g.columns]].copy() if len(g) else pd.DataFrame()
    dag_risk_any = int(dag_rows.get("dag_risk_paths", pd.Series([], dtype=str)).astype(str).str.len().gt(0).any()) if len(dag_rows) else 0
    dag_cov_violation_any = int(pd.to_numeric(dag_rows.get("dag_covariate_violation_flag", 0), errors="coerce").fillna(0).astype(int).max()) if len(dag_rows) else 0
    dag_negative_control_any = "|".join(sorted(set([x for x in dag_rows.get("dag_negative_control", pd.Series([], dtype=str)).astype(str).tolist() if x and x != "nan"]))) if len(dag_rows) else ""
    discovery_adjustment_any = "|".join(sorted(set([x for x in dag_rows.get("discovery_adjustment_set", pd.Series([], dtype=str)).astype(str).tolist() if x and x != "nan"]))) if len(dag_rows) else ""
    discovery_forbidden_any = "|".join(sorted(set([x for x in dag_rows.get("discovery_forbidden_adjustments", pd.Series([], dtype=str)).astype(str).tolist() if x and x != "nan"]))) if len(dag_rows) else ""
    discovery_negative_control_any = "|".join(sorted(set([x for x in dag_rows.get("discovery_negative_control", pd.Series([], dtype=str)).astype(str).tolist() if x and x != "nan"]))) if len(dag_rows) else ""

    diagnostic_series = g.get("diagnostic_grade", pd.Series([], dtype=str)).astype(str) if len(g) else pd.Series([], dtype=str)
    diagnostic_grade_counts = {k: int((diagnostic_series == k).sum()) for k in ["strong", "moderate", "weak", "fail"]}
    if diagnostic_grade_counts["fail"] > 0:
        group_diagnostic_grade = "fail"
    elif diagnostic_grade_counts["weak"] > 0:
        group_diagnostic_grade = "weak"
    elif diagnostic_grade_counts["moderate"] > 0:
        group_diagnostic_grade = "moderate"
    elif diagnostic_grade_counts["strong"] > 0:
        group_diagnostic_grade = "strong"
    else:
        group_diagnostic_grade = "unknown"

    sensitivity_series = g.get("sensitivity_level", pd.Series([], dtype=str)).astype(str) if len(g) else pd.Series([], dtype=str)
    if (sensitivity_series == "high").any():
        group_sensitivity_level = "high"
    elif (sensitivity_series == "medium").any():
        group_sensitivity_level = "medium"
    elif (sensitivity_series == "low").any():
        group_sensitivity_level = "low"
    else:
        group_sensitivity_level = "unknown"

    identification_series = g.get("identification_strength", pd.Series([], dtype=str)).astype(str) if len(g) else pd.Series([], dtype=str)
    identification_strength_counts = {k: int((identification_series == k).sum()) for k in ["strong", "moderate", "weak", "none"]}
    if identification_strength_counts["none"] > 0:
        group_identification_strength = "none"
    elif identification_strength_counts["weak"] > 0:
        group_identification_strength = "weak"
    elif identification_strength_counts["moderate"] > 0:
        group_identification_strength = "moderate"
    elif identification_strength_counts["strong"] > 0:
        group_identification_strength = "strong"
    else:
        group_identification_strength = "unknown"

    strategy_series = g.get("identification_strategy", pd.Series([], dtype=str)).astype(str) if len(g) else pd.Series([], dtype=str)
    group_identification_strategy = strategy_series.value_counts().index[0] if len(strategy_series) else "unknown"
    identifiable_rate = float(pd.to_numeric(g.get("identifiable", 0), errors="coerce").fillna(0).astype(int).mean()) if len(g) else 0.0
    pearl_authorized_rate = float(pd.to_numeric(g.get("pearl_estimation_authorized", 0), errors="coerce").fillna(0).astype(int).mean()) if len(g) else 0.0
    contract_required_any = int(pd.to_numeric(g.get("contract_required", 0), errors="coerce").fillna(0).astype(int).max()) if len(g) else 0
    contract_row_present_rate = float(pd.to_numeric(g.get("contract_row_present", 0), errors="coerce").fillna(0).astype(int).mean()) if len(g) else 0.0
    graph_authority_level = _mode_or_empty(g.get("graph_authority_level", pd.Series([], dtype=str)).astype(str)) if len(g) else ""
    contract_source_authority = _mode_or_empty(g.get("contract_source_authority", pd.Series([], dtype=str)).astype(str)) if len(g) else ""
    pearl_authority_reason = _mode_or_empty(g.get("pearl_authority_reason", pd.Series([], dtype=str)).astype(str)) if len(g) else ""
    causal_claim_status = "identified_estimated" if pearl_authorized_rate > 0 and n > 0 else "diagnostic_only_not_pearl_authorized"

    if dag_risk_any:
        conf = float(max(0.0, conf - DAG_RISK_CONF_PENALTY))

    status, reason_list = _status_from_metrics(n, success_lb, conf, negctrl_pass, direction_match_rate, weak_support_rate)
    if pearl_authorized_rate <= 0:
        status = "rejected" if contract_required_any else status
        conf = min(float(conf), 0.25) if np.isfinite(conf) else 0.0
        if "PEARL_NOT_AUTHORIZED" not in reason_list:
            reason_list.append("PEARL_NOT_AUTHORIZED")
        if pearl_authority_reason and pearl_authority_reason.upper() not in reason_list:
            reason_list.append(pearl_authority_reason.upper())
    if dag_risk_any:
        if status == "confirmed":
            status = "validated"
        elif status == "validated":
            status = "candidate"
        if "DAG_RISK_PATH" not in reason_list:
            reason_list.append("DAG_RISK_PATH")

    decision = {
        "confirmed": "approve",
        "validated": "approve_with_warning",
        "candidate": "hold",
        "exploratory": "hold",
        "rejected": "reject",
    }[status]
    if dag_risk_any and DAG_RISK_FORCE_WARNING and decision == "approve":
        decision = "approve_with_warning"

    risk_level = _risk_level(
        status,
        negctrl_pass,
        pretrend_pass_rate if np.isfinite(pretrend_pass_rate) else 1,
        bool(np.isfinite(direction_match_rate) and direction_match_rate >= 0.5),
        conf,
    )
    if dag_risk_any and risk_level == "LOW":
        risk_level = "MEDIUM"

    ident = _aggregate_insight_identity(iid, g, l29_map)
    feedback_weight = U.bounded_weighted_score(
        confidence=conf,
        success_lb=success_lb,
        identifiable_rate=identifiable_rate,
        direction_match_rate=direction_match_rate,
        balance_pass_rate=balance_pass_rate,
    )
    return {
        "insight_id": iid,
        **ident,
        "feedback_weight": feedback_weight,
        "estimand": _mode_or_empty(g.get("estimand", pd.Series([], dtype=str)).astype(str)) if len(g) else "ATT_proxy",
        "causal_claim_status": causal_claim_status,
        "pearl_authorized_rate": float(pearl_authorized_rate),
        "contract_required_any": int(contract_required_any),
        "contract_row_present_rate": float(contract_row_present_rate),
        "contract_source_authority": contract_source_authority,
        "graph_authority_level": graph_authority_level,
        "pearl_authority_reason": pearl_authority_reason,
        "n_trials": int(n),
        "n_trials_total": int(len(g)),
        "n_trials_eligible": int(n),
        "n_wins": int(wins),
        "success_rate_lb": float(success_lb),
        "avg_z_cf": float(avg_signed_z) if np.isfinite(avg_signed_z) else np.nan,
        "median_z_cf": float(median_signed_z) if np.isfinite(median_signed_z) else np.nan,
        "att_mean": float(effect_mean) if np.isfinite(effect_mean) else np.nan,
        "effect_mean": float(effect_mean) if np.isfinite(effect_mean) else np.nan,
        "effect_ci_low": float(ci_lo) if np.isfinite(ci_lo) else np.nan,
        "effect_ci_high": float(ci_hi) if np.isfinite(ci_hi) else np.nan,
        "direction_match_rate": float(direction_match_rate) if np.isfinite(direction_match_rate) else np.nan,
        "weak_support_rate": float(weak_support_rate) if np.isfinite(weak_support_rate) else np.nan,
        "balance_pass_rate": float(balance_pass_rate) if np.isfinite(balance_pass_rate) else np.nan,
        "propensity_pass_rate": float(propensity_pass_rate) if np.isfinite(propensity_pass_rate) else np.nan,
        "pretrend_pass_rate": float(pretrend_pass_rate) if np.isfinite(pretrend_pass_rate) else np.nan,
        "overlap_pass_rate": float(overlap_pass_rate) if np.isfinite(overlap_pass_rate) else np.nan,
        "robustness_min_ratio": float(robustness_min_ratio) if np.isfinite(robustness_min_ratio) else np.nan,
        "effect_sign_stable_rate": float(sign_stable_rate) if np.isfinite(sign_stable_rate) else np.nan,
        "placebo_pvalue_median": float(placebo_p_median) if np.isfinite(placebo_p_median) else np.nan,
        "negctrl_success_lb": float(negctrl_lb) if np.isfinite(negctrl_lb) else np.nan,
        "negctrl_pass": int(negctrl_pass),
        "confidence": float(conf),
        "status": status,
        "decision": decision,
        "risk_level": risk_level,
        "reason_codes": "|".join(reason_list),
        "l29_confidence": float(l29_conf) if np.isfinite(l29_conf) else np.nan,
        "dag_risk_any": int(dag_risk_any),
        "dag_covariate_violation_any": int(dag_cov_violation_any),
        "dag_negative_control_any": dag_negative_control_any,
        "dag_action_known_rate": float(pd.to_numeric(g.get("dag_action_known", 0), errors="coerce").fillna(0).mean()) if len(g) else np.nan,
        "dag_target_known_rate": float(pd.to_numeric(g.get("dag_target_known", 0), errors="coerce").fillna(0).mean()) if len(g) else np.nan,
        "dag_adjustment_confidence": _mode_or_empty(g.get("dag_adjustment_confidence", pd.Series([], dtype=str)).astype(str)) if len(g) else "",
        "dag_adjustment_source": _mode_or_empty(g.get("dag_adjustment_source", pd.Series([], dtype=str)).astype(str)) if len(g) else "",
        "dag_adjustment_notes": _mode_or_empty(g.get("dag_adjustment_notes", pd.Series([], dtype=str)).astype(str)) if len(g) else "",
        "discovery_adjustment_set": discovery_adjustment_any,
        "discovery_forbidden_adjustments": discovery_forbidden_any,
        "discovery_negative_control": discovery_negative_control_any,
        "dag_path_confidence": _mode_or_empty(g.get("dag_path_confidence", pd.Series([], dtype=str)).astype(str)) if len(g) else "",
        "diagnostic_grade": group_diagnostic_grade,
        "sensitivity_level": group_sensitivity_level,
        "identification_strategy": group_identification_strategy,
        "identification_strength": group_identification_strength,
        "identifiable_rate": identifiable_rate,
        "diagnostic_fail_count": diagnostic_grade_counts["fail"],
        "identification_none_count": identification_strength_counts["none"],
        "identification_weak_count": identification_strength_counts["weak"],
        "identification_moderate_count": identification_strength_counts["moderate"],
        "identification_strong_count": identification_strength_counts["strong"],
        "diagnostic_weak_count": diagnostic_grade_counts["weak"],
        "diagnostic_moderate_count": diagnostic_grade_counts["moderate"],
        "diagnostic_strong_count": diagnostic_grade_counts["strong"],
    }

def _aggregate_all_insights(df_trials: pd.DataFrame, l29_map: Dict[str, Dict[str, object]]) -> pd.DataFrame:
    out_rows = []
    if len(df_trials) > 0:
        for iid, g in df_trials.groupby("insight_id"):
            out_rows.append(_aggregate_insight(iid, g, l29_map))
    return pd.DataFrame(out_rows)