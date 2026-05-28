"""Sensitivity analysis planning for unobserved confounding.

This module is intentionally conservative: it does not claim that an effect is
causal. It turns the Identification/Contract handoff into a transparent
robustness plan so downstream estimators know which sensitivity checks are
required before any stronger claim is made.
"""
from __future__ import annotations

from runtime_env import configure_scientific_runtime
configure_scientific_runtime()


import os
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from . import common as C
from . import _utils as U

OUT_DIR = C.OUT_DIR
SENSITIVITY_ANALYSIS_CSV = os.path.join(OUT_DIR, "estimation", "sensitivity_analysis.csv")

SENSITIVITY_COLUMNS = [
    "sensitivity_id", "insight_id", "source", "target", "treatment_col", "outcome_col", "lag",
    "authority_level", "estimation_enabled", "identification_status", "estimand_type",
    "adjustment_set", "adjustment_set_status", "candidate_adjustment_set", "backdoor_status",
    "negative_controls", "forbidden_adjustment_set", "mci_status", "mci_q_value", "mci_n_eff",
    "pc1_parent_support", "scm_role_hint", "unobserved_confounding_risk", "sensitivity_level",
    "sensitivity_status", "recommended_sensitivity_method", "minimum_report_before_effect_claim",
    "reason",
]


def _as_str(value) -> str:
    return U.as_str(value).strip()


def _boolish(value) -> bool:
    return _as_str(value).lower() in {"1", "true", "yes", "y", "on"}


def _safe_float(value, default=np.nan) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    return x if np.isfinite(x) else default


def _first(row, keys: Iterable[str], default: str = "") -> str:
    for key in keys:
        try:
            value = _as_str(row.get(key, ""))
        except (AttributeError, TypeError, ValueError):
            value = ""
        if value and value.lower() not in {"nan", "none", "null"}:
            return value
    return default


def classify_sensitivity(row) -> tuple[str, str, str, str, str]:
    """Return risk, level, status, method, reason for a handoff/plan row."""
    authority = _as_str(row.get("authority_level", "")).lower()
    enabled = _boolish(row.get("estimation_enabled", "")) or _boolish(row.get("eligible_for_estimation", ""))
    identified = _boolish(row.get("identified", ""))
    adj_status = _as_str(row.get("adjustment_set_status", "")).lower()
    backdoor = _as_str(row.get("backdoor_status", "")).lower()
    strategy = _as_str(row.get("identification_strategy", "")).lower()
    estimand = _as_str(row.get("estimand_type", "")).lower()
    mci_status = _as_str(row.get("mci_status", "")).lower()
    q = _safe_float(row.get("mci_q_value", np.nan))
    neg = _as_str(row.get("negative_controls", ""))
    forbidden = _as_str(row.get("forbidden_adjustment_set", ""))
    blocked_by = _as_str(row.get("blocked_by", "")).lower()
    assumptions = _as_str(row.get("assumption_notes", "")).lower()
    role = _as_str(row.get("scm_role_hint", "")).lower()

    if authority in {"raw_discovery_only", "discovery_only", "weak_or_unaligned"} or "hard_blocked" in role:
        return ("not_applicable", "none", "not_applicable_no_estimation_authority", "none", "row_is_not_authorized_for_estimation")
    if authority == "graph_review" or "collider" in blocked_by:
        return ("high", "strong", "required_before_estimation", "graph_review_plus_unobserved_confounding_stress_test", "graph_review_or_collider_warning_requires_sensitivity_before_effect_estimate")
    if not enabled and not identified and authority != "identified_needs_estimation":
        return ("medium", "planning", "plan_only_until_identification_enabled", "diagnostic_e_value_or_omitted_confounder_grid", "not_formally_estimable_yet_but_structural_prior_can_be_reviewed")

    risk_points = 0
    reasons = []
    if forbidden:
        risk_points += 2; reasons.append("forbidden_adjustments_present")
    if neg:
        risk_points += 1; reasons.append("negative_control_available_or_flagged")
    if mci_status in {"diagnostic_fail", "fail"}:
        risk_points += 2; reasons.append("mci_diagnostic_fail")
    elif mci_status == "diagnostic_support":
        risk_points += 1; reasons.append("mci_only_diagnostic_support")
    if np.isfinite(q) and q > 0.15:
        risk_points += 1; reasons.append("mci_q_above_pass_threshold")
    if adj_status not in {"valid_empty", "valid_nonempty"} and backdoor not in {"backdoor_adjustment_candidate", "adjustment_candidate", "backdoor_valid"}:
        risk_points += 2; reasons.append("no_valid_adjustment_set")
    if "latent" in assumptions or "unobserved" in assumptions:
        risk_points += 2; reasons.append("latent_or_unobserved_confounding_note")
    if "frontdoor" in strategy or "frontdoor" in estimand:
        risk_points += 1; reasons.append("frontdoor_limited_route")

    if risk_points >= 4:
        return ("high", "strong", "required_before_effect_claim", "omitted_confounder_grid_plus_negative_control_stress", "|".join(reasons) or "high_unobserved_confounding_risk")
    if risk_points >= 2:
        return ("medium", "standard", "required_before_strong_claim", "e_value_or_rosenbaum_bounds_plus_covariate_robustness", "|".join(reasons) or "medium_unobserved_confounding_risk")
    return ("low", "light", "recommended_as_reporting_check", "covariate_drop_one_and_e_value_light", "adjustment_set_and_mci_support_are_reasonable_but_not_proof")


def build_sensitivity_analysis(handoff_or_plan: Optional[pd.DataFrame]) -> pd.DataFrame:
    if handoff_or_plan is None or len(handoff_or_plan) == 0:
        return pd.DataFrame(columns=SENSITIVITY_COLUMNS)
    rows = []
    for idx, row in handoff_or_plan.iterrows():
        source = _first(row, ["source", "treatment_col"])
        target = _first(row, ["target", "outcome_col"])
        lag = _first(row, ["lag"], "")
        iid = _first(row, ["insight_id", "plan_id"], f"{source}->{target}@{lag or '0'}")
        risk, level, status, method, reason = classify_sensitivity(row)
        rows.append({
            "sensitivity_id": f"sensitivity::{iid}",
            "insight_id": iid,
            "source": source,
            "target": target,
            "treatment_col": _first(row, ["treatment_col", "source"], source),
            "outcome_col": _first(row, ["outcome_col", "target"], target),
            "lag": lag,
            "authority_level": _first(row, ["authority_level"]),
            "estimation_enabled": _first(row, ["estimation_enabled"]),
            "identification_status": _first(row, ["identification_status"]),
            "estimand_type": _first(row, ["estimand_type"]),
            "adjustment_set": _first(row, ["adjustment_set", "total_adjustment_set"]),
            "adjustment_set_status": _first(row, ["adjustment_set_status"]),
            "candidate_adjustment_set": _first(row, ["candidate_adjustment_set"]),
            "backdoor_status": _first(row, ["backdoor_status"]),
            "negative_controls": _first(row, ["negative_controls"]),
            "forbidden_adjustment_set": _first(row, ["forbidden_adjustment_set"]),
            "mci_status": _first(row, ["mci_status"]),
            "mci_q_value": _first(row, ["mci_q_value"]),
            "mci_n_eff": _first(row, ["mci_n_eff"]),
            "pc1_parent_support": _first(row, ["pc1_parent_support"]),
            "scm_role_hint": _first(row, ["scm_role_hint"]),
            "unobserved_confounding_risk": risk,
            "sensitivity_level": level,
            "sensitivity_status": status,
            "recommended_sensitivity_method": method,
            "minimum_report_before_effect_claim": "required" if status.startswith("required") else "recommended" if status.startswith("recommended") else "not_applicable",
            "reason": reason,
        })
    return pd.DataFrame(rows, columns=SENSITIVITY_COLUMNS)


def write_sensitivity_analysis(handoff_or_plan: Optional[pd.DataFrame], out_dir: str = OUT_DIR) -> str:
    path = os.path.join(out_dir, "estimation", "sensitivity_analysis.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = build_sensitivity_analysis(handoff_or_plan)
    out.to_csv(path, index=False)
    return path
