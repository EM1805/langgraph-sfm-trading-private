"""Backdoor estimators for the Pearl-oriented estimation layer.

This module is intentionally lightweight (numpy/pandas only).  It consumes the
canonical causal contract and estimates only effects that the contract marks as
formally identified and compatible with a backdoor / adjustment estimand.

It does not discover graph structure and it does not authorize causal claims on
its own.  It only quantifies effects that SCM/Identification have already
allowed.
"""
from __future__ import annotations

from runtime_env import configure_scientific_runtime
configure_scientific_runtime()


import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from . import _utils as U
from . import stat_core as SC


BACKDOOR_STRATEGY_TOKENS = (
    "backdoor",
    "adjustment",
    "identified_backdoor",
    "dsep_backdoor",
)

RAW_AUTHORITY_TOKENS = {"raw_discovery_only", "pcmci_raw_input_only", "discovery_only"}
AUTHORIZED_LEVELS = {"identified_estimable"}
AUTHORIZED_STATUSES = {"identified", "backdoor_identified", "identified_backdoor"}


_as_str = U.as_str
_safe_float = U.safe_float
parse_list = U.parse_list

def _adjustment_status_is_valid(row) -> bool:
    status = _as_str(row.get("adjustment_set_status", "")).lower()
    if status in {"valid_empty", "valid_nonempty"}:
        return True
    if status in {"missing", "invalid"}:
        return False
    if parse_list(row.get("adjustment_set", row.get("suggested_adjustment_set", ""))):
        return True
    mins = _as_str(row.get("minimal_adjustment_sets", ""))
    return mins.startswith("[[") or mins == "[[]]"


def _contract_row_authorized(row: pd.Series) -> Tuple[bool, str]:
    source_authority = _as_str(row.get("source_authority", row.get("contract_source_authority", ""))).lower()
    parts = {p.strip() for p in source_authority.split("|") if p.strip()}
    if parts and parts.issubset(RAW_AUTHORITY_TOKENS):
        return False, "raw_discovery_not_pearl_authority"

    authority_level = _as_str(row.get("authority_level", row.get("graph_authority_level", ""))).lower()
    status = _as_str(row.get("identification_status", row.get("graph_identification_status", ""))).lower()
    identified = int(_safe_float(row.get("identified", row.get("graph_identified", 0)), 0)) == 1
    strategy = _as_str(row.get("identification_strategy", row.get("graph_identification_strategy", ""))).lower()
    adj = parse_list(row.get("adjustment_set", row.get("suggested_adjustment_set", "")))
    estimator_enabled = _as_str(row.get("estimation_enabled", "")).lower() in {"1", "true", "yes"}
    adjustment_valid = _adjustment_status_is_valid(row)
    backdoor_like = ("backdoor" in strategy) or adjustment_valid or bool(adj)
    if authority_level == "identified_estimable" and (estimator_enabled or (backdoor_like and adjustment_valid)):
        return True, "formal_backdoor_identification_authorized"
    if identified and backdoor_like and adjustment_valid and authority_level != "identified_needs_estimation":
        return True, "formal_backdoor_identification_authorized"
    if identified or authority_level == "identified_needs_estimation" or status in AUTHORIZED_STATUSES:
        return False, "identified_but_no_enabled_backdoor_estimator"
    return False, "contract_present_but_not_formally_identified"


def _is_backdoor_estimand(row: pd.Series) -> Tuple[bool, str]:
    strategy = _as_str(row.get("identification_strategy", row.get("graph_identification_strategy", ""))).lower()
    estimand = _as_str(row.get("estimand_type", row.get("preferred_estimand", row.get("effect_scope", "")))).lower()
    haystack = " ".join([strategy, estimand])
    if any(tok in haystack for tok in BACKDOOR_STRATEGY_TOKENS):
        return True, strategy or "backdoor"
    # Many early contracts use identified_estimable without a precise strategy.
    # Treat it as backdoor-compatible when the contract proves a valid adjustment
    # status, including the valid-empty set.
    if _adjustment_status_is_valid(row):
        return True, strategy or "backdoor_adjustment"
    return False, strategy or "unsupported_estimand"


def _binary_support(a: np.ndarray) -> Tuple[bool, int, int, str]:
    finite = a[np.isfinite(a)]
    if len(finite) == 0:
        return False, 0, 0, "no_treatment_values"
    uniq = np.unique(finite)
    binary_like = len(uniq) <= 3 and np.nanmin(finite) >= 0 and np.nanmax(finite) <= 1
    if binary_like:
        treated = int(np.sum(finite > 0.5))
        control = int(np.sum(finite <= 0.5))
        ok = treated >= 5 and control >= 5
        return ok, treated, control, "ok" if ok else "insufficient_treated_or_control"
    # Continuous treatments still get a slope estimate, but treated/control counts
    # are reported by median split for support diagnostics only.
    med = float(np.nanmedian(finite))
    treated = int(np.sum(finite > med))
    control = int(np.sum(finite <= med))
    ok = treated >= 5 and control >= 5 and float(np.nanstd(finite)) > 1e-9
    return ok, treated, control, "continuous_treatment_slope" if ok else "insufficient_continuous_support"


def estimate_backdoor_effect(df: pd.DataFrame, row: pd.Series, *, bootstrap_b: int = 200) -> Dict[str, object]:
    """Estimate one contract row via ridge regression adjustment.

    The returned effect is the coefficient on the treatment in:
        Y ~ A + adjustment_set
    where adjustment variables are standardized and treatment keeps its natural
    scale.  For binary treatment this is a risk/mean difference; for continuous
    treatment it is a per-unit slope.
    """
    insight_id = _as_str(row.get("insight_id", ""))
    treatment = _as_str(row.get("treatment_col", row.get("source", "")))
    outcome = _as_str(row.get("outcome_col", row.get("target_col", row.get("target", ""))))
    effect_id = _as_str(row.get("effect_id", "")) or f"pearl_backdoor::{insight_id or treatment + '->' + outcome}"

    base = {
        "effect_id": effect_id,
        "insight_id": insight_id,
        "source": treatment,
        "target": outcome,
        "treatment_col": treatment,
        "outcome_col": outcome,
        "estimator_used": "backdoor_ridge_adjustment",
        "estimand_type": _as_str(row.get("estimand_type", "backdoor_adjusted_effect")) or "backdoor_adjusted_effect",
        "identification_strategy": _as_str(row.get("identification_strategy", row.get("graph_identification_strategy", "backdoor"))) or "backdoor",
        "authority_level": _as_str(row.get("authority_level", row.get("graph_authority_level", ""))),
        "source_authority": _as_str(row.get("source_authority", row.get("contract_source_authority", ""))),
        "adjustment_set": "",
        "forbidden_controls": "",
        "estimand_formula": "",
        "effect_estimate": np.nan,
        "ci_low": np.nan,
        "ci_high": np.nan,
        "support_n": 0,
        "treated_n": 0,
        "control_n": 0,
        "overlap_status": "not_evaluated",
        "balance_status": "not_evaluated",
        "causal_claim_status": "diagnostic_only_not_pearl_authorized",
        "pearl_authority_reason": "not_evaluated",
        "reason_codes": "",
    }

    authorized, auth_reason = _contract_row_authorized(row)
    base["pearl_authority_reason"] = auth_reason
    if not authorized:
        base["reason_codes"] = auth_reason.upper()
        return base

    backdoor_ok, strategy = _is_backdoor_estimand(row)
    base["identification_strategy"] = strategy
    if not backdoor_ok:
        base["causal_claim_status"] = "identified_but_estimator_not_enabled"
        base["estimator_used"] = "none_for_non_backdoor_estimand"
        base["reason_codes"] = "NON_BACKDOOR_ESTIMAND"
        return base

    if not treatment or treatment not in df.columns:
        base["reason_codes"] = "MISSING_TREATMENT_COLUMN"
        base["causal_claim_status"] = "identified_but_unestimated"
        return base
    if not outcome or outcome not in df.columns:
        base["reason_codes"] = "MISSING_OUTCOME_COLUMN"
        base["causal_claim_status"] = "identified_but_unestimated"
        return base

    # Prefer formal identification first; then bridge-normalized Discovery covariates;
    # keep legacy aliases only as fallbacks.
    adjustment = parse_list(
        row.get(
            "adjustment_set",
            row.get("candidate_covariates", row.get("suggested_adjustment_set", row.get("candidate_adjustment_set", ""))),
        )
    )
    forbidden = parse_list(row.get("forbidden_adjustment_set", row.get("forbidden_controls", row.get("forbidden_adjustments", ""))))
    adjustment = [c for c in adjustment if c not in {treatment, outcome}]
    forbidden_hit = [c for c in adjustment if c in set(forbidden)]
    base["adjustment_set"] = "|".join(adjustment)
    base["forbidden_controls"] = "|".join(forbidden)
    if forbidden_hit:
        base["causal_claim_status"] = "failed_forbidden_controls"
        base["reason_codes"] = "FORBIDDEN_CONTROL_IN_ADJUSTMENT_SET"
        return base

    y = pd.to_numeric(df[outcome], errors="coerce").astype(float).to_numpy(dtype=float)
    a = pd.to_numeric(df[treatment], errors="coerce").astype(float).to_numpy(dtype=float)
    Z, used_adjustment = SC.numeric_matrix(df, adjustment, exclude={treatment, outcome}, standardize=True, min_non_null=6)
    base["adjustment_set"] = "|".join(used_adjustment)
    base["estimand_formula"] = "E[Y|do(A=a)] via backdoor adjustment over {%s}" % ",".join(used_adjustment)

    mask = np.isfinite(y) & np.isfinite(a)
    if Z.shape[1] > 0:
        mask = mask & np.all(np.isfinite(Z), axis=1)
    n = int(np.sum(mask))
    base["support_n"] = n
    support_ok, treated_n, control_n, support_reason = _binary_support(a[mask])
    base["treated_n"] = treated_n
    base["control_n"] = control_n
    base["overlap_status"] = "ok" if support_ok else support_reason
    if n < max(12, 4 + Z.shape[1]):
        base["causal_claim_status"] = "identified_but_weak_support"
        base["reason_codes"] = "INSUFFICIENT_SUPPORT"
        return base
    if not support_ok:
        base["causal_claim_status"] = "identified_but_weak_overlap"
        base["reason_codes"] = support_reason.upper()
        return base

    y_m = y[mask]
    a_m = a[mask]
    Z_m = Z[mask] if Z.shape[1] > 0 else np.empty((n, 0), dtype=float)
    res = SC.linear_treatment_effect(a_m, y_m, Z_m, ridge=1.0)
    effect = float(res.effect) if np.isfinite(res.effect) else np.nan
    ci_low, ci_high = SC.bootstrap_treatment_effect_ci(
        a_m,
        y_m,
        Z_m,
        b=bootstrap_b,
        seed=991,
        ridge=1.0,
        alpha=0.10,
    )

    if not np.isfinite(effect):
        base["causal_claim_status"] = "identified_but_unestimated"
        base["reason_codes"] = "NUMERICAL_FAILURE"
        return base

    # Conservative balance summary on used covariates.  This is diagnostic only;
    # matching / propensity checks remain in Level 3.2 trial evaluation.
    if Z_m.shape[1] == 0:
        balance_status = "no_adjustment_covariates"
    else:
        hi = a_m > (0.5 if np.nanmax(a_m) <= 1.0 and np.nanmin(a_m) >= 0.0 else np.nanmedian(a_m))
        lo = ~hi
        if int(np.sum(hi)) >= 3 and int(np.sum(lo)) >= 3:
            smd = np.abs(np.nanmean(Z_m[hi], axis=0) - np.nanmean(Z_m[lo], axis=0))
            balance_status = "ok" if float(np.nanmean(smd)) <= 0.75 else "imbalanced_covariates"
        else:
            balance_status = "not_enough_groups"

    base.update({
        "effect_estimate": effect,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "balance_status": balance_status,
        "causal_claim_status": "identified_estimated_backdoor",
        "reason_codes": "",
    })
    return base


def estimate_backdoor_effects(df: pd.DataFrame, contract: pd.DataFrame, *, bootstrap_b: int = 200) -> pd.DataFrame:
    if df is None or contract is None or len(contract) == 0:
        return pd.DataFrame()
    rows: List[Dict[str, object]] = []
    for _, row in contract.iterrows():
        rows.append(estimate_backdoor_effect(df, row, bootstrap_b=bootstrap_b))
    return pd.DataFrame(rows)
