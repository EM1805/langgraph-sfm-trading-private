"""Estimation handoff reader and plan builder.

This module is the canonical Estimation-side entry point for the downstream
handoff produced by the causal contract layer.  It keeps Estimation conservative:
raw Discovery/PCMCI/SCM rows may become a plan row, but only contract-authorized
rows are marked as directly estimable.
"""
from __future__ import annotations

from runtime_env import configure_scientific_runtime
configure_scientific_runtime()


import os
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from . import common as C
from . import _utils as U
from . import sensitivity as SENS
from . import estimator_registry as ER

OUT_DIR = C.OUT_DIR
ESTIMATION_DIR = os.path.join(OUT_DIR, "estimation")
ESTIMATION_HANDOFF_CSV = os.path.join(ESTIMATION_DIR, "estimation_handoff.csv")
ESTIMATION_PLAN_CSV = os.path.join(ESTIMATION_DIR, "estimation_plan.csv")

HANDOFF_COLUMNS = [
    "insight_id", "source", "target", "treatment_col", "outcome_col", "lag",
    "authority_level", "authority_reason", "estimation_enabled", "allowed_for_estimation", "identification_status",
    "identified", "identification_strategy", "estimand_type", "estimand_expression",
    "adjustment_set_status", "adjustment_set", "total_adjustment_set",
    "candidate_adjustment_set", "backdoor_status", "blocked_by", "assumption_notes",
    "adjustment_set_source", "eligible_for_estimation",
    "forbidden_adjustment_set", "negative_controls", "conditioning_set_used",
    "conditioning_set_size", "mci_status", "mci_q_value", "mci_n_eff",
    "pc1_parent_support", "scm_role_hint", "source_authority", "source_artifacts",
]

PLAN_COLUMNS = [
    "plan_id", "insight_id", "source", "target", "treatment_col", "outcome_col", "lag",
    "estimand_type", "identification_status", "identified", "authority_level", "estimation_enabled",
    "allowed_for_estimation", "estimation_status", "recommended_estimator", "estimator_authority",
    "unobserved_confounding_risk", "sensitivity_level", "sensitivity_status",
    "recommended_sensitivity_method", "minimum_report_before_effect_claim",
    "conditioning_set_used", "adjustment_set", "adjustment_set_status",
    "candidate_adjustment_set", "backdoor_status", "blocked_by", "assumption_notes",
    "adjustment_set_source", "eligible_for_estimation",
    "forbidden_adjustment_set", "negative_controls", "mci_status", "mci_q_value",
    "mci_n_eff", "pc1_parent_support", "scm_role_hint", "reason",
]

_ESTIMABLE_LEVELS = {"identified_estimable"}
_REVIEW_LEVELS = {"identified_needs_estimation", "graph_review"}
_BLOCKED_LEVELS = {"raw_discovery_only", "discovery_only", "weak_or_unaligned", "blocked_id_algorithm"}


def _as_str(value) -> str:
    return U.as_str(value).strip()


def _safe_float(value, default=np.nan) -> float:
    return U.safe_float(value, default)


def _read_csv(path: str) -> pd.DataFrame:
    if not path or not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except (OSError, ValueError, TypeError, pd.errors.ParserError):
        return pd.DataFrame()
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()


def _existing_handoff_paths(out_dir: str = OUT_DIR) -> List[str]:
    return [
        os.path.join(out_dir, "estimation", "estimation_handoff.csv"),
        os.path.join(out_dir, "causal_contract.csv"),
        os.path.join(out_dir, "discovery_estimation_bridge.csv"),
    ]


def _first(row, keys: Iterable[str], default: str = "") -> str:
    for key in keys:
        try:
            value = _as_str(row.get(key, ""))
        except (AttributeError, TypeError, ValueError):
            value = ""
        if value and value.lower() not in {"nan", "none", "null"}:
            return value
    return default


def _boolish(value) -> bool:
    return _as_str(value).lower() in {"1", "true", "yes", "y", "on"}


def _normalize_handoff_frame(df: pd.DataFrame, source_name: str = "") -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=HANDOFF_COLUMNS)
    rows = []
    for idx, row in df.iterrows():
        source = _first(row, ["source", "treatment_col", "action_source", "action_name"])
        target = _first(row, ["target", "outcome_col", "target_col"])
        iid = _first(row, ["insight_id", "candidate_id", "edge_id", "path_id"])
        if not iid and (source or target):
            iid = f"{source}->{target}@{_first(row, ['lag', 'tau', 'time_lag'], '0')}"
        out = {c: "" for c in HANDOFF_COLUMNS}
        out.update({
            "insight_id": iid or f"handoff_{idx:05d}",
            "source": source,
            "target": target,
            "treatment_col": _first(row, ["treatment_col", "source", "action_source", "action_name"], source),
            "outcome_col": _first(row, ["outcome_col", "target_col", "target"], target),
            "lag": _first(row, ["lag", "tau", "time_lag"], ""),
            "authority_level": _first(row, ["authority_level", "graph_authority_level"], "raw_discovery_only" if source_name == "discovery_estimation_bridge" else ""),
            "authority_reason": _first(row, ["authority_reason", "pearl_authority_reason", "bridge_reason_codes"], ""),
            "estimation_enabled": _first(row, ["estimation_enabled", "estimator_enabled"], ""),
            "allowed_for_estimation": _first(row, ["allowed_for_estimation", "allowed", "can_estimate"], ""),
            "identification_status": _first(row, ["identification_status", "graph_identification_status", "status"], ""),
            "identified": _first(row, ["identified", "graph_identified", "is_identified"], ""),
            "identification_strategy": _first(row, ["identification_strategy", "graph_identification_strategy", "strategy"], ""),
            "estimand_type": _first(row, ["estimand_type", "preferred_estimand", "effect_scope"], ""),
            "estimand_expression": _first(row, ["estimand_expression", "estimand_formula"], ""),
            "adjustment_set_status": _first(row, ["adjustment_set_status"], ""),
            "adjustment_set": _first(row, ["adjustment_set", "total_adjustment_set", "candidate_adjustment_set", "suggested_adjustment_set", "candidate_covariates"], ""),
            "total_adjustment_set": _first(row, ["total_adjustment_set", "adjustment_set", "candidate_adjustment_set", "suggested_adjustment_set", "candidate_covariates"], ""),
            "candidate_adjustment_set": _first(row, ["candidate_adjustment_set", "adjustment_set", "total_adjustment_set"], ""),
            "backdoor_status": _first(row, ["backdoor_status"], ""),
            "blocked_by": _first(row, ["blocked_by"], ""),
            "assumption_notes": _first(row, ["assumption_notes"], ""),
            "adjustment_set_source": _first(row, ["adjustment_set_source"], ""),
            "eligible_for_estimation": _first(row, ["eligible_for_estimation"], ""),
            "forbidden_adjustment_set": _first(row, ["forbidden_adjustment_set", "post_treatment_columns", "forbidden_variables"], ""),
            "negative_controls": _first(row, ["negative_controls", "suggested_negative_control", "negative_control_col"], ""),
            "conditioning_set_used": _first(row, ["conditioning_set_used", "mci_conditioning_set_used", "candidate_covariates", "suggested_adjustment_set"], ""),
            "conditioning_set_size": _first(row, ["conditioning_set_size", "mci_conditioning_set_size"], ""),
            "mci_status": _first(row, ["mci_status"], ""),
            "mci_q_value": _first(row, ["mci_q_value"], ""),
            "mci_n_eff": _first(row, ["mci_n_eff"], ""),
            "pc1_parent_support": _first(row, ["pc1_parent_support", "pc1_parent_support_status", "pc1_is_selected_parent"], ""),
            "scm_role_hint": _first(row, ["scm_role_hint"], ""),
            "source_authority": _first(row, ["source_authority", "contract_source_authority"], source_name),
            "source_artifacts": _first(row, ["source_artifacts"], source_name),
        })
        rows.append(out)
    out_df = pd.DataFrame(rows, columns=HANDOFF_COLUMNS)
    return out_df.drop_duplicates(subset=["insight_id"], keep="first").reset_index(drop=True)


def load_estimation_handoff(out_dir: str = OUT_DIR) -> pd.DataFrame:
    """Load Estimation's preferred handoff.

    Priority:
    1. out/estimation/estimation_handoff.csv, curated by causal_contract.py
    2. out/causal_contract.csv, full contract fallback
    3. out/discovery_estimation_bridge.csv, legacy diagnostic fallback
    """
    for path in _existing_handoff_paths(out_dir):
        df = _read_csv(path)
        if df is None or len(df) == 0:
            continue
        name = os.path.splitext(os.path.basename(path))[0]
        return _normalize_handoff_frame(df, source_name=name)
    return pd.DataFrame(columns=HANDOFF_COLUMNS)


def prefer_handoff_contract(causal_contract: Optional[pd.DataFrame], handoff: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Return the contract-like frame Estimation should use for estimators.

    A non-empty estimation_handoff.csv is intentionally narrower than the full
    contract, so it gets priority.  If it is absent, preserve the full causal
    contract to keep older runs compatible.
    """
    if handoff is not None and len(handoff) > 0:
        return handoff.copy()
    if causal_contract is not None and len(causal_contract) > 0:
        return causal_contract.copy()
    return pd.DataFrame(columns=HANDOFF_COLUMNS)

def _recommended_estimator(row: pd.Series) -> tuple[str, str]:
    """Delegate estimator selection to the centralized registry.

    The function name is kept for backwards compatibility with existing tests and
    callers, but estimator names and authority labels now live in
    estimation_parts.estimator_registry.
    """
    return ER.select_estimator_for_row(row.to_dict() if hasattr(row, "to_dict") else row)


def _estimation_status(row: pd.Series) -> tuple[str, str]:
    authority = _as_str(row.get("authority_level", "")).lower()
    enabled = _boolish(row.get("estimation_enabled", ""))
    allowed_text = _as_str(row.get("allowed_for_estimation", ""))
    allowed = _boolish(allowed_text) if allowed_text else (authority == "identified_estimable" and enabled)
    identified_text = _as_str(row.get("identified", ""))
    identified = _boolish(identified_text) if identified_text else authority == "identified_estimable"
    status = _as_str(row.get("identification_status", "")).lower()
    role = _as_str(row.get("scm_role_hint", "")).lower()
    mci_status = _as_str(row.get("mci_status", "")).lower()

    if authority in _ESTIMABLE_LEVELS and enabled and allowed and identified:
        return "can_estimate_now", "contract_enabled_identified_estimable"
    if status in {"not_identified", "unidentified", "blocked_id_algorithm", "blocked", "simulable_not_identified"} or not identified:
        return "blocked", "causal_query_not_identified_by_scm_id"
    if authority == "identified_needs_estimation" or (identified and not enabled):
        return "needs_estimator_or_data", "identified_but_estimator_not_enabled_plan_only"
    if authority == "graph_review" or (status and status not in {"not_identified", "unidentified", ""}):
        return "needs_graph_review", "graph_or_identification_review_required"
    if authority in _BLOCKED_LEVELS:
        return "blocked", "contract_not_formally_identified"
    if "hard_blocked" in role:
        return "blocked", "hard_blocked_by_upstream_scm_role"
    if "temporal_parent" in role or mci_status in {"diagnostic_support", "pass"}:
        return "diagnostic_only", "structural_prior_not_estimation_authority"
    return "blocked", "no_estimation_authority"


def build_estimation_plan(handoff: Optional[pd.DataFrame]) -> pd.DataFrame:
    if handoff is None or len(handoff) == 0:
        return pd.DataFrame(columns=PLAN_COLUMNS)
    rows = []
    for idx, row in handoff.iterrows():
        status, reason = _estimation_status(row)
        estimator, estimator_authority = _recommended_estimator(row)
        sens_risk, sens_level, sens_status, sens_method, sens_reason = SENS.classify_sensitivity(row)
        source = _as_str(row.get("source", row.get("treatment_col", "")))
        target = _as_str(row.get("target", row.get("outcome_col", "")))
        iid = _as_str(row.get("insight_id", "")) or f"{source}->{target}@{_as_str(row.get('lag', '0'))}"
        rows.append({
            "plan_id": f"estimation_plan::{iid}",
            "insight_id": iid,
            "source": source,
            "target": target,
            "treatment_col": _as_str(row.get("treatment_col", source)),
            "outcome_col": _as_str(row.get("outcome_col", target)),
            "lag": _as_str(row.get("lag", "")),
            "estimand_type": _as_str(row.get("estimand_type", "")),
            "identification_status": _as_str(row.get("identification_status", "")),
            "identified": _as_str(row.get("identified", "")),
            "authority_level": _as_str(row.get("authority_level", "")),
            "estimation_enabled": _as_str(row.get("estimation_enabled", "")),
            "allowed_for_estimation": _as_str(row.get("allowed_for_estimation", "")),
            "estimation_status": status,
            "recommended_estimator": estimator,
            "estimator_authority": estimator_authority,
            "unobserved_confounding_risk": sens_risk,
            "sensitivity_level": sens_level,
            "sensitivity_status": sens_status,
            "recommended_sensitivity_method": sens_method,
            "minimum_report_before_effect_claim": "required" if sens_status.startswith("required") else "recommended" if sens_status.startswith("recommended") else "not_applicable",
            "conditioning_set_used": _as_str(row.get("conditioning_set_used", "")),
            "adjustment_set": _as_str(row.get("adjustment_set", row.get("total_adjustment_set", ""))),
            "adjustment_set_status": _as_str(row.get("adjustment_set_status", "")),
            "candidate_adjustment_set": _as_str(row.get("candidate_adjustment_set", "")),
            "backdoor_status": _as_str(row.get("backdoor_status", "")),
            "blocked_by": _as_str(row.get("blocked_by", "")),
            "assumption_notes": _as_str(row.get("assumption_notes", "")),
            "adjustment_set_source": _as_str(row.get("adjustment_set_source", "")),
            "eligible_for_estimation": _as_str(row.get("eligible_for_estimation", "")),
            "forbidden_adjustment_set": _as_str(row.get("forbidden_adjustment_set", "")),
            "negative_controls": _as_str(row.get("negative_controls", "")),
            "mci_status": _as_str(row.get("mci_status", "")),
            "mci_q_value": _as_str(row.get("mci_q_value", "")),
            "mci_n_eff": _as_str(row.get("mci_n_eff", "")),
            "pc1_parent_support": _as_str(row.get("pc1_parent_support", "")),
            "scm_role_hint": _as_str(row.get("scm_role_hint", "")),
            "reason": reason + ("|" + sens_reason if sens_reason else ""),
        })
    return pd.DataFrame(rows, columns=PLAN_COLUMNS)


def write_estimation_plan(handoff: Optional[pd.DataFrame], out_dir: str = OUT_DIR) -> str:
    path = os.path.join(out_dir, "estimation", "estimation_plan.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plan = build_estimation_plan(handoff)
    plan.to_csv(path, index=False)
    return path
