"""Output writers for Estimation Level 3.2.

This module keeps CSV-writing and authority effect-frame assembly
out of engine.py so the engine can remain an orchestrator.
"""
from __future__ import annotations

from runtime_env import configure_scientific_runtime
configure_scientific_runtime()


import os
from typing import Optional

import numpy as np
import pandas as pd

from . import common as C
from . import contract_gate as CG
from . import pearl_backdoor as PB
from . import handoff_reader as HR
from . import sensitivity as SEN
from . import effect_estimates as EE
from . import _utils as U

BOOT_B = C.BOOT_B
OUT_DIR = C.OUT_DIR
OUT_L3 = C.OUT_L3
OUT_LEDGER = C.OUT_LEDGER
OUT_TRIALS = C.OUT_TRIALS
EFFECT_ESTIMATES_CSV = os.path.join(OUT_DIR, "estimation", "effect_estimates.csv")
ESTIMATION_AUTHORITY_REPORT_CSV = os.path.join(OUT_DIR, "estimation", "estimation_authority_report.csv")
ESTIMATION_PLAN_CSV = os.path.join(OUT_DIR, "estimation", "estimation_plan.csv")

_mode_or_empty = U.mode_or_empty
_as_str = U.as_str

def _authority_effects_frame(df_l3: pd.DataFrame, df_data: Optional[pd.DataFrame] = None, df_contract: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    columns = [
        "effect_id", "insight_id", "source", "target", "treatment_col", "outcome_col", "action_name",
        "estimand_type", "identification_strategy", "estimator_used", "estimand_formula",
        "authority_level", "source_authority", "adjustment_set", "forbidden_controls",
        "causal_claim_status", "effect_estimate", "ci_low", "ci_high", "confidence",
        "support_n", "eligible_trials", "treated_n", "control_n", "pearl_authorized_rate",
        "overlap_status", "balance_status", "sensitivity_level", "diagnostic_grade", "reason_codes",
        "pearl_authority_reason",
    ]
    if df_l3 is None or len(df_l3) == 0:
        base = pd.DataFrame(columns=columns)
    else:
        base = pd.DataFrame()
        base["insight_id"] = df_l3.get("insight_id", pd.Series([], dtype=str)).astype(str)
        base["effect_id"] = base["insight_id"].map(lambda x: f"pearl_effect::{x}")
        base["source"] = df_l3.get("source", "")
        target = df_l3.get("target_col", df_l3.get("outcome_col", df_l3.get("target", "")))
        base["target"] = target
        base["treatment_col"] = df_l3.get("source", "")
        base["outcome_col"] = target
        base["action_name"] = df_l3.get("action_name", "")
        base["estimand_type"] = df_l3.get("estimand", "ATT_proxy")
        base["identification_strategy"] = df_l3.get("identification_strategy", "")
        base["estimator_used"] = "level32_trial_matched_counterfactual"
        base["estimand_formula"] = ""
        base["authority_level"] = df_l3.get("graph_authority_level", "")
        base["source_authority"] = df_l3.get("contract_source_authority", "")
        base["adjustment_set"] = df_l3.get("discovery_adjustment_set", "")
        base["forbidden_controls"] = df_l3.get("discovery_forbidden_adjustments", "")
        base["causal_claim_status"] = df_l3.get("causal_claim_status", "diagnostic_only_not_pearl_authorized")
        base["effect_estimate"] = pd.to_numeric(df_l3.get("effect_mean", np.nan), errors="coerce")
        base["ci_low"] = pd.to_numeric(df_l3.get("effect_ci_low", np.nan), errors="coerce")
        base["ci_high"] = pd.to_numeric(df_l3.get("effect_ci_high", np.nan), errors="coerce")
        base["confidence"] = pd.to_numeric(df_l3.get("confidence", np.nan), errors="coerce")
        base["support_n"] = pd.to_numeric(df_l3.get("n_trials_total", np.nan), errors="coerce")
        base["eligible_trials"] = pd.to_numeric(df_l3.get("n_trials_eligible", np.nan), errors="coerce")
        base["treated_n"] = ""
        base["control_n"] = ""
        base["pearl_authorized_rate"] = pd.to_numeric(df_l3.get("pearl_authorized_rate", np.nan), errors="coerce")
        base["overlap_status"] = df_l3.get("overlap_pass_rate", "")
        base["balance_status"] = df_l3.get("balance_pass_rate", "")
        base["sensitivity_level"] = df_l3.get("sensitivity_level", "")
        base["diagnostic_grade"] = df_l3.get("diagnostic_grade", "")
        base["reason_codes"] = df_l3.get("reason_codes", "")
        base["pearl_authority_reason"] = df_l3.get("pearl_authority_reason", "")

    try:
        bd = PB.estimate_backdoor_effects(df_data, df_contract, bootstrap_b=BOOT_B) if (df_data is not None and df_contract is not None and len(df_contract) > 0) else pd.DataFrame()
    except (OSError, ValueError, TypeError, RuntimeError, np.linalg.LinAlgError) as exc:
        bd = pd.DataFrame([{"effect_id": "pearl_backdoor_error", "insight_id": "", "causal_claim_status": "backdoor_estimator_error", "estimator_used": "backdoor_ridge_adjustment", "reason_codes": "BACKDOOR_ESTIMATOR_ERROR:%s" % str(exc)[:120]}])

    if bd is not None and len(bd) > 0:
        rows = []
        base_by_id = {}
        if len(base) > 0 and "insight_id" in base.columns:
            for _, br in base.iterrows():
                base_by_id[_as_str(br.get("insight_id", ""))] = br.to_dict()
        for _, rr in bd.iterrows():
            iid = _as_str(rr.get("insight_id", ""))
            if not iid:
                continue
            cur = dict(base_by_id.get(iid, {"insight_id": iid}))
            for c, v in rr.items():
                cur[c] = v
            if not _as_str(cur.get("source", "")):
                cur["source"] = _as_str(cur.get("treatment_col", ""))
            if not _as_str(cur.get("target", "")):
                cur["target"] = _as_str(cur.get("outcome_col", ""))
            base_by_id[iid] = cur
        seen = set()
        if len(base) > 0 and "insight_id" in base.columns:
            for _, br in base.iterrows():
                iid = _as_str(br.get("insight_id", ""))
                if iid in base_by_id and iid not in seen:
                    rows.append(base_by_id[iid]); seen.add(iid)
        for iid, cur in base_by_id.items():
            if iid and iid not in seen:
                rows.append(cur); seen.add(iid)
        base = pd.DataFrame(rows) if rows else base

    for c in columns:
        if c not in base.columns:
            base[c] = ""
    return base[columns].copy()

def _write_estimation_authority_report(insights: pd.DataFrame, df_contract: Optional[pd.DataFrame] = None, contract_required: bool = False) -> str:
    os.makedirs(os.path.dirname(ESTIMATION_AUTHORITY_REPORT_CSV), exist_ok=True)
    report = CG.build_authority_report(insights, df_contract, contract_required=contract_required)
    report.to_csv(ESTIMATION_AUTHORITY_REPORT_CSV, index=False)
    return ESTIMATION_AUTHORITY_REPORT_CSV

def _write_authority_effects(df_l3: pd.DataFrame, df_data: Optional[pd.DataFrame] = None, df_contract: Optional[pd.DataFrame] = None) -> str:
    os.makedirs(os.path.dirname(EFFECT_ESTIMATES_CSV), exist_ok=True)
    out = _authority_effects_frame(df_l3, df_data=df_data, df_contract=df_contract)
    out.to_csv(EFFECT_ESTIMATES_CSV, index=False)
    return EFFECT_ESTIMATES_CSV

def _write_outputs(df_l3: pd.DataFrame, df_trials: pd.DataFrame, df_ledger: pd.DataFrame, df_data: Optional[pd.DataFrame] = None, df_contract: Optional[pd.DataFrame] = None, insights: Optional[pd.DataFrame] = None, contract_required: bool = False, estimation_handoff: Optional[pd.DataFrame] = None) -> str:
    df_trials.to_csv(OUT_TRIALS, index=False)
    df_ledger.to_csv(OUT_LEDGER, index=False)
    df_l3.to_csv(OUT_L3, index=False)
    plan_input = estimation_handoff if estimation_handoff is not None and len(estimation_handoff) > 0 else df_contract
    plan = HR.build_estimation_plan(plan_input)
    HR.write_estimation_plan(plan_input, OUT_DIR)
    SEN.write_sensitivity_analysis(plan_input, OUT_DIR)
    EE.write_effect_estimates_from_frames(df_data, plan, OUT_DIR)
    try:
        from contracts.causal_report import write_causal_report
        write_causal_report(out_dir=OUT_DIR)
    except (OSError, ValueError, TypeError, RuntimeError, ImportError):
        pass
    _write_estimation_authority_report(insights if insights is not None else df_l3, df_contract, contract_required=contract_required)
    return ""