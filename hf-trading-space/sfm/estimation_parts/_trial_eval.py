"""Trial-level evaluation helpers for estimation Level 3.2.

This module owns per-trial setup, matching, support diagnostics and row/ledger
serialization. The public names intentionally keep the previous private function
names so legacy imports/tests that reached into estimation_parts.engine can be
aliased without changing behavior.
"""


from runtime_env import configure_scientific_runtime
configure_scientific_runtime()

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from runtime_compat import assert_scientific_stack
assert_scientific_stack()
import pandas as pd

from .global_diagnostics import build_diagnostics as build_global_diagnostics
from scm_parts.identification_legacy import build_identification
from . import common as C
from . import propensity as P
from . import matching as M
from . import diagnostics as D
from . import effects as E
from . import discovery_bridge as B
from . import contract_gate as CG
from . import _utils as U

BOOT_B = C.BOOT_B
DAG_RISK_CONF_PENALTY = C.DAG_RISK_CONF_PENALTY
DAG_RISK_FORCE_WARNING = C.DAG_RISK_FORCE_WARNING
DATE_COL = C.DATE_COL
DEFAULT_DATA_CSV = C.DEFAULT_DATA_CSV
ENABLE_PRETREND_CHECK = C.ENABLE_PRETREND_CHECK
ENABLE_PROPENSITY = C.ENABLE_PROPENSITY
FALLBACK_DATA_CSV = C.FALLBACK_DATA_CSV
INSIGHTS_L2 = C.INSIGHTS_L2
DISCOVERY_BRIDGE_CSV = C.DISCOVERY_BRIDGE_CSV
K_CONTROLS = C.K_CONTROLS
NEGCTRL_ENABLE = C.NEGCTRL_ENABLE
NEGCTRL_MAX_SUCCESS_LB = C.NEGCTRL_MAX_SUCCESS_LB
NEGCTRL_OUTCOME_COL = C.NEGCTRL_OUTCOME_COL
OUT_DIR = C.OUT_DIR
OUT_L3 = C.OUT_L3
OUT_LEDGER = C.OUT_LEDGER
OUT_TRIALS = C.OUT_TRIALS
PRETREND_DAYS = C.PRETREND_DAYS
PROPENSITY_ACTION_COL = C.PROPENSITY_ACTION_COL
PROPENSITY_MAX_DIFF = C.PROPENSITY_MAX_DIFF
TARGET_COL = C.TARGET_COL
Z_SUCCESS = C.Z_SUCCESS

_estimate_effect_bundle = E.estimate_effect_bundle
_apply_dag_covariates = C._apply_dag_covariates
_as_str = C._as_str
_compute_propensity = P._compute_propensity
_covariate_balance = D._covariate_balance
_hybrid_match_controls = M._hybrid_match_controls
_overlap_check = D._overlap_check
_pretrend_check = P._pretrend_check
_rosenbaum_sensitivity = D._rosenbaum_sensitivity
_safe_float = C._safe_float
_select_negative_control_col = C._select_negative_control_col
_sensitivity_metrics = D._sensitivity_metrics
_signed_z = P._signed_z

@dataclass
class TrialContext:
    raw_trial: Any
    trial_id: Dict[str, object]
    insight_meta: Dict[str, object]
    l29_meta: Dict[str, object]
    expected_direction: str
    setup: Dict[str, object]
    controls: List[int]
    match_meta: Dict[str, object]
    effect_meta: Dict[str, object]
    signed_z: float
    direction_ok_flag: float
    success_flag: float
    smd_mean: float
    balance_pass: int
    propensity_meta: Dict[str, object]
    pretrend_meta: Dict[str, object]
    overlap_meta: Dict[str, object]
    z_negctrl: float
    negctrl_flag: float
    sensitivity_meta: Dict[str, object]
    rosenbaum_meta: Dict[str, object]
    diagnostic: Any
    identification: Any
    reason_codes: List[str]
    eligible_flag: int


def _trial_identity(tr: pd.Series) -> Dict[str, object]:
    iid = _as_str(tr.get("insight_id", "")).strip()
    t_raw = _safe_float(tr.get("t_index", np.nan), np.nan)
    t_idx = int(t_raw) if np.isfinite(t_raw) else -1
    return {
        "insight_id": iid,
        "trial_id": _as_str(tr.get("trial_id", tr.get("insight_id", ""))),
        "t_index": t_idx,
        "date": _as_str(tr.get("date", "")),
        "action_name": _as_str(tr.get("action_name", "")),
    }


def _trial_is_usable(trial_id: Dict[str, object], df: pd.DataFrame) -> bool:
    iid = _as_str(trial_id.get("insight_id", "")).strip()
    t_idx = int(trial_id.get("t_index", -1))
    return bool(iid) and 0 <= t_idx < len(df)


def _expected_direction(tr: pd.Series, insight_meta: Dict[str, object]) -> str:
    direction = _as_str(tr.get("expected_direction_on_target", "")) or _as_str(insight_meta.get("expected_direction_on_target", ""))
    if direction:
        return direction
    beta_k = _safe_float(insight_meta.get("beta_k", np.nan), np.nan)
    return "increase" if np.isfinite(beta_k) and beta_k > 0 else "decrease"


def _pearl_contract_authority(insight_meta: Dict[str, object]) -> Dict[str, object]:
    """Return Pearl-estimation authority derived from causal_contract.csv.

    Compatibility wrapper: the canonical contract-gate rules live in
    estimation_parts.contract_gate so they can be tested/imported without
    loading the full Level 3.2 engine.
    """
    return CG.pearl_contract_authority(insight_meta)


def _trial_setup(tr, insight_meta, dag, df, base_covs):
    window_days = int(_safe_float(tr.get("window_days", insight_meta.get("lag", 1)), 1))
    action_source = _as_str(insight_meta.get("treatment_col", insight_meta.get("source", tr.get("source", ""))))
    outcome_col = _as_str(insight_meta.get("outcome_col", insight_meta.get("target_col", tr.get("target", TARGET_COL)))) or TARGET_COL
    if outcome_col not in df.columns:
        outcome_col = TARGET_COL if TARGET_COL in df.columns else outcome_col
    # Prefer Discovery bridge canonical fields. Legacy aliases remain fallbacks for
    # older artifacts, but Estimation no longer reinterprets raw PCMCI hints first.
    discovery_adjustment_set = (
        _as_str(insight_meta.get("candidate_covariates", ""))
        or _as_str(insight_meta.get("suggested_adjustment_set", ""))
        or _as_str(insight_meta.get("candidate_adjustment_set", ""))
    )
    discovery_forbidden_adjustments = (
        _as_str(insight_meta.get("post_treatment_columns", ""))
        or _as_str(insight_meta.get("forbidden_adjustment_set", ""))
        or _as_str(insight_meta.get("forbidden_variables", ""))
    )
    discovery_negative_control = (
        _as_str(insight_meta.get("negative_control_col", ""))
        or _as_str(insight_meta.get("suggested_negative_control", ""))
        or _as_str(insight_meta.get("negative_controls", ""))
        or _as_str(insight_meta.get("negative_control_hint", ""))
    )
    (
        covs_trial,
        dag_adjustment_set,
        dag_forbidden_adjustments,
        dag_covariate_violation_flag,
        dag_adjustment_source,
        dag_adjustment_confidence_effective,
        dag_adjustment_notes,
    ) = _apply_dag_covariates(dag, action_source, outcome_col, base_covs, insight_meta)
    dag_ann = dag.l32_annotation(action_source, outcome_col) if dag is not None and action_source else {}
    dag_risk_paths = _as_str(dag_ann.get("dag_risk_paths", ""))
    # The bridge-selected negative control is the canonical PCMCI handoff, but it
    # must be an actual data column. Never pass a pipe-separated or missing symbol
    # into the effect evaluator as if it were a column name.
    bridge_negative_control = B.select_first_valid_column(discovery_negative_control, df)
    dag_negative_control = bridge_negative_control or _select_negative_control_col(dag, action_source, df) or ""
    pearl_authority = _pearl_contract_authority(insight_meta)
    return {
        "window_days": window_days,
        "action_source": action_source,
        "outcome_col": outcome_col,
        "preferred_estimand": _as_str(insight_meta.get("preferred_estimand", "")),
        "pearl_authority": pearl_authority,
        "contract_required": int(pearl_authority.get("contract_required", 0)),
        "contract_row_present": int(pearl_authority.get("contract_row_present", 0)),
        "contract_source_authority": _as_str(pearl_authority.get("source_authority", "")),
        "graph_authority_level": _as_str(pearl_authority.get("authority_level", "")),
        "pearl_estimation_authorized": int(pearl_authority.get("pearl_estimation_authorized", 0)),
        "pearl_authority_reason": _as_str(pearl_authority.get("pearl_authority_reason", "")),
        "causal_claim_status": _as_str(pearl_authority.get("causal_claim_status", "")),
        "graph_identification_strategy": _as_str(insight_meta.get("graph_identification_strategy", insight_meta.get("identification_strategy", ""))),
        "graph_identified": int(_safe_float(insight_meta.get("graph_identified", insight_meta.get("identified", 0)), 0)),
        "graph_estimand_expression": _as_str(insight_meta.get("graph_estimand_expression", insight_meta.get("estimand_expression", ""))),
        "graph_assumptions": _as_str(insight_meta.get("graph_assumptions", insight_meta.get("assumptions", ""))),
        "graph_failed_assumptions": _as_str(insight_meta.get("graph_failed_assumptions", insight_meta.get("failed_assumptions", ""))),
        "treatment_role": _as_str(insight_meta.get("treatment_role", insight_meta.get("source_role", ""))),
        "treatment_kind": _as_str(insight_meta.get("treatment_kind", "")),
        "covs_trial": covs_trial,
        "dag_adjustment_set": dag_adjustment_set,
        "dag_forbidden_adjustments": dag_forbidden_adjustments,
        "dag_covariate_violation_flag": dag_covariate_violation_flag,
        "dag_adjustment_source": dag_adjustment_source,
        "dag_adjustment_confidence_effective": dag_adjustment_confidence_effective,
        "dag_adjustment_notes": dag_adjustment_notes,
        "dag_ann": dag_ann,
        "dag_risk_paths": dag_risk_paths,
        "dag_negative_control": dag_negative_control,
        "discovery_adjustment_set": discovery_adjustment_set,
        "discovery_forbidden_adjustments": discovery_forbidden_adjustments,
        "discovery_negative_control": discovery_negative_control,
    }


def _effect_bundle(df, t_idx, controls, window_days, covs_trial, outcome_col):
    base = {
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
        "valid_control_ids": list(controls or []),
    }
    if not controls:
        return base
    emeta = _estimate_effect_bundle(df, t_idx, controls, window_days, outcome_col, covs_trial) or {}
    base.update(emeta)
    return base


def _propensity_metrics(df, t_idx, controls):
    out = {
        "propensity_t": np.nan,
        "propensity_c_mean": np.nan,
        "propensity_pass": 1,
    }
    if not controls or "__propensity__" not in df.columns:
        return out
    p = pd.to_numeric(df["__propensity__"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(p[t_idx]):
        return out
    out["propensity_t"] = float(p[t_idx])
    pcs = p[np.asarray(controls, dtype=int)]
    pcs = pcs[np.isfinite(pcs)]
    if len(pcs) > 0:
        out["propensity_c_mean"] = float(np.mean(pcs))
        out["propensity_pass"] = int(abs(out["propensity_t"] - out["propensity_c_mean"]) <= PROPENSITY_MAX_DIFF)
    return out


def _pretrend_metrics(df, t_idx, controls, outcome_col=None):
    out = {
        "pretrend_pass": 1,
        "pretrend_trial": np.nan,
        "pretrend_ctrl": np.nan,
        "pretrend_diff": np.nan,
        "pretrend_reason": "",
    }
    if ENABLE_PRETREND_CHECK and controls:
        vals = _pretrend_check(df, t_idx, controls, PRETREND_DAYS, outcome_col=outcome_col)
        out.update({
            "pretrend_pass": vals[0],
            "pretrend_trial": vals[1],
            "pretrend_ctrl": vals[2],
            "pretrend_diff": vals[3],
            "pretrend_reason": vals[4],
        })
    return out


def _overlap_metrics(df, t_idx, controls):
    out = {
        "overlap_pass": 1,
        "overlap_gap": 0.0,
        "overlap_treated": np.nan,
        "overlap_min_ctrl": np.nan,
        "overlap_max_ctrl": np.nan,
    }
    if controls and "__propensity__" in df.columns:
        vals = _overlap_check(df, t_idx, controls)
        out.update({
            "overlap_pass": vals[0],
            "overlap_gap": vals[1],
            "overlap_treated": vals[2],
            "overlap_min_ctrl": vals[3],
            "overlap_max_ctrl": vals[4],
        })
    return out


def _diagnostic_and_identification(df, t_idx, controls, covs_trial, action_source, outcome_col, dag_forbidden_adjustments, effect, effect_se_proxy, pretrend_pass, dag_setup, l29_reason_codes, match_meta=None, effect_meta=None, overlap_meta=None):
    drift_flag = ("drift" in l29_reason_codes.lower()) if l29_reason_codes else None
    leakage_flag = ("leak" in l29_reason_codes.lower()) if l29_reason_codes else None

    diag = build_global_diagnostics(
        df,
        treated_index=t_idx,
        controls=controls,
        covariates=covs_trial,
        min_rows=max(30, len(covs_trial) * 3 if covs_trial else 30),
        leakage_flag=leakage_flag,
        drift_flag=drift_flag,
        treatment_col=action_source,
        outcome_col=outcome_col,
        post_treatment_columns=dag_forbidden_adjustments.split("|") if dag_forbidden_adjustments else [],
        effect=effect,
        se_proxy=effect_se_proxy,
    )
    dag_ann = dag_setup["dag_ann"]
    ident = build_identification(
        has_controls=bool(controls),
        propensity_available=("__propensity__" in df.columns),
        overlap_ok=diag.overlap_ok,
        balance_ok=diag.balance_ok,
        sample_size_ok=diag.sample_size_ok,
        leakage_ok=diag.leakage_ok,
        drift_ok=diag.drift_ok,
        temporal_order_ok=diag.temporal_order_ok,
        diagnostic_grade=diag.diagnostic_grade,
        sensitivity_level=diag.sensitivity_level,
        pretrend_available=bool(ENABLE_PRETREND_CHECK),
        pretrend_pass=(bool(pretrend_pass == 1) if controls else None),
        dag_adjustment_set=dag_setup["dag_adjustment_set"],
        dag_forbidden_adjustments=dag_setup["dag_forbidden_adjustments"],
        dag_risk_paths=dag_setup["dag_risk_paths"],
        dag_covariate_violation_flag=dag_setup["dag_covariate_violation_flag"],
        dag_action_known=dag_ann.get("dag_action_known", 0),
        dag_target_known=dag_ann.get("dag_target_known", 0),
        dag_action_type=_as_str(dag_ann.get("dag_action_type", "unknown")),
        dag_target_type=_as_str(dag_ann.get("dag_target_type", "unknown")),
        dag_action_time_role=_as_str(dag_ann.get("dag_action_time_role", "unknown")),
        dag_target_time_role=_as_str(dag_ann.get("dag_target_time_role", "unknown")),
        dag_adjustment_confidence=dag_setup["dag_adjustment_confidence_effective"] or _as_str(dag_ann.get("dag_adjustment_confidence", "unknown")),
        dag_direct_edge_confidence=_as_str(dag_ann.get("dag_direct_edge_confidence", "unknown")),
        dag_path_confidence=_as_str(dag_ann.get("dag_path_confidence", "unknown")),
        dag_mediators=_as_str(dag_ann.get("dag_mediators", "")).split("|") if _as_str(dag_ann.get("dag_mediators", "")) else [],
        dag_colliders=_as_str(dag_ann.get("dag_colliders", "")).split("|") if _as_str(dag_ann.get("dag_colliders", "")) else [],
        dag_negative_controls=_as_str(dag_ann.get("dag_negative_controls", "")).split("|") if _as_str(dag_ann.get("dag_negative_controls", "")) else [],
        dag_path_id=_as_str(dag_ann.get("dag_path_id", "")),
        dag_treatment_node=_as_str(dag_ann.get("dag_treatment_node", "")),
        dag_outcome_node=_as_str(dag_ann.get("dag_outcome_node", "")),
        match_quality=_safe_float((match_meta or {}).get("match_quality", np.nan), np.nan),
        shared_support_ratio=_safe_float((match_meta or {}).get("shared_support_ratio", np.nan), np.nan),
        overlap_gap_value=_safe_float((overlap_meta or {}).get("overlap_gap", np.nan), np.nan),
        placebo_pvalue=_safe_float((effect_meta or {}).get("placebo_pvalue", np.nan), np.nan),
        effect_sign_stable=_safe_float((effect_meta or {}).get("effect_sign_stable", np.nan), np.nan),
        dominant_control_weight=_safe_float((effect_meta or {}).get("match_weight_max", np.nan), np.nan),
        method_agreement=_safe_float((effect_meta or {}).get("method_agreement", np.nan), np.nan),
        validation_score=np.nan,
    )
    graph_strategy = _as_str(dag_setup.get("graph_identification_strategy", "")).strip()
    if graph_strategy:
        ident.strategy = graph_strategy
    try:
        graph_identified = int(_safe_float(dag_setup.get("graph_identified", 0), 0))
    except (TypeError, ValueError, OverflowError):
        graph_identified = 0
    if graph_identified in (0, 1):
        ident.identifiable = bool(graph_identified)
    graph_assumptions = [x for x in _as_str(dag_setup.get("graph_assumptions", "")).split("|") if x]
    graph_failed = [x for x in _as_str(dag_setup.get("graph_failed_assumptions", "")).split("|") if x]
    if graph_assumptions:
        ident.assumptions = sorted(set(list(ident.assumptions) + graph_assumptions))
    if graph_failed:
        ident.failed_assumptions = sorted(set(list(ident.failed_assumptions) + graph_failed))
    if graph_strategy or graph_assumptions or graph_failed:
        ident.notes = list(ident.notes) + ["graph_identifier_applied"]
    return diag, ident


def _build_reason_codes(controls, mmeta, balance_pass, propensity_pass, pretrend_pass, overlap_pass, negctrl_flag, dag_setup, sens, ros, signed_z):
    reason_codes = []
    if len(controls) == 0:
        reason_codes.append(_as_str(mmeta.get("reason", "NO_MATCHES")).upper())
    if balance_pass != 1:
        reason_codes.append("POOR_BALANCE")
    if propensity_pass != 1:
        reason_codes.append("PROPENSITY_GAP")
    if pretrend_pass != 1:
        reason_codes.append("PRETREND_MISMATCH")
    if overlap_pass != 1:
        reason_codes.append("OVERLAP_FAIL")
    if np.isfinite(negctrl_flag) and int(negctrl_flag) == 1:
        reason_codes.append("NEGCTRL_MOVE")
    if dag_setup["dag_negative_control"] and dag_setup["dag_negative_control"] != NEGCTRL_OUTCOME_COL:
        reason_codes.append("DAG_NEGCTRL")
    if dag_setup["dag_risk_paths"]:
        reason_codes.append("DAG_RISK_PATH")
    if sens.get("robustness_pass", 0) != 1:
        reason_codes.append("LOW_ROBUSTNESS")
    if ros.get("rosenbaum_pass", 0) != 1:
        reason_codes.append("ROSENBAUM_FRAGILE")
    if not np.isfinite(signed_z):
        reason_codes.append("NO_EFFECT_ESTIMATE")
    return sorted(set(reason_codes))


def _evaluate_matching(df: pd.DataFrame, t_idx: int, setup: Dict[str, object]) -> Tuple[List[int], Dict[str, object], Dict[str, object]]:
    controls, match_meta = _hybrid_match_controls(df, t_idx, setup["covs_trial"], outcome_col=setup.get("outcome_col", TARGET_COL))
    effect_meta = _effect_bundle(df, t_idx, controls, setup["window_days"], setup["covs_trial"], setup.get("outcome_col", TARGET_COL))
    return controls, match_meta, effect_meta


def _evaluate_support_metrics(df: pd.DataFrame, t_idx: int, controls: List[int], setup: Dict[str, object], effect_meta: Dict[str, object], expected_direction: str) -> Dict[str, object]:
    signed_z = _signed_z(effect_meta.get("z_raw", np.nan), expected_direction)
    direction_ok_flag = int(np.isfinite(signed_z) and float(signed_z) > 0.0) if controls else np.nan
    success_flag = int(np.isfinite(signed_z) and float(signed_z) >= float(Z_SUCCESS)) if controls else np.nan
    smd_mean, balance_pass = _covariate_balance(df, t_idx, controls, setup["covs_trial"]) if controls else (np.nan, 0)
    propensity_meta = _propensity_metrics(df, t_idx, controls)
    pretrend_meta = _pretrend_metrics(df, t_idx, controls, outcome_col=setup.get("outcome_col", TARGET_COL))
    overlap_meta = _overlap_metrics(df, t_idx, controls)

    z_negctrl = np.nan
    negctrl_flag = np.nan
    if NEGCTRL_ENABLE and controls and setup["dag_negative_control"]:
        nc_meta = _estimate_effect_bundle(df, t_idx, controls, setup["window_days"], setup["dag_negative_control"], setup["covs_trial"])
        z_negctrl = nc_meta.get("z_raw", np.nan)
        negctrl_flag = int(np.isfinite(z_negctrl) and abs(float(z_negctrl)) >= float(Z_SUCCESS))

    sensitivity_meta = _sensitivity_metrics(effect_meta.get("effect", np.nan), effect_meta.get("effect_se_proxy", np.nan))
    rosenbaum_meta = _rosenbaum_sensitivity(effect_meta.get("effect", np.nan), effect_meta.get("effect_se_proxy", np.nan), effect_meta.get("match_effective_n", np.nan))

    return {
        "signed_z": signed_z,
        "direction_ok_flag": direction_ok_flag,
        "success_flag": success_flag,
        "smd_mean": smd_mean,
        "balance_pass": balance_pass,
        "propensity_meta": propensity_meta,
        "pretrend_meta": pretrend_meta,
        "overlap_meta": overlap_meta,
        "z_negctrl": z_negctrl,
        "negctrl_flag": negctrl_flag,
        "sensitivity_meta": sensitivity_meta,
        "rosenbaum_meta": rosenbaum_meta,
    }


def _build_trial_context(tr, df, i_map, l29_map, dag, base_covs) -> Optional[TrialContext]:
    trial_id = _trial_identity(tr)
    if not _trial_is_usable(trial_id, df):
        return None

    iid = trial_id["insight_id"]
    insight_meta = i_map.get(iid, {})
    l29_meta = l29_map.get(iid, {})
    expected_direction = _expected_direction(tr, insight_meta)
    setup = _trial_setup(tr, insight_meta, dag, df, base_covs)
    controls, match_meta, effect_meta = _evaluate_matching(df, int(trial_id["t_index"]), setup)
    support = _evaluate_support_metrics(df, int(trial_id["t_index"]), controls, setup, effect_meta, expected_direction)

    diag, ident = _diagnostic_and_identification(
        df,
        int(trial_id["t_index"]),
        controls,
        setup["covs_trial"],
        setup["action_source"],
        setup.get("outcome_col", TARGET_COL),
        setup["dag_forbidden_adjustments"],
        effect_meta.get("effect", np.nan),
        effect_meta.get("effect_se_proxy", np.nan),
        support["pretrend_meta"]["pretrend_pass"],
        setup,
        str(l29_meta.get("reason_codes", "")),
        match_meta=match_meta,
        effect_meta=effect_meta,
        overlap_meta=support["overlap_meta"],
    )

    pearl_authorized = int(setup.get("pearl_estimation_authorized", 0)) == 1
    if not pearl_authorized:
        try:
            ident.identifiable = False
            ident.strategy = _as_str(ident.strategy or "") or "not_pearl_authorized"
            ident.failed_assumptions = sorted(set(list(getattr(ident, "failed_assumptions", []) or []) + [setup.get("pearl_authority_reason", "not_pearl_authorized")]))
            ident.notes = list(getattr(ident, "notes", []) or []) + ["estimation_contract_gate_applied"]
        except (TypeError, ValueError, AttributeError):
            pass

    eligible_flag = int(
        pearl_authorized
        and (len(controls) > 0)
        and (support["balance_pass"] == 1)
        and (support["propensity_meta"]["propensity_pass"] == 1)
        and (support["pretrend_meta"]["pretrend_pass"] == 1)
        and (support["overlap_meta"]["overlap_pass"] == 1)
    )
    reason_codes = _build_reason_codes(
        controls,
        match_meta,
        support["balance_pass"],
        support["propensity_meta"]["propensity_pass"],
        support["pretrend_meta"]["pretrend_pass"],
        support["overlap_meta"]["overlap_pass"],
        support["negctrl_flag"],
        setup,
        support["sensitivity_meta"],
        support["rosenbaum_meta"],
        support["signed_z"],
    )
    if not pearl_authorized:
        reason_codes.append("PEARL_NOT_AUTHORIZED")
        pr = _as_str(setup.get("pearl_authority_reason", ""))
        if pr:
            reason_codes.append(pr.upper())
        reason_codes = sorted(set(reason_codes))

    return TrialContext(
        raw_trial=tr,
        trial_id=trial_id,
        insight_meta=insight_meta,
        l29_meta=l29_meta,
        expected_direction=expected_direction,
        setup=setup,
        controls=controls,
        match_meta=match_meta,
        effect_meta=effect_meta,
        signed_z=support["signed_z"],
        direction_ok_flag=support["direction_ok_flag"],
        success_flag=support["success_flag"],
        smd_mean=support["smd_mean"],
        balance_pass=support["balance_pass"],
        propensity_meta=support["propensity_meta"],
        pretrend_meta=support["pretrend_meta"],
        overlap_meta=support["overlap_meta"],
        z_negctrl=support["z_negctrl"],
        negctrl_flag=support["negctrl_flag"],
        sensitivity_meta=support["sensitivity_meta"],
        rosenbaum_meta=support["rosenbaum_meta"],
        diagnostic=diag,
        identification=ident,
        reason_codes=reason_codes,
        eligible_flag=eligible_flag,
    )


def _trial_row_from_context(ctx: TrialContext) -> Dict[str, object]:
    trial_id = ctx.trial_id
    emeta = ctx.effect_meta
    pmeta = ctx.propensity_meta
    premeta = ctx.pretrend_meta
    ovmeta = ctx.overlap_meta
    sens = ctx.sensitivity_meta
    ros = ctx.rosenbaum_meta
    diag = ctx.diagnostic
    ident = ctx.identification
    setup = ctx.setup
    dag_ann = setup["dag_ann"]
    return {
        **trial_id,
        "window_days": int(setup["window_days"]),
        "expected_direction_on_target": ctx.expected_direction,
        "estimand": _as_str(setup.get("preferred_estimand", "")) or "ATT_proxy",
        "pearl_estimation_authorized": int(setup.get("pearl_estimation_authorized", 0)),
        "causal_claim_status": _as_str(setup.get("causal_claim_status", "")),
        "pearl_authority_reason": _as_str(setup.get("pearl_authority_reason", "")),
        "contract_required": int(setup.get("contract_required", 0)),
        "contract_row_present": int(setup.get("contract_row_present", 0)),
        "contract_source_authority": _as_str(setup.get("contract_source_authority", "")),
        "graph_authority_level": _as_str(setup.get("graph_authority_level", "")),
        "graph_identification_status": _as_str(setup.get("graph_identification_status", setup.get("graph_identification_strategy", ""))),
        "adherence_flag": _safe_float(ctx.raw_trial.get("adherence_flag", np.nan), np.nan),
        "matched_n": int(len(ctx.controls)),
        "eligible_flag": int(ctx.eligible_flag),
        "match_reason": _as_str(ctx.match_meta.get("reason", "")),
        "match_mode": _as_str(ctx.match_meta.get("match_mode", "strict")),
        "match_avg_dist_top": float(ctx.match_meta.get("avg_dist_top", np.nan)) if ctx.match_meta else np.nan,
        "match_avg_pretrend_gap": float(_safe_float(ctx.match_meta.get("avg_pretrend_gap", np.nan), np.nan)) if ctx.match_meta else np.nan,
        "common_support_low": float(_safe_float(ctx.match_meta.get("support_low", np.nan), np.nan)) if ctx.match_meta else np.nan,
        "common_support_high": float(_safe_float(ctx.match_meta.get("support_high", np.nan), np.nan)) if ctx.match_meta else np.nan,
        "common_support_margin": float(_safe_float(ctx.match_meta.get("support_margin", np.nan), np.nan)) if ctx.match_meta else np.nan,
        "trial_propensity_match": float(_safe_float(ctx.match_meta.get("trial_propensity", np.nan), np.nan)) if ctx.match_meta else np.nan,
        "support_rejects": int(_safe_float(ctx.match_meta.get("support_rejects", 0), 0)) if ctx.match_meta else 0,
        "caliper_rejects": int(_safe_float(ctx.match_meta.get("caliper_rejects", 0), 0)) if ctx.match_meta else 0,
        "pretrend_rejects": int(_safe_float(ctx.match_meta.get("pretrend_rejects", 0), 0)) if ctx.match_meta else 0,
        "balance_smd_mean": float(ctx.smd_mean) if np.isfinite(ctx.smd_mean) else np.nan,
        "balance_pass": int(ctx.balance_pass),
        "propensity_treated": float(pmeta["propensity_t"]) if np.isfinite(pmeta["propensity_t"]) else np.nan,
        "propensity_controls_mean": float(pmeta["propensity_c_mean"]) if np.isfinite(pmeta["propensity_c_mean"]) else np.nan,
        "propensity_pass": int(pmeta["propensity_pass"]),
        "overlap_pass": int(ovmeta["overlap_pass"]),
        "overlap_gap": float(ovmeta["overlap_gap"]) if np.isfinite(ovmeta["overlap_gap"]) else np.nan,
        "overlap_treated": float(ovmeta["overlap_treated"]) if np.isfinite(ovmeta["overlap_treated"]) else np.nan,
        "overlap_min_ctrl": float(ovmeta["overlap_min_ctrl"]) if np.isfinite(ovmeta["overlap_min_ctrl"]) else np.nan,
        "overlap_max_ctrl": float(ovmeta["overlap_max_ctrl"]) if np.isfinite(ovmeta["overlap_max_ctrl"]) else np.nan,
        "pretrend_pass": int(premeta["pretrend_pass"]),
        "pretrend_slope_trial": float(premeta["pretrend_trial"]) if np.isfinite(premeta["pretrend_trial"]) else np.nan,
        "pretrend_slope_controls_mean": float(premeta["pretrend_ctrl"]) if np.isfinite(premeta["pretrend_ctrl"]) else np.nan,
        "pretrend_diff_abs": float(premeta["pretrend_diff"]) if np.isfinite(premeta["pretrend_diff"]) else np.nan,
        "pretrend_reason": _as_str(premeta["pretrend_reason"]),
        "effect_window_raw": float(emeta["trial_mean"] - emeta["ctrl_mean_raw"]) if np.isfinite(emeta["trial_mean"]) and np.isfinite(emeta["ctrl_mean_raw"]) else np.nan,
        "controls_window_mean_raw": float(emeta["ctrl_mean_raw"]) if np.isfinite(emeta["ctrl_mean_raw"]) else np.nan,
        "controls_window_mean": float(emeta["ctrl_mean_adj"]) if np.isfinite(emeta["ctrl_mean_adj"]) else np.nan,
        "effect_window": float(emeta["effect"]) if np.isfinite(emeta["effect"]) else np.nan,
        "trial_window_mean": float(emeta["trial_mean"]) if np.isfinite(emeta["trial_mean"]) else np.nan,
        "effect_median": float(emeta["effect_median"]) if np.isfinite(emeta["effect_median"]) else np.nan,
        "effect_model": _as_str(emeta.get("effect_method", "matched_aipw")),
        "effect_estimator_primary": _as_str(emeta.get("effect_method", "matched_aipw")),
        "effect_se_proxy": float(emeta["effect_se_proxy"]) if np.isfinite(emeta["effect_se_proxy"]) else np.nan,
        "effect_bias_corrected": float(_safe_float(emeta.get("effect_bias_corrected", np.nan), np.nan)),
        "effect_ipw": float(_safe_float(emeta.get("effect_ipw", np.nan), np.nan)),
        "effect_dr_att": float(_safe_float(emeta.get("effect_dr_att", np.nan), np.nan)),
        "effect_dr_ate": float(_safe_float(emeta.get("effect_dr_ate", np.nan), np.nan)),
        "att_ipw": float(_safe_float(emeta.get("att_ipw", np.nan), np.nan)),
        "att_mean": float(_safe_float(emeta.get("att_mean", np.nan), np.nan)),
        "method_agreement": float(_safe_float(emeta.get("method_agreement", np.nan), np.nan)),
        "propensity_overlap_quality": float(_safe_float(emeta.get("propensity_overlap_quality", np.nan), np.nan)),
        "effect_ci_low_trial": float(_safe_float(emeta.get("effect_ci_low", np.nan), np.nan)),
        "effect_ci_high_trial": float(_safe_float(emeta.get("effect_ci_high", np.nan), np.nan)),
        "effect_sign_stable": int(_safe_float(emeta.get("effect_sign_stable", 0), 0)),
        "placebo_pvalue": float(_safe_float(emeta.get("placebo_pvalue", np.nan), np.nan)),
        "aipw_residual_term": float(emeta["residual_term"]) if np.isfinite(emeta["residual_term"]) else np.nan,
        "match_effective_n": float(emeta["match_effective_n"]) if np.isfinite(emeta["match_effective_n"]) else np.nan,
        "effective_support_n": float(_safe_float(emeta.get("match_effective_n", np.nan), np.nan)),
        "match_weight_max": float(emeta["match_weight_max"]) if np.isfinite(emeta["match_weight_max"]) else np.nan,
        "bias_correction_applied": float(emeta["bias_correction_applied"]) if np.isfinite(emeta["bias_correction_applied"]) else np.nan,
        "history_treated_n": float(_safe_float(emeta.get("history_treated_n", np.nan), np.nan)),
        "history_control_n": float(_safe_float(emeta.get("history_control_n", np.nan), np.nan)),
        "z_cf": float(ctx.signed_z) if np.isfinite(ctx.signed_z) else np.nan,
        "z_cf_raw": float(emeta["z_raw"]) if np.isfinite(emeta["z_raw"]) else np.nan,
        "direction_ok_flag": ctx.direction_ok_flag,
        "success_flag": ctx.success_flag,
        "z_negctrl": float(ctx.z_negctrl) if np.isfinite(ctx.z_negctrl) else np.nan,
        "success_flag_negctrl": ctx.negctrl_flag,
        "robustness_ratio": float(sens["robustness_ratio"]) if np.isfinite(sens["robustness_ratio"]) else np.nan,
        "robustness_evalue_like": float(sens["robustness_evalue_like"]) if np.isfinite(sens["robustness_evalue_like"]) else np.nan,
        "robustness_pass": int(sens["robustness_pass"]),
        "sensitivity_null_shift": float(sens["sensitivity_null_shift"]) if np.isfinite(sens["sensitivity_null_shift"]) else np.nan,
        "sensitivity_null_shift_sd": float(sens["sensitivity_null_shift_sd"]) if np.isfinite(sens["sensitivity_null_shift_sd"]) else np.nan,
        "sensitivity_ci_low_shift": float(sens["sensitivity_ci_low_shift"]) if np.isfinite(sens["sensitivity_ci_low_shift"]) else np.nan,
        "sensitivity_ci_low_shift_sd": float(sens["sensitivity_ci_low_shift_sd"]) if np.isfinite(sens["sensitivity_ci_low_shift_sd"]) else np.nan,
        "rosenbaum_n": ros["rosenbaum_n"],
        "rosenbaum_p_gamma_1": float(ros["rosenbaum_p_gamma_1"]) if np.isfinite(ros["rosenbaum_p_gamma_1"]) else np.nan,
        "rosenbaum_gamma_critical": float(ros["rosenbaum_gamma_critical"]) if np.isfinite(ros["rosenbaum_gamma_critical"]) else np.nan,
        "rosenbaum_pass": int(ros["rosenbaum_pass"]),
        "rosenbaum_method": _as_str(ros.get("rosenbaum_method", "")),
        "reason_codes": "|".join(ctx.reason_codes),
        "controls_idx": "|".join([str(i) for i in emeta.get("valid_control_ids", ctx.controls)[:K_CONTROLS]]),
        "covariates_used": _as_str(ctx.match_meta.get("covariates_used", "")),
        "outcome_col": setup.get("outcome_col", TARGET_COL),
        "preferred_estimand": setup.get("preferred_estimand", ""),
        "treatment_role": setup.get("treatment_role", ""),
        "treatment_kind": setup.get("treatment_kind", ""),
        "dag_action_source": setup["action_source"],
        "dag_action_resolved": _as_str(dag_ann.get("dag_action_resolved", "")),
        "dag_target_resolved": _as_str(dag_ann.get("dag_target_resolved", "")),
        "dag_action_known": int(_safe_float(dag_ann.get("dag_action_known", 0), 0)),
        "dag_target_known": int(_safe_float(dag_ann.get("dag_target_known", 0), 0)),
        "dag_action_type": _as_str(dag_ann.get("dag_action_type", "unknown")),
        "dag_target_type": _as_str(dag_ann.get("dag_target_type", "unknown")),
        "dag_action_time_role": _as_str(dag_ann.get("dag_action_time_role", "unknown")),
        "dag_target_time_role": _as_str(dag_ann.get("dag_target_time_role", "unknown")),
        "dag_adjustment_confidence": setup["dag_adjustment_confidence_effective"] or _as_str(dag_ann.get("dag_adjustment_confidence", "unknown")),
        "dag_adjustment_source": setup["dag_adjustment_source"],
        "dag_adjustment_notes": setup["dag_adjustment_notes"],
        "discovery_adjustment_set": setup["discovery_adjustment_set"],
        "discovery_forbidden_adjustments": setup["discovery_forbidden_adjustments"],
        "discovery_negative_control": setup["discovery_negative_control"],
        "dag_direct_edge_confidence": _as_str(dag_ann.get("dag_direct_edge_confidence", "unknown")),
        "dag_path_confidence": _as_str(dag_ann.get("dag_path_confidence", "unknown")),
        "dag_adjustment_set": setup["dag_adjustment_set"],
        "dag_forbidden_adjustments": setup["dag_forbidden_adjustments"],
        "dag_risk_paths": setup["dag_risk_paths"],
        "dag_negative_control": setup["dag_negative_control"],
        "dag_covariate_violation_flag": int(setup["dag_covariate_violation_flag"]),
        "diagnostic_grade": diag.diagnostic_grade,
        "sensitivity_level": diag.sensitivity_level,
        "diagnostic_notes": "|".join(diag.notes),
        "identification_strategy": ident.strategy,
        "graph_identification_strategy": setup.get("graph_identification_strategy", ""),
        "graph_identified": int(setup.get("graph_identified", 0)),
        "graph_estimand_expression": setup.get("graph_estimand_expression", ""),
        "identification_strength": ident.identification_strength,
        "identification_score": float(getattr(ident, "identification_score", np.nan)) if np.isfinite(_safe_float(getattr(ident, "identification_score", np.nan), np.nan)) else np.nan,
        "identifiable": int(ident.identifiable),
        "identification_notes": "|".join(ident.notes),
        "identification_failed_assumptions": "|".join(ident.failed_assumptions),
        "sample_size_ok": int(diag.sample_size_ok),
        "leakage_ok": int(diag.leakage_ok) if diag.leakage_ok is not None else np.nan,
        "drift_ok": int(diag.drift_ok) if diag.drift_ok is not None else np.nan,
        "temporal_order_ok": int(diag.temporal_order_ok) if diag.temporal_order_ok is not None else np.nan,
    }


def _ledger_row_from_context(ctx: TrialContext) -> Dict[str, object]:
    setup = ctx.setup
    dag_ann = setup["dag_ann"]
    emeta = ctx.effect_meta
    diag = ctx.diagnostic
    sens = ctx.sensitivity_meta
    ros = ctx.rosenbaum_meta
    return {
        "insight_id": ctx.trial_id["insight_id"],
        "trial_id": ctx.trial_id["trial_id"],
        "event": "trial_evaluated",
        "details": json.dumps({
            "matched_n": int(len(ctx.controls)),
            "eligible_flag": int(ctx.eligible_flag),
            "reason_codes": ctx.reason_codes,
            "effect_window": float(emeta["effect"]) if np.isfinite(emeta["effect"]) else None,
            "signed_z": float(ctx.signed_z) if np.isfinite(ctx.signed_z) else None,
            "overlap_pass": int(ctx.overlap_meta["overlap_pass"]),
            "robustness_ratio": float(sens["robustness_ratio"]) if np.isfinite(sens["robustness_ratio"]) else None,
            "rosenbaum_gamma_critical": float(ros["rosenbaum_gamma_critical"]) if np.isfinite(ros["rosenbaum_gamma_critical"]) else None,
            "dag_action_resolved": _as_str(dag_ann.get("dag_action_resolved", "")) or None,
            "dag_target_resolved": _as_str(dag_ann.get("dag_target_resolved", "")) or None,
            "dag_action_known": int(_safe_float(dag_ann.get("dag_action_known", 0), 0)),
            "dag_target_known": int(_safe_float(dag_ann.get("dag_target_known", 0), 0)),
            "dag_action_type": _as_str(dag_ann.get("dag_action_type", "unknown")) or None,
            "dag_target_type": _as_str(dag_ann.get("dag_target_type", "unknown")) or None,
            "dag_action_time_role": _as_str(dag_ann.get("dag_action_time_role", "unknown")) or None,
            "dag_target_time_role": _as_str(dag_ann.get("dag_target_time_role", "unknown")) or None,
            "dag_adjustment_confidence": _as_str(dag_ann.get("dag_adjustment_confidence", "unknown")) or None,
            "dag_direct_edge_confidence": _as_str(dag_ann.get("dag_direct_edge_confidence", "unknown")) or None,
            "dag_path_confidence": _as_str(dag_ann.get("dag_path_confidence", "unknown")) or None,
            "dag_adjustment_set": setup["dag_adjustment_set"] or None,
            "dag_forbidden_adjustments": setup["dag_forbidden_adjustments"] or None,
            "discovery_adjustment_set": setup["discovery_adjustment_set"] or None,
            "discovery_forbidden_adjustments": setup["discovery_forbidden_adjustments"] or None,
            "dag_risk_paths": setup["dag_risk_paths"] or None,
            "dag_negative_control": setup["dag_negative_control"] or None,
            "discovery_negative_control": setup["discovery_negative_control"] or None,
            "dag_covariate_violation_flag": int(setup["dag_covariate_violation_flag"]),
            "diagnostic_grade": diag.diagnostic_grade,
            "sensitivity_level": diag.sensitivity_level,
            "pearl_estimation_authorized": int(setup.get("pearl_estimation_authorized", 0)),
            "causal_claim_status": _as_str(setup.get("causal_claim_status", "")),
            "pearl_authority_reason": _as_str(setup.get("pearl_authority_reason", "")),
            "diagnostic_notes": diag.notes,
        }, ensure_ascii=False),
    }


def _evaluate_trial(tr, df, i_map, l29_map, dag, base_covs):
    ctx = _build_trial_context(tr, df, i_map, l29_map, dag, base_covs)
    if ctx is None:
        return None
    return _trial_row_from_context(ctx), _ledger_row_from_context(ctx)






def _evaluate_trials(trials, df, i_map, l29_map, dag, base_covs):
    """Evaluate a batch of experiment trials and return trial and ledger rows."""
    trial_rows = []
    ledger_rows = []
    if trials is None:
        return trial_rows, ledger_rows
    for _, tr in trials.iterrows():
        evaluated = _evaluate_trial(tr, df, i_map, l29_map, dag, base_covs)
        if evaluated is None:
            continue
        trial_row, ledger_row = evaluated
        trial_rows.append(trial_row)
        ledger_rows.append(ledger_row)
    return trial_rows, ledger_rows
