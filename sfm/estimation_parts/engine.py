

from runtime_env import configure_scientific_runtime
configure_scientific_runtime()

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from runtime_compat import assert_scientific_stack
assert_scientific_stack()
import pandas as pd

from . import common as C
from . import propensity as P
from . import matching as M
from . import diagnostics as D
from . import effects as E
from . import discovery_bridge as B
from . import pearl_backdoor as PB
from . import contract_gate as CG
from . import handoff_reader as HR
from . import _utils as U
from . import _outputs as O
from . import _aggregation as A
from . import _trial_eval as T


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
IDENTIFIED_EFFECTS_CSV = os.path.join(OUT_DIR, "identification", "identified_effects.csv")
LEGACY_IDENTIFIED_EFFECTS_CSV = os.path.join(OUT_DIR, "identified_effects.csv")
CAUSAL_CONTRACT_CSV = os.path.join(OUT_DIR, "causal_contract.csv")
EFFECT_ESTIMATES_CSV = os.path.join(OUT_DIR, "estimation", "effect_estimates.csv")
ESTIMATION_AUTHORITY_REPORT_CSV = os.path.join(OUT_DIR, "estimation", "estimation_authority_report.csv")
ESTIMATION_PLAN_CSV = os.path.join(OUT_DIR, "estimation", "estimation_plan.csv")
OUT_L3 = C.OUT_L3
OUT_LEDGER = C.OUT_LEDGER
OUT_TRIALS = C.OUT_TRIALS
SCM_STRUCTURAL_MODELS_CSV = os.path.join(OUT_DIR, "scm", "structural_models.csv")
PRETREND_DAYS = C.PRETREND_DAYS
PROPENSITY_ACTION_COL = C.PROPENSITY_ACTION_COL
PROPENSITY_MAX_DIFF = C.PROPENSITY_MAX_DIFF
TARGET_COL = C.TARGET_COL
Z_SUCCESS = C.Z_SUCCESS

_estimate_effect_bundle = E.estimate_effect_bundle
_apply_dag_covariates = C._apply_dag_covariates
_as_str = C._as_str
_bootstrap_ci = P._bootstrap_ci
_compute_propensity = P._compute_propensity
_confidence_score = D._confidence_score
_covariate_balance = D._covariate_balance
_ensure_calendar_covs = C._ensure_calendar_covs
_ensure_out = C._ensure_out
_hybrid_match_controls = M._hybrid_match_controls
_load_dag = C._load_dag
_load_l29_summary = P._load_l29_summary
_numeric_cols = C._numeric_cols
_overlap_check = D._overlap_check
_pick_trials_path = P._pick_trials_path
_pretrend_check = P._pretrend_check
_risk_level = D._risk_level
_rosenbaum_sensitivity = D._rosenbaum_sensitivity
_safe_float = C._safe_float
_select_negative_control_col = C._select_negative_control_col
_sensitivity_metrics = D._sensitivity_metrics
_signed_z = P._signed_z
_sr_lower_bound = C._sr_lower_bound
_status_from_metrics = D._status_from_metrics
_try_parse_date = C._try_parse_date


@dataclass
class InputBundle:
    trials_path: str
    insights: pd.DataFrame
    trials: pd.DataFrame
    l29: pd.DataFrame
    data_path: Optional[str]
    data: Optional[pd.DataFrame]
    bridge: Optional[pd.DataFrame] = None
    identified_effects: Optional[pd.DataFrame] = None
    causal_contract: Optional[pd.DataFrame] = None
    estimation_handoff: Optional[pd.DataFrame] = None
    contract_required: bool = False


@dataclass
class OutputBundle:
    l3: pd.DataFrame
    trials: pd.DataFrame
    ledger: pd.DataFrame


def _mode_or_empty(series):
    return U.mode_or_empty(series)

def _resolve_data_path(explicit_path=None, df_trials=None):
    """Resolve the best data CSV without reading large files into memory.

    Older versions read up to 2000 rows per candidate. This fast path uses
    file existence, row counts and headers only; it avoids expensive heuristic
    reads on production-size CSV files.
    """
    if explicit_path and os.path.exists(explicit_path):
        return explicit_path

    candidates = []
    for p in [os.path.join(OUT_DIR, "data_clean.csv"), DEFAULT_DATA_CSV, FALLBACK_DATA_CSV]:
        if p and os.path.exists(p) and p not in candidates:
            candidates.append(p)
    if not candidates:
        return FALLBACK_DATA_CSV

    max_tidx = -1
    if df_trials is not None and len(df_trials):
        tidx = pd.to_numeric(df_trials.get("t_index", pd.Series([], dtype=float)), errors="coerce")
        finite = tidx[np.isfinite(tidx)] if len(tidx) else pd.Series([], dtype=float)
        if len(finite):
            max_tidx = int(np.max(finite))

    best_path = candidates[0]
    best_score = -10**9
    for path in candidates:
        n_rows = U.count_csv_rows_fast(path)
        header = set(U.read_csv_header(path))
        score = n_rows
        if max_tidx >= 0 and n_rows > max_tidx:
            score += 1000
        # Prefer canonical cleaned data when coverage is otherwise similar.
        if os.path.normpath(path).endswith(os.path.normpath(os.path.join(OUT_DIR, "data_clean.csv"))):
            score += 25
        if DATE_COL in header:
            score += 10
        if TARGET_COL in header:
            score += 10
        if score > best_score:
            best_score = score
            best_path = path
    return best_path

def _load_inputs(data_path=None) -> InputBundle:
    if not os.path.exists(INSIGHTS_L2):
        raise FileNotFoundError("Missing %s (run Level 2.5 first)." % INSIGHTS_L2)

    trials_path = _pick_trials_path()
    if not os.path.exists(trials_path):
        return InputBundle(trials_path, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None, None, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    df_i = pd.read_csv(INSIGHTS_L2)
    df_r = pd.read_csv(trials_path)
    if "insight_id" in df_i.columns and "insight_id" in df_r.columns:
        valid_ids = set(df_i["insight_id"].astype(str).tolist())
        df_r = df_r[df_r["insight_id"].astype(str).isin(valid_ids)].copy()
    df_l29 = _load_l29_summary()
    if not df_l29.empty and "insight_id" in df_l29.columns and "insight_id" in df_i.columns:
        valid_ids = set(df_i["insight_id"].astype(str).tolist())
        df_l29 = df_l29[df_l29["insight_id"].astype(str).isin(valid_ids)].copy()

    # Load a bridge only if it was generated from the current insights_level2.csv.
    # If the bridge is missing/stale, rebuild it immediately from df_i so Estimation
    # never consumes stale PCMCI metadata.
    bridge = B.load_bridge(OUT_DIR, source_insights_path=INSIGHTS_L2, source_df=df_i)
    if bridge.empty and not df_i.empty:
        try:
            _, _, bridge = B.write_bridge_from_insights(df_i, OUT_DIR, source_insights_path=INSIGHTS_L2)
        except (OSError, ValueError, TypeError, RuntimeError, pd.errors.ParserError):
            bridge = pd.DataFrame()
    if not bridge.empty and "insight_id" in bridge.columns and "insight_id" in df_i.columns:
        valid_ids = set(df_i["insight_id"].astype(str).tolist())
        bridge = bridge[bridge["insight_id"].astype(str).isin(valid_ids)].copy()

    identified_effects = pd.DataFrame()
    causal_contract = pd.DataFrame()
    estimation_handoff = pd.DataFrame()
    structural_models = pd.DataFrame()
    contract_required = os.path.exists(CAUSAL_CONTRACT_CSV)

    # Step 122: if the canonical contract file exists, estimation is contract-driven.
    # Legacy identified_effects.csv is used only when the contract file is absent.
    # A header-only contract intentionally means: no Pearl-authorized effects yet.

    # Step 118: estimation treats out/causal_contract.csv as the canonical
    # Discovery/SCM/Identification handoff. identified_effects.csv is now only
    # a fallback for old runs that have not generated a contract yet.
    if os.path.exists(CAUSAL_CONTRACT_CSV):
        try:
            causal_contract = pd.read_csv(CAUSAL_CONTRACT_CSV)
        except (OSError, ValueError, TypeError, pd.errors.ParserError):
            causal_contract = pd.DataFrame()
    estimation_handoff = HR.load_estimation_handoff(OUT_DIR)
    if not estimation_handoff.empty:
        # Step 190: Estimation is handoff-driven. The curated handoff is
        # narrower than the full causal contract and is the preferred estimator input.
        causal_contract = HR.prefer_handoff_contract(causal_contract, estimation_handoff)
    if not causal_contract.empty and "insight_id" in causal_contract.columns and "insight_id" in df_i.columns:
        # Keep curated estimation_handoff rows even when Level 3 trial insights are sparse.
        # Trial aggregation still requires matching trials; Pearl/backdoor estimators can use
        # the contract-like handoff directly.
        if estimation_handoff.empty:
            valid_ids = set(df_i["insight_id"].astype(str).tolist())
            causal_contract = causal_contract[causal_contract["insight_id"].astype(str).isin(valid_ids)].copy()

    if (not contract_required) and causal_contract.empty:
        ident_path = IDENTIFIED_EFFECTS_CSV if os.path.exists(IDENTIFIED_EFFECTS_CSV) else LEGACY_IDENTIFIED_EFFECTS_CSV
        if os.path.exists(ident_path):
            try:
                identified_effects = pd.read_csv(ident_path)
            except (OSError, ValueError, TypeError, pd.errors.ParserError):
                identified_effects = pd.DataFrame()
        if not identified_effects.empty and "insight_id" in identified_effects.columns and "insight_id" in df_i.columns:
            valid_ids = set(df_i["insight_id"].astype(str).tolist())
            identified_effects = identified_effects[identified_effects["insight_id"].astype(str).isin(valid_ids)].copy()
    if os.path.exists(SCM_STRUCTURAL_MODELS_CSV):
        try:
            structural_models = pd.read_csv(SCM_STRUCTURAL_MODELS_CSV)
        except (OSError, ValueError, TypeError, pd.errors.ParserError):
            structural_models = pd.DataFrame()
    resolved_path = _resolve_data_path(data_path, df_r)
    df = pd.read_csv(resolved_path)
    df = _try_parse_date(df)
    df = _ensure_calendar_covs(df)
    bridge = B.sanitize_bridge_for_data(bridge, df)
    bundle = InputBundle(trials_path, df_i, df_r, df_l29, resolved_path, df, bridge, identified_effects, causal_contract, estimation_handoff, contract_required)
    bundle.structural_models = structural_models
    return bundle


def _build_covariates(df: pd.DataFrame) -> List[str]:
    outcome_col = C.resolve_outcome_col(df, TARGET_COL)
    if outcome_col not in df.columns:
        raise ValueError("Target column '%s' not found in data." % outcome_col)

    out = df
    outcome_num = pd.to_numeric(out[outcome_col], errors="coerce").astype(float)
    if "outcome_prev" not in out.columns:
        out["outcome_prev"] = outcome_num.shift(1)
    # Backwards-compatible aliases: legacy modules/tests may still ask for target_*.
    if "target_prev" not in out.columns:
        out["target_prev"] = out["outcome_prev"]

    # Past-only temporal covariates make the offline counterfactual more stable
    # without leaking post-treatment information into matching / estimation.
    for lag in (1, 2, 3):
        outcome_lag = "outcome_lag_%d" % lag
        legacy_lag = "target_lag_%d" % lag
        if outcome_lag not in out.columns:
            out[outcome_lag] = outcome_num.shift(lag)
        if legacy_lag not in out.columns:
            out[legacy_lag] = out[outcome_lag]
    if "outcome_delta_prev" not in out.columns:
        out["outcome_delta_prev"] = out["outcome_lag_1"] - out["outcome_lag_2"]
    if "target_delta_prev" not in out.columns:
        out["target_delta_prev"] = out["outcome_delta_prev"]
    for window in (3, 7):
        outcome_mean = "outcome_roll_mean_%d" % window
        outcome_std = "outcome_roll_std_%d" % window
        legacy_mean = "target_roll_mean_%d" % window
        legacy_std = "target_roll_std_%d" % window
        if outcome_mean not in out.columns:
            out[outcome_mean] = outcome_num.shift(1).rolling(window, min_periods=max(2, window // 2)).mean()
        if outcome_std not in out.columns:
            out[outcome_std] = outcome_num.shift(1).rolling(window, min_periods=max(2, window // 2)).std()
        if legacy_mean not in out.columns:
            out[legacy_mean] = out[outcome_mean]
        if legacy_std not in out.columns:
            out[legacy_std] = out[outcome_std]

    primary = [
        "outcome_prev",
        "outcome_lag_2",
        "outcome_lag_3",
        "outcome_delta_prev",
        "outcome_roll_mean_3",
        "outcome_roll_std_3",
        "outcome_roll_mean_7",
        "time_idx",
        "dow_sin",
        "dow_cos",
    ]
    covs: List[str] = []
    for c in primary:
        if c in out.columns and c not in covs:
            covs.append(c)

    derived_exclusions = {
        outcome_col,
        DATE_COL,
        NEGCTRL_OUTCOME_COL,
        PROPENSITY_ACTION_COL,
        "__propensity__",
    }
    lag_like = {c for c in out.columns if c.startswith(("target_lag_", "target_roll_", "outcome_lag_", "outcome_roll_"))}
    derived_exclusions.update(lag_like)
    derived_exclusions.update({"outcome_prev", "target_prev", "outcome_delta_prev", "target_delta_prev", "time_idx", "dow_sin", "dow_cos"})

    extra_numeric = []
    for c in _numeric_cols(out):
        if c in derived_exclusions or c in covs:
            continue
        extra_numeric.append(c)

    # Keep a compact but richer covariate set: at most 4 extra source features.
    for c in extra_numeric[:4]:
        covs.append(c)
        lag_col = "%s_lag_1" % c
        roll_col = "%s_roll_mean_3" % c
        src = pd.to_numeric(out[c], errors="coerce").astype(float)
        if lag_col not in out.columns:
            out[lag_col] = src.shift(1)
        if roll_col not in out.columns:
            out[roll_col] = src.shift(1).rolling(3, min_periods=2).mean()
        if lag_col not in covs:
            covs.append(lag_col)
        if roll_col not in covs:
            covs.append(roll_col)
        if len(covs) >= 16:
            break
    return covs[:16]


def _ensure_action_indicator(df: pd.DataFrame, df_trials: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if PROPENSITY_ACTION_COL in out.columns:
        return out

    out[PROPENSITY_ACTION_COL] = 0
    active = pd.to_numeric(
        df_trials.get("action_active", df_trials.get("adherence_flag", pd.Series(np.ones(len(df_trials))))),
        errors="coerce",
    ).fillna(0).to_numpy(dtype=float)
    tidx = pd.to_numeric(
        df_trials.get("t_index", pd.Series([-999] * len(df_trials))),
        errors="coerce",
    ).fillna(-999).astype(int).to_numpy(dtype=int)

    for i, t in enumerate(tidx):
        if 0 <= int(t) < len(out):
            out.loc[int(t), PROPENSITY_ACTION_COL] = int(float(active[i]) > 0.5)
    return out


def _build_metadata_maps(df_i: pd.DataFrame, df_l29: pd.DataFrame, df_bridge: Optional[pd.DataFrame] = None, df_identified_effects: Optional[pd.DataFrame] = None, df_structural_models: Optional[pd.DataFrame] = None, df_causal_contract: Optional[pd.DataFrame] = None, contract_required: bool = False):
    i_map = df_i.set_index("insight_id").to_dict(orient="index") if "insight_id" in df_i.columns else {}
    if contract_required:
        for iid, meta in list(i_map.items()):
            base = dict(meta)
            base["__contract_required"] = 1
            base["__contract_row_present"] = 0
            base["contract_mode"] = "causal_contract_required"
            i_map[iid] = base
    if df_bridge is not None and len(df_bridge) > 0 and "insight_id" in df_bridge.columns:
        for iid, meta in df_bridge.set_index("insight_id").to_dict(orient="index").items():
            base = dict(i_map.get(iid, {}))
            base.update({k: v for k, v in meta.items() if not (isinstance(v, float) and np.isnan(v))})
            if _as_str(base.get("treatment_col", "")) and not _as_str(base.get("source", "")):
                base["source"] = _as_str(base.get("treatment_col", ""))
            if _as_str(base.get("outcome_col", "")) and not _as_str(base.get("target_col", "")):
                base["target_col"] = _as_str(base.get("outcome_col", ""))
            i_map[iid] = base
    if df_causal_contract is not None and len(df_causal_contract) > 0 and "insight_id" in df_causal_contract.columns:
        for iid, meta in df_causal_contract.set_index("insight_id").to_dict(orient="index").items():
            base = dict(i_map.get(iid, {}))
            clean_meta = {k: v for k, v in meta.items() if not (isinstance(v, float) and np.isnan(v))}
            clean_meta["__contract_row_present"] = 1
            clean_meta["__contract_required"] = 1
            clean_meta["contract_mode"] = "causal_contract_primary"
            base.update(clean_meta)
            if _as_str(base.get("treatment_col", "")) and not _as_str(base.get("source", "")):
                base["source"] = _as_str(base.get("treatment_col", ""))
            if _as_str(base.get("outcome_col", "")) and not _as_str(base.get("target_col", "")):
                base["target_col"] = _as_str(base.get("outcome_col", ""))
            if _as_str(base.get("adjustment_set", "")):
                base["candidate_covariates"] = _as_str(base.get("adjustment_set", ""))
                base["suggested_adjustment_set"] = _as_str(base.get("adjustment_set", ""))
            if _as_str(base.get("forbidden_adjustment_set", "")):
                base["forbidden_adjustment_set"] = _as_str(base.get("forbidden_adjustment_set", ""))
            if _as_str(base.get("negative_controls", "")):
                base["suggested_negative_control"] = _as_str(base.get("negative_controls", ""))
            if _as_str(base.get("estimand_type", "")):
                base["preferred_estimand"] = _as_str(base.get("estimand_type", ""))
            if _as_str(base.get("source_authority", "")):
                base["contract_source_authority"] = _as_str(base.get("source_authority", ""))
            if _as_str(base.get("source_artifacts", "")):
                base["contract_source_artifacts"] = _as_str(base.get("source_artifacts", ""))
            base["adjustment_set_confidence"] = "high" if _as_str(base.get("adjustment_set", "")) else _as_str(base.get("adjustment_set_confidence", ""))
            base["graph_identification_strategy"] = _as_str(base.get("identification_strategy", ""))
            base["graph_identified"] = int(_safe_float(base.get("identified", 0), 0))
            base["graph_identification_status"] = _as_str(base.get("identification_status", ""))
            base["graph_effect_scope"] = _as_str(base.get("effect_scope", ""))
            base["graph_authority_level"] = _as_str(base.get("authority_level", ""))
            base["graph_authority_reason"] = _as_str(base.get("authority_reason", ""))
            i_map[iid] = base
    if df_identified_effects is not None and len(df_identified_effects) > 0 and "insight_id" in df_identified_effects.columns:
        for iid, meta in df_identified_effects.set_index("insight_id").to_dict(orient="index").items():
            base = dict(i_map.get(iid, {}))
            clean_meta = {k: v for k, v in meta.items() if not (isinstance(v, float) and np.isnan(v))}
            clean_meta["__contract_row_present"] = 0
            clean_meta["__contract_required"] = 0
            clean_meta["contract_mode"] = "legacy_identified_effects_fallback"
            base.update(clean_meta)
            if _as_str(base.get("treatment_col", "")) and not _as_str(base.get("source", "")):
                base["source"] = _as_str(base.get("treatment_col", ""))
            if _as_str(base.get("outcome_col", "")) and not _as_str(base.get("target_col", "")):
                base["target_col"] = _as_str(base.get("outcome_col", ""))
            if _as_str(base.get("adjustment_set", "")):
                base["candidate_covariates"] = _as_str(base.get("adjustment_set", ""))
                base["suggested_adjustment_set"] = _as_str(base.get("adjustment_set", ""))
            if _as_str(base.get("forbidden_adjustments", "")):
                base["forbidden_adjustment_set"] = _as_str(base.get("forbidden_adjustments", ""))
            if _as_str(base.get("negative_controls", "")):
                base["suggested_negative_control"] = _as_str(base.get("negative_controls", ""))
            if _as_str(base.get("estimand_type", "")):
                base["preferred_estimand"] = _as_str(base.get("estimand_type", ""))
            if _as_str(base.get("source_authority", "")):
                base["contract_source_authority"] = _as_str(base.get("source_authority", ""))
            if _as_str(base.get("source_artifacts", "")):
                base["contract_source_artifacts"] = _as_str(base.get("source_artifacts", ""))
            base["adjustment_set_confidence"] = "high" if _as_str(base.get("adjustment_set", "")) else _as_str(base.get("adjustment_set_confidence", ""))
            base["graph_identification_strategy"] = _as_str(base.get("identification_strategy", ""))
            base["graph_identified"] = int(_safe_float(base.get("identified", 0), 0))
            base["graph_identification_status"] = _as_str(base.get("identification_status", ""))
            base["graph_identification_route"] = _as_str(base.get("identification_route", ""))
            base["graph_identification_vs_simulation"] = _as_str(base.get("identification_vs_simulation", ""))
            base["graph_simulation_status"] = _as_str(base.get("simulation_status", ""))
            base["graph_formal_effects_report"] = _as_str(base.get("formal_effects_report", ""))
            base["graph_total_effect_status"] = _as_str(base.get("total_effect_status", ""))
            base["graph_total_effect_route"] = _as_str(base.get("total_effect_route", ""))
            base["graph_total_effect_estimand"] = _as_str(base.get("total_effect_estimand", ""))
            base["graph_direct_effect_status"] = _as_str(base.get("direct_effect_status", ""))
            base["graph_direct_effect_route"] = _as_str(base.get("direct_effect_route", ""))
            base["graph_direct_effect_estimand"] = _as_str(base.get("direct_effect_estimand", ""))
            base["graph_controlled_direct_effect_status"] = _as_str(base.get("controlled_direct_effect_status", ""))
            base["graph_controlled_direct_effect_route"] = _as_str(base.get("controlled_direct_effect_route", ""))
            base["graph_controlled_direct_effect_estimand"] = _as_str(base.get("controlled_direct_effect_estimand", ""))
            base["graph_natural_direct_effect_status"] = _as_str(base.get("natural_direct_effect_status", ""))
            base["graph_natural_direct_effect_route"] = _as_str(base.get("natural_direct_effect_route", ""))
            base["graph_natural_direct_effect_estimand"] = _as_str(base.get("natural_direct_effect_estimand", ""))
            base["graph_natural_indirect_effect_status"] = _as_str(base.get("natural_indirect_effect_status", ""))
            base["graph_natural_indirect_effect_route"] = _as_str(base.get("natural_indirect_effect_route", ""))
            base["graph_natural_indirect_effect_estimand"] = _as_str(base.get("natural_indirect_effect_estimand", ""))
            base["graph_natural_effects_status"] = _as_str(base.get("natural_effects_status", ""))
            base["graph_natural_effects_route"] = _as_str(base.get("natural_effects_route", ""))
            base["graph_effect_specific_routes"] = _as_str(base.get("effect_specific_routes", ""))
            base["graph_estimand_expression"] = _as_str(base.get("estimand_expression", ""))
            base["graph_assumptions"] = _as_str(base.get("assumptions", ""))
            base["graph_failed_assumptions"] = _as_str(base.get("failed_assumptions", ""))
            base["graph_effect_scope"] = _as_str(base.get("effect_scope", ""))
            base["graph_total_adjustment_set"] = _as_str(base.get("total_adjustment_set", base.get("adjustment_set", "")))
            base["graph_direct_adjustment_set"] = _as_str(base.get("direct_adjustment_set", ""))
            base["graph_direct_effect_identifiable"] = int(_safe_float(base.get("direct_effect_identifiable", 0), 0))
            base["graph_frontdoor_candidate"] = int(_safe_float(base.get("frontdoor_candidate", 0), 0))
            base["graph_frontdoor_identifiable"] = int(_safe_float(base.get("frontdoor_identifiable", 0), 0))
            base["graph_frontdoor_report"] = _as_str(base.get("frontdoor_report", ""))
            base["graph_dsep_report"] = _as_str(base.get("dsep_report", ""))
            base["graph_minimal_adjustment_sets"] = _as_str(base.get("minimal_adjustment_sets", ""))
            base["graph_minimal_direct_adjustment_sets"] = _as_str(base.get("minimal_direct_adjustment_sets", ""))
            base["graph_dsep_backdoor_identifiable"] = int(_safe_float(base.get("dsep_backdoor_identifiable", base.get("backdoor_identifiable", 0)), 0))
            base["graph_dsep_direct_effect_identifiable"] = int(_safe_float(base.get("dsep_direct_effect_identifiable", base.get("direct_effect_identifiable", 0)), 0))
            base["graph_nested_effects_report"] = _as_str(base.get("nested_effects_report", ""))
            base["graph_cde_identified"] = int(_safe_float(base.get("cde_identified", 0), 0))
            base["graph_nde_identified"] = int(_safe_float(base.get("nde_identified", 0), 0))
            base["graph_nie_identified"] = int(_safe_float(base.get("nie_identified", 0), 0))
            base["graph_cross_world_plausible"] = int(_safe_float(base.get("cross_world_plausible", 0), 0))
            i_map[iid] = base
    if df_structural_models is not None and len(df_structural_models) > 0 and "node_id" in df_structural_models.columns:
        scm_map = df_structural_models.set_index("node_id").to_dict(orient="index")
        for iid, base in list(i_map.items()):
            src = _as_str(base.get("source", base.get("treatment_col", "")))
            out = _as_str(base.get("target_col", base.get("outcome_col", "")))
            s_src = scm_map.get(src, {})
            s_out = scm_map.get(out, {})
            if s_src or s_out:
                base = dict(base)
                base["scm_support"] = 1
                base["scm_treatment_fit_score"] = _safe_float(s_src.get("fit_score", ""), 0.0)
                base["scm_outcome_fit_score"] = _safe_float(s_out.get("fit_score", ""), 0.0)
                base["scm_outcome_parents"] = _as_str(s_out.get("parents", ""))
                base["scm_outcome_structural_family"] = _as_str(s_out.get("structural_family", ""))
                base["scm_outcome_model_class"] = _as_str(s_out.get("model_class", ""))
                base["scm_outcome_feature_terms"] = _as_str(s_out.get("feature_terms", ""))
                base["scm_outcome_noise_family"] = _as_str(s_out.get("noise_family", ""))
                base["scm_outcome_residual_sd"] = _safe_float(s_out.get("residual_sd", ""), 0.0)
                base["scm_outcome_support_min"] = s_out.get("support_min", "")
                base["scm_outcome_support_max"] = s_out.get("support_max", "")
                base["scm_outcome_structural_equation"] = _as_str(s_out.get("structural_equation", ""))
                base["scm_outcome_latent_equation"] = _as_str(s_out.get("latent_equation", ""))
                base["scm_treatment_model_class"] = _as_str(s_src.get("model_class", ""))
                base["scm_treatment_noise_family"] = _as_str(s_src.get("noise_family", ""))
                base["scm_treatment_structural_equation"] = _as_str(s_src.get("structural_equation", ""))
                i_map[iid] = base
    l29_map = df_l29.set_index("insight_id").to_dict(orient="index") if (not df_l29.empty and "insight_id" in df_l29.columns) else {}
    return i_map, l29_map













# Output helpers live in estimation_parts._outputs; aliases preserve selected old
# private names for compatibility with tests and external scripts.
_authority_effects_frame = O._authority_effects_frame
_write_estimation_authority_report = O._write_estimation_authority_report
_write_authority_effects = O._write_authority_effects
_write_outputs = O._write_outputs

def _empty_outputs():
    pd.DataFrame(columns=["insight_id", "action_name", "date", "t_index", "adherence_flag", "dose", "notes"]).to_csv(OUT_TRIALS, index=False)
    pd.DataFrame(columns=["insight_id", "status", "decision", "confidence"]).to_csv(OUT_L3, index=False)
    pd.DataFrame(columns=["insight_id", "event", "details"]).to_csv(OUT_LEDGER, index=False)
    empty_plan = pd.DataFrame(columns=HR.PLAN_COLUMNS)
    os.makedirs(os.path.join(OUT_DIR, "estimation"), exist_ok=True)
    HR.write_estimation_plan(empty_plan, OUT_DIR)
    SEN.write_sensitivity_analysis(empty_plan, OUT_DIR)
    EE.write_effect_estimates_from_frames(None, empty_plan, OUT_DIR)
    try:
        from contracts.causal_report import write_causal_report
        write_causal_report(out_dir=OUT_DIR)
    except (OSError, ValueError, TypeError, RuntimeError, ImportError):
        pass


# Trial evaluation helpers live in estimation_parts._trial_eval.
TrialContext = T.TrialContext
_trial_identity = T._trial_identity
_trial_is_usable = T._trial_is_usable
_expected_direction = T._expected_direction
_pearl_contract_authority = T._pearl_contract_authority
_trial_setup = T._trial_setup
_effect_bundle = T._effect_bundle
_propensity_metrics = T._propensity_metrics
_pretrend_metrics = T._pretrend_metrics
_overlap_metrics = T._overlap_metrics
_diagnostic_and_identification = T._diagnostic_and_identification
_build_reason_codes = T._build_reason_codes
_evaluate_matching = T._evaluate_matching
_evaluate_support_metrics = T._evaluate_support_metrics
_build_trial_context = T._build_trial_context
_trial_row_from_context = T._trial_row_from_context
_ledger_row_from_context = T._ledger_row_from_context
_evaluate_trial = T._evaluate_trial
_evaluate_trials = T._evaluate_trials


# Aggregation helpers live in estimation_parts._aggregation.
_aggregate_insight_identity = A._aggregate_insight_identity
_aggregate_insight = A._aggregate_insight
_aggregate_all_insights = A._aggregate_all_insights

def _print_summary(data_path: str, trials_path: str, df_l3: pd.DataFrame, df_trials: pd.DataFrame, feedback_path: str = "") -> None:
    print("\n=== PCB LEVEL 3.2 (AIPW + overlap + serious sensitivity) ===")
    print("Data:", data_path)
    print("Inputs:", INSIGHTS_L2, "+", trials_path)
    print("Saved:", OUT_TRIALS)
    print("Saved:", OUT_LEDGER)
    print("Saved:", OUT_L3)
    print("Saved:", os.path.join(OUT_DIR, "estimation", "effect_estimates.csv"))
    print("Saved:", ESTIMATION_AUTHORITY_REPORT_CSV)
    print("Saved:", ESTIMATION_PLAN_CSV)
    if feedback_path:
        print("Saved:", feedback_path)
    print("Trials enriched:", int(len(df_trials)))
    print("Insights evaluated:", int(len(df_l3)))
    if len(df_l3) > 0:
        cols = [c for c in ["insight_id", "status", "decision", "confidence", "n_trials", "success_rate_lb", "att_mean"] if c in df_l3.columns]
        print(df_l3[cols].head(10).to_string(index=False))


def main(data_path=None):
    _ensure_out()

    bundle = _load_inputs(data_path)
    if bundle.data is None or bundle.data_path is None:
        _empty_outputs()
        print("No trials found for Level 3.2")
        return 0

    df = bundle.data
    covs = _build_covariates(df)
    df = _ensure_action_indicator(df, bundle.trials)
    if ENABLE_PROPENSITY:
        df["__propensity__"] = _compute_propensity(df, PROPENSITY_ACTION_COL, covs)

    i_map, l29_map = _build_metadata_maps(bundle.insights, bundle.l29, bundle.bridge, bundle.identified_effects, getattr(bundle, "structural_models", pd.DataFrame()), getattr(bundle, "causal_contract", pd.DataFrame()), contract_required=getattr(bundle, "contract_required", False))
    # Step 121: causal_contract is the canonical downstream handoff. The seed DAG
    # is still useful as a conservative prior/fallback, but raw edges never become
    # an estimation authority.
    dag = _load_dag()

    trial_rows, ledger_rows = _evaluate_trials(bundle.trials, df, i_map, l29_map, dag, covs)
    df_trials = pd.DataFrame(trial_rows)
    df_ledger = pd.DataFrame(ledger_rows)
    df_l3 = _aggregate_all_insights(df_trials, l29_map)
    output_path = _write_outputs(df_l3, df_trials, df_ledger, df_data=df, df_contract=getattr(bundle, "causal_contract", pd.DataFrame()), insights=bundle.insights, contract_required=getattr(bundle, "contract_required", False), estimation_handoff=getattr(bundle, "estimation_handoff", pd.DataFrame()))
    _print_summary(bundle.data_path, bundle.trials_path, df_l3, df_trials, output_path)
    return OutputBundle(df_l3, df_trials, df_ledger)


def build_argparser():
    p = argparse.ArgumentParser(
        prog="estimation.py",
        description="PCB Level 3.2 — AIPW-style counterfactual validation with overlap and sensitivity analysis",
    )
    p.add_argument("--data", default=None, help="Optional data path")
    return p


def cli(argv=None):
    args = build_argparser().parse_args(sys.argv[1:] if argv is None else argv)
    main(data_path=args.data)
    return 0
