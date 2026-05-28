# FILE: config.py
# Python 3.10+ compatible
#
# Central config loader for Amantia (local-first).
# If pcb.json exists in the project root, CLI commands and offline/runtime
# modules can override defaults such as outcome_col/date_col/out_dir without
# editing code. The canonical pipeline is defined by cli.py, not by legacy
# numbered pipeline flags.

import json
import os

DEFAULT_CONFIG = {
    "outcome_col": "harm_event",
    "target_col": "harm_event",  # legacy alias for outcome_col
    "date_col": "date",
    "out_dir": "out",
    "data_primary": "data.csv",
    "data_fallback": "data.csv",
    "use_data_clean_if_present": True,
    "level25": {
        "max_lag": 7,
        "ar_order": 2,
        "min_obs": 40,
        "detrend_mode": "none",
        "legacy_granger_alpha": 0.05,  # legacy key retained for old configs; ignored by the current ranking path
        "min_rss_reduction": 0.02,
        "min_causal_score": 0.35,
        "discovery_mode": "conservative",
        "keep_min_selection_score": 0.35,
        "keep_min_evidence_score": 0.34,
        "keep_min_priority": 0.35,
        "causal_min_incremental_r2": 0.02,
        "family_map_path": "candidate_family_map.yaml",
        "use_bh_fdr": True,
        "bh_fdr_alpha": 0.10,
        "placebo_future_enable": True,
        "placebo_future_alpha": 0.20,
        "placebo_perm_enable": True,
        "placebo_perm_B": 20,
        "placebo_block_len": 7,
        "negctrl_enable": True,
        "negctrl_outcome_col": "negative_control_outcome",
        "negctrl_alpha": 0.10,
        "stability_enable": True,
        "stability_window": 30,
        "stability_stride": 7,
        "stability_min_windows": 3,
        "stability_min_score": 0.60,
        "enable_guardrails": True,
        "drop_flagged": True,
        "lag0_corr_hard": 0.95,
        "leakage_corr_future_hard": 0.97,
        "leakage_gap_min": 0.10,
        "drift_corr_source": 0.85,
        "drift_corr_target": 0.40,
    },
    "level28": {
        "z_trigger": 0.80,
        "max_alerts": 15,
        "min_strength": 0.45,
        "min_causal_score": 0.35,
        "max_q_value": 0.15,
        "trigger_score_min": 0.62,
        "w_surprise": 0.40,
        "w_causal": 0.30,
        "w_stability": 0.20,
        "w_fdr": 0.10,
        "lookback_days": 90,
        "hard_weekday_match": True,
        "min_baseline_n": 10,
        "use_robust_baseline": True,
    },

    "level29": {
        "default_window_days": 3,
        "default_cost": 1.0,
        "default_dose": "",
        "default_notes": "",
        "lookback_days": 120,
        "lookback_rows": 120,
        "hard_weekday_match": False,
        "min_baseline_n": 8,
        "z_clip": 6.0,
        "z_success_thresh": 0.20,
        "use_robust_baseline": True,
        "auto_dedupe_log": True
    },
    "level32": {
        "enable_metrics": True,
        "enable_matches_table": True,
        "enable_placebo": True,
        "negative_control_enable": True,
        "negative_control_outcome_col": "negative_control_outcome",
        "negative_control_max_success_lb": 0.55,
        "enable_propensity": True,
        "propensity_max_diff": 0.20,
        "propensity_action_col": "action_active",
        "enable_pretrend_check": True,
        "pretrend_days": 7,
        "pretrend_max_diff": 0.30,
    }
}


def _merge_defaults(defaults, overrides):
    merged = dict(defaults)
    overrides = overrides or {}
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = _merge_defaults(merged[k], v)
        else:
            merged[k] = v
    return merged


def _normalize_aliases(cfg):
    cfg = dict(cfg or {})
    # Canonical data-outcome name is outcome_col.
    # target_col/target are retained only as backward-compatible aliases.
    if "outcome_col" not in cfg:
        if "target_col" in cfg:
            cfg["outcome_col"] = cfg["target_col"]
        elif "target" in cfg:
            cfg["outcome_col"] = cfg["target"]
    if "target_col" not in cfg and "outcome_col" in cfg:
        cfg["target_col"] = cfg["outcome_col"]
    if "target" in cfg and "target_col" not in cfg:
        cfg["target_col"] = cfg["target"]

    data_schema = dict(cfg.get("data_schema", {}) or {})
    if "outcome_col" not in data_schema:
        if "target_col" in data_schema:
            data_schema["outcome_col"] = data_schema["target_col"]
        elif cfg.get("outcome_col"):
            data_schema["outcome_col"] = cfg.get("outcome_col")
    if "target_col" not in data_schema and data_schema.get("outcome_col"):
        data_schema["target_col"] = data_schema["outcome_col"]
    if data_schema:
        cfg["data_schema"] = data_schema

    lv25 = dict(cfg.get("level25", {}) or {})
    aliases_25 = {
        "negative_control_outcome_col": "negctrl_outcome_col",
        "negative_control_enable": "negctrl_enable",
        "placebo_perm_B": "placebo_perm_b",
        "min_support_n": "min_obs",
        "min_strength": "min_causal_score",
        # legacy Level 2.5 names -> current PCMCI discovery names
        "min_causal_score": "keep_min_selection_score",
        "min_rss_reduction": "causal_min_incremental_r2",
        "stability_min_windows": "min_stability_windows",
    }
    for src, dst in list(aliases_25.items()):
        if src in lv25 and dst not in lv25:
            lv25[dst] = lv25[src]
        if dst in lv25 and src not in lv25:
            lv25[src] = lv25[dst]
    # Evidence threshold tracks selection unless explicitly configured. This keeps
    # old configs useful for the new discovery scorer without hiding legacy keys.
    if "keep_min_evidence_score" not in lv25 and "keep_min_selection_score" in lv25:
        try:
            lv25["keep_min_evidence_score"] = max(0.0, float(lv25["keep_min_selection_score"]) - 0.01)
        except (TypeError, ValueError):
            lv25["keep_min_evidence_score"] = lv25["keep_min_selection_score"]
    cfg["level25"] = lv25

    lv32 = dict(cfg.get("level32", {}) or {})
    if "negctrl_outcome_col" in lv32 and "negative_control_outcome_col" not in lv32:
        lv32["negative_control_outcome_col"] = lv32["negctrl_outcome_col"]
    cfg["level32"] = lv32
    return cfg


def load_config(path="pcb.json"):
    cfg = _merge_defaults(DEFAULT_CONFIG, {})
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            user = json.load(f) or {}
        cfg = _merge_defaults(cfg, _normalize_aliases(user))
    cfg = _normalize_aliases(cfg)
    return cfg
