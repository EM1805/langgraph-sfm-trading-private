
from runtime_env import configure_scientific_runtime
configure_scientific_runtime()

import os
import sys
import json
import argparse
import math
import warnings
import numpy as np
from runtime_compat import assert_scientific_stack
assert_scientific_stack()
import pandas as pd

from . import _utils as U

# Offline prior graph support was removed from the public input contract.
OfflinePriorGraph = None  # type: ignore

try:
    from scm_parts.adjustment import recommend_adjustments  # type: ignore
except ImportError:
    recommend_adjustments = None  # type: ignore

OUT_DIR = "out"

_CONFIG_LOAD_ERROR = None

def get_config():
    """Load Estimation config explicitly and warn on fallback.

    This preserves current constants for compatibility, but removes the previous
    silent failure mode where common.py initialized with an empty config without
    any user-visible diagnostic.
    """
    global _CONFIG_LOAD_ERROR
    try:
        from config import load_config
        cfg = load_config()
        return cfg if isinstance(cfg, dict) else {}
    except (OSError, ValueError, TypeError) as exc:
        _CONFIG_LOAD_ERROR = exc
        warnings.warn(
            "Estimation config could not be loaded; using defaults. error=%s" % exc,
            RuntimeWarning,
            stacklevel=2,
        )
        return {}

_CFG = get_config()

OUT_DIR = str(_CFG.get("out_dir", OUT_DIR))
INSIGHTS_L2 = os.path.join(OUT_DIR, "insights_level2.csv")
EXP_RESULTS = os.path.join(OUT_DIR, "experiment_results.csv")
EXP_SUMMARY_L29 = os.path.join(OUT_DIR, "experiment_summary_level29.csv")
OUT_L3 = os.path.join(OUT_DIR, "insights_level3.csv")
OUT_LEDGER = os.path.join(OUT_DIR, "insights_level3_ledger.csv")
OUT_TRIALS = os.path.join(OUT_DIR, "experiment_trials_enriched_level32.csv")

DEFAULT_DATA_CSV = "data.csv"
FALLBACK_DATA_CSV = os.path.join(OUT_DIR, "demo_data.csv")
OUTCOME_COL = str(_CFG.get("outcome_col", _CFG.get("target_col", "harm_event")))
TARGET_COL = OUTCOME_COL  # legacy alias for older estimation modules
DATE_COL = str(_CFG.get("date_col", "date"))

_L32 = _CFG.get("level32", {}) if isinstance(_CFG, dict) else {}
NEGCTRL_ENABLE = bool(_L32.get("negative_control_enable", True))
NEGCTRL_OUTCOME_COL = str(_L32.get("negative_control_outcome_col", "negative_control_outcome"))
NEGCTRL_MAX_SUCCESS_LB = float(_L32.get("negative_control_max_success_lb", 0.55))
ENABLE_PROPENSITY = bool(_L32.get("enable_propensity", True))
PROPENSITY_MAX_DIFF = float(_L32.get("propensity_max_diff", 0.20))
PROPENSITY_ACTION_COL = str(_L32.get("propensity_action_col", "action_active"))
ENABLE_PRETREND_CHECK = bool(_L32.get("enable_pretrend_check", True))
PRETREND_DAYS = int(_L32.get("pretrend_days", 7))
PRETREND_MAX_DIFF = float(_L32.get("pretrend_max_diff", 0.30))
LOOKBACK_DAYS = int(_L32.get("lookback_days", 120))
LOOKBACK_ROWS = int(_L32.get("lookback_rows", 120))
K_CONTROLS = int(_L32.get("k_controls", 12))
MIN_MATCHED = int(_L32.get("min_matched", 5))
MATCH_DIST_MAX = float(_L32.get("match_dist_max", 1.5))
Z_SUCCESS = float(_L32.get("z_success", 0.20))
Z_CLIP = float(_L32.get("z_clip", 6.0))
MIN_TRIALS_CANDIDATE = int(_L32.get("min_trials_candidate", 1))
MIN_TRIALS_VALIDATED = int(_L32.get("min_trials_validated", 2))
MIN_TRIALS_CONFIRMED = int(_L32.get("min_trials_confirmed", 3))
MIN_SUCCESS_LB_VALIDATED = float(_L32.get("min_success_lb_validated", 0.55))
MIN_SUCCESS_LB_CONFIRMED = float(_L32.get("min_success_lb_confirmed", 0.67))
MIN_CONFIDENCE_VALIDATED = float(_L32.get("min_confidence_validated", 0.55))
MIN_CONFIDENCE_CONFIRMED = float(_L32.get("min_confidence_confirmed", 0.72))
BOOT_B = int(_L32.get("bootstrap_b", 200))
BOOTSTRAP_SEED = int(_L32.get("bootstrap_seed", 123))
_Z90 = 1.2815515655446004

OVERLAP_MIN_MARGIN = float(_L32.get("overlap_min_margin", 0.02))
ROBUSTNESS_MIN_RATIO = float(_L32.get("robustness_min_ratio", 0.25))
ROSENBAUM_GAMMA_GRID = [1.0, 1.10, 1.25, 1.50, 1.75, 2.0, 2.5, 3.0]
RIDGE_L2 = float(_L32.get("ridge_l2", 1.0))
def _cfg_get_nested_path(*keys, default=""):
    """Return the first configured path found across top-level and nested sections."""
    if not isinstance(_CFG, dict):
        return str(default)
    for key in keys:
        if isinstance(key, tuple) and len(key) == 2:
            section, subkey = key
            block = _CFG.get(section, {})
            if isinstance(block, dict) and block.get(subkey):
                return str(block.get(subkey))
        elif _CFG.get(key):
            return str(_CFG.get(key))
    return str(default)

OFFLINE_PRIOR_GRAPH_PATH = _cfg_get_nested_path(
    default="",
)
DAG_PATH = OFFLINE_PRIOR_GRAPH_PATH  # internal/backward-compatible alias
DAG_RISK_CONF_PENALTY = float(_L32.get("dag_risk_conf_penalty", 0.12))
DAG_RISK_FORCE_WARNING = bool(_L32.get("dag_risk_force_warning", True))

def resolve_outcome_col(df, requested=None):
    """Resolve the outcome column robustly for test data and alternate schemas.

    Priority:
    1) explicit requested column when present
    2) configured OUTCOME_COL when present
    3) common fallback aliases used by tests and legacy datasets
    4) first numeric non-date column as a last resort
    """
    cols = list(getattr(df, "columns", []))
    if not cols:
        return str(requested or OUTCOME_COL)

    candidates = []
    if requested:
        candidates.append(str(requested))
    candidates.extend([str(OUTCOME_COL), "harm_event", "outcome", "y", "target"])
    seen = set()
    for c in candidates:
        if c and c not in seen:
            seen.add(c)
            if c in cols:
                return c

    for c in cols:
        if c == DATE_COL:
            continue
        try:
            series = pd.to_numeric(df[c], errors="coerce")
            if series.notna().any():
                return c
        except (TypeError, ValueError, OverflowError):
            continue
    return str(requested or OUTCOME_COL)


def _warn_once(key, message):
    seen = getattr(_warn_once, "_seen", set())
    if key in seen:
        return
    seen.add(key)
    setattr(_warn_once, "_seen", seen)
    try:
        print(message, file=sys.stderr)
    except OSError:
        pass

def _candidate_dag_paths():
    """Offline prior graph input was removed from the public input contract."""
    return []

def _load_dag():
    return None

def _apply_dag_covariates(dag, action_source, target_col, covs, insight_meta=None):
    if dag is None or not action_source:
        return _merge_adjustment_hints(action_source, target_col, list(covs), insight_meta, [], set(), "", "unknown", "")
    if recommend_adjustments is not None:
        rec = recommend_adjustments(
            dag=dag,
            action=action_source,
            target=target_col,
            candidate_covariates=covs,
        )
        preferred = list(rec.adjust_for)
        forbidden = set(rec.avoid)
        source = str(rec.source or "")
        confidence = str(rec.confidence or "unknown")
        notes = "|".join(rec.notes or [])
    else:
        ann = dag.l32_annotation(action_source, target_col)
        preferred = [c for c in _as_str(ann.get("dag_adjustment_set", "")).split("|") if c]
        forbidden = set(c for c in _as_str(ann.get("dag_forbidden_adjustments", "")).split("|") if c)
        source = "dag_annotation_fallback"
        confidence = _as_str(ann.get("dag_adjustment_confidence", "unknown")) or "unknown"
        notes = ""
    return _merge_adjustment_hints(action_source, target_col, list(covs), insight_meta, preferred, forbidden, source, confidence, notes)

def _select_negative_control_col(dag, action_source, df):
    cols = []
    if dag is not None and action_source:
        ann = dag.l32_annotation(action_source, resolve_outcome_col(df, TARGET_COL))
        cols.extend([c for c in _as_str(ann.get("dag_negative_controls", "")).split("|") if c])
    cols.append(NEGCTRL_OUTCOME_COL)
    seen = set()
    for c in cols:
        if c in seen:
            continue
        seen.add(c)
        if c and c in getattr(df, "columns", []):
            return c
    return ""

def _ensure_out():
    os.makedirs(OUT_DIR, exist_ok=True)

def _safe_float(x, default=np.nan):
    return U.safe_float(x, default)

def _as_str(x):
    return U.as_str(x)

def _split_pipe(value):
    return U.parse_list(value)

def _merge_adjustment_hints(action_source, target_col, covs, insight_meta, dag_preferred, dag_forbidden, dag_source, dag_confidence, dag_notes):
    insight_meta = insight_meta or {}
    bridge_version = _as_str(insight_meta.get("bridge_version", "")).strip()
    discovery_pref = []
    discovery_forbidden = []

    # Canonical mode: once discovery_bridge.py has normalized PCMCI output,
    # Estimation must consume only the bridge fields. This removes the old
    # duplicate reinterpretation of raw Discovery aliases in multiple places.
    if bridge_version:
        discovery_pref.extend(_split_pipe(insight_meta.get("candidate_covariates", "")))
        discovery_forbidden.extend(_split_pipe(insight_meta.get("post_treatment_columns", "")))
        discovery_forbidden.extend(_split_pipe(insight_meta.get("forbidden_adjustment_set", "")))
    else:
        for key in ("candidate_covariates", "suggested_adjustment_set", "candidate_adjustment_set", "graph_adjustment_hint", "confounder_hint"):
            discovery_pref.extend(_split_pipe(insight_meta.get(key, "")))
        for key in ("forbidden_adjustment_set", "forbidden_variables", "post_treatment_columns", "graph_forbidden_adjustment_hint", "forbidden_adjustment_hint"):
            discovery_forbidden.extend(_split_pipe(insight_meta.get(key, "")))

    forbidden = set([c for c in list(dag_forbidden) + discovery_forbidden if c])
    preferred = []
    for c in list(dag_preferred) + discovery_pref:
        if not c or c in forbidden or c in {action_source, target_col}:
            continue
        if c not in preferred:
            preferred.append(c)

    clean_covs = [c for c in covs if c not in forbidden and c not in {action_source, target_col}]
    violation = int(any(c in forbidden for c in covs))

    core_covs = []
    for c in ["outcome_prev", "outcome_lag_2", "outcome_lag_3", "target_prev", "target_lag_2", "target_lag_3", "time_idx", "dow_sin", "dow_cos", "weekday"]:
        if c in clean_covs and c not in core_covs:
            core_covs.append(c)

    if preferred:
        covs2 = []
        for c in core_covs + preferred:
            if c in clean_covs and c not in covs2:
                covs2.append(c)
        extras = []
        for c in clean_covs:
            if c not in covs2:
                extras.append(c)
            if len(extras) >= 2:
                break
        covs2.extend(extras)
        prioritized = True
    else:
        covs2 = list(clean_covs)
        prioritized = False

    for c in preferred:
        if c not in covs2 and c not in forbidden and c not in {action_source, target_col}:
            covs2.append(c)

    conf_rank = {"unknown": 0, "low": 1, "medium": 2, "high": 3}
    discovery_conf = _as_str(insight_meta.get("adjustment_set_confidence", "")) or ("medium" if discovery_pref else "unknown")
    eff_conf = dag_confidence if conf_rank.get(dag_confidence, 0) >= conf_rank.get(discovery_conf, 0) else discovery_conf

    source_bits = []
    if dag_preferred or dag_forbidden:
        source_bits.append(dag_source or "dag")
    if discovery_pref or discovery_forbidden:
        source_bits.append("discovery_bridge" if bridge_version else "discovery_hints")
    eff_source = "+".join([b for b in source_bits if b]) or (dag_source or ("discovery_bridge" if bridge_version else ("discovery_hints" if (discovery_pref or discovery_forbidden) else "")))

    notes = [n for n in _split_pipe(dag_notes) if n]
    if discovery_pref:
        notes.append("DISCOVERY_BRIDGE_COVARIATES" if bridge_version else "DISCOVERY_ADJUSTMENT_HINT")
    if bridge_version or _as_str(insight_meta.get("bridge_reason_codes", "")):
        notes.append("DISCOVERY_BRIDGE_CANONICAL")
    if discovery_forbidden:
        notes.append("DISCOVERY_BRIDGE_FORBIDDEN" if bridge_version else "DISCOVERY_FORBIDDEN_HINT")
    if dag_preferred and discovery_pref:
        notes.append("FUSED_DAG_AND_DISCOVERY")
    elif discovery_pref and not dag_preferred:
        notes.append("DISCOVERY_PRIMARY_ADJUSTMENT")
    if prioritized:
        notes.append("PRIORITIZED_ADJUSTMENT_SET")

    return covs2, "|".join(preferred), "|".join(sorted(forbidden)), violation, eff_source, eff_conf, "|".join(dict.fromkeys([n for n in notes if n]))

def _load_data_path(path=None):
    if path and os.path.exists(path):
        return path
    if os.path.exists(DEFAULT_DATA_CSV):
        return DEFAULT_DATA_CSV
    dc = os.path.join(OUT_DIR, "data_clean.csv")
    if os.path.exists(dc):
        return dc
    return FALLBACK_DATA_CSV

def _try_parse_date(df):
    if DATE_COL in df.columns:
        d = pd.to_datetime(df[DATE_COL], errors="coerce")
        if d.notna().mean() > 0.2:
            df = df.copy()
            df[DATE_COL] = d
    return df

def _has_date(df):
    return DATE_COL in df.columns and pd.api.types.is_datetime64_any_dtype(df[DATE_COL])

def _past_indices(df, t_idx):
    if t_idx <= 0:
        return np.array([], dtype=int)
    if _has_date(df):
        d = df[DATE_COL].iloc[t_idx]
        if pd.notna(d):
            start = d.normalize() - pd.Timedelta(days=int(LOOKBACK_DAYS))
            mask = (df[DATE_COL] < d) & (df[DATE_COL] >= start)
            return df.index[mask].to_numpy(dtype=int)
    start = max(0, int(t_idx) - int(LOOKBACK_ROWS))
    return np.arange(start, int(t_idx), dtype=int)

def _robust_center_scale(x):
    x = np.asarray(x, dtype=float)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return np.nan, 1.0
    med = float(np.median(finite))
    mad = float(np.median(np.abs(finite - med)))
    scale = 1.4826 * mad if np.isfinite(mad) and mad > 1e-9 else float(np.std(finite))
    if not np.isfinite(scale) or scale < 1e-9:
        scale = 1.0
    return med, scale

def _robust_z(v, sample):
    med, scale = _robust_center_scale(sample)
    return float(np.clip((float(v) - med) / scale, -Z_CLIP, Z_CLIP))

def _sr_lower_bound(wins, n, z=_Z90):
    wins = float(wins)
    n = int(n)
    if n <= 0:
        return 0.0
    phat = wins / float(n)
    z2 = float(z) ** 2
    denom = 1.0 + z2 / float(n)
    center = phat + z2 / (2.0 * float(n))
    margin = float(z) * math.sqrt((phat * (1.0 - phat) + z2 / (4.0 * float(n))) / float(n))
    lb = (center - margin) / denom
    return float(max(0.0, min(1.0, lb)))

def _numeric_cols(df):
    cols = []
    for c in df.columns:
        if c == DATE_COL:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() >= max(10, int(0.3 * len(df))):
            cols.append(c)
    return cols

def _ensure_calendar_covs(df):
    out = df.copy()
    out["time_idx"] = np.arange(len(out), dtype=float)
    if _has_date(out):
        dow = out[DATE_COL].dt.dayofweek.astype(float)
        out["dow_sin"] = np.sin(2.0 * np.pi * dow / 7.0)
        out["dow_cos"] = np.cos(2.0 * np.pi * dow / 7.0)
    return out

DISCOVERY_BRIDGE_CSV = os.path.join(OUT_DIR, "discovery_estimation_bridge.csv")
DISCOVERY_BRIDGE_MANIFEST = os.path.join(OUT_DIR, "discovery_estimation_manifest.json")
