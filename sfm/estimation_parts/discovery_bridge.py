"""Discovery -> Estimation bridge.

This module is the canonical translator between PCMCI/Discovery outputs and
Estimation inputs. Discovery is allowed to emit many legacy/proposal columns;
Estimation should consume the normalized fields emitted here.
"""


from runtime_env import configure_scientific_runtime
configure_scientific_runtime()

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from runtime_compat import assert_scientific_stack
assert_scientific_stack()
import pandas as pd

from . import common as C

OUT_DIR = C.OUT_DIR
BRIDGE_CSV = os.path.join(OUT_DIR, "discovery_estimation_bridge.csv")
BRIDGE_MANIFEST = os.path.join(OUT_DIR, "discovery_estimation_manifest.json")
BRIDGE_VERSION = 5

# Ordered aliases: first non-empty value wins unless a field is intentionally merged.
TREATMENT_ALIASES = ("treatment_col", "source", "action_source", "action_name")
OUTCOME_ALIASES = ("outcome_col", "target_col", "target")
ADJUSTMENT_ALIASES = (
    "candidate_covariates",
    "suggested_adjustment_set",
    "candidate_adjustment_set",
    "graph_adjustment_hint",
    "confounder_hint",
    "supporting_features",
    "local_dag_confounders",
)
FORBIDDEN_ALIASES = (
    "post_treatment_columns",
    "forbidden_adjustment_set",
    "forbidden_variables",
    "forbidden_adjustment_hint",
    "graph_forbidden_adjustment_hint",
)
NEGATIVE_CONTROL_ALIASES = (
    "negative_control_col",
    "suggested_negative_control",
    "negative_control_hint",
    "graph_negative_controls",
    "dag_negative_controls",
)

DISCOVERY_EFFECT_PROXY_ALIASES = (
    "discovery_effect_proxy",
    "discovery_adjusted_delta",
    "discovery_signal_strength",
    # Legacy Discovery-only aliases. These are not Pearl-authorized estimates.
    "causal_effect",
    "adjusted_delta_test",
    "adjusted_delta",
)

DISCOVERY_SCORE_HANDOFF_FIELDS = (
    "hypothesis_signal_score",
    "hypothesis_signal_grade",
    "hypothesis_signal_reason_codes",
    "safety_risk_score",
    "safety_risk_grade",
    "safety_risk_reason_codes",
    "safety_blocking",
    "signal_safety_cell",
    "signal_safety_policy",
    "signal_safety_matrix_track",
    "signal_safety_blocking",
    "signal_safety_reason_code",
    "signal_safety_matrix_version",
)



def _file_sha256(path: str) -> str:
    """Return a stable hash for bridge freshness checks."""
    if not path or not os.path.exists(path):
        return ""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
    except OSError:
        return ""
    return h.hexdigest()


def _read_manifest(out_dir: str = OUT_DIR) -> Dict[str, object]:
    path = os.path.join(out_dir, "discovery_estimation_manifest.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {}


def _source_metadata(source_insights_path: str = "", source_df: Optional[pd.DataFrame] = None) -> Dict[str, object]:
    return {
        "source_insights_csv": source_insights_path or "",
        "source_insights_hash": _file_sha256(source_insights_path) if source_insights_path else "",
        "source_insights_mtime": os.path.getmtime(source_insights_path) if source_insights_path and os.path.exists(source_insights_path) else None,
        "source_insights_n_rows": int(len(source_df)) if source_df is not None else None,
    }


def bridge_is_fresh(out_dir: str = OUT_DIR, source_insights_path: str = "", source_df: Optional[pd.DataFrame] = None) -> bool:
    """True when the saved bridge matches the current insights artifact."""
    manifest = _read_manifest(out_dir)
    if not manifest:
        return False
    try:
        if int(manifest.get("bridge_version", -1)) != int(BRIDGE_VERSION):
            return False
    except (TypeError, ValueError):
        return False
    if source_insights_path:
        current_hash = _file_sha256(source_insights_path)
        recorded_hash = str(manifest.get("source_insights_hash", "") or "")
        if current_hash and recorded_hash and current_hash != recorded_hash:
            return False
        if current_hash and not recorded_hash:
            return False
    if source_df is not None and manifest.get("source_insights_n_rows") is not None:
        try:
            if int(manifest.get("source_insights_n_rows")) != int(len(source_df)):
                return False
        except (TypeError, ValueError):
            return False
    return True


def select_first_valid_column(value, df: Optional[pd.DataFrame]) -> str:
    """Pick the first pipe/JSON/list item that is an actual dataframe column."""
    if df is None:
        return ""
    cols = set(map(str, getattr(df, "columns", [])))
    for item in _split_pipe(value):
        item = _as_str(item)
        if item in cols:
            return item
    return ""


def _filter_existing_columns(value, df: Optional[pd.DataFrame]) -> Tuple[List[str], List[str]]:
    if df is None:
        return _dedupe(_split_pipe(value)), []
    cols = set(map(str, getattr(df, "columns", [])))
    valid, dropped = [], []
    for item in _split_pipe(value):
        item = _as_str(item)
        if not item:
            continue
        if item in cols:
            valid.append(item)
        else:
            dropped.append(item)
    return _dedupe(valid), _dedupe(dropped)


def sanitize_bridge_for_data(bridge: pd.DataFrame, df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Validate canonical bridge columns against the dataset consumed by Estimation."""
    if bridge is None or len(bridge) == 0 or df is None:
        return bridge if bridge is not None else pd.DataFrame()
    out = bridge.copy()
    if "bridge_sanitization_codes" not in out.columns:
        out["bridge_sanitization_codes"] = ""
    if "bridge_dropped_columns" not in out.columns:
        out["bridge_dropped_columns"] = ""
    for idx, row in out.iterrows():
        codes: List[str] = _split_pipe(row.get("bridge_sanitization_codes", ""))
        dropped_all: List[str] = _split_pipe(row.get("bridge_dropped_columns", ""))

        covs, dropped = _filter_existing_columns(row.get("candidate_covariates", ""), df)
        if dropped:
            codes.append("COVARIATE_DROPPED_NOT_IN_DATA")
            dropped_all.extend(dropped)
        cov_str = "|".join(covs)
        out.at[idx, "candidate_covariates"] = cov_str
        out.at[idx, "candidate_adjustment_set"] = cov_str
        out.at[idx, "suggested_adjustment_set"] = cov_str

        forbidden, dropped = _filter_existing_columns(row.get("post_treatment_columns", row.get("forbidden_adjustment_set", "")), df)
        if dropped:
            codes.append("FORBIDDEN_COLUMN_DROPPED_NOT_IN_DATA")
            dropped_all.extend(dropped)
        forbidden_str = "|".join(forbidden)
        out.at[idx, "post_treatment_columns"] = forbidden_str
        out.at[idx, "forbidden_adjustment_set"] = forbidden_str

        raw_neg = row.get("negative_control_col", "") or row.get("suggested_negative_control", "") or row.get("negative_controls", "")
        neg = select_first_valid_column(raw_neg, df)
        raw_negs = _split_pipe(raw_neg)
        if raw_negs and not neg:
            codes.append("NEGCTRL_COLUMN_MISSING")
            dropped_all.extend(raw_negs)
        elif raw_negs and neg != raw_negs[0]:
            codes.append("NEGCTRL_FIRST_INVALID_USED_NEXT_VALID")
        out.at[idx, "negative_control_col"] = neg
        valid_negs, dropped = _filter_existing_columns(row.get("negative_controls", row.get("suggested_negative_control", raw_neg)), df)
        if dropped:
            dropped_all.extend(dropped)
        negs_str = "|".join(valid_negs)
        out.at[idx, "negative_controls"] = negs_str
        out.at[idx, "suggested_negative_control"] = negs_str

        out.at[idx, "bridge_sanitization_codes"] = "|".join(_dedupe(codes))
        out.at[idx, "bridge_dropped_columns"] = "|".join(_dedupe(dropped_all))
    return out


def _as_str(x) -> str:
    return C._as_str(x).strip()


def _split_pipe(value) -> List[str]:
    """Parse bridge lists from pipe strings, simple JSON lists, or Python lists."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        raw_items = list(value)
    else:
        raw = _as_str(value)
        if not raw or raw.lower() in {"nan", "none", "null"}:
            return []
        raw_items = None
        if raw.startswith("[") and raw.endswith("]"):
            try:
                decoded = json.loads(raw)
                if isinstance(decoded, list):
                    raw_items = decoded
            except (TypeError, ValueError, json.JSONDecodeError):
                raw_items = None
        if raw_items is None:
            # Keep comma fallback conservative; pipe remains the canonical separator.
            sep = "|" if "|" in raw else ","
            raw_items = raw.split(sep)
    return [_as_str(part) for part in raw_items if _as_str(part)]


def _dedupe(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        item = _as_str(item)
        if not item or item in seen or item.lower() in {"nan", "none", "null"}:
            continue
        seen.add(item)
        out.append(item)
    return out


def _first_present(row: pd.Series, aliases: Iterable[str], default: str = "") -> str:
    for key in aliases:
        value = _as_str(row.get(key, ""))
        if value:
            return value
    return default




def _handoff_value(row: pd.Series, field: str):
    """Copy namespaced Discovery score/safety fields without deriving them from legacy scores."""
    return row.get(field, "") if field in row.index else ""

def _merge_fields(row: pd.Series, aliases: Iterable[str]) -> List[str]:
    values: List[str] = []
    for key in aliases:
        values.extend(_split_pipe(row.get(key, "")))
    return _dedupe(values)


def _present_columns(row: pd.Series, values: Iterable[str]) -> List[str]:
    """Return valid symbolic columns; drop sentence-like hints and self references later."""
    cols = []
    for value in values:
        value = _as_str(value)
        if not value:
            continue
        # Avoid treating free-text rationale as a variable name.
        if len(value) > 80 or " " in value and not any(ch in value for ch in ["_", ":", "/"]):
            continue
        cols.append(value)
    return _dedupe(cols)


def _infer_treatment_kind(row: pd.Series) -> str:
    source_role = _as_str(row.get("source_role", ""))
    edge_family = _as_str(row.get("edge_family", ""))
    action_type = _as_str(row.get("action_type", ""))
    treatment_role = _as_str(row.get("treatment_role", ""))
    treatment = _first_present(row, TREATMENT_ALIASES)
    if treatment_role in {"decision", "guardrail", "context"}:
        return treatment_role
    if source_role in {"decision", "guardrail", "context"}:
        return source_role
    blob = "|".join([edge_family, action_type, treatment]).lower()
    if any(k in blob for k in ["veto", "guardrail", "review", "approval"]):
        return "guardrail"
    if any(k in blob for k in ["action", "intervention", "treatment", "decision"]):
        return "decision"
    return "context"


def _infer_estimand(row: pd.Series, outcome_col: str) -> str:
    explicit = _as_str(row.get("preferred_estimand", ""))
    if explicit:
        return explicit
    outcome_kind = _as_str(row.get("outcome_kind", "")).lower()
    y = (outcome_col or "").lower()
    if outcome_kind == "binary" or y in {"harm_event", "rollback_needed"} or any(k in y for k in ["harm", "rollback", "incident"]):
        return "risk_difference_att"
    if outcome_kind == "count":
        return "count_effect_att"
    if outcome_kind == "continuous_time":
        return "mean_difference_att"
    return "effect_att"


def _infer_post_treatment(row: pd.Series, treatment_col: str, outcome_col: str) -> List[str]:
    forbidden = _present_columns(row, _merge_fields(row, FORBIDDEN_ALIASES))
    return [c for c in forbidden if c not in {treatment_col, outcome_col}]


def _infer_covariates(row: pd.Series, treatment_col: str, outcome_col: str, forbidden: List[str]) -> List[str]:
    preferred = _present_columns(row, _merge_fields(row, ADJUSTMENT_ALIASES))
    blocked = set(forbidden) | {treatment_col, outcome_col}
    return [c for c in preferred if c not in blocked]


def _infer_negative_controls(row: pd.Series, treatment_col: str, outcome_col: str, forbidden: List[str]) -> List[str]:
    controls = _present_columns(row, _merge_fields(row, NEGATIVE_CONTROL_ALIASES))
    blocked = set(forbidden) | {treatment_col, outcome_col}
    return [c for c in controls if c not in blocked]


def _discovery_confidence_tier(row: pd.Series) -> str:
    explicit = _as_str(row.get("discovery_confidence_tier", ""))
    if explicit:
        return explicit
    sel = C._safe_float(row.get("selection_score", np.nan), 0.0)
    evid = C._safe_float(row.get("discovery_evidence_score", np.nan), 0.0)
    red = int(C._safe_float(row.get("discovery_hard_red_flags", np.nan), 0.0))
    plaus = C._safe_float(row.get("causal_plausibility_score", np.nan), 0.0)
    orient = C._safe_float(row.get("orientation_score", np.nan), 0.0)
    cons = C._safe_float(row.get("temporal_consensus_score", np.nan), 0.0)
    negc = C._safe_float(row.get("negative_control_score", np.nan), 0.0)
    q_value = C._safe_float(row.get("mci_q_value", row.get("q_value", np.nan)), np.nan)
    if red == 0 and sel >= 0.72 and evid >= 0.68 and (not np.isfinite(q_value) or q_value <= 0.10):
        return "high"
    if red <= 1 and sel >= 0.58 and evid >= 0.52 and (not np.isfinite(q_value) or q_value <= 0.20):
        return "medium"
    if plaus >= 0.78 and orient >= 0.62 and cons >= 0.62 and negc >= 0.55:
        return "high"
    if plaus >= 0.62 and cons >= 0.50:
        return "medium"
    return "low"


def _validation_design(row: pd.Series, tier: str, negative_controls: List[str]) -> str:
    explicit = _as_str(row.get("validation_design", row.get("trial_design_hint", "")))
    if explicit:
        return explicit
    if negative_controls and tier in {"high", "medium"}:
        return "matched_counterfactual_with_negative_control"
    if tier in {"high", "medium"}:
        return "matched_counterfactual"
    return "diagnostic_only"


def _reason_codes(row: pd.Series, covs: List[str], forbidden: List[str], neg_controls: List[str], tier: str) -> List[str]:
    reasons = []
    if covs:
        reasons.append("BRIDGE_ADJUSTMENT_SET_NORMALIZED")
    else:
        reasons.append("NO_DISCOVERY_ADJUSTMENT_SET")
    if forbidden:
        reasons.append("BRIDGE_FORBIDDEN_SET_NORMALIZED")
    if neg_controls:
        reasons.append("BRIDGE_NEGATIVE_CONTROL_NORMALIZED")
    if _as_str(row.get("mci_q_value", row.get("q_value", ""))):
        reasons.append("MCI_FDR_METADATA_AVAILABLE")
    if tier == "low":
        reasons.append("LOW_DISCOVERY_CONFIDENCE")
    return _dedupe(reasons)


def _empty_bridge() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "bridge_version", "insight_id", "treatment_col", "treatment_role", "treatment_kind",
        "outcome_col", "outcome_role", "preferred_estimand", "candidate_covariates",
        "candidate_adjustment_set", "suggested_adjustment_set", "post_treatment_columns",
        "forbidden_adjustment_set", "negative_control_col", "suggested_negative_control",
        "negative_controls", "source_role", "edge_family", "validation_design",
        "discovery_confidence_tier", "discovery_signal_score",
        "hypothesis_signal_score", "hypothesis_signal_grade", "hypothesis_signal_reason_codes",
        "safety_risk_score", "safety_risk_grade", "safety_risk_reason_codes", "safety_blocking",
        "signal_safety_cell", "signal_safety_policy", "signal_safety_matrix_track",
        "signal_safety_blocking", "signal_safety_reason_code", "signal_safety_matrix_version",
        "discovery_effect_proxy",
        "effect_proxy_semantics", "mci_q_value", "bridge_reason_codes", "bridge_source_columns",
        "bridge_sanitization_codes", "bridge_dropped_columns",
    ])


def build_bridge_from_insights(df_i: pd.DataFrame) -> pd.DataFrame:
    """Build the canonical handoff table consumed by estimation.

    The output intentionally mirrors canonical fields to legacy names so older code keeps
    running, but the canonical fields are: treatment_col, outcome_col,
    candidate_covariates, post_treatment_columns, negative_control_col,
    discovery_confidence_tier, validation_design, and bridge_reason_codes.
    """
    if df_i is None or len(df_i) == 0:
        return _empty_bridge()

    rows = []
    for _, row in df_i.iterrows():
        insight_id = _as_str(row.get("insight_id", ""))
        outcome_col = _first_present(row, OUTCOME_ALIASES)
        treatment_col = _first_present(row, TREATMENT_ALIASES)
        if not insight_id or not treatment_col or not outcome_col:
            continue
        treatment_role = _as_str(row.get("treatment_role", row.get("source_role", "")))
        outcome_role = _as_str(row.get("outcome_role", "outcome")) or "outcome"
        forbidden = _infer_post_treatment(row, treatment_col, outcome_col)
        covs = _infer_covariates(row, treatment_col, outcome_col, forbidden)
        negative_controls = _infer_negative_controls(row, treatment_col, outcome_col, forbidden)
        tier = _discovery_confidence_tier(row)
        q_value = C._safe_float(row.get("mci_q_value", row.get("q_value", np.nan)), np.nan)
        signal = C._safe_float(row.get("selection_score", row.get("discovery_evidence_score", row.get("strength", np.nan))), np.nan)
        effect_proxy = np.nan
        for proxy_col in DISCOVERY_EFFECT_PROXY_ALIASES:
            candidate = C._safe_float(row.get(proxy_col, np.nan), np.nan)
            if np.isfinite(candidate):
                effect_proxy = candidate
                break
        source_cols = [c for c in list(ADJUSTMENT_ALIASES + FORBIDDEN_ALIASES + NEGATIVE_CONTROL_ALIASES + DISCOVERY_EFFECT_PROXY_ALIASES + DISCOVERY_SCORE_HANDOFF_FIELDS) if c in row.index and _as_str(row.get(c, ""))]
        reasons = _reason_codes(row, covs, forbidden, negative_controls, tier)
        rows.append({
            "bridge_version": BRIDGE_VERSION,
            "insight_id": insight_id,
            "treatment_col": treatment_col,
            "treatment_role": treatment_role,
            "treatment_kind": _infer_treatment_kind(row),
            "outcome_col": outcome_col,
            "outcome_role": outcome_role,
            "preferred_estimand": _infer_estimand(row, outcome_col),
            "candidate_covariates": "|".join(covs),
            # Legacy aliases consumed by estimation/common.
            "candidate_adjustment_set": "|".join(covs),
            "suggested_adjustment_set": "|".join(covs),
            "post_treatment_columns": "|".join(forbidden),
            "forbidden_adjustment_set": "|".join(forbidden),
            "negative_control_col": negative_controls[0] if negative_controls else "",
            "suggested_negative_control": "|".join(negative_controls),
            "negative_controls": "|".join(negative_controls),
            "source_role": _as_str(row.get("source_role", "")),
            "edge_family": _as_str(row.get("edge_family", "")),
            "validation_design": _validation_design(row, tier, negative_controls),
            "discovery_confidence_tier": tier,
            "discovery_signal_score": signal if np.isfinite(signal) else np.nan,
            # Namespaced signal/safety fields are canonical for causal_confidence.
            # Do not infer these from legacy score columns here; pass through only.
            "hypothesis_signal_score": _handoff_value(row, "hypothesis_signal_score"),
            "hypothesis_signal_grade": _handoff_value(row, "hypothesis_signal_grade"),
            "hypothesis_signal_reason_codes": _handoff_value(row, "hypothesis_signal_reason_codes"),
            "safety_risk_score": _handoff_value(row, "safety_risk_score"),
            "safety_risk_grade": _handoff_value(row, "safety_risk_grade"),
            "safety_risk_reason_codes": _handoff_value(row, "safety_risk_reason_codes"),
            "safety_blocking": _handoff_value(row, "safety_blocking"),
            "signal_safety_cell": _handoff_value(row, "signal_safety_cell"),
            "signal_safety_policy": _handoff_value(row, "signal_safety_policy"),
            "signal_safety_matrix_track": _handoff_value(row, "signal_safety_matrix_track"),
            "signal_safety_blocking": _handoff_value(row, "signal_safety_blocking"),
            "signal_safety_reason_code": _handoff_value(row, "signal_safety_reason_code"),
            "signal_safety_matrix_version": _handoff_value(row, "signal_safety_matrix_version"),
            "discovery_effect_proxy": effect_proxy if np.isfinite(effect_proxy) else np.nan,
            "effect_proxy_semantics": "discovery_screening_proxy_not_causal_claim",
            "mci_q_value": q_value if np.isfinite(q_value) else np.nan,
            "bridge_reason_codes": "|".join(reasons),
            "bridge_source_columns": "|".join(source_cols),
            "bridge_sanitization_codes": "",
            "bridge_dropped_columns": "",
        })
    out = pd.DataFrame(rows)
    if len(out) == 0:
        return _empty_bridge()
    out = out.drop_duplicates(subset=["insight_id"], keep="first").reset_index(drop=True)
    return out


def write_bridge_from_insights(df_i: pd.DataFrame, out_dir: str = OUT_DIR, source_insights_path: str = "") -> Tuple[str, str, pd.DataFrame]:
    os.makedirs(out_dir, exist_ok=True)
    bridge = build_bridge_from_insights(df_i)
    bridge_csv = os.path.join(out_dir, "discovery_estimation_bridge.csv")
    manifest_json = os.path.join(out_dir, "discovery_estimation_manifest.json")
    bridge.to_csv(bridge_csv, index=False)
    source_meta = _source_metadata(source_insights_path, df_i)
    manifest: Dict[str, object] = {
        "bridge_version": BRIDGE_VERSION,
        "bridge_created_at": datetime.now(timezone.utc).isoformat(),
        **source_meta,
        "bridge_contract": (
            "Canonical PCMCI/Discovery-to-Estimation handoff. Estimation should consume "
            "candidate_covariates, post_treatment_columns, negative_control_col, "
            "discovery_confidence_tier, namespaced hypothesis_signal/safety_risk/signal_safety fields, validation_design, and bridge_reason_codes. "
            "Legacy adjustment/forbidden aliases are mirrored for compatibility. Discovery effect values are exported only as discovery_effect_proxy and are not causal claims."
        ),
        "bridge_csv": bridge_csv,
        "n_rows": int(len(bridge)),
        "columns": bridge.columns.tolist(),
        "canonical_fields": [
            "insight_id", "treatment_col", "outcome_col", "candidate_covariates",
            "post_treatment_columns", "negative_control_col", "discovery_confidence_tier",
            "hypothesis_signal_score", "hypothesis_signal_grade", "safety_risk_score",
            "safety_risk_grade", "signal_safety_matrix_track", "signal_safety_blocking",
            "discovery_effect_proxy", "effect_proxy_semantics", "validation_design", "bridge_reason_codes",
        ],
    }
    with open(manifest_json, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return bridge_csv, manifest_json, bridge


def load_bridge(out_dir: str = OUT_DIR, source_insights_path: str = "", source_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    path = os.path.join(out_dir, "discovery_estimation_bridge.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    if source_insights_path or source_df is not None:
        if not bridge_is_fresh(out_dir, source_insights_path=source_insights_path, source_df=source_df):
            return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError):
        return pd.DataFrame()
    if "insight_id" not in df.columns:
        return pd.DataFrame()
    return df.drop_duplicates(subset=["insight_id"], keep="first").reset_index(drop=True)
