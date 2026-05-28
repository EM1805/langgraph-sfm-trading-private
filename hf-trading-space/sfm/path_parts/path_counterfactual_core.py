from __future__ import annotations

import math
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from runtime.cf_types import ActionIntent, HistoricalEvent, PathTreatmentSpec
from runtime.action_registry_v2 import get_action_spec, load_action_registry
from runtime.action_semantics import derive_action_effects
from runtime.path_library import load_path_library
from runtime.runtime_context import extract_runtime_context
from runtime.path_evidence import _load_jsonl, _match_score
from runtime.causal_strata import build_causal_stratum, summarize_strata, treated_control_design, path_treatment_status, get_path_treatment_spec
from runtime.validation_plan_loader import merge_validation_plan_into_path
from runtime.identification_engine import build_identification_assessment, compute_identification_score, classify_identification, confounding_risk_from_components, sensitivity_ratio
from runtime.shared_utils import boolish


CATEGORICAL_KEYS: Sequence[str] = (
    "resource_sensitivity",
    "blast_radius",
    "recipient_scope",
    "environment",
    "action_name",
)
BOOLEAN_KEYS: Sequence[str] = (
    "approval_present",
    "rollback_available",
    "novel_action",
    "attachment_present",
)
SEVERITY_MAP = {"low": 0.2, "medium": 0.55, "high": 0.9}
RNG = random.Random(17)
_BASELINE_HAZARD_CACHE: Dict[Tuple[Any, ...], float] = {}
_MAX_BASELINE_CACHE = 20000

COUNTERFACTUAL_CONFIG: Dict[str, Any] = {
    "runtime_budget": {
        # Heavy matched counterfactual is opt-in for runtime use.
        # Default gateway path must be bounded and conservative: it should never
        # block a veto decision just because matching/support estimation is slow.
        "enable_heavy_runtime": False,
        "max_runtime_events": 160,
        "max_runtime_paths": 12,
        "bootstrap_draws": 6,
        "placebo_draws": 3,
        "max_placebo_outcomes": 2,
    },
    "nearest_neighbor": {
        "base_weight": 0.62,
        "propensity_weight": 0.24,
        "strata_weight": 0.14,
        "propensity_gap_scale": 0.25,
        "hard_floor": 0.50,
    },
    "logistic": {
        "steps": 250,
        "lr": 0.30,
        "l2": 0.01,
        "clip": 20.0,
        "p_min": 0.01,
        "p_max": 0.99,
    },
    "panel": {
        "pretrend_gap_max": 0.12,
        "min_support": 2,
    },
    "common_support": {
        "ci_low_q": 0.05,
        "ci_high_q": 0.95,
    },
    "placebo": {
        "default_threshold": 0.20,
        "fail_component_max": 0.34,
        "fail_rate_max": 0.28,
        "random_treatment_delta_fail": 0.20,
        "permuted_outcome_delta_fail": 0.20,
        "temporal_delta_fail": 0.12,
        "path_thresholds": {
            "external_data_leakage": 0.12,
            "destructive_mutation": 0.14,
            "operational_failure": 0.16,
        },
    },
    "pretrend": {
        "strong_mean_gap_max": 0.08,
        "strong_max_dev_max": 0.16,
        "weak_mean_gap_max": 0.12,
        "weak_max_dev_max": 0.22,
    },
}


def _cfg(*keys: str, default: Any = None) -> Any:
    cur: Any = COUNTERFACTUAL_CONFIG
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur




def _heavy_counterfactual_enabled() -> bool:
    """Return True only when the expensive matched CF engine is explicitly enabled.

    The veto gateway is safety-critical; an unavailable/slow estimator must degrade
    to an auditable "insufficient support" result, not hang the policy decision.
    Set AMANTIA_ENABLE_HEAVY_COUNTERFACTUAL=1 for offline audits.
    """
    env = str(os.environ.get("AMANTIA_ENABLE_HEAVY_COUNTERFACTUAL", "")).strip().lower()
    if env in {"1", "true", "yes", "on"}:
        return True
    return bool(_cfg("runtime_budget", "enable_heavy_runtime", default=False))


def _safe_event_count(event_log_path: str | Path) -> int:
    try:
        path = Path(event_log_path)
        if not path.exists():
            return 0
        with path.open("r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except OSError:
        return 0


def _counterfactual_fallback_path(path: Dict[str, Any], reason: str, event_count: int = 0) -> Dict[str, Any]:
    """Conservative bounded fallback used when runtime CF cannot be evaluated."""
    out = dict(path or {})
    out.update({
        "counterfactual_method": "runtime_bounded_fallback",
        "counterfactual_method_family": "bounded_conservative_fallback",
        "counterfactual_runtime_skipped": True,
        "counterfactual_skip_reason": str(reason),
        "counterfactual_event_log_support": int(event_count),
        "counterfactual_treated_support": 0,
        "counterfactual_control_support": 0,
        "counterfactual_treated_harm_rate": 0.0,
        "counterfactual_control_harm_rate": 0.0,
        "counterfactual_risk_delta": 0.0,
        "counterfactual_did_effect": 0.0,
        "counterfactual_dr_risk_delta": 0.0,
        "counterfactual_dr_did_effect": 0.0,
        "counterfactual_evidence_strength": "insufficient",
        "counterfactual_evidence_detail": "insufficient_runtime_support",
        "counterfactual_control_match_quality": 0.0,
        "counterfactual_overlap_ok": False,
        "counterfactual_balance_ok": False,
        "counterfactual_pretrend_ok": False,
        "counterfactual_placebo_pass": False,
        "counterfactual_valid_strata": 0,
        "counterfactual_strata_support": 0.0,
        "counterfactual_effective_support_n": 0,
        "counterfactual_dominant_control_weight": 1.0,
        "effect_estimate": {
            "risk_delta": 0.0,
            "did_effect": 0.0,
            "dr_risk_delta": 0.0,
            "dr_did_effect": 0.0,
            "bootstrap_ci": [0.0, 0.0],
            "method": "runtime_bounded_fallback",
        },
        "identification_assessment": {
            "identification_score": 0.0,
            "identification_label": "insufficient",
            "identification_class": "insufficient",
            "confounding_risk": "unknown",
            "reason": str(reason),
        },
    })
    return out


def _counterfactual_fallback_paths(paths: Sequence[Dict[str, Any]] | None, reason: str, event_count: int = 0) -> List[Dict[str, Any]]:
    return [_counterfactual_fallback_path(dict(p or {}), reason, event_count=event_count) for p in list(paths or [])]


PATH_HARM_PRIOR_NOTES: Dict[str, str] = {
    "production": "Baseline operational seriousness prior for production-scope actions; calibrated as a conservative ranking prior from internal scenario review, not a causal effect.",
    "external": "Extra prior for external recipient/share scope because externalization errors are harder to reverse.",
    "high_sensitivity": "Added when data/resource sensitivity is high; reflects expected harm escalation under exposure or destructive change.",
    "medium_sensitivity": "Smaller version of the sensitivity prior for mid-tier assets.",
    "high_blast": "Added when the blast radius is broad or high; represents broader propagation risk.",
    "no_approval": "Governance prior for actions lacking approval when approval would normally be expected.",
    "no_rollback": "Operational prior for mutations without rollback or recovery path.",
    "attachment": "Leakage-specific prior for attachment-bearing communication actions.",
    "novel": "Small novelty prior for actions with little historical support or unfamiliar operational profile.",
}

# Ranking priors for the current safety domain. These are intentionally modest and
# combine with empirical / counterfactual evidence later; they should not be read
# as probabilities or causal effect estimates.
PATH_HARM_PRIORS: Dict[str, Dict[str, float]] = {
    "harm_leakage": {"production": 0.18, "external": 0.28, "high_sensitivity": 0.22, "no_approval": 0.12, "attachment": 0.08},
    "harm_data_loss": {"production": 0.18, "high_sensitivity": 0.18, "high_blast": 0.24, "no_rollback": 0.22, "no_approval": 0.10},
    "harm_operational_failure": {"production": 0.16, "medium_sensitivity": 0.08, "high_blast": 0.18, "novel": 0.16, "no_rollback": 0.06},
    "harm_policy_bypass": {"production": 0.12, "high_sensitivity": 0.12, "no_approval": 0.24, "high_blast": 0.12},
    "harm_access_abuse": {"production": 0.12, "high_sensitivity": 0.16, "high_blast": 0.12, "no_approval": 0.16, "novel": 0.08},
}


def _event_to_intent(event: Dict[str, Any]) -> Dict[str, Any]:
    return {"action_name": event.get("action_name"), "environment": event.get("environment", "unknown"), "params": dict(event.get("params", {}) or {})}


def _parse_event_time(value: Any):
    if not value:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        if s.endswith("Z"):
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
        dt = datetime.fromisoformat(s)
        return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def _cohort_key(intent: Dict[str, Any], path_id: str = "") -> str:
    params = dict(intent.get("params", {}) or {})
    hard_keys, _soft_keys = _get_match_keys(path_id, intent) if path_id else ([], [])
    keys = [k for k in hard_keys if k != str(get_path_treatment_spec(path_id, intent).get("contrast_key", ""))] if path_id else []
    if not keys:
        keys = ["action_family", "environment", "resource_sensitivity"]
    pieces = []
    for key in keys:
        if key == "action_family":
            val = _action_family(str(intent.get("action_name", "")))
        elif key == "environment":
            val = intent.get("environment", "")
        else:
            val = params.get(key)
        pieces.append(f"{key}={_canonical_match_value(key, val)}")
    return "|".join(pieces)


def _build_temporal_cohorts(events: List[Dict[str, Any]], path_id: str) -> Dict[str, List[Tuple[datetime, Dict[str, Any]]]]:
    out: Dict[str, List[Tuple[datetime, Dict[str, Any]]]] = {}
    for ev in events:
        ts = _parse_event_time(ev.get("event_time"))
        if ts is None:
            continue
        intent = _event_to_intent(ev)
        key = _cohort_key(intent, path_id)
        out.setdefault(key, []).append((ts, ev))
    for key in out:
        out[key].sort(key=lambda x: x[0])
    return out


def _longitudinal_panel_effect(treated, controls, harm, events, path_id: str, pre_k: int = 3, post_k: int = 1):
    cohorts = _build_temporal_cohorts(events, path_id)
    if not treated or not controls or not cohorts:
        return {
            "panel_event_window_effect": 0.0,
            "panel_pretrend_gap": 0.0,
            "panel_support": 0,
            "panel_unit_support": 0,
            "panel_pre_avg": 0.0,
            "panel_post_avg": 0.0,
            "panel_control_pre_avg": 0.0,
            "panel_control_post_avg": 0.0,
            "panel_event_study_ok": False,
            "panel_effect_agreement": 0.0,
        }

    def event_stats(weighted_events):
        deltas = []
        pre_vals = []
        post_vals = []
        units = set()
        support = 0
        for ev, w in weighted_events:
            ts = _parse_event_time(ev.get("event_time"))
            if ts is None:
                continue
            intent = _event_to_intent(ev)
            key = _cohort_key(intent, path_id)
            cohort = cohorts.get(key, [])
            if len(cohort) < 2:
                continue
            idx = next((i for i, (c_ts, c_ev) in enumerate(cohort) if c_ev.get("event_id") == ev.get("event_id")), None)
            if idx is None:
                continue
            prev = [c_ev for _t, c_ev in cohort[max(0, idx - pre_k):idx]]
            nxt = [c_ev for _t, c_ev in cohort[idx:min(len(cohort), idx + post_k + 1)]]
            if not prev or not nxt:
                continue
            pre_rate = sum(_harm_indicator(x, harm) for x in prev) / max(1, len(prev))
            post_rate = sum(_harm_indicator(x, harm) for x in nxt) / max(1, len(nxt))
            deltas.append((post_rate - pre_rate, w))
            pre_vals.append((pre_rate, w))
            post_vals.append((post_rate, w))
            units.add(key)
            support += 1
        return {
            "delta": _weighted_mean(deltas),
            "pre": _weighted_mean(pre_vals),
            "post": _weighted_mean(post_vals),
            "units": len(units),
            "support": support,
        }

    t = event_stats(treated)
    c = event_stats(controls)
    effect = float(t["delta"]) - float(c["delta"])
    pre_gap = abs(float(t["pre"]) - float(c["pre"]))
    agreement = max(0.0, 1.0 - min(1.0, abs(effect)))
    return {
        "panel_event_window_effect": round(effect, 3),
        "panel_pretrend_gap": round(pre_gap, 3),
        "panel_support": int(min(t["support"], c["support"])),
        "panel_unit_support": int(min(t["units"], c["units"])),
        "panel_pre_avg": round(float(t["pre"]), 3),
        "panel_post_avg": round(float(t["post"]), 3),
        "panel_control_pre_avg": round(float(c["pre"]), 3),
        "panel_control_post_avg": round(float(c["post"]), 3),
        "panel_event_study_ok": bool(pre_gap <= float(_cfg("panel", "pretrend_gap_max", default=0.12)) and min(t["support"], c["support"]) >= int(_cfg("panel", "min_support", default=2))),
        "panel_effect_agreement": round(agreement, 3),
    }


def _action_family(action_name: str) -> str:
    action_name = str(action_name or "")
    if "email" in action_name or "share_file" in action_name:
        return "externalization"
    if "delete" in action_name:
        return "destructive_mutation"
    if "config" in action_name or "deploy" in action_name:
        return "configuration_change"
    if "permission" in action_name or "approval" in action_name:
        return "privilege_workflow"
    return action_name or "unknown"


def _path_activated_for_intent(intent: Dict[str, Any], path_spec: Dict[str, Any], registry: Dict[str, Any]) -> bool:
    action_spec = get_action_spec(str(intent.get("action_name", "")), registry) or {}
    direct_effects = set(derive_action_effects(intent, action_spec=action_spec) or [])
    flags = {str(k): bool(v) for k, v in (extract_runtime_context(intent, action_spec=action_spec) or {}).items()}
    triggers_any = set(path_spec.get("triggers_any", []) or [])
    if triggers_any and not (direct_effects & triggers_any):
        return False
    required_any = set(path_spec.get("required_context_any", []) or [])
    if required_any and not any(flags.get(x, False) for x in required_any):
        return False
    return True


def _context_similarity(intent: Dict[str, Any], event_intent: Dict[str, Any]) -> float:
    score = _match_score(intent, event_intent)
    ip = dict(intent.get("params", {}) or {})
    ep = dict(event_intent.get("params", {}) or {})
    for key in ("resource_sensitivity", "blast_radius", "recipient_scope"):
        if key in ip and key in ep and str(ip.get(key)) == str(ep.get(key)):
            score += 0.08
    for key in ("approval_present", "rollback_available", "novel_action", "attachment_present"):
        if key in ip and key in ep and boolish(ip.get(key)) == boolish(ep.get(key)):
            score += 0.04
    if str(intent.get("environment", "")) == str(event_intent.get("environment", "")):
        score += 0.08
    if str(intent.get("action_name", "")) == str(event_intent.get("action_name", "")):
        score += 0.10
    if _action_family(str(intent.get("action_name", ""))) == _action_family(str(event_intent.get("action_name", ""))):
        score += 0.08
    return min(1.0, score)


def _stratum_score(intent: Dict[str, Any], event_intent: Dict[str, Any]) -> float:
    ip = dict(intent.get("params", {}) or {})
    ep = dict(event_intent.get("params", {}) or {})
    score = 0.0
    if str(intent.get("environment", "")) == str(event_intent.get("environment", "")):
        score += 0.22
    if _action_family(str(intent.get("action_name", ""))) == _action_family(str(event_intent.get("action_name", ""))):
        score += 0.18
    if str(intent.get("action_name", "")) == str(event_intent.get("action_name", "")):
        score += 0.10
    for key, weight in (("resource_sensitivity", 0.14), ("blast_radius", 0.14), ("recipient_scope", 0.08)):
        if str(ip.get(key, "")) == str(ep.get(key, "")) and str(ip.get(key, "")):
            score += weight
    for key, weight in (("approval_present", 0.08), ("rollback_available", 0.08), ("novel_action", 0.04), ("attachment_present", 0.04)):
        if key in ip and key in ep and boolish(ip.get(key)) == boolish(ep.get(key)):
            score += weight
    return round(min(1.0, score), 3)


def _event_features(intent: Dict[str, Any], exclude_keys: Sequence[str] = ()) -> Dict[str, float]:
    params = dict(intent.get("params", {}) or {})
    excluded = {str(k) for k in (exclude_keys or ())}
    out: Dict[str, float] = {}
    for key in CATEGORICAL_KEYS:
        if key in excluded:
            continue
        val = intent.get(key) if key in {"environment", "action_name"} else params.get(key)
        if val is None:
            continue
        out[f"{key}={val}"] = 1.0
    out[f"action_family={_action_family(str(intent.get('action_name', '')))}"] = 1.0
    for key in BOOLEAN_KEYS:
        if key in excluded:
            continue
        if key in params:
            out[f"{key}=true"] = 1.0 if boolish(params.get(key)) else 0.0
    return out





def _merge_unique(values: Sequence[str]) -> List[str]:
    out: List[str] = []
    for value in values:
        key = str(value or '').strip()
        if key and key not in out:
            out.append(key)
    return out


def _plan_guided_match_keys(path: Dict[str, Any] | None) -> Tuple[List[str], List[str]]:
    plan = dict((path or {}).get('validation_plan', {}) or {})
    if not plan:
        return [], []
    treatment_col = str(plan.get('treatment_col', '')).strip()
    outcome_col = str(plan.get('outcome_col', '')).strip()
    post_treatment = {str(x).strip() for x in list(plan.get('post_treatment_columns', []) or []) if str(x).strip()}
    stratum_keys = [str(x).strip() for x in list(plan.get('candidate_stratum', []) or []) if str(x).strip()]
    covariates = [str(x).strip() for x in list(plan.get('candidate_covariates', []) or []) if str(x).strip()]
    blocked = {treatment_col, outcome_col} | post_treatment
    hard = [k for k in stratum_keys if k and k not in blocked]
    soft = [k for k in covariates if k and k not in blocked and k not in hard]
    # validation plans often surface runtime-relevant context variables; keep only keys
    # that the runtime matcher can actually resolve from intent/params.
    allowed = set(CATEGORICAL_KEYS) | set(BOOLEAN_KEYS) | {
        'action_family', 'environment', 'resource_sensitivity', 'blast_radius',
        'recipient_scope', 'approval_present', 'rollback_available', 'novel_action',
        'attachment_present', 'service_criticality'
    }
    hard = [k for k in hard if k in allowed]
    soft = [k for k in soft if k in allowed]
    return _merge_unique(hard), _merge_unique(soft)


def _get_match_keys(path_id: str, intent: Dict[str, Any], path: Dict[str, Any] | None = None) -> Tuple[List[str], List[str]]:
    spec = get_path_treatment_spec(path_id, intent) if path_id else {}
    hard = [str(x) for x in list(spec.get("hard_match_keys", []) or []) if str(x).strip()]
    soft = [str(x) for x in list(spec.get("soft_balance_keys", []) or []) if str(x).strip()]
    required = [str(x) for x in list(spec.get("required_confounders", []) or []) if str(x).strip()]
    if not hard and required:
        hard = required[: min(3, len(required))]
    soft_merged: List[str] = []
    for key in soft + required:
        if key not in hard and key not in soft_merged:
            soft_merged.append(key)
    plan_hard, plan_soft = _plan_guided_match_keys(path)
    merged_hard = _merge_unique(list(hard) + plan_hard)
    merged_soft = _merge_unique(list(soft_merged) + [k for k in plan_soft if k not in merged_hard])
    return merged_hard, merged_soft


def _resolve_intent_value(intent: Dict[str, Any], key: str) -> Any:
    params = dict(intent.get("params", {}) or {})
    if key in params:
        return params.get(key)
    return intent.get(key)


def _canonical_match_value(key: str, value: Any) -> str:
    if value is None:
        return "na"
    if key in BOOLEAN_KEYS or key.endswith("_present") or key.endswith("_available") or key == "novel_action":
        return "1" if boolish(value) else "0"
    sval = str(value).strip().lower().replace(" ", "_")
    if key == "environment":
        return {"production": "prod", "prod": "prod", "staging": "staging", "stage": "staging", "development": "dev", "dev": "dev", "qa": "test", "test": "test"}.get(sval, sval or "na")
    if key == "recipient_scope":
        return {"outside": "external", "public": "external", "external": "external", "inside": "internal", "internal": "internal"}.get(sval, sval or "na")
    if key in {"resource_sensitivity", "service_criticality", "blast_radius"}:
        return {"critical": "high", "very_high": "high", "high": "high", "pii": "high", "restricted": "high", "confidential": "high", "internal": "medium", "private": "medium", "medium": "medium", "moderate": "medium", "public": "low", "low": "low", "minor": "low", "tier1": "high", "tier_1": "high", "tier2": "medium", "tier_2": "medium", "tier3": "low", "tier_3": "low"}.get(sval, sval or "na")
    return sval or "na"


def _value_severity(key: str, value: Any) -> float:
    if key in BOOLEAN_KEYS or key.endswith("_present") or key.endswith("_available") or key == "novel_action":
        return 1.0 if boolish(value) else 0.0
    canon = _canonical_match_value(key, value)
    if key in {"resource_sensitivity", "service_criticality", "blast_radius"}:
        return SEVERITY_MAP.get(canon, 0.0)
    if key == "recipient_scope":
        return 1.0 if canon == "external" else 0.0
    return 1.0 if canon != "na" else 0.0


def _hard_match_ratio(intent: Dict[str, Any], event_intent: Dict[str, Any], hard_keys: Sequence[str]) -> float:
    if not hard_keys:
        return 1.0
    matched = 0
    total = 0
    for key in hard_keys:
        target = _canonical_match_value(key, _resolve_intent_value(intent, key))
        cand = _canonical_match_value(key, _resolve_intent_value(event_intent, key))
        if target == "na" or cand == "na":
            continue
        total += 1
        if target == cand:
            matched += 1
    if total == 0:
        return 0.0
    return round(matched / total, 3)


def _soft_balance_score(intent: Dict[str, Any], event_intent: Dict[str, Any], soft_keys: Sequence[str]) -> float:
    if not soft_keys:
        return 1.0
    vals = []
    for key in soft_keys:
        target = _resolve_intent_value(intent, key)
        cand = _resolve_intent_value(event_intent, key)
        if target is None or cand is None:
            continue
        t = _value_severity(key, target)
        c = _value_severity(key, cand)
        vals.append(max(0.0, 1.0 - abs(t - c)))
    if not vals:
        return 0.5
    return round(sum(vals) / len(vals), 3)


def _nearest_neighbor_reweight(candidates: List[Tuple[float, Dict[str, Any], float, float]], target_prop: float, top_k: int, hard_floor: float = 0.5) -> List[Tuple[float, Dict[str, Any], float, float]]:
    rescored: List[Tuple[float, Dict[str, Any], float, float]] = []
    for base_weight, event, prop, strat in candidates:
        gap = abs(float(prop) - float(target_prop))
        score = 0.62 * float(base_weight) + 0.24 * max(0.0, 1.0 - min(1.0, gap / 0.25)) + 0.14 * float(strat)
        rescored.append((round(max(hard_floor, min(1.0, score)), 4), event, prop, strat))
    rescored.sort(key=lambda x: x[0], reverse=True)
    return rescored[:top_k]

def _excluded_match_keys_for_path(path_id: str, intent: Dict[str, Any]) -> List[str]:
    spec = get_path_treatment_spec(path_id, intent) if path_id else {}
    confounders = {str(x) for x in list(spec.get("required_confounders", []) or [])}
    treatment_var = str(spec.get("treatment_var", "") or "")
    keep = {k for k in confounders if k not in {"action_family", "environment"}}
    excluded: List[str] = []
    for key in list(CATEGORICAL_KEYS) + list(BOOLEAN_KEYS):
        if key == treatment_var:
            excluded.append(key)
        elif confounders and key not in keep:
            excluded.append(key)
    return excluded


def _fit_logistic(features: List[Dict[str, float]], labels: List[int], steps: int | None = None, lr: float | None = None, l2: float | None = None) -> Dict[str, float]:
    if not features:
        return {}
    steps = int(steps if steps is not None else _cfg("logistic", "steps", default=250))
    lr = float(lr if lr is not None else _cfg("logistic", "lr", default=0.30))
    l2 = float(l2 if l2 is not None else _cfg("logistic", "l2", default=0.01))
    clip = float(_cfg("logistic", "clip", default=20.0))
    keys = sorted({k for row in features for k in row.keys()})
    if not keys:
        return {}
    w = {k: 0.0 for k in keys}
    bias = math.log((sum(labels) + 0.5) / (max(1, len(labels) - sum(labels)) + 0.5))
    for _ in range(steps):
        grad = {k: 0.0 for k in keys}
        grad_b = 0.0
        for row, y in zip(features, labels):
            z = bias + sum(w[k] * row.get(k, 0.0) for k in keys)
            z = max(-clip, min(clip, z))
            p = 1.0 / (1.0 + math.exp(-z))
            err = p - float(y)
            grad_b += err
            for k in keys:
                grad[k] += err * row.get(k, 0.0)
        n = max(1, len(features))
        bias -= lr * grad_b / n
        for k in keys:
            w[k] -= lr * ((grad[k] / n) + l2 * w[k])
    w["__bias__"] = bias
    return w


def _predict_logistic(model: Dict[str, float], row: Dict[str, float]) -> float:
    if not model:
        return 0.5
    p_min = float(_cfg("logistic", "p_min", default=0.01))
    p_max = float(_cfg("logistic", "p_max", default=0.99))
    clip = float(_cfg("logistic", "clip", default=20.0))
    z = float(model.get("__bias__", 0.0))
    for k, v in row.items():
        if k == "__bias__":
            continue
        z += float(model.get(k, 0.0)) * float(v)
    z = max(-clip, min(clip, z))
    p = 1.0 / (1.0 + math.exp(-z))
    return max(p_min, min(p_max, p))


def _context_prior(intent: Dict[str, Any], harm: str) -> float:
    params = dict(intent.get("params", {}) or {})
    score = 0.02
    priors = dict(PATH_HARM_PRIORS.get(str(harm), {}) or {})
    if str(intent.get("environment", "")) == "production":
        score += priors.get("production", 0.08)
    rs = str(params.get("resource_sensitivity", ""))
    br = str(params.get("blast_radius", ""))
    if rs == "high":
        score += priors.get("high_sensitivity", 0.16)
    elif rs == "medium":
        score += priors.get("medium_sensitivity", 0.06)
    if br == "high":
        score += priors.get("high_blast", 0.16)
    elif br == "medium":
        score += priors.get("medium_blast", 0.06)
    if str(params.get("recipient_scope", "")) == "external":
        score += priors.get("external", 0.08)
    if not boolish(params.get("approval_present", True)):
        score += priors.get("no_approval", 0.10)
    if not boolish(params.get("rollback_available", True)):
        score += priors.get("no_rollback", 0.10)
    if boolish(params.get("novel_action", False)):
        score += priors.get("novel", 0.08)
    if boolish(params.get("attachment_present", False)):
        score += priors.get("attachment", 0.05)
    return max(0.0, min(1.0, score))


def _harm_indicator(event: Dict[str, Any], harm: str) -> float:
    return 1.0 if harm and harm in list(event.get("observed_harms", []) or []) else 0.0


def _weighted_mean(values_and_weights: Iterable[Tuple[float, float]]) -> float:
    pairs = list(values_and_weights)
    total_w = sum(max(0.0, w) for _v, w in pairs)
    if total_w <= 0:
        return 0.0
    return sum(v * max(0.0, w) for v, w in pairs) / total_w


def _weighted_harm_rate(events_and_weights: List[Tuple[Dict[str, Any], float]], harm: str) -> float:
    return _weighted_mean((_harm_indicator(e, harm), w) for e, w in events_and_weights)


def _effective_support(events_and_weights: List[Tuple[Dict[str, Any], float]]) -> int:
    return sum(1 for _, w in events_and_weights if w > 0)


def _effective_sample_size(events_and_weights: List[Tuple[Dict[str, Any], float]]) -> float:
    weights = [max(0.0, float(w)) for _e, w in events_and_weights if max(0.0, float(w)) > 0.0]
    if not weights:
        return 0.0
    num = sum(weights) ** 2
    den = sum(w * w for w in weights)
    if den <= 0:
        return 0.0
    return round(num / den, 3)


def _dominant_control_weight(events_and_weights: List[Tuple[Dict[str, Any], float]]) -> float:
    weights = [max(0.0, float(w)) for _e, w in events_and_weights if max(0.0, float(w)) > 0.0]
    total = sum(weights)
    if total <= 0 or not weights:
        return 1.0
    return round(max(weights) / total, 3)


def _weighted_feature_mean(events_and_weights: List[Tuple[Dict[str, Any], float]], feature: str) -> float:
    values: List[Tuple[float, float]] = []
    for e, w in events_and_weights:
        it = _event_to_intent(e)
        params = dict(it.get("params", {}) or {})
        if feature in {"environment", "action_name"}:
            values.append((1.0 if str(it.get(feature, "")) else 0.0, w))
        elif feature in BOOLEAN_KEYS:
            values.append((1.0 if boolish(params.get(feature, False)) else 0.0, w))
        elif feature in {"resource_sensitivity", "blast_radius"}:
            values.append((SEVERITY_MAP.get(str(params.get(feature, "low")), 0.0), w))
        elif feature == "recipient_scope":
            values.append((1.0 if str(params.get(feature, "")) == "external" else 0.0, w))
    return _weighted_mean(values)


def _feature_smds(treated: List[Tuple[Dict[str, Any], float]], controls: List[Tuple[Dict[str, Any], float]]) -> Dict[str, float]:
    feats = ["resource_sensitivity", "blast_radius", "recipient_scope", "approval_present", "rollback_available", "novel_action", "attachment_present"]
    out: Dict[str, float] = {}
    for feat in feats:
        mt = _weighted_feature_mean(treated, feat)
        mc = _weighted_feature_mean(controls, feat)
        out[feat] = round(abs(mt - mc), 3)
    return out


def _mean_abs_smd(treated: List[Tuple[Dict[str, Any], float]], controls: List[Tuple[Dict[str, Any], float]]) -> float:
    smds = _feature_smds(treated, controls)
    return round(sum(smds.values()) / max(1, len(smds)), 3)


def _intent_cache_key(intent: Dict[str, Any], path_id: str = "") -> Tuple[Any, ...]:
    params = dict(intent.get("params", {}) or {})
    keys, _soft = _get_match_keys(path_id, intent) if path_id else (["action_family", "environment", "resource_sensitivity"], [])
    pieces: List[Tuple[str, str]] = []
    for key in keys:
        pieces.append((str(key), _canonical_match_value(key, _resolve_intent_value(intent, key))))
    for key in ("recipient_scope", "blast_radius", "approval_present", "rollback_available", "attachment_present", "novel_action"):
        if key in params:
            pieces.append((key, _canonical_match_value(key, params.get(key))))
    return (str(intent.get("action_name", "")), str(intent.get("environment", "")), tuple(sorted(set(pieces))))


def _historical_baseline_hazard(intent: Dict[str, Any], events: List[Dict[str, Any]], harm: str, path_spec: Dict[str, Any], registry: Dict[str, Any]) -> float:
    base_prior = _context_prior(intent, harm)
    if not events or not harm:
        return round(base_prior, 3)
    path_id = str(path_spec.get("path_id", "") or "")
    cache_key = (id(events), path_id, str(harm), _intent_cache_key(intent, path_id))
    cached = _BASELINE_HAZARD_CACHE.get(cache_key)
    if cached is not None:
        return cached
    weighted: List[Tuple[float, float]] = []
    target_family = _action_family(str(intent.get("action_name", "")))
    for event in events:
        ev_intent = _event_to_intent(event)
        if _path_activated_for_intent(ev_intent, path_spec, registry):
            continue
        sim = _context_similarity(intent, ev_intent)
        strat = _stratum_score(intent, ev_intent)
        fam = _action_family(str(ev_intent.get("action_name", "")))
        weight = 0.50 * sim + 0.40 * strat
        if fam == target_family:
            weight += 0.10
        if weight >= 0.45:
            weighted.append((_harm_indicator(event, harm), min(1.0, weight)))
    if not weighted:
        out = round(base_prior, 3)
    else:
        hist = _weighted_mean(weighted)
        out = round(0.55 * hist + 0.45 * base_prior, 3)
    if len(_BASELINE_HAZARD_CACHE) > _MAX_BASELINE_CACHE:
        _BASELINE_HAZARD_CACHE.clear()
    _BASELINE_HAZARD_CACHE[cache_key] = out
    return out


def _did_effect(treated, controls, harm, events, path_spec, registry):
    treat_post = _weighted_harm_rate(treated, harm)
    ctrl_post = _weighted_harm_rate(controls, harm)
    treat_pre = _weighted_mean((_historical_baseline_hazard(_event_to_intent(e), events, harm, path_spec, registry), w) for e, w in treated)
    ctrl_pre = _weighted_mean((_historical_baseline_hazard(_event_to_intent(e), events, harm, path_spec, registry), w) for e, w in controls)
    effect = (treat_post - treat_pre) - (ctrl_post - ctrl_pre)
    return round(effect, 3), round(treat_pre, 3), round(ctrl_pre, 3), round(treat_post, 3), round(ctrl_post, 3), round((treat_post - ctrl_post), 3)


def _bootstrap_ci(treated, controls, harm, events, path_spec, registry, draws: int = 80):
    draws = max(0, min(int(draws), int(_cfg("runtime_budget", "bootstrap_draws", default=12))))
    if len(treated) < 2 or len(controls) < 2:
        return 0.0, 0.0
    vals = []
    for _ in range(draws):
        t_sample = [treated[RNG.randrange(len(treated))] for _ in range(len(treated))]
        c_sample = [controls[RNG.randrange(len(controls))] for _ in range(len(controls))]
        eff, *_ = _did_effect(t_sample, c_sample, harm, events, path_spec, registry)
        vals.append(float(eff))
    vals.sort()
    low_q = float(_cfg("common_support", "ci_low_q", default=0.05))
    high_q = float(_cfg("common_support", "ci_high_q", default=0.95))
    lo = vals[max(0, int(low_q * len(vals)) - 1)]
    hi = vals[min(len(vals) - 1, int(high_q * len(vals)))]
    return round(lo, 3), round(hi, 3)


def _placebo_threshold_for_path(path_id: str) -> float:
    return {
        "external_data_leakage": 0.12,
        "destructive_mutation": 0.14,
        "operational_failure": 0.16,
    }.get(str(path_id or ""), 0.20)


def _available_placebo_harms(events, exclude, preferred=None):
    harms = []
    seen = set()
    for h in list(preferred or []):
        sh = str(h or "").strip()
        if sh and sh != exclude and sh not in seen:
            harms.append(sh); seen.add(sh)
    for e in events:
        for h in list(e.get("observed_harms", []) or []):
            sh = str(h or "").strip()
            if sh and sh != exclude and sh not in seen:
                harms.append(sh); seen.add(sh)
    return harms


def _randomized_treatment_placebo(treated, controls, harm, events, path_spec, registry, draws: int = 16):
    draws = max(0, min(int(draws), int(_cfg("runtime_budget", "placebo_draws", default=6))))
    pool = list(treated) + list(controls)
    if len(pool) < 4:
        return {"random_treatment_placebo_max_abs_delta": 0.0, "random_treatment_placebo_fail_rate": 0.0}
    t_n = max(1, len(treated)); c_n = max(1, len(controls)); deltas = []
    for _ in range(draws):
        shuffled = pool[:]; RNG.shuffle(shuffled)
        pseudo_t = shuffled[:t_n]; pseudo_c = shuffled[t_n:t_n + c_n]
        if not pseudo_c:
            continue
        eff, *_ = _did_effect(pseudo_t, pseudo_c, harm, events, path_spec, registry)
        deltas.append(abs(float(eff)))
    if not deltas:
        return {"random_treatment_placebo_max_abs_delta": 0.0, "random_treatment_placebo_fail_rate": 0.0}
    fail_cutoff = float(_cfg("placebo", "random_treatment_delta_fail", default=0.20))
    fails = sum(1 for d in deltas if d >= fail_cutoff)
    return {"random_treatment_placebo_max_abs_delta": round(max(deltas), 3), "random_treatment_placebo_fail_rate": round(fails / len(deltas), 3)}


def _harm_permutation_placebo(treated, controls, harm, events, path_spec, registry, draws: int = 16):
    draws = max(0, min(int(draws), int(_cfg("runtime_budget", "placebo_draws", default=6))))
    pool = list(treated) + list(controls)
    if len(pool) < 4:
        return {"permuted_outcome_placebo_max_abs_delta": 0.0, "permuted_outcome_placebo_fail_rate": 0.0}
    y = [_harm_indicator(e, harm) for e, _w in pool]; ws = [w for _e, w in pool]; deltas = []; n_t = len(treated)
    for _ in range(draws):
        yp = y[:]; RNG.shuffle(yp)
        pseudo_pool = [(dict(e, __perm_harm__=yp[i]), ws[i]) for i, (e, _w) in enumerate(pool)]
        pseudo_t = pseudo_pool[:n_t]; pseudo_c = pseudo_pool[n_t:]
        treat_post = _weighted_mean((float(e.get("__perm_harm__", 0.0)), w) for e, w in pseudo_t)
        ctrl_post = _weighted_mean((float(e.get("__perm_harm__", 0.0)), w) for e, w in pseudo_c)
        treat_pre = _weighted_mean((_historical_baseline_hazard(_event_to_intent(e), events, harm, path_spec, registry), w) for e, w in pseudo_t)
        ctrl_pre = _weighted_mean((_historical_baseline_hazard(_event_to_intent(e), events, harm, path_spec, registry), w) for e, w in pseudo_c)
        deltas.append(abs((treat_post - treat_pre) - (ctrl_post - ctrl_pre)))
    fail_cutoff = float(_cfg("placebo", "random_treatment_delta_fail", default=0.20))
    fails = sum(1 for d in deltas if d >= fail_cutoff)
    return {"permuted_outcome_placebo_max_abs_delta": round(max(deltas), 3) if deltas else 0.0, "permuted_outcome_placebo_fail_rate": round(fails / len(deltas), 3) if deltas else 0.0}




def _temporal_placebo(treated, controls, harm, events, path_spec, registry, draws: int = 16):
    draws = max(0, min(int(draws), int(_cfg("runtime_budget", "placebo_draws", default=6))))
    if not treated or not controls:
        return {"temporal_placebo_max_abs_delta": 0.0, "temporal_placebo_fail_rate": 0.0, "temporal_placebo_gap": 0.0}
    deltas = []
    for _ in range(draws):
        t_sample = [treated[RNG.randrange(len(treated))] for _ in range(len(treated))]
        c_sample = [controls[RNG.randrange(len(controls))] for _ in range(len(controls))]
        t_pre = _weighted_mean((_historical_baseline_hazard(_event_to_intent(e), events, harm, path_spec, registry), w) for e, w in t_sample)
        c_pre = _weighted_mean((_historical_baseline_hazard(_event_to_intent(e), events, harm, path_spec, registry), w) for e, w in c_sample)
        deltas.append(abs(t_pre - c_pre))
    if not deltas:
        return {"temporal_placebo_max_abs_delta": 0.0, "temporal_placebo_fail_rate": 0.0, "temporal_placebo_gap": 0.0}
    gap = float(sum(deltas) / len(deltas))
    fail_cutoff = float(_cfg("placebo", "temporal_delta_fail", default=0.12))
    fails = sum(1 for d in deltas if d >= fail_cutoff)
    return {
        "temporal_placebo_max_abs_delta": round(max(deltas), 3),
        "temporal_placebo_fail_rate": round(fails / len(deltas), 3),
        "temporal_placebo_gap": round(gap, 3),
    }

def _placebo_stats(treated, controls, events, harm, path_spec, registry):
    path_id = str(path_spec.get("path_id", "") or "")
    threshold = _placebo_threshold_for_path(path_id)
    preferred = list(path_spec.get("negative_controls", []) or [])
    placebo_harms = _available_placebo_harms(events, harm, preferred=preferred)
    outcome_deltas = []
    outcome_details = []
    max_placebos = int(_cfg("runtime_budget", "max_placebo_outcomes", default=3))
    for ph in placebo_harms[:max_placebos]:
        eff, *_ = _did_effect(treated, controls, ph, events, path_spec, registry)
        outcome_deltas.append(float(eff))
        outcome_details.append({"placebo_outcome": ph, "abs_delta": round(abs(float(eff)), 3)})
    outcome_fail_rate = 0.0; outcome_max = 0.0
    if outcome_deltas:
        outcome_fail_rate = sum(1 for d in outcome_deltas if abs(d) >= threshold) / len(outcome_deltas)
        outcome_max = max(abs(d) for d in outcome_deltas)
    rand_placebo = _randomized_treatment_placebo(treated, controls, harm, events, path_spec, registry)
    perm_placebo = _harm_permutation_placebo(treated, controls, harm, events, path_spec, registry)
    temporal_placebo = _temporal_placebo(treated, controls, harm, events, path_spec, registry)
    fail_components = [
        outcome_fail_rate,
        float(rand_placebo.get("random_treatment_placebo_fail_rate", 0.0)),
        float(perm_placebo.get("permuted_outcome_placebo_fail_rate", 0.0)),
        float(temporal_placebo.get("temporal_placebo_fail_rate", 0.0)),
    ]
    placebo_fail_rate = round(sum(fail_components) / len(fail_components), 3)
    return {
        "placebo_count": len(placebo_harms[:max_placebos]),
        "placebo_max_abs_delta": round(outcome_max, 3),
        "placebo_fail_rate": placebo_fail_rate,
        "placebo_pass": max(fail_components) <= float(_cfg("placebo", "fail_component_max", default=0.34)) and placebo_fail_rate <= float(_cfg("placebo", "fail_rate_max", default=0.28)),
        "placebo_path_threshold": round(threshold, 3),
        "placebo_negative_controls_used": placebo_harms[:max_placebos],
        "placebo_outcome_details": outcome_details,
        **rand_placebo, **perm_placebo, **temporal_placebo
    }


def _pretrend_assessment(treated, controls, harm, events, path_spec, registry):
    if not treated or not controls:
        return {
            "counterfactual_pretrend_gap": 1.0,
            "counterfactual_pretrend_mean_gap": 1.0,
            "counterfactual_pretrend_max_dev": 1.0,
            "counterfactual_pretrend_testable": False,
            "counterfactual_pretrend_ok": False,
            "counterfactual_parallel_trend_flag": "not_testable",
        }
    treat_pre_vals = [(_historical_baseline_hazard(_event_to_intent(e), events, harm, path_spec, registry), w) for e, w in treated]
    ctrl_pre_vals = [(_historical_baseline_hazard(_event_to_intent(e), events, harm, path_spec, registry), w) for e, w in controls]
    treat_pre = _weighted_mean(treat_pre_vals)
    ctrl_pre = _weighted_mean(ctrl_pre_vals)
    treat_pre_only = [float(v) for v, _w in treat_pre_vals]
    ctrl_pre_only = [float(v) for v, _w in ctrl_pre_vals]
    mean_gap = abs(treat_pre - ctrl_pre)
    max_dev = 0.0
    if treat_pre_only and ctrl_pre_only:
        t_lo, t_hi = min(treat_pre_only), max(treat_pre_only)
        c_lo, c_hi = min(ctrl_pre_only), max(ctrl_pre_only)
        max_dev = max(abs(t_lo - c_lo), abs(t_hi - c_hi), mean_gap)
    testable = len(treat_pre_only) >= 2 and len(ctrl_pre_only) >= 2
    strong_ok = testable and mean_gap <= float(_cfg("pretrend", "strong_mean_gap_max", default=0.08)) and max_dev <= float(_cfg("pretrend", "strong_max_dev_max", default=0.16))
    weak_ok = testable and mean_gap <= float(_cfg("pretrend", "weak_mean_gap_max", default=0.12)) and max_dev <= float(_cfg("pretrend", "weak_max_dev_max", default=0.22))
    flag = "plausible" if strong_ok else ("weak" if weak_ok else ("not_testable" if not testable else "divergent"))
    return {
        "counterfactual_pretrend_gap": round(mean_gap, 3),
        "counterfactual_pretrend_mean_gap": round(mean_gap, 3),
        "counterfactual_pretrend_max_dev": round(max_dev, 3),
        "counterfactual_pretrend_testable": bool(testable),
        "counterfactual_pretrend_ok": bool(strong_ok),
        "counterfactual_parallel_trend_flag": flag,
    }


def _sensitivity(effect, prop_gap, balance_smd, n_t, n_c, control_q, placebo_fail_rate, pretrend_ok, stratum_support, exact_stratum_control=0, overlap_ok=False, dominant_control_weight=1.0, method_agreement_ratio=0.0, temporal_placebo_gap=0.0, panel_pretrend_gap=0.0, panel_support=0):
    score = compute_identification_score(
        overlap_ok=bool(overlap_ok),
        balance_ok=balance_smd <= 0.22,
        placebo_pass=placebo_fail_rate <= 0.40,
        pretrend_ok=bool(pretrend_ok),
        valid_strata=1 if stratum_support > 0 else 0,
        exact_stratum_control=int(exact_stratum_control),
        causal_stratum_control=int(exact_stratum_control),
        prop_gap=float(prop_gap),
        balance_smd=float(balance_smd),
        shared_support_ratio=float(stratum_support),
        design_strength="medium" if overlap_ok else "low",
    )
    ident = classify_identification(score)
    conf_risk = confounding_risk_from_components(score, bool(overlap_ok), balance_smd <= 0.22, placebo_fail_rate <= 0.40, bool(pretrend_ok))
    robustness = sensitivity_ratio(float(effect), float(prop_gap), float(balance_smd), int(n_t), int(n_c), float(control_q), float(placebo_fail_rate), bool(pretrend_ok), float(stratum_support), int(exact_stratum_control), bool(overlap_ok))
    bias_load = max(0.01, float(prop_gap) + 0.75 * float(balance_smd) + 0.55 * float(placebo_fail_rate) + 0.35 * float(temporal_placebo_gap) + 0.30 * max(0.0, float(dominant_control_weight) - 0.35) + 0.25 * max(0.0, 0.75 - float(method_agreement_ratio)))
    robustness_value = min(5.0, abs(float(effect)) / bias_load)
    hidden_bias_tolerance = min(1.0, 0.30 * float(robustness_value) + 0.25 * float(stratum_support) + 0.20 * float(method_agreement_ratio) + (0.15 if overlap_ok else 0.0) + (0.10 if pretrend_ok else 0.0))
    return {
        "counterfactual_sensitivity_ratio": round(robustness, 3),
        "counterfactual_identification_support": ident,
        "counterfactual_confounding_risk": conf_risk,
        "counterfactual_robustness_value": round(robustness_value, 3),
        "counterfactual_hidden_bias_tolerance": round(hidden_bias_tolerance, 3),
        "counterfactual_sensitivity_bias_load": round(bias_load, 3),
    }

def _counterfactual_label(n_t, n_c, treat_rate, control_rate, did_effect, control_q, prop_gap, balance_smd, placebo_fail_rate, pretrend_ok, stratum_support, exact_stratum_control=0, overlap_ok=False, effective_support_n=0, dominant_control_weight=1.0, pretrend_testable=False, panel_effect=0.0, panel_event_study_ok=False):
    risk_delta = float(treat_rate) - float(control_rate)
    if n_t < 2 or n_c < 2 or effective_support_n < 2:
        return "not_identified"
    if not overlap_ok or prop_gap > 0.16 or dominant_control_weight > 0.60:
        return "weak_match_only"
    if not pretrend_testable:
        return "weak_match_only"
    if not pretrend_ok or placebo_fail_rate > 0.40 or balance_smd > 0.28:
        return "effect_without_identification" if max(did_effect, panel_effect) > 0.06 else "weak_match_only"
    if exact_stratum_control < 2 or stratum_support < 0.50 or control_q < 0.48:
        return "effect_without_identification" if did_effect > 0.08 else "weak_match_only"
    if did_effect >= 0.18 and risk_delta >= 0.15 and placebo_fail_rate <= 0.20 and balance_smd <= 0.14 and prop_gap <= 0.10 and exact_stratum_control >= 2 and dominant_control_weight <= 0.42 and panel_event_study_ok:
        return "strong_counterfactual_support"
    if max(did_effect, panel_effect) >= 0.08 and risk_delta >= 0.06:
        return "directionally_supported"
    return "effect_without_identification" if did_effect > 0.0 else "weak_match_only"


def _legacy_counterfactual_tier(detail_label: str) -> str:
    mapping = {
        "strong_counterfactual_support": "high",
        "directionally_supported": "medium",
        "effect_without_identification": "low",
        "weak_match_only": "low",
        "not_identified": "none",
    }
    return mapping.get(str(detail_label or ""), "none")



def _exact_stratum_key(intent: Dict[str, Any]) -> Tuple[str, str, str, str, int, int]:
    params = dict(intent.get("params", {}) or {})
    return (
        str(intent.get("environment", "")),
        _action_family(str(intent.get("action_name", ""))),
        str(params.get("resource_sensitivity", "")),
        str(params.get("blast_radius", "")),
        1 if boolish(params.get("approval_present", False)) else 0,
        1 if boolish(params.get("rollback_available", False)) else 0,
    )


def _trim_overlap(candidates: List[Tuple[float, Dict[str, Any], float, float]], target_prop: float, width: float = 0.22, min_local_support: int = 2) -> Tuple[List[Tuple[float, Dict[str, Any], float, float]], Dict[str, Any]]:
    trimmed: List[Tuple[float, Dict[str, Any], float, float]] = []
    lo = max(0.01, target_prop - width)
    hi = min(0.99, target_prop + width)
    local_width = max(0.06, width * 0.5)
    local_lo = max(0.01, target_prop - local_width)
    local_hi = min(0.99, target_prop + local_width)
    local_support = 0
    total_weight = 0.0
    local_weight = 0.0
    for item in candidates:
        prop = float(item[2])
        weight = max(0.0, float(item[0]))
        if lo <= prop <= hi:
            trimmed.append(item)
            total_weight += weight
            if local_lo <= prop <= local_hi:
                local_support += 1
                local_weight += weight
    overlap_info = {
        "overlap_local_support": int(local_support),
        "overlap_effective_mass": round(local_weight / total_weight, 3) if total_weight > 0 else 0.0,
        "overlap_hard_pass": bool(local_support >= min_local_support and (local_weight / total_weight if total_weight > 0 else 0.0) >= 0.40),
    }
    return trimmed, overlap_info




def _common_support_trim(candidates_t: List[Tuple[float, Dict[str, Any], float, float]], candidates_c: List[Tuple[float, Dict[str, Any], float, float]], buffer: float = 0.03) -> Tuple[List[Tuple[float, Dict[str, Any], float, float]], List[Tuple[float, Dict[str, Any], float, float]], Dict[str, Any]]:
    if not candidates_t or not candidates_c:
        return candidates_t, candidates_c, {
            "common_support_low": 0.0,
            "common_support_high": 1.0,
            "common_support_trim_treated": 0,
            "common_support_trim_control": 0,
            "common_support_ratio": 0.0,
            "common_support_ok": False,
        }
    t_props = [float(x[2]) for x in candidates_t]
    c_props = [float(x[2]) for x in candidates_c]
    low = max(min(t_props), min(c_props)) - buffer
    high = min(max(t_props), max(c_props)) + buffer
    low = max(0.01, low)
    high = min(0.99, high)
    trimmed_t = [x for x in candidates_t if low <= float(x[2]) <= high]
    trimmed_c = [x for x in candidates_c if low <= float(x[2]) <= high]
    ratio_t = (len(trimmed_t) / len(candidates_t)) if candidates_t else 0.0
    ratio_c = (len(trimmed_c) / len(candidates_c)) if candidates_c else 0.0
    ratio = min(ratio_t, ratio_c)
    return trimmed_t, trimmed_c, {
        "common_support_low": round(low, 3),
        "common_support_high": round(high, 3),
        "common_support_trim_treated": max(0, len(candidates_t) - len(trimmed_t)),
        "common_support_trim_control": max(0, len(candidates_c) - len(trimmed_c)),
        "common_support_ratio": round(ratio, 3),
        "common_support_ok": bool(len(trimmed_t) >= 2 and len(trimmed_c) >= 2 and ratio >= 0.55),
    }


def _required_balance_audit(treated: List[Tuple[Dict[str, Any], float]], controls: List[Tuple[Dict[str, Any], float]], features: Sequence[str]) -> Dict[str, Any]:
    feats = [str(f) for f in (features or []) if str(f)]
    if not treated or not controls or not feats:
        return {"required_balance_smds": {}, "required_balance_smd_mean": 1.0, "required_balance_ok": False}
    out = {}
    vals = []
    for feat in feats:
        mt = _weighted_feature_mean(treated, feat)
        mc = _weighted_feature_mean(controls, feat)
        smd = round(abs(mt - mc), 3)
        out[feat] = smd
        vals.append(smd)
    mean_smd = round(sum(vals) / max(1, len(vals)), 3)
    return {
        "required_balance_smds": out,
        "required_balance_smd_mean": mean_smd,
        "required_balance_ok": bool(mean_smd <= 0.18 and max(vals or [1.0]) <= 0.28),
    }


def _method_agreement_stats(raw_delta: float, did_effect: float, dr_risk_delta: float, dr_did_effect: float, stratified_risk_delta: float, stratified_did_effect: float, panel_event_window_effect: float = 0.0) -> Dict[str, Any]:
    methods = {
        "risk_delta": float(raw_delta),
        "did_effect": float(did_effect),
        "dr_risk_delta": float(dr_risk_delta),
        "dr_did_effect": float(dr_did_effect),
        "stratified_risk_delta": float(stratified_risk_delta),
        "stratified_did_effect": float(stratified_did_effect),
    }
    if abs(float(panel_event_window_effect)) > 0.0:
        methods["panel_event_window_effect"] = float(panel_event_window_effect)
    vals = list(methods.values())
    active = [v for v in vals if abs(v) >= 0.03]
    pos = sum(1 for v in active if v > 0)
    neg = sum(1 for v in active if v < 0)
    sign_agreement = 1.0 if not active else round(max(pos, neg) / len(active), 3)
    spread = round((max(vals) - min(vals)) if vals else 0.0, 3)
    mean_abs = round(sum(abs(v) for v in vals) / max(1, len(vals)), 3)
    strong = bool(sign_agreement >= 0.80 and spread <= 0.22)
    return {
        "method_estimates": {k: round(v, 3) for k, v in methods.items()},
        "method_agreement_ratio": sign_agreement,
        "method_effect_spread": spread,
        "method_mean_abs_effect": mean_abs,
        "method_agreement_ok": strong,
    }

def _fit_outcome_model(weighted_events: List[Tuple[Dict[str, Any], float]], harm: str, exclude_keys: Sequence[str] = ()) -> Dict[str, float]:
    if not weighted_events:
        return {}
    feats = [_event_features(_event_to_intent(e), exclude_keys=exclude_keys) for e, _w in weighted_events]
    labels = [_harm_indicator(e, harm) for e, _w in weighted_events]
    return _fit_logistic(feats, labels, steps=220, lr=0.24, l2=0.02)


def _predict_outcome_model(model: Dict[str, float], intent: Dict[str, Any], exclude_keys: Sequence[str] = ()) -> float:
    return _predict_logistic(model, _event_features(intent, exclude_keys=exclude_keys)) if model else 0.0


def _dr_effect(treated, controls, harm, events, path_spec, registry, exclude_keys: Sequence[str] = ()):
    if not treated or not controls:
        return 0.0, 0.0
    ctrl_model = _fit_outcome_model(controls, harm, exclude_keys=exclude_keys)
    treat_post = _weighted_harm_rate(treated, harm)
    ctrl_pre = _weighted_mean((_historical_baseline_hazard(_event_to_intent(e), events, harm, path_spec, registry), w) for e, w in controls)
    treat_pre = _weighted_mean((_historical_baseline_hazard(_event_to_intent(e), events, harm, path_spec, registry), w) for e, w in treated)
    pred_treat_as_control = _weighted_mean((_predict_outcome_model(ctrl_model, _event_to_intent(e), exclude_keys=exclude_keys), w) for e, w in treated)
    dr_risk = treat_post - pred_treat_as_control
    dr_did = (treat_post - treat_pre) - (pred_treat_as_control - ctrl_pre)
    return round(dr_risk, 3), round(dr_did, 3)

def _build_matched_sets(intent, events, path_spec, registry, harm, path_id: str = "", path: Dict[str, Any] | None = None, top_k_treat: int = 6, top_k_control: int = 8, min_treat_similarity: float = 0.50, min_control_similarity: float = 0.34, caliper: float = 0.14):
    design = treated_control_design(intent, path_id) if path_id else {}
    excluded_match_keys = _excluded_match_keys_for_path(path_id, intent) if path_id else []
    hard_keys, soft_keys = _get_match_keys(path_id, intent, path=path) if path_id else ([], [])
    labels = []; feats = []; intent_features = _event_features(intent, exclude_keys=excluded_match_keys)
    for event in events:
        ev_intent = _event_to_intent(event)
        feats.append(_event_features(ev_intent, exclude_keys=excluded_match_keys))
        labels.append(1 if _path_activated_for_intent(ev_intent, path_spec, registry) else 0)
    model = _fit_logistic(feats, labels)
    target_prop = _predict_logistic(model, intent_features)
    candidates_t = []; candidates_c = []
    relaxed_candidates_c = []; nearest_candidates_c = []
    target_action = str(intent.get("action_name", "")); target_env = str(intent.get("environment", "")); target_family = _action_family(target_action)
    target_stratum = _exact_stratum_key(intent)
    target_causal_stratum = build_causal_stratum(intent, path_id=path_id)
    target_status = path_treatment_status(intent, path_id)
    for event, row in zip(events, feats):
        ev_intent = _event_to_intent(event)
        sim = _context_similarity(intent, ev_intent)
        strat = _stratum_score(intent, ev_intent)
        prop = _predict_logistic(model, row)
        prop_gap = abs(prop - target_prop)
        same_action = str(event.get("action_name", "")) == target_action
        same_env = str(event.get("environment", "")) == target_env
        same_family = _action_family(str(event.get("action_name", ""))) == target_family
        exact_stratum = _exact_stratum_key(ev_intent) == target_stratum
        same_causal_stratum = build_causal_stratum(ev_intent, path_id=path_id) == target_causal_stratum
        activated = _path_activated_for_intent(ev_intent, path_spec, registry)
        path_status = path_treatment_status(ev_intent, path_id)
        hard_ratio = _hard_match_ratio(intent, ev_intent, hard_keys)
        soft_score = _soft_balance_score(intent, ev_intent, soft_keys)
        base_weight = 0.30 * sim + 0.20 * strat + 0.18 * max(0.0, 1.0 - prop_gap) + 0.20 * hard_ratio + 0.12 * soft_score
        if same_family: base_weight += 0.08
        if same_action: base_weight += 0.08
        if same_env: base_weight += 0.05
        if exact_stratum: base_weight += 0.12
        if same_causal_stratum: base_weight += 0.12
        is_treated = (path_status == "treated") if target_status != "other" else activated
        is_control = (path_status == "control") if target_status != "other" else (not activated)
        if is_treated and hard_ratio >= 0.50 and sim >= min_treat_similarity and strat >= 0.42 and prop_gap <= max(0.22, caliper + 0.06):
            candidates_t.append((min(1.0, base_weight), event, prop, strat))
        elif is_control:
            strict_ok = hard_ratio >= 0.67 and sim >= min_control_similarity and strat >= 0.34 and prop_gap <= caliper
            relaxed_ok = hard_ratio >= 0.50 and sim >= max(0.26, min_control_similarity - 0.06) and strat >= 0.28 and prop_gap <= min(0.22, caliper + 0.06)
            nn_ok = hard_ratio >= 0.34 and sim >= 0.22 and strat >= 0.24 and prop_gap <= 0.28
            if strict_ok:
                candidates_c.append((min(1.0, base_weight), event, prop, strat))
            elif relaxed_ok:
                relaxed_candidates_c.append((min(0.95, base_weight), event, prop, strat))
            elif nn_ok:
                nearest_candidates_c.append((min(0.90, base_weight), event, prop, strat))

    trimmed_t, overlap_t = _trim_overlap(candidates_t, target_prop, width=0.20, min_local_support=2)
    trimmed_c, overlap_c = _trim_overlap(candidates_c, target_prop, width=0.14, min_local_support=3)
    if trimmed_t:
        candidates_t = trimmed_t
    if trimmed_c:
        candidates_c = trimmed_c
    common_t, common_c, common_support = _common_support_trim(candidates_t, candidates_c, buffer=0.03)
    if common_t:
        candidates_t = common_t
    if common_c:
        candidates_c = common_c

    exact_controls = [x for x in candidates_c if _exact_stratum_key(_event_to_intent(x[1])) == target_stratum]
    exact_treated = [x for x in candidates_t if _exact_stratum_key(_event_to_intent(x[1])) == target_stratum]
    causal_controls = [x for x in candidates_c if build_causal_stratum(_event_to_intent(x[1]), path_id=path_id) == target_causal_stratum]
    if len(exact_controls) >= 2:
        candidates_c = exact_controls + [x for x in candidates_c if x not in exact_controls]
        fallback_level = "exact_stratum"
    elif len(causal_controls) >= 3:
        candidates_c = causal_controls + [x for x in candidates_c if x not in causal_controls]
        fallback_level = "causal_stratum"
    else:
        fallback_level = "global"

    adaptive_relax_used = False
    nn_fallback_used = False
    if len(candidates_c) < max(3, min(top_k_control, 4)):
        shared_mass = min(overlap_t.get("overlap_effective_mass", 0.0), overlap_c.get("overlap_effective_mass", 0.0))
        if shared_mass < 0.35 or len(candidates_c) < 2:
            adaptive_relax_used = bool(relaxed_candidates_c)
            candidates_c = candidates_c + [x for x in relaxed_candidates_c if x not in candidates_c]
            if len(candidates_c) < max(3, min(top_k_control, 4)) and nearest_candidates_c:
                nn_fallback_used = True
                candidates_c = candidates_c + [x for x in _nearest_neighbor_reweight(nearest_candidates_c, target_prop, top_k=max(top_k_control * 2, 8)) if x not in candidates_c]
                fallback_level = "nearest_neighbor" if fallback_level == "global" else f"{fallback_level}+nearest_neighbor"
            elif adaptive_relax_used and fallback_level == "global":
                fallback_level = "relaxed_soft_balance"

    if len(exact_treated) >= 2:
        candidates_t = exact_treated + [x for x in candidates_t if x not in exact_treated]

    candidates_t = _nearest_neighbor_reweight(candidates_t, target_prop, top_k=max(top_k_treat * 2, 8), hard_floor=0.45) if candidates_t else []
    candidates_c = _nearest_neighbor_reweight(candidates_c, target_prop, top_k=max(top_k_control * 2, 10), hard_floor=0.40) if candidates_c else []
    sel_t = candidates_t[:top_k_treat]
    sel_c = candidates_c[:top_k_control]
    weighted_t = [(e, round(max(0.05, s), 4)) for s, e, _p, _st in sel_t]
    weighted_c = [(e, round(max(0.05, s), 4)) for s, e, _p, _st in sel_c]
    treated_props = [p for _s, _e, p, _st in sel_t]
    control_props = [p for _s, _e, p, _st in sel_c]
    treated_strat = [st for _s, _e, _p, st in sel_t]
    control_strat = [st for _s, _e, _p, st in sel_c]
    exact_c = sum(1 for _s, e, _p, _st in sel_c if _exact_stratum_key(_event_to_intent(e)) == target_stratum)
    exact_t = sum(1 for _s, e, _p, _st in sel_t if _exact_stratum_key(_event_to_intent(e)) == target_stratum)
    causal_c = sum(1 for _s, e, _p, _st in sel_c if build_causal_stratum(_event_to_intent(e), path_id=path_id) == target_causal_stratum)
    causal_t = sum(1 for _s, e, _p, _st in sel_t if build_causal_stratum(_event_to_intent(e), path_id=path_id) == target_causal_stratum)
    avg_treated_prop = sum(treated_props) / max(1, len(treated_props)) if treated_props else target_prop
    avg_control_prop = sum(control_props) / max(1, len(control_props)) if control_props else target_prop
    prop_gap_mean = abs(avg_treated_prop - avg_control_prop)
    overlap_ok = bool(
        prop_gap_mean <= 0.14
        and overlap_t.get("overlap_hard_pass", False)
        and overlap_c.get("overlap_hard_pass", False)
        and common_support.get("common_support_ok", False)
    )
    quality = {
        "treated_match_quality": round(sum(w for _, w in weighted_t) / max(1, len(weighted_t)), 3) if weighted_t else 0.0,
        "control_match_quality": round(sum(w for _, w in weighted_c) / max(1, len(weighted_c)), 3) if weighted_c else 0.0,
        "treated_same_action": sum(1 for e, _w in weighted_t if str(e.get("action_name", "")) == target_action),
        "control_same_action": sum(1 for e, _w in weighted_c if str(e.get("action_name", "")) == target_action),
        "control_same_env": sum(1 for e, _w in weighted_c if str(e.get("environment", "")) == target_env),
        "control_same_family": sum(1 for e, _w in weighted_c if _action_family(str(e.get("action_name", ""))) == target_family),
        "propensity_target": round(target_prop, 3),
        "propensity_treated_mean": round(avg_treated_prop, 3),
        "propensity_control_mean": round(avg_control_prop, 3),
        "propensity_overlap_gap": round(prop_gap_mean, 3),
        "treated_stratum_mean": round(sum(treated_strat) / max(1, len(treated_strat)), 3) if treated_strat else 0.0,
        "control_stratum_mean": round(sum(control_strat) / max(1, len(control_strat)), 3) if control_strat else 0.0,
        "stratum_support": round(sum(control_strat) / max(1, len(control_strat)), 3) if control_strat else 0.0,
        "exact_stratum_treated": exact_t,
        "exact_stratum_control": exact_c,
        "causal_stratum_treated": causal_t,
        "causal_stratum_control": causal_c,
        "overlap_trimmed": bool(trimmed_t or trimmed_c),
        "overlap_ok": overlap_ok,
        "common_support_low": common_support["common_support_low"],
        "common_support_high": common_support["common_support_high"],
        "common_support_ratio": common_support["common_support_ratio"],
        "common_support_ok": common_support["common_support_ok"],
        "common_support_trim_treated": common_support["common_support_trim_treated"],
        "common_support_trim_control": common_support["common_support_trim_control"],
        "overlap_local_support": int(min(overlap_t.get("overlap_local_support", 0), overlap_c.get("overlap_local_support", 0))),
        "overlap_effective_mass": round(min(overlap_t.get("overlap_effective_mass", 0.0), overlap_c.get("overlap_effective_mass", 0.0)), 3),
        "effective_support_n": int(min(_effective_support(weighted_t), _effective_support(weighted_c))),
        "effective_sample_size": round(min(_effective_sample_size(weighted_t), _effective_sample_size(weighted_c)), 3),
        "dominant_control_weight": _dominant_control_weight(weighted_c),
        "fallback_level_used": fallback_level,
        "adaptive_relax_used": bool(adaptive_relax_used),
        "nearest_neighbor_fallback_used": bool(nn_fallback_used),
        "hard_match_keys": list(hard_keys),
        "soft_balance_keys": list(soft_keys),
    }
    quality["feature_smds"] = _feature_smds(weighted_t, weighted_c) if weighted_t and weighted_c else {}
    quality["balance_smd_mean"] = _mean_abs_smd(weighted_t, weighted_c) if weighted_t and weighted_c else 1.0
    quality["balance_ok"] = bool(quality["balance_smd_mean"] <= 0.12 and quality["dominant_control_weight"] <= 0.55)
    return weighted_t, weighted_c, quality





def _group_by_stratum(events_and_weights, path_id: str):
    grouped = {}
    for e, w in events_and_weights:
        key = build_causal_stratum(_event_to_intent(e), path_id=path_id)
        grouped.setdefault(key, []).append((e, w))
    return grouped


def _stratified_effects(treated, controls, harm: str, events, path_spec, registry, path_id: str):
    if not treated or not controls:
        return {
            "valid_shared_strata": 0,
            "shared_strata": [],
            "stratified_risk_delta": 0.0,
            "stratified_did_effect": 0.0,
            "shared_support_ratio": 0.0,
            "exact_shared_control": 0,
            "exact_shared_treated": 0,
        }
    tg = _group_by_stratum(treated, path_id)
    cg = _group_by_stratum(controls, path_id)
    shared = [k for k in tg.keys() if k in cg.keys()]
    if not shared:
        return {
            "valid_shared_strata": 0,
            "shared_strata": [],
            "stratified_risk_delta": 0.0,
            "stratified_did_effect": 0.0,
            "shared_support_ratio": 0.0,
            "exact_shared_control": 0,
            "exact_shared_treated": 0,
        }
    weighted_delta = []
    weighted_did = []
    exact_c = 0
    exact_t = 0
    total_shared_weight = 0.0
    for key in shared:
        tset = tg[key]
        cset = cg[key]
        t_rate = _weighted_harm_rate(tset, harm)
        c_rate = _weighted_harm_rate(cset, harm)
        t_pre = _weighted_mean((_historical_baseline_hazard(_event_to_intent(e), events, harm, path_spec, registry), w) for e, w in tset)
        c_pre = _weighted_mean((_historical_baseline_hazard(_event_to_intent(e), events, harm, path_spec, registry), w) for e, w in cset)
        delta = t_rate - c_rate
        did = (t_rate - t_pre) - (c_rate - c_pre)
        t_w = sum(max(0.0, w) for _e, w in tset)
        c_w = sum(max(0.0, w) for _e, w in cset)
        pair_w = min(t_w, c_w)
        total_shared_weight += pair_w
        weighted_delta.append((delta, pair_w))
        weighted_did.append((did, pair_w))
        exact_t += len(tset)
        exact_c += len(cset)
    treat_total = sum(max(0.0, w) for _e, w in treated)
    ctrl_total = sum(max(0.0, w) for _e, w in controls)
    denom = max(1e-6, min(treat_total, ctrl_total))
    return {
        "valid_shared_strata": len(shared),
        "shared_strata": shared,
        "stratified_risk_delta": round(_weighted_mean(weighted_delta), 3),
        "stratified_did_effect": round(_weighted_mean(weighted_did), 3),
        "shared_support_ratio": round(min(1.0, total_shared_weight / denom), 3),
        "exact_shared_control": exact_c,
        "exact_shared_treated": exact_t,
    }

def _identification_score(overlap_ok, balance_ok, placebo_pass, pretrend_ok, valid_strata, exact_stratum_control, causal_stratum_control, prop_gap, balance_smd, shared_support_ratio=0.0, design_strength="low", effective_support_n=0, dominant_control_weight=1.0, pretrend_testable=False, panel_pretrend_gap=0.0, panel_support=0):
    match_score = 0.0
    match_score += 0.24 if overlap_ok else 0.0
    match_score += 0.16 if balance_ok else max(0.0, 0.08 - balance_smd * 0.10)
    match_score += min(0.14, 0.03 * max(0, exact_stratum_control))
    match_score += min(0.08, 0.02 * max(0, causal_stratum_control))
    match_score += 0.12 * max(0.0, min(1.0, shared_support_ratio))
    match_score += min(0.10, 0.025 * max(0, effective_support_n))
    temporal_score = 0.18 if pretrend_ok and pretrend_testable else (0.06 if pretrend_testable else 0.0)
    falsification_score = 0.14 if placebo_pass else 0.0
    design_bonus = 0.10 if design_strength == "high" else (0.05 if design_strength == "medium" else 0.0)
    penalties = 0.0
    penalties += min(0.12, max(0.0, prop_gap - 0.10) * 0.7)
    penalties += 0.12 if dominant_control_weight > 0.55 else (0.05 if dominant_control_weight > 0.42 else 0.0)
    penalties += 0.10 if panel_pretrend_gap > 0.12 else (0.04 if panel_pretrend_gap > 0.08 else 0.0)
    penalties -= 0.04 if panel_support >= 3 else 0.0
    if not pretrend_testable:
        penalties += 0.12
    score = min(match_score, max(0.0, temporal_score + falsification_score + design_bonus + 0.12 * max(0.0, min(1.0, shared_support_ratio))))
    score -= penalties
    return round(max(0.0, min(1.0, score)), 3)


def _counterfactual_evidence_for_paths_impl(intent, paths, event_log_path: str | Path = "historical_action_events.jsonl", registry_path: str | Path = "action_registry.yaml", path_library_path: str | Path = "dangerous_paths.yaml"):
    events = _load_jsonl(event_log_path)
    if not paths or not events:
        return list(paths or [])
    registry = load_action_registry(str(registry_path)); library = load_path_library(str(path_library_path)).get("paths", {}) or {}; enriched = []
    for path in paths:
        path_id = str(path.get("path_id", "")); path_spec = dict(library.get(path_id, {}) or {}); harm = str(path.get("graph_harm") or "")
        design = treated_control_design(intent, path_id)
        contrast_key = str(design.get("contrast_key", "") or "")
        spec = get_path_treatment_spec(path_id, intent)
        plan = dict(path.get('validation_plan', {}) or {})
        excluded_match_keys = [contrast_key] if contrast_key else []
        treated, controls, quality = _build_matched_sets(intent, events, path_spec, registry, harm, path_id=path_id, path=path)
        n_t = _effective_support(treated); n_c = _effective_support(controls)
        treat_rate = _weighted_harm_rate(treated, harm); control_rate = _weighted_harm_rate(controls, harm)
        did_effect, treat_pre, ctrl_pre, treat_post, ctrl_post, raw_delta = _did_effect(treated, controls, harm, events, path_spec, registry)
        dr_risk_delta, dr_did_effect = _dr_effect(treated, controls, harm, events, path_spec, registry, exclude_keys=excluded_match_keys)
        treat_strata = summarize_strata(treated, path_id=path_id)
        control_strata = summarize_strata(controls, path_id=path_id)
        strat = _stratified_effects(treated, controls, harm, events, path_spec, registry, path_id)
        # prefer within-stratum estimates when shared strata exist
        if int(strat.get("valid_shared_strata", 0)) > 0:
            did_effect = round((did_effect + float(strat.get("stratified_did_effect", 0.0)) * 2.0) / 3.0, 3)
            raw_delta = round((raw_delta + float(strat.get("stratified_risk_delta", 0.0)) * 2.0) / 3.0, 3)
            dr_did_effect = round((dr_did_effect + float(strat.get("stratified_did_effect", 0.0))) / 2.0, 3)
            dr_risk_delta = round((dr_risk_delta + float(strat.get("stratified_risk_delta", 0.0))) / 2.0, 3)
        avg_strata_effect = round((did_effect + dr_did_effect) / 2.0, 3)
        valid_strata = int(max(strat.get("valid_shared_strata", 0), min(treat_strata.get("count", 0), control_strata.get("count", 0))))
        strata_consistency = round(max(float(treat_strata.get("consistency", 0.0)), float(control_strata.get("consistency", 0.0)), float(strat.get("shared_support_ratio", 0.0))), 3)
        strata_support = round(max(float(strat.get("shared_support_ratio", 0.0)), (float(treat_strata.get("support", 0.0)) + float(control_strata.get("support", 0.0))) / 2.0), 3)
        pretrend = _pretrend_assessment(treated, controls, harm, events, path_spec, registry)
        placebo = _placebo_stats(treated, controls, events, harm, path_spec, registry)
        balance_audit = _required_balance_audit(treated, controls, list(spec.get("required_confounders", []) or []))
        panel_stats = _longitudinal_panel_effect(treated, controls, harm, events, path_id)
        method_agreement = _method_agreement_stats(raw_delta, did_effect, dr_risk_delta, dr_did_effect, float(strat.get("stratified_risk_delta", 0.0)), float(strat.get("stratified_did_effect", 0.0)), float(panel_stats.get("panel_event_window_effect", 0.0)))
        design_strength = "high" if (
            int(strat.get("exact_shared_control", 0)) >= 2
            and bool(quality.get("overlap_ok", False))
            and bool(quality.get("balance_ok", False))
            and bool(balance_audit.get("required_balance_ok", False))
            and bool(method_agreement.get("method_agreement_ok", False))
        ) else ("medium" if int(strat.get("valid_shared_strata", 0)) >= 1 else ("medium" if quality["causal_stratum_control"] >= 1 else "low"))
        exact_control = max(int(quality.get("exact_stratum_control", 0)), int(strat.get("exact_shared_control", 0)))
        causal_control = max(int(quality.get("causal_stratum_control", 0)), int(strat.get("exact_shared_control", 0)))
        ident_assessment = build_identification_assessment(
            overlap_ok=bool(quality.get("overlap_ok", False)),
            balance_ok=bool(quality.get("balance_ok", False)) and bool(balance_audit.get("required_balance_ok", False)),
            placebo_pass=bool(placebo.get("placebo_pass", False)),
            pretrend_ok=bool(pretrend.get("counterfactual_pretrend_ok", False)),
            valid_strata=valid_strata,
            exact_stratum_control=exact_control,
            causal_stratum_control=causal_control,
            prop_gap=float(quality.get("propensity_overlap_gap", 1.0)),
            balance_smd=float(quality.get("balance_smd_mean", 1.0)),
            shared_support_ratio=strata_support,
            design_strength=design_strength,
            support_treated=n_t,
            support_control=n_c,
            placebo_fail_rate=float(placebo.get("placebo_fail_rate", 1.0)),
            control_match_quality=float(quality.get("control_match_quality", 0.0)),
            effect_for_sensitivity=dr_did_effect if dr_did_effect else did_effect,
            contrast_key=design["contrast_key"],
        )
        sensitivity = _sensitivity(
            effect=dr_did_effect if dr_did_effect else did_effect,
            prop_gap=float(quality.get("propensity_overlap_gap", 1.0)),
            balance_smd=max(float(quality.get("balance_smd_mean", 1.0)), float(balance_audit.get("required_balance_smd_mean", 1.0))),
            n_t=n_t,
            n_c=n_c,
            control_q=float(quality.get("control_match_quality", 0.0)),
            placebo_fail_rate=float(placebo.get("placebo_fail_rate", 1.0)),
            pretrend_ok=bool(pretrend.get("counterfactual_pretrend_ok", False)),
            stratum_support=strata_support,
            exact_stratum_control=max(int(quality.get("exact_stratum_control", 0)), int(strat.get("exact_shared_control", 0))),
            overlap_ok=bool(quality.get("overlap_ok", False)),
            dominant_control_weight=float(quality.get("dominant_control_weight", 1.0)),
            method_agreement_ratio=float(method_agreement.get("method_agreement_ratio", 0.0)),
            temporal_placebo_gap=float(placebo.get("temporal_placebo_gap", 0.0)),
        )
        ident_score = ident_assessment["identification_score"]
        detail_label = _counterfactual_label(
            n_t,
            n_c,
            treat_rate,
            control_rate,
            dr_did_effect if dr_did_effect else did_effect,
            float(quality.get("control_match_quality", 0.0)),
            float(quality.get("propensity_overlap_gap", 1.0)),
            float(quality.get("balance_smd_mean", 1.0)),
            float(placebo.get("placebo_fail_rate", 1.0)),
            bool(pretrend.get("counterfactual_pretrend_ok", False)),
            strata_support,
            max(int(quality.get("exact_stratum_control", 0)), int(strat.get("exact_shared_control", 0))),
            bool(quality.get("overlap_ok", False)),
            int(quality.get("effective_support_n", 0)),
            float(quality.get("dominant_control_weight", 1.0)),
            bool(pretrend.get("counterfactual_pretrend_testable", False)),
        )
        label = _legacy_counterfactual_tier(detail_label)
        ci_lo, ci_hi = _bootstrap_ci(treated, controls, harm, events, path_spec, registry)
        out = dict(path)
        out.update({
            # Backward-compatible public method label retained for older tests/consumers.
            # The richer panel-enhanced implementation is exposed separately below.
            "counterfactual_method": "matched_propensity_did_proxy_v3",
            "counterfactual_method_family": "matched_propensity_did_proxy_v4_panel",
            "counterfactual_treated_support": n_t,
            "counterfactual_control_support": n_c,
            "counterfactual_treated_harm_rate": round(treat_rate, 3),
            "counterfactual_control_harm_rate": round(control_rate, 3),
            "counterfactual_risk_delta": round(raw_delta, 3),
            "counterfactual_did_effect": round(did_effect, 3),
            "counterfactual_dr_risk_delta": round(dr_risk_delta, 3),
            "counterfactual_dr_did_effect": round(dr_did_effect, 3),
            "counterfactual_treated_pre_risk": treat_pre,
            "counterfactual_control_pre_risk": ctrl_pre,
            "counterfactual_treated_post_harm": treat_post,
            "counterfactual_control_post_harm": ctrl_post,
            "counterfactual_evidence_strength": label,
            "counterfactual_evidence_detail": detail_label,
            "counterfactual_control_match_quality": quality["control_match_quality"],
            "counterfactual_plan_covariates_used": list(plan.get("candidate_covariates", []) or []),
            "counterfactual_plan_post_treatment_columns": list(plan.get("post_treatment_columns", []) or []),
            "counterfactual_validation_design": str(plan.get("validation_design", "") or ""),
            "counterfactual_preferred_estimand": str(plan.get("preferred_estimand", "") or ""),
            "counterfactual_treated_match_quality": quality["treated_match_quality"],
            "counterfactual_control_same_action": quality["control_same_action"],
            "counterfactual_control_same_env": quality["control_same_env"],
            "counterfactual_control_same_family": quality["control_same_family"],
            "counterfactual_propensity_target": quality["propensity_target"],
            "counterfactual_propensity_overlap_gap": quality["propensity_overlap_gap"],
            "counterfactual_balance_smd_mean": quality["balance_smd_mean"],
            "counterfactual_balance_ok": quality["balance_ok"],
            "counterfactual_feature_smds": quality["feature_smds"],
            "counterfactual_required_balance_smd_mean": balance_audit["required_balance_smd_mean"],
            "counterfactual_required_balance_ok": balance_audit["required_balance_ok"],
            "counterfactual_required_balance_smds": balance_audit["required_balance_smds"],
            "counterfactual_common_support_low": quality["common_support_low"],
            "counterfactual_common_support_high": quality["common_support_high"],
            "counterfactual_common_support_ratio": quality["common_support_ratio"],
            "counterfactual_common_support_ok": quality["common_support_ok"],
            "counterfactual_common_support_trim_treated": quality["common_support_trim_treated"],
            "counterfactual_common_support_trim_control": quality["common_support_trim_control"],
            "counterfactual_method_estimates": method_agreement["method_estimates"],
            "counterfactual_method_agreement_ratio": method_agreement["method_agreement_ratio"],
            "counterfactual_method_effect_spread": method_agreement["method_effect_spread"],
            "counterfactual_method_mean_abs_effect": method_agreement["method_mean_abs_effect"],
            "counterfactual_method_agreement_ok": method_agreement["method_agreement_ok"],
            "counterfactual_panel_event_window_effect": panel_stats.get("panel_event_window_effect", 0.0),
            "counterfactual_panel_pretrend_gap": panel_stats.get("panel_pretrend_gap", 0.0),
            "counterfactual_panel_support": panel_stats.get("panel_support", 0),
            "counterfactual_panel_unit_support": panel_stats.get("panel_unit_support", 0),
            "counterfactual_panel_pre_avg": panel_stats.get("panel_pre_avg", 0.0),
            "counterfactual_panel_post_avg": panel_stats.get("panel_post_avg", 0.0),
            "counterfactual_panel_control_pre_avg": panel_stats.get("panel_control_pre_avg", 0.0),
            "counterfactual_panel_control_post_avg": panel_stats.get("panel_control_post_avg", 0.0),
            "counterfactual_panel_event_study_ok": bool(panel_stats.get("panel_event_study_ok", False)),
            "counterfactual_panel_effect_agreement": panel_stats.get("panel_effect_agreement", 0.0),
            "counterfactual_runtime_required_confounders": list(spec.get("required_confounders", []) or []),
            "counterfactual_runtime_mediators": list(spec.get("mediators", []) or []),
            "counterfactual_runtime_colliders": list(spec.get("colliders", []) or []),
            "counterfactual_runtime_forbidden_adjustments": list(spec.get("forbidden_adjustments", []) or []),
            "counterfactual_runtime_negative_controls": list(spec.get("negative_controls", []) or []),
            "counterfactual_runtime_path_hint_notes": str(spec.get("path_hint_notes", "") or ""),
            "counterfactual_stratum_support": quality["stratum_support"],
            "counterfactual_exact_stratum_treated": quality["exact_stratum_treated"],
            "counterfactual_exact_stratum_control": quality["exact_stratum_control"],
            "counterfactual_overlap_ok": quality["overlap_ok"],
            "counterfactual_overlap_trimmed": quality["overlap_trimmed"],
            "counterfactual_overlap_local_support": quality["overlap_local_support"],
            "counterfactual_overlap_effective_mass": quality["overlap_effective_mass"],
            "counterfactual_effective_support_n": quality["effective_support_n"],
            "counterfactual_effective_sample_size": quality["effective_sample_size"],
            "counterfactual_dominant_control_weight": quality["dominant_control_weight"],
            "counterfactual_fallback_level_used": quality["fallback_level_used"],
            "counterfactual_adaptive_relax_used": quality["adaptive_relax_used"],
            "counterfactual_nearest_neighbor_fallback_used": quality["nearest_neighbor_fallback_used"],
            "counterfactual_hard_match_keys": quality["hard_match_keys"],
            "counterfactual_soft_balance_keys": quality["soft_balance_keys"],
            "counterfactual_treated_stratum_mean": quality["treated_stratum_mean"],
            "counterfactual_control_stratum_mean": quality["control_stratum_mean"],
            "counterfactual_bootstrap_ci_low": ci_lo,
            "counterfactual_bootstrap_ci_high": ci_hi,
            "counterfactual_strata_count": int(max(treat_strata.get("count", 0), control_strata.get("count", 0))),
            "counterfactual_valid_strata": valid_strata,
            "counterfactual_avg_strata_effect": avg_strata_effect,
            "counterfactual_strata_consistency": strata_consistency,
            "counterfactual_strata_support": strata_support,
            "counterfactual_shared_strata": int(strat.get("valid_shared_strata", 0)),
            "counterfactual_shared_support_ratio": float(strat.get("shared_support_ratio", 0.0)),
            "counterfactual_stratified_risk_delta": float(strat.get("stratified_risk_delta", 0.0)),
            "counterfactual_stratified_did_effect": float(strat.get("stratified_did_effect", 0.0)),
            "treated_definition": design["treated_definition"],
            "control_definition": design["control_definition"],
            "within_stratum": design["within_stratum"],
            "counterfactual_stratum_key": design["stratum_key"],
            "counterfactual_causal_stratum_treated": quality["causal_stratum_treated"],
            "counterfactual_causal_stratum_control": quality["causal_stratum_control"],
            "counterfactual_contrast_key": design["contrast_key"],
            "counterfactual_contrast_treated_value": design["contrast_treated_value"],
            "counterfactual_contrast_control_value": design["contrast_control_value"],
            "counterfactual_treatment_var": design.get("treatment_var", design["contrast_key"]),
            "counterfactual_outcome_var": design.get("outcome_var", harm),
            "counterfactual_required_confounders": list(design.get("required_confounders", []) or []),
            "counterfactual_design_strength": design_strength,
        })
        out.update(pretrend); out.update(placebo); out.update(sensitivity); out.update(panel_stats)
        out["effect_estimate"] = {"risk_delta": round(raw_delta, 3), "did_effect": round(did_effect, 3), "dr_risk_delta": round(dr_risk_delta, 3), "dr_did_effect": round(dr_did_effect, 3), "panel_event_window_effect": round(float(panel_stats.get("panel_event_window_effect", 0.0)), 3), "treated_post_harm": treat_post, "control_post_harm": ctrl_post, "bootstrap_ci": [ci_lo, ci_hi], "method": "matched_propensity_did_proxy_v3_plus_panel"}
        out["identification_assessment"] = {
            **ident_assessment,
            "propensity_overlap_gap": quality["propensity_overlap_gap"],
            "balance_smd_mean": quality["balance_smd_mean"],
            "stratum_support": strata_support,
            "causal_strata_support": strata_support,
            "strata_consistency": strata_consistency,
            "required_balance_smd_mean": balance_audit["required_balance_smd_mean"],
            "required_balance_ok": balance_audit["required_balance_ok"],
            "common_support_ratio": quality["common_support_ratio"],
            "common_support_ok": quality["common_support_ok"],
            "method_agreement_ratio": method_agreement["method_agreement_ratio"],
            "method_agreement_ok": method_agreement["method_agreement_ok"],
            "method_effect_spread": method_agreement["method_effect_spread"],
            "shared_strata": int(strat.get("valid_shared_strata", 0)),
            "placebo_fail_rate": float(placebo["placebo_fail_rate"]),
            "panel_event_window_effect": float(panel_stats.get("panel_event_window_effect", 0.0)),
            "panel_pretrend_gap": float(panel_stats.get("panel_pretrend_gap", 0.0)),
            "panel_support": int(panel_stats.get("panel_support", 0)),
            "panel_event_study_ok": bool(panel_stats.get("panel_event_study_ok", False)),
        }
        score = float(out.get("risk_score", 0.0))
        if label == "high": score += 0.08
        elif label == "medium": score += 0.045
        elif label == "low": score += 0.015
        if not placebo.get("placebo_pass", True): score -= 0.03
        if float(quality.get("propensity_overlap_gap", 1.0)) > 0.20: score -= 0.02
        if not bool(quality.get("overlap_ok", False)): score -= 0.015
        if float(quality.get("balance_smd_mean", 1.0)) > 0.30: score -= 0.02
        if bool(panel_stats.get("panel_event_study_ok", False)): score += 0.02
        if float(panel_stats.get("panel_pretrend_gap", 0.0)) > 0.15: score -= 0.02
        if float(quality.get("stratum_support", 0.0)) < 0.45: score -= 0.015
        if int(quality.get("exact_stratum_control", 0)) < 1: score -= 0.015
        if int(quality.get("causal_stratum_control", 0)) < 1: score -= 0.015
        if strata_support >= 0.6 and valid_strata >= 1: score += 0.01
        if not bool(pretrend.get("counterfactual_pretrend_ok", False)): score -= 0.015
        if n_c == 0: score -= 0.03
        elif float(quality.get("control_match_quality", 0.0)) < 0.4: score -= 0.015
        out["risk_score"] = round(min(0.99, max(0.0, score)), 3)
        enriched.append(out)
    return sorted(enriched, key=lambda x: (-float(x.get("risk_score", 0.0)), x.get("path_id", "")))



def _plan_guided_stratum_keys(path: Dict[str, Any]) -> List[str]:
    plan = dict(path.get('validation_plan', {}) or {})
    plan_keys = [str(x).strip() for x in list(plan.get('candidate_stratum', []) or []) if str(x).strip()]
    graph_keys = [str(x).strip() for x in list(path.get('graph_preferred_stratum_keys', []) or []) if str(x).strip()]
    merged: List[str] = []
    for key in plan_keys + graph_keys:
        if key and key not in merged:
            merged.append(key)
    return merged

def _plan_negative_control(path: Dict[str, Any]) -> str:
    plan = dict(path.get('validation_plan', {}) or {})
    nc = str(plan.get('suggested_negative_control', '')).strip()
    if nc:
        return nc
    arr = list(path.get('graph_negative_controls', []) or [])
    if arr:
        return str(arr[0]).strip()
    spec = get_path_treatment_spec(str(path.get('path_id', '') or ''), {"action_name": path.get("action_name"), "environment": path.get("environment"), "params": dict(path.get("action_params", {}) or {})})
    arr = list(spec.get('negative_controls', []) or [])
    return str(arr[0]).strip() if arr else ''

def _enrich_paths_with_validation_guided_counterfactual_impl(intent: Dict[str, Any] | None, paths: List[Dict[str, Any]], event_log_path: str | Path = 'historical_action_events.jsonl', validation_plan_path: str | Path = 'out/validation_plan_level2.csv') -> List[Dict[str, Any]]:
    merged = []
    for p in list(paths or []):
        q = merge_validation_plan_into_path(dict(p or {}), validation_plan_path=validation_plan_path)
        q['counterfactual_used_validation_plan'] = bool(q.get('validation_plan'))
        q['counterfactual_runtime_stratum_keys'] = _plan_guided_stratum_keys(q)
        q['counterfactual_runtime_negative_control'] = _plan_negative_control(q)
        merged.append(q)

    event_count = _safe_event_count(event_log_path)
    max_events = int(_cfg('runtime_budget', 'max_runtime_events', default=160))
    max_paths = int(_cfg('runtime_budget', 'max_runtime_paths', default=12))

    if not _heavy_counterfactual_enabled():
        out = _counterfactual_fallback_paths(merged, 'heavy_counterfactual_disabled_for_runtime', event_count=event_count)
    elif event_count > max_events:
        out = _counterfactual_fallback_paths(merged, f'event_log_too_large_for_runtime:{event_count}>{max_events}', event_count=event_count)
    elif len(merged) > max_paths:
        out = _counterfactual_fallback_paths(merged, f'too_many_paths_for_runtime:{len(merged)}>{max_paths}', event_count=event_count)
    else:
        runtime_intent = dict(intent or {})
        try:
            out = counterfactual_evidence_for_paths(runtime_intent, merged, event_log_path=event_log_path) if 'counterfactual_evidence_for_paths' in globals() else merged
        except (OSError, ValueError, TypeError, RuntimeError, KeyError) as exc:
            out = _counterfactual_fallback_paths(merged, f'counterfactual_runtime_error:{type(exc).__name__}', event_count=event_count)

    final = []
    for p in out:
        q = merge_validation_plan_into_path(dict(p), validation_plan_path=validation_plan_path)
        q['counterfactual_used_validation_plan'] = bool(q.get('validation_plan'))
        q.setdefault('counterfactual_runtime_stratum_keys', _plan_guided_stratum_keys(q))
        q.setdefault('counterfactual_runtime_negative_control', _plan_negative_control(q))
        final.append(q)
    return final



def _coerce_action_intent(intent: Dict[str, Any] | ActionIntent | None) -> Dict[str, Any]:
    if intent is None:
        return {}
    if isinstance(intent, ActionIntent):
        return intent.to_dict()
    if isinstance(intent, dict):
        params = dict(intent.get("params", {}) or {})
        return {
            "action_name": intent.get("action_name"),
            "environment": intent.get("environment"),
            "params": params,
            **{k: v for k, v in intent.items() if k not in {"action_name", "environment", "params"}},
        }
    return {"action_name": str(intent), "params": {}}


def _coerce_path_list(paths: Sequence[Dict[str, Any] | PathTreatmentSpec] | None) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for path in list(paths or []):
        if isinstance(path, PathTreatmentSpec):
            out.append(path.to_path_dict())
        else:
            out.append(dict(path or {}))
    return out


class CounterfactualEstimator:
    """Thin orchestrator around the runtime counterfactual engine.

    It centralizes configuration and source paths, while keeping the existing
    functional internals intact for backwards compatibility and testability.
    """

    def __init__(
        self,
        event_log_path: str | Path = "historical_action_events.jsonl",
        registry_path: str | Path = "action_registry.yaml",
        path_library_path: str | Path = "dangerous_paths.yaml",
        validation_plan_path: str | Path = "out/validation_plan_level2.csv",
    ) -> None:
        self.event_log_path = Path(event_log_path)
        self.registry_path = Path(registry_path)
        self.path_library_path = Path(path_library_path)
        self.validation_plan_path = Path(validation_plan_path)

    def evaluate_paths(
        self,
        intent: Dict[str, Any] | ActionIntent | None,
        paths: Sequence[Dict[str, Any] | PathTreatmentSpec] | None,
    ) -> List[Dict[str, Any]]:
        runtime_intent = _coerce_action_intent(intent)
        runtime_paths = _coerce_path_list(paths)
        return _counterfactual_evidence_for_paths_impl(
            runtime_intent,
            runtime_paths,
            event_log_path=self.event_log_path,
            registry_path=self.registry_path,
            path_library_path=self.path_library_path,
        )

    def enrich_with_validation_plan(
        self,
        intent: Dict[str, Any] | ActionIntent | None,
        paths: Sequence[Dict[str, Any] | PathTreatmentSpec] | None,
    ) -> List[Dict[str, Any]]:
        runtime_intent = _coerce_action_intent(intent)
        runtime_paths = _coerce_path_list(paths)
        return _enrich_paths_with_validation_guided_counterfactual_impl(
            runtime_intent,
            runtime_paths,
            event_log_path=self.event_log_path,
            validation_plan_path=self.validation_plan_path,
        )


def counterfactual_evidence_for_paths(intent, paths, event_log_path: str | Path = "historical_action_events.jsonl", registry_path: str | Path = "action_registry.yaml", path_library_path: str | Path = "dangerous_paths.yaml"):
    estimator = CounterfactualEstimator(
        event_log_path=event_log_path,
        registry_path=registry_path,
        path_library_path=path_library_path,
    )
    return estimator.evaluate_paths(intent, paths)


def enrich_paths_with_validation_guided_counterfactual(intent: Dict[str, Any] | ActionIntent | None, paths: List[Dict[str, Any] | PathTreatmentSpec], event_log_path: str | Path = 'historical_action_events.jsonl', validation_plan_path: str | Path = 'out/validation_plan_level2.csv') -> List[Dict[str, Any]]:
    estimator = CounterfactualEstimator(
        event_log_path=event_log_path,
        validation_plan_path=validation_plan_path,
    )
    return estimator.enrich_with_validation_plan(intent, paths)
