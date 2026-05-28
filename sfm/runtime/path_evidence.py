
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List

from .runtime_calibration import PATH_EVIDENCE
from .shared_utils import boolish

def _event_from_flat_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a flat agent-action CSV row into the runtime event schema.

    The runtime counterfactual engine consumes JSONL events with top-level
    action metadata plus a nested params dictionary. Some package artifacts are
    stored as data/action_event_panel.csv instead, where params are flat columns.
    This adapter keeps the IO contract stable for both formats.
    """
    core = {
        "event_id", "event_time", "timestamp", "action_name", "environment",
        "observed_harms", "outcome", "actor_role", "review_type",
        "incident_detected", "harm_severity", "params",
    }
    params = {k: v for k, v in row.items() if k not in core and v not in (None, "")}
    observed = row.get("observed_harms", "")
    if isinstance(observed, str):
        observed_harms = [x.strip() for x in observed.replace("|", ",").split(",") if x.strip()]
    elif isinstance(observed, list):
        observed_harms = observed
    else:
        observed_harms = []
    return {
        "event_id": row.get("event_id", ""),
        "event_time": row.get("event_time") or row.get("timestamp"),
        "action_name": row.get("action_name", ""),
        "environment": row.get("environment", "unknown"),
        "params": params,
        "observed_harms": observed_harms,
        "outcome": row.get("outcome", ""),
        "actor_role": row.get("actor_role", ""),
        "review_type": row.get("review_type", ""),
        "incident_detected": boolish(row.get("incident_detected", False)),
        "harm_severity": row.get("harm_severity", ""),
    }


def _load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    out: List[Dict[str, Any]] = []
    if p.suffix.lower() == ".csv":
        import csv
        with p.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if isinstance(row, dict):
                    out.append(_event_from_flat_row(dict(row)))
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out

def _match_score(intent: Dict[str, Any], event: Dict[str, Any]) -> float:
    score = 0.0
    if str(intent.get("action_name", "")) == str(event.get("action_name", "")):
        score += 0.55
    if str(intent.get("environment", "")) == str(event.get("environment", "")):
        score += float(PATH_EVIDENCE["match"]["environment"])
    ip = dict(intent.get("params", {}) or {})
    ep = dict(event.get("params", {}) or {})
    for key in ("resource_sensitivity", "recipient_scope", "blast_radius"):
        if key in ip and key in ep and str(ip.get(key)) == str(ep.get(key)):
            score += float(PATH_EVIDENCE["match"]["categorical_key"])
    for key in ("attachment_present", "approval_present", "rollback_available", "novel_action"):
        if key in ip and key in ep and boolish(ip.get(key)) == boolish(ep.get(key)):
            score += float(PATH_EVIDENCE["match"]["boolean_key"])
    return score

def _evidence_label(support: int, harm_rate: float, delta: float) -> str:
    high = PATH_EVIDENCE["label_thresholds"]["high"]
    medium = PATH_EVIDENCE["label_thresholds"]["medium"]
    low = PATH_EVIDENCE["label_thresholds"]["low"]
    if support >= int(high["min_support"]) and harm_rate >= float(high["min_harm_rate"]) and delta >= float(high["min_delta"]):
        return "high"
    if support >= int(medium["min_support"]) and harm_rate >= float(medium["min_harm_rate"]) and delta >= float(medium["min_delta"]):
        return "medium"
    if support >= int(low["min_support"]) and harm_rate > float(low["min_harm_rate"]):
        return "low"
    return "none"

def empirical_evidence_for_paths(intent: Dict[str, Any], paths: List[Dict[str, Any]], event_log_path: str | Path = "historical_action_events.jsonl", min_similarity: float = float(PATH_EVIDENCE["default_min_similarity"])) -> List[Dict[str, Any]]:
    events = _load_jsonl(event_log_path)
    if not paths or not events:
        return list(paths or [])
    total = len(events)
    baseline_counts: Dict[str, int] = {}
    for p in paths:
        harm = str(p.get("graph_harm") or "")
        baseline_counts[harm] = sum(1 for e in events if harm and harm in list(e.get("observed_harms", []) or []))
    enriched: List[Dict[str, Any]] = []
    similar = [e for e in events if _match_score(intent, e) >= min_similarity]
    for p in paths:
        harm = str(p.get("graph_harm") or "")
        support = len(similar)
        harmful = sum(1 for e in similar if harm and harm in list(e.get("observed_harms", []) or []))
        safe = sum(1 for e in similar if not list(e.get("observed_harms", []) or []))
        harm_rate = harmful / support if support else 0.0
        baseline_rate = baseline_counts.get(harm, 0) / total if total else 0.0
        delta = harm_rate - baseline_rate
        label = _evidence_label(support, harm_rate, delta)
        out = dict(p)
        out["empirical_support"] = support
        out["empirical_harmful_support"] = harmful
        out["empirical_safe_support"] = safe
        out["empirical_harm_rate"] = round(harm_rate, 3)
        out["empirical_baseline_rate"] = round(baseline_rate, 3)
        out["empirical_risk_delta"] = round(delta, 3)
        out["empirical_evidence_strength"] = label
        score = float(out.get("risk_score", 0.0))
        score += float(PATH_EVIDENCE["risk_score_bonus"].get(label, 0.0))
        out["risk_score"] = round(min(0.99, score), 3)
        enriched.append(out)
    return sorted(enriched, key=lambda x: (-float(x.get("risk_score", 0.0)), x.get("path_id", "")))
