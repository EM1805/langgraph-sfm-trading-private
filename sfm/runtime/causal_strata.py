from __future__ import annotations

from pathlib import Path
import threading
from typing import Any, Dict, List, Tuple

try:
    import yaml  # type: ignore
except (ImportError, ModuleNotFoundError):  # pragma: no cover
    yaml = None

try:  # pragma: no cover - allow both package and top-level imports
    from .action_registry_v2 import get_action_spec, load_action_registry
    from .shared_utils import boolish
except (ImportError, ModuleNotFoundError):  # pragma: no cover
    from runtime.action_registry_v2 import get_action_spec, load_action_registry
    from runtime.shared_utils import boolish


_DAG_HINT_CACHE: Dict[str, Dict[str, Any]] | None = None
_ACTION_REGISTRY_CACHE: Dict[str, Any] | None = None
_CACHE_LOCK = threading.RLock()


CANONICAL_VALUE_MAPS: Dict[str, Dict[str, str]] = {
    "recipient_scope": {"outside": "external", "public": "external", "inside": "internal"},
    "share_scope": {"outside": "external", "public": "external", "inside": "internal"},
    "resource_sensitivity": {
        "pii": "high",
        "restricted": "high",
        "secret": "high",
        "confidential": "high",
        "internal": "medium",
        "private": "medium",
        "public": "low",
    },
    "blast_radius": {
        "very_high": "high",
        "critical": "high",
        "global": "high",
        "broad": "high",
        "medium_high": "medium",
        "moderate": "medium",
        "small": "low",
        "minor": "low",
    },
    "service_criticality": {
        "tier1": "high",
        "tier_1": "high",
        "critical": "high",
        "tier2": "medium",
        "tier_2": "medium",
        "important": "medium",
        "tier3": "low",
        "tier_3": "low",
        "non_critical": "low",
    },
    "environment": {
        "prod": "production",
        "production": "production",
        "staging": "staging",
        "stage": "staging",
        "dev": "dev",
        "development": "dev",
        "test": "test",
        "qa": "test",
    },
}

ACTION_TYPE_TO_FAMILY: Dict[str, str] = {
    "communication": "externalization",
    "sharing": "externalization",
    "mutation": "destructive_mutation",
    "ops": "configuration_change",
    "governance": "privilege_workflow",
    "admin": "privilege_workflow",
}


STRATUM_DISPLAY_KEY_ALIASES: Dict[str, str] = {
    "approval_present": "approval",
    "rollback_available": "rollback",
    "attachment_present": "attachment",
}

DEFAULT_STRATUM_KEYS: List[str] = [
    "action_family",
    "environment",
    "resource_sensitivity",
    "blast_radius",
    "approval_present",
    "rollback_available",
    "recipient_scope",
    "novel_action",
    "attachment_present",
    "service_criticality",
]


def _candidate_graph_paths(graph_path: str | Path = "operational_causal_graph.yaml") -> List[Path]:
    requested = Path(graph_path)
    candidates = [requested]
    here = Path(__file__).resolve().parent
    candidates.extend([
        here / requested.name,
        here.parent / requested.name,
        Path.cwd() / requested.name,
    ])
    uniq: List[Path] = []
    seen: set[str] = set()
    for p in candidates:
        key = str(p.resolve()) if p.exists() else str(p)
        if key not in seen:
            uniq.append(p)
            seen.add(key)
    return uniq


def _load_graph_path_hints(graph_path: str | Path = "operational_causal_graph.yaml") -> Dict[str, Dict[str, Any]]:
    global _DAG_HINT_CACHE
    with _CACHE_LOCK:
        if _DAG_HINT_CACHE is not None:
            return _DAG_HINT_CACHE
    try:
        if yaml is None:
            _DAG_HINT_CACHE = {}
            return _DAG_HINT_CACHE
        for path in _candidate_graph_paths(graph_path):
            if path.exists():
                spec = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
                _DAG_HINT_CACHE = dict(spec.get("path_hints", {}) or {})
                return _DAG_HINT_CACHE
        _DAG_HINT_CACHE = {}
        return _DAG_HINT_CACHE
    except (OSError, ValueError, TypeError, RuntimeError, KeyError):
        _DAG_HINT_CACHE = {}
        return _DAG_HINT_CACHE


def _load_action_registry_cached() -> Dict[str, Any]:
    global _ACTION_REGISTRY_CACHE
    if _ACTION_REGISTRY_CACHE is not None:
        return _ACTION_REGISTRY_CACHE
    for path in [
        Path(__file__).resolve().parent / "action_registry.yaml",
        Path(__file__).resolve().parent.parent / "action_registry.yaml",
        Path("action_registry.yaml"),
    ]:
        try:
            if path.exists():
                _ACTION_REGISTRY_CACHE = load_action_registry(path)
                return _ACTION_REGISTRY_CACHE
        except (OSError, ValueError, TypeError, RuntimeError, KeyError):
            continue
    _ACTION_REGISTRY_CACHE = {"actions": {}}
    return _ACTION_REGISTRY_CACHE


def _graph_hint_for_path(path_id: str, outcome_var: str = "") -> Dict[str, Any]:
    hints = _load_graph_path_hints()
    for harm, hint in hints.items():
        h = dict(hint or {})
        if str(h.get("path_id", "")) == str(path_id):
            if outcome_var and not h.get("outcome_node"):
                h["outcome_node"] = outcome_var
            return h
        if outcome_var and str(harm) == str(outcome_var):
            h.setdefault("path_id", path_id)
            h.setdefault("outcome_node", outcome_var)
            return h
    return {}



def canonicalize_stratum_value(key: str, value: Any) -> Any:
    if value is None or value == "":
        return "na"
    if key.endswith("_present") or key.endswith("_available") or key in {"novel_action"}:
        return 1 if boolish(value) else 0
    if isinstance(value, bool):
        return 1 if value else 0
    sval = str(value).strip().lower().replace(" ", "_")
    mapped = CANONICAL_VALUE_MAPS.get(key, {}).get(sval)
    if mapped is not None:
        return mapped
    return sval if sval else "na"


def _resolve_from_sources(intent: Dict[str, Any], key: str) -> Any:
    params = dict(intent.get("params", {}) or {})
    if key in params:
        return params.get(key)
    if key in intent:
        return intent.get(key)
    derived = dict(intent.get("derived_context", {}) or {})
    if key in derived:
        return derived.get(key)
    return None


def _action_spec_for_intent(intent: Dict[str, Any]) -> Dict[str, Any]:
    action_name = str(intent.get("action_name", "") or "").strip()
    if not action_name:
        return {}
    registry = _load_action_registry_cached()
    try:
        return dict(get_action_spec(action_name, registry) or {})
    except (OSError, ValueError, TypeError, RuntimeError, KeyError):
        return {}


def action_family(action_name: str, intent: Dict[str, Any] | None = None) -> str:
    action_name = str(action_name or "")
    intent = dict(intent or {})

    explicit_family = str(intent.get("action_family", "") or "").strip()
    if explicit_family:
        return explicit_family

    action_spec = _action_spec_for_intent({"action_name": action_name, **intent})
    action_type = str(action_spec.get("action_type", "") or action_spec.get("maps_to", {}).get("action_type", "")).strip().lower()
    if action_type in ACTION_TYPE_TO_FAMILY:
        return ACTION_TYPE_TO_FAMILY[action_type]

    name_lower = action_name.lower()
    if "email" in name_lower or "share_file" in name_lower or "share" in name_lower:
        return "externalization"
    if "delete" in name_lower or "overwrite" in name_lower or "remove" in name_lower:
        return "destructive_mutation"
    if "config" in name_lower or "deploy" in name_lower or "release" in name_lower:
        return "configuration_change"
    if "permission" in name_lower or "approval" in name_lower or "access" in name_lower:
        return "privilege_workflow"
    return action_name or "unknown"


PATH_TREATMENT_SPECS: Dict[str, Dict[str, Any]] = {
    "external_data_leakage": {
        "treatment_var": "recipient_scope",
        "treated_value": "external",
        "control_value": "internal",
        "outcome_var": "harm_leakage",
        "required_confounders": ["action_family", "environment", "resource_sensitivity", "approval_present", "attachment_present"],
        "hard_match_keys": ["action_family", "environment", "resource_sensitivity"],
        "soft_balance_keys": ["approval_present", "attachment_present"],
        "label": "external boundary",
    },
    "destructive_mutation": {
        "treatment_var": "rollback_available",
        "treated_value": False,
        "control_value": True,
        "outcome_var": "harm_data_loss",
        "required_confounders": ["action_family", "environment", "resource_sensitivity", "approval_present", "blast_radius"],
        "hard_match_keys": ["action_family", "environment", "resource_sensitivity"],
        "soft_balance_keys": ["approval_present", "blast_radius"],
        "label": "rollback availability",
    },
    "operational_failure": {
        "treatment_var": "novel_action",
        "treated_value": True,
        "control_value": False,
        "outcome_var": "harm_operational_failure",
        "required_confounders": ["action_family", "environment", "approval_present", "blast_radius", "service_criticality"],
        "hard_match_keys": ["action_family", "environment", "service_criticality"],
        "soft_balance_keys": ["approval_present", "blast_radius"],
        "label": "novel change",
    },
    "policy_bypass": {
        "treatment_var": "approval_present",
        "treated_value": False,
        "control_value": True,
        "outcome_var": "harm_policy_bypass",
        "required_confounders": ["action_family", "environment", "resource_sensitivity", "blast_radius"],
        "hard_match_keys": ["action_family", "environment", "resource_sensitivity"],
        "soft_balance_keys": ["blast_radius"],
        "label": "approval presence",
    },
    "privilege_escalation": {
        "treatment_var": "approval_present",
        "treated_value": False,
        "control_value": True,
        "outcome_var": "harm_unauthorized_access",
        "required_confounders": ["action_family", "environment", "resource_sensitivity", "blast_radius"],
        "hard_match_keys": ["action_family", "environment", "resource_sensitivity"],
        "soft_balance_keys": ["blast_radius"],
        "label": "approval presence",
    },
}


def get_path_treatment_spec(path_id: str, intent: Dict[str, Any] | None = None) -> Dict[str, Any]:
    intent = dict(intent or {})
    family = action_family(str(intent.get("action_name", "")), intent=intent)
    spec = dict(PATH_TREATMENT_SPECS.get(path_id, {}) or {})
    if not spec:
        spec = {
            "treatment_var": "approval_present" if family in {"privilege_workflow", "configuration_change"} else "rollback_available",
            "treated_value": False,
            "control_value": True,
            "outcome_var": "",
            "required_confounders": ["action_family", "environment", "resource_sensitivity", "blast_radius"],
            "hard_match_keys": ["action_family", "environment"],
            "soft_balance_keys": ["resource_sensitivity", "blast_radius"],
            "label": "safety control",
        }
    graph_hint = _graph_hint_for_path(path_id, outcome_var=str(spec.get("outcome_var", "") or ""))
    if graph_hint:
        spec["treatment_var"] = str(graph_hint.get("contrast_key") or graph_hint.get("treatment_node") or spec.get("treatment_var", ""))
        if "treated_value" in graph_hint:
            spec["treated_value"] = graph_hint.get("treated_value")
        if "control_value" in graph_hint:
            spec["control_value"] = graph_hint.get("control_value")
        spec["outcome_var"] = str(graph_hint.get("outcome_node") or spec.get("outcome_var", ""))
        path_adjust = [str(x) for x in (graph_hint.get("adjust_for", []) or graph_hint.get("preferred_stratum_keys", []) or []) if str(x).strip()]
        if path_adjust:
            spec["required_confounders"] = path_adjust
        spec["mediators"] = [str(x) for x in (graph_hint.get("mediators", []) or []) if str(x).strip()]
        spec["colliders"] = [str(x) for x in (graph_hint.get("colliders", []) or []) if str(x).strip()]
        spec["negative_controls"] = [str(x) for x in (graph_hint.get("negative_controls", []) or []) if str(x).strip()]
        spec["forbidden_adjustments"] = [str(x) for x in (graph_hint.get("avoid", []) or []) if str(x).strip()]
        spec["path_hint_notes"] = str(graph_hint.get("notes", "") or "")

    spec.setdefault("required_confounders", ["action_family", "environment"])
    spec.setdefault("hard_match_keys", list(spec.get("required_confounders", [])[:2] or ["action_family", "environment"]))
    remaining = [k for k in spec.get("required_confounders", []) if k not in set(spec.get("hard_match_keys", []) or [])]
    spec.setdefault("soft_balance_keys", remaining)
    spec.setdefault("mediators", [])
    spec.setdefault("colliders", [])
    spec.setdefault("negative_controls", [])
    spec.setdefault("forbidden_adjustments", [])
    spec["observed_treatment_value"] = _resolve_from_sources(intent, str(spec.get("treatment_var", "")))
    spec["action_family"] = family
    return spec


def event_to_intent(event: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "action_name": event.get("action_name"),
        "environment": event.get("environment", "unknown"),
        "action_family": event.get("action_family"),
        "params": dict(event.get("params", {}) or {}),
    }


def _contrast_spec(path_id: str, intent: Dict[str, Any]) -> Dict[str, Any]:
    spec = get_path_treatment_spec(path_id, intent)
    return {
        "contrast_key": spec.get("treatment_var", ""),
        "treated_value": spec.get("treated_value"),
        "control_value": spec.get("control_value"),
        "label": spec.get("label", "safety control"),
        "outcome_var": spec.get("outcome_var", ""),
        "required_confounders": list(spec.get("required_confounders", []) or []),
        "hard_match_keys": list(spec.get("hard_match_keys", []) or []),
        "soft_balance_keys": list(spec.get("soft_balance_keys", []) or []),
    }


def _value_eq(a: Any, b: Any, key: str = "") -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return boolish(a) == boolish(b)
    return canonicalize_stratum_value(key, a) == canonicalize_stratum_value(key, b)


def path_treatment_status(intent: Dict[str, Any], path_id: str) -> str:
    spec = _contrast_spec(path_id, intent)
    key = str(spec.get("contrast_key"))
    val = _resolve_from_sources(intent, key)
    if val is None or val == "":
        return "missing_treatment"
    if _value_eq(val, spec.get("treated_value"), key=key):
        return "treated"
    if _value_eq(val, spec.get("control_value"), key=key):
        return "control"
    return "outside_design"


def build_causal_stratum(intent: Dict[str, Any], path_id: str | None = None) -> str:
    keys = DEFAULT_STRATUM_KEYS
    if path_id:
        spec = get_path_treatment_spec(path_id, intent)
        keys = list(spec.get("required_confounders", []) or [])

    pieces: List[str] = []
    for key in keys:
        if key == "action_family":
            pieces.append(f"action_family={action_family(str(intent.get('action_name', '')), intent=intent)}")
            continue
        if key == "environment":
            pieces.append(f"environment={canonicalize_stratum_value('environment', intent.get('environment', 'unknown'))}")
            continue
        val = canonicalize_stratum_value(key, _resolve_from_sources(intent, key))
        display_key = STRATUM_DISPLAY_KEY_ALIASES.get(key, key)
        pieces.append(f"{display_key}={val}")
    return "|".join(pieces)


def treated_control_design(intent: Dict[str, Any], path_id: str) -> Dict[str, Any]:
    spec = get_path_treatment_spec(path_id, intent)
    family = action_family(str(intent.get("action_name", "")), intent=intent)
    contrast_key = str(spec.get("treatment_var", ""))
    treated_label = f"{family} with {contrast_key}={spec.get('treated_value')}"
    control_label = f"same family and hard-match stratum with {contrast_key}={spec.get('control_value')}"
    return {
        "treated_definition": treated_label,
        "control_definition": control_label,
        "within_stratum": True,
        "stratum_key": build_causal_stratum(intent, path_id=path_id),
        "contrast_key": contrast_key,
        "contrast_treated_value": spec.get("treated_value"),
        "contrast_control_value": spec.get("control_value"),
        "contrast_label": spec.get("label"),
        "treatment_var": contrast_key,
        "outcome_var": spec.get("outcome_var", ""),
        "required_confounders": list(spec.get("required_confounders", []) or []),
        "hard_match_keys": list(spec.get("hard_match_keys", []) or []),
        "soft_balance_keys": list(spec.get("soft_balance_keys", []) or []),
    }


def summarize_strata(events_and_weights: List[Tuple[Dict[str, Any], float]], path_id: str | None = None) -> Dict[str, float]:
    if not events_and_weights:
        return {
            "count": 0,
            "support": 0.0,
            "consistency": 0.0,
            "effective_n": 0.0,
            "top_stratum_share": 0.0,
            "shared_strata_ratio": 0.0,
        }

    totals: Dict[str, float] = {}
    treated: Dict[str, float] = {}
    control: Dict[str, float] = {}
    for event, w in events_and_weights:
        weight = max(0.0, float(w))
        intent = event_to_intent(event)
        key = build_causal_stratum(intent, path_id=path_id)
        totals[key] = totals.get(key, 0.0) + weight
        if path_id:
            status = path_treatment_status(intent, path_id)
            if status == "treated":
                treated[key] = treated.get(key, 0.0) + weight
            elif status == "control":
                control[key] = control.get(key, 0.0) + weight

    weights = sorted(totals.values(), reverse=True)
    total = sum(weights)
    concentration = weights[0] / total if total > 0 and weights else 0.0
    consistency = sum(w * w for w in weights) / (total * total) if total > 0 else 0.0
    effective_n = (total * total / sum(w * w for w in weights)) if weights and sum(w * w for w in weights) > 0 else 0.0

    treated_keys = set(treated)
    control_keys = set(control)
    shared_keys = treated_keys & control_keys
    denom = len(treated_keys | control_keys) or 1
    shared_ratio = len(shared_keys) / denom

    return {
        "count": float(len(weights)),
        "support": round(concentration, 3),
        "consistency": round(consistency, 3),
        "effective_n": round(effective_n, 3),
        "top_stratum_share": round(concentration, 3),
        "n_unique_strata_treated": float(len(treated_keys)),
        "n_unique_strata_control": float(len(control_keys)),
        "shared_strata_ratio": round(shared_ratio, 3),
    }
