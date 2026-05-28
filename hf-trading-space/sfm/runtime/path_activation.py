from __future__ import annotations
from typing import Any, Dict, List, Optional
import warnings

from .causal_graph_runtime import OperationalCausalGraph
from .runtime_calibration import PATH_ACTIVATION

_SEVERITY_SCORE = dict(PATH_ACTIVATION["severity_score"])
_CONF_BONUS = dict(PATH_ACTIVATION["graph_confidence_bonus"])


def _load_graph(graph_path: Optional[str]) -> Optional[OperationalCausalGraph]:
    if not graph_path:
        return None
    try:
        return OperationalCausalGraph.load(graph_path)
    except (OSError, ValueError, TypeError, RuntimeError, KeyError) as exc:
        warnings.warn(
            f"[path_activation] Could not load operational graph '{graph_path}': {exc}. "
            "Continuing with library-only path activation.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None


def activate_paths(direct_effects: List[str], context_flags: Dict[str, Any], path_library: Dict[str, Any], graph_path: Optional[str] = None) -> List[Dict[str, Any]]:
    direct = set(direct_effects or [])
    flags = {str(k): bool(v) for k, v in (context_flags or {}).items()}
    graph = _load_graph(graph_path)
    graph_hits = graph.reachable_harms(sorted(direct)) if graph is not None else []
    harm_to_graph = {g["harm"]: g for g in graph_hits}
    out: List[Dict[str, Any]] = []
    for path_id, spec in (path_library.get("paths", {}) or {}).items():
        triggers_any = set(spec.get("triggers_any", []) or [])
        if triggers_any and not (direct & triggers_any):
            continue
        required_any = set(spec.get("required_context_any", []) or [])
        if required_any and not any(flags.get(x, False) for x in required_any):
            continue
        amplifiers = [x for x in (spec.get("amplifiers", []) or []) if flags.get(str(x), False)]
        hard_block_hits = [x for x in (spec.get("hard_block_if", []) or []) if flags.get(str(x), False)]
        severity = str(spec.get("severity", "medium") or "medium").lower()
        base_score = _SEVERITY_SCORE.get(severity, 0.55)
        graph_target = str(spec.get("graph_harm", "") or "").strip()
        graph_support = harm_to_graph.get(graph_target) if graph_target else None
        graph_conf = str((graph_support or {}).get("path_confidence", "unknown") or "unknown").lower()
        graph_nodes = list((graph_support or {}).get("path_nodes", []) or [])
        path_hint = dict((graph_support or {}).get("path_hint", {}) or {})
        graph_bonus = _CONF_BONUS.get(graph_conf, 0.0)
        if graph_support and any(x in graph_nodes for x in sorted(direct & triggers_any)):
            graph_bonus += float(PATH_ACTIVATION["trigger_graph_alignment_bonus"])
        risk_score = min(0.99, base_score + float(PATH_ACTIVATION["amplifier_increment"]) * len(amplifiers) + float(PATH_ACTIVATION["hard_block_increment"]) * len(hard_block_hits) + graph_bonus)
        evidence_strength = str(spec.get("default_evidence_strength", "structural") or "structural")
        if graph_support and graph_conf in {"medium", "high"}:
            evidence_strength = "structural_plus_graph"
        out.append({
            "path_id": str(path_id),
            "description": str(spec.get("description", "") or ""),
            "severity": severity,
            "reversibility": str(spec.get("reversibility", "unknown") or "unknown"),
            "evidence_strength": evidence_strength,
            "activated_by": sorted(direct & triggers_any) if triggers_any else [],
            "amplifiers": amplifiers,
            "hard_block_hits": hard_block_hits,
            "graph_harm": graph_target or None,
            "graph_supported": bool(graph_support),
            "graph_path_confidence": graph_conf,
            "graph_path_nodes": graph_nodes,
            "graph_contrast_key": path_hint.get("contrast_key", ""),
            "graph_treated_value": path_hint.get("treated_value"),
            "graph_control_value": path_hint.get("control_value"),
            "graph_preferred_stratum_keys": list(path_hint.get("preferred_stratum_keys", []) or []),
            "graph_adjust_for": list(path_hint.get("adjust_for", []) or []),
            "graph_avoid": list(path_hint.get("avoid", []) or []),
            "graph_negative_controls": list(path_hint.get("negative_controls", []) or []),
            "graph_alternative_explanations": list(path_hint.get("alternative_explanations", []) or []),
            "risk_score": round(risk_score, 3),
        })
    return sorted(out, key=lambda x: (-x["risk_score"], x["path_id"]))
