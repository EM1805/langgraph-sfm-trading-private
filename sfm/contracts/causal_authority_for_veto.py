#!/usr/bin/env python3
"""Build causal authority cards for the runtime veto layer.

This module is the explicit bridge between the two graph families:

* operational_causal_graph.yaml: runtime/path-specific graph used by veto
* out/causal_contract.csv: offline PCMCI/SCM/identification/estimation handoff

It does not make runtime decisions.  It writes conservative authority cards that
say whether a runtime dangerous path has enough offline causal support to be
called a causal veto, or whether it should remain a policy/structural veto.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

AUTHORITY_CARD_VERSION = 2

IDENTIFIED_LEVELS = {
    "identified",
    "identified_estimable",
    "estimable",
    "scm_identified",
    "backdoor_identified",
    "frontdoor_identified",
}

WEAK_OR_PENDING_LEVELS = {
    "",
    "unknown",
    "pending",
    "discovery_only",
    "structural_only",
    "not_identified",
    "insufficient",
}


def _load_yaml_or_json(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml  # type: ignore
        payload = yaml.safe_load(text) or {}
        return payload if isinstance(payload, dict) else {}
    except (ImportError, ModuleNotFoundError, ValueError, TypeError) as exc:
        # Reuse the package's small YAML parser so this bridge stays dependency-light.
        from runtime.action_registry_v2 import _minimal_yaml_load
        payload = _minimal_yaml_load(text)
        return payload if isinstance(payload, dict) else {}


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, tuple):
        return [str(v).strip() for v in value if str(v).strip()]
    s = str(value).strip()
    if not s:
        return []
    if s.startswith("[") and s.endswith("]"):
        try:
            payload = json.loads(s.replace("'", '"'))
            return _as_list(payload)
        except (json.JSONDecodeError, TypeError, ValueError):
            s = s[1:-1]
    return [x.strip() for x in s.replace(";", ",").split(",") if x.strip()]


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def _node_aliases(graph: Dict[str, Any]) -> Dict[str, Set[str]]:
    aliases: Dict[str, Set[str]] = {}
    for node in graph.get("nodes", []) or []:
        if not isinstance(node, dict):
            continue
        node_id = str(node.get("id") or "").strip()
        if not node_id:
            continue
        vals = {node_id}
        vals.update(_as_list(node.get("aliases")))
        aliases[node_id] = {_norm(v) for v in vals if _norm(v)}
    return aliases


def _dangerous_paths_by_id(path_library: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    paths = path_library.get("paths", path_library)
    if not isinstance(paths, dict):
        return {}
    return {str(k): dict(v or {}) for k, v in paths.items() if isinstance(v, dict)}


def _read_contract(path: str | Path) -> List[Dict[str, str]]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _read_csv_rows(path: str | Path | None) -> List[Dict[str, str]]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _default_runtime_evidence_paths(causal_contract_path: str | Path) -> Dict[str, str]:
    contract = Path(causal_contract_path)
    out_dir = contract.parent if contract.parent.name else Path("out")
    candidates = {
        "effect_estimates": [out_dir / "effect_estimates.csv", out_dir / "estimation" / "effect_estimates.csv"],
        "sensitivity_analysis": [out_dir / "sensitivity_analysis.csv", out_dir / "estimation" / "sensitivity_analysis.csv"],
        "sensitivity_quantitative": [out_dir / "estimation" / "sensitivity_quantitative.csv"],
    }
    resolved: Dict[str, str] = {}
    for key, paths in candidates.items():
        for path in paths:
            if path.exists():
                resolved[key] = str(path)
                break
        else:
            resolved[key] = str(paths[0])
    return resolved


def _safe_float_or_none(value: Any) -> Optional[float]:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if x != x or x in {float("inf"), float("-inf")}:
        return None
    return x


def _first_nonempty(row: Dict[str, str], keys: Sequence[str]) -> str:
    for key in keys:
        val = str(row.get(key, "") or "").strip()
        if val and val.lower() not in {"nan", "none", "null", "nat"}:
            return val
    return ""


def _row_matches_contract(row: Dict[str, str], contract: Dict[str, str]) -> int:
    if not contract:
        return 0
    score = 0
    if _norm(row.get("insight_id")) and _norm(row.get("insight_id")) == _norm(contract.get("insight_id")):
        score += 8
    if _norm(row.get("source")) and _norm(row.get("source")) in {_norm(contract.get("source")), _norm(contract.get("treatment_col"))}:
        score += 2
    if _norm(row.get("treatment_col")) and _norm(row.get("treatment_col")) in {_norm(contract.get("source")), _norm(contract.get("treatment_col"))}:
        score += 2
    if _norm(row.get("target")) and _norm(row.get("target")) in {_norm(contract.get("target")), _norm(contract.get("outcome_col"))}:
        score += 2
    if _norm(row.get("outcome_col")) and _norm(row.get("outcome_col")) in {_norm(contract.get("target")), _norm(contract.get("outcome_col"))}:
        score += 2
    if _norm(row.get("lag")) and _norm(row.get("lag")) == _norm(contract.get("lag")):
        score += 1
    return score


def _best_evidence_row(rows: Sequence[Dict[str, str]], contract: Optional[Dict[str, str]]) -> Tuple[Optional[Dict[str, str]], int]:
    if not contract:
        return None, 0
    best: Optional[Dict[str, str]] = None
    best_score = 0
    for row in rows:
        score = _row_matches_contract(row, contract)
        if score > best_score:
            best = row
            best_score = score
    if best_score <= 0:
        return None, 0
    return best, best_score


def _effect_direction(effect_value: Optional[float], ci_low: Optional[float], ci_high: Optional[float]) -> str:
    if effect_value is None:
        return "unknown"
    if ci_low is not None and ci_high is not None:
        if ci_low > 0 and ci_high > 0:
            return "positive_supported"
        if ci_low < 0 and ci_high < 0:
            return "negative_supported"
        return "directional_only_ci_crosses_zero"
    if effect_value > 0:
        return "positive_directional"
    if effect_value < 0:
        return "negative_directional"
    return "near_zero"


def _runtime_effect_confidence(effect_row: Optional[Dict[str, str]], sensitivity_row: Optional[Dict[str, str]]) -> str:
    if not effect_row:
        return "not_available"
    effect = _safe_float_or_none(effect_row.get("effect_estimate"))
    ci_low = _safe_float_or_none(effect_row.get("ci_low"))
    ci_high = _safe_float_or_none(effect_row.get("ci_high"))
    if effect is None:
        return "not_available"
    robustness = _norm(effect_row.get("robustness_status"))
    est_status = _norm(effect_row.get("estimation_status"))
    sens_status = _norm(effect_row.get("sensitivity_status"))
    sens_risk = _norm((sensitivity_row or {}).get("unobserved_confounding_risk"))
    if ci_low is not None and ci_high is not None and ci_low <= 0 <= ci_high:
        return "directional_only_ci_crosses_zero"
    if "fail" in robustness or "blocked" in est_status or sens_risk == "high":
        return "low_due_to_robustness_or_sensitivity"
    if "pass" in robustness or robustness in {"stable", "robust"}:
        if sens_risk in {"", "low"} and sens_status in {"", "recommended_as_reporting_check", "not_applicable"}:
            return "moderate_estimation_support"
        return "limited_estimation_support"
    return "limited_estimation_support"


def _estimation_evidence_summary(
    contract: Optional[Dict[str, str]],
    effect_rows: Sequence[Dict[str, str]],
    sensitivity_rows: Sequence[Dict[str, str]],
) -> Dict[str, Any]:
    effect_row, effect_match_score = _best_evidence_row(effect_rows, contract)
    sensitivity_row, sensitivity_match_score = _best_evidence_row(sensitivity_rows, contract)
    effect = _safe_float_or_none((effect_row or {}).get("effect_estimate"))
    ci_low = _safe_float_or_none((effect_row or {}).get("ci_low"))
    ci_high = _safe_float_or_none((effect_row or {}).get("ci_high"))
    return {
        "estimation_evidence_available": bool(effect_row),
        "estimation_match_score": effect_match_score,
        "sensitivity_match_score": sensitivity_match_score,
        "effect_id": (effect_row or {}).get("effect_id", ""),
        "estimator_used": (effect_row or {}).get("estimator_used", ""),
        "effect_claim_status": (effect_row or {}).get("effect_claim_status", ""),
        "effect_estimate": effect,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_level": (effect_row or {}).get("ci_level", ""),
        "support_n": _safe_float_or_none((effect_row or {}).get("support_n")),
        "treated_n": _safe_float_or_none((effect_row or {}).get("treated_n")),
        "control_n": _safe_float_or_none((effect_row or {}).get("control_n")),
        "robustness_status": (effect_row or {}).get("robustness_status", ""),
        "estimation_status": (effect_row or {}).get("estimation_status", ""),
        "sensitivity_status": _first_nonempty(effect_row or {}, ["sensitivity_status"]) or (sensitivity_row or {}).get("sensitivity_status", ""),
        "sensitivity_quant_status": (effect_row or {}).get("sensitivity_quant_status", ""),
        "partial_r2_treatment": _safe_float_or_none((effect_row or {}).get("partial_r2_treatment")),
        "partial_r2_needed_to_explain_away": _safe_float_or_none((effect_row or {}).get("partial_r2_needed_to_explain_away")),
        "unobserved_confounding_risk": (sensitivity_row or {}).get("unobserved_confounding_risk", ""),
        "sensitivity_level": (sensitivity_row or {}).get("sensitivity_level", ""),
        "recommended_sensitivity_method": (sensitivity_row or {}).get("recommended_sensitivity_method", ""),
        "effect_direction": _effect_direction(effect, ci_low, ci_high),
        "runtime_effect_confidence": _runtime_effect_confidence(effect_row, sensitivity_row),
    }


def _values_for(node: str, alias_map: Dict[str, Set[str]]) -> Set[str]:
    vals = set(alias_map.get(node, set()))
    vals.add(_norm(node))
    return {v for v in vals if v}


def _contract_match_score(
    row: Dict[str, str],
    treatment_values: Set[str],
    outcome_values: Set[str],
    aggregate_outcome_values: Set[str],
) -> Tuple[int, List[str]]:
    row_sources = {_norm(row.get("source")), _norm(row.get("treatment_col"))}
    row_targets = {_norm(row.get("target")), _norm(row.get("outcome_col"))}
    row_sources.discard("")
    row_targets.discard("")
    reasons: List[str] = []
    score = 0
    if row_sources & treatment_values:
        score += 4
        reasons.append("treatment_alias_match")
    if row_targets & outcome_values:
        score += 4
        reasons.append("specific_outcome_alias_match")
    elif row_targets & aggregate_outcome_values:
        score += 2
        reasons.append("aggregate_harm_event_match")
    auth = _norm(row.get("authority_level") or row.get("identification_status"))
    identified = _norm(row.get("identified")) in {"1", "true", "yes", "y"}
    if identified or auth in IDENTIFIED_LEVELS:
        score += 2
        reasons.append("identified_or_estimable")
    return score, reasons


def _best_contract_row(
    contract_rows: Sequence[Dict[str, str]],
    treatment_node: str,
    outcome_node: str,
    alias_map: Dict[str, Set[str]],
) -> Tuple[Optional[Dict[str, str]], List[str], int]:
    treatment_values = _values_for(treatment_node, alias_map)
    outcome_values = _values_for(outcome_node, alias_map)
    aggregate_values = _values_for("harm_event", alias_map)
    best: Optional[Dict[str, str]] = None
    best_score = 0
    best_reasons: List[str] = []
    for row in contract_rows:
        score, reasons = _contract_match_score(row, treatment_values, outcome_values, aggregate_values)
        if score > best_score:
            best = row
            best_score = score
            best_reasons = reasons
    if best_score <= 0:
        return None, [], 0
    return best, best_reasons, best_score


def _authority_from_match(path_hint: Dict[str, Any], contract: Optional[Dict[str, str]], match_score: int) -> Tuple[str, str, str]:
    """Return (authority_level, runtime_use, reason)."""
    if not contract:
        return (
            "structural_runtime_only",
            "policy_or_structural_veto_only",
            "no_matching_offline_causal_contract_row",
        )
    auth = _norm(contract.get("authority_level") or contract.get("identification_status"))
    identified = _norm(contract.get("identified")) in {"1", "true", "yes", "y"}
    if (identified or auth in IDENTIFIED_LEVELS) and match_score >= 8:
        return (
            "identified_runtime_path",
            "causal_veto_allowed",
            "specific_runtime_path_matches_identified_offline_contract",
        )
    if (identified or auth in IDENTIFIED_LEVELS) and match_score >= 6:
        return (
            "identified_aggregate_harm_bridge",
            "causal_veto_allowed_with_aggregate_outcome_caution",
            "runtime_path_matches_identified_contract_through_harm_event_alias",
        )
    if auth and auth not in WEAK_OR_PENDING_LEVELS:
        return (
            "offline_support_weak_or_partial",
            "policy_veto_with_causal_context",
            "offline_contract_exists_but_is_not_strong_enough_for_causal_veto",
        )
    return (
        "offline_authority_pending",
        "policy_or_structural_veto_only",
        "offline_contract_match_exists_but_identification_or_estimation_is_pending",
    )


def build_causal_authority_cards(
    operational_graph_path: str | Path = "operational_causal_graph.yaml",
    path_library_path: str | Path = "dangerous_paths.yaml",
    causal_contract_path: str | Path = "out/causal_contract.csv",
    effect_estimates_path: str | Path | None = None,
    sensitivity_analysis_path: str | Path | None = None,
) -> List[Dict[str, Any]]:
    graph = _load_yaml_or_json(operational_graph_path)
    library = _load_yaml_or_json(path_library_path)
    contract_rows = _read_contract(causal_contract_path)
    evidence_defaults = _default_runtime_evidence_paths(causal_contract_path)
    effect_rows = _read_csv_rows(effect_estimates_path or evidence_defaults["effect_estimates"])
    sensitivity_rows = _read_csv_rows(sensitivity_analysis_path or evidence_defaults["sensitivity_analysis"])
    aliases = _node_aliases(graph)
    dangerous_paths = _dangerous_paths_by_id(library)
    path_hints = graph.get("path_hints", {}) or {}
    if not isinstance(path_hints, dict):
        path_hints = {}

    cards: List[Dict[str, Any]] = []
    for outcome_key, hint_obj in sorted(path_hints.items(), key=lambda kv: str(kv[0])):
        if not isinstance(hint_obj, dict):
            continue
        hint = dict(hint_obj)
        path_id = str(hint.get("path_id") or outcome_key)
        treatment = str(hint.get("treatment_node") or hint.get("contrast_key") or "")
        outcome = str(hint.get("outcome_node") or outcome_key)
        contract, match_reasons, match_score = _best_contract_row(contract_rows, treatment, outcome, aliases)
        authority_level, runtime_use, authority_reason = _authority_from_match(hint, contract, match_score)
        path_spec = dangerous_paths.get(path_id, {})
        estimation_evidence = _estimation_evidence_summary(contract, effect_rows, sensitivity_rows)
        card = {
            "authority_card_version": AUTHORITY_CARD_VERSION,
            "path_id": path_id,
            "runtime_outcome": outcome,
            "runtime_treatment": treatment,
            "contrast_key": hint.get("contrast_key", treatment),
            "treated_value": hint.get("treated_value", ""),
            "control_value": hint.get("control_value", ""),
            "graph_harm": path_spec.get("graph_harm", outcome),
            "severity": path_spec.get("severity", ""),
            "reversibility": path_spec.get("reversibility", ""),
            "adjustment_set": _as_list(hint.get("adjust_for")),
            "preferred_stratum_keys": _as_list(hint.get("preferred_stratum_keys")),
            "forbidden_adjustment_set": _as_list(hint.get("avoid")),
            "mediators": _as_list(hint.get("mediators")),
            "colliders": _as_list(hint.get("colliders")),
            "negative_controls": _as_list(hint.get("negative_controls")),
            "alternative_explanations": _as_list(hint.get("alternative_explanations")),
            "operational_graph_path": str(operational_graph_path),
            "causal_contract_path": str(causal_contract_path),
            "matched_contract": bool(contract),
            "contract_match_score": match_score,
            "contract_match_reasons": match_reasons,
            "contract_insight_id": (contract or {}).get("insight_id", ""),
            "contract_source": (contract or {}).get("source", ""),
            "contract_target": (contract or {}).get("target", ""),
            "contract_identification_status": (contract or {}).get("identification_status", ""),
            "contract_authority_level": (contract or {}).get("authority_level", ""),
            "contract_estimand_type": (contract or {}).get("estimand_type", ""),
            "contract_effect_scope": (contract or {}).get("effect_scope", ""),
            "estimation_evidence": estimation_evidence,
            "authority_level": authority_level,
            "runtime_use": runtime_use,
            "authority_reason": authority_reason,
            "notes": hint.get("notes", ""),
        }
        cards.append(card)
    return cards


def write_causal_authority_cards(
    operational_graph_path: str | Path = "operational_causal_graph.yaml",
    path_library_path: str | Path = "dangerous_paths.yaml",
    causal_contract_path: str | Path = "out/causal_contract.csv",
    out_jsonl: str | Path = "out/veto/causal_authority_cards.jsonl",
    out_summary: str | Path = "out/veto/causal_authority_summary.json",
    effect_estimates_path: str | Path | None = None,
    sensitivity_analysis_path: str | Path | None = None,
) -> Dict[str, Any]:
    evidence_defaults = _default_runtime_evidence_paths(causal_contract_path)
    resolved_effect_estimates_path = effect_estimates_path or evidence_defaults["effect_estimates"]
    resolved_sensitivity_analysis_path = sensitivity_analysis_path or evidence_defaults["sensitivity_analysis"]
    cards = build_causal_authority_cards(
        operational_graph_path=operational_graph_path,
        path_library_path=path_library_path,
        causal_contract_path=causal_contract_path,
        effect_estimates_path=resolved_effect_estimates_path,
        sensitivity_analysis_path=resolved_sensitivity_analysis_path,
    )
    jsonl_path = Path(out_jsonl)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as f:
        for card in cards:
            f.write(json.dumps(card, sort_keys=True) + "\n")
    counts: Dict[str, int] = {}
    uses: Dict[str, int] = {}
    for card in cards:
        counts[card["authority_level"]] = counts.get(card["authority_level"], 0) + 1
        uses[card["runtime_use"]] = uses.get(card["runtime_use"], 0) + 1
    summary = {
        "status": "ok",
        "authority_card_version": AUTHORITY_CARD_VERSION,
        "meaning": "Bridge from operational runtime paths to offline PCMCI/SCM/estimation causal contract.",
        "n_cards": len(cards),
        "authority_level_counts": counts,
        "runtime_use_counts": uses,
        "outputs": {
            "causal_authority_cards_jsonl": str(jsonl_path),
            "causal_authority_summary_json": str(out_summary),
        },
        "inputs": {
            "operational_graph_path": str(operational_graph_path),
            "path_library_path": str(path_library_path),
            "causal_contract_path": str(causal_contract_path),
            "effect_estimates_path": str(resolved_effect_estimates_path),
            "sensitivity_analysis_path": str(resolved_sensitivity_analysis_path),
        },
    }
    summary_path = Path(out_summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Write causal authority cards for the runtime veto layer")
    parser.add_argument("--graph", default="operational_causal_graph.yaml")
    parser.add_argument("--paths", default="dangerous_paths.yaml")
    parser.add_argument("--contract", default="out/causal_contract.csv")
    parser.add_argument("--effect-estimates", default=None, help="Optional effect_estimates.csv used to enrich runtime authority cards")
    parser.add_argument("--sensitivity-analysis", default=None, help="Optional sensitivity_analysis.csv used to enrich runtime authority cards")
    parser.add_argument("--out", default="out/veto/causal_authority_cards.jsonl")
    parser.add_argument("--summary-out", default="out/veto/causal_authority_summary.json")
    args = parser.parse_args(argv)
    summary = write_causal_authority_cards(
        operational_graph_path=args.graph,
        path_library_path=args.paths,
        causal_contract_path=args.contract,
        out_jsonl=args.out,
        out_summary=args.summary_out,
        effect_estimates_path=args.effect_estimates,
        sensitivity_analysis_path=args.sensitivity_analysis,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
