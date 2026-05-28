"""Canonical causal contract for PCMCI/Discovery -> SCM -> Estimation.

This module is intentionally lightweight and conservative.  It does not create
new causal claims; it only normalizes the artifacts already emitted by discovery,
SCM identification, and SCM fitting into a single table that downstream
estimation can consume consistently.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

CONTRACT_VERSION = 16

CONTRACT_COLUMNS: List[str] = [
    "insight_id",
    "source",
    "target",
    "treatment_col",
    "outcome_col",
    "lag",
    "edge_type",
    "discovery_track",
    "discovery_confidence_tier",
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
    "legacy_discovery_score",
    "legacy_pcmci_score",
    "mci_status",
    "mci_q_value",
    "mci_n_eff",
    "conditioning_set_used",
    "conditioning_set_size",
    "pc1_parent_support",
    "scm_role_hint",
    "temporal_consensus_score",
    "causal_plausibility_score",
    # SCM ID-algorithm authority fields. These are copied from
    # out/scm/id_algorithm_audit.csv and are treated as the strongest
    # identification gate when present.
    "id_status",
    "id_identified",
    "id_algorithm_level",
    "symbolic_formula_status",
    "symbolic_formula_kind",
    "symbolic_formula_json",
    "symbolic_formula_latex",
    "symbolic_sum_over",
    "symbolic_product_terms",
    "symbolic_removed_terms",
    "symbolic_unresolved_terms",
    "symbolic_evaluator_status",
    "symbolic_formula_evaluable",
    "symbolic_numeric_estimator_ready",
    "symbolic_estimator_route",
    "symbolic_estimator_family",
    "symbolic_effect_estimate_semantics",
    "symbolic_required_columns",
    "symbolic_evaluator_blocker",
    "symbolic_evaluator_reason_codes",
    "hedge_detected",
    "hedge_status",
    "recursive_id_status",
    "c_factor_status",
    "district_status",
    "id_block_reason",
    "id_reason_codes",
    "identification_status",
    "identified",
    "identification_strategy",
    "identification_route",
    "effect_scope",
    "estimand_type",
    "estimand_expression",
    "estimand_authority_status",
    "effect_claim_authority",
    "estimation_enabled",
    "allowed_for_estimation",
    "adjustment_set_status",
    "adjustment_set",
    "total_adjustment_set",
    "direct_adjustment_set",
    "candidate_adjustment_set",
    "backdoor_status",
    "blocked_by",
    "assumption_notes",
    "adjustment_set_source",
    "eligible_for_estimation",
    "unobserved_confounding_risk",
    "sensitivity_level",
    "sensitivity_status",
    "recommended_sensitivity_method",
    "forbidden_adjustment_set",
    "mediators",
    "frontdoor_status",
    "frontdoor_verification_level",
    "colliders",
    "negative_controls",
    "minimal_adjustment_sets",
    "minimal_direct_adjustment_sets",
    "structural_family",
    "structural_model_status",
    "structural_r2",
    "structural_model_role",
    "structural_fit_authority_level",
    "causal_authority_from_fit",
    "source_artifacts",
    "source_authority",
    "canonical_id_authority",
    "authority_level",
    "authority_reason",
]


def _read_csv(path: str | os.PathLike) -> List[Dict[str, str]]:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return []
    try:
        with p.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return []
            return [{k: ("" if v is None else str(v)) for k, v in row.items()} for row in reader]
    except (OSError, csv.Error, UnicodeDecodeError, ValueError, TypeError):
        return []


def _write_csv(path: str | os.PathLike, rows: Iterable[Dict[str, object]], columns: List[str]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})


def _first(row: Dict[str, str], keys: Iterable[str], default: str = "") -> str:
    for key in keys:
        value = str(row.get(key, "") or "").strip()
        if value and value.lower() not in {"nan", "none", "null"}:
            return value
    return default


def _to_key(row: Dict[str, str], fallback_prefix: str, index: int) -> str:
    iid = _first(row, ["insight_id", "candidate_id", "edge_id", "path_id"])
    if iid:
        return iid
    source = _first(row, ["source", "treatment_col", "from", "cause"])
    target = _first(row, ["target", "target_col", "outcome_col", "to", "effect"])
    lag = _first(row, ["lag", "tau", "time_lag"], "0")
    if source or target:
        return f"{source}->{target}@{lag}"
    return f"{fallback_prefix}_{index:05d}"


def _pair_key(row: Dict[str, str]) -> str:
    """Return a source->target key that intentionally ignores lag."""
    source = _first(row, ["source", "treatment_col", "from", "cause"])
    target = _first(row, ["target", "target_col", "outcome_col", "to", "effect"])
    if source or target:
        return f"{source}->{target}"
    return ""


def _best_existing_key(rows_by_key: Dict[str, Dict[str, str]], row: Dict[str, str], fallback_key: str) -> str:
    """Merge SCM identification rows back into bridge/discovery rows.

    Identification often drops PCMCI lag metadata because it works on a graph
    pair.  Without this helper, X->Y@0 identified rows can become disconnected
    from the original X->Y@lag discovery/bridge row.
    """
    iid = _first(row, ["insight_id", "candidate_id", "edge_id", "path_id"])
    if iid and iid in rows_by_key:
        return iid
    exact = _to_key(row, "candidate", 0)
    if exact in rows_by_key:
        return exact
    pair = _pair_key(row)
    if pair:
        matches = [key for key, existing in rows_by_key.items() if _pair_key(existing) == pair]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            def priority(k: str) -> int:
                auth = str(rows_by_key[k].get("source_authority", "")).lower()
                if "discovery_handoff_candidate" in auth:
                    return 0
                if "discovery_ranked_candidate" in auth:
                    return 1
                return 2
            return sorted(matches, key=priority)[0]
    return fallback_key


def _append_pipe(existing: str, value: str) -> str:
    parts = [x.strip() for x in str(existing or "").split("|") if x.strip()]
    for item in [x.strip() for x in str(value or "").split("|") if x.strip()]:
        if item not in parts:
            parts.append(item)
    return "|".join(parts)


def _merge_nonempty(base: Dict[str, str], updates: Dict[str, object], *, overwrite: bool = True) -> Dict[str, str]:
    for key, value in updates.items():
        if key not in CONTRACT_COLUMNS:
            continue
        text = "" if value is None else str(value)
        if not text or text.lower() in {"nan", "none", "null"}:
            continue
        if key in {"source_artifacts", "source_authority"}:
            base[key] = _append_pipe(base.get(key, ""), text)
            continue
        if overwrite or not str(base.get(key, "") or "").strip():
            base[key] = text
    return base


def _normalize_discovery_row(row: Dict[str, str]) -> Dict[str, str]:
    source = _first(row, ["source", "treatment_col", "from", "cause"])
    target = _first(row, ["target", "target_col", "outcome_col", "to", "effect"])
    return {
        "insight_id": _first(row, ["insight_id", "candidate_id", "edge_id"]),
        "source": source,
        "target": target,
        "treatment_col": _first(row, ["treatment_col", "source"], source),
        "outcome_col": _first(row, ["outcome_col", "target_col", "target"], target),
        "lag": _first(row, ["lag", "tau", "time_lag"], ""),
        "edge_type": _first(row, ["edge_type", "edge_family", "relationship_type"], "temporal_candidate"),
        "discovery_track": _first(row, ["discovery_track", "track"]),
        "discovery_confidence_tier": _first(row, ["discovery_confidence_tier", "confidence_tier", "confidence"]),
        # Canonical causal confidence must use namespaced signal/safety fields only.
        # Legacy generic scores are kept in explicit legacy_* audit fields.
        "hypothesis_signal_score": _first(row, ["hypothesis_signal_score"]),
        "hypothesis_signal_grade": _first(row, ["hypothesis_signal_grade"]),
        "hypothesis_signal_reason_codes": _first(row, ["hypothesis_signal_reason_codes"]),
        "safety_risk_score": _first(row, ["safety_risk_score"]),
        "safety_risk_grade": _first(row, ["safety_risk_grade"]),
        "safety_risk_reason_codes": _first(row, ["safety_risk_reason_codes"]),
        "safety_blocking": _first(row, ["safety_blocking"]),
        "signal_safety_cell": _first(row, ["signal_safety_cell"]),
        "signal_safety_policy": _first(row, ["signal_safety_policy"]),
        "signal_safety_matrix_track": _first(row, ["signal_safety_matrix_track"]),
        "signal_safety_blocking": _first(row, ["signal_safety_blocking"]),
        "signal_safety_reason_code": _first(row, ["signal_safety_reason_code"]),
        "signal_safety_matrix_version": _first(row, ["signal_safety_matrix_version"]),
        "legacy_discovery_score": _first(row, ["legacy_discovery_score", "discovery_score", "selection_score", "score", "strength", "discovery_evidence_score"]),
        "legacy_pcmci_score": _first(row, ["legacy_pcmci_score", "pcmci_score", "mci_score", "ci_score", "selection_score"]),
        "mci_status": _first(row, ["mci_status"]),
        "mci_q_value": _first(row, ["mci_q_value"]),
        "mci_n_eff": _first(row, ["mci_n_eff"]),
        "conditioning_set_used": _first(row, ["conditioning_set_used", "mci_conditioning_set_used", "candidate_covariates", "suggested_adjustment_set"]),
        "conditioning_set_size": _first(row, ["conditioning_set_size", "mci_conditioning_set_size"]),
        "pc1_parent_support": _first(row, ["pc1_parent_support", "pc1_parent_support_status", "pc1_is_selected_parent"]),
        "scm_role_hint": _first(row, ["scm_role_hint"]),
        "temporal_consensus_score": _first(row, ["temporal_consensus_score", "consensus_score"]),
        "causal_plausibility_score": _first(row, ["causal_plausibility_score", "plausibility_score"]),
        "adjustment_set": _first(row, ["suggested_adjustment_set", "candidate_adjustment_set", "candidate_covariates", "graph_adjustment_hint"]),
        "forbidden_adjustment_set": _first(row, ["forbidden_adjustment_set", "post_treatment_columns", "forbidden_variables"]),
        "negative_controls": _first(row, ["suggested_negative_control", "negative_controls", "negative_control"]),
        "estimand_type": _first(row, ["preferred_estimand", "estimand_type"]),
    }


def _normalize_identification_row(row: Dict[str, str]) -> Dict[str, str]:
    source = _first(row, ["treatment_col", "source"])
    target = _first(row, ["outcome_col", "target_col", "target"])
    status = _first(row, ["identification_status", "status"])
    identified = _first(row, ["identified", "is_identified"], "")
    if not status:
        status = "identified" if identified in {"1", "true", "True"} else "not_identified"

    # identifier.py is a legacy reporting mirror. When it carries canonical
    # id_algorithm fields, normalize them into the same contract fields used by
    # out/scm/id_algorithm_audit.csv so _id_algorithm_authority remains the
    # single gate for hard blocks and estimation authority.
    canonical_available = _truthy_text(_first(row, ["canonical_id_available"]))
    canonical_identified = _truthy_text(_first(row, ["canonical_id_identified"])) or _first(row, ["canonical_id_status"]).lower() == "identified"
    id_status = _first(row, ["id_status", "canonical_id_strategy"])
    id_identified = _first(row, ["id_identified"])
    if canonical_available:
        id_identified = "1" if canonical_identified else "0"
        if _first(row, ["canonical_id_status"]).lower() == "blocked":
            identified = "0"
            status = "blocked_by_id_algorithm"
        elif canonical_identified:
            identified = "1"
            status = "identified"

    return {
        "insight_id": _first(row, ["insight_id", "candidate_id", "edge_id"]),
        "source": source,
        "target": target,
        "treatment_col": source,
        "outcome_col": target,
        "id_status": id_status,
        "id_identified": id_identified,
        "id_algorithm_level": _first(row, ["id_algorithm_level", "canonical_id_level"]),
        "symbolic_formula_status": _first(row, ["symbolic_formula_status"]),
        "symbolic_formula_kind": _first(row, ["symbolic_formula_kind"]),
        "symbolic_formula_json": _first(row, ["symbolic_formula_json"]),
        "symbolic_formula_latex": _first(row, ["symbolic_formula_latex"]),
        "symbolic_sum_over": _first(row, ["symbolic_sum_over"]),
        "symbolic_product_terms": _first(row, ["symbolic_product_terms"]),
        "symbolic_removed_terms": _first(row, ["symbolic_removed_terms"]),
        "symbolic_unresolved_terms": _first(row, ["symbolic_unresolved_terms"]),
        "hedge_detected": _first(row, ["hedge_detected"]),
        "hedge_status": _first(row, ["hedge_status"]),
        "recursive_id_status": _first(row, ["recursive_id_status"]),
        "c_factor_status": _first(row, ["c_factor_status"]),
        "district_status": _first(row, ["district_status"]),
        "id_block_reason": _first(row, ["id_block_reason", "canonical_id_reason_codes"]),
        "id_reason_codes": _first(row, ["id_reason_codes", "canonical_id_reason_codes"]),
        "identification_status": status,
        "identified": identified,
        "identification_strategy": _first(row, ["identification_strategy", "strategy", "canonical_id_strategy"]),
        "identification_route": _first(row, ["identification_route", "route", "canonical_id_level"]),
        "effect_scope": _first(row, ["effect_scope"]),
        "estimand_type": _first(row, ["estimand_type", "preferred_estimand"]),
        "estimand_expression": _first(row, ["estimand_expression", "canonical_id_formula"]),
        "estimand_authority_status": _first(row, ["identification_vs_simulation", "estimand_authority_status"]),
        "effect_claim_authority": _first(row, ["effect_claim_authority", "frontdoor_authority", "cde_authority"]),
        "estimation_enabled": _first(row, ["estimation_enabled"]),
        "adjustment_set_status": _first(row, ["adjustment_set_status"]),
        "adjustment_set": _first(row, ["adjustment_set"]),
        "total_adjustment_set": _first(row, ["total_adjustment_set", "adjustment_set"]),
        "direct_adjustment_set": _first(row, ["direct_adjustment_set"]),
        "candidate_adjustment_set": _first(row, ["candidate_adjustment_set", "adjustment_set"]),
        "backdoor_status": _first(row, ["backdoor_status"]),
        "blocked_by": _first(row, ["blocked_by", "failed_assumptions"]),
        "assumption_notes": _first(row, ["assumption_notes", "notes"]),
        "adjustment_set_source": _first(row, ["adjustment_set_source"], "identified_effects"),
        "eligible_for_estimation": _first(row, ["eligible_for_estimation", "estimation_enabled"]),
        "forbidden_adjustment_set": _first(row, ["forbidden_adjustments", "forbidden_adjustment_set"]),
        "mediators": _first(row, ["mediators"]),
        "frontdoor_status": _first(row, ["frontdoor_status"]),
        "frontdoor_verification_level": _first(row, ["frontdoor_verification_level"]),
        "colliders": _first(row, ["colliders"]),
        "negative_controls": _first(row, ["negative_controls"]),
        "minimal_adjustment_sets": _first(row, ["minimal_adjustment_sets"]),
        "minimal_direct_adjustment_sets": _first(row, ["minimal_direct_adjustment_sets"]),
    }


def _normalize_id_algorithm_row(row: Dict[str, str]) -> Dict[str, str]:
    """Normalize scm/id_algorithm_audit.csv into causal_contract fields."""
    source = _first(row, ["treatment", "treatment_col", "source"])
    target = _first(row, ["outcome", "outcome_col", "target"])
    id_status = _first(row, ["id_strategy", "id_status", "identification_status"])
    identified = "1" if _truthy_text(_first(row, ["identifiable", "id_identified", "identified"])) else "0"
    symbolic_status = _first(row, ["symbolic_formula_status"])

    strategy = id_status
    if id_status == "backdoor_adjustment":
        strategy = "backdoor_adjustment"
    elif id_status == "frontdoor_limited":
        strategy = "frontdoor_limited"
    elif id_status == "observed_dag_truncated_factorization":
        strategy = "truncated_factorization"
    elif id_status == "no_directed_effect":
        strategy = "graphical_zero_effect"

    hedge_detected = "1" if _truthy_text(_first(row, ["possible_hedge", "hedge_detected"])) else "0"
    adjustment_set = _first(row, ["adjustment_set"])
    backdoor_status = _first(row, ["backdoor_status"])
    adjustment_status = ""
    if id_status == "backdoor_adjustment" and backdoor_status == "valid_backdoor_adjustment":
        adjustment_status = "valid_nonempty" if adjustment_set else "valid_empty"

    block_reason = _first(row, ["failure_reason", "reason_codes"])
    return {
        "insight_id": _first(row, ["insight_id", "candidate_id", "edge_id"]),
        "source": source,
        "target": target,
        "treatment_col": source,
        "outcome_col": target,
        "id_status": id_status,
        "id_identified": identified,
        "id_algorithm_level": _first(row, ["id_algorithm_level"]),
        "symbolic_formula_status": symbolic_status,
        "symbolic_formula_kind": _first(row, ["symbolic_formula_kind"]),
        "symbolic_formula_json": _first(row, ["symbolic_formula_json"]),
        "symbolic_formula_latex": _first(row, ["symbolic_formula_latex"]),
        "symbolic_sum_over": _first(row, ["symbolic_sum_over"]),
        "symbolic_product_terms": _first(row, ["symbolic_product_terms"]),
        "symbolic_removed_terms": _first(row, ["symbolic_removed_terms"]),
        "symbolic_unresolved_terms": _first(row, ["symbolic_unresolved_terms"]),
        "hedge_detected": hedge_detected,
        "hedge_status": _first(row, ["hedge_status"]),
        "recursive_id_status": _first(row, ["recursive_status", "recursive_id_status"]),
        "c_factor_status": _first(row, ["c_factor_status"]),
        "district_status": _first(row, ["district_status"]),
        "id_block_reason": block_reason,
        "id_reason_codes": _first(row, ["reason_codes"]),
        "identification_status": id_status,
        "identified": identified,
        "identification_strategy": strategy,
        "identification_route": _first(row, ["id_algorithm_level"]),
        "estimand_type": "total_effect",
        "estimand_expression": _first(row, ["estimand_formula", "symbolic_formula_latex"]),
        "adjustment_set_status": adjustment_status,
        "adjustment_set": adjustment_set,
        "total_adjustment_set": adjustment_set,
        "candidate_adjustment_set": adjustment_set,
        "backdoor_status": backdoor_status,
        "blocked_by": block_reason if id_status.startswith("blocked") or "blocked" in id_status else "",
        "assumption_notes": _first(row, ["reason_codes"]),
        "adjustment_set_source": "scm_id_algorithm",
        "eligible_for_estimation": identified,
        "mediators": _first(row, ["mediators", "frontdoor_active_mediators"]),
        "frontdoor_status": _first(row, ["frontdoor_status"]),
        "source_artifacts": "id_algorithm_audit",
        "source_authority": "scm_id_algorithm",
    }


def _normalize_symbolic_evaluation_row(row: Dict[str, str]) -> Dict[str, str]:
    """Normalize scm/symbolic_evaluation.csv into causal_contract fields."""
    source = _first(row, ["treatment", "treatment_col", "source"])
    target = _first(row, ["outcome", "outcome_col", "target"])
    return {
        "insight_id": _first(row, ["insight_id", "candidate_id", "edge_id"]),
        "source": source,
        "target": target,
        "treatment_col": source,
        "outcome_col": target,
        "symbolic_evaluator_status": _first(row, ["symbolic_evaluator_status"]),
        "symbolic_formula_evaluable": _first(row, ["formula_evaluable", "symbolic_formula_evaluable"]),
        "symbolic_numeric_estimator_ready": _first(row, ["numeric_estimator_ready", "symbolic_numeric_estimator_ready"]),
        "symbolic_estimator_route": _first(row, ["estimator_route", "symbolic_estimator_route"]),
        "symbolic_estimator_family": _first(row, ["estimator_family", "symbolic_estimator_family"]),
        "symbolic_effect_estimate_semantics": _first(row, ["effect_estimate_semantics", "symbolic_effect_estimate_semantics"]),
        "symbolic_required_columns": _first(row, ["required_columns", "symbolic_required_columns"]),
        "symbolic_evaluator_blocker": _first(row, ["blocker", "symbolic_evaluator_blocker"]),
        "symbolic_evaluator_reason_codes": _first(row, ["reason_codes", "symbolic_evaluator_reason_codes"]),
    }


def _structural_models_by_node(rows: List[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for row in rows:
        node = _first(row, ["node_id", "target", "outcome_col", "node"])
        if node:
            out[node] = row
    return out



def _parse_json_list(value: str) -> List[object]:
    text = str(value or "").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except (OSError, csv.Error, UnicodeDecodeError, ValueError, TypeError):
        return []
    return parsed if isinstance(parsed, list) else []


def _has_valid_minimal_set(value: str) -> bool:
    """Return True when identifier proved at least one minimal set.

    ``[[]]`` means the empty adjustment set is a valid proof; ``[]`` or blank
    means no valid set was found/reported.
    """
    parsed = _parse_json_list(value)
    return bool(parsed) and any(isinstance(item, list) for item in parsed)



def _truthy_text(value: object) -> bool:
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "y", "on", "identified", "identified_estimable"}:
        return True
    if text in {"0", "false", "no", "n", "off", "", "nan", "none", "null"}:
        return False
    try:
        return float(text) != 0.0
    except (TypeError, ValueError, OverflowError):
        return False


ID_SUPPORTED_SYMBOLIC_STATUSES = {"identified_symbolic_formula"}
ID_BACKDOOR_STATUSES = {"backdoor_adjustment", "identified_backdoor", "identified_backdoor_adjustment"}
ID_FRONTDOOR_STATUSES = {"frontdoor_limited", "identified_frontdoor_limited", "frontdoor_adjustment"}
ID_FACTOR_ONLY_STATUSES = {
    "observed_dag_truncated_factorization",
    "identified_observed_dag_base_case",
    "identified_observed_dag_after_recursive_reduction",
    "no_directed_effect",
}
ID_HARD_BLOCK_TOKENS = (
    "blocked",
    "unsupported_requires_full_id",
    "requires_full_id",
    "requires_symbolic_c_factor",
    "invalid_backdoor",
    "invalid_frontdoor",
    "directed_cycle",
)


def _has_id_algorithm_fields(row: Dict[str, str]) -> bool:
    return any(str(row.get(k, "") or "").strip() for k in [
        "id_status", "symbolic_formula_status", "hedge_status",
        "recursive_id_status", "c_factor_status", "district_status",
    ])


def _pipe_tokens(value: object) -> List[str]:
    return [tok.strip().lower() for tok in str(value or "").replace(",", "|").split("|") if tok.strip()]


def _has_canonical_id_authority(row: Dict[str, str]) -> bool:
    artifacts = set(_pipe_tokens(row.get("source_artifacts")))
    sources = set(_pipe_tokens(row.get("source_authority")))
    if "id_algorithm_audit" in artifacts or "scm_id_algorithm" in sources:
        return True
    if _truthy_text(row.get("canonical_id_available")):
        return True
    if str(row.get("id_algorithm_level", "") or "").strip():
        return True
    return False


def _id_algorithm_block_reasons(row: Dict[str, str]) -> List[str]:
    """Return hard-block reason codes from the SCM ID algorithm audit."""
    reasons: List[str] = []
    id_status = str(row.get("id_status", "") or row.get("identification_status", "")).strip().lower()
    symbolic = str(row.get("symbolic_formula_status", "") or "").strip().lower()
    recursive = str(row.get("recursive_id_status", "") or "").strip().lower()
    cfactor = str(row.get("c_factor_status", "") or "").strip().lower()
    district = str(row.get("district_status", "") or "").strip().lower()
    hedge_status = str(row.get("hedge_status", "") or "").strip().lower()
    if any(tok in id_status for tok in ID_HARD_BLOCK_TOKENS):
        reasons.append(f"ID_STATUS_{id_status.upper()}")
    if symbolic and symbolic not in ID_SUPPORTED_SYMBOLIC_STATUSES:
        reasons.append(f"SYMBOLIC_FORMULA_{symbolic.upper()}")
    if _truthy_text(row.get("hedge_detected")) or hedge_status.startswith("possible_hedge") or "possible_hedge" in hedge_status:
        reasons.append("ID_HEDGE_DETECTED")
    if recursive.startswith("blocked") or "requires_symbolic_c_factor" in recursive:
        reasons.append(f"RECURSIVE_ID_{recursive.upper()}")
    if "unresolved" in cfactor or "requires_recursive" in cfactor:
        reasons.append(f"C_FACTOR_{cfactor.upper()}")
    if "possible_hedge" in district:
        reasons.append(f"DISTRICT_{district.upper()}")
    return reasons


def _id_algorithm_authority(row: Dict[str, str]) -> Optional[Tuple[str, str]]:
    """Conservative ID-audit override for causal contract authority."""
    if not _has_id_algorithm_fields(row):
        return None

    id_status = str(row.get("id_status", "") or row.get("identification_status", "")).strip().lower()
    symbolic = str(row.get("symbolic_formula_status", "") or "").strip().lower()
    identified = _truthy_text(row.get("id_identified", row.get("identified")))
    blocks = _id_algorithm_block_reasons(row)
    if blocks:
        row["estimation_enabled"] = "0"
        row["identified"] = "0"
        row["id_block_reason"] = _append_pipe(row.get("id_block_reason", ""), "|".join(blocks))
        row["blocked_by"] = _append_pipe(row.get("blocked_by", ""), "|".join(blocks))
        row["estimand_authority_status"] = row.get("estimand_authority_status") or "blocked_by_id_algorithm"
        row["effect_claim_authority"] = row.get("effect_claim_authority") or "no_effect_claim_id_algorithm_blocked"
        return "blocked_id_algorithm", "id_algorithm_hard_block:" + "|".join(blocks)

    if not identified:
        row["estimation_enabled"] = "0"
        return "graph_review", "id_algorithm_not_identified_but_not_hard_blocked"

    if symbolic and symbolic not in ID_SUPPORTED_SYMBOLIC_STATUSES:
        row["estimation_enabled"] = "0"
        return "graph_review", f"id_algorithm_symbolic_formula_status={symbolic}"

    if id_status in ID_BACKDOOR_STATUSES:
        row["adjustment_set_status"] = _adjustment_set_status(row)
        if row["adjustment_set_status"] not in {"valid_empty", "valid_nonempty"}:
            row["estimation_enabled"] = "0"
            return "identified_needs_estimation", "id_algorithm_backdoor_identified_but_adjustment_status_missing"
        row["estimation_enabled"] = "1"
        row["estimand_authority_status"] = row.get("estimand_authority_status") or "id_algorithm_backdoor_symbolic_formula_verified"
        row["effect_claim_authority"] = row.get("effect_claim_authority") or "formal_backdoor_identification_from_id_algorithm"
        return "identified_estimable", "id_algorithm_backdoor_verified_symbolic_formula"

    if id_status in ID_FRONTDOOR_STATUSES:
        frontdoor_status = str(row.get("frontdoor_status", "") or "").strip().lower()
        if frontdoor_status not in {"valid_limited_frontdoor", "frontdoor_valid", "valid_frontdoor", "frontdoor_mediators_valid"}:
            row["estimation_enabled"] = "0"
            return "identified_needs_estimation", "id_algorithm_frontdoor_identified_but_frontdoor_status_not_ready"
        if not str(row.get("mediators", "") or "").strip():
            row["estimation_enabled"] = "0"
            return "identified_needs_estimation", "id_algorithm_frontdoor_identified_but_mediators_missing"
        row["estimation_enabled"] = "1"
        row["estimand_authority_status"] = row.get("estimand_authority_status") or "id_algorithm_frontdoor_limited_symbolic_formula_verified"
        row["effect_claim_authority"] = row.get("effect_claim_authority") or "frontdoor_limited_estimator_ready_from_id_algorithm"
        return "identified_estimable", "id_algorithm_frontdoor_limited_verified_symbolic_formula"

    if id_status in ID_FACTOR_ONLY_STATUSES:
        row["estimation_enabled"] = "0"
        evaluator_status = str(row.get("symbolic_evaluator_status", "") or "").strip().lower()
        formula_evaluable = _truthy_text(row.get("symbolic_formula_evaluable"))
        numeric_ready = _truthy_text(row.get("symbolic_numeric_estimator_ready"))
        if numeric_ready:
            row["estimand_authority_status"] = row.get("estimand_authority_status") or "id_algorithm_formula_symbolic_evaluator_numeric_ready"
            row["effect_claim_authority"] = row.get("effect_claim_authority") or "symbolic_formula_numeric_route_available_contract_gated"
            row["estimation_enabled"] = "1"
            return "identified_estimable", f"id_algorithm_symbolic_evaluator_ready_route={id_status}"
        if formula_evaluable:
            row["estimand_authority_status"] = row.get("estimand_authority_status") or "id_algorithm_formula_symbolically_evaluable_estimator_not_connected"
            row["effect_claim_authority"] = row.get("effect_claim_authority") or "symbolic_formula_evaluable_no_numeric_estimator_yet"
            return "identified_needs_estimation", f"id_algorithm_symbolic_formula_evaluable_route={id_status};evaluator={evaluator_status or 'missing'}"
        row["estimand_authority_status"] = row.get("estimand_authority_status") or "id_algorithm_formula_identified_estimator_not_connected"
        row["effect_claim_authority"] = row.get("effect_claim_authority") or "symbolic_formula_identified_no_strong_do_estimator_yet"
        return "identified_needs_estimation", f"id_algorithm_identified_formula_route={id_status}"

    row["estimation_enabled"] = "0"
    return "identified_needs_estimation", f"id_algorithm_identified_unsupported_estimator_route={id_status or 'unknown'}"

def _adjustment_set_status(row: Dict[str, str]) -> str:
    explicit = str(row.get("adjustment_set_status", "") or "").strip().lower()
    if explicit in {"valid_empty", "valid_nonempty", "missing", "invalid"}:
        return explicit
    adj = str(row.get("adjustment_set", "") or row.get("total_adjustment_set", "") or "").strip()
    if adj:
        return "valid_nonempty"
    if _has_valid_minimal_set(str(row.get("minimal_adjustment_sets", "") or "")):
        return "valid_empty"
    bd_flag = str(row.get("backdoor_identifiable", row.get("dsep_backdoor_identifiable", "")) or "").strip().lower()
    if bd_flag in {"1", "true", "yes"}:
        return "valid_empty"
    return "missing"

def _authority(row: Dict[str, str]) -> Tuple[str, str]:
    identified = str(row.get("identified", "")).strip().lower() in {"1", "true", "yes"}
    status = str(row.get("identification_status", "")).strip().lower()
    adj = str(row.get("adjustment_set", "") or row.get("total_adjustment_set", "")).strip()
    adj_status = _adjustment_set_status(row)
    row["adjustment_set_status"] = adj_status
    adjustment_valid = adj_status in {"valid_empty", "valid_nonempty"}
    scm_status = str(row.get("structural_model_status", "")).strip().lower()
    conf = str(row.get("discovery_confidence_tier", "")).strip().lower()
    source_authority = str(row.get("source_authority", "")).strip().lower()
    raw_bits = [x for x in source_authority.split("|") if x]
    non_fit_bits = [x for x in raw_bits if x not in {"scm_fit", "scm_fit_diagnostic_only"}]
    if (not identified) and non_fit_bits and all(x in {"raw_discovery_only", "pcmci_raw_input_only"} for x in non_fit_bits):
        return "raw_discovery_only", "raw_edges_or_pcmci_links_are_seed_only_even_if_structural_fit_exists"
    id_override = _id_algorithm_authority(row)
    canonical_id_authority = _has_canonical_id_authority(row)
    row["canonical_id_authority"] = "1" if canonical_id_authority else "0"
    if id_override is not None:
        return id_override
    if identified and not canonical_id_authority:
        row["estimation_enabled"] = "0"
        row["estimand_authority_status"] = row.get("estimand_authority_status") or "identified_legacy_reporting_missing_canonical_id_authority"
        row["effect_claim_authority"] = row.get("effect_claim_authority") or "no_numeric_do_without_canonical_id_authority"
        row["blocked_by"] = _append_pipe(row.get("blocked_by", ""), "MISSING_CANONICAL_ID_AUTHORITY")
        return "identified_needs_estimation", "missing_canonical_id_authority_for_numeric_estimation"

    strategy = str(row.get("identification_strategy", "")).strip().lower()
    route = str(row.get("identification_route", "")).strip().lower()
    estimand = str(row.get("estimand_type", row.get("effect_scope", ""))).strip().lower()
    joined = " ".join([strategy, route, estimand])
    backdoor_like = ("backdoor" in joined) or (estimand in {"total_effect", "effect", ""} and adjustment_valid)
    frontdoor_like = "frontdoor" in joined
    frontdoor_status = str(row.get("frontdoor_status", "") or "").strip().lower()
    frontdoor_ready = frontdoor_like and (frontdoor_status in {"frontdoor_valid", "valid_frontdoor", "frontdoor_mediators_valid"} or str(row.get("effect_claim_authority", "")).strip().lower() == "frontdoor_limited_estimator_ready") and bool(str(row.get("mediators", "") or "").strip())
    path_specific = any(tok in joined for tok in ["controlled_direct", "natural_direct", "natural_indirect", "cde", "nde", "nie"])
    if identified and backdoor_like and adjustment_valid and scm_status not in {"failed", "error"}:
        row["estimation_enabled"] = "1"
        row["estimand_authority_status"] = row.get("estimand_authority_status") or "identified_backdoor_estimation_enabled"
        row["effect_claim_authority"] = row.get("effect_claim_authority") or "formal_backdoor_identification"
        return "identified_estimable", "backdoor_identification_and_valid_adjustment_status"
    if identified and frontdoor_ready and scm_status not in {"failed", "error"}:
        row["estimation_enabled"] = "1"
        row["estimand_authority_status"] = row.get("estimand_authority_status") or "identified_frontdoor_estimation_enabled_limited"
        row["effect_claim_authority"] = row.get("effect_claim_authority") or "frontdoor_limited_estimator_ready"
        if not row.get("frontdoor_status"):
            row["frontdoor_status"] = "frontdoor_valid"
        return "identified_estimable", "frontdoor_limited_identification_and_estimator_enabled"
    if identified and (frontdoor_like or path_specific):
        row["estimation_enabled"] = "0"
        row["estimand_authority_status"] = row.get("estimand_authority_status") or "identified_but_estimator_not_enabled"
        row["effect_claim_authority"] = row.get("effect_claim_authority") or "graphical_lite_or_path_specific_not_estimation_authority"
        return "identified_needs_estimation", "identified_graphical_route_but_no_enabled_estimator"
    if identified:
        row["estimation_enabled"] = "0"
        row["estimand_authority_status"] = row.get("estimand_authority_status") or "identified_needs_estimator"
        return "identified_needs_estimation", "graph_identification_available_but_adjustment_or_model_incomplete"
    if status and status not in {"not_identified", "unidentified", "simulable_not_identified"}:
        row["estimation_enabled"] = "0"
        return "graph_review", f"identification_status={status}"
    if conf in {"high", "medium"}:
        return "discovery_only", "pcmci_discovery_signal_without_formal_identification"
    return "weak_or_unaligned", "insufficient_cross_layer_support"



def _normalize_scm_edge_row(row: Dict[str, str]) -> Dict[str, str]:
    source = _first(row, ["source", "treatment_col"])
    target = _first(row, ["target", "outcome_col"])
    return {
        "insight_id": _first(row, ["insight_id", "candidate_id", "edge_id"]),
        "source": source,
        "target": target,
        "treatment_col": _first(row, ["treatment_col", "source"], source),
        "outcome_col": _first(row, ["outcome_col", "target"], target),
        "lag": _first(row, ["lag", "tau", "time_lag"], ""),
        "edge_type": _first(row, ["edge_kind", "edge_type"], "lagged_structural_candidate"),
        "legacy_discovery_score": _first(row, ["selection_score", "identification_priority"]),
        "legacy_pcmci_score": _first(row, ["mci_score", "selection_score"]),
        "mci_status": _first(row, ["mci_status"]),
        "mci_q_value": _first(row, ["mci_q_value"]),
        "conditioning_set_used": _first(row, ["conditioning_set_used", "parent_set"]),
        "conditioning_set_size": _first(row, ["conditioning_set_size"]),
        "adjustment_set": _first(row, ["conditioning_set_used", "parent_set", "candidate_covariates"]),
        "forbidden_adjustment_set": _first(row, ["forbidden_adjustment_set", "post_treatment_columns"]),
        "scm_role_hint": _first(row, ["scm_role_hint"]),
    }


def _normalize_adjustment_set_row(row: Dict[str, str]) -> Dict[str, str]:
    source = _first(row, ["source", "treatment_col"])
    target = _first(row, ["target", "outcome_col"])
    identified = _first(row, ["identified"], "")
    eligible = _first(row, ["eligible_for_estimation"], "")
    return {
        "insight_id": _first(row, ["insight_id", "candidate_id", "edge_id"]),
        "source": source,
        "target": target,
        "treatment_col": _first(row, ["treatment_col", "source"], source),
        "outcome_col": _first(row, ["outcome_col", "target"], target),
        "lag": _first(row, ["lag", "tau", "time_lag"], ""),
        "estimand_type": _first(row, ["estimand_type", "effect_scope"], "total_effect"),
        "identification_status": _first(row, ["identification_status"], "identified" if identified in {"1", "true", "True"} else "not_identified"),
        "identified": identified,
        "identification_strategy": _first(row, ["identification_strategy", "strategy"]),
        "frontdoor_status": _first(row, ["frontdoor_status"]),
        "adjustment_set_status": _first(row, ["adjustment_set_status"], "missing"),
        "adjustment_set": _first(row, ["adjustment_set", "candidate_adjustment_set", "total_adjustment_set"]),
        "total_adjustment_set": _first(row, ["total_adjustment_set", "adjustment_set"]),
        "direct_adjustment_set": _first(row, ["direct_adjustment_set"]),
        "candidate_adjustment_set": _first(row, ["candidate_adjustment_set", "adjustment_set"]),
        "backdoor_status": _first(row, ["backdoor_status"]),
        "blocked_by": _first(row, ["blocked_by"]),
        "assumption_notes": _first(row, ["assumption_notes"]),
        "adjustment_set_source": "identification_adjustment_sets",
        "eligible_for_estimation": eligible,
        "forbidden_adjustment_set": _first(row, ["forbidden_adjustment_set", "forbidden_adjustments"]),
        "mediators": _first(row, ["mediators"]),
        "colliders": _first(row, ["colliders"]),
        "conditioning_set_used": _first(row, ["conditioning_set_used"]),
        "conditioning_set_size": _first(row, ["conditioning_set_size"]),
        "mci_status": _first(row, ["mci_status"]),
        "pc1_parent_support": _first(row, ["pc1_parent_support"]),
        "scm_role_hint": _first(row, ["scm_role_hint"]),
        "source_artifacts": "adjustment_sets",
        "source_authority": "formal_adjustment_set_summary",
    }


def _is_canonical_contract_row(row: Dict[str, str]) -> bool:
    """Rows allowed in the public handoff contract.

    Raw Discovery/SCM seed rows remain useful for audit, but they should not be
    mixed into causal_contract.csv because estimation/effect outputs are keyed to
    the canonical handoff rows only.
    """
    authority = str(row.get("authority_level", "") or "").strip()
    enabled = str(row.get("estimation_enabled", "") or "").strip().lower()
    source = str(row.get("source_artifacts", "") or "").strip()
    if authority == "blocked_id_algorithm" and source == "identified_effects":
        # Legacy identifier blocked rows are audit evidence only.  Canonical
        # hard-block rows should come from scm/id_algorithm_audit.csv, while the
        # older identified_effects wrapper remains visible in causal_contract_audit.csv.
        return False
    return authority in {"identified_estimable", "identified_needs_estimation", "graph_review", "blocked_id_algorithm"} or enabled in {"1", "1.0", "true", "yes", "y", "on"}


def _split_contract_rows(rows: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    canonical: List[Dict[str, str]] = []
    audit: List[Dict[str, str]] = []
    for row in rows:
        (canonical if _is_canonical_contract_row(row) else audit).append(row)
    return canonical, audit

def build_causal_contract(out_dir: str = "out") -> Tuple[List[Dict[str, str]], Dict[str, object]]:
    out = Path(out_dir)
    sources = {
        "bridge": out / "discovery_estimation_bridge.csv",
        "pcmci_scm_bridge": out / "pcmci_scm_bridge.csv",
        "pcmci_scm_bridge_layered": out / "discovery" / "pcmci_scm_bridge.csv",
        "scm_edges": out / "scm" / "scm_edges.csv",
        "insights": out / "insights_level2.csv",
        "ranking_insights": out / "ranking" / "insights_level2.csv",
        "edges": out / "edges.csv",
        "pcmci_links": out / "discovery" / "pcmci_links.csv",
        "identified": out / "identification" / "identified_effects.csv",
        "legacy_identified": out / "identified_effects.csv",
        "adjustment_sets": out / "identification" / "adjustment_sets.csv",
        "structural_models": out / "scm" / "structural_models.csv",
        "id_algorithm_audit": out / "scm" / "id_algorithm_audit.csv",
        "symbolic_evaluation": out / "scm" / "symbolic_evaluation.csv",
    }
    rows_by_key: Dict[str, Dict[str, str]] = {}
    counts: Dict[str, int] = {}

    source_authority_by_name = {
        "bridge": "discovery_handoff_candidate",
        "insights": "discovery_ranked_candidate",
        "ranking_insights": "discovery_ranked_candidate",
        "edges": "raw_discovery_only",
        "pcmci_links": "pcmci_raw_input_only",
        "pcmci_scm_bridge": "pcmci_scm_structural_prior",
        "pcmci_scm_bridge_layered": "pcmci_scm_structural_prior",
        "scm_edges": "scm_structural_prior",
    }
    for name in ["bridge", "pcmci_scm_bridge", "pcmci_scm_bridge_layered", "insights", "ranking_insights", "edges", "pcmci_links"]:
        raw = _read_csv(sources[name])
        counts[name] = len(raw)
        for idx, row in enumerate(raw):
            norm = _normalize_discovery_row(row)
            norm["source_artifacts"] = name
            norm["source_authority"] = source_authority_by_name.get(name, "discovery_candidate")
            key = _to_key(norm or row, name, idx)
            base = rows_by_key.setdefault(key, {c: "" for c in CONTRACT_COLUMNS})
            if not base.get("insight_id"):
                base["insight_id"] = key
            # Raw edges and raw PCMCI links are allowed to seed rows, but they
            # must not overwrite bridge/insight/identification metadata.
            _merge_nonempty(base, norm, overwrite=False)

    raw_scm_edges = _read_csv(sources["scm_edges"])
    counts["scm_edges"] = len(raw_scm_edges)
    for idx, row in enumerate(raw_scm_edges):
        norm = _normalize_scm_edge_row(row)
        norm["source_artifacts"] = "scm_edges"
        norm["source_authority"] = "scm_structural_prior"
        fallback_key = _to_key(norm or row, "scm_edge", idx)
        key = _best_existing_key(rows_by_key, norm or row, fallback_key)
        base = rows_by_key.setdefault(key, {c: "" for c in CONTRACT_COLUMNS})
        if not base.get("insight_id"):
            base["insight_id"] = key
        _merge_nonempty(base, norm, overwrite=False)

    ident_path = sources["identified"] if sources["identified"].exists() else sources["legacy_identified"]
    raw_ident = _read_csv(ident_path)
    counts["identified_effects"] = len(raw_ident)
    for idx, row in enumerate(raw_ident):
        norm = _normalize_identification_row(row)
        norm["source_artifacts"] = "identified_effects"
        norm["source_authority"] = "formal_identification"
        fallback_key = _to_key(norm or row, "identified", idx)
        key = _best_existing_key(rows_by_key, norm or row, fallback_key)
        base = rows_by_key.setdefault(key, {c: "" for c in CONTRACT_COLUMNS})
        if not base.get("insight_id"):
            base["insight_id"] = key
        _merge_nonempty(base, norm, overwrite=True)

    raw_adjustments = _read_csv(sources["adjustment_sets"])
    counts["adjustment_sets"] = len(raw_adjustments)
    for idx, row in enumerate(raw_adjustments):
        norm = _normalize_adjustment_set_row(row)
        fallback_key = _to_key(norm or row, "adjustment_set", idx)
        key = _best_existing_key(rows_by_key, norm or row, fallback_key)
        base = rows_by_key.setdefault(key, {c: "" for c in CONTRACT_COLUMNS})
        if not base.get("insight_id"):
            base["insight_id"] = key
        _merge_nonempty(base, norm, overwrite=True)

    raw_id_algorithm = _read_csv(sources["id_algorithm_audit"])
    counts["id_algorithm_audit"] = len(raw_id_algorithm)
    for idx, row in enumerate(raw_id_algorithm):
        norm = _normalize_id_algorithm_row(row)
        fallback_key = _to_key(norm or row, "id_algorithm", idx)
        key = _best_existing_key(rows_by_key, norm or row, fallback_key)
        base = rows_by_key.setdefault(key, {c: "" for c in CONTRACT_COLUMNS})
        if not base.get("insight_id"):
            base["insight_id"] = key
        _merge_nonempty(base, norm, overwrite=True)

    raw_symbolic_evaluation = _read_csv(sources["symbolic_evaluation"])
    counts["symbolic_evaluation"] = len(raw_symbolic_evaluation)
    for idx, row in enumerate(raw_symbolic_evaluation):
        norm = _normalize_symbolic_evaluation_row(row)
        fallback_key = _to_key(norm or row, "symbolic_evaluation", idx)
        key = _best_existing_key(rows_by_key, norm or row, fallback_key)
        base = rows_by_key.setdefault(key, {c: "" for c in CONTRACT_COLUMNS})
        if not base.get("insight_id"):
            base["insight_id"] = key
        _merge_nonempty(base, norm, overwrite=True)

    raw_models = _read_csv(sources["structural_models"])
    counts["structural_models"] = len(raw_models)
    models_by_node = _structural_models_by_node(raw_models)
    for row in rows_by_key.values():
        outcome = _first(row, ["outcome_col", "target"])
        model = models_by_node.get(outcome, {})
        if model:
            _merge_nonempty(row, {
                "structural_family": _first(model, ["family", "model_family", "structural_family"]),
                "structural_model_status": _first(model, ["status", "model_status", "fit_status"], "diagnostic_available"),
                "structural_r2": _first(model, ["r2", "train_r2", "cv_r2", "fit_score"]),
                "structural_model_role": "diagnostic_simulation_only",
                "structural_fit_authority_level": _first(model, ["fit_authority_level"], "diagnostic_simulation_only"),
                "causal_authority_from_fit": _first(model, ["causal_authority_from_fit"], "0"),
                "source_artifacts": "structural_models",
                "source_authority": "scm_fit_diagnostic_only",
            }, overwrite=True)

    rows: List[Dict[str, str]] = []
    for key in sorted(rows_by_key):
        row = rows_by_key[key]
        if not row.get("insight_id"):
            row["insight_id"] = key
        # Mirror canonical treatment/outcome/source/target fields.
        if not row.get("treatment_col") and row.get("source"):
            row["treatment_col"] = row["source"]
        if not row.get("source") and row.get("treatment_col"):
            row["source"] = row["treatment_col"]
        if not row.get("outcome_col") and row.get("target"):
            row["outcome_col"] = row["target"]
        if not row.get("target") and row.get("outcome_col"):
            row["target"] = row["outcome_col"]
        level, reason = _authority(row)
        row["authority_level"] = level
        row["authority_reason"] = reason
        # Step 2 SCM-first gate: Estimation may only run numeric effects when
        # SCM/ID produced an identified_estimable contract row. All other rows
        # remain visible for audit/planning but are hard-disabled for estimation.
        allowed = (
            level == "identified_estimable"
            and _truthy_text(row.get("identified"))
            and _truthy_text(row.get("estimation_enabled"))
            and _truthy_text(row.get("canonical_id_authority"))
        )
        row["allowed_for_estimation"] = "1" if allowed else "0"
        if not allowed:
            row["estimation_enabled"] = "0"
        rows.append({c: row.get(c, "") for c in CONTRACT_COLUMNS})

    canonical_rows, audit_rows = _split_contract_rows(rows)
    estimation_handoff_path = Path(out_dir) / "estimation" / "estimation_handoff.csv"
    estimation_handoff_rows = canonical_rows

    manifest = {
        "contract_version": CONTRACT_VERSION,
        "contract": "PCMCI/Discovery + SCM identification + SCM ID-algorithm audit + SCM fit diagnostics normalized for estimation",
        "graph_authority_policy": "one_runtime_graph_one_offline_scm_graph_one_handoff_contract_no_personal_graph",
        "canonical_runtime_graph": "operational_causal_graph.yaml",
        "canonical_offline_scm_graph": "out/scm/scm_graph.json",
        "canonical_handoff_contract": "out/causal_contract.csv",
        "canonical_audit_contract": "out/causal_contract_audit.csv",
        "canonical_do_authority_audit": "out/scm/do_authority_audit.csv",
        "non_canonical_inputs": {
            "out/edges.csv": "raw_discovery_seed_only_not_downstream_authority",
            "out/discovery/pcmci_links.csv": "raw_pcmci_seed_only_not_downstream_authority",
            "out/pcmci_scm_bridge.csv": "structural_prior_for_scm_and_identification_not_estimation_authority",
            "out/scm/scm_edges.csv": "scm_structural_prior_not_estimation_authority",
            "out/scm/id_algorithm_audit.csv": "scm_id_algorithm_authority_gate_for_identification",
            "out/scm/symbolic_evaluation.csv": "symbolic_formula_route_plan_for_estimator_handoff",
            "out/identification/adjustment_sets.csv": "canonical_identification_adjustment_summary",
            "out/identified_effects.csv": "legacy_mirror_only",
        },
        "n_rows": len(canonical_rows),
        "n_audit_rows": len(audit_rows),
        "n_total_rows_before_canonical_split": len(rows),
        "source_counts": counts,
        "merge_policy": "identified_effects rows merge by insight_id first, then by source-target pair when lag is absent, preferring bridge-backed rows",
        "downstream_rule": "estimation and veto must not consume raw edges directly; structural fit is diagnostic/simulation-only; SCM id_algorithm_audit blocks hedge/c-factor/full-ID gaps; symbolic_evaluation routes formula JSON into estimator plans; backdoor and limited frontdoor may enable strong-do only when causal_contract explicitly sets estimation_enabled=1 and canonical_id_authority=1; truncated factorization is formula-evaluable and numeric-ready only through contract-gated symbolic_numeric.py; CDE/NDE/NIE remain non-estimator routes; do_outputs writes do_authority_audit.csv as the end-to-end ID-to-estimate ledger",
        "columns": CONTRACT_COLUMNS,
        "estimation_handoff_csv": str(estimation_handoff_path),
        "n_estimation_handoff_rows": len(estimation_handoff_rows),
        "audit_policy": "Rows without handoff/estimation authority are written to causal_contract_audit.csv, not causal_contract.csv.",
        "_audit_rows": audit_rows,
    }
    return canonical_rows, manifest


def write_causal_contract(out_dir: str = "out") -> Dict[str, str]:
    rows, manifest = build_causal_contract(out_dir=out_dir)
    out = Path(out_dir)
    csv_path = out / "causal_contract.csv"
    audit_csv_path = out / "causal_contract_audit.csv"
    manifest_path = out / "causal_contract_manifest.json"
    estimation_dir = out / "estimation"
    estimation_handoff_path = estimation_dir / "estimation_handoff.csv"
    audit_rows = list(manifest.pop("_audit_rows", []))
    _write_csv(csv_path, rows, CONTRACT_COLUMNS)
    _write_csv(audit_csv_path, audit_rows, CONTRACT_COLUMNS)
    handoff_columns = [
        "insight_id", "source", "target", "treatment_col", "outcome_col", "lag",
        "authority_level", "authority_reason", "estimation_enabled", "allowed_for_estimation", "identification_status",
        "identified", "identification_strategy", "estimand_type", "estimand_expression",
        "adjustment_set_status", "adjustment_set", "total_adjustment_set",
        "candidate_adjustment_set", "backdoor_status", "blocked_by", "assumption_notes",
        "adjustment_set_source", "eligible_for_estimation",
        "unobserved_confounding_risk", "sensitivity_level", "sensitivity_status",
        "recommended_sensitivity_method",
        "forbidden_adjustment_set", "negative_controls", "conditioning_set_used",
        "conditioning_set_size", "mci_status", "mci_q_value", "mci_n_eff",
        "pc1_parent_support", "scm_role_hint",
        "id_status", "symbolic_formula_status", "symbolic_formula_kind",
        "symbolic_evaluator_status", "symbolic_formula_evaluable",
        "symbolic_numeric_estimator_ready", "symbolic_estimator_route",
        "hedge_detected", "hedge_status", "recursive_id_status",
        "c_factor_status", "district_status", "id_block_reason",
        "source_authority", "source_artifacts", "canonical_id_authority",
    ]
    estimation_handoff_rows = list(rows)
    _write_csv(estimation_handoff_path, estimation_handoff_rows, handoff_columns)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump({**manifest, "causal_contract_csv": str(csv_path), "causal_contract_audit_csv": str(audit_csv_path)}, f, ensure_ascii=False, indent=2)
    return {
        "causal_contract_csv": str(csv_path),
        "causal_contract_audit_csv": str(audit_csv_path),
        "causal_contract_manifest": str(manifest_path),
        "estimation_handoff_csv": str(estimation_handoff_path),
    }


def cli(argv: Optional[List[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Write canonical causal contract from discovery/SCM artifacts")
    ap.add_argument("--out-dir", default="out")
    args = ap.parse_args(argv)
    paths = write_causal_contract(out_dir=args.out_dir)
    print(json.dumps({"status": "ok", **paths}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
