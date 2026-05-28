"""Compact cross-layer causal report for Amantia.

This report is intentionally non-authoritative: it summarizes existing
Discovery, PCMCI/SCM, Identification and Estimation/Sensitivity artifacts in a
single review table. The authority gate remains causal_contract.csv and any
runtime do/veto contracts.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

REPORT_VERSION = 7

REPORT_COLUMNS = [
    "report_id", "insight_id", "source", "target", "lag",
    "final_recommendation", "claim_status", "query_status", "estimated",
    "id_gate_status", "estimation_gate_status", "allowed_for_estimation",
    "authority_level", "discovery_track", "selection_score", "hypothesis_signal_grade", "safety_risk_grade",
    "mci_status", "mci_q_value", "mci_n_eff",
    "scm_role_hint", "id_status", "symbolic_formula_status",
    "symbolic_evaluator_status", "symbolic_formula_evaluable",
    "symbolic_numeric_estimator_ready", "symbolic_estimator_route",
    "do_authorized", "do_mode", "effect_semantics", "support_n_low", "support_n_high",
    "support_min", "support_ratio", "support_n_mid", "overlap_score",
    "bootstrap_status", "bootstrap_success_n", "ci_width", "ci_width_to_effect_ratio",
    "extrapolation_risk", "sensitivity_warning",
    "hedge_detected", "hedge_status",
    "recursive_id_status", "c_factor_status", "district_status",
    "identification_status", "backdoor_status", "adjustment_set",
    "estimation_status", "recommended_estimator", "effect_claim_status", "effect_estimate",
    "ci_low", "ci_high", "robustness_status", "partial_r2_needed_to_explain_away",
    "sensitivity_level", "sensitivity_status", "sensitivity_quant_status",
    "negative_control_status", "negative_control_pass", "placebo_status", "placebo_pass",
    "minimum_report_before_effect_claim", "main_blocker", "next_action", "reason",
]


def _as_str(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    if text.lower() in {"nan", "none", "nat"}:
        return ""
    return text.strip()


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            return [dict(r) for r in csv.DictReader(f)]
    except (OSError, csv.Error, UnicodeDecodeError, ValueError):
        return []


def _write_csv(path: Path, rows: List[Dict[str, object]], columns: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _identity(row: Dict[str, object]) -> Tuple[str, str, str, str]:
    iid = _as_str(row.get("insight_id")) or _as_str(row.get("proposal_id"))
    source = _as_str(row.get("source") or row.get("treatment_col") or row.get("treatment"))
    target = _as_str(row.get("target") or row.get("target_col") or row.get("outcome_col") or row.get("outcome"))
    lag = _as_str(row.get("lag"))
    return iid, source, target, lag


def _key(row: Dict[str, object]) -> str:
    iid, source, target, lag = _identity(row)
    if iid:
        return f"id::{iid}"
    return f"edge::{source}->{target}@{lag}"


def _edge_key(row: Dict[str, object]) -> str:
    _, source, target, lag = _identity(row)
    return f"edge::{source}->{target}@{lag}"


def _edge_prefix(row: Dict[str, object]) -> str:
    _iid, source, target, _lag = _identity(row)
    return f"edge::{source}->{target}@" if (source or target) else ""


def _merge_rows(base: Dict[str, Dict[str, str]], rows: List[Dict[str, str]], *, prefer_existing: bool = False) -> None:
    for row in rows:
        if not row:
            continue
        key = _key(row)
        edge_key = _edge_key(row)
        if key == "id::" or key == "edge::->@":
            continue
        candidates = [key]
        if edge_key and edge_key != key:
            candidates.append(edge_key)
        prefix = _edge_prefix(row)
        prefix_match = next((k for k in base if prefix and k.startswith(prefix)), "")
        existing_key = next((k for k in candidates if k in base), prefix_match or key)
        cur = base.setdefault(existing_key, {})
        for k, v in row.items():
            val = _as_str(v)
            if not val:
                continue
            if prefer_existing and _as_str(cur.get(k)):
                continue
            cur[k] = val
        # Cross-index by edge for later sources without insight_id.
        if edge_key and edge_key not in base:
            base[edge_key] = cur


def _boolish(value: object) -> bool:
    return _as_str(value).lower() in {"1", "true", "yes", "y", "on"}


def _effect_is_numeric(row: Dict[str, str]) -> bool:
    effect = _as_str(row.get("effect_estimate"))
    if not effect:
        return False
    try:
        float(effect)
        return True
    except (TypeError, ValueError):
        return False


def _query_gate_status(row: Dict[str, str]) -> Tuple[str, str, str, str]:
    """Return query_status, estimated, id_gate_status, estimation_gate_status.

    This is intentionally a readable ledger label, not an authority grant. The
    underlying authority still comes from causal_contract.csv / ID audit.
    """
    authority = _as_str(row.get("authority_level")).lower()
    id_status = _as_str(row.get("identification_status") or row.get("id_status")).lower()
    estimation_status = _as_str(row.get("estimation_status")).lower()
    claim = _as_str(row.get("effect_claim_status")).lower()
    allowed_text = _as_str(row.get("allowed_for_estimation"))
    identified_text = _as_str(row.get("identified") or row.get("id_identified"))
    enabled_text = _as_str(row.get("estimation_enabled"))
    reason = _as_str(row.get("reason_codes") or row.get("reason") or row.get("blocked_by") or row.get("id_block_reason"))

    estimated = "1" if _effect_is_numeric(row) and not claim.startswith("not_estimated") else "0"

    if id_status in {"not_identified", "unidentified", "blocked_id_algorithm", "blocked", "simulable_not_identified"} or authority == "blocked_id_algorithm" or identified_text == "0":
        id_gate = "blocked_not_identified"
    elif authority == "identified_estimable" or identified_text == "1":
        id_gate = "identified"
    elif authority == "identified_needs_estimation":
        id_gate = "identified_plan_only"
    elif authority == "graph_review" or "review" in id_status:
        id_gate = "review_required"
    elif id_status:
        id_gate = id_status
    else:
        id_gate = "not_evaluated"

    if estimated == "1":
        est_gate = "estimated"
        query_status = "estimated"
    elif allowed_text == "0" or enabled_text == "0" or claim == "not_estimated_contract_gate":
        est_gate = "blocked_by_contract_gate"
        query_status = "blocked"
    elif estimation_status in {"blocked", "not_estimated_contract_gate"}:
        est_gate = "blocked"
        query_status = "blocked"
    elif estimation_status == "needs_estimator_or_data" or authority == "identified_needs_estimation":
        est_gate = "plan_only"
        query_status = "identified_not_estimated"
    elif estimation_status == "needs_graph_review" or authority == "graph_review":
        est_gate = "graph_review_required"
        query_status = "review_required"
    elif estimation_status == "diagnostic_only":
        est_gate = "diagnostic_only"
        query_status = "diagnostic_only"
    elif authority == "identified_estimable" and allowed_text in {"", "1"}:
        est_gate = "ready_or_pending"
        query_status = "identified_pending_estimate"
    elif reason:
        est_gate = reason
        query_status = "not_estimated"
    else:
        est_gate = "not_evaluated"
        query_status = "not_evaluated"

    return query_status, estimated, id_gate, est_gate


def _recommendation(row: Dict[str, str]) -> Tuple[str, str, str, str]:
    """Return recommendation, claim_status, main_blocker, next_action."""
    authority = _as_str(row.get("authority_level")).lower()
    estimation_status = _as_str(row.get("estimation_status")).lower()
    identification_status = _as_str(row.get("identification_status")).lower()
    backdoor = _as_str(row.get("backdoor_status")).lower()
    sensitivity = _as_str(row.get("sensitivity_status")).lower()
    discovery_track = _as_str(row.get("discovery_track")).lower()
    risk_grade = _as_str(row.get("safety_risk_grade")).lower()
    mci_status = _as_str(row.get("mci_status")).lower()
    scm_role = _as_str(row.get("scm_role_hint")).lower()
    blocked_by = _as_str(row.get("blocked_by"))
    reason = _as_str(row.get("reason")) or _as_str(row.get("drop_reason"))
    do_mode = _as_str(row.get("do_mode"))
    robustness_status = _as_str(row.get("robustness_status")).lower()
    sensitivity_warning = _as_str(row.get("sensitivity_warning"))

    if robustness_status.startswith("blocked") or do_mode.startswith("blocked_low_support"):
        return (
            "blocked",
            "no_causal_claim_numeric_support_failed",
            sensitivity_warning or robustness_status or reason or "numeric_robustness_gate_blocked",
            "do_not_claim; improve support/overlap or inspect symbolic_numeric_diagnostics.csv",
        )
    if robustness_status.startswith("diagnostic_only") or do_mode.startswith("diagnostic_only"):
        return (
            "diagnostic_only",
            "numeric_robustness_warning",
            sensitivity_warning or robustness_status or reason or "numeric_robustness_warning",
            "treat estimate as diagnostic until bootstrap/support improves",
        )

    if _boolish(row.get("do_authorized")):
        return (
            "do_estimate_ready",
            "authorized_do_estimate",
            "",
            "review effect size, CI, support and sensitivity before external claim",
        )
    if "blocked" in authority or estimation_status == "blocked" or "hard_blocked" in scm_role:
        return (
            "blocked",
            "no_causal_claim",
            blocked_by or reason or "blocked_by_contract_or_hard_graph_gate",
            "do_not_estimate; inspect gate_audit and SCM assumptions",
        )
    if authority == "identified_estimable" and estimation_status == "can_estimate_now":
        if sensitivity.startswith("required"):
            return (
                "estimate_after_sensitivity",
                "identified_but_sensitivity_required",
                "sensitivity_required",
                "run recommended sensitivity method before any effect claim",
            )
        return (
            "estimate_now",
            "identified_estimable",
            "",
            "run estimator and report uncertainty/sensitivity",
        )
    if authority == "identified_needs_estimation" or estimation_status == "needs_estimator_or_data":
        return (
            "prepare_estimation",
            "identified_needs_estimation",
            "estimator_or_data_not_ready",
            "complete estimator inputs, support checks and sensitivity plan",
        )
    if authority == "graph_review" or estimation_status == "needs_graph_review" or "review" in identification_status:
        return (
            "graph_review",
            "not_yet_estimable",
            blocked_by or "graph_or_adjustment_set_review_required",
            "review adjustment set, colliders, mediators and assumptions",
        )
    if backdoor in {"backdoor_adjustment_candidate", "adjustment_candidate"} and not _boolish(row.get("estimation_enabled")):
        return (
            "identification_review",
            "adjustment_candidate_not_authorized",
            "authority_not_enabled_for_estimation",
            "promote through causal_contract only after review",
        )
    if mci_status in {"pass", "diagnostic_support"} or "temporal_parent" in scm_role:
        return (
            "diagnostic_only",
            "structural_prior_only",
            "not_formally_identified",
            "use as SCM prior; do not make effect claim yet",
        )
    if discovery_track in {"weak_structured", "exploratory", "diagnostic_only"}:
        return (
            "observe_more",
            "discovery_only",
            "insufficient_downstream_authority" if not risk_grade else f"safety_risk_{risk_grade}",
            "collect more data or strengthen SCM/identification evidence",
        )
    return (
        "no_action",
        "insufficient_evidence",
        reason or "no_positive_authority_found",
        "keep in audit only",
    )


def _compact_row(row: Dict[str, str], idx: int) -> Dict[str, object]:
    iid, source, target, lag = _identity(row)
    if not iid:
        iid = f"{source}->{target}@{lag}" if source or target else f"row_{idx}"
    rec, claim, blocker, next_action = _recommendation(row)
    query_status, estimated, id_gate, estimation_gate = _query_gate_status(row)
    reason_parts = []
    for c in ("reason", "assumption_notes", "drop_reason", "risk_flags", "sensitivity_warning"):
        val = _as_str(row.get(c))
        if val and val not in reason_parts:
            reason_parts.append(val)
    return {
        "report_id": f"causal_report::{iid}",
        "insight_id": iid,
        "source": source,
        "target": target,
        "lag": lag,
        "final_recommendation": rec,
        "claim_status": claim,
        "query_status": query_status,
        "estimated": estimated,
        "id_gate_status": id_gate,
        "estimation_gate_status": estimation_gate,
        "allowed_for_estimation": _as_str(row.get("allowed_for_estimation")),
        "authority_level": _as_str(row.get("authority_level")),
        "discovery_track": _as_str(row.get("discovery_track")),
        "selection_score": _as_str(row.get("selection_score")),
        "hypothesis_signal_grade": _as_str(row.get("hypothesis_signal_grade")),
        "safety_risk_grade": _as_str(row.get("safety_risk_grade")),
        "mci_status": _as_str(row.get("mci_status")),
        "mci_q_value": _as_str(row.get("mci_q_value")),
        "mci_n_eff": _as_str(row.get("mci_n_eff")),
        "scm_role_hint": _as_str(row.get("scm_role_hint")),
        "id_status": _as_str(row.get("id_status")),
        "symbolic_formula_status": _as_str(row.get("symbolic_formula_status")),
        "symbolic_evaluator_status": _as_str(row.get("symbolic_evaluator_status")),
        "symbolic_formula_evaluable": _as_str(row.get("symbolic_formula_evaluable")),
        "symbolic_numeric_estimator_ready": _as_str(row.get("symbolic_numeric_estimator_ready")),
        "symbolic_estimator_route": _as_str(row.get("symbolic_estimator_route")),
        "do_authorized": _as_str(row.get("do_authorized")),
        "do_mode": _as_str(row.get("do_mode")),
        "effect_semantics": _as_str(row.get("effect_semantics")),
        "support_n_low": _as_str(row.get("support_n_low")),
        "support_n_high": _as_str(row.get("support_n_high")),
        "support_min": _as_str(row.get("support_min")),
        "support_ratio": _as_str(row.get("support_ratio")),
        "support_n_mid": _as_str(row.get("support_n_mid")),
        "overlap_score": _as_str(row.get("overlap_score")),
        "bootstrap_status": _as_str(row.get("bootstrap_status")),
        "bootstrap_success_n": _as_str(row.get("bootstrap_success_n")),
        "ci_width": _as_str(row.get("ci_width")),
        "ci_width_to_effect_ratio": _as_str(row.get("ci_width_to_effect_ratio")),
        "extrapolation_risk": _as_str(row.get("extrapolation_risk")),
        "sensitivity_warning": _as_str(row.get("sensitivity_warning")),
        "hedge_detected": _as_str(row.get("hedge_detected")),
        "hedge_status": _as_str(row.get("hedge_status")),
        "recursive_id_status": _as_str(row.get("recursive_id_status")),
        "c_factor_status": _as_str(row.get("c_factor_status")),
        "district_status": _as_str(row.get("district_status")),
        "identification_status": _as_str(row.get("identification_status")),
        "backdoor_status": _as_str(row.get("backdoor_status")),
        "adjustment_set": _as_str(row.get("candidate_adjustment_set") or row.get("adjustment_set") or row.get("conditioning_set_used")),
        "estimation_status": _as_str(row.get("estimation_status")),
        "recommended_estimator": _as_str(row.get("recommended_estimator")),
        "effect_claim_status": _as_str(row.get("effect_claim_status")),
        "effect_estimate": _as_str(row.get("effect_estimate")),
        "ci_low": _as_str(row.get("ci_low")),
        "ci_high": _as_str(row.get("ci_high")),
        "robustness_status": _as_str(row.get("robustness_status")),
        "partial_r2_needed_to_explain_away": _as_str(row.get("partial_r2_needed_to_explain_away")),
        "sensitivity_level": _as_str(row.get("sensitivity_level")),
        "sensitivity_status": _as_str(row.get("sensitivity_status")),
        "sensitivity_quant_status": _as_str(row.get("sensitivity_quant_status")),
        "negative_control_status": _as_str(row.get("negative_control_status")),
        "negative_control_pass": _as_str(row.get("negative_control_pass")),
        "placebo_status": _as_str(row.get("placebo_status")),
        "placebo_pass": _as_str(row.get("placebo_pass")),
        "minimum_report_before_effect_claim": _as_str(row.get("minimum_report_before_effect_claim")),
        "main_blocker": blocker,
        "next_action": next_action,
        "reason": "|".join(reason_parts),
    }


def _row_edge_token(row: Dict[str, object]) -> str:
    return f"{_as_str(row.get('source') or row.get('treatment_col'))}->{_as_str(row.get('target') or row.get('outcome_col'))}@{_as_str(row.get('lag'))}"


def _canonical_tokens(rows: List[Dict[str, str]]) -> Tuple[set[str], set[str]]:
    ids = {_as_str(r.get("insight_id")) for r in rows if _as_str(r.get("insight_id"))}
    edges = {_row_edge_token(r) for r in rows if _as_str(r.get("source") or r.get("treatment_col")) or _as_str(r.get("target") or r.get("outcome_col"))}
    return ids, edges


def _split_report_rows_by_contract_scope(
    rows: List[Dict[str, object]],
    canonical_source_rows: List[Dict[str, str]],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Keep public report scoped to contract/plan/effect rows, audit raw seeds.

    Step 3 SCM-first reporting: causal_report.csv is the human query ledger for
    rows that reached the contract/estimation/effect layer. Raw discovery or SCM
    seed rows that never reached the handoff stay in causal_report_audit.csv.
    """
    canonical_ids, canonical_edges = _canonical_tokens(canonical_source_rows)
    if not canonical_ids and not canonical_edges:
        return rows, []
    canonical: List[Dict[str, object]] = []
    audit: List[Dict[str, object]] = []
    for row in rows:
        iid = _as_str(row.get("insight_id"))
        edge = _row_edge_token(row)
        if (iid and iid in canonical_ids) or (edge and edge in canonical_edges):
            canonical.append(row)
        else:
            audit.append(row)
    return canonical, audit


def _source_paths(out: Path) -> Dict[str, Path]:
    return {
        "gate_audit": out / "gate_audit.csv",
        "discovery_scoring": out / "discovery_scoring.csv",
        "pcmci_scm_bridge": out / "pcmci_scm_bridge.csv",
        "scm_edges": out / "scm" / "scm_edges.csv",
        "id_algorithm_audit": out / "scm" / "id_algorithm_audit.csv",
        "symbolic_evaluation": out / "scm" / "symbolic_evaluation.csv",
        "symbolic_numeric_estimates": out / "scm" / "symbolic_numeric_estimates.csv",
        "symbolic_numeric_diagnostics": out / "scm" / "symbolic_numeric_diagnostics.csv",
        "do_estimates": out / "scm" / "do_estimates.csv",
        "do_diagnostics": out / "scm" / "do_diagnostics.csv",
        "unified_gate_audit": out / "gate_audit.csv",
        "adjustment_sets": out / "identification" / "adjustment_sets.csv",
        "causal_contract": out / "causal_contract.csv",
        "estimation_handoff": out / "estimation" / "estimation_handoff.csv",
        "estimation_plan": out / "estimation" / "estimation_plan.csv",
        "sensitivity_analysis": out / "estimation" / "sensitivity_analysis.csv",
        "effect_estimates": out / "estimation" / "effect_estimates.csv",
        "robustness_diagnostics": out / "estimation" / "robustness_diagnostics.csv",
        "sensitivity_quantitative": out / "estimation" / "sensitivity_quantitative.csv",
        "negative_control_checks": out / "estimation" / "negative_control_checks.csv",
        "placebo_checks": out / "estimation" / "placebo_checks.csv",
    }


def build_causal_report(out_dir: str = "out") -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    out = Path(out_dir)
    paths = _source_paths(out)
    base: Dict[str, Dict[str, str]] = {}

    # Broad discovery rows first, then higher-authority downstream artifacts override/add fields.
    for name in [
        "gate_audit", "discovery_scoring", "pcmci_scm_bridge", "scm_edges", "id_algorithm_audit", "symbolic_evaluation",
        "symbolic_numeric_estimates", "symbolic_numeric_diagnostics", "do_estimates", "do_diagnostics", "adjustment_sets",
        "causal_contract", "estimation_handoff", "estimation_plan", "sensitivity_analysis",
        "effect_estimates", "negative_control_checks", "placebo_checks", "robustness_diagnostics", "sensitivity_quantitative",
    ]:
        _merge_rows(base, _read_csv(paths[name]), prefer_existing=False)

    unique_rows: List[Dict[str, str]] = []
    seen_obj = set()
    for _, row in base.items():
        oid = id(row)
        if oid in seen_obj:
            continue
        seen_obj.add(oid)
        if _identity(row)[1] or _identity(row)[2] or _identity(row)[0]:
            unique_rows.append(row)

    report_all = [_compact_row(r, i) for i, r in enumerate(unique_rows)]
    canonical_scope_rows = (
        _read_csv(paths["causal_contract"])
        + _read_csv(paths["estimation_handoff"])
        + _read_csv(paths["estimation_plan"])
        + _read_csv(paths["effect_estimates"])
        + _read_csv(paths["do_estimates"])
        + _read_csv(paths["symbolic_numeric_estimates"])
    )
    report, audit_rows = _split_report_rows_by_contract_scope(report_all, canonical_scope_rows)
    order = {"do_estimate_ready": 0, "estimate_now": 1, "estimate_after_sensitivity": 2, "prepare_estimation": 3, "graph_review": 4, "identification_review": 5, "diagnostic_only": 6, "observe_more": 7, "blocked": 8, "no_action": 9}
    report.sort(key=lambda r: (order.get(_as_str(r.get("final_recommendation")), 99), _as_str(r.get("source")), _as_str(r.get("target")), _as_str(r.get("lag"))))
    audit_rows.sort(key=lambda r: (order.get(_as_str(r.get("final_recommendation")), 99), _as_str(r.get("source")), _as_str(r.get("target")), _as_str(r.get("lag"))))

    counts: Dict[str, int] = {}
    claim_counts: Dict[str, int] = {}
    authority_counts: Dict[str, int] = {}
    id_status_counts: Dict[str, int] = {}
    symbolic_evaluator_counts: Dict[str, int] = {}
    do_mode_counts: Dict[str, int] = {}
    robustness_counts: Dict[str, int] = {}
    bootstrap_counts: Dict[str, int] = {}
    query_status_counts: Dict[str, int] = {}
    id_gate_counts: Dict[str, int] = {}
    estimation_gate_counts: Dict[str, int] = {}
    n_authorized_do_estimates = 0
    n_numeric_estimates = 0
    for r in report:
        counts[_as_str(r.get("final_recommendation"))] = counts.get(_as_str(r.get("final_recommendation")), 0) + 1
        claim_counts[_as_str(r.get("claim_status"))] = claim_counts.get(_as_str(r.get("claim_status")), 0) + 1
        query_status_counts[_as_str(r.get("query_status"))] = query_status_counts.get(_as_str(r.get("query_status")), 0) + 1
        id_gate_counts[_as_str(r.get("id_gate_status"))] = id_gate_counts.get(_as_str(r.get("id_gate_status")), 0) + 1
        estimation_gate_counts[_as_str(r.get("estimation_gate_status"))] = estimation_gate_counts.get(_as_str(r.get("estimation_gate_status")), 0) + 1
        authority_counts[_as_str(r.get("authority_level"))] = authority_counts.get(_as_str(r.get("authority_level")), 0) + 1
        id_status_counts[_as_str(r.get("id_status"))] = id_status_counts.get(_as_str(r.get("id_status")), 0) + 1
        symbolic_evaluator_counts[_as_str(r.get("symbolic_evaluator_status"))] = symbolic_evaluator_counts.get(_as_str(r.get("symbolic_evaluator_status")), 0) + 1
        do_mode_counts[_as_str(r.get("do_mode"))] = do_mode_counts.get(_as_str(r.get("do_mode")), 0) + 1
        robustness_counts[_as_str(r.get("robustness_status"))] = robustness_counts.get(_as_str(r.get("robustness_status")), 0) + 1
        bootstrap_counts[_as_str(r.get("bootstrap_status"))] = bootstrap_counts.get(_as_str(r.get("bootstrap_status")), 0) + 1
        if _boolish(r.get("do_authorized")):
            n_authorized_do_estimates += 1
        if _boolish(r.get("estimated")):
            n_numeric_estimates += 1
    manifest = {
        "report_version": REPORT_VERSION,
        "semantics": "Compact review report only; causal_contract/do_contract remain authority gates.",
        "n_rows": len(report),
        "n_audit_rows": len(audit_rows),
        "n_total_rows_before_canonical_split": len(report_all),
        "audit_policy": "Rows outside causal_contract/estimation_plan/effect scope are written to causal_report_audit.csv, not causal_report.csv.",
        "_audit_rows": audit_rows,
        "recommendation_counts": counts,
        "claim_status_counts": claim_counts,
        "query_status_counts": query_status_counts,
        "id_gate_status_counts": id_gate_counts,
        "estimation_gate_status_counts": estimation_gate_counts,
        "authority_level_counts": authority_counts,
        "id_status_counts": id_status_counts,
        "symbolic_evaluator_status_counts": symbolic_evaluator_counts,
        "do_mode_counts": do_mode_counts,
        "robustness_status_counts": robustness_counts,
        "bootstrap_status_counts": bootstrap_counts,
        "n_authorized_do_estimates": n_authorized_do_estimates,
        "n_numeric_estimates": n_numeric_estimates,
        "source_files": {k: str(v) for k, v in paths.items() if v.exists()},
    }
    return report, manifest


def _write_markdown(path: Path, rows: List[Dict[str, object]], manifest: Dict[str, object], max_rows: int = 25) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Amantia compact causal report",
        "",
        "This report is review-oriented only. It does not grant causal authority; `causal_contract.csv` and runtime contracts remain authoritative.",
        "",
        f"Rows: {manifest.get('n_rows', 0)}",
        f"Numeric estimates: {manifest.get('n_numeric_estimates', 0)}",
        "",
        "## Query status counts",
        "",
    ]
    query_counts = manifest.get("query_status_counts", {}) or {}
    if isinstance(query_counts, dict) and query_counts:
        for k, v in sorted(query_counts.items(), key=lambda kv: str(kv[0])):
            lines.append(f"- {k or 'blank'}: {v}")
    else:
        lines.append("- no rows")
    lines += [
        "",
        "## Recommendation counts",
        "",
    ]
    counts = manifest.get("recommendation_counts", {}) or {}
    if isinstance(counts, dict) and counts:
        for k, v in sorted(counts.items(), key=lambda kv: str(kv[0])):
            lines.append(f"- {k}: {v}")
    else:
        lines.append("- no rows")
    lines += ["", "## Authority counts", ""]
    authority_counts = manifest.get("authority_level_counts", {}) or {}
    if isinstance(authority_counts, dict) and authority_counts:
        for k, v in sorted(authority_counts.items(), key=lambda kv: str(kv[0])):
            lines.append(f"- {k or 'blank'}: {v}")
    else:
        lines.append("- no rows")
    lines += ["", "## ID status counts", ""]
    id_counts = manifest.get("id_status_counts", {}) or {}
    if isinstance(id_counts, dict) and id_counts:
        for k, v in sorted(id_counts.items(), key=lambda kv: str(kv[0])):
            lines.append(f"- {k or 'blank'}: {v}")
    else:
        lines.append("- no rows")
    lines += ["", "## Symbolic evaluator counts", ""]
    sym_counts = manifest.get("symbolic_evaluator_status_counts", {}) or {}
    if isinstance(sym_counts, dict) and sym_counts:
        for k, v in sorted(sym_counts.items(), key=lambda kv: str(kv[0])):
            lines.append(f"- {k or 'blank'}: {v}")
    else:
        lines.append("- no rows")
    lines += ["", "## Do-estimation counts", ""]
    lines.append(f"- authorized do-estimates: {manifest.get('n_authorized_do_estimates', 0)}")
    do_counts = manifest.get("do_mode_counts", {}) or {}
    if isinstance(do_counts, dict) and do_counts:
        for k, v in sorted(do_counts.items(), key=lambda kv: str(kv[0])):
            if k:
                lines.append(f"- {k}: {v}")
    else:
        lines.append("- no do-estimate rows")
    lines += ["", "## Do-estimation robustness", ""]
    robust_counts = manifest.get("robustness_status_counts", {}) or {}
    if isinstance(robust_counts, dict) and robust_counts:
        for k, v in sorted(robust_counts.items(), key=lambda kv: str(kv[0])):
            if k:
                lines.append(f"- {k}: {v}")
    else:
        lines.append("- no robustness rows")
    bootstrap_counts = manifest.get("bootstrap_status_counts", {}) or {}
    if isinstance(bootstrap_counts, dict) and bootstrap_counts:
        lines.append("")
        lines.append("Bootstrap status:")
        for k, v in sorted(bootstrap_counts.items(), key=lambda kv: str(kv[0])):
            if k:
                lines.append(f"- {k}: {v}")
    lines += ["", "## SCM Identification & Do-Estimation", "", "| query status | estimated | recommendation | source | target | lag | ID gate | estimation gate | claim | strategy | effect | CI | support | robustness | next action |", "|---|---:|---|---|---|---:|---|---|---|---|---:|---|---|---|---|"]
    for r in rows[:max_rows]:
        def cell(c: str) -> str:
            return _as_str(r.get(c)).replace("|", "/")
        ci = f"{cell('ci_low')}..{cell('ci_high')}" if cell('ci_low') or cell('ci_high') else ""
        support = f"{cell('support_n_low')}/{cell('support_n_high')}" if cell('support_n_low') or cell('support_n_high') else ""
        robustness = cell('robustness_status') or cell('bootstrap_status') or cell('sensitivity_warning')
        strategy = cell('do_mode') or cell('recommended_estimator') or cell('symbolic_estimator_route') or cell('id_status')
        lines.append(
            f"| {cell('query_status')} | {cell('estimated')} | {cell('final_recommendation')} | {cell('source')} | {cell('target')} | {cell('lag')} | "
            f"{cell('id_gate_status')} | {cell('estimation_gate_status')} | {cell('claim_status')} | {strategy} | {cell('effect_estimate')} | {ci} | {support} | {robustness} | {cell('next_action')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_causal_report(out_dir: str = "out") -> Dict[str, str]:
    out = Path(out_dir)
    rows, manifest = build_causal_report(out_dir=out_dir)
    csv_path = out / "causal_report.csv"
    audit_csv_path = out / "causal_report_audit.csv"
    json_path = out / "causal_report.json"
    audit_json_path = out / "causal_report_audit.json"
    md_path = out / "causal_report.md"
    manifest_path = out / "causal_report_manifest.json"
    audit_rows = list(manifest.pop("_audit_rows", []))
    _write_csv(csv_path, rows, REPORT_COLUMNS)
    _write_csv(audit_csv_path, audit_rows, REPORT_COLUMNS)
    _write_json(json_path, rows)
    _write_json(audit_json_path, audit_rows)
    _write_json(manifest_path, manifest)
    _write_markdown(md_path, rows, manifest)
    return {
        "causal_report_csv": str(csv_path),
        "causal_report_audit_csv": str(audit_csv_path),
        "causal_report_json": str(json_path),
        "causal_report_audit_json": str(audit_json_path),
        "causal_report_md": str(md_path),
        "causal_report_manifest_json": str(manifest_path),
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Write compact cross-layer causal report")
    ap.add_argument("--out-dir", default="out")
    args = ap.parse_args()
    print(json.dumps(write_causal_report(out_dir=args.out_dir), indent=2))


__all__ = ["REPORT_COLUMNS", "REPORT_VERSION", "build_causal_report", "write_causal_report"]
