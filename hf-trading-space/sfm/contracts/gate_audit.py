"""Unified gate audit for Amantia.

Step 13 scope
-------------
The Discovery gate audit remains useful, but once SCM identification, symbolic
formula routing, causal_contract.csv, and do-estimation exist, users need a
single review file that explains why a candidate is authorized, blocked, or
kept diagnostic-only.

This module writes ``out/gate_audit.csv`` as a cross-layer audit. The original
Discovery audit is preserved at ``out/discovery/gate_audit.csv`` when present.
The audit is explanatory only; it does not grant authority. Authority remains
in ``causal_contract.csv`` and ``scm_parts.do_contract``.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

GATE_AUDIT_VERSION = 3

UNIFIED_GATE_AUDIT_COLUMNS: List[str] = [
    "gate_id", "proposal_id", "insight_id", "source", "target", "lag",
    "gate_decision", "gate_stage", "gate_reason", "next_action",
    "authority_level", "estimation_enabled", "identification_status", "id_status",
    "symbolic_formula_status", "symbolic_evaluator_status", "symbolic_formula_evaluable",
    "symbolic_numeric_estimator_ready", "symbolic_estimator_route",
    "hedge_detected", "hedge_status", "recursive_id_status", "c_factor_status", "district_status",
    "do_authorized", "do_mode", "effect_estimate", "ci_low", "ci_high",
    "support_n_low", "support_n_high", "support_min", "support_ratio", "support_n_mid",
    "overlap_score", "bootstrap_status", "bootstrap_success_n", "ci_width",
    "ci_width_to_effect_ratio", "robustness_status", "extrapolation_risk", "sensitivity_warning",
    "effect_semantics", "reason_codes",
    "discovery_track", "final_decision", "drop_reason", "keep_flag",
    "mci_status", "hypothesis_signal_grade", "safety_risk_grade", "selection_score",
    "scm_role_hint", "backdoor_status", "adjustment_set", "recommended_estimator",
]


def _s(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null", "nat"} else text


def _truthy(value: object) -> bool:
    text = _s(value).lower()
    if text in {"1", "true", "yes", "y", "on", "enabled", "identified_estimable"}:
        return True
    if text in {"", "0", "false", "no", "n", "off", "disabled"}:
        return False
    try:
        return float(text) != 0.0
    except (TypeError, ValueError, OverflowError):
        return False


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
    iid = _s(row.get("insight_id")) or _s(row.get("proposal_id")) or _s(row.get("effect_id"))
    source = _s(row.get("source") or row.get("treatment_col") or row.get("treatment"))
    target = _s(row.get("target") or row.get("target_col") or row.get("outcome_col") or row.get("outcome"))
    lag = _s(row.get("lag"))
    return iid, source, target, lag


def _key(row: Dict[str, object]) -> str:
    iid, source, target, lag = _identity(row)
    if iid and not iid.startswith("do:"):
        return f"id::{iid}"
    return f"edge::{source}->{target}@{lag}"


def _edge_key(row: Dict[str, object]) -> str:
    _iid, source, target, lag = _identity(row)
    return f"edge::{source}->{target}@{lag}"


def _edge_prefix(row: Dict[str, object]) -> str:
    _iid, source, target, _lag = _identity(row)
    return f"edge::{source}->{target}@" if (source or target) else ""


def _merge_rows(base: Dict[str, Dict[str, str]], rows: List[Dict[str, str]]) -> None:
    for row in rows:
        if not row:
            continue
        key = _key(row)
        edge_key = _edge_key(row)
        if key == "edge::->@" and not _s(row.get("insight_id")):
            continue
        prefix = _edge_prefix(row)
        prefix_match = next((k for k in base if prefix and k.startswith(prefix)), "")
        existing_key = key if key in base else edge_key if edge_key in base else prefix_match or key
        cur = base.setdefault(existing_key, {})
        for k, v in row.items():
            val = _s(v)
            if val:
                cur[k] = val
        if edge_key and edge_key != existing_key and edge_key not in base:
            base[edge_key] = cur


def _paths(out: Path) -> Dict[str, Path]:
    return {
        "discovery_gate_audit": out / "discovery" / "gate_audit.csv",
        "root_gate_audit_existing": out / "gate_audit.csv",
        "discovery_scoring": out / "discovery_scoring.csv",
        "pcmci_scm_bridge": out / "pcmci_scm_bridge.csv",
        "scm_edges": out / "scm" / "scm_edges.csv",
        "id_algorithm_audit": out / "scm" / "id_algorithm_audit.csv",
        "symbolic_evaluation": out / "scm" / "symbolic_evaluation.csv",
        "symbolic_numeric_estimates": out / "scm" / "symbolic_numeric_estimates.csv",
        "symbolic_numeric_diagnostics": out / "scm" / "symbolic_numeric_diagnostics.csv",
        "do_estimates": out / "scm" / "do_estimates.csv",
        "do_diagnostics": out / "scm" / "do_diagnostics.csv",
        "causal_contract": out / "causal_contract.csv",
        "estimation_plan": out / "estimation" / "estimation_plan.csv",
        "effect_estimates": out / "estimation" / "effect_estimates.csv",
        "sensitivity_analysis": out / "estimation" / "sensitivity_analysis.csv",
    }


def _decision(row: Dict[str, str]) -> Tuple[str, str, str, str]:
    authority = _s(row.get("authority_level")).lower()
    id_status = _s(row.get("id_status")).lower()
    sym_status = _s(row.get("symbolic_formula_status")).lower()
    evaluator_status = _s(row.get("symbolic_evaluator_status")).lower()
    c_factor_status = _s(row.get("c_factor_status")).lower()
    recursive_status = _s(row.get("recursive_id_status")).lower()
    final_decision = _s(row.get("final_decision")).lower()
    drop_reason = _s(row.get("drop_reason"))
    reason_codes = _s(row.get("reason_codes"))
    do_mode = _s(row.get("do_mode"))
    robustness_status = _s(row.get("robustness_status")).lower()
    bootstrap_status = _s(row.get("bootstrap_status")).lower()
    sensitivity_warning = _s(row.get("sensitivity_warning"))

    if robustness_status.startswith("blocked") or do_mode.startswith("blocked_low_support"):
        return (
            "blocked_low_support_or_nonfinite",
            "do_outputs",
            sensitivity_warning or robustness_status or reason_codes or "symbolic_numeric_robustness_blocked",
            "do_not_claim; collect more support/overlap or inspect numeric diagnostics",
        )
    if robustness_status.startswith("diagnostic_only") or do_mode.startswith("diagnostic_only"):
        return (
            "diagnostic_only_robustness_warning",
            "do_outputs",
            sensitivity_warning or robustness_status or reason_codes or "symbolic_numeric_robustness_warning",
            "treat as diagnostic only; improve overlap/bootstrap stability before claim",
        )
    if bootstrap_status == "weak" and reason_codes:
        return (
            "diagnostic_only_weak_bootstrap",
            "do_outputs",
            sensitivity_warning or reason_codes,
            "increase data/support or reduce model complexity before causal claim",
        )

    if _truthy(row.get("do_authorized")):
        return (
            "authorized_do_estimate",
            "do_outputs",
            reason_codes or do_mode or "contract_authorized_do_estimate_available",
            "review effect size, CI, support and sensitivity before external claim",
        )
    if "blocked" in authority or id_status.startswith("blocked") or _truthy(row.get("hedge_detected")):
        blocker = id_status or authority or _s(row.get("hedge_status")) or "blocked_by_id_or_contract"
        return ("blocked_id_algorithm", "scm_identification", blocker, "do_not_estimate; inspect SCM ID audit")
    if "unresolved" in c_factor_status or "requires" in recursive_status or "unsupported" in sym_status:
        return (
            "blocked_requires_full_id_or_c_factor",
            "scm_identification",
            c_factor_status or recursive_status or sym_status,
            "keep diagnostic-only until recursive ID/symbolic c-factor route is implemented",
        )
    if authority == "identified_estimable" and _truthy(row.get("estimation_enabled")):
        return (
            "authorized_estimation_pending_do_output",
            "causal_contract",
            _s(row.get("identification_status")) or id_status or "identified_estimable_contract",
            "run do_outputs or inspect do_diagnostics for support/data blockers",
        )
    if _truthy(row.get("symbolic_numeric_estimator_ready")):
        return (
            "symbolic_numeric_ready",
            "symbolic_evaluator",
            evaluator_status or "symbolic_formula_numeric_ready",
            "evaluate numeric symbolic formula and write do_estimates.csv",
        )
    if _truthy(row.get("symbolic_formula_evaluable")):
        return (
            "symbolic_formula_evaluable",
            "symbolic_evaluator",
            evaluator_status or "formula_evaluable_but_numeric_route_pending",
            "connect/verify numeric evaluator before effect claim",
        )
    if final_decision in {"drop", "blocked", "reject", "dropped"} or _s(row.get("keep_flag")).lower() in {"0", "false", "no"}:
        return ("blocked_discovery_gate", "discovery", drop_reason or "discovery_gate_rejected_candidate", "keep in audit only")
    if authority or id_status or evaluator_status:
        return (
            "diagnostic_only",
            "contract_or_identification",
            _s(row.get("reason")) or _s(row.get("assumption_notes")) or authority or id_status or evaluator_status,
            "do not make effect claim; strengthen ID or estimator route",
        )
    return ("diagnostic_only", "discovery", drop_reason or "no_downstream_authority", "collect more data or strengthen SCM evidence")


def _compact_row(row: Dict[str, str], idx: int) -> Dict[str, object]:
    iid, source, target, lag = _identity(row)
    decision, stage, reason, next_action = _decision(row)
    return {
        "gate_id": f"gate::{iid or source + '->' + target + '@' + lag or idx}",
        "proposal_id": _s(row.get("proposal_id")),
        "insight_id": _s(row.get("insight_id")) or (iid if not iid.startswith("do:") else ""),
        "source": source,
        "target": target,
        "lag": lag,
        "gate_decision": decision,
        "gate_stage": stage,
        "gate_reason": reason,
        "next_action": next_action,
        "authority_level": _s(row.get("authority_level")),
        "estimation_enabled": _s(row.get("estimation_enabled")),
        "identification_status": _s(row.get("identification_status")),
        "id_status": _s(row.get("id_status")),
        "symbolic_formula_status": _s(row.get("symbolic_formula_status")),
        "symbolic_evaluator_status": _s(row.get("symbolic_evaluator_status")),
        "symbolic_formula_evaluable": _s(row.get("symbolic_formula_evaluable")),
        "symbolic_numeric_estimator_ready": _s(row.get("symbolic_numeric_estimator_ready")),
        "symbolic_estimator_route": _s(row.get("symbolic_estimator_route")),
        "hedge_detected": _s(row.get("hedge_detected")),
        "hedge_status": _s(row.get("hedge_status")),
        "recursive_id_status": _s(row.get("recursive_id_status")),
        "c_factor_status": _s(row.get("c_factor_status")),
        "district_status": _s(row.get("district_status")),
        "do_authorized": _s(row.get("do_authorized")),
        "do_mode": _s(row.get("do_mode")),
        "effect_estimate": _s(row.get("effect_estimate")),
        "ci_low": _s(row.get("ci_low")),
        "ci_high": _s(row.get("ci_high")),
        "support_n_low": _s(row.get("support_n_low")),
        "support_n_high": _s(row.get("support_n_high")),
        "support_min": _s(row.get("support_min")),
        "support_ratio": _s(row.get("support_ratio")),
        "support_n_mid": _s(row.get("support_n_mid")),
        "overlap_score": _s(row.get("overlap_score")),
        "bootstrap_status": _s(row.get("bootstrap_status")),
        "bootstrap_success_n": _s(row.get("bootstrap_success_n")),
        "ci_width": _s(row.get("ci_width")),
        "ci_width_to_effect_ratio": _s(row.get("ci_width_to_effect_ratio")),
        "robustness_status": _s(row.get("robustness_status")),
        "extrapolation_risk": _s(row.get("extrapolation_risk")),
        "sensitivity_warning": _s(row.get("sensitivity_warning")),
        "effect_semantics": _s(row.get("effect_semantics")),
        "reason_codes": _s(row.get("reason_codes")),
        "discovery_track": _s(row.get("discovery_track")),
        "final_decision": _s(row.get("final_decision")),
        "drop_reason": _s(row.get("drop_reason")),
        "keep_flag": _s(row.get("keep_flag")),
        "mci_status": _s(row.get("mci_status")),
        "hypothesis_signal_grade": _s(row.get("hypothesis_signal_grade")),
        "safety_risk_grade": _s(row.get("safety_risk_grade")),
        "selection_score": _s(row.get("selection_score")),
        "scm_role_hint": _s(row.get("scm_role_hint")),
        "backdoor_status": _s(row.get("backdoor_status")),
        "adjustment_set": _s(row.get("adjustment_set") or row.get("candidate_adjustment_set") or row.get("conditioning_set_used")),
        "recommended_estimator": _s(row.get("recommended_estimator")),
    }


def build_unified_gate_audit(out_dir: str = "out") -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    out = Path(out_dir)
    paths = _paths(out)
    base: Dict[str, Dict[str, str]] = {}
    # Discovery first, then progressively higher authority layers override/add fields.
    for name in [
        "discovery_gate_audit", "root_gate_audit_existing", "discovery_scoring", "pcmci_scm_bridge", "scm_edges",
        "id_algorithm_audit", "symbolic_evaluation", "symbolic_numeric_estimates", "symbolic_numeric_diagnostics",
        "causal_contract", "estimation_plan", "sensitivity_analysis", "effect_estimates", "do_estimates", "do_diagnostics",
    ]:
        if name == "root_gate_audit_existing" and paths.get("discovery_gate_audit", Path()).exists():
            # Avoid duplicating the same Discovery audit when output_writer has already mirrored it.
            continue
        _merge_rows(base, _read_csv(paths[name]))

    rows: List[Dict[str, object]] = []
    seen = set()
    for row in base.values():
        obj_id = id(row)
        if obj_id in seen:
            continue
        seen.add(obj_id)
        if _identity(row)[0] or _identity(row)[1] or _identity(row)[2]:
            rows.append(_compact_row(row, len(rows)))
    order = {
        "authorized_do_estimate": 0,
        "diagnostic_only_robustness_warning": 1,
        "diagnostic_only_weak_bootstrap": 2,
        "blocked_low_support_or_nonfinite": 3,
        "authorized_estimation_pending_do_output": 4,
        "symbolic_numeric_ready": 5,
        "symbolic_formula_evaluable": 6,
        "diagnostic_only": 7,
        "blocked_requires_full_id_or_c_factor": 8,
        "blocked_id_algorithm": 9,
        "blocked_discovery_gate": 10,
    }
    rows.sort(key=lambda r: (order.get(_s(r.get("gate_decision")), 99), _s(r.get("source")), _s(r.get("target")), _s(r.get("lag"))))
    decision_counts: Dict[str, int] = {}
    stage_counts: Dict[str, int] = {}
    for row in rows:
        decision_counts[_s(row.get("gate_decision"))] = decision_counts.get(_s(row.get("gate_decision")), 0) + 1
        stage_counts[_s(row.get("gate_stage"))] = stage_counts.get(_s(row.get("gate_stage")), 0) + 1
    manifest = {
        "gate_audit_version": GATE_AUDIT_VERSION,
        "semantics": "Unified explanatory audit. Authority remains causal_contract.csv/do_contract.py.",
        "n_rows": len(rows),
        "gate_decision_counts": decision_counts,
        "gate_stage_counts": stage_counts,
        "source_files": {k: str(v) for k, v in paths.items() if v.exists()},
    }
    return rows, manifest


def write_unified_gate_audit(out_dir: str = "out") -> Dict[str, str]:
    out = Path(out_dir)
    rows, manifest = build_unified_gate_audit(out_dir=out_dir)
    csv_path = out / "gate_audit.csv"
    json_path = out / "gate_audit.json"
    manifest_path = out / "gate_audit_manifest.json"
    _write_csv(csv_path, rows, UNIFIED_GATE_AUDIT_COLUMNS)
    _write_json(json_path, rows)
    _write_json(manifest_path, manifest)
    return {
        "gate_audit_csv": str(csv_path),
        "gate_audit_json": str(json_path),
        "gate_audit_manifest_json": str(manifest_path),
    }


__all__ = [
    "GATE_AUDIT_VERSION",
    "UNIFIED_GATE_AUDIT_COLUMNS",
    "build_unified_gate_audit",
    "write_unified_gate_audit",
]
