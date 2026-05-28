from __future__ import annotations

"""End-to-end authority audit for SCM do-estimation.

Stdlib-only ledger tying ``id_algorithm_audit -> causal_contract -> do_estimates``
together. It reports authority; it never promotes a claim.
"""

import csv
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

AUDIT_VERSION = 1

AUTHORITY_AUDIT_COLUMNS: List[str] = [
    "audit_id", "treatment", "outcome", "final_decision",
    "causal_estimate_authorized", "diagnostic_only", "estimate_row_present",
    "diagnostic_row_present", "contract_row_present", "contract_index",
    "authority_level", "authority_reason", "canonical_id_authority",
    "source_artifacts", "source_authority", "id_status", "id_identified",
    "id_algorithm_level", "symbolic_formula_status", "symbolic_evaluator_status",
    "symbolic_numeric_estimator_ready", "symbolic_estimator_route",
    "hedge_detected", "hedge_status", "recursive_id_status", "c_factor_status",
    "district_status", "identification_strategy", "identification_route",
    "estimation_enabled", "do_mode", "effect_semantics", "effect_estimate",
    "ci_low", "ci_high", "robustness_status", "support_min", "support_ratio",
    "extrapolation_risk", "reason_codes", "audit_reason_codes",
]

ID_HARD_BLOCK_TOKENS = (
    "blocked", "unsupported_requires_full_id", "requires_full_id",
    "requires_symbolic_c_factor", "invalid_backdoor", "invalid_frontdoor",
    "directed_cycle",
)

AUTHORITY_FATAL_CODES = {
    "MISSING_CANONICAL_ID_AUTHORITY",
    "ID_STATUS_BLOCKED",
    "SYMBOLIC_FORMULA_NOT_IDENTIFIED",
    "ID_HEDGE_DETECTED",
    "ID_RECURSIVE_BLOCKED",
    "ID_C_FACTOR_UNRESOLVED",
    "ID_DISTRICT_POSSIBLE_HEDGE",
    "ESTIMATION_NOT_ENABLED",
    "IDENTIFIED_BUT_ESTIMATOR_NOT_ENABLED",
    "GRAPH_REVIEW_ONLY",
    "CONTRACT_BLOCKED",
}


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


def _write_csv(path: str | os.PathLike, rows: Iterable[Mapping[str, object]], columns: List[str]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})


def _s(value: object) -> str:
    text = "" if value is None else str(value).strip()
    return "" if text.lower() in {"nan", "none", "null", "nat"} else text


def _truthy(value: object) -> bool:
    text = _s(value).lower()
    if text in {"1", "true", "yes", "y", "on", "enabled", "identified", "identified_estimable"}:
        return True
    if text in {"", "0", "false", "no", "n", "off", "disabled"}:
        return False
    try:
        return float(text) != 0.0
    except (TypeError, ValueError, OverflowError):
        return False


def _tokens(value: object) -> List[str]:
    return [tok.strip().lower() for tok in _s(value).replace(",", "|").split("|") if tok.strip()]


def _append_code(codes: List[str], code: str) -> None:
    if code and code not in codes:
        codes.append(code)


def _key(row: Mapping[str, object]) -> Tuple[str, str]:
    treatment = _s(row.get("treatment") or row.get("treatment_col") or row.get("source") or row.get("cause") or row.get("from"))
    outcome = _s(row.get("outcome") or row.get("outcome_col") or row.get("target") or row.get("effect") or row.get("to"))
    return treatment, outcome


def _index_latest(rows: Iterable[Mapping[str, object]]) -> Dict[Tuple[str, str], Dict[str, str]]:
    indexed: Dict[Tuple[str, str], Dict[str, str]] = {}
    for row in rows:
        key = _key(row)
        if key != ("", ""):
            indexed[key] = dict(row)
    return indexed


def _id_block_codes(contract: Mapping[str, object]) -> List[str]:
    codes: List[str] = []
    id_status = _s(contract.get("id_status") or contract.get("identification_status")).lower()
    symbolic = _s(contract.get("symbolic_formula_status")).lower()
    recursive = _s(contract.get("recursive_id_status")).lower()
    cfactor = _s(contract.get("c_factor_status")).lower()
    district = _s(contract.get("district_status")).lower()
    hedge = _s(contract.get("hedge_status")).lower()
    if any(tok in id_status for tok in ID_HARD_BLOCK_TOKENS):
        _append_code(codes, "ID_STATUS_BLOCKED")
    if symbolic and symbolic != "identified_symbolic_formula":
        _append_code(codes, "SYMBOLIC_FORMULA_NOT_IDENTIFIED")
    if _truthy(contract.get("hedge_detected")) or "possible_hedge" in hedge:
        _append_code(codes, "ID_HEDGE_DETECTED")
    if recursive.startswith("blocked") or "requires_symbolic_c_factor" in recursive:
        _append_code(codes, "ID_RECURSIVE_BLOCKED")
    if "unresolved" in cfactor or "requires_recursive" in cfactor:
        _append_code(codes, "ID_C_FACTOR_UNRESOLVED")
    if "possible_hedge" in district:
        _append_code(codes, "ID_DISTRICT_POSSIBLE_HEDGE")
    return codes


def _canonical_authority(contract: Mapping[str, object]) -> bool:
    if _truthy(contract.get("canonical_id_authority")):
        return True
    artifacts = set(_tokens(contract.get("source_artifacts")))
    sources = set(_tokens(contract.get("source_authority")))
    if "id_algorithm_audit" in artifacts or "scm_id_algorithm" in sources:
        return True
    if _s(contract.get("id_algorithm_level")):
        return True
    return False


def _classify_row(*, contract: Mapping[str, object], estimate: Optional[Mapping[str, object]], diagnostic: Optional[Mapping[str, object]]) -> Tuple[str, int, int, str]:
    codes: List[str] = []
    estimate = estimate or {}
    diagnostic = diagnostic or {}
    contract_present = bool(contract)
    estimate_present = bool(estimate)
    diagnostic_present = bool(diagnostic)
    contract_auth = _s(contract.get("authority_level"))
    estimate_do_authorized = _truthy(estimate.get("do_authorized"))
    diag_do_authorized = _truthy(diagnostic.get("do_authorized"))
    diagnostic_allowed = _truthy(estimate.get("diagnostic_estimation_allowed") or diagnostic.get("diagnostic_estimation_allowed"))
    do_mode = _s(estimate.get("do_mode") or diagnostic.get("do_mode"))

    if not contract_present:
        _append_code(codes, "NO_CAUSAL_CONTRACT_ROW")
    if contract_present and not _canonical_authority(contract):
        _append_code(codes, "MISSING_CANONICAL_ID_AUTHORITY")
    for code in _id_block_codes(contract):
        _append_code(codes, code)
    if contract_present and not _truthy(contract.get("estimation_enabled")):
        _append_code(codes, "ESTIMATION_NOT_ENABLED")
    if contract_present and contract_auth == "identified_needs_estimation":
        _append_code(codes, "IDENTIFIED_BUT_ESTIMATOR_NOT_ENABLED")
    if contract_present and contract_auth == "graph_review":
        _append_code(codes, "GRAPH_REVIEW_ONLY")
    if contract_present and contract_auth.startswith("blocked"):
        _append_code(codes, "CONTRACT_BLOCKED")
    if estimate_present and not estimate_do_authorized and not diagnostic_allowed:
        _append_code(codes, _s(estimate.get("reason_codes")) or "ESTIMATE_ROW_NOT_AUTHORIZED")
    if diagnostic_present and not diag_do_authorized and not diagnostic_allowed:
        _append_code(codes, _s(diagnostic.get("diagnostic_notes")) or "DIAGNOSTIC_ROW_BLOCKED")
    if not estimate_present and contract_present and _truthy(contract.get("estimation_enabled")) and not codes:
        _append_code(codes, "ESTIMATION_ENABLED_BUT_NO_ESTIMATE_ROW")

    has_fatal_code = any(code in AUTHORITY_FATAL_CODES for code in codes)
    if (
        estimate_do_authorized
        and contract_present
        and _canonical_authority(contract)
        and _truthy(contract.get("estimation_enabled"))
        and not has_fatal_code
    ):
        return "authorized_do_estimate", 1, 0, "|".join(codes)
    if diagnostic_allowed or "diagnostic" in do_mode:
        return "diagnostic_only_no_causal_authority", 0, 1, "|".join(codes)
    if any(code in codes for code in ["ID_STATUS_BLOCKED", "SYMBOLIC_FORMULA_NOT_IDENTIFIED", "ID_HEDGE_DETECTED", "ID_RECURSIVE_BLOCKED", "ID_C_FACTOR_UNRESOLVED", "ID_DISTRICT_POSSIBLE_HEDGE"]):
        return "blocked_by_id_algorithm", 0, 0, "|".join(codes)
    if "MISSING_CANONICAL_ID_AUTHORITY" in codes:
        return "blocked_missing_canonical_id_authority", 0, 0, "|".join(codes)
    if "IDENTIFIED_BUT_ESTIMATOR_NOT_ENABLED" in codes or contract_auth == "identified_needs_estimation":
        return "identified_needs_estimator", 0, 0, "|".join(codes)
    if "GRAPH_REVIEW_ONLY" in codes or contract_auth == "graph_review":
        return "graph_review_only", 0, 0, "|".join(codes)
    if not contract_present:
        return "orphan_estimate_without_contract", 0, 0, "|".join(codes)
    return "blocked_or_not_ready", 0, 0, "|".join(codes)


def _make_audit_row(*, audit_id: str, treatment: str, outcome: str, contract: Mapping[str, object], estimate: Mapping[str, object], diagnostic: Mapping[str, object], contract_index: int, final_decision: str, causal_estimate_authorized: int, diagnostic_only: int, audit_reason_codes: str) -> Dict[str, object]:
    reason_codes = _s(estimate.get("reason_codes") or diagnostic.get("diagnostic_notes") or contract.get("id_reason_codes") or contract.get("authority_reason"))
    return {
        "audit_id": audit_id, "treatment": treatment, "outcome": outcome,
        "final_decision": final_decision,
        "causal_estimate_authorized": int(causal_estimate_authorized),
        "diagnostic_only": int(diagnostic_only),
        "estimate_row_present": int(bool(estimate)), "diagnostic_row_present": int(bool(diagnostic)),
        "contract_row_present": int(bool(contract)), "contract_index": contract_index,
        "authority_level": _s(contract.get("authority_level") or estimate.get("authority_level")),
        "authority_reason": _s(contract.get("authority_reason")),
        "canonical_id_authority": int(_canonical_authority(contract)) if contract else 0,
        "source_artifacts": _s(contract.get("source_artifacts")),
        "source_authority": _s(contract.get("source_authority")),
        "id_status": _s(contract.get("id_status")),
        "id_identified": _s(contract.get("id_identified") or contract.get("identified")),
        "id_algorithm_level": _s(contract.get("id_algorithm_level")),
        "symbolic_formula_status": _s(contract.get("symbolic_formula_status")),
        "symbolic_evaluator_status": _s(contract.get("symbolic_evaluator_status")),
        "symbolic_numeric_estimator_ready": _s(contract.get("symbolic_numeric_estimator_ready")),
        "symbolic_estimator_route": _s(contract.get("symbolic_estimator_route")),
        "hedge_detected": _s(contract.get("hedge_detected")),
        "hedge_status": _s(contract.get("hedge_status")),
        "recursive_id_status": _s(contract.get("recursive_id_status")),
        "c_factor_status": _s(contract.get("c_factor_status")),
        "district_status": _s(contract.get("district_status")),
        "identification_strategy": _s(contract.get("identification_strategy") or estimate.get("identification_strategy")),
        "identification_route": _s(contract.get("identification_route")),
        "estimation_enabled": _s(contract.get("estimation_enabled") or diagnostic.get("estimation_enabled")),
        "do_mode": _s(estimate.get("do_mode") or diagnostic.get("do_mode")),
        "effect_semantics": _s(estimate.get("effect_semantics")),
        "effect_estimate": _s(estimate.get("effect_estimate")),
        "ci_low": _s(estimate.get("ci_low")), "ci_high": _s(estimate.get("ci_high")),
        "robustness_status": _s(estimate.get("robustness_status") or diagnostic.get("robustness_status")),
        "support_min": _s(estimate.get("support_min") or diagnostic.get("support_min")),
        "support_ratio": _s(estimate.get("support_ratio") or diagnostic.get("support_ratio")),
        "extrapolation_risk": _s(estimate.get("extrapolation_risk") or diagnostic.get("extrapolation_risk")),
        "reason_codes": reason_codes, "audit_reason_codes": audit_reason_codes,
    }


def build_do_authority_audit(out_dir: str = "out", contract_path: Optional[str] = None, estimates_path: Optional[str] = None, diagnostics_path: Optional[str] = None) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    out = Path(out_dir)
    contract_rows = _read_csv(contract_path or out / "causal_contract.csv")
    estimate_rows = _read_csv(estimates_path or out / "scm" / "do_estimates.csv")
    diagnostic_rows = _read_csv(diagnostics_path or out / "scm" / "do_diagnostics.csv")
    estimates_by_key = _index_latest(estimate_rows)
    diagnostics_by_key = _index_latest(diagnostic_rows)
    seen: set[Tuple[str, str]] = set()
    rows: List[Dict[str, object]] = []
    for idx, contract in enumerate(contract_rows):
        key = _key(contract)
        if key == ("", ""):
            continue
        seen.add(key)
        estimate = estimates_by_key.get(key, {})
        diagnostic = diagnostics_by_key.get(key, {})
        decision, causal, diagnostic_only, audit_codes = _classify_row(contract=contract, estimate=estimate, diagnostic=diagnostic)
        rows.append(_make_audit_row(audit_id=f"audit:{key[0]}->{key[1]}", treatment=key[0], outcome=key[1], contract=contract, estimate=estimate, diagnostic=diagnostic, contract_index=idx, final_decision=decision, causal_estimate_authorized=causal, diagnostic_only=diagnostic_only, audit_reason_codes=audit_codes))
    for key, estimate in sorted(estimates_by_key.items()):
        if key in seen:
            continue
        diagnostic = diagnostics_by_key.get(key, {})
        decision, causal, diagnostic_only, audit_codes = _classify_row(contract={}, estimate=estimate, diagnostic=diagnostic)
        rows.append(_make_audit_row(audit_id=f"orphan_estimate:{key[0]}->{key[1]}", treatment=key[0], outcome=key[1], contract={}, estimate=estimate, diagnostic=diagnostic, contract_index=-1, final_decision=decision, causal_estimate_authorized=causal, diagnostic_only=diagnostic_only, audit_reason_codes=audit_codes))
    for key, diagnostic in sorted(diagnostics_by_key.items()):
        if key in seen or key in estimates_by_key:
            continue
        decision, causal, diagnostic_only, audit_codes = _classify_row(contract={}, estimate={}, diagnostic=diagnostic)
        rows.append(_make_audit_row(audit_id=f"orphan_diagnostic:{key[0]}->{key[1]}", treatment=key[0], outcome=key[1], contract={}, estimate={}, diagnostic=diagnostic, contract_index=-1, final_decision=decision, causal_estimate_authorized=causal, diagnostic_only=diagnostic_only, audit_reason_codes=audit_codes))
    decision_counts: Dict[str, int] = {}
    for row in rows:
        decision = _s(row.get("final_decision"))
        decision_counts[decision] = decision_counts.get(decision, 0) + 1
    manifest = {
        "do_authority_audit_version": AUDIT_VERSION,
        "contract_rows": len(contract_rows), "estimate_rows": len(estimate_rows),
        "diagnostic_rows": len(diagnostic_rows), "audit_rows": len(rows),
        "decision_counts": decision_counts,
        "policy": "No estimator row is treated as causal authority unless it is contract-backed, canonical-ID-backed, and do_authorized=1.",
        "columns": AUTHORITY_AUDIT_COLUMNS,
    }
    return rows, manifest


def write_do_authority_audit(out_dir: str = "out", contract_path: Optional[str] = None, estimates_path: Optional[str] = None, diagnostics_path: Optional[str] = None) -> Dict[str, str]:
    rows, manifest = build_do_authority_audit(out_dir=out_dir, contract_path=contract_path, estimates_path=estimates_path, diagnostics_path=diagnostics_path)
    out = Path(out_dir) / "scm"
    audit_path = out / "do_authority_audit.csv"
    manifest_path = out / "do_authority_manifest.json"
    _write_csv(audit_path, rows, AUTHORITY_AUDIT_COLUMNS)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump({**manifest, "do_authority_audit_csv": str(audit_path)}, f, ensure_ascii=False, indent=2)
    return {"do_authority_audit": str(audit_path), "do_authority_manifest": str(manifest_path)}


__all__ = ["AUDIT_VERSION", "AUTHORITY_AUDIT_COLUMNS", "AUTHORITY_FATAL_CODES", "build_do_authority_audit", "write_do_authority_audit"]
