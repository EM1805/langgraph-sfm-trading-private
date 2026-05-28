from __future__ import annotations

"""Consistency validator for SCM do-authority outputs.

This module is intentionally stdlib-only. It checks that the canonical outputs
produced by the SCM-ID/do-estimation path agree with each other:

    id_algorithm_audit -> causal_contract -> do_estimates/do_diagnostics
    -> do_authority_audit

The validator is audit-only. It never promotes a claim and never changes an
estimate. Any inconsistency is written as an explicit row so downstream users can
block, warn, or inspect the package without re-deriving the whole pipeline.
"""

import csv
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

VALIDATOR_VERSION = 1

VALIDATION_COLUMNS: List[str] = [
    "check_id",
    "severity",
    "status",
    "treatment",
    "outcome",
    "check_name",
    "message",
    "contract_present",
    "audit_present",
    "estimate_present",
    "diagnostic_present",
    "canonical_id_authority",
    "estimation_enabled",
    "do_authorized",
    "audit_final_decision",
    "audit_causal_estimate_authorized",
    "authority_level",
    "id_status",
    "symbolic_formula_status",
    "reason_codes",
]

PASS = "pass"
WARN = "warn"
FAIL = "fail"

ID_BLOCK_TOKENS = (
    "blocked",
    "unsupported_requires_full_id",
    "requires_full_id",
    "requires_symbolic_c_factor",
    "invalid_backdoor",
    "invalid_frontdoor",
    "directed_cycle",
    "possible_hedge",
)


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


def _canonical_id_authority(row: Mapping[str, object]) -> bool:
    if _truthy(row.get("canonical_id_authority")):
        return True
    artifacts = set(_tokens(row.get("source_artifacts")))
    sources = set(_tokens(row.get("source_authority")))
    if "id_algorithm_audit" in artifacts or "scm_id_algorithm" in sources:
        return True
    if _s(row.get("id_algorithm_level")):
        return True
    return False


def _id_blocked(row: Mapping[str, object]) -> bool:
    id_status = _s(row.get("id_status") or row.get("identification_status")).lower()
    hedge_status = _s(row.get("hedge_status")).lower()
    recursive = _s(row.get("recursive_id_status")).lower()
    c_factor = _s(row.get("c_factor_status")).lower()
    district = _s(row.get("district_status")).lower()
    symbolic = _s(row.get("symbolic_formula_status")).lower()
    if _truthy(row.get("hedge_detected")):
        return True
    blob = "|".join([id_status, hedge_status, recursive, c_factor, district])
    if any(tok in blob for tok in ID_BLOCK_TOKENS):
        return True
    if symbolic and symbolic != "identified_symbolic_formula":
        return True
    return False


def _make_row(
    *,
    check_id: str,
    severity: str,
    status: str,
    treatment: str,
    outcome: str,
    check_name: str,
    message: str,
    contract: Optional[Mapping[str, object]] = None,
    audit: Optional[Mapping[str, object]] = None,
    estimate: Optional[Mapping[str, object]] = None,
    diagnostic: Optional[Mapping[str, object]] = None,
    reason_codes: str = "",
) -> Dict[str, object]:
    contract = contract or {}
    audit = audit or {}
    estimate = estimate or {}
    diagnostic = diagnostic or {}
    return {
        "check_id": check_id,
        "severity": severity,
        "status": status,
        "treatment": treatment,
        "outcome": outcome,
        "check_name": check_name,
        "message": message,
        "contract_present": int(bool(contract)),
        "audit_present": int(bool(audit)),
        "estimate_present": int(bool(estimate)),
        "diagnostic_present": int(bool(diagnostic)),
        "canonical_id_authority": int(_canonical_id_authority(contract)) if contract else _s(audit.get("canonical_id_authority")),
        "estimation_enabled": _s(contract.get("estimation_enabled")),
        "do_authorized": _s(estimate.get("do_authorized") or diagnostic.get("do_authorized")),
        "audit_final_decision": _s(audit.get("final_decision")),
        "audit_causal_estimate_authorized": _s(audit.get("causal_estimate_authorized")),
        "authority_level": _s(contract.get("authority_level") or audit.get("authority_level")),
        "id_status": _s(contract.get("id_status") or audit.get("id_status")),
        "symbolic_formula_status": _s(contract.get("symbolic_formula_status") or audit.get("symbolic_formula_status")),
        "reason_codes": reason_codes or _s(audit.get("audit_reason_codes") or estimate.get("reason_codes") or diagnostic.get("diagnostic_notes") or contract.get("id_reason_codes")),
    }


def build_do_authority_validation(
    out_dir: str = "out",
    contract_path: Optional[str] = None,
    estimates_path: Optional[str] = None,
    diagnostics_path: Optional[str] = None,
    authority_audit_path: Optional[str] = None,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """Return validation rows and a manifest for SCM do-authority outputs."""
    out = Path(out_dir)
    contract_rows = _read_csv(contract_path or out / "causal_contract.csv")
    estimate_rows = _read_csv(estimates_path or out / "scm" / "do_estimates.csv")
    diagnostic_rows = _read_csv(diagnostics_path or out / "scm" / "do_diagnostics.csv")
    audit_rows = _read_csv(authority_audit_path or out / "scm" / "do_authority_audit.csv")

    contracts = _index_latest(contract_rows)
    estimates = _index_latest(estimate_rows)
    diagnostics = _index_latest(diagnostic_rows)
    audits = _index_latest(audit_rows)
    all_keys = sorted(set(contracts) | set(estimates) | set(diagnostics) | set(audits))

    rows: List[Dict[str, object]] = []

    def add(
        key: Tuple[str, str],
        severity: str,
        status: str,
        name: str,
        message: str,
        *,
        reason_codes: str = "",
    ) -> None:
        check_num = len(rows) + 1
        rows.append(_make_row(
            check_id=f"scm_do_authority_validation:{check_num:04d}",
            severity=severity,
            status=status,
            treatment=key[0],
            outcome=key[1],
            check_name=name,
            message=message,
            contract=contracts.get(key),
            audit=audits.get(key),
            estimate=estimates.get(key),
            diagnostic=diagnostics.get(key),
            reason_codes=reason_codes,
        ))

    for key in all_keys:
        contract = contracts.get(key, {})
        estimate = estimates.get(key, {})
        diagnostic = diagnostics.get(key, {})
        audit = audits.get(key, {})
        audit_decision = _s(audit.get("final_decision"))
        audit_authorized = _truthy(audit.get("causal_estimate_authorized"))
        estimate_authorized = _truthy(estimate.get("do_authorized"))
        contract_estimation_enabled = _truthy(contract.get("estimation_enabled"))
        canonical = _canonical_id_authority(contract)
        blocked_by_id = _id_blocked(contract)

        if estimate and not contract:
            add(key, "error", FAIL, "orphan_estimate_without_contract", "A do_estimates row exists without a matching causal_contract row.", reason_codes="NO_CAUSAL_CONTRACT_ROW")
        if diagnostic and not contract:
            add(key, "warning", WARN, "orphan_diagnostic_without_contract", "A diagnostic row exists without a matching causal_contract row.", reason_codes="NO_CAUSAL_CONTRACT_ROW")
        if contract and not audit:
            add(key, "error", FAIL, "missing_do_authority_audit_row", "A causal_contract row has no matching do_authority_audit row.", reason_codes="MISSING_DO_AUTHORITY_AUDIT_ROW")
        if audit and not contract and audit_decision != "orphan_estimate_without_contract" and not audit_decision.startswith("orphan"):
            add(key, "error", FAIL, "audit_without_contract_not_marked_orphan", "A do_authority_audit row lacks a contract but is not marked as orphan.", reason_codes="AUDIT_ORPHAN_CLASSIFICATION_MISMATCH")

        if estimate_authorized and not canonical:
            add(key, "critical", FAIL, "authorized_estimate_missing_canonical_id", "A do_estimates row is authorized but the contract lacks canonical ID authority.", reason_codes="AUTHORIZED_WITHOUT_CANONICAL_ID")
        if estimate_authorized and not contract_estimation_enabled:
            add(key, "critical", FAIL, "authorized_estimate_without_estimation_enabled", "A do_estimates row is authorized while the contract has estimation_enabled=0.", reason_codes="AUTHORIZED_WITHOUT_ESTIMATION_ENABLED")
        if estimate_authorized and blocked_by_id:
            add(key, "critical", FAIL, "authorized_estimate_despite_id_block", "A do_estimates row is authorized although ID metadata contains a hard blocker.", reason_codes="AUTHORIZED_DESPITE_ID_BLOCK")
        if audit_authorized and (not estimate_authorized or not canonical or not contract_estimation_enabled or blocked_by_id):
            add(key, "critical", FAIL, "audit_authorizes_inconsistent_estimate", "do_authority_audit marks a causal estimate as authorized but source rows do not satisfy the authority contract.", reason_codes="AUDIT_AUTHORITY_INCONSISTENCY")
        if estimate_authorized and audit and audit_decision != "authorized_do_estimate":
            add(key, "critical", FAIL, "estimate_authorized_but_audit_not_authorized", "do_estimates has do_authorized=1 but the final audit decision is not authorized_do_estimate.", reason_codes="ESTIMATE_AUDIT_DECISION_MISMATCH")
        if audit_decision == "authorized_do_estimate" and not audit_authorized:
            add(key, "critical", FAIL, "audit_decision_flag_mismatch", "Audit decision is authorized_do_estimate but causal_estimate_authorized is not true.", reason_codes="AUDIT_DECISION_FLAG_MISMATCH")
        if blocked_by_id and audit_decision == "authorized_do_estimate":
            add(key, "critical", FAIL, "audit_authorized_despite_id_block", "Audit decision authorizes an estimate even though ID metadata is blocked.", reason_codes="AUDIT_AUTHORIZED_DESPITE_ID_BLOCK")
        if contract and contract_estimation_enabled and not estimate and not audit_authorized:
            add(key, "warning", WARN, "enabled_contract_missing_estimate", "The contract enables estimation, but no do_estimates row was produced.", reason_codes="ESTIMATION_ENABLED_BUT_NO_ESTIMATE_ROW")


    # Add one compact PASS row only when no warnings/errors were found. This
    # keeps the CSV useful without creating a large number of noisy pass rows.
    if not rows:
        rows.append(_make_row(
            check_id="scm_do_authority_validation:0000",
            severity="info",
            status=PASS,
            treatment="",
            outcome="",
            check_name="all_authority_outputs_consistent",
            message="No SCM do-authority inconsistencies were detected.",
        ))

    counts: Dict[str, int] = {PASS: 0, WARN: 0, FAIL: 0}
    severity_counts: Dict[str, int] = {}
    for row in rows:
        status = _s(row.get("status")) or WARN
        counts[status] = counts.get(status, 0) + 1
        severity = _s(row.get("severity")) or "warning"
        severity_counts[severity] = severity_counts.get(severity, 0) + 1
    validation_status = "fail" if counts.get(FAIL, 0) else "warn" if counts.get(WARN, 0) else "pass"
    manifest = {
        "do_authority_validator_version": VALIDATOR_VERSION,
        "validation_status": validation_status,
        "contract_rows": len(contract_rows),
        "estimate_rows": len(estimate_rows),
        "diagnostic_rows": len(diagnostic_rows),
        "authority_audit_rows": len(audit_rows),
        "validation_rows": len(rows),
        "status_counts": counts,
        "severity_counts": severity_counts,
        "policy": "Strong do estimates must be contract-backed, canonical-ID-backed, estimation_enabled, and consistent with do_authority_audit.",
        "columns": VALIDATION_COLUMNS,
    }
    return rows, manifest


def write_do_authority_validation(
    out_dir: str = "out",
    contract_path: Optional[str] = None,
    estimates_path: Optional[str] = None,
    diagnostics_path: Optional[str] = None,
    authority_audit_path: Optional[str] = None,
) -> Dict[str, str]:
    rows, manifest = build_do_authority_validation(
        out_dir=out_dir,
        contract_path=contract_path,
        estimates_path=estimates_path,
        diagnostics_path=diagnostics_path,
        authority_audit_path=authority_audit_path,
    )
    out = Path(out_dir) / "scm"
    validation_path = out / "do_authority_validation.csv"
    manifest_path = out / "do_authority_validation_manifest.json"
    _write_csv(validation_path, rows, VALIDATION_COLUMNS)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump({**manifest, "do_authority_validation_csv": str(validation_path)}, f, ensure_ascii=False, indent=2)
    return {"do_authority_validation": str(validation_path), "do_authority_validation_manifest": str(manifest_path)}


__all__ = [
    "VALIDATOR_VERSION",
    "VALIDATION_COLUMNS",
    "build_do_authority_validation",
    "write_do_authority_validation",
]
