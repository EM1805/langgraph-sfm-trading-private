"""Contract integrity validator for Amantia runtime handoffs.

The validator is intentionally conservative and dependency-light. It checks
that offline causal authority artifacts are present and that downstream
estimation/report/veto artifacts do not silently upgrade weak or missing
authority into causal claims.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

VALIDATOR_VERSION = 1

REQUIRED_CONTRACT_COLUMNS: Sequence[str] = (
    "insight_id",
    "source",
    "target",
    "treatment_col",
    "outcome_col",
    "authority_level",
    "identification_status",
    "identified",
    "estimation_enabled",
    "allowed_for_estimation",
    "canonical_id_authority",
)

REQUIRED_HANDOFF_COLUMNS: Sequence[str] = (
    "insight_id",
    "source",
    "target",
    "treatment_col",
    "outcome_col",
    "authority_level",
    "estimation_enabled",
    "allowed_for_estimation",
    "identification_status",
)

REQUIRED_EFFECT_COLUMNS: Sequence[str] = (
    "insight_id",
    "source",
    "target",
    "effect_claim_status",
    "effect_estimate",
    "authority_level",
    "identification_status",
    "estimation_status",
)

REQUIRED_GATE_AUDIT_COLUMNS: Sequence[str] = (
    "insight_id",
    "source",
    "target",
    "gate_decision",
    "authority_level",
    "estimation_enabled",
    "identification_status",
)

TRUTHY = {"1", "true", "yes", "y", "on", "enabled", "identified", "identified_estimable"}
FALSEY = {"", "0", "false", "no", "n", "off", "disabled", "none", "null", "nan"}
AUTHORIZED_LEVELS = {"identified_estimable"}
BLOCKED_OR_DIAGNOSTIC_LEVELS = {
    "",
    "raw_discovery_only",
    "discovery_only",
    "weak_or_unaligned",
    "blocked_id_algorithm",
    "blocked",
    "diagnostic_only",
}
CLAIM_STATUSES = {
    "effect_claim_ready",
    "effect_claim_supported",
    "claim_ready",
    "claim_supported",
    "causal_effect_claim",
}


def _s(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null", "nat"} else text


def _norm(value: object) -> str:
    return _s(value).lower()


def _truthy(value: object) -> bool:
    text = _norm(value)
    if text in TRUTHY:
        return True
    if text in FALSEY:
        return False
    try:
        return float(text) != 0.0
    except (TypeError, ValueError, OverflowError):
        return False


def _read_csv(path: Path) -> Tuple[List[Dict[str, str]], List[str], str]:
    if not path.exists():
        return [], [], "missing"
    try:
        if path.stat().st_size == 0:
            return [], [], "empty"
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            columns = list(reader.fieldnames or [])
            rows = [{k: _s(v) for k, v in row.items()} for row in reader]
            return rows, columns, "ok" if columns else "no_header"
    except (OSError, UnicodeDecodeError, csv.Error, ValueError, TypeError) as exc:
        return [], [], f"read_error:{type(exc).__name__}"


def _row_key(row: Mapping[str, str]) -> str:
    insight = _s(row.get("insight_id"))
    if insight:
        return insight
    source = _s(row.get("source") or row.get("treatment_col"))
    target = _s(row.get("target") or row.get("outcome_col"))
    lag = _s(row.get("lag")) or "0"
    if source or target:
        return f"{source}->{target}@{lag}"
    return ""


def _pair_key(row: Mapping[str, str]) -> str:
    source = _s(row.get("source") or row.get("treatment_col"))
    target = _s(row.get("target") or row.get("outcome_col"))
    return f"{source}->{target}" if source or target else ""


def _contract_indexes(rows: Iterable[Mapping[str, str]]) -> Tuple[Dict[str, Mapping[str, str]], Dict[str, List[Mapping[str, str]]]]:
    by_id: Dict[str, Mapping[str, str]] = {}
    by_pair: Dict[str, List[Mapping[str, str]]] = {}
    for row in rows:
        key = _row_key(row)
        if key and key not in by_id:
            by_id[key] = row
        pair = _pair_key(row)
        if pair:
            by_pair.setdefault(pair, []).append(row)
    return by_id, by_pair


def _find_contract_row(row: Mapping[str, str], by_id: Mapping[str, Mapping[str, str]], by_pair: Mapping[str, List[Mapping[str, str]]]) -> Mapping[str, str] | None:
    key = _row_key(row)
    if key and key in by_id:
        return by_id[key]
    pair = _pair_key(row)
    if pair and pair in by_pair and len(by_pair[pair]) == 1:
        return by_pair[pair][0]
    return None


def _artifact_report(name: str, path: Path, required_columns: Sequence[str], *, required: bool, strict: bool) -> Tuple[Dict[str, object], List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, str]], List[str]]:
    errors: List[Dict[str, object]] = []
    warnings: List[Dict[str, object]] = []
    rows, columns, status = _read_csv(path)
    artifact = {
        "name": name,
        "path": str(path),
        "status": status,
        "rows": len(rows),
        "columns": columns,
        "missing_columns": [c for c in required_columns if c not in columns],
    }
    if status != "ok":
        issue = {"code": f"{name}_{status}", "artifact": name, "path": str(path), "message": f"{name} is {status}."}
        if required and strict:
            errors.append(issue)
        else:
            warnings.append(issue)
    missing = artifact["missing_columns"]
    if missing:
        issue = {"code": f"{name}_missing_required_columns", "artifact": name, "missing_columns": missing, "message": f"{name} is missing required columns."}
        if required and strict:
            errors.append(issue)
        else:
            warnings.append(issue)
    return artifact, errors, warnings, rows, columns


def _is_contract_authorized(row: Mapping[str, str]) -> bool:
    authority = _norm(row.get("authority_level"))
    allowed = _truthy(row.get("allowed_for_estimation"))
    enabled = _truthy(row.get("estimation_enabled"))
    identified = _truthy(row.get("identified"))
    canonical = _truthy(row.get("canonical_id_authority"))
    id_status = _norm(row.get("id_status"))
    identification_status = _norm(row.get("identification_status"))
    explicit_id_ok = id_status in {"identified", "id_identified", "ok"} or identification_status in {"identified", "identified_estimable", "backdoor_identified", "frontdoor_identified"}
    return authority in AUTHORIZED_LEVELS and allowed and enabled and (identified or canonical or explicit_id_ok)


def _validate_contract_rows(rows: Sequence[Mapping[str, str]]) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    errors: List[Dict[str, object]] = []
    warnings: List[Dict[str, object]] = []
    seen: set[str] = set()
    for idx, row in enumerate(rows, start=2):
        key = _row_key(row)
        pair = _pair_key(row)
        if key:
            if key in seen:
                warnings.append({"code": "causal_contract_duplicate_key", "row": idx, "key": key, "message": "Duplicate contract key; downstream matching may be ambiguous."})
            seen.add(key)
        if not pair:
            warnings.append({"code": "causal_contract_missing_pair", "row": idx, "key": key, "message": "Contract row has no source/target or treatment/outcome pair."})
        authority = _norm(row.get("authority_level"))
        allowed = _truthy(row.get("allowed_for_estimation"))
        enabled = _truthy(row.get("estimation_enabled"))
        identified = _truthy(row.get("identified"))
        canonical = _truthy(row.get("canonical_id_authority"))
        if allowed and authority not in AUTHORIZED_LEVELS:
            errors.append({
                "code": "allowed_estimation_without_identified_authority",
                "row": idx,
                "key": key,
                "authority_level": authority,
                "message": "allowed_for_estimation is true but authority_level is not identified_estimable.",
            })
        if enabled and allowed and not (identified or canonical):
            warnings.append({
                "code": "estimation_allowed_without_explicit_identified_flag",
                "row": idx,
                "key": key,
                "message": "Estimation is enabled/allowed but identified/canonical_id_authority is not explicitly true.",
            })
        if authority in BLOCKED_OR_DIAGNOSTIC_LEVELS and (allowed or enabled):
            errors.append({
                "code": "diagnostic_or_blocked_row_enabled_for_estimation",
                "row": idx,
                "key": key,
                "authority_level": authority,
                "message": "Diagnostic/blocked authority must not enable estimation.",
            })
    return errors, warnings


def _validate_downstream_against_contract(
    artifact_name: str,
    rows: Sequence[Mapping[str, str]],
    contract_by_id: Mapping[str, Mapping[str, str]],
    contract_by_pair: Mapping[str, List[Mapping[str, str]]],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    errors: List[Dict[str, object]] = []
    warnings: List[Dict[str, object]] = []
    for idx, row in enumerate(rows, start=2):
        c_row = _find_contract_row(row, contract_by_id, contract_by_pair)
        key = _row_key(row)
        if c_row is None:
            warnings.append({
                "code": f"{artifact_name}_row_without_contract_match",
                "artifact": artifact_name,
                "row": idx,
                "key": key,
                "message": "Downstream row has no unambiguous causal_contract match.",
            })
            continue
        contract_authorized = _is_contract_authorized(c_row)
        artifact_authority = _norm(row.get("authority_level"))
        allowed = _truthy(row.get("allowed_for_estimation"))
        enabled = _truthy(row.get("estimation_enabled"))
        claim_status = _norm(row.get("effect_claim_status"))
        estimate_text = _s(row.get("effect_estimate"))
        gate_decision = _norm(row.get("gate_decision"))

        if artifact_name == "estimation_handoff" and (allowed or enabled) and not contract_authorized:
            errors.append({
                "code": "estimation_handoff_upgrades_contract_authority",
                "artifact": artifact_name,
                "row": idx,
                "key": key,
                "contract_authority_level": _s(c_row.get("authority_level")),
                "message": "estimation_handoff enables estimation for a row not authorized by causal_contract.",
            })
        if artifact_name == "effect_estimates":
            has_claim = claim_status in CLAIM_STATUSES
            has_numeric_estimate = bool(estimate_text)
            if has_claim and not contract_authorized:
                errors.append({
                    "code": "effect_claim_without_contract_authority",
                    "artifact": artifact_name,
                    "row": idx,
                    "key": key,
                    "effect_claim_status": claim_status,
                    "message": "effect_estimates contains a causal claim for a row not authorized by causal_contract.",
                })
            if has_numeric_estimate and artifact_authority in BLOCKED_OR_DIAGNOSTIC_LEVELS:
                warnings.append({
                    "code": "diagnostic_effect_estimate_only",
                    "artifact": artifact_name,
                    "row": idx,
                    "key": key,
                    "authority_level": artifact_authority,
                    "message": "Numeric estimate exists under diagnostic/blocked authority; keep it out of causal claims.",
                })
        if artifact_name == "gate_audit" and gate_decision in {"allow", "authorized", "execute"} and not contract_authorized:
            errors.append({
                "code": "gate_allows_without_contract_authority",
                "artifact": artifact_name,
                "row": idx,
                "key": key,
                "gate_decision": gate_decision,
                "message": "gate_audit allows a row that causal_contract does not authorize.",
            })
    return errors, warnings


def validate_contract_integrity(out_dir: str | Path = "out", *, strict: bool = True, write_report: bool = False, report_path: str | Path | None = None) -> Dict[str, object]:
    """Validate Amantia causal handoff integrity.

    Strict mode treats missing canonical artifacts/columns as errors. Non-strict
    mode downgrades absent canonical artifacts to warnings, which is useful
    before a pipeline has generated outputs.
    """
    out = Path(out_dir)
    artifacts: List[Dict[str, object]] = []
    errors: List[Dict[str, object]] = []
    warnings: List[Dict[str, object]] = []

    contract_art, e, w, contract_rows, _ = _artifact_report(
        "causal_contract",
        out / "causal_contract.csv",
        REQUIRED_CONTRACT_COLUMNS,
        required=True,
        strict=strict,
    )
    artifacts.append(contract_art); errors.extend(e); warnings.extend(w)

    handoff_art, e, w, handoff_rows, _ = _artifact_report(
        "estimation_handoff",
        out / "estimation" / "estimation_handoff.csv",
        REQUIRED_HANDOFF_COLUMNS,
        required=True,
        strict=strict,
    )
    artifacts.append(handoff_art); errors.extend(e); warnings.extend(w)

    effect_art, e, w, effect_rows, _ = _artifact_report(
        "effect_estimates",
        out / "estimation" / "effect_estimates.csv",
        REQUIRED_EFFECT_COLUMNS,
        required=False,
        strict=False,
    )
    artifacts.append(effect_art); errors.extend(e); warnings.extend(w)

    gate_art, e, w, gate_rows, _ = _artifact_report(
        "gate_audit",
        out / "gate_audit.csv",
        REQUIRED_GATE_AUDIT_COLUMNS,
        required=False,
        strict=False,
    )
    artifacts.append(gate_art); errors.extend(e); warnings.extend(w)

    if contract_rows:
        row_errors, row_warnings = _validate_contract_rows(contract_rows)
        errors.extend(row_errors)
        warnings.extend(row_warnings)
        by_id, by_pair = _contract_indexes(contract_rows)
        for name, rows in (
            ("estimation_handoff", handoff_rows),
            ("effect_estimates", effect_rows),
            ("gate_audit", gate_rows),
        ):
            downstream_errors, downstream_warnings = _validate_downstream_against_contract(name, rows, by_id, by_pair)
            errors.extend(downstream_errors)
            warnings.extend(downstream_warnings)

    status = "pass" if not errors else "fail_closed"
    report: Dict[str, object] = {
        "ok": not errors,
        "status": status,
        "validator_version": VALIDATOR_VERSION,
        "strict": bool(strict),
        "out_dir": str(out),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
        "artifacts": artifacts,
        "policy": {
            "causal_contract_is_authority": True,
            "raw_discovery_never_grants_estimation_authority": True,
            "effect_estimates_must_not_upgrade_contract_authority": True,
            "runtime_veto_authority_must_be_derived_from_contract": True,
        },
    }
    if write_report:
        dest = Path(report_path) if report_path else out / "contract_integrity_report.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        report["report_path"] = str(dest)
    return report


__all__ = ["validate_contract_integrity", "VALIDATOR_VERSION"]
