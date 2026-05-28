"""Cross-layer causal confidence and migration reports for Amantia.

This module deliberately separates Discovery screening scores from causal
claim authority. It reads existing artifacts and produces auditable reports;
it does not create new causal permissions and it does not override the causal
contract.

Causal confidence uses only namespaced signal/safety fields. Legacy Discovery
scores such as selection_score or discovery_evidence_score are reported as
ignored migration signals, not used as confidence inputs.
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

CONFIDENCE_VERSION = 4
MIGRATION_REPORT_VERSION = 1
CONFIDENCE_INPUT_POLICY = "strict_namespaced_inputs_no_legacy_score_fallback_prefer_signal_safety_rows"

CAUSAL_CONFIDENCE_COLUMNS: List[str] = [
    "insight_id", "source", "target", "treatment_col", "outcome_col", "lag",
    "hypothesis_signal_score", "hypothesis_signal_grade", "safety_risk_score",
    "safety_risk_grade", "safety_cleanliness_score", "discovery_signal_confidence",
    "discovery_confidence_tier", "signal_safety_cell", "signal_safety_policy",
    "signal_safety_matrix_track", "signal_safety_blocking", "authority_level",
    "identification_status", "identified", "identification_strategy", "estimation_enabled",
    "scm_identification_score", "estimation_validation_score", "do_authorized", "do_mode",
    "do_effect_estimate", "pearl_claim_status", "causal_confidence_score",
    "causal_confidence_grade", "confidence_cap", "confidence_cap_reason",
    "confidence_formula_version", "causal_confidence_semantics", "confidence_input_policy",
    "legacy_score_ignored", "legacy_score_ignored_fields", "score_namespace_version",
    "reason_codes",
]

CAUSAL_CONFIDENCE_MIGRATION_COLUMNS: List[str] = [
    "insight_id", "source", "target", "lag",
    "has_hypothesis_signal_score", "has_hypothesis_signal_grade",
    "has_safety_risk_score", "has_safety_risk_grade", "has_signal_safety_matrix",
    "has_causal_contract_row", "has_identified_estimable_contract", "has_do_estimate",
    "has_pearl_effect", "legacy_score_present", "legacy_score_ignored",
    "legacy_score_ignored_fields", "missing_required_inputs", "migration_action",
    "migration_severity", "migration_reason_codes", "migration_report_version",
]

LEGACY_SCORE_KEYS = [
    "discovery_score", "selection_score", "discovery_evidence_score", "pcmci_score",
    "score", "discovery_confidence_tier", "confidence_tier", "confidence",
]

DISCOVERY_NAMESPACED_KEYS = [
    "hypothesis_signal_score", "hypothesis_signal_grade", "hypothesis_signal_reason_codes",
    "safety_risk_score", "safety_risk_grade", "safety_risk_reason_codes", "safety_blocking",
    "signal_safety_cell", "signal_safety_policy", "signal_safety_matrix_track",
    "signal_safety_blocking", "signal_safety_reason_code", "signal_safety_matrix_version",
]


def _norm(value: object) -> str:
    text = "" if value is None else str(value).strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return float(out)


def _clip01(value: object) -> float:
    return max(0.0, min(1.0, _safe_float(value, 0.0)))


def _bool01(value: object) -> str:
    return "1" if bool(value) else "0"


def _read_csv(path: str | os.PathLike) -> List[Dict[str, str]]:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return []
    try:
        with p.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return []
            return [{k: _norm(v) for k, v in row.items()} for row in reader]
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
        value = _norm(row.get(key, ""))
        if value:
            return value
    return default


def _key(row: Dict[str, str]) -> str:
    iid = _first(row, ["insight_id", "candidate_id", "edge_id", "effect_id"])
    if iid:
        return iid
    source = _first(row, ["source", "treatment_col", "treatment", "from", "cause"])
    target = _first(row, ["target", "outcome_col", "outcome", "target_col", "to", "effect"])
    lag = _first(row, ["lag", "tau", "time_lag"], "")
    if source or target:
        return f"{source}->{target}@{lag}"
    return ""


def _pair_key(row: Dict[str, str]) -> str:
    source = _first(row, ["source", "treatment_col", "treatment", "from", "cause"])
    target = _first(row, ["target", "outcome_col", "outcome", "target_col", "to", "effect"])
    return f"{source}->{target}" if (source or target) else ""


def _namespaced_signal_safety_count(row: Dict[str, str]) -> int:
    return sum(1 for key in DISCOVERY_NAMESPACED_KEYS if _norm(row.get(key)))


def _discovery_row_quality(row: Dict[str, str]) -> Tuple[int, int, int]:
    """Prefer rows carrying explicit signal/safety fields over bridge/legacy-only rows."""
    namespaced = _namespaced_signal_safety_count(row)
    has_matrix = int(_row_has(row, ["signal_safety_cell", "signal_safety_matrix_track", "signal_safety_blocking"]))
    has_signal = int(_row_has(row, ["hypothesis_signal_score", "hypothesis_signal_grade", "discovery_signal_confidence"]))
    return namespaced, has_matrix, has_signal


def _prefer_row(current: Optional[Dict[str, str]], candidate: Dict[str, str]) -> Dict[str, str]:
    if current is None:
        return candidate
    if _discovery_row_quality(candidate) > _discovery_row_quality(current):
        return candidate
    return current


def _index_rows(rows: List[Dict[str, str]]) -> Tuple[Dict[str, Dict[str, str]], Dict[str, Dict[str, str]]]:
    by_key: Dict[str, Dict[str, str]] = {}
    by_pair: Dict[str, Dict[str, str]] = {}
    for row in rows:
        k = _key(row)
        p = _pair_key(row)
        if k:
            by_key[k] = _prefer_row(by_key.get(k), row)
        if p:
            by_pair[p] = _prefer_row(by_pair.get(p), row)
    return by_key, by_pair


def _dedupe_preferred_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    by_key, by_pair = _index_rows(rows)
    chosen: List[Dict[str, str]] = []
    seen_ids = set()
    for row in list(by_key.values()) + list(by_pair.values()):
        marker = id(row)
        if marker not in seen_ids:
            chosen.append(row)
            seen_ids.add(marker)
    return chosen


def _grade_from_score(score: float) -> str:
    if score >= 0.85:
        return "very_high"
    if score >= 0.70:
        return "high"
    if score >= 0.45:
        return "medium"
    if score > 0.0:
        return "low"
    return "blocked"


def _legacy_score_fields_present(row: Dict[str, str]) -> List[str]:
    return [key for key in LEGACY_SCORE_KEYS if _norm(row.get(key))]


def _row_has(row: Dict[str, str], keys: Iterable[str]) -> bool:
    return any(_norm(row.get(k)) for k in keys)


def _signal_confidence(row: Dict[str, str]) -> Tuple[float, str, List[str]]:
    """Return namespaced Discovery signal confidence without legacy fallback."""
    ignored = _legacy_score_fields_present(row)
    if _norm(row.get("hypothesis_signal_score")):
        return _clip01(row.get("hypothesis_signal_score")), "hypothesis_signal_score", ignored
    if _norm(row.get("discovery_signal_confidence")):
        return _clip01(row.get("discovery_signal_confidence")), "discovery_signal_confidence", ignored
    grade = _first(row, ["hypothesis_signal_grade"]).lower()
    if grade in {"strong", "middle", "weak", "block"}:
        return {"strong": 0.82, "middle": 0.58, "weak": 0.30, "block": 0.0}[grade], "hypothesis_signal_grade", ignored
    return 0.0, "missing_namespaced_signal_no_legacy_fallback", ignored


def _safety_risk(row: Dict[str, str]) -> Tuple[float, str, int]:
    grade = _first(row, ["safety_risk_grade"], "good").lower()
    if grade not in {"good", "mediocre", "dangerous", "critical"}:
        grade = "good"
    if _norm(row.get("safety_risk_score")):
        risk = _clip01(row.get("safety_risk_score"))
    else:
        risk = {"good": 0.05, "mediocre": 0.35, "dangerous": 0.65, "critical": 1.0}[grade]
    blocking = int(_safe_float(row.get("safety_blocking", row.get("signal_safety_blocking", 0)), 0))
    return risk, grade, blocking


def _scm_identification_score(row: Dict[str, str]) -> float:
    authority = _norm(row.get("authority_level")).lower()
    identified = _norm(row.get("identified")).lower() in {"1", "true", "yes"}
    enabled = _norm(row.get("estimation_enabled")).lower() in {"1", "true", "yes"}
    strategy = _norm(row.get("identification_strategy")).lower()
    if authority == "identified_estimable" and identified and enabled:
        return 0.88 if "frontdoor" in strategy else 0.95
    if authority == "identified_estimable" or (identified and enabled):
        return 0.85
    if authority == "identified_needs_estimation" or identified:
        return 0.62
    if authority == "graph_review":
        return 0.42
    if authority == "discovery_only":
        return 0.25
    return 0.0


def _estimation_validation_score(row: Dict[str, str], do_row: Optional[Dict[str, str]], pearl_row: Optional[Dict[str, str]]) -> Tuple[float, str, str, str]:
    if do_row:
        do_auth = _norm(do_row.get("do_authorized")).lower() in {"1", "true", "yes"}
        diagnostic_allowed = _norm(do_row.get("diagnostic_estimation_allowed")).lower() in {"1", "true", "yes"}
        if do_auth:
            return 0.92, _norm(do_row.get("do_authorized", "1")), _norm(do_row.get("do_mode")), _norm(do_row.get("effect_estimate"))
        if diagnostic_allowed:
            return 0.35, _norm(do_row.get("do_authorized", "0")), _norm(do_row.get("do_mode")), _norm(do_row.get("effect_estimate"))
    if pearl_row:
        status = _norm(pearl_row.get("causal_claim_status") or pearl_row.get("claim_status") or pearl_row.get("status"))
        if status and "authorized" in status.lower():
            return 0.70, "", "", ""
        if status:
            return 0.35, "", "", ""
    return 0.0, "", "", ""


def _cap_score(score: float, row: Dict[str, str], safety_grade: str, safety_blocking: int, scm_score: float, estimation_score: float) -> Tuple[float, str, str]:
    cap = 1.0
    reasons: List[str] = []
    matrix_block = _norm(row.get("signal_safety_blocking")).lower() in {"1", "true", "yes"}
    if safety_grade == "critical" or safety_blocking or matrix_block:
        return 0.0, "0.00", "SAFETY_CRITICAL_OR_MATRIX_BLOCKING"
    if safety_grade == "dangerous":
        cap = min(cap, 0.35)
        reasons.append("SAFETY_DANGEROUS_CAP_0_35")
    authority = _norm(row.get("authority_level")).lower()
    if authority != "identified_estimable" or scm_score < 0.80:
        cap = min(cap, 0.70)
        reasons.append("NO_IDENTIFIED_ESTIMABLE_CONTRACT_CAP_0_70")
    if estimation_score <= 0.0:
        cap = min(cap, 0.85)
        reasons.append("NO_DO_OR_ESTIMATION_VALIDATION_CAP_0_85")
    return min(score, cap), f"{cap:.2f}", "|".join(reasons) if reasons else "NONE"


def _merge_discovery_metadata(contract_rows: List[Dict[str, str]], discovery_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    by_key, by_pair = _index_rows(discovery_rows)
    out: List[Dict[str, str]] = []
    for row in contract_rows:
        merged = dict(row)
        extra = by_key.get(_key(row)) or by_pair.get(_pair_key(row)) or {}
        for key in [
            "hypothesis_signal_score", "hypothesis_signal_grade", "hypothesis_signal_reason_codes",
            "safety_risk_score", "safety_risk_grade", "safety_risk_reason_codes", "safety_blocking",
            "signal_safety_cell", "signal_safety_policy", "signal_safety_matrix_track", "signal_safety_blocking",
            "signal_safety_reason_code", "signal_safety_matrix_version",
            "discovery_evidence_score", "selection_score", "confidence_tier", "discovery_confidence_tier",
        ]:
            if not _norm(merged.get(key)) and _norm(extra.get(key)):
                merged[key] = extra.get(key, "")
        out.append(merged)
    return out


def _load_artifact_rows(out: Path) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    contract_rows = _read_csv(out / "causal_contract.csv")
    discovery_rows: List[Dict[str, str]] = []
    for rel in ["insights_level2.csv", os.path.join("ranking", "insights_level2.csv"), "discovery_estimation_bridge.csv"]:
        discovery_rows.extend(_read_csv(out / rel))
    do_rows = _read_csv(out / "scm" / "do_estimates.csv")
    effect_rows = _read_csv(out / "estimation" / "effect_estimates.csv")
    return contract_rows, discovery_rows, do_rows, effect_rows


def _rows_for_reports(out: Path) -> List[Dict[str, str]]:
    contract_rows, discovery_rows, _do_rows, _effect_rows = _load_artifact_rows(out)
    if contract_rows:
        return _merge_discovery_metadata(contract_rows, discovery_rows)
    return _dedupe_preferred_rows(discovery_rows)


def _migration_action(row: Dict[str, str], do_row: Optional[Dict[str, str]], pearl_row: Optional[Dict[str, str]]) -> Tuple[str, str, List[str], List[str]]:
    missing: List[str] = []
    reasons: List[str] = []
    has_signal_score = _row_has(row, ["hypothesis_signal_score", "discovery_signal_confidence"])
    has_signal_grade = _row_has(row, ["hypothesis_signal_grade"])
    has_safety_score = _row_has(row, ["safety_risk_score"])
    has_safety_grade = _row_has(row, ["safety_risk_grade"])
    has_matrix = _row_has(row, ["signal_safety_cell", "signal_safety_matrix_track", "signal_safety_blocking"])
    authority = _norm(row.get("authority_level")).lower()
    identified = _norm(row.get("identified")).lower() in {"1", "true", "yes"}
    estimation_enabled = _norm(row.get("estimation_enabled")).lower() in {"1", "true", "yes"}
    identified_estimable = authority == "identified_estimable" and identified and estimation_enabled

    if not (has_signal_score or has_signal_grade):
        missing.append("hypothesis_signal_score_or_grade")
        reasons.append("MISSING_NAMESPACED_SIGNAL")
    if not (has_safety_score or has_safety_grade):
        missing.append("safety_risk_score_or_grade")
        reasons.append("MISSING_SAFETY_RISK")
    if not has_matrix:
        missing.append("signal_safety_matrix")
        reasons.append("MISSING_SIGNAL_SAFETY_MATRIX")
    if not authority:
        missing.append("causal_contract_row")
        reasons.append("MISSING_CAUSAL_CONTRACT_ROW")
    elif not identified_estimable:
        missing.append("identified_estimable_contract")
        reasons.append("CONTRACT_NOT_IDENTIFIED_ESTIMABLE")
    if identified_estimable and not (do_row or pearl_row):
        missing.append("do_or_estimation_validation")
        reasons.append("MISSING_DO_OR_ESTIMATION_VALIDATION")
    if _legacy_score_fields_present(row):
        reasons.append("LEGACY_DISCOVERY_SCORE_PRESENT_BUT_IGNORED")

    if not missing:
        return "OK_READY_FOR_CAUSAL_CONFIDENCE", "ok", missing, reasons or ["READY"]
    if "hypothesis_signal_score_or_grade" in missing or "safety_risk_score_or_grade" in missing or "signal_safety_matrix" in missing:
        return "REGENERATE_DISCOVERY_STEP164_PLUS", "high", missing, reasons
    if "causal_contract_row" in missing or "identified_estimable_contract" in missing:
        return "RUN_SCM_CONTRACT", "medium", missing, reasons
    if "do_or_estimation_validation" in missing:
        return "RUN_DO_ENGINE_OR_ESTIMATION", "low", missing, reasons
    return "REVIEW_CONFIDENCE_INPUTS", "medium", missing, reasons


def build_causal_confidence_migration_report(out_dir: str = "out") -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    out = Path(out_dir)
    rows = _rows_for_reports(out)
    _contract_rows, _discovery_rows, do_rows, effect_rows = _load_artifact_rows(out)
    do_by_key, do_by_pair = _index_rows(do_rows)
    pearl_by_key, pearl_by_pair = _index_rows(effect_rows)
    report: List[Dict[str, object]] = []
    action_counts: Dict[str, int] = {}
    severity_counts: Dict[str, int] = {}

    for row in rows:
        do_row = do_by_key.get(_key(row)) or do_by_pair.get(_pair_key(row))
        pearl_row = pearl_by_key.get(_key(row)) or pearl_by_pair.get(_pair_key(row))
        action, severity, missing, reasons = _migration_action(row, do_row, pearl_row)
        action_counts[action] = action_counts.get(action, 0) + 1
        severity_counts[severity] = severity_counts.get(severity, 0) + 1
        legacy = _legacy_score_fields_present(row)
        authority = _norm(row.get("authority_level")).lower()
        identified = _norm(row.get("identified")).lower() in {"1", "true", "yes"}
        estimation_enabled = _norm(row.get("estimation_enabled")).lower() in {"1", "true", "yes"}
        report.append({
            "insight_id": _first(row, ["insight_id", "candidate_id", "edge_id"], _key(row)),
            "source": _first(row, ["source", "treatment_col", "treatment"]),
            "target": _first(row, ["target", "outcome_col", "outcome", "target_col"]),
            "lag": _first(row, ["lag", "tau", "time_lag"]),
            "has_hypothesis_signal_score": _bool01(_row_has(row, ["hypothesis_signal_score", "discovery_signal_confidence"])),
            "has_hypothesis_signal_grade": _bool01(_row_has(row, ["hypothesis_signal_grade"])),
            "has_safety_risk_score": _bool01(_row_has(row, ["safety_risk_score"])),
            "has_safety_risk_grade": _bool01(_row_has(row, ["safety_risk_grade"])),
            "has_signal_safety_matrix": _bool01(_row_has(row, ["signal_safety_cell", "signal_safety_matrix_track", "signal_safety_blocking"])),
            "has_causal_contract_row": _bool01(bool(authority)),
            "has_identified_estimable_contract": _bool01(authority == "identified_estimable" and identified and estimation_enabled),
            "has_do_estimate": _bool01(do_row is not None),
            "has_pearl_effect": _bool01(pearl_row is not None),
            "legacy_score_present": _bool01(bool(legacy)),
            "legacy_score_ignored": _bool01(bool(legacy)),
            "legacy_score_ignored_fields": "|".join(legacy),
            "missing_required_inputs": "|".join(missing),
            "migration_action": action,
            "migration_severity": severity,
            "migration_reason_codes": "|".join(reasons),
            "migration_report_version": MIGRATION_REPORT_VERSION,
        })
    manifest = {
        "migration_report_version": MIGRATION_REPORT_VERSION,
        "n_rows": len(report),
        "action_counts": action_counts,
        "severity_counts": severity_counts,
        "purpose": "Explain low/zero causal confidence caused by missing namespaced inputs versus genuine causal weakness.",
        "recommended_order": [
            "REGENERATE_DISCOVERY_STEP164_PLUS",
            "RUN_SCM_CONTRACT",
            "RUN_DO_ENGINE_OR_ESTIMATION",
            "OK_READY_FOR_CAUSAL_CONFIDENCE",
        ],
        "columns": CAUSAL_CONFIDENCE_MIGRATION_COLUMNS,
    }
    return report, manifest


def build_causal_confidence_report(out_dir: str = "out") -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    out = Path(out_dir)
    rows = _rows_for_reports(out)
    _contract_rows, _discovery_rows, do_rows, effect_rows = _load_artifact_rows(out)
    do_by_key, do_by_pair = _index_rows(do_rows)
    pearl_by_key, pearl_by_pair = _index_rows(effect_rows)

    report: List[Dict[str, object]] = []
    cap_counts: Dict[str, int] = {}
    grade_counts: Dict[str, int] = {}
    for row in rows:
        sig, signal_source, legacy_ignored = _signal_confidence(row)
        risk, risk_grade, safety_blocking = _safety_risk(row)
        cleanliness = 1.0 - risk
        scm_score = _scm_identification_score(row)
        do_row = do_by_key.get(_key(row)) or do_by_pair.get(_pair_key(row))
        pearl_row = pearl_by_key.get(_key(row)) or pearl_by_pair.get(_pair_key(row))
        est_score, do_auth, do_mode, do_effect = _estimation_validation_score(row, do_row, pearl_row)
        raw_score = 0.40 * sig + 0.25 * cleanliness + 0.20 * scm_score + 0.15 * est_score
        final_score, cap, cap_reason = _cap_score(raw_score, row, risk_grade, safety_blocking, scm_score, est_score)
        grade = _grade_from_score(final_score)
        cap_counts[cap_reason] = cap_counts.get(cap_reason, 0) + 1
        grade_counts[grade] = grade_counts.get(grade, 0) + 1
        reason_codes = [
            f"DISCOVERY_SIGNAL={sig:.2f}", f"SIGNAL_SOURCE={signal_source}",
            f"SAFETY_CLEAN={cleanliness:.2f}", f"SCM={scm_score:.2f}", f"ESTIMATION={est_score:.2f}",
        ]
        if legacy_ignored:
            reason_codes.append("LEGACY_DISCOVERY_SCORE_FIELDS_IGNORED")
        if signal_source == "missing_namespaced_signal_no_legacy_fallback":
            reason_codes.append("SIGNAL_SCORE_MISSING_NO_LEGACY_FALLBACK")
        if cap_reason != "NONE":
            reason_codes.append(cap_reason)
        report.append({
            "insight_id": _first(row, ["insight_id", "candidate_id", "edge_id"], _key(row)),
            "source": _first(row, ["source", "treatment_col", "treatment"]),
            "target": _first(row, ["target", "outcome_col", "outcome", "target_col"]),
            "treatment_col": _first(row, ["treatment_col", "source", "treatment"]),
            "outcome_col": _first(row, ["outcome_col", "target", "target_col", "outcome"]),
            "lag": _first(row, ["lag", "tau", "time_lag"]),
            "hypothesis_signal_score": f"{sig:.6f}",
            "hypothesis_signal_grade": _first(row, ["hypothesis_signal_grade"]),
            "safety_risk_score": f"{risk:.6f}",
            "safety_risk_grade": risk_grade,
            "safety_cleanliness_score": f"{cleanliness:.6f}",
            "discovery_signal_confidence": f"{sig:.6f}",
            "discovery_confidence_tier": _first(row, ["discovery_confidence_tier", "confidence_tier", "confidence"]),
            "signal_safety_cell": _first(row, ["signal_safety_cell"]),
            "signal_safety_policy": _first(row, ["signal_safety_policy"]),
            "signal_safety_matrix_track": _first(row, ["signal_safety_matrix_track"]),
            "signal_safety_blocking": _first(row, ["signal_safety_blocking"], "0"),
            "authority_level": _first(row, ["authority_level"]),
            "identification_status": _first(row, ["identification_status"]),
            "identified": _first(row, ["identified"]),
            "identification_strategy": _first(row, ["identification_strategy"]),
            "estimation_enabled": _first(row, ["estimation_enabled"]),
            "scm_identification_score": f"{scm_score:.6f}",
            "estimation_validation_score": f"{est_score:.6f}",
            "do_authorized": do_auth,
            "do_mode": do_mode,
            "do_effect_estimate": do_effect,
            "pearl_claim_status": _first(pearl_row or {}, ["causal_claim_status", "claim_status", "status"]),
            "causal_confidence_score": f"{final_score:.6f}",
            "causal_confidence_grade": grade,
            "confidence_cap": cap,
            "confidence_cap_reason": cap_reason,
            "confidence_formula_version": CONFIDENCE_VERSION,
            "causal_confidence_semantics": "cross_layer_confidence_not_causal_authority;causal_contract_remains_authority_gate",
            "confidence_input_policy": CONFIDENCE_INPUT_POLICY,
            "legacy_score_ignored": "1" if legacy_ignored else "0",
            "legacy_score_ignored_fields": "|".join(legacy_ignored),
            "score_namespace_version": "3",
            "reason_codes": "|".join(reason_codes),
        })
    manifest = {
        "causal_confidence_version": CONFIDENCE_VERSION,
        "discovery_row_preference": "prefer rows with hypothesis_signal_*, safety_risk_*, signal_safety_* fields",
        "confidence_input_policy": CONFIDENCE_INPUT_POLICY,
        "legacy_score_policy": "legacy Discovery scores are ignored by causal_confidence; rows with namespaced signal/safety fields are preferred over bridge or legacy-only rows",
        "n_rows": len(report),
        "grade_counts": grade_counts,
        "cap_reason_counts": cap_counts,
        "formula": "0.40*namespaced_hypothesis_signal + 0.25*(1-safety_risk) + 0.20*scm_identification + 0.15*estimation_validation",
        "hard_caps": {
            "critical_or_blocking_safety": 0.0,
            "dangerous_safety": 0.35,
            "no_identified_estimable_contract": 0.70,
            "no_do_or_estimation_validation": 0.85,
        },
        "semantics": "Confidence report only; does not grant causal authority or bypass causal_contract/do_contract.",
        "inputs": {
            "causal_contract": str(out / "causal_contract.csv"),
            "discovery_bridge": str(out / "discovery_estimation_bridge.csv"),
            "insights_level2": str(out / "insights_level2.csv"),
            "do_estimates": str(out / "scm" / "do_estimates.csv"),
            "effect_estimates": str(out / "estimation" / "effect_estimates.csv"),
        },
        "columns": CAUSAL_CONFIDENCE_COLUMNS,
    }
    return report, manifest


def write_causal_confidence_report(out_dir: str = "out") -> Dict[str, str]:
    rows, manifest = build_causal_confidence_report(out_dir=out_dir)
    migration_rows, migration_manifest = build_causal_confidence_migration_report(out_dir=out_dir)
    out = Path(out_dir)
    report_path = out / "causal_confidence_report.csv"
    migration_path = out / "causal_confidence_migration_report.csv"
    manifest_path = out / "causal_confidence_manifest.json"
    migration_manifest_path = out / "causal_confidence_migration_manifest.json"
    _write_csv(report_path, rows, CAUSAL_CONFIDENCE_COLUMNS)
    _write_csv(migration_path, migration_rows, CAUSAL_CONFIDENCE_MIGRATION_COLUMNS)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump({
            **manifest,
            "causal_confidence_report_csv": str(report_path),
            "causal_confidence_migration_report_csv": str(migration_path),
            "causal_confidence_migration_manifest": str(migration_manifest_path),
        }, f, ensure_ascii=False, indent=2)
    with migration_manifest_path.open("w", encoding="utf-8") as f:
        json.dump({
            **migration_manifest,
            "causal_confidence_migration_report_csv": str(migration_path),
        }, f, ensure_ascii=False, indent=2)
    return {
        "causal_confidence_report_csv": str(report_path),
        "causal_confidence_manifest": str(manifest_path),
        "causal_confidence_migration_report_csv": str(migration_path),
        "causal_confidence_migration_manifest": str(migration_manifest_path),
    }


def cli(argv: Optional[List[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Write cross-layer causal confidence and migration reports")
    ap.add_argument("--out-dir", default="out")
    args = ap.parse_args(argv)
    paths = write_causal_confidence_report(out_dir=args.out_dir)
    print(json.dumps({"status": "ok", **paths}, indent=2))
    return 0


__all__ = [
    "CAUSAL_CONFIDENCE_COLUMNS",
    "CAUSAL_CONFIDENCE_MIGRATION_COLUMNS",
    "CONFIDENCE_VERSION",
    "MIGRATION_REPORT_VERSION",
    "CONFIDENCE_INPUT_POLICY",
    "build_causal_confidence_report",
    "build_causal_confidence_migration_report",
    "write_causal_confidence_report",
]


if __name__ == "__main__":
    raise SystemExit(cli())
