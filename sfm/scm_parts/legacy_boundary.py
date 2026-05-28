from __future__ import annotations

"""Boundary helpers for legacy SCM identification artifacts.

The active causal authority for SCM identification is the canonical ID audit
produced by :mod:`scm_parts.id_algorithm`, plus downstream gates such as
:mod:`scm_parts.do_contract`.  Legacy reporting artifacts are still useful for
compatibility, diagnostics, and historical CSV outputs, but they must never be
interpreted as canonical causal-identification authority.
"""

from typing import Dict, Iterable, Mapping, MutableMapping, Set

CANONICAL_ID_AUTHORITY_ARTIFACTS: Set[str] = {"id_algorithm_audit"}
CANONICAL_ID_AUTHORITY_SOURCES: Set[str] = {"scm_id_algorithm"}

LEGACY_IDENTIFICATION_ARTIFACTS: Set[str] = {
    "identified_effects",
    "identified_effects_csv",
    "identification/identified_effects.csv",
    "adjustment_sets",
    "adjustment_sets_csv",
    "identification/adjustment_sets.csv",
    "legacy_identifier",
    "legacy_identifier_report",
}
LEGACY_IDENTIFICATION_SOURCES: Set[str] = {
    "scm_parts.identifier",
    "scm_parts.identification_legacy",
    "legacy_identifier",
    "legacy_identifier_report",
    "legacy_identification_scoring",
}

LEGACY_IDENTIFIER_SOURCE_AUTHORITY = "legacy_identifier_report"
LEGACY_IDENTIFIER_AUDIT_SOURCE = "scm_parts.identifier"
LEGACY_IDENTIFIER_AUTHORITY_LEVEL = "legacy_reporting_only"


def _norm(value: object) -> str:
    text = "" if value is None else str(value).strip().lower()
    return "" if text in {"nan", "none", "null"} else text


def pipe_tokens(value: object) -> Set[str]:
    """Normalize pipe/comma-delimited provenance fields for authority checks."""
    return {token for token in (_norm(part) for part in str(value or "").replace(",", "|").split("|")) if token}


def _append_pipe_token(value: object, token: str) -> str:
    raw = "" if value is None else str(value).strip()
    items = [part.strip() for part in raw.replace(",", "|").split("|") if part.strip()]
    if token and token not in items:
        items.append(token)
    return "|".join(items)


def intersects_any(tokens: Iterable[str], candidates: Iterable[str]) -> bool:
    return bool(set(tokens).intersection(set(candidates)))


def is_legacy_identification_provenance(row: Mapping[str, object]) -> bool:
    """Return True when provenance points to legacy reporting/scoring output."""
    artifacts = pipe_tokens(row.get("source_artifacts"))
    sources = pipe_tokens(row.get("source_authority"))
    audit_source = pipe_tokens(row.get("audit_source"))
    return (
        intersects_any(artifacts, LEGACY_IDENTIFICATION_ARTIFACTS)
        or intersects_any(sources, LEGACY_IDENTIFICATION_SOURCES)
        or intersects_any(audit_source, LEGACY_IDENTIFICATION_SOURCES)
    )


def is_canonical_id_provenance(row: Mapping[str, object]) -> bool:
    """Return True only for canonical ID-algorithm provenance markers."""
    artifacts = pipe_tokens(row.get("source_artifacts"))
    sources = pipe_tokens(row.get("source_authority"))
    if intersects_any(artifacts, CANONICAL_ID_AUTHORITY_ARTIFACTS):
        return True
    if intersects_any(sources, CANONICAL_ID_AUTHORITY_SOURCES):
        return True
    if _norm(row.get("canonical_id_available")) in {"1", "true", "yes", "identified_estimable"}:
        return True
    if str(row.get("id_algorithm_level") or "").strip():
        return True
    return False


def legacy_boundary_reason(row: Mapping[str, object]) -> str:
    """Explain how a row should be treated at the legacy/canonical boundary."""
    legacy = is_legacy_identification_provenance(row)
    canonical = is_canonical_id_provenance(row)
    if canonical:
        return "canonical_id_authority"
    if legacy:
        return "legacy_identification_reporting_only"
    return "missing_canonical_id_authority"


def annotate_legacy_identifier_row(row: Mapping[str, object], *, artifact: str = "legacy_identifier_report") -> Dict[str, object]:
    """Return a copy of a legacy identifier row with explicit reporting-only provenance.

    This helper is intended for legacy CSV writers such as ``identifier.py``.
    It preserves existing fields, appends reporting-only provenance tokens, and
    never sets canonical ID authority fields.
    """
    out: Dict[str, object] = dict(row)
    out["source_artifacts"] = _append_pipe_token(out.get("source_artifacts"), artifact)
    out["source_authority"] = _append_pipe_token(out.get("source_authority"), LEGACY_IDENTIFIER_SOURCE_AUTHORITY)
    out["audit_source"] = _append_pipe_token(out.get("audit_source"), LEGACY_IDENTIFIER_AUDIT_SOURCE)
    out["legacy_identifier_authority"] = LEGACY_IDENTIFIER_AUTHORITY_LEVEL
    out["legacy_boundary_reason"] = legacy_boundary_reason(out)
    out.setdefault("canonical_id_available", 0)
    return out


def annotate_legacy_identifier_rows(rows: Iterable[Mapping[str, object]], *, artifact: str = "legacy_identifier_report") -> list[Dict[str, object]]:
    """Annotate a sequence of legacy identifier rows for reporting-only output."""
    return [annotate_legacy_identifier_row(row, artifact=artifact) for row in rows]


def annotate_legacy_identifier_row_inplace(row: MutableMapping[str, object], *, artifact: str = "legacy_identifier_report") -> MutableMapping[str, object]:
    """Mutate a legacy identifier row with reporting-only provenance and return it."""
    row.update(annotate_legacy_identifier_row(row, artifact=artifact))
    return row


__all__ = [
    "CANONICAL_ID_AUTHORITY_ARTIFACTS",
    "CANONICAL_ID_AUTHORITY_SOURCES",
    "LEGACY_IDENTIFICATION_ARTIFACTS",
    "LEGACY_IDENTIFICATION_SOURCES",
    "LEGACY_IDENTIFIER_SOURCE_AUTHORITY",
    "LEGACY_IDENTIFIER_AUDIT_SOURCE",
    "LEGACY_IDENTIFIER_AUTHORITY_LEVEL",
    "pipe_tokens",
    "is_legacy_identification_provenance",
    "is_canonical_id_provenance",
    "legacy_boundary_reason",
    "annotate_legacy_identifier_row",
    "annotate_legacy_identifier_rows",
    "annotate_legacy_identifier_row_inplace",
]
