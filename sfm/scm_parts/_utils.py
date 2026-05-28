from __future__ import annotations

"""Small shared helpers for SCM modules.

This module intentionally contains only dependency-light constants and helpers
so graph/identification code can share schema decisions without importing the
heavier builder/runtime stack.
"""

from typing import Any, Iterable, List, Sequence

CONFIDENCE_RANK = {"unknown": 0, "low": 1, "medium": 2, "high": 3}
STRENGTH_RANK = {"none": 0, "weak": 1, "moderate": 2, "strong": 3}

SOURCE_FIELD_CANDIDATES: Sequence[str] = (
    "source",
    "treatment_col",
    "treatment",
    "cause",
    "action_col",
    "action",
    "from",
)

TARGET_FIELD_CANDIDATES: Sequence[str] = (
    "target",
    "outcome_col",
    "outcome",
    "harm_event",
    "target_col",
    "effect",
    "to",
)


def as_str(value: Any) -> str:
    """Return a clean string for graph IDs; never leak pandas/CSV null tokens."""
    if value is None:
        return ""
    try:
        if value != value:  # NaN without importing math/pandas
            return ""
    except Exception:
        pass
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null", "nat"} else text


def first_present(row: Any, names: Iterable[str], default: Any = "") -> Any:
    """Return the first non-empty value from a dict/Series-like row."""
    for name in names:
        try:
            value = row.get(name, "")
        except (TypeError, ValueError, AttributeError):
            value = ""
        text = as_str(value)
        if text:
            return text
    return default


def split_field_list(value: Any) -> List[str]:
    """Split pipe/comma encoded metadata fields into clean unique strings."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        raw_parts = list(value)
    else:
        raw = str(value).strip()
        if not raw:
            return []
        sep = "|" if "|" in raw else ","
        raw_parts = raw.split(sep)
    out: List[str] = []
    seen = set()
    for part in raw_parts:
        item = as_str(part)
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def normalize_confidence(value: Any) -> str:
    raw = str(value or "").strip().lower()
    return raw if raw in CONFIDENCE_RANK else "unknown"


def edge_endpoints_from_row(row: Any) -> tuple[str, str]:
    """Resolve source/target using the shared SCM handoff schema."""
    return (
        as_str(first_present(row, SOURCE_FIELD_CANDIDATES, "")).strip(),
        as_str(first_present(row, TARGET_FIELD_CANDIDATES, "")).strip(),
    )
