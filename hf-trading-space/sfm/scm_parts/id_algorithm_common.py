from __future__ import annotations

"""Shared helpers for Amantia SCM ID modules.

Kept deliberately small so the recursive ID engine, diagnostics, and legacy
public API can share formatting/parsing behavior without circular imports.
"""

import json
import re
from typing import Iterable, List, Mapping, Sequence, Set

def _s(value: object) -> str:
    raw = "" if value is None else str(value).strip()
    return "" if raw.lower() in {"nan", "none", "null"} else raw


def _dedupe(values: Iterable[object]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for value in values or []:
        item = _s(value)
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _format_component(nodes: Iterable[str]) -> str:
    return "{" + ",".join(sorted(_dedupe(nodes))) + "}"


def _format_components(components: Iterable[Iterable[str]]) -> str:
    return "|".join(_format_component(c) for c in components)


def _format_path(path: Sequence[str]) -> str:
    return "->".join(path)


def _format_paths(paths: Sequence[Sequence[str]], *, limit: int = 8) -> str:
    shown = [_format_path(p) for p in list(paths)[:limit]]
    if len(paths) > limit:
        shown.append(f"...(+{len(paths) - limit})")
    return "|".join(shown)


def parse_field_list(value: object) -> List[str]:
    """Parse pipe/comma/JSON-ish list fields used in contracts/bridge rows."""
    text = _s(value)
    if not text or text in {"[]", "{}"}:
        return []
    for ch in "[]{}'\"":
        text = text.replace(ch, "")
    return _dedupe(part.strip() for part in text.replace(",", "|").split("|"))


def parse_formula_term_list(value: object) -> List[str]:
    """Parse formula term fields without splitting conditional bars inside P(.).

    Existing audit columns use ``|`` as a separator, while probability factors
    also contain a conditional bar, e.g. ``P(Y | Z)``.  This helper splits only
    at bars that introduce another probability/Q term.
    """
    text = _s(value)
    if not text:
        return []
    return _dedupe(part.strip() for part in re.split(r"\|(?=(?:P\(|Q\[))", text) if part.strip())


def _conditional_factor(node: str, parents: Sequence[str]) -> str:
    pa = ",".join(_dedupe(parents))
    return f"P({node} | {pa})" if pa else f"P({node})"


def _joint_symbol(nodes: Sequence[str]) -> str:
    items = _dedupe(nodes)
    return ",".join(items) if items else ""


def _json_formula(payload: Mapping[str, object]) -> str:
    """Stable compact JSON for downstream audit/report consumers."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))
