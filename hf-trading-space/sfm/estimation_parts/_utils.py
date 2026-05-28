"""Shared lightweight utilities for the Estimation layer.

Keep this module free from project config loading and heavy runtime side effects.
It is safe to import from common.py, contract_gate.py, pearl_backdoor.py,
and small helper modules.
"""
from __future__ import annotations

from runtime_env import configure_scientific_runtime
configure_scientific_runtime()


import json
import math
import os
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd


def as_str(value) -> str:
    """Return a normalized string; NaN/None become empty string."""
    if value is None:
        return ""
    try:
        if isinstance(value, float) and np.isnan(value):
            return ""
    except (TypeError, ValueError, FloatingPointError):
        pass
    try:
        return str(value).strip()
    except (TypeError, ValueError):
        return ""


def safe_float(value, default=np.nan) -> float:
    """Convert to finite float, otherwise return default."""
    try:
        v = float(value)
        return v if np.isfinite(v) else float(default)
    except (TypeError, ValueError, OverflowError):
        return float(default)


def parse_list(value) -> List[str]:
    """Parse JSON-list, pipe, semicolon or comma separated fields with dedupe."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        raw = list(value)
    else:
        s = as_str(value)
        if not s:
            return []
        raw = None
        if s.startswith("[") and s.endswith("]"):
            try:
                decoded = json.loads(s)
                if isinstance(decoded, list):
                    raw = decoded
            except (json.JSONDecodeError, TypeError, ValueError):
                raw = None
        if raw is None:
            sep = "|" if "|" in s else (";" if ";" in s else ",")
            raw = s.split(sep)
    out: List[str] = []
    for item in raw:
        t = as_str(item).strip().strip("'\"")
        if t and t.lower() not in {"nan", "none", "null"} and t not in out:
            out.append(t)
    return out


def split_pipe(value) -> List[str]:
    return parse_list(as_str(value).replace(",", "|") if value is not None else value)


def mode_or_empty(series) -> str:
    try:
        s = pd.Series(series).astype(str)
        s = s[(s != "") & (s != "nan") & (s != "None")]
        if len(s) == 0:
            return ""
        vc = s.value_counts(dropna=True)
        return str(vc.index[0]) if len(vc) else ""
    except (TypeError, ValueError):
        return ""


FEEDBACK_WEIGHT_DEFAULTS = {
    "confidence": 0.55,
    "success_lb": 0.20,
    "identifiable_rate": 0.10,
    "direction_match_rate": 0.10,
    "balance_pass_rate": 0.05,
}


def bounded_weighted_score(*, confidence: float, success_lb: float, identifiable_rate: float,
                           direction_match_rate: float, balance_pass_rate: float,
                           weights: dict | None = None) -> float:
    """Canonical feedback-weight formula, replacing the inline magic-number expression."""
    w = dict(FEEDBACK_WEIGHT_DEFAULTS)
    if weights:
        w.update(weights)

    def pos(x, default=0.0):
        try:
            v = float(x)
            return max(0.0, v) if math.isfinite(v) else default
        except (TypeError, ValueError, OverflowError):
            return default

    score = (
        w["confidence"] * pos(confidence)
        + w["success_lb"] * pos(success_lb)
        + w["identifiable_rate"] * pos(identifiable_rate)
        + w["direction_match_rate"] * pos(direction_match_rate)
        + w["balance_pass_rate"] * pos(balance_pass_rate)
    )
    return float(min(1.0, max(0.0, score)))


def count_csv_rows_fast(path: str) -> int:
    """Count data rows without loading the CSV into a DataFrame."""
    try:
        with open(path, "rb") as fh:
            n_lines = sum(1 for _ in fh)
        return max(0, n_lines - 1)
    except OSError:
        return 0


def read_csv_header(path: str) -> Sequence[str]:
    try:
        return list(pd.read_csv(path, nrows=0).columns)
    except (OSError, ValueError, TypeError, pd.errors.ParserError):
        return []
