"""Explicit signal × safety decision matrix for Discovery tracks.

The matrix keeps the positive hypothesis signal axis separate from the
negative safety-risk axis.  It is intentionally policy based: a strong signal
cannot compensate for a critical safety risk.

This module does not identify or authorize causal effects.  It only decides
how a discovery candidate should be routed before SCM/contract estimation.
"""

from __future__ import annotations

from typing import Dict, Tuple


SIGNAL_GRADES = ("block", "weak", "middle", "strong")
SAFETY_RISK_GRADES = ("good", "mediocre", "dangerous", "critical")

# Matrix cells are policy decisions, not numeric score arithmetic.
# The track is the maximum route the candidate may take before the usual
# numeric gates in hypotheses.py are applied.
_BASE_MATRIX: Dict[Tuple[str, str], Dict[str, object]] = {
    ("strong", "good"): {
        "policy": "candidate_for_scm",
        "track": "high_confidence",
        "blocking": 0,
        "reason": "MATRIX_STRONG_SIGNAL_GOOD_SAFETY",
    },
    ("strong", "mediocre"): {
        "policy": "candidate_low_confidence",
        "track": "exploratory",
        "blocking": 0,
        "reason": "MATRIX_STRONG_SIGNAL_MEDIOCRE_SAFETY",
    },
    ("middle", "good"): {
        "policy": "candidate_low_confidence",
        "track": "exploratory",
        "blocking": 0,
        "reason": "MATRIX_MIDDLE_SIGNAL_GOOD_SAFETY",
    },
    ("middle", "mediocre"): {
        "policy": "diagnostic_only",
        "track": "weak_structured",
        "blocking": 0,
        "reason": "MATRIX_MIDDLE_SIGNAL_MEDIOCRE_SAFETY",
    },
    ("weak", "good"): {
        "policy": "observe_more",
        "track": "weak_structured",
        "blocking": 0,
        "reason": "MATRIX_WEAK_SIGNAL_GOOD_SAFETY",
    },
    ("weak", "mediocre"): {
        "policy": "diagnostic_only",
        "track": "weak_structured",
        "blocking": 0,
        "reason": "MATRIX_WEAK_SIGNAL_MEDIOCRE_SAFETY",
    },
}

_DANGEROUS_BY_MODE: Dict[str, Dict[str, Dict[str, object]]] = {
    "conservative": {
        # Step 178: less conservative routing. Dangerous safety means
        # diagnostic-only for middle/strong signals, not automatic deletion.
        "strong": {"policy": "diagnostic_only", "track": "weak_structured", "blocking": 0, "reason": "MATRIX_STRONG_SIGNAL_DANGEROUS_SAFETY_DIAGNOSTIC_CONSERVATIVE"},
        "middle": {"policy": "diagnostic_only", "track": "weak_structured", "blocking": 0, "reason": "MATRIX_MIDDLE_SIGNAL_DANGEROUS_SAFETY_DIAGNOSTIC_CONSERVATIVE"},
        "weak": {"policy": "blocked", "track": "dropped", "blocking": 1, "reason": "MATRIX_WEAK_SIGNAL_DANGEROUS_SAFETY_BLOCKED"},
    },
    "balanced": {
        "strong": {"policy": "diagnostic_only", "track": "weak_structured", "blocking": 0, "reason": "MATRIX_STRONG_SIGNAL_DANGEROUS_SAFETY_DIAGNOSTIC"},
        "middle": {"policy": "diagnostic_only", "track": "weak_structured", "blocking": 0, "reason": "MATRIX_MIDDLE_SIGNAL_DANGEROUS_SAFETY_DIAGNOSTIC"},
        "weak": {"policy": "blocked", "track": "dropped", "blocking": 1, "reason": "MATRIX_WEAK_SIGNAL_DANGEROUS_SAFETY_BLOCKED"},
    },
    "exploratory": {
        "strong": {"policy": "diagnostic_only", "track": "weak_structured", "blocking": 0, "reason": "MATRIX_STRONG_SIGNAL_DANGEROUS_SAFETY_DIAGNOSTIC"},
        "middle": {"policy": "diagnostic_only", "track": "weak_structured", "blocking": 0, "reason": "MATRIX_MIDDLE_SIGNAL_DANGEROUS_SAFETY_DIAGNOSTIC"},
        "weak": {"policy": "diagnostic_only", "track": "weak_structured", "blocking": 0, "reason": "MATRIX_WEAK_SIGNAL_DANGEROUS_SAFETY_DIAGNOSTIC_EXPLORATORY"},
    },
}


def normalize_signal_grade(value) -> str:
    text = str(value or "").strip().lower()
    return text if text in SIGNAL_GRADES else "block"


def normalize_safety_risk_grade(value) -> str:
    text = str(value or "").strip().lower()
    return text if text in SAFETY_RISK_GRADES else "good"


def normalize_discovery_mode(value) -> str:
    text = str(value or "").strip().lower()
    return text if text in ("conservative", "balanced", "exploratory") else "conservative"


def classify_signal_safety_decision(signal_grade, safety_risk_grade, mode="conservative") -> dict:
    """Return the explicit matrix decision for one signal/safety cell."""

    signal = normalize_signal_grade(signal_grade)
    risk = normalize_safety_risk_grade(safety_risk_grade)
    mode = normalize_discovery_mode(mode)
    cell = f"{signal}__{risk}"

    if signal == "block":
        out = {"policy": "blocked", "track": "dropped", "blocking": 1, "reason": "MATRIX_SIGNAL_BLOCK"}
    elif risk == "critical":
        out = {"policy": "blocked", "track": "dropped", "blocking": 1, "reason": "MATRIX_CRITICAL_SAFETY_BLOCK"}
    elif risk == "dangerous":
        out = dict(_DANGEROUS_BY_MODE[mode].get(signal, _DANGEROUS_BY_MODE[mode]["weak"]))
    else:
        out = dict(_BASE_MATRIX.get((signal, risk), {"policy": "blocked", "track": "dropped", "blocking": 1, "reason": "MATRIX_UNSUPPORTED_CELL_BLOCK"}))

    out.update({
        "signal_safety_cell": cell,
        "signal_safety_policy": out["policy"],
        "signal_safety_matrix_track": out["track"],
        "signal_safety_blocking": int(out["blocking"]),
        "signal_safety_reason_code": out["reason"],
        "signal_safety_matrix_version": 1,
    })
    return out


def is_matrix_blocking(signal_grade, safety_risk_grade, mode="conservative") -> bool:
    return bool(classify_signal_safety_decision(signal_grade, safety_risk_grade, mode=mode)["signal_safety_blocking"])


def matrix_table(mode="conservative") -> Dict[str, Dict[str, dict]]:
    """Return a nested, human-readable matrix for tests/docs."""

    mode = normalize_discovery_mode(mode)
    return {
        signal: {
            risk: classify_signal_safety_decision(signal, risk, mode=mode)
            for risk in SAFETY_RISK_GRADES
        }
        for signal in SIGNAL_GRADES
    }


__all__ = [
    "SIGNAL_GRADES",
    "SAFETY_RISK_GRADES",
    "classify_signal_safety_decision",
    "is_matrix_blocking",
    "matrix_table",
    "normalize_discovery_mode",
    "normalize_safety_risk_grade",
    "normalize_signal_grade",
]
