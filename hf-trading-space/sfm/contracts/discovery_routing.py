"""Separate hypothesis signal grading from safety-risk grading.

Discovery routing uses two independent axes:

- hypothesis_signal_* answers: "how promising is this candidate signal?"
- safety_risk_* answers: "how suspicious/dangerous is this candidate?"

A strong signal never clears a critical safety risk.  These scores are
routing metadata only; they do not identify or authorize causal effects.
"""

from __future__ import annotations

from runtime_env import configure_scientific_runtime
configure_scientific_runtime()


from typing import Iterable, List, Tuple

import math
import numpy as np
import pandas as pd

from .signal_safety_matrix import classify_signal_safety_decision


def _as_str(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def _safe_float(value, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return out


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, _safe_float(value, 0.0))))


def _tokens(text: str) -> List[str]:
    out: List[str] = []
    seen = set()
    for part in _as_str(text).replace(",", "|").split("|"):
        token = part.strip()
        if token and token not in seen:
            seen.add(token)
            out.append(token)
    return out


def _join(tokens: Iterable[str]) -> str:
    out: List[str] = []
    seen = set()
    for token in tokens:
        text = _as_str(token).strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return "|".join(out)


def hypothesis_signal_grade(score: float, blocking: bool = False) -> str:
    """Map a signal score to block/weak/middle/strong."""

    if blocking:
        return "block"
    score = _clip01(score)
    if score < 0.25:
        return "block"
    if score < 0.45:
        return "weak"
    if score < 0.68:
        return "middle"
    return "strong"


def safety_risk_grade(score: float) -> str:
    """Map a safety risk score to good/mediocre/dangerous/critical.

    Step 178 intentionally makes this axis less conservative: ordinary
    estimation-falsification/structural concerns should usually route a
    candidate to diagnostic review (dangerous/mediocre), not auto-escalate
    every proposal to critical.  Critical is reserved for very strong safety
    evidence such as leakage/collider-style structural traps or extreme risk.
    """

    score = _clip01(score)
    if score >= 0.90:
        return "critical"
    if score >= 0.68:
        return "dangerous"
    if score >= 0.35:
        return "mediocre"
    return "good"


def compute_hypothesis_signal(row, cfg=None) -> dict:
    """Compute the positive hypothesis-signal axis for one proposal row."""

    selection = _clip01(_safe_float(row.get("selection_score", row.get("priority_score", 0.0)), 0.0))
    evidence = _clip01(_safe_float(row.get("discovery_evidence_score", 0.0), 0.0))
    plaus = _clip01(_safe_float(row.get("causal_plausibility_score", 0.0), 0.0))
    priority = _clip01(_safe_float(row.get("priority_score", 0.0), 0.0))
    mci = _clip01(_safe_float(row.get("mci_score", 0.5), 0.5))
    pc1 = _clip01(_safe_float(row.get("pc1_score", 0.5), 0.5))
    stability = _clip01(_safe_float(row.get("rolling_stability", 0.0), 0.0))
    support_votes = _safe_float(row.get("discovery_support_votes", 0), 0.0)
    contradiction_votes = _safe_float(row.get("discovery_contradiction_votes", 0), 0.0)

    raw = (
        0.30 * selection
        + 0.30 * evidence
        + 0.15 * plaus
        + 0.10 * priority
        + 0.07 * mci
        + 0.04 * pc1
        + 0.04 * stability
    )
    vote_bonus = min(0.04, 0.01 * max(0.0, support_votes - contradiction_votes))
    score = _clip01(raw + vote_bonus)

    exclusion = _as_str(row.get("exclusion_reasons", ""))
    # Step 178: do not let legacy prune/weak-signal tokens create a second,
    # hidden SIGNAL_BLOCK system.  The signal grade is now score-based; legacy
    # tokens are still reported as diagnostics but no longer override a middle
    # or strong hypothesis signal.
    signal_diagnostic_tokens = {
        "FAILS_CONDITIONAL_TEST",
        "MCI_PRUNE",
        "MCI_Q_PRUNE",
        "PC1_PRUNE",
            "WEAK_SIGNAL",
    }
    found = [token for token in signal_diagnostic_tokens if token in exclusion]
    blocking = score < 0.22

    reasons = []
    if score >= 0.68:
        reasons.append("SIGNAL_STRONG_SCORE")
    elif score >= 0.45:
        reasons.append("SIGNAL_MIDDLE_SCORE")
    elif score >= 0.25:
        reasons.append("SIGNAL_WEAK_SCORE")
    else:
        reasons.append("SIGNAL_BLOCK_SCORE")
    if mci >= 0.60:
        reasons.append("MCI_SUPPORT")
    if pc1 >= 0.60:
        reasons.append("PC1_SUPPORT")
    if stability >= 0.60:
        reasons.append("STABLE_SIGNAL")
    reasons.extend(found)

    return {
        "hypothesis_signal_score": float(score),
        "hypothesis_signal_grade": hypothesis_signal_grade(score, blocking=blocking),
        "hypothesis_signal_reason_codes": _join(reasons),
    }


def compute_safety_risk(row, cfg=None) -> dict:
    """Compute the negative safety-risk axis for one proposal row."""

    exclusion = _as_str(row.get("exclusion_reasons", ""))
    risk_flags = _as_str(row.get("risk_flags", ""))
    reasons: List[str] = []
    risk = 0.0

    def mark(score: float, reason: str) -> None:
        nonlocal risk
        risk = max(risk, _clip01(score))
        reasons.append(reason)

    # Step 178: only leakage-style evidence remains an immediate critical risk.
    # Other failed diagnostics are safety warnings/diagnostic routes, not
    # automatic critical blocks.
    if "LEAKAGE_BLOCK" in exclusion or "leakage_suspect" in risk_flags or "future_stronger_than_lead" in risk_flags:
        mark(1.0, "LEAKAGE_RISK")
    leakage_score = _safe_float(row.get("leakage_score", np.nan), np.nan)
    if np.isfinite(leakage_score) and leakage_score >= 0.70:
        mark(leakage_score, "LEAKAGE_SCORE_HIGH")

    # Placebo and negative-control diagnostics are intentionally ignored by
    # Discovery safety-risk scoring. They are Estimation-owned falsification
    # gates and appear in effect_estimates.csv plus dedicated check artifacts.

    structural = _safe_float(row.get("structural_risk_score", np.nan), np.nan)
    if np.isfinite(structural) and structural >= 0.90:
        mark(min(0.86, structural), "STRUCTURAL_RISK_HIGH_DIAGNOSTIC")
    elif np.isfinite(structural) and structural >= 0.55:
        mark(min(0.62, structural), "STRUCTURAL_RISK_MEDIUM_DIAGNOSTIC")

    confounding = _safe_float(row.get("confounding_risk_score", np.nan), np.nan)
    if np.isfinite(confounding) and confounding >= 0.85:
        mark(0.50, "CONFOUNDING_RISK_HIGH_DIAGNOSTIC")
    elif np.isfinite(confounding) and confounding >= 0.70:
        mark(0.40, "CONFOUNDING_RISK_SOFT_DIAGNOSTIC")
    if "CONFOUNDER_PATTERN_PRUNE" in exclusion:
        # Legacy token: do not let it dominate final safety by itself.
        mark(0.42, "CONFOUNDER_PATTERN_SOFT_DIAGNOSTIC")
    if "COLLIDER_PATTERN_PRUNE" in exclusion:
        mark(0.90, "COLLIDER_PATTERN_RISK")

    regime_shift = _safe_float(row.get("regime_shift_score", np.nan), np.nan)
    if np.isfinite(regime_shift) and regime_shift >= 0.65:
        mark(0.58, "DRIFT_OR_REGIME_RISK")

    grade = safety_risk_grade(risk)
    blocking = int(any(r in reasons for r in ("LEAKAGE_RISK", "COLLIDER_PATTERN_RISK")) or (grade == "critical" and risk >= 0.95))
    return {
        "safety_risk_score": float(_clip01(risk)),
        "safety_risk_grade": grade,
        "safety_risk_reason_codes": _join(reasons) if reasons else "NO_MAJOR_SAFETY_RISK",
        "safety_blocking": int(blocking),
    }


def score_row(row, cfg=None) -> dict:
    out = {}
    out.update(compute_hypothesis_signal(row, cfg=cfg))
    out.update(compute_safety_risk(row, cfg=cfg))
    return out


def apply_signal_and_safety_scores(proposals: pd.DataFrame, cfg=None) -> pd.DataFrame:
    """Attach separate signal and safety-risk columns to a proposal frame."""

    if proposals is None or len(proposals) == 0:
        return proposals
    out = proposals.copy()
    scores = out.apply(lambda r: pd.Series(score_row(r, cfg=cfg)), axis=1)
    for col in scores.columns:
        out[col] = scores[col]

    mode = getattr(cfg, "discovery_mode", "conservative") if cfg is not None else "conservative"
    decisions = out.apply(
        lambda r: pd.Series(classify_signal_safety_decision(
            r.get("hypothesis_signal_grade", "block"),
            r.get("safety_risk_grade", "good"),
            mode=mode,
        )),
        axis=1,
    )
    for col in decisions.columns:
        out[col] = decisions[col]
    return out


__all__ = [
    "apply_signal_and_safety_scores",
    "compute_hypothesis_signal",
    "compute_safety_risk",
    "hypothesis_signal_grade",
    "safety_risk_grade",
    "classify_signal_safety_decision",
    "score_row",
]
