"""Routing/track classification helpers for Discovery proposals.

This module keeps proposal/drop classification rules out of the main
``ProposalEngine`` orchestration code.  The rules are intentionally
conservative and preserve the previous behavior, but make the reasons
explicit, deduplicated, and unit-testable.
"""

from __future__ import annotations

from runtime_env import configure_scientific_runtime
configure_scientific_runtime()


from typing import Iterable, List

import numpy as np
import pandas as pd

from .signal_safety_matrix import classify_signal_safety_decision


def _calibration():
    """Import calibration lazily to avoid pcmci_discovery_parts package init cycles."""
    from pcmci_discovery_parts import calibration as CAL
    return CAL


def _as_str(value) -> str:
    if value is None:
        return ""
    try:
        if isinstance(value, float) and np.isnan(value):
            return ""
    except TypeError:
        pass
    return str(value)


def _safe_float(value, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return out


_EXPLORATORY_HARD_DROP_TOKENS = (
    "LEAKAGE_BLOCK",
    "SENSITIVE_FEATURE",
    "BLOCKLISTED",
)

_BALANCED_HARD_DROP_TOKENS = (
    "LEAKAGE_BLOCK",
    "FAILS_CONDITIONAL_TEST",
)

_CONSERVATIVE_HARD_DROP_TOKENS = (
    # Step 178: keep hard drops for true safety/structural traps only.
    # Placebo/MCI/PC1/CI failures remain diagnostic evidence and drop reasons,
    # but no longer create a second hard-blocking path by themselves.
    "LEAKAGE_BLOCK",
    "COLLIDER_PATTERN_PRUNE",
    "SENSITIVE_FEATURE",
    "BLOCKLISTED",
)


_PRUNE_TOKENS = (
    "PC1_PRUNE",
    "MCI_PRUNE",
    "MCI_Q_PRUNE",
    "CONFOUNDER_PATTERN_PRUNE",
    "COLLIDER_PATTERN_PRUNE",
)


def _dedupe_reason_tokens(tokens: Iterable[str]) -> List[str]:
    """Return non-empty tokens in first-seen order."""

    out: List[str] = []
    seen = set()
    for token in tokens:
        text = _as_str(token).strip()
        if not text:
            continue
        # Existing reason fields can already be pipe-delimited.
        for part in [p.strip() for p in text.split("|") if p.strip()]:
            if part not in seen:
                seen.add(part)
                out.append(part)
    return out


def hard_drop_tokens_for_mode(discovery_mode: str) -> tuple:
    """Return exclusion tokens that remain hard drops for a discovery mode."""

    mode = _as_str(discovery_mode).lower() or "conservative"
    if mode == "exploratory":
        return _EXPLORATORY_HARD_DROP_TOKENS
    if mode == "balanced":
        return _BALANCED_HARD_DROP_TOKENS
    return _CONSERVATIVE_HARD_DROP_TOKENS


def hard_drop_mask(proposals: pd.DataFrame, cfg) -> pd.Series:
    """Boolean mask for proposals blocked by hard safety/structure reasons."""

    if proposals is None or len(proposals) == 0:
        return pd.Series([], dtype=bool)
    base = pd.Series(False, index=proposals.index)
    if "exclusion_reasons" in proposals.columns:
        tokens = hard_drop_tokens_for_mode(getattr(cfg, "discovery_mode", "conservative"))
        pattern = "|".join(tokens)
        base = proposals["exclusion_reasons"].astype(str).str.contains(pattern, regex=True, na=False)

    mode = getattr(cfg, "discovery_mode", "conservative")
    if {"hypothesis_signal_grade", "safety_risk_grade"}.issubset(proposals.columns):
        matrix_block = proposals.apply(
            lambda r: bool(classify_signal_safety_decision(
                r.get("hypothesis_signal_grade", "block"),
                r.get("safety_risk_grade", "good"),
                mode=mode,
            )["signal_safety_blocking"]),
            axis=1,
        )
    else:
        matrix_block = pd.Series(False, index=proposals.index)
    safety_blocking = proposals.get("safety_blocking")
    if safety_blocking is None:
        explicit_safety_block = pd.Series(False, index=proposals.index)
    else:
        explicit_safety_block = (
            pd.to_numeric(safety_blocking, errors="coerce")
            .fillna(0)
            .astype(int)
            .eq(1)
        )
    return base | matrix_block | explicit_safety_block


def build_drop_reason(row, cfg) -> str:
    """Build a deterministic, deduplicated drop reason string for one proposal.

    This replaces the previous inline DataFrame.apply(lambda ...) block in
    ``engine.py``.  It preserves the same thresholds and reason tokens while
    avoiding duplicated prune markers when they are already present inside
    ``exclusion_reasons``.
    """

    exclusion_text = _as_str(row.get("exclusion_reasons", ""))
    reasons: List[str] = []

    if _safe_float(row.get("selection_score", np.nan), 0.0) < getattr(cfg, "keep_min_selection_score", 0.46):
        reasons.append("LOW_SELECTION_SCORE")
    if _safe_float(row.get("discovery_evidence_score", np.nan), 0.0) < getattr(cfg, "keep_min_evidence_score", 0.40):
        reasons.append("LOW_EVIDENCE_SCORE")
    if _safe_float(row.get("priority_score", np.nan), 0.0) < getattr(cfg, "keep_min_priority", 0.0):
        reasons.append("LOW_PRIORITY")
    CAL = _calibration()

    if _safe_float(row.get("causal_plausibility_score", np.nan), 0.0) < CAL.TRACK_WEAK_PLAUSIBILITY:
        reasons.append("LOW_CAUSAL_PLAUSIBILITY")
    if _safe_float(row.get("downside_action_score", np.nan), 0.0) < CAL.TRACK_WEAK_DOWNSIDE:
        reasons.append("LOW_DOWNSIDE")
    if _safe_float(row.get("structural_risk_score", np.nan), 0.0) > CAL.DROP_HIGH_STRUCTURAL:
        reasons.append("HIGH_STRUCTURAL_RISK")
    if _as_str(row.get("hypothesis_signal_grade", "")) == "block":
        reasons.append("SIGNAL_BLOCK")
    if int(_safe_float(row.get("safety_blocking", 0), 0)) == 1:
        reasons.append("SAFETY_BLOCKING")
    matrix_reason = _as_str(row.get("signal_safety_reason_code", ""))
    if matrix_reason:
        reasons.append(matrix_reason)
    matrix_policy = _as_str(row.get("signal_safety_policy", ""))
    if matrix_policy == "blocked":
        reasons.append("SIGNAL_SAFETY_MATRIX_BLOCK")
    safety_grade = _as_str(row.get("safety_risk_grade", ""))
    if safety_grade in ("critical", "dangerous"):
        reasons.append("SAFETY_RISK_" + safety_grade.upper())
    safety_reasons = _as_str(row.get("safety_risk_reason_codes", ""))
    if safety_reasons and safety_reasons != "NO_MAJOR_SAFETY_RISK":
        reasons.append(safety_reasons)

    reasons.append(exclusion_text)
    # Preserve explicit prune tokens even if an exclusion string used extra text,
    # while dedupe prevents repeated tokens such as STRUCTURAL_PRUNE twice.
    for token in _PRUNE_TOKENS:
        if token in exclusion_text:
            reasons.append(token)

    return "|".join(_dedupe_reason_tokens(reasons))


def apply_drop_reasons(proposals: pd.DataFrame, cfg) -> pd.DataFrame:
    """Fill ``drop_reason`` only for proposals whose track remains dropped."""

    if proposals is None or len(proposals) == 0:
        return proposals
    out = proposals.copy()
    out["drop_reason"] = ""
    dropped = out["discovery_track"] == "dropped" if "discovery_track" in out.columns else pd.Series(False, index=out.index)
    if dropped.any():
        out.loc[dropped, "drop_reason"] = out.loc[dropped].apply(lambda r: build_drop_reason(r, cfg), axis=1)
    return out
