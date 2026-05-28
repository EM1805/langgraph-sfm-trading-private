from __future__ import annotations

from typing import Any, Dict

"""
Named calibration constants for runtime ranking and policy heuristics.

These values are intentionally separated from the causal-estimation code so that
future domain retuning does not silently change the meaning of the statistical
outputs. The numbers below are ranking / policy defaults for the current agent
safety domain; they are not causal effect sizes.

Retuning guidance:
- Raise severity / evidence thresholds only when false positives are clearly too high.
- Lower thresholds only with supporting retrospective review data.
- Prefer changing one family of constants at a time and record the rationale.
"""

RUNTIME_CALIBRATION_VERSION = "step11"

PATH_ACTIVATION: Dict[str, Any] = {
    "severity_score": {
        # Structural prior for path seriousness before empirical/counterfactual evidence.
        "low": 0.35,
        "medium": 0.55,
        "high": 0.75,
        "critical": 0.92,
    },
    "graph_confidence_bonus": {
        # Bonus added when graph reachability independently supports a path.
        "unknown": 0.0,
        "low": 0.01,
        "medium": 0.03,
        "high": 0.06,
    },
    "amplifier_increment": 0.06,
    "hard_block_increment": 0.08,
    "trigger_graph_alignment_bonus": 0.02,
}

PATH_EVIDENCE: Dict[str, Any] = {
    "match": {
        # Historical similarity weighting for empirical support lookup.
        "action_name": 0.55,
        "environment": 0.15,
        "categorical_key": 0.10,
        "boolean_key": 0.025,
    },
    "label_thresholds": {
        # Support / rate / lift thresholds tuned for conservative evidence labels.
        "high": {"min_support": 5, "min_harm_rate": 0.55, "min_delta": 0.25},
        "medium": {"min_support": 3, "min_harm_rate": 0.33, "min_delta": 0.10},
        "low": {"min_support": 2, "min_harm_rate": 0.00, "min_delta": 0.00},
    },
    "risk_score_bonus": {
        # Small additive bump so empirical evidence nudges but does not dominate structural risk.
        "high": 0.08,
        "medium": 0.04,
        "low": 0.015,
        "none": 0.0,
    },
    "default_min_similarity": 0.65,
}

PATH_VALIDATION: Dict[str, Any] = {
    "effect_agreement": {
        "discordant_multiplier": 0.35,
        "strong_ratio": 0.72,
        "strong_base": 0.70,
        "strong_slope": 0.30,
        "moderate_base": 0.45,
        "moderate_slope": 0.35,
    },
    "interval_width": {
        "tight_width": 0.12,
        "usable_width": 0.28,
        "tight_denominator_floor": 0.35,
        "tight_effect_multiplier": 2.0,
        "tight_effect_offset": 0.05,
        "usable_base": 0.72,
        "usable_slope": 0.45,
        "wide_base": 0.42,
        "wide_slope": 0.35,
    },
    "negative_control": {
        "discordant_delta_fail": 0.35,
        "pass_rate_max": 0.08,
        "review_rate_max": 0.18,
    },
    "score_components": {
        # These are policy-facing scoring weights, not identification weights.
        "overlap_gap_scale": 0.35,
        "design_medium": 0.68,
        "design_low": 0.34,
        "placebo_gap_scale": 0.70,
        "pretrend_testable_fallback": 0.40,
        "pretrend_untestable_fallback": 0.22,
        "strata_valid_weight": 0.30,
        "strata_exact_control_weight": 0.22,
        "control_concentration_baseline": 0.35,
        "control_concentration_scale": 0.55,
    },
    "bands": {
        "high_score": 0.74,
        "medium_score": 0.50,
        "promotion_ident_high_score": 0.66,
        "promotion_ident_medium_score": 0.52,
        "promotion_match_high": 0.72,
        "promotion_match_medium": 0.55,
    },
}

POLICY_ENGINE: Dict[str, Any] = {
    "identification_score_defaults": {
        "high": 0.75,
        "medium": 0.50,
        "low": 0.25,
        "none": 0.0,
    },
    "evidence_tier": {
        "high_risk_min": 0.85,
        "medium_risk_min": 0.67,
    },
    "identification_tier": {
        "high_score_min": 0.68,
        "medium_score_min": 0.45,
    },
    "production_review_risk_min": 0.70,
}

CALIBRATION_NOTES: Dict[str, str] = {
    "path_activation": "Structural priors and graph bonuses for early path ranking.",
    "path_evidence": "Historical matching and evidence-label thresholds for empirical support.",
    "path_validation": "Policy-facing validation aggregation thresholds and bands.",
    "policy_engine": "Decision-tier thresholds for PASS/REVIEW/BLOCK routing.",
}
