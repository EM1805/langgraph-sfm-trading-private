from __future__ import annotations
from typing import Any, Dict, List

from .runtime_calibration import POLICY_ENGINE


def _safe_str(value: Any) -> str:
    return str(value or "").strip().lower()


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Parse numeric policy inputs without letting dirty runtime data crash veto decisions."""
    try:
        if value is None:
            return default
        out = float(value)
        if out != out:  # NaN
            return default
        if out in (float("inf"), float("-inf")):
            return default
        return out
    except (TypeError, ValueError):
        return default


def _policy_profile_from_path(path: Dict[str, Any]) -> Dict[str, Any]:
    plan = dict(path.get("validation_plan", {}) or {})
    source_role = _safe_str(path.get("source_role") or plan.get("source_role") or plan.get("treatment_role") or path.get("plan_source_role"))
    outcome_role = _safe_str(path.get("outcome_role") or plan.get("outcome_role") or path.get("plan_outcome_role"))
    edge_family = _safe_str(path.get("edge_family") or plan.get("edge_family") or path.get("plan_edge_family"))
    validation_design = _safe_str(path.get("counterfactual_validation_design") or plan.get("validation_design") or path.get("plan_validation_design"))
    preferred_estimand = _safe_str(path.get("counterfactual_preferred_estimand") or plan.get("preferred_estimand") or path.get("plan_preferred_estimand"))

    profile = {
        "policy_case": "default",
        "policy_weight": 1.0,
        "production_review_risk_min": _safe_float(POLICY_ENGINE["production_review_risk_min"], 0.72),
        "high_risk_min": _safe_float(POLICY_ENGINE["evidence_tier"]["high_risk_min"], 0.85),
        "medium_risk_min": _safe_float(POLICY_ENGINE["evidence_tier"]["medium_risk_min"], 0.65),
        "high_ident_score_min": _safe_float(POLICY_ENGINE["identification_tier"]["high_score_min"], 0.70),
        "medium_ident_score_min": _safe_float(POLICY_ENGINE["identification_tier"]["medium_score_min"], 0.45),
        "require_high_ident_for_hard_block": False,
        "notes": [],
        "source_role": source_role,
        "outcome_role": outcome_role,
        "edge_family": edge_family,
        "validation_design": validation_design,
        "preferred_estimand": preferred_estimand,
    }

    if source_role in {"decision", "guardrail"} and outcome_role in {"harm", "outcome"}:
        profile.update({
            "policy_case": "decision_to_harm" if outcome_role == "harm" else "decision_to_outcome",
            "policy_weight": 1.15 if outcome_role == "harm" else 1.08,
            "production_review_risk_min": min(profile["production_review_risk_min"], 0.62 if outcome_role == "harm" else 0.66),
            "medium_risk_min": min(profile["medium_risk_min"], 0.60 if outcome_role == "harm" else 0.64),
            "high_risk_min": min(profile["high_risk_min"], 0.80 if outcome_role == "harm" else 0.83),
            "require_high_ident_for_hard_block": True,
        })
        profile["notes"].append("Policy tightened for action-like causes with direct outcome relevance.")
    elif source_role == "context":
        profile.update({
            "policy_case": "context_to_outcome",
            "policy_weight": 0.93,
            "production_review_risk_min": max(profile["production_review_risk_min"], 0.74),
            "medium_risk_min": max(profile["medium_risk_min"], 0.70),
            "high_ident_score_min": max(profile["high_ident_score_min"], 0.70),
            "medium_ident_score_min": max(profile["medium_ident_score_min"], 0.48),
            "require_high_ident_for_hard_block": True,
        })
        profile["notes"].append("Policy requires stronger causal support for context-only signals.")

    if edge_family in {"decision_to_outcome", "guardrail_to_outcome", "decision_to_harm", "guardrail_to_harm"}:
        profile["policy_weight"] = max(profile["policy_weight"], 1.10)
        profile["production_review_risk_min"] = min(profile["production_review_risk_min"], 0.64)
    elif edge_family in {"context_to_outcome", "context_to_harm"}:
        profile["policy_weight"] = min(profile["policy_weight"], 0.95)

    if validation_design in {"matching+did", "priority_validation", "within_stratum_matched_did"} or preferred_estimand in {"risk_difference_att", "att", "ate"}:
        profile["medium_ident_score_min"] = min(profile["medium_ident_score_min"], 0.43)
        profile["notes"].append("Validation design supports stronger policy trust in causal estimates.")

    return profile


def _tier_from_path(path: Dict[str, Any]) -> Dict[str, str]:
    profile = _policy_profile_from_path(path)
    risk = _safe_float(path.get("risk_score"), 0.0)
    severity = str(path.get("severity", "low"))
    empirical = str(path.get("empirical_evidence_strength", "none"))
    counterfactual = str(path.get("counterfactual_evidence_strength", "none"))
    identification = str(path.get("identification_support", path.get("counterfactual_identification_support", "none")))
    placebo_pass = bool(path.get("placebo_pass", True))
    confounding = str(path.get("counterfactual_confounding_risk", path.get("sensitivity_confounding_risk", "unknown")))
    graph_supported = bool(path.get("graph_supported", False))
    design_strength = str(path.get("counterfactual_design_strength", "medium" if str(path.get("counterfactual_identification_support", "none")) in {"medium", "high"} else "low"))
    default_ident_scores = POLICY_ENGINE["identification_score_defaults"]
    ident_default = _safe_float(default_ident_scores.get(identification, default_ident_scores["none"]), 0.0)
    ident_score = _safe_float(path.get("identification_assessment", {}).get("identification_score", ident_default), ident_default) if isinstance(path.get("identification_assessment"), dict) else ident_default

    weighted_risk = min(1.0, risk * _safe_float(profile["policy_weight"], 1.0))

    if weighted_risk >= _safe_float(profile["high_risk_min"], 0.85) and (empirical in {"medium", "high"} or counterfactual in {"medium", "high"}):
        evidence_tier = "high"
    elif weighted_risk >= _safe_float(profile["medium_risk_min"], 0.65) and (empirical != "none" or counterfactual != "none" or graph_supported):
        evidence_tier = "medium"
    elif risk > 0:
        evidence_tier = "low"
    else:
        evidence_tier = "none"

    if identification == "high" and placebo_pass and confounding in {"low", "medium"} and design_strength in {"medium", "high"} and ident_score >= _safe_float(profile["high_ident_score_min"], 0.70):
        identification_tier = "high"
    elif identification in {"medium", "high"} and confounding != "high" and ident_score >= _safe_float(profile["medium_ident_score_min"], 0.45):
        identification_tier = "medium"
    elif identification in {"low", "medium", "high"}:
        identification_tier = "low"
    else:
        identification_tier = "none"

    structural_tier = "high" if severity in {"critical", "high"} and graph_supported else ("medium" if graph_supported else "low")
    return {
        "evidence_tier": evidence_tier,
        "identification_tier": identification_tier,
        "structural_tier": structural_tier,
        "design_strength": design_strength,
        "weighted_risk_score": round(weighted_risk, 3),
        "policy_case": profile["policy_case"],
        "policy_weight": round(_safe_float(profile["policy_weight"], 1.0), 3),
        "policy_thresholds": {
            "production_review_risk_min": profile["production_review_risk_min"],
            "high_risk_min": profile["high_risk_min"],
            "medium_risk_min": profile["medium_risk_min"],
            "high_ident_score_min": profile["high_ident_score_min"],
            "medium_ident_score_min": profile["medium_ident_score_min"],
        },
        "policy_source_role": profile["source_role"],
        "policy_outcome_role": profile["outcome_role"],
        "policy_edge_family": profile["edge_family"],
        "policy_validation_design": profile["validation_design"],
        "policy_preferred_estimand": profile["preferred_estimand"],
        "policy_notes": list(profile["notes"]),
        "require_high_ident_for_hard_block": bool(profile["require_high_ident_for_hard_block"]),
    }


def _attach_policy_metadata(paths: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for p in paths:
        merged = dict(p)
        merged.update(_tier_from_path(merged))
        out.append(merged)
    return out


def decide_veto(paths: List[Dict[str, Any]], context_flags: Dict[str, Any]) -> Dict[str, Any]:
    if not paths:
        return {
            "decision": "PASS",
            "reason_codes": ["NO_DANGEROUS_PATH"],
            "max_risk_score": 0.0,
            "activated_path_count": 0,
            "evidence_tier": "none",
            "identification_tier": "none",
            "structural_tier": "none",
            "design_strength": "none",
            "decision_basis": {
                "top_path_id": None,
                "evidence_tier": "none",
                "identification_tier": "none",
                "structural_tier": "none",
                "design_strength": "none",
                "identification_score": 0.0,
            },
            "notes": [],
        }

    paths = _attach_policy_metadata(paths)
    paths = sorted(paths, key=lambda x: (-_safe_float(x.get("weighted_risk_score", x.get("risk_score", 0.0)), 0.0), x.get("path_id", "")))
    top = paths[0]
    production = bool(context_flags.get("production_environment", False))
    severe_paths = [p for p in paths if str(p.get("severity", "low")) in {"high", "critical"}]
    top_evidence = str(top.get("evidence_tier", "none"))
    top_identification = str(top.get("identification_tier", "none"))
    top_structural = str(top.get("structural_tier", "low"))
    top_policy_case = str(top.get("policy_case", "default"))
    top_review_min = _safe_float(top.get("policy_thresholds", {}).get("production_review_risk_min", POLICY_ENGINE["production_review_risk_min"]), 0.72)

    has_hard_block = any(p.get("hard_block_hits") for p in paths)
    notes: List[str] = []

    if has_hard_block:
        decision = "HARD_BLOCK"
        reason_codes = ["PATH_HARD_BLOCK", str(top["path_id"]).upper()]
        if top_policy_case != "default":
            notes.extend(top.get("policy_notes", []))
    elif production and top_structural == "high" and top_evidence == "high":
        if top_identification == "high":
            decision = "HARD_BLOCK"
            reason_codes = ["STRUCTURAL_AND_CAUSAL_HIGH", str(top["path_id"]).upper()]
        else:
            decision = "REVIEW"
            reason_codes = ["HIGH_RISK_WEAK_IDENTIFICATION", str(top["path_id"]).upper()]
            notes.append("Risk appears high, but counterfactual identification remains limited.")
    elif production and top_structural == "high":
        decision = "REVIEW"
        reason_codes = ["GRAPH_SUPPORTED_SEVERE_PATH", str(top["path_id"]).upper()]
        if top_identification in {"low", "none"}:
            notes.append("Structural risk dominates because causal identification is weak.")
    elif production and top_evidence == "high":
        decision = "REVIEW"
        reason_codes = ["EMPIRICALLY_ELEVATED_PATH", str(top["path_id"]).upper()]
        if top_identification in {"low", "none"}:
            notes.append("Empirical/counterfactual effect looks elevated, but identification is weak.")
    elif production and top_structural == "medium" and _safe_float(top.get("weighted_risk_score", top.get("risk_score", 0.0)), 0.0) >= top_review_min:
        decision = "REVIEW"
        reason_codes = ["ELEVATED_STRUCTURAL_PATH_IN_PRODUCTION", str(top["path_id"]).upper()]
    elif severe_paths and production:
        decision = "REVIEW"
        reason_codes = ["SEVERE_PATH_IN_PRODUCTION", str(top["path_id"]).upper()]
    elif production and top_structural == "medium" and top_identification in {"low", "none"}:
        decision = "REVIEW"
        reason_codes = ["STRUCTURAL_MEDIUM_WEAK_IDENTIFICATION", str(top["path_id"]).upper()]
        notes.append("Moderate structural risk in production with weak identification is routed to review.")
    elif production and top_structural == "medium" and top_evidence == "medium":
        decision = "REVIEW"
        reason_codes = ["ELEVATED_PATH_IN_PRODUCTION", str(top["path_id"]).upper()]
        if top_identification in {"low", "none"}:
            notes.append("Moderate structural risk in production with weak identification is routed to review.")
    elif production and top_evidence == "medium" and top_identification in {"low", "none"}:
        decision = "REVIEW"
        reason_codes = ["MEDIUM_EVIDENCE_WEAK_IDENTIFICATION", str(top["path_id"]).upper()]
        notes.append("Moderate evidence in production with weak identification is routed to review.")
    elif top_evidence == "medium":
        decision = "PASS_WITH_WARNING"
        reason_codes = ["ELEVATED_PATH_WARNING", str(top["path_id"]).upper()]
    elif severe_paths:
        decision = "PASS_WITH_WARNING"
        reason_codes = ["SEVERE_PATH_WARNING", str(top["path_id"]).upper()]
    else:
        decision = "PASS_WITH_WARNING"
        reason_codes = ["MODERATE_PATH_WARNING", str(top["path_id"]).upper()]

    if top_policy_case != "default":
        notes.extend([n for n in top.get("policy_notes", []) if n not in notes])

    return {
        "decision": decision,
        "reason_codes": reason_codes,
        "max_risk_score": _safe_float(top.get("risk_score"), 0.0),
        "max_weighted_risk_score": _safe_float(top.get("weighted_risk_score", top.get("risk_score")), 0.0),
        "activated_path_count": len(paths),
        "evidence_tier": top_evidence,
        "identification_tier": top_identification,
        "structural_tier": top_structural,
        "design_strength": top.get("design_strength", "low"),
        "decision_basis": {
            "top_path_id": top.get("path_id"),
            "evidence_tier": top_evidence,
            "identification_tier": top_identification,
            "structural_tier": top_structural,
            "design_strength": top.get("design_strength", "low"),
            "identification_score": _safe_float(top.get("identification_assessment", {}).get("identification_score"), 0.0) if isinstance(top.get("identification_assessment"), dict) else 0.0,
            "counterfactual_method": top.get("counterfactual_method"),
            "placebo_pass": bool(top.get("placebo_pass", True)),
            "counterfactual_confounding_risk": top.get("counterfactual_confounding_risk"),
            "weighted_risk_score": _safe_float(top.get("weighted_risk_score", top.get("risk_score", 0.0)), 0.0),
            "policy_case": top.get("policy_case", "default"),
            "policy_weight": _safe_float(top.get("policy_weight"), 1.0),
            "policy_source_role": top.get("policy_source_role", ""),
            "policy_outcome_role": top.get("policy_outcome_role", ""),
            "policy_edge_family": top.get("policy_edge_family", ""),
            "policy_validation_design": top.get("policy_validation_design", ""),
            "policy_preferred_estimand": top.get("policy_preferred_estimand", ""),
            "policy_thresholds": top.get("policy_thresholds", {}),
        },
        "notes": notes,
    }
