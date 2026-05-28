from __future__ import annotations

from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Tuple

from .path_evidence import _load_jsonl
from .path_counterfactual import _build_matched_sets, _weighted_harm_rate
from .runtime_calibration import PATH_VALIDATION
from .action_registry_v2 import load_action_registry
from .path_library import load_path_library


def _available_harms(events: List[Dict[str, Any]]) -> List[str]:
    harms = set()
    for e in events:
        for h in list(e.get("observed_harms", []) or []):
            if isinstance(h, str) and h:
                harms.add(h)
    return sorted(harms)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError, OverflowError):
        return int(default)


def _effect_agreement(path: Dict[str, Any]) -> Tuple[str, float, Dict[str, float]]:
    metrics = {
        "risk_delta": _safe_float(path.get("counterfactual_risk_delta", 0.0)),
        "did_effect": _safe_float(path.get("counterfactual_did_effect", 0.0)),
        "dr_risk_delta": _safe_float(path.get("counterfactual_dr_risk_delta", 0.0)),
        "dr_did_effect": _safe_float(path.get("counterfactual_dr_did_effect", 0.0)),
        "stratified_risk_delta": _safe_float(path.get("counterfactual_stratified_risk_delta", 0.0)),
        "stratified_did_effect": _safe_float(path.get("counterfactual_stratified_did_effect", 0.0)),
    }
    vals = [v for v in metrics.values() if abs(v) > 1e-9]
    if not vals:
        return "unknown", 0.0, metrics
    signs = {1 if v > 0 else -1 for v in vals}
    magnitudes = [abs(v) for v in vals]
    med = median(magnitudes) if magnitudes else 0.0
    spread = (max(magnitudes) - min(magnitudes)) if len(magnitudes) >= 2 else 0.0
    agreement_ratio = max(0.0, 1.0 - (spread / max(0.08, med + 1e-9)))
    if len(signs) > 1:
        return "discordant", round(max(0.0, 0.35 * agreement_ratio), 3), metrics
    if agreement_ratio >= 0.72:
        return "strong", round(min(1.0, 0.70 + 0.30 * agreement_ratio), 3), metrics
    if agreement_ratio >= 0.42:
        return "moderate", round(min(1.0, 0.45 + 0.35 * agreement_ratio), 3), metrics
    return "weak", round(max(0.1, 0.25 + 0.25 * agreement_ratio), 3), metrics


def _ci_status(ci_low: float, ci_high: float, effect: float) -> Tuple[str, float, float]:
    if ci_low == 0.0 and ci_high == 0.0 and abs(effect) < 1e-9:
        return "unknown", 0.0, 1.0
    width = max(0.0, ci_high - ci_low)
    crosses_zero = ci_low <= 0.0 <= ci_high
    eff_abs = abs(effect)
    if not crosses_zero and width <= max(0.28, 1.4 * eff_abs):
        return "tight", round(max(0.0, 1.0 - width / max(0.35, 2.0 * eff_abs + 0.05)), 3), round(width, 3)
    if width <= max(0.45, 2.8 * eff_abs + 0.05):
        return "usable", round(max(0.0, 0.72 - 0.45 * width), 3), round(width, 3)
    return "wide", round(max(0.0, 0.42 - 0.35 * width), 3), round(width, 3)


def _negative_control_status(max_abs_delta: float, n_t: int, n_c: int, available_count: int) -> str:
    if n_t < 2 or n_c < 2 or available_count <= 0:
        return "unknown"
    if max_abs_delta >= float(PATH_VALIDATION["negative_control"]["discordant_delta_fail"]):
        return "fail"
    if max_abs_delta >= 0.20:
        return "warning"
    return "pass"



def _plan_coherence(path: Dict[str, Any]) -> Tuple[str, float, Dict[str, Any]]:
    plan = dict(path.get('validation_plan', {}) or {})
    if not plan:
        return 'unknown', 0.35, {'has_plan': False}
    covs = [str(x).strip() for x in list(plan.get('candidate_covariates', []) or []) if str(x).strip()]
    post = [str(x).strip() for x in list(plan.get('post_treatment_columns', []) or []) if str(x).strip()]
    vd = str(plan.get('validation_design', '') or '').strip().lower()
    estimand = str(plan.get('preferred_estimand', '') or '').strip().lower()
    used_covs = [str(x).strip() for x in list(path.get('counterfactual_plan_covariates_used', []) or []) if str(x).strip()]
    used_post = [str(x).strip() for x in list(path.get('counterfactual_plan_post_treatment_columns', []) or []) if str(x).strip()]
    design = str(path.get('counterfactual_validation_design', '') or '').strip().lower()
    used_estimand = str(path.get('counterfactual_preferred_estimand', '') or '').strip().lower()
    cov_overlap = (len(set(covs) & set(used_covs)) / max(1, len(set(covs)))) if covs else 1.0
    post_overlap = (len(set(post) & set(used_post)) / max(1, len(set(post)))) if post else 1.0
    design_match = 1.0 if (vd and vd == design) or (not vd and not design) else (0.6 if vd and design and vd in design else 0.4)
    estimand_match = 1.0 if (estimand and estimand == used_estimand) or (not estimand and not used_estimand) else (0.6 if estimand and used_estimand and estimand in used_estimand else 0.4)
    score = round(max(0.0, min(1.0, 0.40 * cov_overlap + 0.20 * post_overlap + 0.20 * design_match + 0.20 * estimand_match)), 3)
    label = 'strong' if score >= 0.8 else ('moderate' if score >= 0.55 else 'weak')
    return label, score, {
        'has_plan': True,
        'covariate_overlap': round(cov_overlap, 3),
        'post_treatment_overlap': round(post_overlap, 3),
        'design_match': round(design_match, 3),
        'estimand_match': round(estimand_match, 3),
    }


def _validation_score(path: Dict[str, Any], neg_status: str, max_neg_delta: float) -> Tuple[float, Dict[str, float], List[str]]:
    n_t = _safe_int(path.get("counterfactual_treated_support", 0))
    n_c = _safe_int(path.get("counterfactual_control_support", 0))
    match_q = _safe_float(path.get("counterfactual_control_match_quality", 0.0))
    prop_gap = _safe_float(path.get("counterfactual_propensity_overlap_gap", 1.0), 1.0)
    balance_smd = _safe_float(path.get("counterfactual_balance_smd_mean", 1.0), 1.0)
    overlap_ok = bool(path.get("counterfactual_overlap_ok", False))
    design_strength = str(path.get("counterfactual_design_strength", "low"))
    shared_support = _safe_float(path.get("counterfactual_shared_support_ratio", path.get("counterfactual_strata_support", 0.0)), 0.0)
    effective_support_n = _safe_int(path.get("counterfactual_effective_support_n", 0))
    dominant_control_weight = _safe_float(path.get("counterfactual_dominant_control_weight", 1.0), 1.0)
    placebo_pass = bool(path.get("placebo_pass", True))
    placebo_fail_rate = _safe_float(path.get("placebo_fail_rate", 0.0), 0.0)
    pretrend_ok = bool(path.get("counterfactual_pretrend_ok", False))
    pretrend_testable = bool(path.get("counterfactual_pretrend_testable", False))
    exact_stratum_control = _safe_int(path.get("counterfactual_exact_stratum_control", 0))
    valid_strata = _safe_int(path.get("counterfactual_valid_strata", 0))
    graph_supported = bool(path.get("graph_supported", False))

    effect_for_ci = _safe_float(path.get("counterfactual_dr_did_effect", path.get("counterfactual_did_effect", 0.0)), 0.0)
    ci_status, ci_score, ci_width = _ci_status(
        _safe_float(path.get("counterfactual_bootstrap_ci_low", 0.0), 0.0),
        _safe_float(path.get("counterfactual_bootstrap_ci_high", 0.0), 0.0),
        effect_for_ci,
    )
    agreement_label, agreement_score, agreement_components = _effect_agreement(path)
    plan_coherence_label, plan_coherence_score, plan_coherence_meta = _plan_coherence(path)

    components = {
        "sample_support": min(1.0, (min(n_t, n_c) / 8.0)) if min(n_t, n_c) > 0 else 0.0,
        "match_quality": max(0.0, min(1.0, match_q)),
        "overlap": 1.0 if overlap_ok else max(0.0, 1.0 - min(1.0, prop_gap / 0.35)),
        "balance": 1.0 if balance_smd <= 0.12 else max(0.0, 1.0 - min(1.0, (balance_smd - 0.12) / 0.45)),
        "shared_support": max(0.0, min(1.0, shared_support)),
        "design": 1.0 if design_strength == "high" else (0.68 if design_strength == "medium" else 0.34),
        "placebo": 1.0 if placebo_pass else max(0.0, 1.0 - min(1.0, placebo_fail_rate / 0.7)),
        "pretrend": 1.0 if (pretrend_ok and pretrend_testable) else (0.40 if pretrend_testable else 0.22),
        "ci": ci_score,
        "agreement": agreement_score,
        "strata": min(1.0, 0.30 * max(0, valid_strata) + 0.22 * max(0, exact_stratum_control)),
        "control_concentration": max(0.0, 1.0 - max(0.0, dominant_control_weight - 0.35) / 0.55),
        "structural_support": 0.9 if graph_supported else 0.45,
        "plan_coherence": plan_coherence_score,
    }

    weights = {
        "sample_support": 0.08,
        "match_quality": 0.12,
        "overlap": 0.11,
        "balance": 0.09,
        "shared_support": 0.10,
        "design": 0.07,
        "placebo": 0.10,
        "pretrend": 0.06,
        "ci": 0.08,
        "agreement": 0.10,
        "strata": 0.05,
        "control_concentration": 0.05,
        "structural_support": 0.04,
        "plan_coherence": 0.05,
    }
    score = sum(weights[k] * components[k] for k in weights)
    notes: List[str] = []

    if neg_status == "fail":
        score -= 0.14
        notes.append("negative_control_fail")
    elif neg_status == "warning":
        score -= 0.07
        notes.append("negative_control_warning")
    if plan_coherence_label == "weak":
        score -= 0.04
        notes.append("validation_plan_coherence_weak")
    elif plan_coherence_label == "strong":
        notes.append("validation_plan_coherence_strong")
    if agreement_label == "discordant":
        score -= 0.12
        notes.append("effect_methods_disagree")
    elif agreement_label == "weak":
        score -= 0.04
        notes.append("effect_methods_weak")
    if not overlap_ok and prop_gap > 0.20:
        score -= 0.06
        notes.append("overlap_gap_high")
    if balance_smd > 0.25:
        score -= 0.05
        notes.append("balance_smd_high")
    if dominant_control_weight > 0.60:
        score -= 0.05
        notes.append("control_weight_concentrated")
    if effective_support_n < 4:
        score -= 0.05
        notes.append("effective_support_low")
    if not placebo_pass and placebo_fail_rate >= 0.5:
        score -= 0.05
        notes.append("placebo_fail_rate_high")
    if not pretrend_ok and pretrend_testable:
        score -= 0.04
        notes.append("pretrend_fail")
    if ci_status == "wide":
        notes.append("bootstrap_ci_wide")
    if max_neg_delta >= 0.20:
        notes.append("negative_control_delta_elevated")

    score = round(max(0.0, min(1.0, score)), 3)
    components["ci_width"] = ci_width
    components["agreement_label_score"] = agreement_score
    components["negative_control_max_abs_delta"] = round(abs(max_neg_delta), 3)
    return score, {
        **{k: round(v, 3) for k, v in components.items()},
        "ci_status": ci_status,
        "effect_agreement": agreement_label,
        "effect_components": agreement_components,
        "plan_coherence_label": plan_coherence_label,
        "plan_coherence_meta": plan_coherence_meta,
    }, notes


def _validation_band(score: float, neg_status: str, conf_risk: str) -> str:
    if score >= float(PATH_VALIDATION["bands"]["high_score"]) and neg_status == "pass" and conf_risk in {"low", "medium"}:
        return "high"
    if score >= float(PATH_VALIDATION["bands"]["medium_score"]) and neg_status != "fail" and conf_risk != "high":
        return "medium"
    if score > 0.0:
        return "low"
    return "none"


def enrich_paths_with_validation(
    intent: Dict[str, Any],
    paths: List[Dict[str, Any]],
    event_log_path: str | Path = "historical_action_events.jsonl",
    registry_path: str | Path = "action_registry.yaml",
    path_library_path: str | Path = "dangerous_paths.yaml",
) -> List[Dict[str, Any]]:
    events = _load_jsonl(event_log_path)
    if not paths or not events:
        return list(paths or [])

    registry = load_action_registry(str(registry_path))
    library = load_path_library(str(path_library_path)).get("paths", {}) or {}
    harms = _available_harms(events)
    enriched: List[Dict[str, Any]] = []

    for path in paths:
        out = dict(path)
        path_id = str(out.get("path_id", ""))
        path_spec = dict(library.get(path_id, {}) or {})
        harm = str(out.get("graph_harm") or "")

        # Runtime path validation must stay bounded. If the counterfactual layer
        # already returned the explicit runtime fallback, do not immediately call
        # the expensive matcher again from the validation layer. This keeps the
        # veto gateway responsive even when historical logs are non-trivial or
        # the scientific/offline stack is unavailable. Offline audits can still
        # enable the heavy counterfactual path upstream.
        if bool(out.get("counterfactual_runtime_skipped", False)):
            treated, controls, quality = [], [], {"control_match_quality": 0.0}
        else:
            treated, controls, quality = _build_matched_sets(
                intent,
                events,
                path_spec,
                registry,
                harm,
                path_id=path_id,
                path=out,
            )
        n_t = _safe_int(out.get("counterfactual_treated_support", len(treated)))
        n_c = _safe_int(out.get("counterfactual_control_support", len(controls)))
        match_q = _safe_float(out.get("counterfactual_control_match_quality", quality.get("control_match_quality", 0.0)))

        neg_deltas: List[Tuple[str, float]] = []
        eligible_neg = 0
        for h in harms:
            if not h or h == harm:
                continue
            prevalence = sum(1 for e in events if h in list(e.get("observed_harms", []) or []))
            if prevalence < 2:
                continue
            eligible_neg += 1
            td = _weighted_harm_rate(treated, h)
            cd = _weighted_harm_rate(controls, h)
            neg_deltas.append((h, round(td - cd, 3)))
        max_neg_harm = ""
        max_neg_delta = 0.0
        if neg_deltas:
            max_neg_harm, max_neg_delta = max(neg_deltas, key=lambda x: abs(x[1]))
        neg_status = _negative_control_status(abs(max_neg_delta), n_t, n_c, eligible_neg)

        validation_score, validation_components, validation_notes = _validation_score(out, neg_status, max_neg_delta)
        cf_ident = str(out.get("counterfactual_identification_support", "none"))
        cf_conf_risk = str(out.get("counterfactual_confounding_risk", "unknown"))
        band = _validation_band(validation_score, neg_status, cf_conf_risk)

        if cf_ident == "high" and band == "medium" and validation_score >= float(PATH_VALIDATION["bands"]["promotion_ident_high_score"]) and neg_status != "fail":
            band = "high"
        elif cf_ident == "medium" and band == "low" and validation_score >= float(PATH_VALIDATION["bands"]["promotion_ident_medium_score"]) and neg_status != "fail":
            band = "medium"
        elif cf_ident == "low" and band == "high":
            band = "medium"

        conf_risk = "high"
        if validation_score >= float(PATH_VALIDATION["bands"]["high_score"]) and match_q >= float(PATH_VALIDATION["bands"]["promotion_match_high"]) and neg_status == "pass" and cf_conf_risk != "high":
            conf_risk = "low"
        elif validation_score >= max(0.48, float(PATH_VALIDATION["bands"]["medium_score"]) - 0.02) and match_q >= float(PATH_VALIDATION["bands"]["promotion_match_medium"]) and neg_status != "fail":
            conf_risk = "medium"
        out["negative_control_status"] = neg_status
        out["negative_control_max_harm"] = max_neg_harm or None
        out["negative_control_max_delta"] = round(max_neg_delta, 3)
        out["identification_support"] = band
        out["identification_support_score"] = validation_score
        out["validation_score"] = validation_score
        out["validation_components"] = validation_components
        out["validation_notes"] = validation_notes
        out["sensitivity_confounding_risk"] = conf_risk
        out["evidence_profile"] = {
            "structural": "high" if bool(out.get("hard_block_hits")) else "medium" if bool(out.get("graph_supported", False)) else "low",
            "empirical": str(out.get("empirical_evidence_strength", "none")),
            "counterfactual": str(out.get("counterfactual_evidence_strength", "none")),
            "identification": band,
            "negative_control": neg_status,
            "validation_score": validation_score,
            "effect_agreement": validation_components.get("effect_agreement", "unknown"),
            "ci_status": validation_components.get("ci_status", "unknown"),
        }
        enriched.append(out)

    return sorted(
        enriched,
        key=lambda x: (
            {"high": 0, "medium": 1, "low": 2, "none": 3}.get(str(x.get("identification_support", "none")), 4),
            -_safe_float(x.get("validation_score", 0.0)),
            -_safe_float(x.get("risk_score", 0.0)),
            x.get("path_id", ""),
        ),
    )
