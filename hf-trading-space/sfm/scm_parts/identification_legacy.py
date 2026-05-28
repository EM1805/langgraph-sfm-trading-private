# DECLASSIFIED LEGACY MODULE
# Retained only for historical estimation scoring. Canonical SCM authority is id_algorithm + do_contract.

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional

from ._utils import CONFIDENCE_RANK, STRENGTH_RANK


@dataclass
class IdentificationResult:
    strategy: str
    identifiable: bool
    identification_strength: str
    identification_score: float
    assumptions: List[str]
    failed_assumptions: List[str]
    notes: List[str]
    adjustment_set: List[str]
    forbidden_adjustments: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip() for v in value if str(v).strip()]
    raw = str(value).strip()
    if not raw:
        return []
    sep = '|' if '|' in raw else ','
    return [part.strip() for part in raw.split(sep) if part.strip()]


def _norm_grade(value: Any) -> str:
    v = str(value or '').strip().lower()
    return v if v in {'fail', 'weak', 'moderate', 'strong'} else 'unknown'


def _norm_conf(value: Any) -> str:
    raw = str(value or '').strip().lower()
    return raw if raw in CONFIDENCE_RANK else 'unknown'


def _strength_label(rank: int) -> str:
    rank = max(0, min(3, int(rank)))
    for k, v in STRENGTH_RANK.items():
        if v == rank:
            return k
    return 'none'


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
        return out
    except (TypeError, ValueError, OverflowError):
        return default


def _bool_score(value: Optional[bool], true_score: float = 1.0, false_score: float = 0.0, none_score: float = 0.5) -> float:
    if value is True:
        return float(true_score)
    if value is False:
        return float(false_score)
    return float(none_score)


def _grade_score(grade: str) -> float:
    g = _norm_grade(grade)
    return {"fail": 0.10, "weak": 0.40, "moderate": 0.68, "strong": 0.88}.get(g, 0.35)


def _confidence_score(conf: str) -> float:
    c = _norm_conf(conf)
    return {"unknown": 0.35, "low": 0.45, "medium": 0.72, "high": 0.90}.get(c, 0.35)


def _choose_strategy(*, has_controls: bool, pretrend_available: bool, propensity_available: bool, overlap_ok: Optional[bool]) -> str:
    if not has_controls:
        return 'not_identified'
    if propensity_available and overlap_ok is True:
        return 'backdoor_matching'
    if propensity_available:
        return 'backdoor_weighting'
    if pretrend_available:
        return 'quasi_prepost'
    return 'matched_observational'


def build_identification(
    *,
    has_controls: bool,
    propensity_available: bool,
    overlap_ok: Optional[bool],
    balance_ok: Optional[bool],
    sample_size_ok: Optional[bool],
    leakage_ok: Optional[bool],
    drift_ok: Optional[bool],
    temporal_order_ok: Optional[bool],
    diagnostic_grade: str,
    sensitivity_level: str,
    pretrend_available: bool = False,
    pretrend_pass: Optional[bool] = None,
    dag_adjustment_set: Optional[Iterable[str]] = None,
    dag_forbidden_adjustments: Optional[Iterable[str]] = None,
    dag_risk_paths: str = '',
    dag_covariate_violation_flag: Any = 0,
    dag_action_known: Any = 0,
    dag_target_known: Any = 0,
    dag_action_type: str = 'unknown',
    dag_target_type: str = 'unknown',
    dag_action_time_role: str = 'unknown',
    dag_target_time_role: str = 'unknown',
    dag_adjustment_confidence: str = 'unknown',
    dag_direct_edge_confidence: str = 'unknown',
    dag_path_confidence: str = 'unknown',
    dag_mediators: Optional[Iterable[str]] = None,
    dag_colliders: Optional[Iterable[str]] = None,
    dag_negative_controls: Optional[Iterable[str]] = None,
    dag_path_id: str = '',
    dag_treatment_node: str = '',
    dag_outcome_node: str = '',
    validation_score: Any = None,
    match_quality: Any = None,
    shared_support_ratio: Any = None,
    overlap_gap_value: Any = None,
    placebo_pvalue: Any = None,
    effect_sign_stable: Any = None,
    dominant_control_weight: Any = None,
    method_agreement: Any = None,
) -> IdentificationResult:
    strategy = _choose_strategy(
        has_controls=has_controls,
        pretrend_available=pretrend_available,
        propensity_available=propensity_available,
        overlap_ok=overlap_ok,
    )

    assumptions: List[str] = [
        'temporal_order_valid',
        'sufficient_sample_size',
        'no_obvious_leakage',
        'regime_stability_plausible',
        'dag_action_target_mapped',
        'dag_structural_support',
    ]
    if has_controls:
        assumptions.append('measured_confounding_adjustable')
    if str(dag_path_id or '').strip():
        assumptions.append('dag_path_design_specified')
    if propensity_available:
        assumptions.append('positivity_overlap')
    if pretrend_available:
        assumptions.append('parallel_pretrends_plausible')

    try:
        action_known = int(float(dag_action_known)) != 0
    except (TypeError, ValueError, OverflowError):
        action_known = bool(dag_action_known)
    try:
        target_known = int(float(dag_target_known)) != 0
    except (TypeError, ValueError, OverflowError):
        target_known = bool(dag_target_known)

    adjustment_set = _as_list(dag_adjustment_set)
    forbidden_adjustments = _as_list(dag_forbidden_adjustments)
    mediators = _as_list(dag_mediators)
    colliders = _as_list(dag_colliders)
    negative_controls = _as_list(dag_negative_controls)
    dgrade = _norm_grade(diagnostic_grade)
    adj_conf = _norm_conf(dag_adjustment_confidence)
    direct_conf = _norm_conf(dag_direct_edge_confidence)
    path_conf = _norm_conf(dag_path_confidence)

    dag_mapping_ok = action_known and target_known
    dag_structural_support_ok = CONFIDENCE_RANK.get(path_conf, 0) >= 1 or CONFIDENCE_RANK.get(direct_conf, 0) >= 1
    dag_treatment_ok = (not str(dag_treatment_node or '').strip()) or (action_known and (str(dag_treatment_node).strip() == str(dag_adjustment_set or dag_treatment_node).strip() or True))
    dag_outcome_ok = (not str(dag_outcome_node or '').strip()) or target_known
    dag_action_type_ok = str(dag_action_type or '').strip().lower() in {'treatment_candidate', 'exposure', 'action'}
    dag_target_type_ok = str(dag_target_type or '').strip().lower() in {'outcome', 'risk_outcome'}
    dag_time_ok = str(dag_action_time_role or '').strip().lower() != 'post_treatment'

    checks = {
        'temporal_order_valid': temporal_order_ok,
        'sufficient_sample_size': sample_size_ok,
        'no_obvious_leakage': leakage_ok,
        'regime_stability_plausible': drift_ok,
        'measured_confounding_adjustable': has_controls if has_controls else False,
        'positivity_overlap': overlap_ok if propensity_available else None,
        'parallel_pretrends_plausible': pretrend_pass if pretrend_available else None,
        'covariate_balance_acceptable': balance_ok,
        'dag_action_target_mapped': dag_mapping_ok,
        'dag_structural_support': dag_structural_support_ok,
        'dag_action_type_valid': dag_action_type_ok if action_known else None,
        'dag_target_type_valid': dag_target_type_ok if target_known else None,
        'dag_action_not_post_treatment': dag_time_ok if action_known else None,
        'dag_path_design_specified': bool(str(dag_path_id or '').strip()) if str(dag_path_id or '').strip() else None,
        'dag_treatment_node_mapped': dag_treatment_ok if str(dag_treatment_node or '').strip() else None,
        'dag_outcome_node_mapped': dag_outcome_ok if str(dag_outcome_node or '').strip() else None,
    }
    failed = sorted({k for k, v in checks.items() if v is False})
    notes: List[str] = []

    if adjustment_set:
        notes.append('DAG_ADJUSTMENT_SET')
    if str(dag_path_id or '').strip():
        notes.append('DAG_PATH_HINT_USED:' + str(dag_path_id).strip())
    if mediators:
        notes.append('DAG_MEDIATORS_MARKED')
    if colliders:
        notes.append('DAG_COLLIDERS_MARKED')
    if negative_controls:
        notes.append('DAG_NEGATIVE_CONTROLS_MARKED')
    if forbidden_adjustments:
        notes.append('DAG_FORBIDDEN_ADJUSTMENTS')
    if adj_conf != 'unknown':
        notes.append(f'DAG_ADJUSTMENT_{adj_conf.upper()}')
    if path_conf != 'unknown':
        notes.append(f'DAG_PATH_{path_conf.upper()}')
    if direct_conf not in {'unknown', 'low'}:
        notes.append(f'DAG_DIRECT_{direct_conf.upper()}')
    if str(dag_risk_paths or '').strip():
        failed.append('dag_risk_path_present')
        notes.append('DAG_RISK_PATH')
    try:
        if int(float(dag_covariate_violation_flag)) != 0:
            failed.append('dag_covariate_violation')
            notes.append('DAG_COVARIATE_VIOLATION')
    except (TypeError, ValueError, OverflowError):
        pass
    if not dag_mapping_ok:
        notes.append('DAG_MAPPING_WEAK')
    if not dag_structural_support_ok:
        notes.append('DAG_PATH_UNKNOWN')
    if action_known and not dag_action_type_ok:
        notes.append('DAG_ACTION_TYPE_WEAK')
    if target_known and not dag_target_type_ok:
        notes.append('DAG_TARGET_TYPE_WEAK')
    if action_known and not dag_time_ok:
        notes.append('DAG_POST_TREATMENT_ACTION')

    validation_score_f = _safe_float(validation_score)
    match_quality_f = _safe_float(match_quality)
    shared_support_ratio_f = _safe_float(shared_support_ratio)
    overlap_gap_f = _safe_float(overlap_gap_value)
    placebo_pvalue_f = _safe_float(placebo_pvalue)
    dominant_control_weight_f = _safe_float(dominant_control_weight)
    method_agreement_f = _safe_float(method_agreement)

    component_scores: List[float] = []
    component_scores.append(0.16 * _bool_score(sample_size_ok, 1.0, 0.0, 0.35))
    component_scores.append(0.14 * _bool_score(temporal_order_ok, 1.0, 0.0, 0.35))
    component_scores.append(0.10 * _bool_score(balance_ok, 1.0, 0.0, 0.45))
    component_scores.append(0.12 * _bool_score(overlap_ok, 1.0, 0.0, 0.45))
    component_scores.append(0.08 * _bool_score(pretrend_pass if pretrend_available else None, 1.0, 0.0, 0.55))
    component_scores.append(0.10 * _grade_score(diagnostic_grade))
    component_scores.append(0.09 * _confidence_score(adj_conf))
    component_scores.append(0.07 * _confidence_score(path_conf))
    component_scores.append(0.05 * _confidence_score(direct_conf))
    component_scores.append(0.09 * (1.0 if adjustment_set else 0.0))

    empirical_score = 0.0
    empirical_weight = 0.0
    if match_quality_f == match_quality_f:
        empirical_score += 0.20 * max(0.0, min(1.0, match_quality_f))
        empirical_weight += 0.20
    if shared_support_ratio_f == shared_support_ratio_f:
        empirical_score += 0.20 * max(0.0, min(1.0, shared_support_ratio_f))
        empirical_weight += 0.20
    if overlap_gap_f == overlap_gap_f:
        empirical_score += 0.15 * max(0.0, min(1.0, 1.0 - min(1.0, overlap_gap_f)))
        empirical_weight += 0.15
    if placebo_pvalue_f == placebo_pvalue_f:
        empirical_score += 0.15 * max(0.0, min(1.0, placebo_pvalue_f))
        empirical_weight += 0.15
    if method_agreement_f == method_agreement_f:
        empirical_score += 0.15 * max(0.0, min(1.0, method_agreement_f))
        empirical_weight += 0.15
    if dominant_control_weight_f == dominant_control_weight_f:
        empirical_score += 0.10 * max(0.0, min(1.0, 1.0 - dominant_control_weight_f))
        empirical_weight += 0.10
    if effect_sign_stable not in (None, ''):
        try:
            empirical_score += 0.05 * (1.0 if int(float(effect_sign_stable)) != 0 else 0.0)
            empirical_weight += 0.05
        except (TypeError, ValueError, OverflowError):
            pass
    if validation_score_f == validation_score_f:
        empirical_score += 0.25 * max(0.0, min(1.0, validation_score_f))
        empirical_weight += 0.25
    empirical_component = (empirical_score / empirical_weight) if empirical_weight > 0 else float('nan')
    if empirical_component == empirical_component:
        component_scores.append(0.20 * empirical_component)

    identification_score = max(0.0, min(1.0, float(sum(component_scores))))

    hard_fail = False
    if strategy == 'not_identified':
        hard_fail = True
        identifiable = False
        strength = 'none'
        identification_score = min(identification_score, 0.08)
    else:
        if action_known and not dag_time_ok:
            hard_fail = True
            identifiable = False
            identification_score = min(identification_score, 0.05)
        if not has_controls:
            identification_score = min(identification_score, 0.18)
        if forbidden_adjustments and int(bool(dag_covariate_violation_flag)):
            identification_score = min(identification_score, 0.34)
        if (mediators or colliders) and not forbidden_adjustments:
            identification_score = min(identification_score, 0.42)
        if str(dag_path_id or '').strip() and not adjustment_set:
            identification_score = min(identification_score, 0.44)
        if sensitivity_level == 'high':
            identification_score = min(identification_score, 0.56)
        if not dag_mapping_ok:
            identification_score = min(identification_score, 0.46)
        if not dag_structural_support_ok:
            identification_score = min(identification_score, 0.44)
        if overlap_ok is False or balance_ok is False:
            identification_score = min(identification_score, 0.52)
        if pretrend_available and pretrend_pass is False:
            identification_score = min(identification_score, 0.58)
        if leakage_ok is False or drift_ok is False:
            identification_score = min(identification_score, 0.48)

        if hard_fail or identification_score < 0.22:
            strength = 'none'
            identifiable = False
        elif identification_score < 0.45:
            strength = 'weak'
            identifiable = True
        elif identification_score < 0.72:
            strength = 'moderate'
            identifiable = True
        else:
            strength = 'strong'
            identifiable = True
    if not has_controls:
        notes.append('NO_MATCHED_CONTROLS')
    if balance_ok is False:
        notes.append('BALANCE_NOT_ACCEPTABLE')
    if overlap_ok is False:
        notes.append('WEAK_OVERLAP')
    if leakage_ok is False:
        notes.append('LEAKAGE_RISK')
    if drift_ok is False:
        notes.append('DRIFT_RISK')
    if validation_score_f == validation_score_f:
        notes.append('IDENT_VALIDATION_SCORE_{:.3f}'.format(validation_score_f))
    notes.append('IDENT_SCORE_{:.3f}'.format(identification_score))
    if match_quality_f == match_quality_f:
        notes.append('IDENT_MATCH_QUALITY_{:.3f}'.format(match_quality_f))
    if shared_support_ratio_f == shared_support_ratio_f:
        notes.append('IDENT_SHARED_SUPPORT_{:.3f}'.format(shared_support_ratio_f))
    if method_agreement_f == method_agreement_f:
        notes.append('IDENT_METHOD_AGREEMENT_{:.3f}'.format(method_agreement_f))
    if placebo_pvalue_f == placebo_pvalue_f:
        notes.append('IDENT_PLACEBO_P_{:.3f}'.format(placebo_pvalue_f))
    if dominant_control_weight_f == dominant_control_weight_f and dominant_control_weight_f > 0.65:
        notes.append('IDENT_CONTROL_WEIGHT_CONCENTRATED')
    if strength == 'strong':
        notes.append('IDENTIFICATION_EVIDENCE_CONSISTENT')
    elif strength == 'moderate':
        notes.append('IDENTIFICATION_EVIDENCE_PLAUSIBLE')
    elif strength == 'weak':
        notes.append('IDENTIFICATION_EVIDENCE_FRAGILE')

    return IdentificationResult(
        strategy=strategy,
        identifiable=bool(identifiable),
        identification_strength=strength,
        identification_score=float(identification_score),
        assumptions=assumptions,
        failed_assumptions=sorted(set(failed)),
        notes=notes,
        adjustment_set=adjustment_set,
        forbidden_adjustments=forbidden_adjustments,
    )


__all__ = ['IdentificationResult', 'build_identification']
