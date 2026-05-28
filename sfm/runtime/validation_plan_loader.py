from __future__ import annotations
import csv
from pathlib import Path
from typing import Any, Dict, List

_HARM_ALIASES = {
    "harm_operational_failure": ["operational_failure", "service_degradation", "outage"],
    "harm_leakage": ["external_data_leakage", "leakage", "data_exfiltration"],
    "harm_data_loss": ["destructive_mutation", "data_loss"],
    "harm_policy_bypass": ["policy_bypass", "approval_bypass"],
    "harm_access_abuse": ["privilege_escalation", "unauthorized_access"],
}

def _split_pipe(value: Any) -> List[str]:
    if value is None:
        return []
    s = str(value).strip()
    if not s:
        return []
    return [x.strip() for x in s.split('|') if x.strip()]

def _safe_read_csv(path: str | Path) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open('r', encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))

def _harm_tokens(graph_harm: str, path_id: str) -> List[str]:
    vals: List[str] = []
    for x in [str(graph_harm or '').strip(), str(path_id or '').strip()]:
        if x and x not in vals:
            vals.append(x)
    for x in _HARM_ALIASES.get(str(graph_harm or '').strip(), []):
        if x not in vals:
            vals.append(x)
    return vals

def _score_row(row: Dict[str, Any], path_id: str, graph_harm: str) -> float:
    score = 0.0
    row_path = str(row.get('path_id_candidate', row.get('path_id', ''))).strip()
    row_harm = str(row.get('harm_node_candidate', row.get('graph_harm', ''))).strip()
    if row_path and row_path == path_id:
        score += 1.5
    if row_harm and graph_harm and row_harm == graph_harm:
        score += 1.0
    joined = ' | '.join(str(row.get(k,'')) for k in [
        'path_id_candidate','harm_node_candidate','candidate_action_family','validation_plan_rationale',
        'treatment_col','treatment_role','outcome_col','preferred_estimand','edge_family','source_role'
    ]).lower()
    for tok in _harm_tokens(graph_harm, path_id):
        if tok and tok.lower() in joined:
            score += 0.35
    if graph_harm and str(row.get('outcome_col','')).strip() == graph_harm:
        score += 0.4
    track = str(row.get('discovery_track','')).strip()
    if track == 'high_confidence':
        score += 0.15
    elif track == 'exploratory':
        score += 0.05
    return score

def load_validation_plan_for_path(path_id: str, graph_harm: str = '', validation_plan_path: str | Path = 'out/validation_plan_level2.csv') -> Dict[str, Any]:
    rows = _safe_read_csv(validation_plan_path)
    if not rows:
        return {}
    scored = sorted(((_score_row(r, path_id, graph_harm), r) for r in rows), key=lambda x: x[0], reverse=True)
    if not scored or scored[0][0] <= 0.34:
        return {}
    s, best = scored[0]
    return {
        'plan_source': str(validation_plan_path),
        'plan_score': round(s, 3),
        'matched_by': 'path_or_harm_semantic_match',
        'recommended_validation': str(best.get('recommended_validation', '')).strip(),
        'validation_priority_tier': str(best.get('validation_priority_tier', '')).strip(),
        'candidate_stratum': _split_pipe(best.get('candidate_stratum', '')),
        'candidate_stratum_confidence': str(best.get('candidate_stratum_confidence', '')).strip(),
        'suggested_adjustment_set': _split_pipe(best.get('suggested_adjustment_set', '')),
        'forbidden_adjustment_set': _split_pipe(best.get('forbidden_adjustment_set', '')),
        'suggested_negative_control': str(best.get('suggested_negative_control', '')).strip(),
        'validation_plan_rationale': str(best.get('validation_plan_rationale', '')).strip(),
        'matched_plan_row_path_id': str(best.get('path_id_candidate', best.get('path_id', ''))).strip(),
        'matched_plan_row_harm': str(best.get('harm_node_candidate', best.get('graph_harm', ''))).strip(),
        'treatment_col': str(best.get('treatment_col', best.get('source', ''))).strip(),
        'treatment_role': str(best.get('treatment_role', best.get('source_role', ''))).strip(),
        'outcome_col': str(best.get('outcome_col', '')).strip(),
        'outcome_role': str(best.get('outcome_role', best.get('target_role', ''))).strip(),
        'preferred_estimand': str(best.get('preferred_estimand', best.get('validation_design', ''))).strip(),
        'validation_design': str(best.get('validation_design', '')).strip(),
        'candidate_covariates': _split_pipe(best.get('candidate_covariates', '')),
        'post_treatment_columns': _split_pipe(best.get('post_treatment_columns', '')),
        'edge_family': str(best.get('edge_family', '')).strip(),
        'source_role': str(best.get('source_role', '')).strip(),
    }

def merge_validation_plan_into_path(path: Dict[str, Any], validation_plan_path: str | Path = 'out/validation_plan_level2.csv') -> Dict[str, Any]:
    out = dict(path)
    plan = load_validation_plan_for_path(str(path.get('path_id', '')), str(path.get('graph_harm', '')), validation_plan_path)
    out['validation_plan'] = plan
    if plan:
        out['plan_recommended_validation'] = plan.get('recommended_validation', '')
        out['plan_candidate_stratum'] = plan.get('candidate_stratum', [])
        out['plan_suggested_adjustment_set'] = plan.get('suggested_adjustment_set', [])
        out['plan_forbidden_adjustment_set'] = plan.get('forbidden_adjustment_set', [])
        out['plan_suggested_negative_control'] = plan.get('suggested_negative_control', '')
        out['plan_candidate_stratum_confidence'] = plan.get('candidate_stratum_confidence', '')
        out['plan_matched_by'] = plan.get('matched_by', '')
        out['plan_treatment_col'] = plan.get('treatment_col', '')
        out['plan_treatment_role'] = plan.get('treatment_role', '')
        out['plan_outcome_col'] = plan.get('outcome_col', '')
        out['plan_preferred_estimand'] = plan.get('preferred_estimand', '')
        out['plan_validation_design'] = plan.get('validation_design', '')
        out['plan_candidate_covariates'] = plan.get('candidate_covariates', [])
        out['plan_post_treatment_columns'] = plan.get('post_treatment_columns', [])
        out['plan_outcome_role'] = plan.get('outcome_role', '')
        out['plan_edge_family'] = plan.get('edge_family', '')
        out['plan_source_role'] = plan.get('source_role', '')
    return out
