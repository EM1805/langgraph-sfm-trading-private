from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional

from ._utils import CONFIDENCE_RANK


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


def _norm_conf(value: Any) -> str:
    raw = str(value or '').strip().lower()
    return raw if raw in CONFIDENCE_RANK else 'unknown'


@dataclass
class AdjustmentRecommendation:
    action: str
    target: str
    adjust_for: List[str]
    avoid: List[str]
    confidence: str
    action_known: bool
    target_known: bool
    source: str
    notes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def recommend_adjustments(
    *,
    dag: Any,
    action: str,
    target: Optional[str],
    candidate_covariates: Optional[Iterable[str]] = None,
) -> AdjustmentRecommendation:
    action_raw = str(action or '').strip()
    target_raw = str(target or '').strip()
    candidate_covariates = [str(c).strip() for c in (candidate_covariates or []) if str(c).strip()]
    if dag is None:
        return AdjustmentRecommendation(
            action=action_raw,
            target=target_raw,
            adjust_for=[],
            avoid=[],
            confidence='unknown',
            action_known=False,
            target_known=False,
            source='none',
            notes=['NO_DAG'],
        )

    ann = dag.l32_annotation(action_raw, target_raw)
    adjust_for = _as_list(ann.get('dag_adjustment_set', ''))
    avoid = _as_list(ann.get('dag_forbidden_adjustments', ''))
    mediators = set(_as_list(ann.get('dag_mediators', '')))
    colliders = set(_as_list(ann.get('dag_colliders', '')))
    negative_controls = set(_as_list(ann.get('dag_negative_controls', '')))
    confidence = _norm_conf(ann.get('dag_adjustment_confidence', 'unknown'))
    path_id = str(ann.get('dag_path_id', '')).strip()
    source = 'path_hint' if path_id and (adjust_for or avoid) else ('recommended_adjustments' if adjust_for or avoid else 'ancestor_fallback')
    notes: List[str] = []

    if not adjust_for and candidate_covariates:
        # If the DAG could not propose explicit covariates, keep only candidate covariates that are known DAG nodes
        # and marked adjustable when possible.
        blocked = set(avoid).union(mediators).union(colliders).union(negative_controls)
        for c in candidate_covariates:
            if c in blocked:
                continue
            node = dag.get_node(c)
            if node is not None and bool(getattr(node, 'adjustable', False)) and c not in adjust_for:
                adjust_for.append(c)
        if adjust_for:
            source = 'candidate_covariates'
            confidence = max(confidence, 'low', key=lambda x: CONFIDENCE_RANK.get(x, 0))
            notes.append('CANDIDATE_COVARIATE_FALLBACK')

    if path_id:
        notes.append('PATH_HINT_USED:' + path_id)
    if candidate_covariates and (avoid or mediators or colliders or negative_controls):
        blocked = set(avoid).union(mediators).union(colliders).union(negative_controls)
        violating = sorted({c for c in candidate_covariates if c in blocked})
        if violating:
            notes.append('FORBIDDEN_IN_INPUT:' + '|'.join(violating))
    if mediators:
        notes.append('MEDIATORS_EXCLUDED')
    if colliders:
        notes.append('COLLIDERS_EXCLUDED')
    if negative_controls:
        notes.append('NEGATIVE_CONTROLS_RESERVED')
    if not adjust_for:
        notes.append('NO_ADJUSTMENT_SET')
    if avoid:
        notes.append('AVOID_SET_PRESENT')

    return AdjustmentRecommendation(
        action=str(ann.get('dag_action_resolved', action_raw)),
        target=str(ann.get('dag_target_resolved', target_raw)),
        adjust_for=adjust_for,
        avoid=avoid,
        confidence=confidence,
        action_known=bool(int(ann.get('dag_action_known', 0) or 0)),
        target_known=bool(int(ann.get('dag_target_known', 0) or 0)),
        source=source,
        notes=notes,
    )


__all__ = ['AdjustmentRecommendation', 'recommend_adjustments']
