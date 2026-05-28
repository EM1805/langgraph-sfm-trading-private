from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


_RISK_PENALTY = {
    "none": 0.0,
    "low": 0.05,
    "medium": 0.18,
    "moderate": 0.18,
    "high": 0.38,
    "critical": 0.65,
    "unknown": 0.12,
}
_CONFIDENCE_BONUS = {
    "high": 0.05,
    "medium": 0.02,
    "low": -0.03,
    "unknown": 0.0,
}


def _clean_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _clean_lower(value: Any, default: str = "unknown") -> str:
    return _clean_str(value, default).lower()


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray, Mapping)):
        return list(value)
    return [value]


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _clip01(value: Optional[float], default: float = 0.5) -> float:
    if value is None:
        return default
    return max(0.0, min(1.0, float(value)))


def _first(payload: Mapping[str, Any], keys: Sequence[str], default: Any = "") -> Any:
    for key in keys:
        if key in payload and payload.get(key) not in (None, ""):
            return payload.get(key)
    return default


def _action_name(option: Any) -> str:
    if isinstance(option, Mapping):
        return _clean_str(
            option.get("action")
            or option.get("action_name")
            or option.get("candidate_action")
            or option.get("selected_action")
            or option.get("name")
        )
    return _clean_str(option)


def _normalize_option(option: Any) -> Dict[str, Any]:
    if isinstance(option, Mapping):
        out = dict(option)
    else:
        out = {"action": _action_name(option)}
    name = _action_name(out)
    if name:
        out.setdefault("action", name)
        out.setdefault("action_name", name)
    return out


def _merge_scores_into_options(options: List[Dict[str, Any]], raw: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Support compact score dictionaries such as action_scores={name: 0.8}."""
    action_scores = _as_dict(raw.get("action_scores"))
    risk_scores = _as_dict(raw.get("risk_scores"))
    harm_scores = _as_dict(raw.get("harm_scores") or raw.get("harm_probabilities"))
    effect_scores = _as_dict(raw.get("effect_scores") or raw.get("effect_estimates"))
    for option in options:
        name = _action_name(option)
        if not name:
            continue
        if name in action_scores and "expected_success" not in option:
            option["expected_success"] = action_scores[name]
        if name in risk_scores and "risk" not in option and "risk_level" not in option:
            option["risk"] = risk_scores[name]
        if name in harm_scores and "harm_probability" not in option:
            option["harm_probability"] = harm_scores[name]
        if name in effect_scores and "effect_estimate" not in option:
            option["effect_estimate"] = effect_scores[name]
    return options


@dataclass
class CounterfactualQuery:
    """Stable request for comparing candidate actions.

    This adapter is intentionally lightweight. It does not claim to simulate a
    full Pearl structural counterfactual. It ranks action alternatives using
    supplied expected outcome/risk/effect evidence and produces an auditable
    recommendation for the Decision Gate.
    """

    current_action: str = ""
    candidate_actions: List[Dict[str, Any]] = field(default_factory=list)
    outcome: str = "task_success"
    protected_outcome: str = "user_or_system_harm"
    min_margin: float = 0.05
    risk_weight: float = 1.0
    query_id: str = ""
    source: str = "counterfactual_adapter"
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Any) -> "CounterfactualQuery":
        raw = _as_dict(payload)
        nested = _as_dict(raw.get("counterfactual_query"))
        if nested:
            merged = dict(raw)
            merged.update(nested)
            raw = merged

        raw_options = (
            raw.get("candidate_actions")
            or raw.get("action_options")
            or raw.get("alternatives")
            or raw.get("options")
            or []
        )
        options = [_normalize_option(item) for item in _as_list(raw_options)]
        options = [item for item in options if _action_name(item)]
        options = _merge_scores_into_options(options, raw)

        current = _clean_str(_first(raw, ["current_action", "selected_action", "action_name", "candidate_action", "action"]))
        if current and not any(_action_name(item) == current for item in options):
            options.append(_normalize_option({"action": current}))

        return cls(
            current_action=current,
            candidate_actions=options,
            outcome=_clean_str(_first(raw, ["outcome", "target_outcome", "intended_outcome"], "task_success")),
            protected_outcome=_clean_str(_first(raw, ["protected_outcome", "harm_outcome"], "user_or_system_harm")),
            min_margin=_safe_float(_first(raw, ["min_margin", "decision_margin"], 0.05), 0.05) or 0.05,
            risk_weight=_safe_float(_first(raw, ["risk_weight"], 1.0), 1.0) or 1.0,
            query_id=_clean_str(_first(raw, ["query_id", "request_id"])),
            source=_clean_str(_first(raw, ["source"], "counterfactual_adapter")),
            raw=dict(raw),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CounterfactualResult:
    compared: bool = False
    comparison_status: str = "not_compared"
    current_action: str = ""
    recommended_action: str = ""
    current_score: Optional[float] = None
    recommended_score: Optional[float] = None
    score_margin: Optional[float] = None
    outcome: str = "task_success"
    protected_outcome: str = "user_or_system_harm"
    alternatives_count: int = 0
    action_rankings: List[Dict[str, Any]] = field(default_factory=list)
    simulator_used: str = "none"
    authority_status: str = "diagnostic_only"
    reason: str = ""
    reason_codes: List[str] = field(default_factory=list)
    raw_backend_result: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _option_score(option: Mapping[str, Any], query: CounterfactualQuery) -> Dict[str, Any]:
    name = _action_name(option)
    expected = _safe_float(_first(option, ["expected_success", "success_probability", "p_success"], None))
    effect = _safe_float(_first(option, ["effect_estimate", "effect", "ate", "expected_effect"], None))
    harm_prob = _safe_float(_first(option, ["harm_probability", "expected_harm", "p_harm"], None), 0.0) or 0.0
    risk = _clean_lower(_first(option, ["risk", "risk_level"], "unknown"), "unknown")
    confidence = _clean_lower(_first(option, ["confidence", "support", "evidence"], "unknown"), "unknown")

    # If only an effect estimate is supplied, treat 0.5 as the neutral baseline.
    if expected is None and effect is not None:
        expected = 0.5 + effect
    expected = _clip01(expected, default=0.5)
    harm_prob = _clip01(harm_prob, default=0.0)

    risk_penalty = _RISK_PENALTY.get(risk, _RISK_PENALTY["unknown"]) * max(0.0, query.risk_weight)
    confidence_bonus = _CONFIDENCE_BONUS.get(confidence, 0.0)
    score = expected - risk_penalty - harm_prob + confidence_bonus

    return {
        "action": name,
        "score": round(float(score), 6),
        "expected_success": round(float(expected), 6),
        "risk": risk,
        "harm_probability": round(float(harm_prob), 6),
        "effect_estimate": effect,
        "confidence": confidence,
        "risk_penalty": round(float(risk_penalty), 6),
        "confidence_bonus": round(float(confidence_bonus), 6),
        "raw_option": dict(option),
    }


class CounterfactualEngine:
    """Lightweight action-alternative comparator for Amantia.

    Heavy path counterfactual modules can remain offline. This facade gives the
    online Decision Gate a stable, stdlib-only way to compare safe alternatives
    when the caller supplies expected outcome/risk evidence.
    """

    def compare(self, payload: Any) -> CounterfactualResult:
        query = payload if isinstance(payload, CounterfactualQuery) else CounterfactualQuery.from_dict(payload)
        if len(query.candidate_actions) < 2:
            return CounterfactualResult(
                compared=False,
                comparison_status="insufficient_alternatives",
                current_action=query.current_action,
                recommended_action=query.current_action,
                outcome=query.outcome,
                protected_outcome=query.protected_outcome,
                alternatives_count=len(query.candidate_actions),
                simulator_used="score_based_policy_simulator",
                reason="At least two candidate actions are required for counterfactual comparison.",
                reason_codes=["COUNTERFACTUAL_INSUFFICIENT_ALTERNATIVES"],
                raw_backend_result=query.to_dict(),
            )

        rankings = [_option_score(option, query) for option in query.candidate_actions]
        rankings.sort(key=lambda item: item["score"], reverse=True)
        recommended = rankings[0]

        current = query.current_action or recommended["action"]
        current_rows = [item for item in rankings if item["action"] == current]
        current_row = current_rows[0] if current_rows else {"action": current, "score": None}
        current_score = current_row.get("score")
        recommended_score = recommended.get("score")
        margin = None
        if current_score is not None and recommended_score is not None:
            margin = float(recommended_score) - float(current_score)

        if recommended["action"] == current:
            status = "current_action_preferred"
            reason = "The current action is the top-ranked alternative under the supplied outcome/risk evidence."
            codes = ["COUNTERFACTUAL_CURRENT_ACTION_PREFERRED"]
        elif margin is not None and margin < query.min_margin:
            status = "alternatives_near_tie"
            reason = "A different action scored slightly higher, but the margin is below the configured decision threshold."
            codes = ["COUNTERFACTUAL_NEAR_TIE"]
        else:
            status = "alternative_recommended"
            reason = "A different action is expected to perform better after risk-adjusted comparison."
            codes = ["COUNTERFACTUAL_ALTERNATIVE_RECOMMENDED"]

        return CounterfactualResult(
            compared=True,
            comparison_status=status,
            current_action=current,
            recommended_action=recommended["action"],
            current_score=current_score,
            recommended_score=recommended_score,
            score_margin=round(float(margin), 6) if margin is not None else None,
            outcome=query.outcome,
            protected_outcome=query.protected_outcome,
            alternatives_count=len(rankings),
            action_rankings=rankings,
            simulator_used="score_based_policy_simulator",
            authority_status="diagnostic_only",
            reason=reason,
            reason_codes=codes,
            raw_backend_result=query.to_dict(),
        )

    def compare_many(self, payloads: Iterable[Any]) -> List[CounterfactualResult]:
        return [self.compare(payload) for payload in payloads]


# Alias for natural API naming.
CounterfactualComparisonEngine = CounterfactualEngine


def compare_actions(payload: Any) -> Dict[str, Any]:
    return CounterfactualEngine().compare(payload).to_dict()


def compare_many(payloads: Iterable[Any]) -> List[Dict[str, Any]]:
    return [item.to_dict() for item in CounterfactualEngine().compare_many(payloads)]


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run Amantia CounterfactualEngine adapter on one query JSON.")
    parser.add_argument("--input", required=True, help="Path to counterfactual query JSON.")
    parser.add_argument("--out", default="out/counterfactual_adapter_result.json", help="Output JSON path.")
    args = parser.parse_args(argv)
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    result = CounterfactualEngine().compare(payload).to_dict()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "status": "ok",
        "out": str(out_path),
        "compared": result.get("compared"),
        "comparison_status": result.get("comparison_status"),
        "recommended_action": result.get("recommended_action"),
        "score_margin": result.get("score_margin"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
