from __future__ import annotations

"""Goal discovery diagnostics for Structural Final Model development.

SFM inference is only as good as the candidate final causes it tests.  Earlier
steps required callers to supply ``candidate_goals`` explicitly.  This module
adds a conservative discovery layer that proposes plausible goal variables from
available action outcomes, SCM/belief graph descendants, agent utility weights,
and empirical outcome records.

The output is intentionally diagnostic: discovered goals are hypotheses to be
passed into the SFM pipeline, not proof of an agent's true ends.
"""

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .schema import FinalCauseQuery, GoalSpec
from .utility import _directional_utility, _outcome_value, _risk_penalty


_META_ACTION_KEYS = {
    "action",
    "action_name",
    "candidate_action",
    "selected_action",
    "name",
    "label",
    "risk",
    "risk_level",
    "metadata",
    "notes",
    "description",
}
_CONTAINER_KEYS = {
    "expected_outcomes",
    "outcome_scores",
    "goal_scores",
    "utility_scores",
    "utilities",
    "effect_estimates",
    "effects",
    "effect_scores",
    "metrics",
}
_POSITIVE_PRIOR_TOKENS = {
    "success",
    "satisfaction",
    "reward",
    "utility",
    "helpful",
    "resolved",
    "completion",
    "quality",
    "accuracy",
    "benefit",
    "retention",
    "conversion",
    "revenue",
}
_NEGATIVE_OR_PROTECTED_TOKENS = {
    "harm",
    "risk",
    "damage",
    "loss",
    "toxicity",
    "unsafe",
    "latency",
    "cost",
    "error",
    "failure",
    "violation",
}


@dataclass
class DiscoveredGoalCandidate:
    """One goal hypothesis produced by the discovery layer."""

    goal_variable: str = ""
    desired_direction: str = "increase"
    utility_weight: float = 1.0
    discovery_score: float = 0.0
    selected_by_goal: str = ""
    selected_goal_matches_observed: bool = False
    observed_rank_for_goal: Optional[int] = None
    observed_margin: Optional[float] = None
    evidence_sources: List[str] = field(default_factory=list)
    reason_codes: List[str] = field(default_factory=list)
    limits: List[str] = field(default_factory=list)
    graph_support: Dict[str, bool] = field(default_factory=dict)
    action_score_summary: List[Dict[str, Any]] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_goal_spec(self) -> GoalSpec:
        return GoalSpec(
            goal_variable=self.goal_variable,
            desired_direction=self.desired_direction,
            utility_weight=self.utility_weight,
            metadata={
                "source": "sfm_goal_discovery",
                "discovery_score": self.discovery_score,
                "evidence_sources": list(self.evidence_sources),
                "selected_by_goal": self.selected_by_goal,
                "selected_goal_matches_observed": self.selected_goal_matches_observed,
            },
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GoalDiscoveryReport:
    """Aggregate discovery result for a final-cause query."""

    assessed: bool = False
    discovered: bool = False
    explicit_goals_supplied: bool = False
    used_for_inference: bool = False
    observed_action: str = ""
    selected_goals: List[Dict[str, Any]] = field(default_factory=list)
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    max_goals: int = 3
    authority_status: str = "diagnostic_goal_hypotheses"
    reason: str = ""
    reason_codes: List[str] = field(default_factory=list)
    limits: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _clean_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


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


def _has_explicit_goal_payload(payload: Mapping[str, Any]) -> bool:
    return any(
        key in payload and payload.get(key) not in (None, "", [], {})
        for key in ["candidate_goals", "goals", "goal", "intended_outcome", "outcome"]
    )


def _is_negative_like(outcome: str) -> bool:
    lowered = outcome.lower()
    return any(token in lowered for token in _NEGATIVE_OR_PROTECTED_TOKENS)


def _is_positive_prior(outcome: str) -> bool:
    lowered = outcome.lower()
    return any(token in lowered for token in _POSITIVE_PRIOR_TOKENS)


def _graph_edges(graph: Mapping[str, Any]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for edge in _as_list(graph.get("edges")):
        if isinstance(edge, Mapping):
            src = _clean_str(edge.get("source") or edge.get("src") or edge.get("from") or edge.get("u"))
            dst = _clean_str(edge.get("target") or edge.get("dst") or edge.get("to") or edge.get("v"))
        elif isinstance(edge, (list, tuple)) and len(edge) >= 2:
            src, dst = _clean_str(edge[0]), _clean_str(edge[1])
        else:
            src, dst = "", ""
        if src and dst:
            out.append((src, dst))
    return out


def _descendants(graph: Mapping[str, Any], source: str) -> set[str]:
    source = _clean_str(source)
    if not graph or not source:
        return set()
    adjacency: Dict[str, List[str]] = {}
    for src, dst in _graph_edges(graph):
        adjacency.setdefault(src, []).append(dst)
    seen: set[str] = set()
    frontier = list(adjacency.get(source, []))
    while frontier:
        node = frontier.pop(0)
        if node in seen:
            continue
        seen.add(node)
        frontier.extend(adjacency.get(node, []))
    return seen


def _collect_outcome_names_from_actions(actions: Iterable[Mapping[str, Any]]) -> Dict[str, List[str]]:
    names: Dict[str, List[str]] = {}
    for option in actions:
        for key, value in option.items():
            if key in _CONTAINER_KEYS:
                for nested_key, nested_value in _as_dict(value).items():
                    if _safe_float(nested_value) is not None:
                        names.setdefault(_clean_str(nested_key), []).append(f"action_container:{key}")
            elif key not in _META_ACTION_KEYS and _safe_float(value) is not None:
                names.setdefault(_clean_str(key), []).append("action_numeric_field")
        if any(k in option for k in ["expected_success", "success_probability", "p_success"]):
            names.setdefault("task_success", []).append("expected_success_alias")
        if any(k in option for k in ["harm_probability", "expected_harm", "p_harm"]):
            names.setdefault("user_or_system_harm", []).append("harm_alias")
    return {k: v for k, v in names.items() if k}


def _collect_outcome_names_from_records(records: Iterable[Mapping[str, Any]]) -> Dict[str, List[str]]:
    names: Dict[str, List[str]] = {}
    for record in records:
        for key in ["observed_outcomes", "outcomes", "metrics", "outcome_scores", "goal_scores"]:
            for nested_key, nested_value in _as_dict(record.get(key)).items():
                if _safe_float(nested_value) is not None or isinstance(nested_value, bool):
                    names.setdefault(_clean_str(nested_key), []).append(f"record_container:{key}")
        for key in ["success", "helpful", "resolved"]:
            if key in record:
                names.setdefault("task_success", []).append(f"record_alias:{key}")
        for key in ["harm", "unsafe", "violation"]:
            if key in record:
                names.setdefault("user_or_system_harm", []).append(f"record_alias:{key}")
        for key in ["user_satisfaction", "satisfaction", "latency_ms"]:
            if key in record:
                names.setdefault(key, []).append(f"record_direct:{key}")
    return {k: v for k, v in names.items() if k}


def _direction_for_outcome(query: FinalCauseQuery, outcome: str) -> str:
    utility_weight = _safe_float((query.agent.utility_model or {}).get(outcome))
    if utility_weight is not None and utility_weight < 0:
        return "decrease"
    if outcome == query.protected_outcome or _is_negative_like(outcome):
        return "decrease"
    return "increase"


def _utility_weight_for_outcome(query: FinalCauseQuery, outcome: str, score: float) -> float:
    explicit = _safe_float((query.agent.utility_model or {}).get(outcome))
    if explicit is not None and explicit != 0:
        return round(max(0.1, min(5.0, abs(float(explicit)))), 6)
    return round(max(0.1, min(2.0, 0.5 + float(score))), 6)


def _rank_actions_for_outcome(query: FinalCauseQuery, outcome: str, direction: str) -> Tuple[str, bool, Optional[int], Optional[float], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    for option in query.candidate_actions:
        option_d = _as_dict(option)
        action = _action_name(option_d)
        if not action:
            continue
        value, source = _outcome_value(option_d, outcome, default=0.5)
        normalized = _directional_utility(value, direction, {})
        # Small generic risk adjustment so hazardous high-scoring actions do not
        # automatically become discovered final causes.
        adjusted = float(normalized) - 0.05 * _risk_penalty(option_d)
        rows.append(
            {
                "action": action,
                "value": round(float(value), 6),
                "directional_utility": round(float(normalized), 6),
                "adjusted_score": round(float(adjusted), 6),
                "value_source": source,
            }
        )
    rows.sort(key=lambda row: row["adjusted_score"], reverse=True)
    selected = str(rows[0]["action"]) if rows else ""
    matches = bool(selected and selected == query.observed_action)
    rank = None
    observed_score = None
    for idx, row in enumerate(rows, start=1):
        if row["action"] == query.observed_action:
            rank = idx
            observed_score = float(row["adjusted_score"])
            break
    margin = None
    if observed_score is not None:
        competitors = [float(row["adjusted_score"]) for row in rows if row["action"] != query.observed_action]
        if competitors:
            margin = round(observed_score - max(competitors), 6)
        else:
            margin = 0.0
    return selected, matches, rank, margin, rows


def _empirical_action_support(query: FinalCauseQuery, outcome: str, direction: str) -> Tuple[bool, float, List[str]]:
    if not query.outcome_records or not query.observed_action:
        return False, 0.0, []
    by_action: Dict[str, List[float]] = {}
    for record in query.outcome_records:
        action = _action_name(record)
        if not action:
            continue
        value: Optional[float] = None
        if outcome in record:
            value = _safe_float(record.get(outcome))
        if value is None:
            for key in ["observed_outcomes", "outcomes", "metrics", "outcome_scores", "goal_scores"]:
                nested = _as_dict(record.get(key))
                if outcome in nested:
                    value = _safe_float(nested.get(outcome))
                    break
        if value is None and outcome in {"task_success", "success"}:
            raw = record.get("success")
            if isinstance(raw, bool):
                value = 1.0 if raw else 0.0
        if value is None:
            continue
        by_action.setdefault(action, []).append(_directional_utility(_clip01(float(value)), direction, {}))
    means = {action: sum(values) / len(values) for action, values in by_action.items() if values}
    if query.observed_action not in means or len(means) < 2:
        return False, 0.0, ["empirical_records_insufficient_for_goal_discovery"]
    selected, selected_score = max(means.items(), key=lambda item: item[1])
    observed_score = means[query.observed_action]
    support = selected == query.observed_action
    margin = observed_score - max([score for action, score in means.items() if action != query.observed_action], default=observed_score)
    strength = _clip01(0.5 + margin)
    codes = ["SFM_GOAL_DISCOVERY_EMPIRICAL_OBSERVED_BEST"] if support else ["SFM_GOAL_DISCOVERY_EMPIRICAL_OBSERVED_NOT_BEST"]
    return support, strength if support else 0.0, codes


class GoalDiscoveryEngine:
    """Propose candidate final causes when none are supplied explicitly."""

    def discover(
        self,
        payload: FinalCauseQuery | Mapping[str, Any],
        *,
        max_goals: int = 3,
        min_discovery_score: float = 0.25,
        used_for_inference: bool = False,
    ) -> GoalDiscoveryReport:
        query = payload if isinstance(payload, FinalCauseQuery) else FinalCauseQuery.from_payload(payload)
        raw = _as_dict(query.raw)
        explicit_goals = bool(query.candidate_goals) and _has_explicit_goal_payload(raw)
        reason_codes: List[str] = ["SFM_GOAL_DISCOVERY_ASSESSED"]
        limits: List[str] = ["goal_discovery_outputs_hypotheses_not_confirmed_intentions"]

        action_outcomes = _collect_outcome_names_from_actions(query.candidate_actions)
        record_outcomes = _collect_outcome_names_from_records(query.outcome_records)
        real_desc = _descendants(query.scm_graph, query.action_variable)
        belief_desc = _descendants(query.agent.belief_graph, query.action_variable)
        utility_outcomes = {k for k, v in (query.agent.utility_model or {}).items() if _safe_float(v) is not None and str(k).strip()}

        outcome_names = set(action_outcomes) | set(record_outcomes) | set(real_desc) | set(belief_desc) | set(utility_outcomes)
        outcome_names.discard(query.action_variable)
        outcome_names.discard("agent_action")
        outcome_names.discard("action")

        if not outcome_names:
            return GoalDiscoveryReport(
                assessed=True,
                discovered=False,
                explicit_goals_supplied=explicit_goals,
                used_for_inference=False,
                observed_action=query.observed_action,
                max_goals=max_goals,
                reason="No outcome-like variables were available for goal discovery.",
                reason_codes=reason_codes + ["SFM_GOAL_DISCOVERY_NO_OUTCOME_CANDIDATES"],
                limits=limits + ["candidate_actions_graph_or_outcome_records_required_for_goal_discovery"],
            )

        candidates: List[DiscoveredGoalCandidate] = []
        for outcome in sorted(outcome_names):
            if not outcome:
                continue
            direction = _direction_for_outcome(query, outcome)
            selected, matches_observed, rank, margin, rows = _rank_actions_for_outcome(query, outcome, direction)
            evidence_sources: List[str] = []
            candidate_codes: List[str] = []
            candidate_limits: List[str] = []
            score = 0.0

            if outcome in action_outcomes:
                score += 0.15
                evidence_sources.extend(sorted(set(action_outcomes[outcome])))
                candidate_codes.append("SFM_GOAL_DISCOVERY_ACTION_OUTCOME_SURFACE")
            if outcome in record_outcomes:
                score += 0.10
                evidence_sources.extend(sorted(set(record_outcomes[outcome])))
                candidate_codes.append("SFM_GOAL_DISCOVERY_OUTCOME_RECORD_SURFACE")
            if outcome in real_desc:
                score += 0.25
                evidence_sources.append("real_graph_descendant_of_action")
                candidate_codes.append("SFM_GOAL_DISCOVERY_REAL_GRAPH_DESCENDANT")
            if outcome in belief_desc:
                score += 0.20
                evidence_sources.append("belief_graph_descendant_of_action")
                candidate_codes.append("SFM_GOAL_DISCOVERY_BELIEF_GRAPH_DESCENDANT")
            if outcome in utility_outcomes:
                score += 0.20
                evidence_sources.append("agent_utility_model")
                candidate_codes.append("SFM_GOAL_DISCOVERY_AGENT_UTILITY_MODEL")

            if matches_observed:
                score += 0.25
                candidate_codes.append("SFM_GOAL_DISCOVERY_OBSERVED_ACTION_BEST_FOR_GOAL")
                if margin is not None and margin > 0:
                    score += min(0.10, margin)
            elif rank is not None:
                candidate_codes.append("SFM_GOAL_DISCOVERY_OBSERVED_ACTION_NOT_BEST_FOR_GOAL")
                score -= 0.05

            empirical_support, empirical_strength, empirical_codes = _empirical_action_support(query, outcome, direction)
            candidate_codes.extend(empirical_codes)
            if empirical_support:
                score += 0.15 * empirical_strength
                evidence_sources.append("empirical_outcome_records")

            if _is_positive_prior(outcome):
                score += 0.08
                candidate_codes.append("SFM_GOAL_DISCOVERY_POSITIVE_GOAL_NAME_PRIOR")
            if outcome == query.protected_outcome or _is_negative_like(outcome):
                score -= 0.15
                candidate_limits.append("candidate_goal_looks_like_protected_outcome_or_side_effect")
                candidate_codes.append("SFM_GOAL_DISCOVERY_PROTECTED_OR_SIDE_EFFECT_PRIOR")

            score = round(max(0.0, min(1.0, score)), 6)
            if score <= 0:
                continue
            candidates.append(
                DiscoveredGoalCandidate(
                    goal_variable=outcome,
                    desired_direction=direction,
                    utility_weight=_utility_weight_for_outcome(query, outcome, score),
                    discovery_score=score,
                    selected_by_goal=selected,
                    selected_goal_matches_observed=matches_observed,
                    observed_rank_for_goal=rank,
                    observed_margin=margin,
                    evidence_sources=sorted(set(evidence_sources)),
                    reason_codes=candidate_codes,
                    limits=candidate_limits,
                    graph_support={
                        "real_graph_descendant_of_action": outcome in real_desc,
                        "belief_graph_descendant_of_action": outcome in belief_desc,
                    },
                    action_score_summary=rows,
                    raw={"action_outcome_sources": action_outcomes.get(outcome, []), "record_sources": record_outcomes.get(outcome, [])},
                )
            )

        candidates.sort(key=lambda item: (item.discovery_score, item.selected_goal_matches_observed), reverse=True)
        selected_candidates = [item for item in candidates if item.discovery_score >= min_discovery_score]
        # Do not select protected-looking outcomes when safer positive candidates exist.
        non_protected = [item for item in selected_candidates if "candidate_goal_looks_like_protected_outcome_or_side_effect" not in item.limits]
        if non_protected:
            selected_candidates = non_protected
        selected_candidates = selected_candidates[: max(1, int(max_goals or 1))]

        if selected_candidates:
            reason_codes.append("SFM_GOAL_DISCOVERY_FOUND_CANDIDATES")
        if explicit_goals:
            reason_codes.append("SFM_GOAL_DISCOVERY_EXPLICIT_GOALS_PRESERVED")
            limits.append("explicit_candidate_goals_supplied_discovery_not_used_for_inference")
        elif used_for_inference and selected_candidates:
            reason_codes.append("SFM_GOAL_DISCOVERY_USED_FOR_INFERENCE")
        elif not selected_candidates:
            reason_codes.append("SFM_GOAL_DISCOVERY_NO_CANDIDATE_PASSED_THRESHOLD")
            limits.append("goal_discovery_threshold_not_met")

        return GoalDiscoveryReport(
            assessed=True,
            discovered=bool(selected_candidates),
            explicit_goals_supplied=explicit_goals,
            used_for_inference=bool(used_for_inference and selected_candidates and not explicit_goals),
            observed_action=query.observed_action,
            selected_goals=[candidate.to_goal_spec().to_dict() for candidate in selected_candidates],
            candidates=[candidate.to_dict() for candidate in candidates],
            max_goals=max_goals,
            reason=(
                "Goal discovery ranked outcome-like variables from action outcome surfaces, SCM/belief descendants, "
                "agent utility weights, and empirical outcome records."
            ),
            reason_codes=reason_codes,
            limits=limits,
        )


def discover_candidate_goals(payload: FinalCauseQuery | Mapping[str, Any], **kwargs: Any) -> Dict[str, Any]:
    """Functional API for SFM goal discovery."""

    return GoalDiscoveryEngine().discover(payload, **kwargs).to_dict()
