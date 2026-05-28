from __future__ import annotations

"""Hierarchical SFM diagnostics: instrumental vs final goals.

Step 14 adds a conservative hierarchy layer on top of single-goal, multi-goal,
policy-learning, temporal, and context-conditioned SFM diagnostics.  It asks
whether a candidate goal is plausibly a *means* to a higher-level final goal.

Example: ``response_speed -> user_satisfaction``.  A fast response can be an
instrumental goal; user satisfaction is the more terminal telos.  This module
does not prove ultimate finality.  It exposes explicit or graph-derived
means-end structure so the main SFM result can avoid mistaking an instrument for
an ultimate end.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from .schema import FinalCauseQuery, GoalSpec
from .utility import _directional_utility, _outcome_value


@dataclass
class GoalHierarchyEdge:
    """A directed means-end relation between goals: source -> target."""

    source_goal: str = ""
    target_goal: str = ""
    relation: str = "instrumental_to_final"
    confidence: float = 1.0
    source: str = "explicit"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HierarchicalGoalProfile:
    """Role and action-support profile for one candidate goal."""

    goal_variable: str = ""
    role: str = "isolated_goal"
    desired_direction: str = "increase"
    immediate_targets: List[str] = field(default_factory=list)
    immediate_instruments: List[str] = field(default_factory=list)
    ultimate_goals: List[str] = field(default_factory=list)
    depth_to_nearest_ultimate_goal: Optional[int] = None
    observed_action_goal_score: float = 0.0
    best_action_for_goal: str = ""
    observed_action_rank_for_goal: Optional[int] = None
    observed_action_is_best_for_goal: bool = False
    supported_by_instruments: List[str] = field(default_factory=list)
    supports_ultimate_goals: List[str] = field(default_factory=list)
    support_strength: float = 0.0
    reason_codes: List[str] = field(default_factory=list)
    limits: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HierarchicalGoalAudit:
    """Means-end hierarchy audit over the candidate goal set."""

    assessed: bool = False
    mode: str = "hierarchical_sfm"
    hierarchy_detected: bool = False
    observed_action: str = ""
    terminal_goals: List[str] = field(default_factory=list)
    instrumental_goals: List[str] = field(default_factory=list)
    intermediate_goals: List[str] = field(default_factory=list)
    isolated_goals: List[str] = field(default_factory=list)
    selected_hierarchical_goal: str = ""
    selected_goal_role: str = ""
    selected_ultimate_goal: str = ""
    support_strength: float = 0.0
    hierarchy_edges: List[Dict[str, Any]] = field(default_factory=list)
    goal_profiles: List[Dict[str, Any]] = field(default_factory=list)
    ultimate_goal_by_instrument: Dict[str, List[str]] = field(default_factory=dict)
    reason: str = ""
    reason_codes: List[str] = field(default_factory=list)
    limits: List[str] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)

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


def _safe_float(value: Any, default: float = 1.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if out == out and out not in {float("inf"), float("-inf")} else default


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _dedupe(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    for item in items:
        text = _clean_str(item)
        if text and text not in out:
            out.append(text)
    return out


def _goal_names(query: FinalCauseQuery) -> List[str]:
    return _dedupe(goal.goal_variable for goal in query.candidate_goals if goal.goal_variable)


def _goal_map(query: FinalCauseQuery) -> Dict[str, GoalSpec]:
    return {goal.goal_variable: goal for goal in query.candidate_goals if goal.goal_variable}


def _edge_from_payload(payload: Any, *, source: str = "explicit") -> Optional[GoalHierarchyEdge]:
    if isinstance(payload, GoalHierarchyEdge):
        return payload
    if isinstance(payload, (list, tuple)) and len(payload) >= 2:
        return GoalHierarchyEdge(
            source_goal=_clean_str(payload[0]),
            target_goal=_clean_str(payload[1]),
            relation=_clean_str(payload[2] if len(payload) > 2 else "instrumental_to_final", "instrumental_to_final"),
            source=source,
        )
    if not isinstance(payload, Mapping):
        return None
    data = _as_dict(payload)
    src = _clean_str(
        data.get("source_goal")
        or data.get("instrumental_goal")
        or data.get("means")
        or data.get("from")
        or data.get("source")
        or data.get("cause")
        or data.get("parent")
    )
    dst = _clean_str(
        data.get("target_goal")
        or data.get("final_goal")
        or data.get("end")
        or data.get("to")
        or data.get("target")
        or data.get("effect")
        or data.get("child")
    )
    if not src or not dst:
        return None
    return GoalHierarchyEdge(
        source_goal=src,
        target_goal=dst,
        relation=_clean_str(data.get("relation"), "instrumental_to_final"),
        confidence=_clip01(_safe_float(data.get("confidence"), 1.0)),
        source=source,
        metadata=_as_dict(data.get("metadata")),
    )


def _explicit_edges(raw: Mapping[str, Any]) -> List[GoalHierarchyEdge]:
    edges: List[GoalHierarchyEdge] = []
    for key in [
        "goal_hierarchy_edges",
        "goal_hierarchy",
        "telos_edges",
        "instrumental_edges",
        "means_end_edges",
        "goal_edges",
    ]:
        value = raw.get(key)
        if not value:
            continue
        if isinstance(value, Mapping):
            if isinstance(value.get("edges"), list):
                for item in value.get("edges") or []:
                    edge = _edge_from_payload(item, source=f"explicit:{key}")
                    if edge:
                        edges.append(edge)
            else:
                for src, dst in value.items():
                    if src == "edges":
                        continue
                    for target in _as_list(dst):
                        edge = _edge_from_payload([src, target], source=f"explicit:{key}")
                        if edge:
                            edges.append(edge)
        else:
            for item in _as_list(value):
                edge = _edge_from_payload(item, source=f"explicit:{key}")
                if edge:
                    edges.append(edge)

    instrumental_map = raw.get("instrumental_goals") or raw.get("means_to_ends")
    if isinstance(instrumental_map, Mapping):
        for src, targets in instrumental_map.items():
            for target in _as_list(targets):
                edge = _edge_from_payload([src, target], source="explicit:instrumental_goals")
                if edge:
                    edges.append(edge)
    return edges


def _graph_edges(graph: Mapping[str, Any], *, source: str, goal_set: Set[str], action_variable: str) -> List[GoalHierarchyEdge]:
    out: List[GoalHierarchyEdge] = []
    raw_edges = graph.get("edges") or graph.get("directed_edges") or []
    for item in _as_list(raw_edges):
        edge = _edge_from_payload(item, source=source)
        if edge is None:
            continue
        src = edge.source_goal
        dst = edge.target_goal
        if src == action_variable or dst == action_variable:
            continue
        if src in goal_set and dst in goal_set and src != dst:
            edge.relation = "graph_goal_to_goal_path"
            out.append(edge)
    return out


def _filtered_unique_edges(edges: Iterable[GoalHierarchyEdge], goal_set: Set[str]) -> List[GoalHierarchyEdge]:
    seen: Set[Tuple[str, str]] = set()
    out: List[GoalHierarchyEdge] = []
    for edge in edges:
        src = _clean_str(edge.source_goal)
        dst = _clean_str(edge.target_goal)
        if not src or not dst or src == dst:
            continue
        if src not in goal_set or dst not in goal_set:
            continue
        key = (src, dst)
        if key in seen:
            continue
        seen.add(key)
        edge.source_goal = src
        edge.target_goal = dst
        out.append(edge)
    return out


def _hierarchy_edges(query: FinalCauseQuery) -> List[GoalHierarchyEdge]:
    raw = _as_dict(query.raw)
    goal_set = set(_goal_names(query))
    edges = _explicit_edges(raw)
    infer_from_graph = bool(raw.get("infer_goal_hierarchy_from_graph", raw.get("infer_hierarchy_from_graph", True)))
    if infer_from_graph:
        edges.extend(_graph_edges(query.scm_graph, source="scm_graph", goal_set=goal_set, action_variable=query.action_variable))
        edges.extend(_graph_edges(query.agent.belief_graph, source="agent_belief_graph", goal_set=goal_set, action_variable=query.action_variable))
    return _filtered_unique_edges(edges, goal_set)


def _adjacency(edges: Sequence[GoalHierarchyEdge]) -> Dict[str, List[str]]:
    adj: Dict[str, List[str]] = {}
    for edge in edges:
        adj.setdefault(edge.source_goal, [])
        if edge.target_goal not in adj[edge.source_goal]:
            adj[edge.source_goal].append(edge.target_goal)
        adj.setdefault(edge.target_goal, adj.get(edge.target_goal, []))
    return adj


def _reverse_adjacency(edges: Sequence[GoalHierarchyEdge]) -> Dict[str, List[str]]:
    rev: Dict[str, List[str]] = {}
    for edge in edges:
        rev.setdefault(edge.target_goal, [])
        if edge.source_goal not in rev[edge.target_goal]:
            rev[edge.target_goal].append(edge.source_goal)
        rev.setdefault(edge.source_goal, rev.get(edge.source_goal, []))
    return rev


def _descendants(start: str, adj: Mapping[str, Sequence[str]]) -> List[str]:
    seen: Set[str] = set()
    stack = list(adj.get(start, []))
    out: List[str] = []
    while stack:
        node = stack.pop(0)
        if node in seen:
            continue
        seen.add(node)
        out.append(node)
        for nxt in adj.get(node, []):
            if nxt not in seen:
                stack.append(nxt)
    return out


def _ancestors(start: str, rev: Mapping[str, Sequence[str]]) -> List[str]:
    seen: Set[str] = set()
    stack = list(rev.get(start, []))
    out: List[str] = []
    while stack:
        node = stack.pop(0)
        if node in seen:
            continue
        seen.add(node)
        out.append(node)
        for nxt in rev.get(node, []):
            if nxt not in seen:
                stack.append(nxt)
    return out


def _distance_to_terminal(start: str, adj: Mapping[str, Sequence[str]], terminals: Set[str]) -> Optional[int]:
    if start in terminals:
        return 0
    queue: List[Tuple[str, int]] = [(node, 1) for node in adj.get(start, [])]
    seen: Set[str] = set()
    while queue:
        node, dist = queue.pop(0)
        if node in seen:
            continue
        seen.add(node)
        if node in terminals:
            return dist
        for nxt in adj.get(node, []):
            queue.append((nxt, dist + 1))
    return None


def _is_protected_like(query: FinalCauseQuery, goal: GoalSpec) -> bool:
    name = _clean_str(goal.goal_variable).lower()
    protected = {_clean_str(query.protected_outcome).lower()}
    protected.update(_clean_str(x).lower() for x in goal.protected_outcomes)
    protected.update(_clean_str(x).lower() for x in goal.side_effect_outcomes)
    protected.update(_clean_str(g.goal_variable).lower() for g in query.side_effect_goals)
    return bool(name and (name in protected or any(token in name for token in ["harm", "risk", "damage", "unsafe"])))


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


def _action_goal_ranking(query: FinalCauseQuery, goal: GoalSpec) -> Tuple[List[Dict[str, Any]], float, Optional[int], str, bool]:
    rows: List[Dict[str, Any]] = []
    for option in query.candidate_actions:
        option_d = _as_dict(option)
        action = _action_name(option_d)
        value, source = _outcome_value(option_d, goal.goal_variable)
        utility = _directional_utility(value, goal.desired_direction, goal.metadata)
        rows.append({
            "action": action,
            "goal_variable": goal.goal_variable,
            "value": round(float(value), 6),
            "utility": round(float(utility), 6),
            "value_source": source,
        })
    rows.sort(key=lambda item: (item.get("utility", float("-inf")), item.get("action", "")), reverse=True)
    for idx, row in enumerate(rows, start=1):
        row["rank"] = idx
    observed = query.observed_action
    observed_score = 0.0
    observed_rank: Optional[int] = None
    for row in rows:
        if row.get("action") == observed:
            observed_score = float(row.get("utility", 0.0))
            observed_rank = int(row.get("rank"))
            break
    best_action = str(rows[0].get("action") or "") if rows else ""
    return rows, round(observed_score, 6), observed_rank, best_action, bool(observed_rank == 1)


def _support_strength(role: str, observed_best: bool, observed_score: float, instrument_count: int, depth: Optional[int]) -> float:
    base = float(observed_score or 0.0)
    if observed_best:
        base = max(base, 0.65)
    if role == "final_goal" and instrument_count:
        base += min(0.25, 0.08 * instrument_count)
    if role in {"instrumental_goal", "intermediate_goal"} and depth is not None:
        base += max(0.0, 0.08 - 0.02 * max(depth - 1, 0))
    if role == "protected_or_side_effect":
        base = min(base, 0.49)
    return round(_clip01(base), 6)


class HierarchicalGoalEvaluator:
    """Evaluate explicit or graph-derived means-end structure among goals."""

    def evaluate(self, payload: Any) -> HierarchicalGoalAudit:
        query = FinalCauseQuery.from_payload(payload)
        goals = _goal_map(query)
        reason_codes: List[str] = []
        limits: List[str] = []

        if len(goals) < 2:
            return HierarchicalGoalAudit(
                assessed=False,
                observed_action=query.observed_action,
                reason="Hierarchical SFM requires at least two candidate goals.",
                reason_codes=["SFM_HIERARCHY_REQUIRES_AT_LEAST_TWO_GOALS"],
                limits=["hierarchical_sfm_requires_two_or_more_candidate_goals"],
                raw=query.to_dict(),
            )
        if len(query.candidate_actions) < 1:
            return HierarchicalGoalAudit(
                assessed=False,
                observed_action=query.observed_action,
                reason="Hierarchical SFM requires candidate actions to assess observed action support.",
                reason_codes=["SFM_HIERARCHY_REQUIRES_ACTIONS"],
                limits=["candidate_action_set_required"],
                raw=query.to_dict(),
            )

        edges = _hierarchy_edges(query)
        if not edges:
            return HierarchicalGoalAudit(
                assessed=False,
                observed_action=query.observed_action,
                isolated_goals=list(goals.keys()),
                reason="No explicit or graph-derived goal hierarchy edges were available.",
                reason_codes=["SFM_HIERARCHY_NO_GOAL_EDGES"],
                limits=["goal_hierarchy_edges_required_for_hierarchical_sfm"],
                raw=query.to_dict(),
            )

        adj = _adjacency(edges)
        rev = _reverse_adjacency(edges)
        all_goal_names = set(goals.keys())
        outgoing = {goal: set(adj.get(goal, [])) for goal in all_goal_names}
        incoming = {goal: set(rev.get(goal, [])) for goal in all_goal_names}
        terminal_goals = sorted(goal for goal in all_goal_names if incoming.get(goal) and not outgoing.get(goal))
        instrumental_goals = sorted(goal for goal in all_goal_names if outgoing.get(goal) and not incoming.get(goal))
        intermediate_goals = sorted(goal for goal in all_goal_names if outgoing.get(goal) and incoming.get(goal))
        isolated_goals = sorted(goal for goal in all_goal_names if not outgoing.get(goal) and not incoming.get(goal))
        terminals = set(terminal_goals)

        profiles: List[HierarchicalGoalProfile] = []
        for goal_name, goal in goals.items():
            descendants = _descendants(goal_name, adj)
            ancestors = _ancestors(goal_name, rev)
            ultimate = [node for node in descendants if node in terminals]
            if goal_name in terminals:
                role = "final_goal"
                ultimate = [goal_name]
            elif goal_name in intermediate_goals:
                role = "intermediate_goal"
            elif goal_name in instrumental_goals:
                role = "instrumental_goal"
            elif _is_protected_like(query, goal):
                role = "protected_or_side_effect"
            else:
                role = "isolated_goal"

            rankings, observed_score, observed_rank, best_action, observed_best = _action_goal_ranking(query, goal)
            supported_by = []
            if role == "final_goal":
                supported_by = [ancestor for ancestor in ancestors if ancestor in instrumental_goals or ancestor in intermediate_goals]
            supports_ultimate = ultimate if role in {"instrumental_goal", "intermediate_goal"} else []
            depth = _distance_to_terminal(goal_name, adj, terminals)
            profile_reason_codes: List[str] = []
            profile_limits: List[str] = []
            if role == "final_goal":
                profile_reason_codes.append("SFM_HIERARCHY_GOAL_IS_TERMINAL_FINAL")
            elif role in {"instrumental_goal", "intermediate_goal"}:
                profile_reason_codes.append("SFM_HIERARCHY_GOAL_IS_INSTRUMENTAL")
                if supports_ultimate:
                    profile_reason_codes.append("SFM_HIERARCHY_INSTRUMENTAL_TO_FINAL_PATH_DETECTED")
            elif role == "protected_or_side_effect":
                profile_reason_codes.append("SFM_HIERARCHY_PROTECTED_LIKE_GOAL_CAPPED")
                profile_limits.append("protected_or_side_effect_goal_not_promoted_as_final_cause")
            else:
                profile_reason_codes.append("SFM_HIERARCHY_GOAL_IS_ISOLATED")
            if observed_best:
                profile_reason_codes.append("SFM_HIERARCHY_OBSERVED_ACTION_BEST_FOR_GOAL")
            else:
                profile_reason_codes.append("SFM_HIERARCHY_OBSERVED_ACTION_NOT_BEST_FOR_GOAL")

            profiles.append(
                HierarchicalGoalProfile(
                    goal_variable=goal_name,
                    role=role,
                    desired_direction=goal.desired_direction,
                    immediate_targets=sorted(outgoing.get(goal_name, [])),
                    immediate_instruments=sorted(incoming.get(goal_name, [])),
                    ultimate_goals=sorted(_dedupe(ultimate)),
                    depth_to_nearest_ultimate_goal=depth,
                    observed_action_goal_score=observed_score,
                    best_action_for_goal=best_action,
                    observed_action_rank_for_goal=observed_rank,
                    observed_action_is_best_for_goal=observed_best,
                    supported_by_instruments=sorted(supported_by),
                    supports_ultimate_goals=sorted(supports_ultimate),
                    support_strength=_support_strength(role, observed_best, observed_score, len(supported_by), depth),
                    reason_codes=profile_reason_codes,
                    limits=profile_limits,
                )
            )

        profile_rows = [profile.to_dict() for profile in profiles]
        final_profiles = [p for p in profiles if p.role == "final_goal"]
        if final_profiles:
            selected_profile = max(final_profiles, key=lambda p: (p.support_strength, len(p.supported_by_instruments), p.goal_variable))
        else:
            selected_profile = max(profiles, key=lambda p: (p.support_strength, p.goal_variable))
        selected_ultimate = selected_profile.goal_variable if selected_profile.role == "final_goal" else (
            selected_profile.ultimate_goals[0] if selected_profile.ultimate_goals else ""
        )
        support_strength = selected_profile.support_strength if selected_profile else 0.0

        reason_codes.append("SFM_HIERARCHY_ASSESSED")
        reason_codes.append("SFM_HIERARCHY_DETECTED")
        if terminal_goals:
            reason_codes.append("SFM_HIERARCHY_FINAL_GOALS_IDENTIFIED")
        if instrumental_goals or intermediate_goals:
            reason_codes.append("SFM_HIERARCHY_INSTRUMENTAL_GOALS_IDENTIFIED")
        if selected_ultimate:
            reason_codes.append("SFM_HIERARCHY_SELECTED_ULTIMATE_GOAL")
        if any(code == "SFM_HIERARCHY_INSTRUMENTAL_TO_FINAL_PATH_DETECTED" for p in profiles for code in p.reason_codes):
            reason_codes.append("SFM_HIERARCHY_INSTRUMENTAL_TO_FINAL_PATH_DETECTED")

        ultimate_by_instrument = {
            profile.goal_variable: profile.ultimate_goals
            for profile in profiles
            if profile.role in {"instrumental_goal", "intermediate_goal"} and profile.ultimate_goals
        }

        limits.append("diagnostic_only_not_full_hierarchical_sfm_identification")
        if any(edge.source.startswith("scm_graph") or edge.source.startswith("agent_belief_graph") for edge in edges):
            limits.append("graph_derived_goal_hierarchy_requires_domain_validation")

        return HierarchicalGoalAudit(
            assessed=True,
            hierarchy_detected=True,
            observed_action=query.observed_action,
            terminal_goals=terminal_goals,
            instrumental_goals=instrumental_goals,
            intermediate_goals=intermediate_goals,
            isolated_goals=isolated_goals,
            selected_hierarchical_goal=selected_profile.goal_variable,
            selected_goal_role=selected_profile.role,
            selected_ultimate_goal=selected_ultimate,
            support_strength=round(float(support_strength), 6),
            hierarchy_edges=[edge.to_dict() for edge in edges],
            goal_profiles=profile_rows,
            ultimate_goal_by_instrument=ultimate_by_instrument,
            reason=(
                "Hierarchical SFM audit classified candidate goals into instrumental, intermediate, "
                "terminal/final, and isolated roles using explicit or graph-derived means-end edges."
            ),
            reason_codes=_dedupe([*reason_codes, *(code for profile in profiles for code in profile.reason_codes)]),
            limits=_dedupe([*limits, *(limit for profile in profiles for limit in profile.limits)]),
            raw=query.to_dict(),
        )


def evaluate_hierarchical_goals(payload: Any) -> Dict[str, Any]:
    """Convenience API for hierarchical SFM diagnostics."""

    return HierarchicalGoalEvaluator().evaluate(payload).to_dict()
