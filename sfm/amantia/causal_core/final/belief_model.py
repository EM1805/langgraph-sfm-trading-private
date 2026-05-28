from __future__ import annotations

"""Agent belief graph diagnostics for Structural Final Model development.

An SFM needs two different causal objects:

1. the real/system SCM graph used to evaluate whether an action actually has a
   causal path to a goal;
2. the agent's belief graph used to evaluate whether the action was plausibly
   selected *because the agent believed* it would advance that goal.

This module keeps that distinction explicit.  It intentionally performs only a
small graph audit: directed reachability and agreement/mismatch between the real
and believed action->goal relation.  It is diagnostic, not a replacement for
full SCM identification.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from .schema import FinalCauseQuery, GoalSpec


@dataclass
class BeliefCausalAssessment:
    """Compare real causal support with the agent's believed causal support."""

    assessed: bool = False
    action_variable: str = ""
    goal_variable: str = ""
    belief_model_supplied: bool = False
    real_graph_supplied: bool = False
    real_direct_edge: Optional[bool] = None
    belief_direct_edge: Optional[bool] = None
    real_has_path: Optional[bool] = None
    belief_has_path: Optional[bool] = None
    belief_agrees_with_real: Optional[bool] = None
    belief_error_type: str = "unknown"
    intent_under_agent_beliefs: bool = False
    real_world_goal_path_supported: bool = False
    authority_status: str = "diagnostic_only"
    reason: str = ""
    reason_codes: List[str] = field(default_factory=list)
    real_path: List[str] = field(default_factory=list)
    belief_path: List[str] = field(default_factory=list)
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


def _node_id(node: Any) -> str:
    if isinstance(node, Mapping):
        return _clean_str(node.get("id") or node.get("name") or node.get("variable"))
    return _clean_str(node)


def _edge_pair(edge: Any) -> Tuple[str, str]:
    if isinstance(edge, Mapping):
        src = _clean_str(edge.get("source") or edge.get("src") or edge.get("from") or edge.get("parent") or edge.get("u"))
        dst = _clean_str(edge.get("target") or edge.get("dst") or edge.get("to") or edge.get("child") or edge.get("v"))
        return src, dst
    if isinstance(edge, Sequence) and not isinstance(edge, (str, bytes)) and len(edge) >= 2:
        return _clean_str(edge[0]), _clean_str(edge[1])
    return "", ""


def _extract_edges(graph: Mapping[str, Any]) -> List[Tuple[str, str]]:
    edges: List[Tuple[str, str]] = []
    for edge in graph.get("edges") or graph.get("directed_edges") or []:
        src, dst = _edge_pair(edge)
        if src and dst:
            edges.append((src, dst))
    return edges


def _extract_nodes(graph: Mapping[str, Any], edges: Iterable[Tuple[str, str]]) -> Set[str]:
    nodes: Set[str] = set()
    for node in graph.get("nodes") or []:
        node_id = _node_id(node)
        if node_id:
            nodes.add(node_id)
    for src, dst in edges:
        nodes.add(src)
        nodes.add(dst)
    return nodes


def _adjacency(edges: Iterable[Tuple[str, str]]) -> Dict[str, List[str]]:
    adj: Dict[str, List[str]] = {}
    for src, dst in edges:
        adj.setdefault(src, []).append(dst)
        adj.setdefault(dst, adj.get(dst, []))
    for src in adj:
        adj[src] = sorted(set(adj[src]))
    return adj


def _directed_path(graph: Mapping[str, Any], source: str, target: str) -> Tuple[Optional[bool], List[str], Optional[bool]]:
    """Return (has_path, path, direct_edge), using None when graph is absent."""

    graph = _as_dict(graph)
    if not graph:
        return None, [], None
    edges = _extract_edges(graph)
    nodes = _extract_nodes(graph, edges)
    if not source or not target or source not in nodes or target not in nodes:
        return False, [], False
    direct = (source, target) in set(edges)
    adj = _adjacency(edges)
    queue: List[Tuple[str, List[str]]] = [(source, [source])]
    seen: Set[str] = set()
    while queue:
        node, path = queue.pop(0)
        if node == target:
            return True, path, direct
        if node in seen:
            continue
        seen.add(node)
        for child in adj.get(node, []):
            if child not in seen:
                queue.append((child, [*path, child]))
    return False, [], direct


def _belief_error_type(real_has_path: Optional[bool], belief_has_path: Optional[bool]) -> str:
    if belief_has_path is None:
        return "missing_belief_graph"
    if real_has_path is None:
        return "real_graph_missing"
    if bool(real_has_path) and bool(belief_has_path):
        return "none"
    if not real_has_path and belief_has_path:
        return "false_positive_belief"
    if real_has_path and not belief_has_path:
        return "false_negative_belief"
    return "shared_no_path"


class AgentBeliefEvaluator:
    """Assess action->goal support in the real graph and agent belief graph."""

    def assess(self, query: FinalCauseQuery, goal: GoalSpec) -> BeliefCausalAssessment:
        action = query.action_variable
        goal_var = goal.goal_variable
        real_graph = _as_dict(query.scm_graph)
        belief_graph = _as_dict(query.agent.belief_graph)
        real_has_path, real_path, real_direct = _directed_path(real_graph, action, goal_var)
        belief_has_path, belief_path, belief_direct = _directed_path(belief_graph, action, goal_var)
        error_type = _belief_error_type(real_has_path, belief_has_path)

        agrees: Optional[bool]
        if belief_has_path is None or real_has_path is None:
            agrees = None
        else:
            agrees = bool(real_has_path) == bool(belief_has_path)

        reason_codes: List[str] = []
        if real_graph:
            reason_codes.append("SFM_REAL_GRAPH_SUPPLIED")
            if real_has_path:
                reason_codes.append("SFM_REAL_ACTION_GOAL_PATH_PRESENT")
            else:
                reason_codes.append("SFM_REAL_ACTION_GOAL_PATH_ABSENT")
        else:
            reason_codes.append("SFM_REAL_GRAPH_MISSING")

        if belief_graph:
            reason_codes.append("SFM_AGENT_BELIEF_GRAPH_SUPPLIED")
            if belief_has_path:
                reason_codes.append("SFM_AGENT_BELIEVES_ACTION_CAN_AFFECT_GOAL")
            else:
                reason_codes.append("SFM_AGENT_DOES_NOT_BELIEVE_ACTION_CAN_AFFECT_GOAL")
        else:
            reason_codes.append("SFM_AGENT_BELIEF_GRAPH_MISSING")

        if error_type == "false_positive_belief":
            reason_codes.append("SFM_AGENT_BELIEF_FALSE_POSITIVE_VS_REAL_GRAPH")
        elif error_type == "false_negative_belief":
            reason_codes.append("SFM_AGENT_BELIEF_FALSE_NEGATIVE_VS_REAL_GRAPH")
        elif error_type == "none":
            reason_codes.append("SFM_AGENT_BELIEF_ALIGNED_WITH_REAL_PATH")
        elif error_type == "shared_no_path":
            reason_codes.append("SFM_NO_REAL_OR_BELIEVED_ACTION_GOAL_PATH")

        return BeliefCausalAssessment(
            assessed=True,
            action_variable=action,
            goal_variable=goal_var,
            belief_model_supplied=bool(belief_graph),
            real_graph_supplied=bool(real_graph),
            real_direct_edge=real_direct,
            belief_direct_edge=belief_direct,
            real_has_path=real_has_path,
            belief_has_path=belief_has_path,
            belief_agrees_with_real=agrees,
            belief_error_type=error_type,
            intent_under_agent_beliefs=bool(belief_has_path),
            real_world_goal_path_supported=bool(real_has_path),
            reason=(
                "Compared the real SCM graph with the agent belief graph for directed action-to-goal reachability."
            ),
            reason_codes=reason_codes,
            real_path=real_path,
            belief_path=belief_path,
            raw={"query": query.to_dict(), "goal": goal.to_dict()},
        )


def assess_agent_beliefs(payload: Any, goal: GoalSpec | str | Mapping[str, Any] | None = None) -> Dict[str, Any]:
    """Convenience API for real-vs-believed action->goal diagnostics."""

    query = FinalCauseQuery.from_payload(payload)
    goal_obj = GoalSpec.from_payload(goal) if goal is not None else query.candidate_goals[0]
    return AgentBeliefEvaluator().assess(query, goal_obj).to_dict()
