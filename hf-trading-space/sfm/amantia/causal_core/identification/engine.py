from __future__ import annotations

"""Stable online SCM-ID adapter for Amantia.

This facade accepts compact agent/runtime graph payloads and routes them into
Amantia's existing SCM-ID / Full-ID-shaped backend without letting backend
errors crash the decision gate.  It is deliberately conservative: missing graph
or invalid query returns a blocked result, while complete arbitrary Full-ID
claims remain gated by the backend's ``full_id_claim_allowed`` flag.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def _s(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _dedupe(values: Iterable[Any]) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values or []:
        item = _s(value)
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        if "," in value:
            return _dedupe(part for part in value.split(","))
        if "|" in value:
            return _dedupe(part for part in value.split("|"))
        return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple, set)):
        return _dedupe(value)
    return [_s(value)] if _s(value) else []


def _split_pipe(value: Any) -> List[str]:
    return _as_list(value)


def _edge_source_target(edge: Any) -> Tuple[str, str, str]:
    """Return ``source, target, edge_kind`` from compact/list/dict edge rows."""
    if isinstance(edge, Mapping):
        source = _s(edge.get("source") or edge.get("from") or edge.get("src") or edge.get("u"))
        target = _s(edge.get("target") or edge.get("to") or edge.get("dst") or edge.get("v"))
        kind = _s(edge.get("edge_kind") or edge.get("edge_type") or edge.get("kind") or edge.get("type"))
        if not kind and bool(edge.get("bidirected")):
            kind = "bidirected"
        return source, target, kind or "directed"
    if isinstance(edge, (list, tuple)) and len(edge) >= 2:
        kind = _s(edge[2]) if len(edge) >= 3 else "directed"
        return _s(edge[0]), _s(edge[1]), kind or "directed"
    return "", "", ""


def normalize_scm_graph(graph: Mapping[str, Any] | None) -> Dict[str, Any]:
    """Normalize compact agent graph shapes into SCM-ID graph rows.

    Accepted examples:
    - ``{"nodes": ["X", "Y"], "edges": [["X", "Y"]]}``
    - ``{"nodes": [{"id": "X"}], "edges": [{"source": "X", "target": "Y"}]}``
    - optional ``directed_edges`` / ``bidirected_edges`` lists.
    """
    graph = dict(graph or {}) if isinstance(graph, Mapping) else {}
    node_ids: List[str] = []

    for node in graph.get("nodes", []) or []:
        if isinstance(node, Mapping):
            node_ids.append(_s(node.get("node_id") or node.get("id") or node.get("name") or node.get("label")))
        else:
            node_ids.append(_s(node))

    normalized_edges: List[Dict[str, str]] = []

    def add_edge(source: Any, target: Any, kind: str = "directed") -> None:
        s, t, k = _s(source), _s(target), _s(kind or "directed")
        if not s or not t:
            return
        node_ids.extend([s, t])
        row = {"source": s, "target": t}
        if k and k.lower() not in {"directed", "causal", "arrow"}:
            row["edge_kind"] = k
        normalized_edges.append(row)

    for edge in graph.get("edges", []) or []:
        source, target, kind = _edge_source_target(edge)
        add_edge(source, target, kind)

    for edge in graph.get("directed_edges", []) or []:
        source, target, _kind = _edge_source_target(edge)
        add_edge(source, target, "directed")

    for edge in graph.get("bidirected_edges", []) or []:
        source, target, _kind = _edge_source_target(edge)
        add_edge(source, target, "bidirected")

    nodes = [{"id": node} for node in _dedupe(node_ids)]
    out = dict(graph)
    out["nodes"] = nodes
    out["edges"] = normalized_edges
    if "queries" in graph:
        out["queries"] = graph.get("queries") or []
    return out


@dataclass(frozen=True)
class IdentificationQuery:
    scm_graph: Dict[str, Any]
    treatments: List[str]
    outcomes: List[str]
    conditions: List[str] = field(default_factory=list)
    adjustment_set: List[str] = field(default_factory=list)
    mediators: List[str] = field(default_factory=list)
    strategy_hint: str = ""
    query_id: str = ""
    source: str = "amantia.causal_core.identification"
    max_depth: int = 8

    @property
    def treatment(self) -> str:
        return self.treatments[0] if self.treatments else ""

    @property
    def outcome(self) -> str:
        return self.outcomes[0] if self.outcomes else ""

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any] | None) -> "IdentificationQuery":
        payload = dict(payload or {}) if isinstance(payload, Mapping) else {}
        graph = payload.get("scm_graph") or payload.get("graph") or payload.get("causal_graph") or {}
        if not graph and ("nodes" in payload or "edges" in payload or "directed_edges" in payload or "bidirected_edges" in payload):
            graph = payload
        graph = normalize_scm_graph(graph if isinstance(graph, Mapping) else {})
        treatments = _as_list(payload.get("treatments") or payload.get("treatment") or payload.get("action") or payload.get("action_name"))
        outcomes = _as_list(payload.get("outcomes") or payload.get("outcome") or payload.get("protected_outcome") or payload.get("target"))
        conditions = _as_list(payload.get("conditions") or payload.get("condition_set") or payload.get("conditioning_set"))
        return cls(
            scm_graph=graph,
            treatments=treatments,
            outcomes=outcomes,
            conditions=conditions,
            adjustment_set=_as_list(payload.get("adjustment_set") or payload.get("controls") or payload.get("confounders")),
            mediators=_as_list(payload.get("mediators") or payload.get("mediator_set")),
            strategy_hint=_s(payload.get("strategy_hint")),
            query_id=_s(payload.get("query_id") or payload.get("id")),
            source=_s(payload.get("source")) or "amantia.causal_core.identification",
            max_depth=int(payload.get("max_depth") or 8),
        )


@dataclass(frozen=True)
class IdentificationResult:
    identified: bool
    treatment: str = ""
    outcome: str = ""
    treatments: List[str] = field(default_factory=list)
    outcomes: List[str] = field(default_factory=list)
    conditions: List[str] = field(default_factory=list)
    adjustment_set: List[str] = field(default_factory=list)
    mediators: List[str] = field(default_factory=list)
    identification_strategy: str = ""
    identification_tier: str = "unidentified"
    estimand: str = ""
    formula: str = ""
    authority_status: str = ""
    primary_formula_authority: str = ""
    full_id_claim_allowed: int = 0
    failure_certificate_status: str = ""
    failure_certified: int = 0
    idc_pruned_conditions: str = ""
    idc_pruning_status: str = ""
    reason: str = ""
    reason_codes: List[str] = field(default_factory=list)
    raw_id_result: Dict[str, Any] = field(default_factory=dict)
    query_id: str = ""
    source: str = "amantia.causal_core.identification"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _blocked_result(
    *,
    strategy: str,
    reason: str,
    reason_codes: Sequence[str],
    query: Optional[IdentificationQuery] = None,
    raw: Optional[Mapping[str, Any]] = None,
) -> IdentificationResult:
    query = query or IdentificationQuery({}, [], [])
    return IdentificationResult(
        identified=False,
        treatment=query.treatment,
        outcome=query.outcome,
        treatments=list(query.treatments),
        outcomes=list(query.outcomes),
        conditions=list(query.conditions),
        adjustment_set=list(query.adjustment_set),
        mediators=list(query.mediators),
        identification_strategy=strategy,
        identification_tier="unidentified" if strategy != "blocked_nonidentifiable" else "blocked_nonidentifiable",
        authority_status="blocked",
        reason=reason,
        reason_codes=list(reason_codes),
        raw_id_result=dict(raw or {}),
        query_id=query.query_id,
        source=query.source,
    )


def _has_graph(graph: Mapping[str, Any]) -> bool:
    return bool(graph and (graph.get("nodes") or graph.get("edges")))


def _map_full_id(query: IdentificationQuery, diagnostic: Any) -> IdentificationResult:
    raw = diagnostic.to_dict() if hasattr(diagnostic, "to_dict") else dict(diagnostic or {})
    identified = bool(raw.get("identified"))
    treatments = _split_pipe(raw.get("treatments")) or list(query.treatments)
    outcomes = _split_pipe(raw.get("outcomes")) or list(query.outcomes)
    tier = "identified_canonical" if identified else "blocked_nonidentifiable"
    strategy = _s(raw.get("identification_status")) or ("identified_full_id_backend" if identified else "blocked_nonidentifiable")
    return IdentificationResult(
        identified=identified,
        treatment=treatments[0] if treatments else query.treatment,
        outcome=outcomes[0] if outcomes else query.outcome,
        treatments=treatments,
        outcomes=outcomes,
        conditions=list(query.conditions),
        adjustment_set=list(query.adjustment_set),
        mediators=list(query.mediators),
        identification_strategy=strategy,
        identification_tier=tier,
        estimand=_s(raw.get("formula")),
        formula=_s(raw.get("formula")),
        authority_status="canonical_formula" if identified else "failure_certificate",
        primary_formula_authority=_s(raw.get("primary_formula_authority")),
        full_id_claim_allowed=int(raw.get("full_id_claim_allowed") or 0),
        failure_certificate_status=_s(raw.get("failure_certificate_status")),
        failure_certified=int(raw.get("failure_certified") or 0),
        reason=_s(raw.get("reason_codes")) or strategy,
        reason_codes=[c for c in _split_pipe(raw.get("reason_codes")) if c] or (["SCM_ID_IDENTIFIED"] if identified else ["SCM_ID_BLOCKED"]),
        raw_id_result=raw,
        query_id=query.query_id,
        source=query.source,
    )


def _map_conditional_id(query: IdentificationQuery, diagnostic: Any) -> IdentificationResult:
    raw = diagnostic.to_dict() if hasattr(diagnostic, "to_dict") else dict(diagnostic or {})
    identified = bool(raw.get("identified"))
    treatments = _split_pipe(raw.get("treatments")) or list(query.treatments)
    outcomes = _split_pipe(raw.get("outcomes")) or list(query.outcomes)
    conditions = _split_pipe(raw.get("conditions")) or list(query.conditions)
    return IdentificationResult(
        identified=identified,
        treatment=treatments[0] if treatments else query.treatment,
        outcome=outcomes[0] if outcomes else query.outcome,
        treatments=treatments,
        outcomes=outcomes,
        conditions=conditions,
        adjustment_set=list(query.adjustment_set),
        mediators=list(query.mediators),
        identification_strategy=_s(raw.get("identification_status")) or "identified_conditional" if identified else "blocked_conditional",
        identification_tier="identified_conditional" if identified else "blocked_nonidentifiable",
        estimand=_s(raw.get("formula")),
        formula=_s(raw.get("formula")),
        authority_status="conditional_idc" if identified else "failure_certificate",
        primary_formula_authority=_s(raw.get("primary_formula_authority")),
        full_id_claim_allowed=int(raw.get("full_id_claim_allowed") or 0),
        failure_certificate_status=_s(raw.get("failure_certificate_status")),
        failure_certified=int(raw.get("failure_certified") or 0),
        idc_pruned_conditions=_s(raw.get("idc_pruned_conditions")),
        idc_pruning_status=_s(raw.get("idc_pruning_status")),
        reason=_s(raw.get("reason_codes")) or ("SCM_ID_CONDITIONAL_IDENTIFIED" if identified else "SCM_ID_CONDITIONAL_BLOCKED"),
        reason_codes=[c for c in _split_pipe(raw.get("reason_codes")) if c] or (["SCM_ID_CONDITIONAL_IDENTIFIED"] if identified else ["SCM_ID_CONDITIONAL_BLOCKED"]),
        raw_id_result=raw,
        query_id=query.query_id,
        source=query.source,
    )


def _map_graphical(query: IdentificationQuery, diagnostic: Any) -> IdentificationResult:
    raw = diagnostic.to_dict() if hasattr(diagnostic, "to_dict") else dict(diagnostic or {})
    identified = bool(raw.get("identifiable"))
    strategy = _s(raw.get("id_strategy")) or _s(raw.get("id_algorithm_level")) or "graphical_id_backend"
    formula = _s(raw.get("estimand_formula"))
    return IdentificationResult(
        identified=identified,
        treatment=_s(raw.get("treatment")) or query.treatment,
        outcome=_s(raw.get("outcome")) or query.outcome,
        treatments=[_s(raw.get("treatment")) or query.treatment],
        outcomes=[_s(raw.get("outcome")) or query.outcome],
        conditions=list(query.conditions),
        adjustment_set=list(query.adjustment_set),
        mediators=list(query.mediators),
        identification_strategy=strategy,
        identification_tier="identified_graphical" if identified else "unidentified",
        estimand=formula,
        formula=formula,
        authority_status="graphical_diagnostic" if identified else "blocked",
        reason=_s(raw.get("reason_codes")) or strategy,
        reason_codes=[c for c in _split_pipe(raw.get("reason_codes")) if c] or (["SCM_ID_IDENTIFIED"] if identified else ["SCM_ID_UNIDENTIFIED"]),
        raw_id_result=raw,
        query_id=query.query_id,
        source=query.source,
    )


def _dedupe_pairs(values: Iterable[Tuple[str, str]]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    seen = set()
    for a, b in values:
        key = (a, b)
        if a and b and key not in seen:
            seen.add(key)
            out.append(key)
    return out


def _normalized_graph_components(graph: Mapping[str, Any]) -> Tuple[set, List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Return graph nodes, directed edges and bidirected/confounding edges.

    Step 90 hardening: an LLM-supplied ``adjustment_set`` is never accepted by
    shape alone.  The online adapter validates that it blocks every open
    backdoor path and rejects hidden-confounding or post-treatment adjustment.
    """

    nodes = {
        _s(n.get("id") or n.get("node_id") or n.get("name")) if isinstance(n, Mapping) else _s(n)
        for n in graph.get("nodes", []) or []
    }
    directed: List[Tuple[str, str]] = []
    bidirected: List[Tuple[str, str]] = []
    bidirected_kinds = {
        "bidirected",
        "bi-directed",
        "confounded",
        "confounding",
        "latent",
        "latent_confounding",
        "unobserved_confounding",
        "hidden_confounding",
        "<->",
    }

    for edge in graph.get("edges", []) or []:
        source, target, kind = _edge_source_target(edge)
        if not source or not target:
            continue
        nodes.update([source, target])
        if (kind or "").strip().lower() in bidirected_kinds:
            bidirected.append((source, target))
        else:
            directed.append((source, target))

    for edge in graph.get("directed_edges", []) or []:
        source, target, _kind = _edge_source_target(edge)
        if source and target:
            nodes.update([source, target])
            directed.append((source, target))

    for edge in graph.get("bidirected_edges", []) or []:
        source, target, _kind = _edge_source_target(edge)
        if source and target:
            nodes.update([source, target])
            bidirected.append((source, target))

    return {n for n in nodes if n}, _dedupe_pairs(directed), _dedupe_pairs(bidirected)


def _descendants(start: str, directed: Sequence[Tuple[str, str]]) -> set:
    children: Dict[str, List[str]] = {}
    for source, target in directed:
        children.setdefault(source, []).append(target)

    seen = set()
    stack = list(children.get(start, []))
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(children.get(node, []))
    return seen


def _edge_has_arrowhead_at(left: str, right: str, node: str, directed: Sequence[Tuple[str, str]], bidirected: Sequence[Tuple[str, str]]) -> bool:
    if (left, right) in bidirected or (right, left) in bidirected:
        return node in {left, right}
    if (left, right) in directed:
        return node == right
    if (right, left) in directed:
        return node == left
    return False


def _first_step_is_backdoor(path: Sequence[str], x: str, directed: Sequence[Tuple[str, str]], bidirected: Sequence[Tuple[str, str]]) -> bool:
    if len(path) < 2 or path[0] != x:
        return False
    nxt = path[1]
    return (nxt, x) in directed or (x, nxt) in bidirected or (nxt, x) in bidirected


def _simple_paths(start: str, goal: str, directed: Sequence[Tuple[str, str]], bidirected: Sequence[Tuple[str, str]], max_depth: int) -> List[List[str]]:
    neighbors: Dict[str, List[str]] = {}
    for source, target in directed:
        neighbors.setdefault(source, []).append(target)
        neighbors.setdefault(target, []).append(source)
    for source, target in bidirected:
        neighbors.setdefault(source, []).append(target)
        neighbors.setdefault(target, []).append(source)

    paths: List[List[str]] = []
    stack: List[Tuple[str, List[str]]] = [(start, [start])]
    while stack:
        node, path = stack.pop()
        if len(path) > max_depth + 1:
            continue
        for nxt in neighbors.get(node, []):
            if nxt in path:
                continue
            candidate = path + [nxt]
            if nxt == goal:
                paths.append(candidate)
            else:
                stack.append((nxt, candidate))
    return paths


def _path_is_active(path: Sequence[str], z: set, directed: Sequence[Tuple[str, str]], bidirected: Sequence[Tuple[str, str]]) -> bool:
    if len(path) <= 2:
        return True
    descendants_cache: Dict[str, set] = {}

    for idx in range(1, len(path) - 1):
        prev_node = path[idx - 1]
        node = path[idx]
        next_node = path[idx + 1]
        left_head = _edge_has_arrowhead_at(prev_node, node, node, directed, bidirected)
        right_head = _edge_has_arrowhead_at(next_node, node, node, directed, bidirected)
        is_collider = left_head and right_head

        if is_collider:
            if node not in descendants_cache:
                descendants_cache[node] = _descendants(node, directed)
            if node not in z and not (descendants_cache[node] & z):
                return False
        else:
            if node in z:
                return False
    return True


def _format_path(path: Sequence[str]) -> str:
    return "->".join(path)


def _simple_graphical_identification(query: IdentificationQuery) -> IdentificationResult:
    nodes, directed, bidirected = _normalized_graph_components(query.scm_graph)
    missing = [n for n in list(query.treatments) + list(query.outcomes) if n not in nodes]
    if missing:
        return _blocked_result(
            strategy="blocked_invalid_graphical_query",
            reason="Query nodes missing from graph: " + ",".join(missing),
            reason_codes=["QUERY_NODE_NOT_IN_GRAPH"],
            query=query,
            raw={"adapter": "simple_graphical_identification", "nodes_checked": sorted(nodes)},
        )

    if len(query.treatments) != 1 or len(query.outcomes) != 1:
        return _blocked_result(
            strategy="blocked_graphical_query_scope",
            reason="The online graphical adapter only validates one treatment and one outcome at a time.",
            reason_codes=["GRAPHICAL_ADAPTER_SINGLE_QUERY_ONLY"],
            query=query,
            raw={"adapter": "simple_graphical_identification"},
        )

    x = query.treatment
    y = query.outcome

    if query.mediators and not query.adjustment_set:
        return _blocked_result(
            strategy="blocked_mediator_strategy_requires_backend",
            reason="Mediator/frontdoor-style identification is not certified by the simple online backdoor adapter.",
            reason_codes=["MEDIATOR_IDENTIFICATION_REQUIRES_FULL_BACKEND"],
            query=query,
            raw={"adapter": "simple_graphical_identification", "mediators": list(query.mediators)},
        )

    supplied_z = _dedupe(query.adjustment_set)
    unknown_z = [v for v in supplied_z if v not in nodes]
    forbidden_z = [v for v in supplied_z if v in {x, y}]
    descendants_of_x = _descendants(x, directed)
    post_treatment_z = [v for v in supplied_z if v in descendants_of_x]

    invalid_reasons: List[str] = []
    if unknown_z:
        invalid_reasons.append("ADJUSTMENT_VARIABLE_NOT_IN_GRAPH")
    if forbidden_z:
        invalid_reasons.append("ADJUSTMENT_SET_CONTAINS_TREATMENT_OR_OUTCOME")
    if post_treatment_z:
        invalid_reasons.append("ADJUSTMENT_SET_CONTAINS_POST_TREATMENT_DESCENDANT")

    z = {v for v in supplied_z if v in nodes and v not in {x, y} and v not in descendants_of_x}
    all_paths = _simple_paths(x, y, directed, bidirected, max(1, query.max_depth or 8))
    backdoor_paths = [p for p in all_paths if _first_step_is_backdoor(p, x, directed, bidirected)]
    open_backdoor_paths = [p for p in backdoor_paths if _path_is_active(p, z, directed, bidirected)]

    directly_hidden_confounding = any({a, b} == {x, y} for a, b in bidirected)
    if directly_hidden_confounding:
        invalid_reasons.append("HIDDEN_CONFOUNDING_BIDIRECTED_EDGE")

    if invalid_reasons or open_backdoor_paths:
        reason_codes = list(dict.fromkeys(invalid_reasons + (["INVALID_ADJUSTMENT_SET_BACKDOOR_OPEN"] if open_backdoor_paths else [])))
        return _blocked_result(
            strategy="blocked_invalid_adjustment_set",
            reason=(
                "The supplied adjustment set was not graphically valid for this DAG. "
                "Amantia will not promote an LLM-supplied adjustment set to a causal identification claim."
            ),
            reason_codes=reason_codes,
            query=query,
            raw={
                "adapter": "simple_graphical_identification",
                "nodes_checked": sorted(nodes),
                "directed_edges": directed,
                "bidirected_edges": bidirected,
                "adjustment_set_supplied": supplied_z,
                "adjustment_set_used": sorted(z),
                "unknown_adjustment_variables": unknown_z,
                "forbidden_adjustment_variables": forbidden_z,
                "post_treatment_adjustment_variables": post_treatment_z,
                "backdoor_paths_found": [_format_path(p) for p in backdoor_paths],
                "open_backdoor_paths": [_format_path(p) for p in open_backdoor_paths],
                "full_id_claim_allowed": 0,
            },
        )

    if z:
        z_text = ",".join(sorted(z))
        formula = f"sum_{{{z_text}}} P({y} | {x},{z_text}) P({z_text})"
        strategy = "validated_backdoor_adjustment"
        reason_codes = ["SCM_ID_IDENTIFIED", "VALID_BACKDOOR_ADJUSTMENT_SET"]
    else:
        formula = f"P({y} | {x})"
        strategy = "validated_no_open_backdoor_path"
        reason_codes = ["SCM_ID_IDENTIFIED", "NO_OPEN_BACKDOOR_PATH"]

    return IdentificationResult(
        identified=True,
        treatment=x,
        outcome=y,
        treatments=[x],
        outcomes=[y],
        conditions=list(query.conditions),
        adjustment_set=list(query.adjustment_set),
        mediators=list(query.mediators),
        identification_strategy=strategy,
        identification_tier="identified_graphical",
        estimand=formula,
        formula=formula,
        authority_status="validated_graphical_adapter",
        reason="The online graphical adapter accepted the query only after validating that the adjustment set blocks all open backdoor paths.",
        reason_codes=reason_codes,
        raw_id_result={
            "adapter": "simple_graphical_identification",
            "nodes_checked": sorted(nodes),
            "directed_edges": directed,
            "bidirected_edges": bidirected,
            "adjustment_set_supplied": supplied_z,
            "adjustment_set_used": sorted(z),
            "backdoor_paths_found": [_format_path(p) for p in backdoor_paths],
            "open_backdoor_paths": [],
            "full_id_claim_allowed": 0,
        },
        query_id=query.query_id,
        source=query.source,
    )


class IdentificationEngine:
    """Stable adapter object used by DecisionGate, MCP and scientific veto code."""

    def identify(self, payload: Mapping[str, Any] | IdentificationQuery) -> IdentificationResult:
        query = payload if isinstance(payload, IdentificationQuery) else IdentificationQuery.from_payload(payload)
        if not _has_graph(query.scm_graph):
            return _blocked_result(
                strategy="blocked_missing_graph",
                reason="SCM graph is required for identification.",
                reason_codes=["MISSING_SCM_GRAPH"],
                query=query,
            )
        if not query.treatments or not query.outcomes:
            return _blocked_result(
                strategy="blocked_missing_query",
                reason="Treatment and outcome are required for identification.",
                reason_codes=["MISSING_TREATMENT_OR_OUTCOME"],
                query=query,
            )

        # Step 90: validate online LLM-supplied adjustment sets before any
        # backend fallback. A non-empty adjustment_set is not evidence by
        # itself; it must graphically block all open backdoor paths.
        if (query.adjustment_set or query.mediators) and len(query.treatments) == 1 and len(query.outcomes) == 1:
            return _simple_graphical_identification(query)

        # Direct bidirected treatment-outcome confounding is safety-critical
        # for scientific claims, so keep it in the conservative online path
        # instead of allowing a weaker backend interpretation to promote it.
        if len(query.treatments) == 1 and len(query.outcomes) == 1:
            _nodes, _directed, _bidirected = _normalized_graph_components(query.scm_graph)
            if any({a, b} == {query.treatment, query.outcome} for a, b in _bidirected):
                return _simple_graphical_identification(query)

        try:
            from scm_parts.admg import admg_from_scm_graph
            from scm_parts.id_full import full_id, identify_conditional_effect

            admg = admg_from_scm_graph(query.scm_graph)

            if query.conditions:
                return _map_conditional_id(
                    query,
                    identify_conditional_effect(admg, query.treatments, query.outcomes, query.conditions, max_depth=query.max_depth),
                )

            return _map_full_id(query, full_id(admg, query.treatments, query.outcomes, max_depth=query.max_depth))
        except Exception as exc:  # pragma: no cover - defensive safety boundary
            # If the optional Full-ID backend is not packaged, keep the online
            # runtime useful with the stricter Step 90 backdoor validator.  For
            # other backend failures, fail closed instead of overclaiming.
            if isinstance(exc, (ImportError, ModuleNotFoundError)) and len(query.treatments) == 1 and len(query.outcomes) == 1:
                return _simple_graphical_identification(query)
            return _blocked_result(
                strategy="adapter_runtime_error",
                reason=f"SCM-ID adapter failed safely: {type(exc).__name__}: {exc}",
                reason_codes=["SCM_ID_ADAPTER_ERROR"],
                query=query,
                raw={"error_type": type(exc).__name__, "error_message": str(exc)},
            )

    def identify_many(self, payload: Mapping[str, Any]) -> List[IdentificationResult]:
        payload = dict(payload or {}) if isinstance(payload, Mapping) else {}
        graph = payload.get("scm_graph") or payload.get("graph") or payload.get("causal_graph") or {}
        graph = normalize_scm_graph(graph if isinstance(graph, Mapping) else {})
        queries = graph.get("queries") or payload.get("queries") or []
        if not queries:
            queries = [payload]
        results: List[IdentificationResult] = []
        for idx, row in enumerate(queries):
            row = dict(row or {}) if isinstance(row, Mapping) else {}
            row.setdefault("scm_graph", graph)
            row.setdefault("query_id", row.get("id") or f"q{idx + 1}")
            results.append(self.identify(row))
        return results

    def identify_action_effect(self, action: Mapping[str, Any], graph: Mapping[str, Any]) -> IdentificationResult:
        action = dict(action or {}) if isinstance(action, Mapping) else {}
        query = {
            "scm_graph": graph,
            "treatment": action.get("treatment") or action.get("action_name") or action.get("candidate_action"),
            "outcome": action.get("outcome") or action.get("protected_outcome") or action.get("intended_outcome"),
            "conditions": action.get("conditions") or action.get("condition_set"),
            "adjustment_set": action.get("adjustment_set"),
            "query_id": action.get("query_id") or action.get("id"),
            "source": "identify_action_effect",
        }
        return self.identify(query)


def identify_effect(payload: Mapping[str, Any]) -> Dict[str, Any]:
    return IdentificationEngine().identify(payload).to_dict()


def identify_many(payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    return [result.to_dict() for result in IdentificationEngine().identify_many(payload)]


__all__ = [
    "IdentificationEngine",
    "IdentificationQuery",
    "IdentificationResult",
    "identify_effect",
    "identify_many",
    "normalize_scm_graph",
]
