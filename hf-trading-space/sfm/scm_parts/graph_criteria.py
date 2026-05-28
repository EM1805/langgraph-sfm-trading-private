from __future__ import annotations

"""Shared graph criteria for SCM/ID diagnostics.

This module is the single graph-only home for reusable SCM criteria:

- directed path search and topological checks on ADMG directed parts;
- latent-expanded d-separation for ADMGs with bidirected edges;
- map-based backdoor/adjustment helpers kept for legacy compatibility.

It intentionally performs no file I/O, imports no pandas/numpy, and does not
authorize causal claims.  Causal authority remains in ``scm_parts.id_algorithm``
and the downstream causal contract.
"""

from collections import deque
from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from .admg import ADMG

DirectedEdge = Tuple[str, str]


def _s(value: object) -> str:
    raw = "" if value is None else str(value).strip()
    return "" if raw.lower() in {"nan", "none", "null"} else raw


def dedupe(values: Iterable[object]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for value in values or []:
        item = _s(value)
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _format_path(path: Sequence[str]) -> str:
    return "->".join(path)


def _format_paths(paths: Sequence[Sequence[str]], *, limit: int = 8) -> str:
    shown = [_format_path(p) for p in list(paths)[:limit]]
    if len(paths) > limit:
        shown.append(f"...(+{len(paths) - limit})")
    return "|".join(shown)


# ---------------------------------------------------------------------------
# ADMG directed-graph helpers used by ID/front-door/truncated factorization.
# ---------------------------------------------------------------------------


def directed_path_exists(
    admg: ADMG,
    source: str,
    target: str,
    *,
    forbidden: Optional[Iterable[str]] = None,
) -> bool:
    source = _s(source)
    target = _s(target)
    if not source or not target or source not in admg.node_set or target not in admg.node_set:
        return False
    banned = set(dedupe(forbidden or []))
    children = admg.children()
    stack = [source]
    seen = {source}
    while stack:
        cur = stack.pop()
        for nxt in children.get(cur, set()):
            if nxt in banned and nxt != target:
                continue
            if nxt == target:
                return True
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return False


def directed_paths(admg: ADMG, source: str, target: str, *, max_paths: int = 256) -> List[List[str]]:
    """Enumerate simple directed paths with a hard cap for audit safety."""
    source = _s(source)
    target = _s(target)
    if not source or not target or source not in admg.node_set or target not in admg.node_set:
        return []
    children = admg.children()
    out: List[List[str]] = []
    stack: List[Tuple[str, List[str]]] = [(source, [source])]
    while stack and len(out) < max_paths:
        cur, path = stack.pop()
        for nxt in sorted(children.get(cur, set()), reverse=True):
            if nxt in path:
                continue
            new_path = path + [nxt]
            if nxt == target:
                out.append(new_path)
                if len(out) >= max_paths:
                    break
            else:
                stack.append((nxt, new_path))
    return out


def nodes_on_directed_paths(admg: ADMG, source: str, target: str) -> List[str]:
    """Return observed nodes lying on any directed path source -> ... -> target."""
    source = _s(source)
    target = _s(target)
    if not source or not target or source not in admg.node_set or target not in admg.node_set:
        return []
    desc = admg.descendants([source])
    anc_y = admg.ancestors([target])
    candidates = sorted((desc & anc_y) - {source, target})
    out: List[str] = []
    for z in candidates:
        if directed_path_exists(admg, source, z) and directed_path_exists(admg, z, target):
            out.append(z)
    return out


def has_bidirected_incident(admg: ADMG, node: str) -> bool:
    node = _s(node)
    return any(node in edge for edge in admg.bidirected_edges)


def same_district(admg: ADMG, a: str, b: str) -> bool:
    a = _s(a)
    b = _s(b)
    if not a or not b:
        return False
    for district in admg.districts():
        if a in district and b in district:
            return True
    return False


def directed_cycle_nodes(admg: ADMG) -> List[str]:
    """Return nodes participating in directed cycles, if any."""
    children = admg.children()
    visiting: Set[str] = set()
    visited: Set[str] = set()
    cycle_nodes: Set[str] = set()
    stack: List[str] = []

    def dfs(node: str) -> None:
        visiting.add(node)
        stack.append(node)
        for nxt in children.get(node, set()):
            if nxt not in visited and nxt not in visiting:
                dfs(nxt)
            elif nxt in visiting:
                try:
                    idx = stack.index(nxt)
                    cycle_nodes.update(stack[idx:])
                except ValueError:
                    cycle_nodes.add(nxt)
                cycle_nodes.add(node)
        stack.pop()
        visiting.discard(node)
        visited.add(node)

    for node in sorted(admg.node_set):
        if node not in visited:
            dfs(node)
    return sorted(cycle_nodes)


def topological_order(admg: ADMG) -> List[str]:
    """Return a deterministic topological order for the directed part.

    Empty list means the directed component is cyclic.
    """
    nodes = sorted(admg.node_set)
    children = admg.children()
    parents = admg.parents()
    indegree: Dict[str, int] = {n: len(parents.get(n, set())) for n in nodes}
    ready = [n for n in nodes if indegree.get(n, 0) == 0]
    order: List[str] = []
    while ready:
        cur = ready.pop(0)
        order.append(cur)
        for child in sorted(children.get(cur, set())):
            indegree[child] = indegree.get(child, 0) - 1
            if indegree[child] == 0:
                ready.append(child)
                ready.sort()
    return order if len(order) == len(nodes) else []


# ---------------------------------------------------------------------------
# Latent-expanded ADMG d-separation.
# ---------------------------------------------------------------------------


def _latent_name(a: str, b: str) -> str:
    x, y = sorted((_s(a), _s(b)))
    return f"U__{x}__{y}"


def _expanded_dag_edges(admg: ADMG) -> Tuple[Set[str], List[DirectedEdge]]:
    nodes: Set[str] = set(admg.nodes)
    edges: List[DirectedEdge] = list(admg.directed_edges)
    for a, b in admg.bidirected_edges:
        u = _latent_name(a, b)
        nodes.add(u)
        edges.append((u, a))
        edges.append((u, b))
    return nodes, edges


def _parents_from_edges(nodes: Iterable[str], edges: Iterable[DirectedEdge]) -> Dict[str, Set[str]]:
    g: Dict[str, Set[str]] = {n: set() for n in nodes}
    for a, b in edges:
        g.setdefault(b, set()).add(a)
        g.setdefault(a, set())
    return g


def _ancestors_dag(targets: Iterable[str], parents: Mapping[str, Set[str]]) -> Set[str]:
    wanted = {t for t in dedupe(targets) if t in parents}
    seen = set(wanted)
    stack = list(wanted)
    while stack:
        cur = stack.pop()
        for p in parents.get(cur, set()):
            if p not in seen:
                seen.add(p)
                stack.append(p)
    return seen


def _remove_outgoing(edges: Iterable[DirectedEdge], sources: Iterable[str]) -> List[DirectedEdge]:
    blocked = set(dedupe(sources))
    return [(a, b) for a, b in edges if a not in blocked]


def _all_simple_undirected_paths(
    nodes: Iterable[str],
    edges: Iterable[DirectedEdge],
    source: str,
    target: str,
    *,
    max_paths: int = 512,
) -> List[List[str]]:
    source = _s(source)
    target = _s(target)
    node_set = set(nodes)
    if source not in node_set or target not in node_set:
        return []
    adj: Dict[str, Set[str]] = {n: set() for n in node_set}
    for a, b in edges:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    out: List[List[str]] = []
    stack: List[Tuple[str, List[str]]] = [(source, [source])]
    while stack and len(out) < max_paths:
        cur, path = stack.pop()
        for nxt in sorted(adj.get(cur, set()), reverse=True):
            if nxt in path:
                continue
            new_path = path + [nxt]
            if nxt == target:
                out.append(new_path)
                if len(out) >= max_paths:
                    break
            else:
                stack.append((nxt, new_path))
    return out


def _is_arrow_into(node: str, frm: str, directed_edge_set: Set[DirectedEdge]) -> bool:
    return (frm, node) in directed_edge_set


def _path_active(
    path: Sequence[str],
    directed_edge_set: Set[DirectedEdge],
    conditioned: Set[str],
    conditioned_anc: Set[str],
) -> bool:
    """Return whether a DAG path is active under standard d-separation."""
    if len(path) <= 2:
        return True
    for i in range(1, len(path) - 1):
        prev_node = path[i - 1]
        mid = path[i]
        next_node = path[i + 1]
        collider = _is_arrow_into(mid, prev_node, directed_edge_set) and _is_arrow_into(mid, next_node, directed_edge_set)
        if collider:
            if mid not in conditioned_anc:
                return False
        else:
            if mid in conditioned:
                return False
    return True


@dataclass(frozen=True)
class DSeparationDiagnostic:
    """Audit result for d-separation on the latent-expanded ADMG DAG."""

    separated: bool
    open_path_count: int
    checked_path_count: int
    conditioned_on: str = ""
    removed_outgoing_from: str = ""
    open_paths: str = ""
    status: str = "checked"
    reason_codes: str = ""

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def d_separation_diagnostic(
    admg: ADMG,
    source: str,
    target: str,
    *,
    conditioned_on: Optional[Sequence[str]] = None,
    remove_outgoing_from: Optional[Sequence[str]] = None,
    max_paths: int = 512,
) -> DSeparationDiagnostic:
    """Check d-separation after expanding ADMG bidirected edges into latents."""
    src = _s(source)
    dst = _s(target)
    nodes, edges0 = _expanded_dag_edges(admg)
    if not src or not dst or src not in nodes or dst not in nodes:
        return DSeparationDiagnostic(
            separated=False,
            open_path_count=0,
            checked_path_count=0,
            status="invalid_query",
            reason_codes="MISSING_QUERY_NODE",
        )
    conditioned = {z for z in dedupe(conditioned_on or []) if z in nodes and z not in {src, dst}}
    removed = {z for z in dedupe(remove_outgoing_from or []) if z in nodes}
    edges = _remove_outgoing(edges0, removed)
    parents = _parents_from_edges(nodes, edges)
    conditioned_anc = _ancestors_dag(conditioned, parents) | conditioned
    paths = _all_simple_undirected_paths(nodes, edges, src, dst, max_paths=max_paths)
    directed_edge_set = set(edges)
    open_paths: List[List[str]] = []
    for path in paths:
        if _path_active(path, directed_edge_set, conditioned, conditioned_anc):
            open_paths.append(path)
    return DSeparationDiagnostic(
        separated=len(open_paths) == 0,
        open_path_count=len(open_paths),
        checked_path_count=len(paths),
        conditioned_on="|".join(sorted(conditioned)),
        removed_outgoing_from="|".join(sorted(removed)),
        open_paths=_format_paths(open_paths),
        status="separated" if len(open_paths) == 0 else "not_separated",
        reason_codes="D_SEPARATED" if len(open_paths) == 0 else "OPEN_DCONNECTING_PATHS",
    )


# ---------------------------------------------------------------------------
# Legacy map-based backdoor helpers used by scm_parts.identifier.
# ---------------------------------------------------------------------------


def descendants_from_child_map(start: str, children: Dict[str, Set[str]]) -> Set[str]:
    seen: Set[str] = set()
    dq = deque([start])
    while dq:
        cur = dq.popleft()
        for nxt in children.get(cur, set()):
            if nxt not in seen:
                seen.add(nxt)
                dq.append(nxt)
    seen.discard(start)
    return seen


def ancestors_from_parent_map(start: str, parents: Dict[str, Set[str]]) -> Set[str]:
    seen: Set[str] = set()
    dq = deque([start])
    while dq:
        cur = dq.popleft()
        for nxt in parents.get(cur, set()):
            if nxt not in seen:
                seen.add(nxt)
                dq.append(nxt)
    seen.discard(start)
    return seen


def simple_paths(graph: Dict[str, Set[str]], src: str, dst: str, cutoff: int = 5) -> List[List[str]]:
    if src == dst:
        return [[src]]
    out: List[List[str]] = []
    stack: List[Tuple[str, List[str]]] = [(src, [src])]
    while stack:
        node, path = stack.pop()
        if len(path) > cutoff + 1:
            continue
        for nxt in graph.get(node, set()):
            if nxt in path:
                continue
            nxt_path = path + [nxt]
            if nxt == dst:
                out.append(nxt_path)
            else:
                stack.append((nxt, nxt_path))
    return out


def path_is_backdoor(path: List[str], treatment: str, parents: Dict[str, Set[str]]) -> bool:
    return len(path) >= 2 and path[0] == treatment and path[1] in parents.get(treatment, set())


def is_collider(prev_node: str, node: str, next_node: str, parents: Dict[str, Set[str]]) -> bool:
    return prev_node in parents.get(node, set()) and next_node in parents.get(node, set())


def descendant_map(children: Dict[str, Set[str]], nodes: Iterable[str]) -> Dict[str, Set[str]]:
    return {n: descendants_from_child_map(n, children) for n in nodes}


def path_open_given(
    path: List[str],
    conditioned: Set[str],
    parents: Dict[str, Set[str]],
    descendant_lookup: Dict[str, Set[str]],
) -> bool:
    """Conservative d-separation path-open check for simple map paths."""
    if len(path) <= 2:
        return True
    for i in range(1, len(path) - 1):
        prev_node, node, next_node = path[i - 1], path[i], path[i + 1]
        collider = is_collider(prev_node, node, next_node, parents)
        if collider:
            activated = node in conditioned or bool(descendant_lookup.get(node, set()).intersection(conditioned))
            if not activated:
                return False
        else:
            if node in conditioned:
                return False
    return True


def valid_adjustment_sets_for_backdoor(
    treatment: str,
    outcome: str,
    candidate_controls: List[str],
    forbidden: List[str],
    parents: Dict[str, Set[str]],
    children: Dict[str, Set[str]],
    undirected: Dict[str, Set[str]],
    max_set_size: int = 3,
) -> Dict[str, object]:
    """Return minimal observed adjustment sets that close backdoor paths."""
    forbidden_set = set(forbidden) | {treatment, outcome}
    candidates = [c for c in dedupe(candidate_controls) if c not in forbidden_set]
    backdoor_paths = [p for p in simple_paths(undirected, treatment, outcome, cutoff=6) if path_is_backdoor(p, treatment, parents)]
    path_nodes = set().union(*[set(p) for p in backdoor_paths]) if backdoor_paths else set()
    descendant_lookup = descendant_map(children, path_nodes)

    if not backdoor_paths:
        return {
            "valid_sets": [[]],
            "minimal_sets": [[]],
            "adjustment_set_status": "valid_empty",
            "all_backdoor_paths": [],
            "unblocked_under_empty": [],
            "max_set_size": max_set_size,
            "search_space": candidates,
        }

    valid_sets: List[List[str]] = []
    for k in range(0, min(max_set_size, len(candidates)) + 1):
        for combo in combinations(candidates, k):
            cond = set(combo)
            open_paths = ["->".join(p) for p in backdoor_paths if path_open_given(p, cond, parents, descendant_lookup)]
            if not open_paths:
                valid_sets.append(list(combo))
        if valid_sets:
            break

    empty_open = ["->".join(p) for p in backdoor_paths if path_open_given(p, set(), parents, descendant_lookup)]
    if valid_sets:
        status = "valid_empty" if valid_sets[0] == [] else "valid_nonempty"
    else:
        status = "missing"
    return {
        "valid_sets": valid_sets,
        "minimal_sets": valid_sets[:],
        "adjustment_set_status": status,
        "all_backdoor_paths": ["->".join(p) for p in backdoor_paths],
        "unblocked_under_empty": empty_open,
        "max_set_size": max_set_size,
        "search_space": candidates,
    }


def dsep_query_report(
    src: str,
    dst: str,
    conditioned: Iterable[str],
    parents: Dict[str, Set[str]],
    children: Dict[str, Set[str]],
    undirected: Dict[str, Set[str]],
    cutoff: int = 6,
) -> Dict[str, object]:
    conditioned_set = set(dedupe(conditioned))
    paths = simple_paths(undirected, src, dst, cutoff=cutoff)
    path_nodes = set().union(*[set(p) for p in paths]) if paths else set()
    descendant_lookup = descendant_map(children, path_nodes)
    open_paths: List[str] = []
    blocked_paths: List[str] = []
    path_reports: List[dict] = []
    for p in paths:
        is_open = path_open_given(p, conditioned_set, parents, descendant_lookup)
        label = "->".join(p)
        if is_open:
            open_paths.append(label)
        else:
            blocked_paths.append(label)
        path_reports.append(
            {
                "path": label,
                "open": bool(is_open),
                "starts_with_backdoor": bool(path_is_backdoor(p, src, parents)),
                "contains_collider": any(is_collider(p[i - 1], p[i], p[i + 1], parents) for i in range(1, len(p) - 1)),
            }
        )
    return {
        "x": src,
        "y": dst,
        "conditioned_on": sorted(conditioned_set),
        "d_separated": len(open_paths) == 0,
        "n_paths_considered": len(paths),
        "open_paths": open_paths,
        "blocked_paths": blocked_paths,
        "path_reports": path_reports,
    }


def candidate_backdoor_controls(
    treatment: str,
    outcome: str,
    parents: Dict[str, Set[str]],
    children: Dict[str, Set[str]],
    undirected: Dict[str, Set[str]],
    node_meta: Dict[str, dict],
    mediators: List[str],
    colliders: List[str],
) -> Tuple[List[str], List[str], Dict[str, object]]:
    """Conservative observed-control candidates for backdoor closure."""
    descendants_t = descendants_from_child_map(treatment, children)
    descendants_m: Set[str] = set()
    for m in mediators:
        descendants_m.update(descendants_from_child_map(m, children))
    collider_set = set(colliders)
    bad_controls = set(descendants_t) | set(mediators) | collider_set | descendants_m | {outcome, treatment}
    observed_anc = (ancestors_from_parent_map(treatment, parents) | ancestors_from_parent_map(outcome, parents)) - bad_controls
    observed_anc = {n for n in observed_anc if bool(node_meta.get(n, {}).get("observed", True))}
    paths = simple_paths(undirected, treatment, outcome, cutoff=5)
    candidate_paths = [p for p in paths if path_is_backdoor(p, treatment, parents)]

    coverage = []
    valid_controls: List[str] = []
    for cand in sorted(observed_anc):
        hit = ["->".join(p) for p in candidate_paths if cand in p[1:-1]]
        if hit:
            valid_controls.append(cand)
            coverage.append({"control": cand, "blocks_paths": hit[:10]})
    report = {
        "n_candidate_paths": len(candidate_paths),
        "candidate_backdoor_paths": ["->".join(p) for p in candidate_paths[:15]],
        "candidate_control_coverage": coverage,
        "bad_controls": sorted(bad_controls),
    }
    return dedupe(valid_controls), sorted(bad_controls), report


# Compatibility aliases with the previous backdoor_identifier names.
descendants = descendants_from_child_map
ancestors = ancestors_from_parent_map


__all__ = [
    "DSeparationDiagnostic",
    "ancestors",
    "ancestors_from_parent_map",
    "candidate_backdoor_controls",
    "d_separation_diagnostic",
    "dedupe",
    "descendant_map",
    "descendants",
    "descendants_from_child_map",
    "directed_cycle_nodes",
    "directed_path_exists",
    "directed_paths",
    "dsep_query_report",
    "has_bidirected_incident",
    "is_collider",
    "nodes_on_directed_paths",
    "path_is_backdoor",
    "path_open_given",
    "same_district",
    "simple_paths",
    "topological_order",
    "valid_adjustment_sets_for_backdoor",
]
