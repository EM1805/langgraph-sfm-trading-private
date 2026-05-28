from __future__ import annotations

"""ADMG / c-component utilities for future do-calculus ID support.

This module is intentionally conservative: it does not claim to implement the
Pearl/Shpitser ID algorithm.  It provides the graph representation that the ID
algorithm needs next:

- directed edges: X -> Y
- bidirected edges: X <-> Y, representing possible latent confounding
- districts / c-components: connected components under bidirected edges
- induced and ancestral subgraphs

If no bidirected edges are present, every observed node is its own district.
"""

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

DirectedEdge = Tuple[str, str]
BidirectedEdge = Tuple[str, str]
Graph = Dict[str, Set[str]]


def _s(x: object) -> str:
    return "" if x is None else str(x).strip()


def _dedupe(xs: Iterable[object]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for x in xs:
        v = _s(x)
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _edge_pair(a: object, b: object) -> Optional[Tuple[str, str]]:
    x = _s(a)
    y = _s(b)
    if not x or not y or x == y:
        return None
    return (x, y)


def _bidirected_pair(a: object, b: object) -> Optional[Tuple[str, str]]:
    pair = _edge_pair(a, b)
    if pair is None:
        return None
    x, y = pair
    return tuple(sorted((x, y)))  # canonical undirected/bidirected key


def _split_pair_field(value: object) -> Optional[Tuple[str, str]]:
    raw = _s(value)
    if not raw:
        return None
    for sep in ["<->", "↔", "--", "|", ",", ";"]:
        if sep in raw:
            parts = [p.strip() for p in raw.split(sep) if p.strip()]
            if len(parts) == 2:
                return _bidirected_pair(parts[0], parts[1])
    return None


@dataclass(frozen=True)
class ADMG:
    """A small acyclic-directed-mixed-graph container.

    The class does not enforce acyclicity because Discovery/SCM candidates can
    still contain cycles.  Downstream ID code should check acyclicity before
    claiming a formal do-calculus result.
    """

    nodes: Tuple[str, ...]
    directed_edges: Tuple[DirectedEdge, ...]
    bidirected_edges: Tuple[BidirectedEdge, ...]

    @property
    def node_set(self) -> Set[str]:
        return set(self.nodes)

    def children(self) -> Graph:
        g: Graph = {n: set() for n in self.nodes}
        for a, b in self.directed_edges:
            g.setdefault(a, set()).add(b)
            g.setdefault(b, set())
        return g

    def parents(self) -> Graph:
        g: Graph = {n: set() for n in self.nodes}
        for a, b in self.directed_edges:
            g.setdefault(b, set()).add(a)
            g.setdefault(a, set())
        return g

    def siblings(self) -> Graph:
        g: Graph = {n: set() for n in self.nodes}
        for a, b in self.bidirected_edges:
            g.setdefault(a, set()).add(b)
            g.setdefault(b, set()).add(a)
        return g

    def ancestors(self, targets: Iterable[str]) -> Set[str]:
        parents = self.parents()
        wanted = {t for t in (_s(x) for x in targets) if t in self.node_set}
        seen: Set[str] = set(wanted)
        q = deque(wanted)
        while q:
            cur = q.popleft()
            for p in parents.get(cur, set()):
                if p not in seen:
                    seen.add(p)
                    q.append(p)
        return seen

    def descendants(self, sources: Iterable[str]) -> Set[str]:
        children = self.children()
        wanted = {s for s in (_s(x) for x in sources) if s in self.node_set}
        seen: Set[str] = set(wanted)
        q = deque(wanted)
        while q:
            cur = q.popleft()
            for c in children.get(cur, set()):
                if c not in seen:
                    seen.add(c)
                    q.append(c)
        return seen

    def districts(self, subset: Optional[Iterable[str]] = None) -> List[List[str]]:
        """Return c-components/districts under bidirected connectivity."""
        allowed = self.node_set if subset is None else {n for n in (_s(x) for x in subset) if n in self.node_set}
        sib = self.siblings()
        out: List[List[str]] = []
        seen: Set[str] = set()
        for n in sorted(allowed):
            if n in seen:
                continue
            comp: Set[str] = set()
            q = deque([n])
            seen.add(n)
            while q:
                cur = q.popleft()
                comp.add(cur)
                for nxt in sib.get(cur, set()):
                    if nxt in allowed and nxt not in seen:
                        seen.add(nxt)
                        q.append(nxt)
            out.append(sorted(comp))
        return out

    def induced_subgraph(self, subset: Iterable[str]) -> "ADMG":
        keep = {n for n in (_s(x) for x in subset) if n in self.node_set}
        return ADMG(
            nodes=tuple(sorted(keep)),
            directed_edges=tuple((a, b) for a, b in self.directed_edges if a in keep and b in keep),
            bidirected_edges=tuple((a, b) for a, b in self.bidirected_edges if a in keep and b in keep),
        )

    def ancestral_subgraph(self, targets: Iterable[str]) -> "ADMG":
        return self.induced_subgraph(self.ancestors(targets))


def admg_from_edges(nodes: Iterable[object], directed_edges: Iterable[Tuple[object, object]], bidirected_edges: Iterable[Tuple[object, object]] = ()) -> ADMG:
    node_list = _dedupe(nodes)
    node_set: Set[str] = set(node_list)
    de: List[DirectedEdge] = []
    be: List[BidirectedEdge] = []
    seen_de: Set[DirectedEdge] = set()
    seen_be: Set[BidirectedEdge] = set()

    for a, b in directed_edges:
        pair = _edge_pair(a, b)
        if pair is None:
            continue
        x, y = pair
        node_set.add(x); node_set.add(y)
        if pair not in seen_de:
            seen_de.add(pair)
            de.append(pair)

    for a, b in bidirected_edges:
        pair = _bidirected_pair(a, b)
        if pair is None:
            continue
        x, y = pair
        node_set.add(x); node_set.add(y)
        if pair not in seen_be:
            seen_be.add(pair)
            be.append(pair)

    return ADMG(nodes=tuple(sorted(node_set)), directed_edges=tuple(de), bidirected_edges=tuple(be))


def bidirected_edges_from_rows(rows: Iterable[Mapping[str, object]]) -> List[BidirectedEdge]:
    """Extract explicit latent-confounding / bidirected pairs from row metadata.

    Supported row shapes are intentionally simple and explicit.  We do *not*
    infer latent confounding merely because a row is risky; that would overclaim.
    """
    out: List[BidirectedEdge] = []
    seen: Set[BidirectedEdge] = set()
    for row in rows or []:
        if not isinstance(row, Mapping):
            continue
        pair = None
        for key in [
            "bidirected_pair", "latent_confounding_pair", "unobserved_confounding_pair",
            "latent_pair", "confounded_pair",
        ]:
            pair = _split_pair_field(row.get(key, ""))
            if pair:
                break
        if pair is None:
            a = row.get("bidirected_source", row.get("latent_source", row.get("confounded_source", "")))
            b = row.get("bidirected_target", row.get("latent_target", row.get("confounded_target", "")))
            pair = _bidirected_pair(a, b)
        if pair is None:
            kind = _s(row.get("edge_kind", row.get("edge_type", ""))).lower()
            if kind in {"bidirected", "latent_confounding", "unobserved_confounding"}:
                pair = _bidirected_pair(row.get("source", ""), row.get("target", ""))
        if pair and pair not in seen:
            seen.add(pair)
            out.append(pair)
    return out


def admg_from_scm_graph(scm_graph: Mapping[str, object]) -> ADMG:
    node_ids: List[str] = []
    for n in scm_graph.get("nodes", []) if isinstance(scm_graph, Mapping) else []:
        if isinstance(n, Mapping):
            node_ids.append(_s(n.get("node_id") or n.get("id") or n.get("name")))
        else:
            node_ids.append(_s(n))
    edge_rows = [e for e in scm_graph.get("edges", []) if isinstance(e, Mapping)] if isinstance(scm_graph, Mapping) else []
    directed: List[DirectedEdge] = []
    for e in edge_rows:
        kind = _s(e.get("edge_kind", e.get("edge_type", ""))).lower()
        if kind in {"bidirected", "latent_confounding", "unobserved_confounding", "exogenous_noise", "noise_input", "latent_noise"}:
            continue
        pair = _edge_pair(e.get("source", e.get("from", "")), e.get("target", e.get("to", "")))
        if pair:
            directed.append(pair)
    bidirected = bidirected_edges_from_rows(edge_rows)
    return admg_from_edges(node_ids, directed, bidirected)


def c_component_rows(admg: ADMG) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    districts = admg.districts()
    for idx, comp in enumerate(districts, start=1):
        rows.append({
            "component_id": f"C{idx}",
            "component_nodes": "|".join(comp),
            "component_size": len(comp),
            "has_latent_confounding": int(len(comp) > 1),
            "district_semantics": "bidirected_connected_c_component" if len(comp) > 1 else "singleton_observed_component",
        })
    return rows


def admg_summary(admg: ADMG) -> Dict[str, object]:
    districts = admg.districts()
    nontrivial = [d for d in districts if len(d) > 1]
    return {
        "admg_version": 1,
        "graph_type": "ADMG_support_lite",
        "n_nodes": len(admg.nodes),
        "n_directed_edges": len(admg.directed_edges),
        "n_bidirected_edges": len(admg.bidirected_edges),
        "n_c_components": len(districts),
        "n_nontrivial_c_components": len(nontrivial),
        "id_algorithm_status": "limited_id_algorithm_scaffold_available",
        "semantics": "bidirected_edges_represent_explicit_latent_confounding_candidates_only",
    }


def admg_report_from_scm_graph(scm_graph: Mapping[str, object]) -> Tuple[ADMG, Dict[str, object], List[Dict[str, object]]]:
    admg = admg_from_scm_graph(scm_graph)
    return admg, admg_summary(admg), c_component_rows(admg)


__all__ = [
    "ADMG", "DirectedEdge", "BidirectedEdge", "admg_from_edges", "admg_from_scm_graph",
    "admg_report_from_scm_graph", "admg_summary", "bidirected_edges_from_rows",
    "c_component_rows",
]
