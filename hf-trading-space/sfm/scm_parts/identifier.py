from __future__ import annotations

from runtime_env import configure_scientific_runtime
configure_scientific_runtime()


"""Legacy SCM identification reporting layer.

This module is intentionally not the causal-identification authority.
The authoritative ID decision lives in scm_parts.id_algorithm; shared
graph criteria live in scm_parts.graph_criteria. This file remains as a
compatibility/reporting layer because older contract and output code still
reads out/identification/identified_effects.csv and adjustment_sets.csv.

Do not add new identification logic here. Add graph predicates to
graph_criteria.py and ID decisions/proof traces to id_algorithm.py.
"""

import json
import os
import warnings
from collections import defaultdict, deque
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Set, Tuple

from runtime_compat import assert_scientific_stack
assert_scientific_stack()

import pandas as pd

from scm_parts.identification_legacy import build_identification
from scm_parts import graph_criteria as _gc
try:
    from scm_parts.id_algorithm import identify_effect_from_scm_graph as _authoritative_identify_effect
except Exception:  # pragma: no cover - keep legacy writer importable in minimal environments
    _authoritative_identify_effect = None

ID_DIRNAME = "identification"


def _as_str(x) -> str:
    return "" if x is None else str(x)



def _first_present(row, names, default=""):
    """Return first non-empty value from a pandas row/dict-like object."""
    for name in names:
        try:
            val = row.get(name, "")
        except (TypeError, ValueError, AttributeError):
            val = ""
        if val is not None and str(val).strip() != "":
            return val
    return default


def _normalize_effect_contract(df):
    """Normalize Discovery/bridge effect columns for SCM identification."""
    try:
        df = df.copy()
    except (TypeError, ValueError, AttributeError):
        return df
    if getattr(df, "empty", True):
        return df
    if "treatment_col" not in df.columns:
        for c in ["source", "treatment", "cause", "action_col", "action", "from"]:
            if c in df.columns:
                df["treatment_col"] = df[c]
                break
    if "outcome_col" not in df.columns:
        for c in ["target", "outcome", "harm_event", "target_col", "effect", "to"]:
            if c in df.columns:
                df["outcome_col"] = df[c]
                break
    if "source" not in df.columns and "treatment_col" in df.columns:
        df["source"] = df["treatment_col"]
    if "target" not in df.columns and "outcome_col" in df.columns:
        df["target"] = df["outcome_col"]
    return df

def _split_pipe(value) -> List[str]:
    raw = _as_str(value).strip()
    if not raw:
        return []
    return [p.strip() for p in raw.split("|") if p.strip()]


def _dedupe(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for item in items:
        item = _as_str(item).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out




def _node_id(node) -> str:
    if isinstance(node, dict):
        return _as_str(node.get("node_id") or node.get("id") or node.get("name")).strip()
    return _as_str(node).strip()


def _alias_map_from_nodes(nodes: List[dict]) -> Dict[str, str]:
    aliases: Dict[str, str] = {}
    for n in nodes:
        node_id = _node_id(n)
        if not node_id:
            continue
        aliases[node_id] = node_id
        aliases[node_id.lower()] = node_id
        raw = n.get("aliases") if isinstance(n, dict) else None
        if isinstance(raw, str):
            parts = [x.strip() for x in raw.replace("|", ",").split(",") if x.strip()]
        elif isinstance(raw, (list, tuple, set)):
            parts = [str(x).strip() for x in raw if str(x).strip()]
        else:
            parts = []
        for a in parts:
            aliases[a] = node_id
            aliases[a.lower()] = node_id
    return aliases


def _resolve_node_name(name: str, aliases: Dict[str, str]) -> str:
    v = _as_str(name).strip()
    return aliases.get(v, aliases.get(v.lower(), v)) if v else v


def _resolve_many_names(names: Optional[Iterable[str]], aliases: Dict[str, str]) -> List[str]:
    return _dedupe(_resolve_node_name(str(n), aliases) for n in (names or []))

def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_graph(edges: List[dict]) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]], Dict[Tuple[str, str], dict], Dict[str, Set[str]]]:
    children: Dict[str, Set[str]] = defaultdict(set)
    parents: Dict[str, Set[str]] = defaultdict(set)
    by_pair: Dict[Tuple[str, str], dict] = {}
    undirected: Dict[str, Set[str]] = defaultdict(set)
    for e in edges:
        s = _as_str(e.get("source", "")).strip()
        t = _as_str(e.get("target", "")).strip()
        if not s or not t:
            continue
        children[s].add(t)
        parents[t].add(s)
        by_pair[(s, t)] = e
        undirected[s].add(t)
        undirected[t].add(s)
    return children, parents, by_pair, undirected


def _descendants(start: str, children: Dict[str, Set[str]]) -> Set[str]:
    return _gc.descendants_from_child_map(start, children)


def _ancestors(start: str, parents: Dict[str, Set[str]]) -> Set[str]:
    return _gc.ancestors_from_parent_map(start, parents)

def _simple_paths(graph: Dict[str, Set[str]], src: str, dst: str, cutoff: int = 5) -> List[List[str]]:
    return _gc.simple_paths(graph, src, dst, cutoff=cutoff)

def _find_simple_mediators(treatment: str, outcome: str, children: Dict[str, Set[str]]) -> List[str]:
    meds: Set[str] = set()
    for mid in children.get(treatment, set()):
        if mid == outcome:
            continue
        if outcome in _descendants(mid, children):
            meds.add(mid)
    return sorted(meds)


def _find_path_mediators(treatment: str, outcome: str, children: Dict[str, Set[str]]) -> List[str]:
    meds: Set[str] = set()
    for path in _directed_paths(children, treatment, outcome, cutoff=8):
        for node in path[1:-1]:
            meds.add(node)
    return sorted(meds)


def _find_colliders(parents: Dict[str, Set[str]]) -> List[str]:
    return sorted([node for node, incoming in parents.items() if len(incoming) >= 2])


def _role_maps(nodes: List[dict]) -> Tuple[Dict[str, dict], Dict[str, str]]:
    node_meta = {_as_str(n.get("node_id", "")).strip(): n for n in nodes if _as_str(n.get("node_id", "")).strip()}
    node_role = {k: _as_str(v.get("node_role", "state")).strip() or "state" for k, v in node_meta.items()}
    return node_meta, node_role


# Compatibility wrappers. The implementation is now centralized in
# scm_parts.graph_criteria, not in this legacy reporting module.
def _path_is_backdoor(path: List[str], treatment: str, parents: Dict[str, Set[str]]) -> bool:
    return _gc.path_is_backdoor(path, treatment, parents)


def _is_collider(prev_node: str, node: str, next_node: str, parents: Dict[str, Set[str]]) -> bool:
    return _gc.is_collider(prev_node, node, next_node, parents)


def _descendant_map(children: Dict[str, Set[str]], nodes: Iterable[str]) -> Dict[str, Set[str]]:
    return _gc.descendant_map(children, nodes)


def _path_open_given(path: List[str], conditioned: Set[str], parents: Dict[str, Set[str]], descendant_map: Dict[str, Set[str]]) -> bool:
    return _gc.path_open_given(path, conditioned, parents, descendant_map)


def _valid_adjustment_sets_for_backdoor(treatment: str, outcome: str, candidate_controls: List[str], forbidden: List[str], parents: Dict[str, Set[str]], children: Dict[str, Set[str]], undirected: Dict[str, Set[str]], max_set_size: int = 3) -> Dict[str, object]:
    return _gc.valid_adjustment_sets_for_backdoor(treatment, outcome, candidate_controls, forbidden, parents, children, undirected, max_set_size=max_set_size)


def _dsep_query_report(src: str, dst: str, conditioned: Iterable[str], parents: Dict[str, Set[str]], children: Dict[str, Set[str]], undirected: Dict[str, Set[str]], cutoff: int = 6) -> Dict[str, object]:
    return _gc.dsep_query_report(src, dst, conditioned, parents, children, undirected, cutoff=cutoff)
def analyze_graph_query(scm_graph: dict, treatment: str, outcome: str, conditioned: Optional[Iterable[str]] = None, max_adjustment_set: int = 3) -> Dict[str, object]:
    nodes = list(scm_graph.get('nodes', []))
    edges = list(scm_graph.get('edges', []))
    aliases = _alias_map_from_nodes(nodes)
    input_treatment = _as_str(treatment).strip()
    input_outcome = _as_str(outcome).strip()
    treatment = _resolve_node_name(input_treatment, aliases)
    outcome = _resolve_node_name(input_outcome, aliases)
    conditioned = _resolve_many_names(conditioned or [], aliases)

    children, parents, _by_pair, undirected = _build_graph(edges)
    node_meta, _node_role = _role_maps(nodes)
    mediators = _find_path_mediators(treatment, outcome, children)
    colliders = _find_colliders(parents)
    candidate_controls, forbidden, backdoor_report = _candidate_backdoor_controls(treatment, outcome, parents, children, undirected, node_meta, mediators, colliders)
    adjustment = _valid_adjustment_sets_for_backdoor(treatment, outcome, candidate_controls, forbidden, parents, children, undirected, max_set_size=max_adjustment_set)
    query = _dsep_query_report(treatment, outcome, conditioned, parents, children, undirected, cutoff=6)
    return {
        'treatment': treatment,
        'outcome': outcome,
        'input_treatment': input_treatment,
        'input_outcome': input_outcome,
        'resolved_treatment': treatment,
        'resolved_outcome': outcome,
        'conditioned_on': conditioned,
        'graph_query': query,
        'candidate_adjustment_controls': candidate_controls,
        'forbidden_adjustments': forbidden,
        'minimal_adjustment_sets': adjustment.get('minimal_sets', []),
        'backdoor_paths': adjustment.get('all_backdoor_paths', []),
        'unblocked_backdoor_paths_without_adjustment': adjustment.get('unblocked_under_empty', []),
        'backdoor_report': backdoor_report,
        'identifiable_via_backdoor': bool(adjustment.get('minimal_sets')),
    }


def _candidate_backdoor_controls(treatment: str, outcome: str, parents: Dict[str, Set[str]], children: Dict[str, Set[str]], undirected: Dict[str, Set[str]], node_meta: Dict[str, dict], mediators: List[str], colliders: List[str]) -> Tuple[List[str], List[str], Dict[str, object]]:
    return _gc.candidate_backdoor_controls(treatment, outcome, parents, children, undirected, node_meta, mediators, colliders)

def _choose_estimand(preferred: str, direct_requested: bool, has_mediator: bool, direct_known: bool) -> Tuple[str, str]:
    pref = _as_str(preferred).strip()
    if pref:
        if pref in {"nde", "natural_direct_effect", "controlled_direct_effect", "direct_effect"}:
            return "direct_effect", "direct"
        if pref in {"ate", "total_effect", "average_treatment_effect"}:
            return "total_effect", "total"
        return pref, "custom"
    if direct_requested and direct_known:
        return "direct_effect", "direct"
    if has_mediator:
        return "total_effect", "total"
    return "average_treatment_effect", "direct_or_total"


def _directed_paths(children: Dict[str, Set[str]], src: str, dst: str, cutoff: int = 6) -> List[List[str]]:
    # Legacy adapter used for reporting-only frontdoor summaries. Canonical
    # graph predicates live in graph_criteria.py; this avoids importing ADMG.
    if src == dst:
        return [[src]]
    out: List[List[str]] = []
    stack: List[Tuple[str, List[str]]] = [(src, [src])]
    while stack:
        node, path = stack.pop()
        if len(path) > cutoff + 1:
            continue
        for nxt in children.get(node, set()):
            if nxt in path:
                continue
            nxt_path = path + [nxt]
            if nxt == dst:
                out.append(nxt_path)
            else:
                stack.append((nxt, nxt_path))
    return out

def _backdoor_paths_between(a: str, b: str, parents: Dict[str, Set[str]], undirected: Dict[str, Set[str]], cutoff: int = 5) -> List[List[str]]:
    return [p for p in _simple_paths(undirected, a, b, cutoff=cutoff) if _path_is_backdoor(p, a, parents)]


def _directed_chain_mediators(path: List[str], observed_mediators: Set[str]) -> List[str]:
    return [n for n in path[1:-1] if n in observed_mediators]


def _chain_segments(treatment: str, outcome: str, mediators: List[str]) -> List[Tuple[str, str]]:
    seq = [treatment] + list(mediators) + [outcome]
    return [(seq[i], seq[i + 1]) for i in range(len(seq) - 1)]


def _segment_blocking_controls(src: str, dst: str, treatment: str, chain_mediators: List[str], parents: Dict[str, Set[str]], children: Dict[str, Set[str]], undirected: Dict[str, Set[str]], node_meta: Dict[str, dict], allow_treatment: bool = False) -> Dict[str, object]:
    descendants_t = _descendants(treatment, children)
    descendants_chain: Set[str] = set()
    for m in chain_mediators:
        descendants_chain.update(_descendants(m, children))
    bad_controls = set(descendants_t) | descendants_chain | set(chain_mediators) | {src, dst}
    if not allow_treatment:
        bad_controls.add(treatment)
    observed_anc = (_ancestors(src, parents) | _ancestors(dst, parents)) - bad_controls
    observed_anc = {n for n in observed_anc if bool(node_meta.get(n, {}).get("observed", True))}
    if allow_treatment:
        observed_anc.add(treatment)

    raw_paths = _backdoor_paths_between(src, dst, parents, undirected, cutoff=6)
    uncovered = []
    coverage = []
    controls = []
    for p in raw_paths:
        interior = set(p[1:-1])
        blockers = sorted(interior.intersection(observed_anc))
        if blockers:
            controls.extend(blockers)
            coverage.append({"path": "->".join(p), "blocking_controls": blockers})
        else:
            uncovered.append("->".join(p))
    return {
        "segment": f"{src}->{dst}",
        "admissible_controls": _dedupe(controls),
        "coverage": coverage[:10],
        "uncovered_backdoor_paths": uncovered[:10],
    }


def _frontdoor_blocking_controls(a: str, b: str, treatment: str, mediator: str, parents: Dict[str, Set[str]], children: Dict[str, Set[str]], undirected: Dict[str, Set[str]], node_meta: Dict[str, dict], allow_treatment: bool = False) -> Dict[str, object]:
    descendants_t = _descendants(treatment, children)
    descendants_m = _descendants(mediator, children)
    bad_controls = set(descendants_t) | set(descendants_m) | {mediator, a, b}
    if not allow_treatment:
        bad_controls.add(treatment)
    observed_anc = (_ancestors(a, parents) | _ancestors(b, parents)) - bad_controls
    observed_anc = {n for n in observed_anc if bool(node_meta.get(n, {}).get("observed", True))}
    if allow_treatment:
        observed_anc.add(treatment)
    raw_paths = _backdoor_paths_between(a, b, parents, undirected, cutoff=5)
    uncovered = []
    coverage = []
    controls = []
    for p in raw_paths:
        interior = set(p[1:-1])
        blockers = sorted(interior.intersection(observed_anc))
        if blockers:
            controls.extend(blockers)
            coverage.append({"path": "->".join(p), "blocking_controls": blockers})
        else:
            uncovered.append("->".join(p))
    return {
        "admissible_controls": _dedupe(controls),
        "coverage": coverage[:10],
        "uncovered_backdoor_paths": uncovered[:10],
    }


def _frontdoor_assessment(treatment: str, outcome: str, mediators: List[str], parents: Dict[str, Set[str]], children: Dict[str, Set[str]], undirected: Dict[str, Set[str]], node_meta: Dict[str, dict]) -> Dict[str, object]:
    observed_mediators = [m for m in mediators if bool(node_meta.get(m, {}).get("observed", True))]
    if not observed_mediators:
        return {
            "candidate": False,
            "identifiable": False,
            "graph_verified": False,
            "reason": "no_observed_mediators",
            "verification_level": "none",
            "covered_paths": [],
            "uncovered_directed_paths": [],
            "path_mediator_witness": {},
            "path_chain_witness": {},
            "segment_reports": {},
            "treatment_mediator_backdoors": {},
            "mediator_outcome_backdoors": {},
            "treatment_mediator_controls": {},
            "mediator_outcome_controls": {},
            "chain_segment_controls": {},
            "has_direct_treatment_outcome_edge": False,
        }

    observed_set = set(observed_mediators)
    directed_paths = _directed_paths(children, treatment, outcome, cutoff=8)
    covered_paths = []
    uncovered_paths = []
    path_mediator_witness = {}
    path_chain_witness = {}
    segment_reports = {}
    chain_segment_controls = {}

    for p in directed_paths:
        key = "->".join(p)
        mids = _directed_chain_mediators(p, observed_set)
        path_mediator_witness[key] = mids
        path_chain_witness[key] = mids
        if not mids:
            uncovered_paths.append(key)
            continue

        segments = _chain_segments(treatment, outcome, mids)
        local_reports = {}
        all_segments_closed = True
        for i, (src, dst) in enumerate(segments):
            allow_t = i > 0
            seg_report = _segment_blocking_controls(src, dst, treatment, mids, parents, children, undirected, node_meta, allow_treatment=allow_t)
            local_reports[f"{src}->{dst}"] = seg_report
            chain_segment_controls[f"{key}::{src}->{dst}"] = seg_report.get("admissible_controls", [])
            if seg_report.get("uncovered_backdoor_paths"):
                all_segments_closed = False
        segment_reports[key] = local_reports
        if all_segments_closed:
            covered_paths.append(key)
        else:
            uncovered_paths.append(key)

    tm_reports = {}
    my_reports = {}
    for m in observed_mediators:
        tm_reports[m] = _frontdoor_blocking_controls(treatment, m, treatment, m, parents, children, undirected, node_meta, allow_treatment=False)
        my_reports[m] = _frontdoor_blocking_controls(m, outcome, treatment, m, parents, children, undirected, node_meta, allow_treatment=True)

    tm_backdoors = {m: r.get("uncovered_backdoor_paths", []) for m, r in tm_reports.items()}
    my_backdoors = {m: r.get("uncovered_backdoor_paths", []) for m, r in my_reports.items()}
    tm_controls = {m: r.get("admissible_controls", []) for m, r in tm_reports.items()}
    my_controls = {m: r.get("admissible_controls", []) for m, r in my_reports.items()}

    candidate = bool(covered_paths) and not uncovered_paths
    graph_verified = candidate and all(len(v) == 0 for v in tm_backdoors.values()) and all(len(v) == 0 for v in my_backdoors.values())
    has_direct_edge = outcome in children.get(treatment, set())
    complete = graph_verified and (not has_direct_edge)
    verification_level = "candidate_only"
    if graph_verified:
        verification_level = "graph_verified"
    if complete:
        verification_level = "frontdoor_chain_complete_lite"
    # Limited Step150 policy: when the classical front-door checks pass in this
    # observed graph (complete=True), the route becomes estimator-ready for the
    # dedicated limited front-door estimator. This is still not a general ID
    # algorithm and remains explicitly labelled as limited.
    return {
        "candidate": bool(candidate),
        "identifiable": bool(complete),
        "graphical_lite_identifiable": bool(graph_verified),
        "estimator_ready": bool(complete),
        "identification_authority": "frontdoor_limited_estimator_ready" if complete else "frontdoor_graphical_lite_diagnostic_only",
        "graph_verified": bool(graph_verified),
        "reason": "frontdoor_chain_graphical_lite" if complete else ("frontdoor_graph_verified_with_direct_edge" if graph_verified else ("candidate_only" if candidate else "coverage_failed")),
        "verification_level": verification_level,
        "covered_paths": covered_paths[:15],
        "uncovered_directed_paths": uncovered_paths[:15],
        "path_mediator_witness": path_mediator_witness,
        "path_chain_witness": path_chain_witness,
        "segment_reports": segment_reports,
        "treatment_mediator_backdoors": tm_backdoors,
        "mediator_outcome_backdoors": my_backdoors,
        "treatment_mediator_controls": tm_controls,
        "mediator_outcome_controls": my_controls,
        "chain_segment_controls": chain_segment_controls,
        "has_direct_treatment_outcome_edge": bool(has_direct_edge),
    }



def _formal_identification_report(effect_scope: str, backdoor_identifiable: bool, direct_effect_identifiable: bool, frontdoor_candidate: bool, frontdoor_identifiable: bool, nested_effects: Dict[str, object], direct_known: bool, treatment: str, outcome: str) -> Dict[str, object]:
    cde_ok = bool(nested_effects.get("cde", {}).get("identified"))
    nde_ok = bool(nested_effects.get("nde", {}).get("identified"))
    nie_ok = bool(nested_effects.get("nie", {}).get("identified"))
    cde_support = bool(nested_effects.get("cde", {}).get("strategy"))
    cross_world_required = bool(nested_effects.get("cross_world_assumption_required"))
    cross_world_plausible = bool(nested_effects.get("cross_world_plausible", False))

    total_status = "identified" if (backdoor_identifiable or frontdoor_identifiable) else ("simulable_not_identified" if (direct_known or frontdoor_candidate) else "not_identified")
    direct_status = "identified" if direct_effect_identifiable else ("simulable_not_identified" if (direct_known or cde_ok) else "not_identified")
    cde_status = "identified" if cde_ok else ("simulable_not_identified" if (cde_support and direct_known) else "not_identified")
    nde_status = "identified" if nde_ok else ("simulable_not_identified" if (cross_world_required or frontdoor_candidate) else "not_identified")
    nie_status = "identified" if nie_ok else ("simulable_not_identified" if (cross_world_required or frontdoor_candidate) else "not_identified")
    natural_status = "identified" if (nde_ok and nie_ok) else ("simulable_not_identified" if (cross_world_required or frontdoor_candidate) else "not_identified")

    if frontdoor_identifiable and (not direct_known):
        total_route = "frontdoor"
    elif backdoor_identifiable:
        total_route = "backdoor"
    elif frontdoor_identifiable:
        total_route = "frontdoor"
    elif frontdoor_candidate:
        total_route = "frontdoor_candidate_only"
    elif direct_known:
        total_route = "do_operator_only"
    else:
        total_route = "none"
    direct_route = "backdoor_direct" if direct_effect_identifiable else ("controlled_direct_only" if cde_ok else ("do_operator_only" if direct_known else "none"))
    cde_route = "controlled_direct_only" if cde_ok else ("do_operator_only" if direct_known else "none")
    nde_route = "frontdoor_nested" if nde_ok else ("cross_world_required" if cross_world_required else ("frontdoor_candidate_only" if frontdoor_candidate else "none"))
    nie_route = "frontdoor_nested" if nie_ok else ("cross_world_required" if cross_world_required else ("frontdoor_candidate_only" if frontdoor_candidate else "none"))

    if effect_scope == "direct":
        primary_status = direct_status
        route = direct_route
    else:
        primary_status = total_status
        route = total_route

    effect_specific = {
        "total_effect": {
            "status": total_status,
            "identified": bool(total_status == "identified"),
            "route": total_route,
            "estimand": f"E[{outcome} | do({treatment})]",
            "estimand_type": "total_effect",
        },
        "direct_effect": {
            "status": direct_status,
            "identified": bool(direct_status == "identified"),
            "route": direct_route,
            "estimand": f"E[{outcome} | do({treatment}), hold_mediators]",
            "estimand_type": "direct_effect",
        },
        "controlled_direct_effect": {
            "status": cde_status,
            "identified": bool(cde_status == "identified"),
            "route": cde_route,
            "estimand": f"E[{outcome}_{{{treatment}, M(m)}} - {outcome}_{{0, M(m)}}]",
            "estimand_type": "controlled_direct_effect",
        },
        "natural_direct_effect": {
            "status": nde_status,
            "identified": bool(nde_status == "identified"),
            "route": nde_route,
            "estimand": f"E[{outcome}_{{{treatment}, M({treatment})}} - {outcome}_{{0, M({treatment})}}]",
            "estimand_type": "natural_direct_effect",
            "cross_world_plausible": cross_world_plausible,
        },
        "natural_indirect_effect": {
            "status": nie_status,
            "identified": bool(nie_status == "identified"),
            "route": nie_route,
            "estimand": f"E[{outcome}_{{{treatment}, M({treatment})}} - {outcome}_{{{treatment}, M(0)}}]",
            "estimand_type": "natural_indirect_effect",
            "cross_world_plausible": cross_world_plausible,
        },
        "natural_effects": {
            "status": natural_status,
            "identified": bool(natural_status == "identified"),
            "route": "frontdoor_nested" if (nde_ok and nie_ok) else ("cross_world_required" if cross_world_required else "none"),
            "cross_world_plausible": cross_world_plausible,
            "estimand_type": "natural_effects_bundle",
        },
    }

    simulation_supported = any(v.get("status") != "not_identified" for v in effect_specific.values()) or direct_known or frontdoor_candidate
    simulation_status = "scm_do_simulable" if simulation_supported else "simulation_unsupported"
    if primary_status == "identified":
        identification_vs_simulation = "identified"
    elif simulation_status == "scm_do_simulable":
        identification_vs_simulation = "simulable_only"
    else:
        identification_vs_simulation = "not_identified"

    return {
        "primary_effect_status": primary_status,
        "primary_identification_route": route,
        "identification_vs_simulation": identification_vs_simulation,
        "simulation_status": simulation_status,
        "effect_specific_routes": {k: {"status": v["status"], "route": v["route"], "identified": v["identified"]} for k, v in effect_specific.items()},
        **effect_specific,
    }

def _natural_effects_assessment(treatment: str, outcome: str, mediators: List[str], frontdoor_report: Dict[str, object], direct_known: bool, node_meta: Dict[str, dict], children: Dict[str, Set[str]], parents: Dict[str, Set[str]]) -> Dict[str, object]:
    observed_mediators = [m for m in mediators if bool(node_meta.get(m, {}).get("observed", True))]
    unobserved_mediators = [m for m in mediators if m not in observed_mediators]
    descendants_t = _descendants(treatment, children)
    post_treatment_mo_confounders: Set[str] = set()
    for m in observed_mediators:
        for anc in _ancestors(m, parents).intersection(_ancestors(outcome, parents)):
            if anc in {treatment, outcome, m} or anc in observed_mediators:
                continue
            if anc in descendants_t:
                post_treatment_mo_confounders.add(anc)

    chain_complete = bool(frontdoor_report.get("identifiable", False))
    frontdoor_complete = frontdoor_report.get("verification_level") == "frontdoor_chain_complete_lite"
    mediator_outcome_uncovered = bool(frontdoor_report.get("mediator_outcome_backdoors")) and not bool(frontdoor_report.get("mediator_outcome_controls"))
    cross_world_plausible = frontdoor_complete and not direct_known and not mediator_outcome_uncovered and not post_treatment_mo_confounders and not unobserved_mediators

    # Controlled/path-specific effects are simulation diagnostics here, not
    # formal estimation authority. Observed mediators allow CDE simulation only.
    cde_simulable = bool(observed_mediators)
    cde_identified = False
    nde_identified = bool(cross_world_plausible)
    nie_identified = bool(cross_world_plausible)

    natural_reason = "natural_effects_frontdoor_nested_graphical_lite" if cross_world_plausible else "cross_world_not_graphically_supported"
    cde_reason = "mediator_freeze_simulation_supported_diagnostic_only" if cde_simulable else "missing_observed_mediators"

    return {
        "cde": {
            "identified": bool(cde_identified),
            "simulable": bool(cde_simulable),
            "strategy": "controlled_direct_simulation_supported" if cde_simulable else "not_identified",
            "authority": "diagnostic_simulation_only",
            "reason": cde_reason,
        },
        "nde": {
            "identified": bool(nde_identified),
            "strategy": "frontdoor_nested" if nde_identified else "not_identified",
            "reason": natural_reason,
        },
        "nie": {
            "identified": bool(nie_identified),
            "strategy": "frontdoor_nested" if nie_identified else "not_identified",
            "reason": natural_reason,
        },
        "cross_world_assumption_required": True,
        "cross_world_plausible": bool(cross_world_plausible),
        "frontdoor_chain_complete": bool(chain_complete),
        "frontdoor_verification_level": _as_str(frontdoor_report.get("verification_level", "")),
        "post_treatment_mediator_outcome_confounders": sorted(post_treatment_mo_confounders),
        "unobserved_mediators": sorted(unobserved_mediators),
    }



def _authoritative_id_fields(
    scm_graph: dict,
    treatment: str,
    outcome: str,
    adjustment_set: Iterable[str],
    mediators: Iterable[str],
    strategy_hint: str,
) -> Dict[str, object]:
    """Return canonical id_algorithm fields for legacy identifier rows.

    Failure to run the canonical layer never authorizes anything; it is reported
    as unavailable/error so this compatibility CSV remains safe.
    """
    if _authoritative_identify_effect is None:
        return {
            "canonical_id_available": 0,
            "canonical_id_source": "scm_parts.id_algorithm_unavailable",
            "canonical_id_status": "unavailable",
            "canonical_id_identified": 0,
            "canonical_id_strategy": "",
            "canonical_id_level": "",
            "canonical_id_formula": "",
            "canonical_id_reason_codes": "ID_ALGORITHM_IMPORT_UNAVAILABLE",
            "canonical_formula_tree_json": "{}",
            "canonical_id_proof_steps_json": "[]",
            "id_status": "",
            "id_identified": "0",
            "id_algorithm_level": "",
            "symbolic_formula_status": "",
            "hedge_detected": "0",
            "hedge_status": "",
            "recursive_id_status": "",
            "c_factor_status": "",
            "district_status": "",
            "id_block_reason": "ID_ALGORITHM_IMPORT_UNAVAILABLE",
            "id_reason_codes": "ID_ALGORITHM_IMPORT_UNAVAILABLE",
            "legacy_identifier_authority": "legacy_reporting_only",
        }
    try:
        result = _authoritative_identify_effect(
            scm_graph,
            treatment,
            outcome,
            adjustment_set=list(adjustment_set or []),
            mediators=list(mediators or []),
            strategy_hint=strategy_hint,
        ).to_dict()
    except (OSError, ValueError, TypeError, RuntimeError, KeyError, ImportError) as exc:
        err = f"{type(exc).__name__}:{exc}"
        return {
            "canonical_id_available": 0,
            "canonical_id_source": "scm_parts.id_algorithm_error",
            "canonical_id_status": "error",
            "canonical_id_identified": 0,
            "canonical_id_strategy": "",
            "canonical_id_level": "",
            "canonical_id_formula": "",
            "canonical_id_reason_codes": err,
            "canonical_formula_tree_json": "{}",
            "canonical_id_proof_steps_json": "[]",
            "id_status": "",
            "id_identified": "0",
            "id_algorithm_level": "",
            "symbolic_formula_status": "",
            "hedge_detected": "0",
            "hedge_status": "",
            "recursive_id_status": "",
            "c_factor_status": "",
            "district_status": "",
            "id_block_reason": err,
            "id_reason_codes": err,
            "legacy_identifier_authority": "legacy_reporting_only",
        }
    identifiable = bool(result.get("identifiable"))
    reason_codes = _as_str(result.get("reason_codes", ""))
    failure_reason = _as_str(result.get("failure_reason", "")) or reason_codes
    return {
        "canonical_id_available": 1,
        "canonical_id_source": "scm_parts.id_algorithm",
        "canonical_id_status": "identified" if identifiable else "blocked",
        "canonical_id_identified": 1 if identifiable else 0,
        "canonical_id_strategy": _as_str(result.get("id_strategy", "")),
        "canonical_id_level": _as_str(result.get("id_algorithm_level", "")),
        "canonical_id_formula": _as_str(result.get("estimand_formula", "")),
        "canonical_id_reason_codes": reason_codes,
        "canonical_formula_tree_json": _as_str(result.get("formula_tree_json", "{}")) or "{}",
        "canonical_id_proof_steps_json": _as_str(result.get("id_proof_steps_json", "[]")) or "[]",
        # Mirror the authoritative audit fields into the legacy CSV so that a
        # downstream causal contract can hard-block or authorize from ID even if
        # out/scm/id_algorithm_audit.csv is absent. identifier.py still remains
        # reporting-only; these fields are copied from id_algorithm.py.
        "id_status": _as_str(result.get("id_strategy", "")),
        "id_identified": "1" if identifiable else "0",
        "id_algorithm_level": _as_str(result.get("id_algorithm_level", "")),
        "symbolic_formula_status": _as_str(result.get("symbolic_formula_status", "")),
        "symbolic_formula_kind": _as_str(result.get("symbolic_formula_kind", "")),
        "symbolic_formula_json": _as_str(result.get("symbolic_formula_json", "")),
        "symbolic_formula_latex": _as_str(result.get("symbolic_formula_latex", "")),
        "symbolic_sum_over": _as_str(result.get("symbolic_sum_over", "")),
        "symbolic_product_terms": _as_str(result.get("symbolic_product_terms", "")),
        "symbolic_removed_terms": _as_str(result.get("symbolic_removed_terms", "")),
        "symbolic_unresolved_terms": _as_str(result.get("symbolic_unresolved_terms", "")),
        "hedge_detected": "1" if bool(result.get("possible_hedge")) else "0",
        "hedge_status": _as_str(result.get("hedge_status", "")),
        "recursive_id_status": _as_str(result.get("recursive_status", "")),
        "c_factor_status": _as_str(result.get("c_factor_status", "")),
        "district_status": _as_str(result.get("district_status", "")),
        "id_block_reason": failure_reason if not identifiable else "",
        "id_reason_codes": reason_codes,
        "legacy_identifier_authority": "legacy_reporting_only",
    }


def _apply_canonical_id_authority(row: Dict[str, object]) -> Dict[str, object]:
    """Make legacy identifier rows obey the canonical id_algorithm decision.

    The legacy graph/reporting code can still compute adjustment/path
    diagnostics, but it cannot grant identification authority. When the
    canonical ID layer is available, its result overwrites the public
    identified/status/strategy fields and the legacy values are retained only as
    ``legacy_identifier_*`` audit columns.
    """
    row = dict(row)
    row["legacy_identifier_identified"] = row.get("identified", "")
    row["legacy_identifier_status"] = row.get("identification_status", "")
    row["legacy_identifier_strategy"] = row.get("identification_strategy", "")
    row["legacy_identifier_route"] = row.get("identification_route", "")
    row["id_algorithm_is_authority"] = 0
    row["identification_authority_source"] = "legacy_identifier_reporting_only"

    canonical_available = _as_str(row.get("canonical_id_available", "0")).strip().lower() in {"1", "true", "yes"}
    if not canonical_available:
        row["effect_claim_authority"] = row.get("effect_claim_authority") or "legacy_identifier_reporting_only_no_canonical_id"
        row["estimation_enabled"] = 0
        return row

    row["id_algorithm_is_authority"] = 1
    row["identification_authority_source"] = "scm_parts.id_algorithm"
    canonical_status = _as_str(row.get("canonical_id_status", "")).strip().lower()
    canonical_identified = canonical_status == "identified" or _as_str(row.get("canonical_id_identified", "")).strip().lower() in {"1", "true", "yes"}
    canonical_strategy = _as_str(row.get("canonical_id_strategy", "")).strip()
    canonical_level = _as_str(row.get("canonical_id_level", "")).strip()
    canonical_formula = _as_str(row.get("canonical_id_formula", "")).strip()
    canonical_reasons = _as_str(row.get("canonical_id_reason_codes", "")).strip()

    row["identified"] = 1 if canonical_identified else 0
    row["identification_status"] = "identified" if canonical_identified else "blocked_by_id_algorithm"
    row["identification_strategy"] = canonical_strategy or ("id_algorithm_identified" if canonical_identified else "id_algorithm_blocked")
    row["identification_route"] = canonical_level
    row["id_status"] = row.get("id_status") or row["identification_strategy"]
    row["id_identified"] = "1" if canonical_identified else "0"
    row["id_algorithm_level"] = row.get("id_algorithm_level") or canonical_level
    row["estimand_authority_status"] = "canonical_id_algorithm_identified" if canonical_identified else "blocked_by_canonical_id_algorithm"
    if canonical_formula:
        row["estimand_expression"] = canonical_formula
    row["effect_claim_authority"] = "canonical_id_algorithm" if canonical_identified else "no_effect_claim_id_algorithm_blocked"

    # Only the causal contract + do_contract may enable estimation. The legacy
    # CSV mirrors ID; it never grants estimator authority by itself.
    row["estimation_enabled"] = 0
    if not canonical_identified:
        row["adjustment_set_status"] = "missing"
        row["blocked_by"] = "|".join(_dedupe(_split_pipe(row.get("blocked_by", "")) + ["blocked_by_canonical_id_algorithm"] + _split_pipe(canonical_reasons)))
        row["failed_assumptions"] = "|".join(_dedupe(_split_pipe(row.get("failed_assumptions", "")) + ["blocked_by_canonical_id_algorithm"] + _split_pipe(canonical_reasons)))
    notes = _dedupe(_split_pipe(row.get("notes", "")) + [
        "legacy_identifier_reporting_only",
        "id_algorithm_authority_applied",
        f"canonical_id_strategy={canonical_strategy}",
    ])
    row["notes"] = "|".join(notes)
    return row

def build_identification_assets(scm_graph: dict, bridge = None, insights = None) -> Dict[str, object]:
    bridge = _normalize_effect_contract(bridge.copy() if bridge is not None else pd.DataFrame())
    insights = _normalize_effect_contract(insights.copy() if insights is not None else pd.DataFrame())

    nodes = list(scm_graph.get("nodes", []))
    edges = list(scm_graph.get("edges", []))
    children, parents, by_pair, undirected = _build_graph(edges)
    node_meta, node_role = _role_maps(nodes)
    global_colliders = _find_colliders(parents)

    if len(bridge) > 0:
        effects = bridge.copy()
    elif len(insights) > 0:
        effects = _normalize_effect_contract(insights).copy()
    else:
        effects = pd.DataFrame(columns=["treatment_col", "outcome_col"])

    rows: List[dict] = []
    for _, row in effects.iterrows():
        treatment = _as_str(_first_present(row, ["treatment_col", "source", "treatment", "cause", "action_col", "action"])).strip()
        outcome = _as_str(_first_present(row, ["outcome_col", "target", "outcome", "harm_event", "target_col", "effect"])).strip()
        if not treatment or not outcome:
            continue
        direct_known = bool(by_pair.get((treatment, outcome), {}))
        descendants_t = _descendants(treatment, children)
        ancestors_t = _ancestors(treatment, parents)
        mediators = _dedupe(_split_pipe(row.get("post_treatment_columns", "")) + _split_pipe(row.get("mediators", "")) + _find_path_mediators(treatment, outcome, children))
        colliders = sorted(set(global_colliders).intersection(set(children.get(treatment, set())) | set(children.get(outcome, set())) | set(mediators)))
        total_adjustments, forbidden, backdoor_report = _candidate_backdoor_controls(treatment, outcome, parents, children, undirected, node_meta, mediators, colliders)
        dsep_total = _valid_adjustment_sets_for_backdoor(treatment, outcome, total_adjustments, forbidden, parents, children, undirected, max_set_size=3)
        minimal_total_sets = dsep_total.get("minimal_sets", [])
        total_adjustments = minimal_total_sets[0] if minimal_total_sets else total_adjustments
        direct_candidates = [c for c in _dedupe(total_adjustments + [c for s in minimal_total_sets for c in s]) if c not in mediators and c not in descendants_t]
        dsep_direct = _valid_adjustment_sets_for_backdoor(treatment, outcome, direct_candidates, forbidden, parents, children, undirected, max_set_size=3)
        minimal_direct_sets = dsep_direct.get("minimal_sets", [])
        direct_adjustments = minimal_direct_sets[0] if minimal_direct_sets else [c for c in total_adjustments if c not in mediators and c not in descendants_t]
        graph_query_unadjusted = _dsep_query_report(treatment, outcome, [], parents, children, undirected, cutoff=6)
        graph_query_total = _dsep_query_report(treatment, outcome, total_adjustments, parents, children, undirected, cutoff=6)
        graph_query_direct = _dsep_query_report(treatment, outcome, direct_adjustments, parents, children, undirected, cutoff=6)
        negative_controls = _split_pipe(row.get("negative_controls", row.get("dag_negative_controls", row.get("graph_negative_controls", ""))))
        direct_requested = _as_str(row.get("preferred_estimand", "")).lower() in {"nde", "natural_direct_effect", "controlled_direct_effect", "direct_effect"}
        preferred_estimand, effect_scope = _choose_estimand(row.get("preferred_estimand", ""), direct_requested, len(mediators) > 0, direct_known)

        backdoor_identifiable = bool(minimal_total_sets) or len(parents.get(treatment, set())) == 0
        direct_effect_identifiable = direct_known and (bool(minimal_direct_sets) or len(parents.get(treatment, set())) == 0)
        frontdoor_report = _frontdoor_assessment(treatment, outcome, mediators, parents, children, undirected, node_meta)
        frontdoor_candidate = bool(frontdoor_report.get("candidate", False))
        frontdoor_identifiable = bool(frontdoor_report.get("identifiable", False))
        frontdoor_graphical_lite = bool(frontdoor_report.get("graphical_lite_identifiable", False))
        nested_effects = _natural_effects_assessment(treatment, outcome, mediators, frontdoor_report, direct_known, node_meta, children, parents)

        formal_id = _formal_identification_report(effect_scope, backdoor_identifiable, direct_effect_identifiable, frontdoor_candidate, frontdoor_identifiable, nested_effects, direct_known, treatment, outcome)
        primary_status = _as_str(formal_id.get("primary_effect_status", "not_identified"))
        identification_route = _as_str(formal_id.get("primary_identification_route", "none"))
        if effect_scope == "direct":
            identification_strategy = "backdoor_direct" if direct_effect_identifiable else ("cde_only" if bool(nested_effects.get("cde", {}).get("identified")) else ("simulable_not_identified" if primary_status == "simulable_not_identified" else "not_identified"))
        elif frontdoor_identifiable and (not backdoor_identifiable or (not direct_known and len(total_adjustments) == 0 and len(parents.get(treatment, set())) == 0)):
            identification_strategy = "frontdoor"
        elif backdoor_identifiable:
            identification_strategy = "backdoor"
        elif frontdoor_candidate:
            identification_strategy = "frontdoor_candidate"
        elif primary_status == "simulable_not_identified":
            identification_strategy = "simulable_not_identified"
        else:
            identification_strategy = "not_identified"

        res = build_identification(
            has_controls=len(total_adjustments) > 0,
            propensity_available=len(total_adjustments) > 0,
            overlap_ok=True if len(total_adjustments) > 0 else None,
            balance_ok=True if len(total_adjustments) > 0 else None,
            sample_size_ok=True,
            leakage_ok=True,
            drift_ok=True,
            temporal_order_ok=True,
            diagnostic_grade="strong" if direct_known else ("moderate" if backdoor_identifiable else "weak"),
            sensitivity_level="moderate",
            pretrend_available=False,
            dag_adjustment_set=direct_adjustments if effect_scope == "direct" else total_adjustments,
            dag_forbidden_adjustments=forbidden,
            dag_adjustment_confidence="high" if len(total_adjustments) > 0 else "low",
            dag_direct_edge_confidence="high" if direct_known else "low",
            dag_path_confidence="high" if (direct_known or frontdoor_identifiable) else "low",
            dag_mediators=mediators,
            dag_colliders=colliders,
            dag_negative_controls=negative_controls,
            dag_path_id=f"{treatment}->{outcome}",
            dag_treatment_node=treatment,
            dag_outcome_node=outcome,
            dag_action_known=1,
            dag_target_known=1,
            dag_action_type="action" if node_role.get(treatment, "") in {"action", "guardrail"} else "exposure",
            dag_target_type="outcome" if node_role.get(outcome, "") == "outcome" else "risk_outcome",
            dag_action_time_role=_as_str(node_meta.get(treatment, {}).get("time_role", "lagged_or_state")),
            dag_target_time_role=_as_str(node_meta.get(outcome, {}).get("time_role", "concurrent")),
        ).to_dict()

        identified = primary_status == "identified"
        failed = list(res.get("failed_assumptions", []))
        if effect_scope == "direct" and not direct_effect_identifiable:
            failed.append("no_valid_direct_effect_adjustment_found")
        if effect_scope != "direct" and not backdoor_identifiable and identification_strategy == "not_identified":
            failed.append("no_valid_backdoor_adjustment_found")
        notes = list(res.get("notes", []))
        if mediators:
            notes.append("mediated_path_present")
        if forbidden:
            notes.append("bad_controls_excluded")
        notes.append(f"backdoor_paths_considered={int(backdoor_report.get('n_candidate_paths', 0))}")
        notes.append(f"dsep_min_total_sets={len(minimal_total_sets)}")
        notes.append(f"dsep_min_direct_sets={len(minimal_direct_sets)}")
        if effect_scope == "direct":
            notes.append("direct_effect_mode")
        if frontdoor_candidate:
            notes.append("frontdoor_candidate_present")
        if primary_status == "simulable_not_identified":
            notes.append("do_simulable_but_not_graphically_identified")
            notes.append(f"identification_route={identification_route}")
        if frontdoor_graphical_lite and not frontdoor_identifiable:
            notes.append("frontdoor_graphical_lite_diagnostic_only")
        if frontdoor_identifiable:
            notes.append("frontdoor_limited_estimator_ready")
        if nested_effects.get("cde", {}).get("simulable"):
            notes.append("cde_simulation_supported_diagnostic_only")
        if nested_effects.get("nde", {}).get("identified"):
            notes.append("nde_graphically_supported")
        if nested_effects.get("nie", {}).get("identified"):
            notes.append("nie_graphically_supported")
        elif nested_effects.get("cross_world_assumption_required"):
            notes.append("natural_effects_require_cross_world_assumption")

        if effect_scope == "direct":
            adjustment_set_status = _as_str(dsep_direct.get("adjustment_set_status", "")).strip()
            if not adjustment_set_status:
                adjustment_set_status = "valid_nonempty" if direct_adjustments else ("valid_empty" if direct_effect_identifiable else "missing")
        else:
            adjustment_set_status = _as_str(dsep_total.get("adjustment_set_status", "")).strip()
            if not adjustment_set_status:
                adjustment_set_status = "valid_nonempty" if total_adjustments else ("valid_empty" if backdoor_identifiable else "missing")

        out = {
            "insight_id": _as_str(row.get("insight_id", "")),
            "source": treatment,
            "target": outcome,
            "treatment_col": treatment,
            "outcome_col": outcome,
            "lag": _as_str(_first_present(row, ["lag", "tau", "time_lag"], "")),
            "bridge_version": _as_str(row.get("bridge_version", "")),
            "discovery_confidence_tier": _as_str(row.get("discovery_confidence_tier", "")),
            "discovery_effect_proxy": _as_str(row.get("discovery_effect_proxy", "")),
            "identification_strategy": identification_strategy,
            "identified": int(bool(identified)),
            "estimand_type": preferred_estimand,
            "effect_scope": effect_scope,
            "estimand_expression": (f"E[{outcome} | do({treatment})]" if effect_scope != "direct" else f"E[{outcome} | do({treatment}), hold_mediators]") if identified else "not_identified",
            "adjustment_set_status": adjustment_set_status,
            "adjustment_set": "|".join(direct_adjustments if effect_scope == "direct" else total_adjustments),
            "total_adjustment_set": "|".join(total_adjustments),
            "direct_adjustment_set": "|".join(direct_adjustments),
            "forbidden_adjustments": "|".join(forbidden),
            "mediators": "|".join(mediators),
            "colliders": "|".join(colliders),
            "descendants_of_treatment": "|".join(sorted(descendants_t)),
            "ancestors_of_treatment": "|".join(sorted(ancestors_t)),
            "negative_controls": "|".join(negative_controls),
            "direct_edge_present": int(direct_known),
            "backdoor_identifiable": int(backdoor_identifiable),
            "direct_effect_identifiable": int(direct_effect_identifiable),
            "dsep_backdoor_identifiable": int(backdoor_identifiable),
            "dsep_direct_effect_identifiable": int(direct_effect_identifiable),
            "minimal_adjustment_sets": json.dumps(minimal_total_sets, ensure_ascii=False, sort_keys=True),
            "minimal_direct_adjustment_sets": json.dumps(minimal_direct_sets, ensure_ascii=False, sort_keys=True),
            "dsep_report": json.dumps({
                "total_effect": dsep_total,
                "direct_effect": dsep_direct,
            }, ensure_ascii=False, sort_keys=True),
            "graph_query_unadjusted": json.dumps(graph_query_unadjusted, ensure_ascii=False, sort_keys=True),
            "graph_query_total_adjusted": json.dumps(graph_query_total, ensure_ascii=False, sort_keys=True),
            "graph_query_direct_adjusted": json.dumps(graph_query_direct, ensure_ascii=False, sort_keys=True),
            "minimal_adjustment_proof": json.dumps({
                "effect_scope": effect_scope,
                "selected_total_adjustment": total_adjustments,
                "selected_direct_adjustment": direct_adjustments,
                "minimal_total_sets": minimal_total_sets,
                "minimal_direct_sets": minimal_direct_sets,
                "unadjusted_open_paths": graph_query_unadjusted.get("open_paths", []),
                "open_paths_after_total_adjustment": graph_query_total.get("open_paths", []),
                "open_paths_after_direct_adjustment": graph_query_direct.get("open_paths", []),
                "unblocked_backdoor_paths_without_adjustment": dsep_total.get("unblocked_under_empty", []),
                "unblocked_backdoor_paths_after_total_adjustment": [p for p in dsep_total.get("all_backdoor_paths", []) if p in graph_query_total.get("open_paths", [])],
                "unblocked_backdoor_paths_after_direct_adjustment": [p for p in dsep_direct.get("all_backdoor_paths", []) if p in graph_query_direct.get("open_paths", [])],
            }, ensure_ascii=False, sort_keys=True),
            "frontdoor_candidate": int(frontdoor_candidate),
            "frontdoor_identifiable": int(frontdoor_identifiable),
            "frontdoor_graphical_lite_candidate": int(frontdoor_graphical_lite),
            "frontdoor_authority": _as_str(frontdoor_report.get("identification_authority", "diagnostic_only")),
            "frontdoor_status": "frontdoor_valid" if frontdoor_identifiable else ("frontdoor_candidate" if frontdoor_candidate else "missing"),
            "frontdoor_verification_level": _as_str(frontdoor_report.get("verification_level", "none")),
            "frontdoor_report": json.dumps(frontdoor_report, ensure_ascii=False, sort_keys=True),
            "nested_effects_report": json.dumps(nested_effects, ensure_ascii=False, sort_keys=True),
            "cde_identified": int(bool(nested_effects.get("cde", {}).get("identified"))),
            "cde_simulable": int(bool(nested_effects.get("cde", {}).get("simulable", False))),
            "cde_authority": _as_str(nested_effects.get("cde", {}).get("authority", "diagnostic_simulation_only")),
            "nde_identified": int(bool(nested_effects.get("nde", {}).get("identified"))),
            "nie_identified": int(bool(nested_effects.get("nie", {}).get("identified"))),
            "cde_strategy": _as_str(nested_effects.get("cde", {}).get("strategy", "")),
            "nde_strategy": _as_str(nested_effects.get("nde", {}).get("strategy", "")),
            "nie_strategy": _as_str(nested_effects.get("nie", {}).get("strategy", "")),
            "cross_world_plausible": int(bool(nested_effects.get("cross_world_plausible", False))),
            "identification_status": primary_status,
            "identification_route": identification_route,
            "identification_vs_simulation": _as_str(formal_id.get("identification_vs_simulation", "not_identified")),
            "simulation_status": _as_str(formal_id.get("simulation_status", "simulation_unsupported")),
            "formal_effects_report": json.dumps(formal_id, ensure_ascii=False, sort_keys=True),
            "total_effect_status": _as_str(formal_id.get("total_effect", {}).get("status", "not_identified")),
            "total_effect_route": _as_str(formal_id.get("total_effect", {}).get("route", "none")),
            "total_effect_estimand": _as_str(formal_id.get("total_effect", {}).get("estimand", "")),
            "direct_effect_status": _as_str(formal_id.get("direct_effect", {}).get("status", "not_identified")),
            "direct_effect_route": _as_str(formal_id.get("direct_effect", {}).get("route", "none")),
            "direct_effect_estimand": _as_str(formal_id.get("direct_effect", {}).get("estimand", "")),
            "controlled_direct_effect_status": _as_str(formal_id.get("controlled_direct_effect", {}).get("status", "not_identified")),
            "controlled_direct_effect_route": _as_str(formal_id.get("controlled_direct_effect", {}).get("route", "none")),
            "controlled_direct_effect_estimand": _as_str(formal_id.get("controlled_direct_effect", {}).get("estimand", "")),
            "natural_direct_effect_status": _as_str(formal_id.get("natural_direct_effect", {}).get("status", "not_identified")),
            "natural_direct_effect_route": _as_str(formal_id.get("natural_direct_effect", {}).get("route", "none")),
            "natural_direct_effect_estimand": _as_str(formal_id.get("natural_direct_effect", {}).get("estimand", "")),
            "natural_indirect_effect_status": _as_str(formal_id.get("natural_indirect_effect", {}).get("status", "not_identified")),
            "natural_indirect_effect_route": _as_str(formal_id.get("natural_indirect_effect", {}).get("route", "none")),
            "natural_indirect_effect_estimand": _as_str(formal_id.get("natural_indirect_effect", {}).get("estimand", "")),
            "natural_effects_status": _as_str(formal_id.get("natural_effects", {}).get("status", "not_identified")),
            "natural_effects_route": _as_str(formal_id.get("natural_effects", {}).get("route", "none")),
            "effect_specific_routes": json.dumps(formal_id.get("effect_specific_routes", {}), ensure_ascii=False, sort_keys=True),
            "identification_strength": res.get("identification_strength", "unknown"),
            "identification_score": res.get("identification_score", 0.0),
            "assumptions": "|".join(res.get("assumptions", [])),
            "failed_assumptions": "|".join(_dedupe(failed)),
            "notes": "|".join(_dedupe(notes)),
            "backdoor_path_report": json.dumps(backdoor_report, ensure_ascii=False, sort_keys=True),
        }
        canonical_adjustment = direct_adjustments if effect_scope == "direct" else total_adjustments
        out.update(_authoritative_id_fields(
            scm_graph,
            treatment,
            outcome,
            canonical_adjustment,
            mediators,
            identification_strategy,
        ))
        out = _apply_canonical_id_authority(out)
        rows.append(out)

    identified_effects = pd.DataFrame(rows)
    n_identified = int(identified_effects["identified"].sum()) if len(identified_effects) > 0 and "identified" in identified_effects.columns else 0
    empty_scm_reason = ""
    if len(identified_effects) == 0:
        empty_scm_reason = "no_identifiable_effect_candidates_from_scm_graph_or_bridge"
    elif n_identified == 0:
        empty_scm_reason = "no_effects_passed_conservative_identification_checks"
    summary = {
        "identification_version": "1.2-step18",
        "type": "legacy_identifier_reporting_wrapper",
        "authoritative_id_module": "scm_parts.id_algorithm",
        "shared_graph_criteria_module": "scm_parts.graph_criteria",
        "legacy_identifier_authority": "legacy_reporting_only",
        "empty_scm_reason": empty_scm_reason,
        "n_effects": int(len(identified_effects)),
        "n_identified": n_identified,
        "strategies": identified_effects["identification_strategy"].value_counts().to_dict() if len(identified_effects) > 0 else {},
        "assumptions": [
            "graph derived from SCM outputs",
            "forbidden adjustments include descendants of treatment, mediators, colliders, descendants of mediators, and outcome",
            "backdoor candidates filtered to observed ancestors that appear on simple undirected backdoor paths",
            "direct-effect mode requires a direct edge plus a valid direct adjustment view",
            "frontdoor is upgraded only when all directed treatment->outcome paths are covered by observed mediator chains, each chain segment passes a conservative backdoor blocking check, treatment->mediator backdoors are closed, mediator->outcome backdoors are closed after allowing treatment among blockers, and direct treatment->outcome edges are absent for full graphical-lite verification",
            "natural effects are reported separately from controlled direct effects; when cross-world support is not graphically plausible the package marks NDE/NIE as not identified rather than overclaiming",
            "effect-specific reporting now emits separate status, route, and estimand fields for total, direct, controlled direct, natural direct, and natural indirect effects",
            "adjustment_set_status distinguishes valid_empty adjustment sets from missing/invalid adjustment evidence",
            "frontdoor/CDE/NDE/NIE outputs are diagnostic or graphical-lite unless separately authorized by a downstream estimator; current Pearl estimation authorization is backdoor-only",
            "d-separation style backdoor screening searches minimal observed adjustment sets that block all simple undirected backdoor paths under conservative collider activation rules",
            "graph query exports now include explicit unadjusted and adjusted d-separation reports so the package can show why a pair remains connected or becomes blocked under a chosen adjustment set",
            "SCM-18: identifier.py is a legacy compatibility/reporting wrapper; causal identification authority is id_algorithm.py and graph predicates are centralized in graph_criteria.py",
            "SCM-30: identified_effects.csv mirrors canonical id_algorithm decisions; legacy Pearl-lite diagnostics are retained only as legacy_identifier_* audit columns and cannot enable estimation by themselves",
        ],
    }
    return {"identified_effects": identified_effects, "identification_summary": summary}



ADJUSTMENT_SET_COLUMNS = [
    "insight_id", "source", "target", "treatment_col", "outcome_col", "lag",
    "estimand_type", "identification_status", "identified", "identification_strategy",
    "backdoor_status", "frontdoor_status", "adjustment_set_status",
    "adjustment_set", "total_adjustment_set", "direct_adjustment_set",
    "candidate_adjustment_set", "forbidden_adjustment_set", "mediators", "colliders",
    "blocked_by", "assumption_notes", "conditioning_set_used", "conditioning_set_size",
    "mci_status", "scm_role_hint", "identification_priority", "eligible_for_estimation",
]


def _pipe_minus(items: Iterable[str], banned: Iterable[str]) -> List[str]:
    banned_set = {str(x).strip() for x in banned if str(x).strip()}
    return [x for x in _dedupe(items) if x not in banned_set]


def _status_from_adjustment_row(row) -> Tuple[str, str, str]:
    """Return (backdoor_status, blocked_by, eligible_for_estimation).

    The adjustment handoff is also reporting-only.  When canonical ID fields are
    present, they decide whether the row can be described as an ID-backed
    adjustment candidate.  Non-backdoor ID formulas are not turned into
    adjustment-set authority here.
    """
    canonical_available = _as_str(row.get("canonical_id_available", "")).strip().lower() in {"1", "true", "yes"}
    canonical_identified = _as_str(row.get("canonical_id_status", "")).strip().lower() == "identified" or _as_str(row.get("canonical_id_identified", "")).strip().lower() in {"1", "true", "yes"}
    canonical_strategy = _as_str(row.get("canonical_id_strategy", row.get("id_status", ""))).strip().lower()
    canonical_reasons = _split_pipe(row.get("canonical_id_reason_codes", row.get("id_reason_codes", "")))
    if canonical_available and not canonical_identified:
        return "blocked_by_id_algorithm", "|".join(_dedupe(["blocked_by_canonical_id_algorithm"] + canonical_reasons)), "0"

    identified = _as_str(row.get("identified", "")).strip().lower() in {"1", "true", "yes"}
    strategy = _as_str(row.get("identification_strategy", "")).lower()
    adj_status = _as_str(row.get("adjustment_set_status", "")).strip().lower()
    failed = _split_pipe(row.get("failed_assumptions", ""))
    forbidden = _split_pipe(row.get("forbidden_adjustments", row.get("forbidden_adjustment_set", "")))
    colliders = _split_pipe(row.get("colliders", ""))
    mediators = _split_pipe(row.get("mediators", ""))
    if canonical_available:
        if canonical_identified and "backdoor" in canonical_strategy and adj_status in {"valid_empty", "valid_nonempty"}:
            return "backdoor_adjustment_candidate", "", "1"
        if canonical_identified:
            return "identified_by_id_algorithm_non_adjustment", "non_backdoor_id_formula_not_adjustment_authority", "0"
    if identified and adj_status in {"valid_empty", "valid_nonempty"} and "backdoor" in strategy:
        return "backdoor_adjustment_candidate", "", "1"
    if identified and adj_status in {"valid_empty", "valid_nonempty"}:
        return "adjustment_candidate", "", "0"
    blockers = _dedupe(failed + (["forbidden_adjustments_present"] if forbidden else []) + (["collider_warning"] if colliders else []) + (["mediator_warning"] if mediators else []))
    if blockers:
        return "not_identified", "|".join(blockers), "0"
    if adj_status == "missing":
        return "missing_adjustment_set", "no_valid_adjustment_set_found", "0"
    return "diagnostic_only", "not_formally_identified", "0"


def _build_adjustment_sets_frame(effects: pd.DataFrame, bridge=None) -> pd.DataFrame:
    """Build an explicit Pearl-lite adjustment-set handoff.

    This table summarizes selected adjustment sets, bad controls, mediators,
    colliders, and estimation eligibility. It does not create new causal claims.
    """
    if effects is None or len(effects) == 0:
        return pd.DataFrame(columns=ADJUSTMENT_SET_COLUMNS)
    bridge_by_pair = {}
    if bridge is not None and hasattr(bridge, "iterrows") and len(bridge) > 0:
        for _, brow in bridge.iterrows():
            bs = _first_present(brow, ["source", "treatment_col"], "")
            bt = _first_present(brow, ["target", "outcome_col"], "")
            blag = _first_present(brow, ["lag", "tau", "time_lag"], "")
            if bs or bt:
                bridge_by_pair[(str(bs), str(bt), str(blag))] = brow
                bridge_by_pair.setdefault((str(bs), str(bt), ""), brow)
    rows = []
    for idx, row in effects.iterrows():
        source = _first_present(row, ["source", "treatment_col"], "")
        target = _first_present(row, ["target", "outcome_col"], "")
        lag = _first_present(row, ["lag", "tau", "time_lag"], "")
        brow = bridge_by_pair.get((str(source), str(target), str(lag)), bridge_by_pair.get((str(source), str(target), ""), {}))
        total_adj = _first_present(row, ["total_adjustment_set", "adjustment_set"], "")
        direct_adj = _first_present(row, ["direct_adjustment_set"], "")
        candidate = total_adj or direct_adj or _first_present(brow, ["conditioning_set_used", "mci_conditioning_set_used", "parent_set"], "")
        forbidden = _first_present(row, ["forbidden_adjustments", "forbidden_adjustment_set"], "")
        mediators = _first_present(row, ["mediators"], "")
        colliders = _first_present(row, ["colliders"], "")
        bad_controls = _dedupe(_split_pipe(forbidden) + _split_pipe(mediators) + _split_pipe(colliders) + [source, target])
        cleaned_candidate = "|".join(_pipe_minus(_split_pipe(candidate), bad_controls))
        backdoor_status, blocked_by, eligible = _status_from_adjustment_row(row)
        notes = _dedupe(_split_pipe(_first_present(row, ["notes"], "")) + [
            "uses_identifier_selected_adjustment_set" if cleaned_candidate or _first_present(row, ["adjustment_set_status"], "") == "valid_empty" else "no_selected_adjustment_set",
            "mci_conditioning_set_available" if _first_present(brow, ["conditioning_set_used", "mci_conditioning_set_used"], "") else "mci_conditioning_set_missing",
        ])
        rows.append({
            "insight_id": _first_present(row, ["insight_id", "candidate_id", "edge_id"], f"adjustment::{source}->{target}@{lag or idx}"),
            "source": source,
            "target": target,
            "treatment_col": _first_present(row, ["treatment_col", "source"], source),
            "outcome_col": _first_present(row, ["outcome_col", "target"], target),
            "lag": lag,
            "estimand_type": _first_present(row, ["estimand_type", "effect_scope"], "total_effect"),
            "identification_status": _first_present(row, ["identification_status"], "not_identified"),
            "identified": _first_present(row, ["identified"], "0"),
            "identification_strategy": _first_present(row, ["identification_strategy"], ""),
            "backdoor_status": backdoor_status,
            "frontdoor_status": _first_present(row, ["frontdoor_status"], ""),
            "adjustment_set_status": _first_present(row, ["adjustment_set_status"], "missing"),
            "adjustment_set": cleaned_candidate if cleaned_candidate else _first_present(row, ["adjustment_set"], ""),
            "total_adjustment_set": total_adj,
            "direct_adjustment_set": direct_adj,
            "candidate_adjustment_set": cleaned_candidate,
            "forbidden_adjustment_set": forbidden,
            "mediators": mediators,
            "colliders": colliders,
            "blocked_by": blocked_by,
            "assumption_notes": "|".join(notes),
            "conditioning_set_used": _first_present(brow, ["conditioning_set_used", "mci_conditioning_set_used"], ""),
            "conditioning_set_size": _first_present(brow, ["conditioning_set_size", "mci_conditioning_set_size"], ""),
            "mci_status": _first_present(brow, ["mci_status"], ""),
            "scm_role_hint": _first_present(brow, ["scm_role_hint"], ""),
            "identification_priority": _first_present(brow, ["identification_priority", "selection_score"], ""),
            "eligible_for_estimation": eligible,
        })
    return pd.DataFrame(rows, columns=ADJUSTMENT_SET_COLUMNS)

def write_identification_assets(out_dir: str = "out", scm_graph_path: Optional[str] = None, bridge = None, bridge_csv_path: Optional[str] = None, insights = None) -> Dict[str, str]:
    if scm_graph_path is None:
        scm_graph_path = os.path.join(out_dir, "scm", "scm_graph.json")
        if not os.path.exists(scm_graph_path):
            scm_graph_path = os.path.join(out_dir, "scm_graph.json")
    scm_graph = _load_json(scm_graph_path)
    if bridge is None and bridge_csv_path and os.path.exists(bridge_csv_path):
        try:
            bridge = _normalize_effect_contract(pd.read_csv(bridge_csv_path))
        except (OSError, ValueError, TypeError, pd.errors.ParserError) as exc:
            warnings.warn(f"[amantia][warning] identification assets could not read bridge CSV {bridge_csv_path}: {type(exc).__name__}: {exc}", RuntimeWarning)
            bridge = None

    assets = build_identification_assets(scm_graph, bridge=bridge, insights=insights)
    id_dir = os.path.join(out_dir, ID_DIRNAME)
    os.makedirs(id_dir, exist_ok=True)
    paths = {
        "identified_effects_csv": os.path.join(id_dir, "identified_effects.csv"),
        "identified_effects_json": os.path.join(id_dir, "identified_effects.json"),
        "adjustment_sets_csv": os.path.join(id_dir, "adjustment_sets.csv"),
        "adjustment_sets_json": os.path.join(id_dir, "adjustment_sets.json"),
        "identification_summary_json": os.path.join(id_dir, "identification_summary.json"),
        "legacy_identified_effects_csv": os.path.join(out_dir, "identified_effects.csv"),
        "legacy_identification_summary_json": os.path.join(out_dir, "identification_summary.json"),
    }
    effects = assets["identified_effects"]
    if len(effects) == 0 and len(getattr(effects, "columns", [])) == 0:
        effects = pd.DataFrame(columns=[
            "source", "target", "estimand", "identification_status",
            "adjustment_set", "frontdoor_set", "candidate_covariates",
            "post_treatment_columns", "forbidden_adjustment_set",
            "reason_codes", "authority_level", "insight_id",
        ])
    adjustment_sets = _build_adjustment_sets_frame(effects, bridge=bridge)
    effects.to_csv(paths["identified_effects_csv"], index=False)
    effects.to_csv(paths["legacy_identified_effects_csv"], index=False)
    adjustment_sets.to_csv(paths["adjustment_sets_csv"], index=False)
    with open(paths["identified_effects_json"], "w", encoding="utf-8") as f:
        json.dump(effects.to_dict(orient="records"), f, ensure_ascii=False, indent=2)
    with open(paths["adjustment_sets_json"], "w", encoding="utf-8") as f:
        json.dump(adjustment_sets.to_dict(orient="records"), f, ensure_ascii=False, indent=2)
    with open(paths["identification_summary_json"], "w", encoding="utf-8") as f:
        json.dump(assets["identification_summary"], f, ensure_ascii=False, indent=2)
    with open(paths["legacy_identification_summary_json"], "w", encoding="utf-8") as f:
        json.dump(assets["identification_summary"], f, ensure_ascii=False, indent=2)
    try:
        from contracts.causal_contract import write_causal_contract
        paths.update(write_causal_contract(out_dir=out_dir))
    except (OSError, ValueError, TypeError, RuntimeError, KeyError, ImportError) as exc:
        warnings.warn(f"[amantia][warning] causal contract sync after scm-identify failed: {type(exc).__name__}: {exc}", RuntimeWarning)
    return paths


if __name__ == "__main__":
    out = write_identification_assets()
    print("Saved identified effects:", out["identified_effects_csv"])
    print("Saved identification summary:", out["identification_summary_json"])
