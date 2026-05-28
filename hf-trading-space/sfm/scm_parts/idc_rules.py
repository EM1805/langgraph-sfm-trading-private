from __future__ import annotations

"""Conservative IDC pruning/simplification helpers for SCM-ID.

Step 62 adds a small, auditable IDC simplification layer.  It does **not**
claim complete IDC/do-calculus.  It only drops conditioning variables that are
structurally disconnected from the queried outcomes after the intervention
variables are treated as fixed separators.

Rule implemented here:
    In the mixed-graph skeleton of G with treatment nodes removed, a condition
    node Z is prunable only when its connected component contains no outcome
    node.  Directed and bidirected edges are both treated as undirected for this
    conservative connectivity check.  If a condition shares a component with any
    outcome, it is kept.

This is intentionally stricter than full IDC: many removable conditions will be
kept, but conditions with any suspicious path/confounding signal will not be
silently removed.
"""

from collections import deque
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple

from .admg import ADMG
from .id_algorithm_common import _dedupe, _s

IDC_PRUNING_VERSION = "idc_pruning_v1_step62"
IDC_PRUNING_LEVEL = (
    "conservative_component_disconnectivity_after_doX_removes_only_conditions_"
    "whose_mixed_skeleton_component_excludes_all_outcomes_no_full_id_or_full_idc_claim"
)


@dataclass(frozen=True)
class IDCPruningDiagnostic:
    """Audit payload for Step-62 IDC condition pruning."""

    version: str
    level: str
    status: str
    treatments: Tuple[str, ...]
    outcomes: Tuple[str, ...]
    original_conditions: Tuple[str, ...]
    kept_conditions: Tuple[str, ...]
    pruned_conditions: Tuple[str, ...]
    rule: str
    component_map: Mapping[str, List[str]]
    reason_codes: str
    simplification_used: int = 0

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)



def _tuple(values: Iterable[object]) -> Tuple[str, ...]:
    return tuple(_dedupe(values or []))



def _component_adjacency_after_do(admg: ADMG, treatments: Sequence[str]) -> Dict[str, Set[str]]:
    """Mixed-graph skeleton adjacency after treating treatments as separators."""
    blocked = set(_tuple(treatments))
    nodes = sorted(n for n in admg.node_set if n not in blocked)
    adj: Dict[str, Set[str]] = {n: set() for n in nodes}
    for a, b in admg.directed_edges:
        if a in blocked or b in blocked:
            continue
        if a in adj and b in adj:
            adj[a].add(b)
            adj[b].add(a)
    for a, b in admg.bidirected_edges:
        if a in blocked or b in blocked:
            continue
        if a in adj and b in adj:
            adj[a].add(b)
            adj[b].add(a)
    return adj



def _components(adj: Mapping[str, Set[str]]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    seen: Set[str] = set()
    for node in sorted(adj):
        if node in seen:
            continue
        comp: Set[str] = set()
        q = deque([node])
        seen.add(node)
        while q:
            cur = q.popleft()
            comp.add(cur)
            for nxt in adj.get(cur, set()):
                if nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)
        comp_sorted = sorted(comp)
        for item in comp_sorted:
            out[item] = comp_sorted
    return out



def idc_pruning_diagnostic(
    admg: ADMG,
    treatments: Sequence[object],
    outcomes: Sequence[object],
    conditions: Sequence[object],
) -> IDCPruningDiagnostic:
    """Return the conservative Step-62 condition-pruning decision."""
    x = _tuple(treatments)
    y = _tuple(outcomes)
    z = _tuple(conditions)
    y_set = set(y)

    adj = _component_adjacency_after_do(admg, x)
    comp_by_node = _components(adj)
    kept: List[str] = []
    pruned: List[str] = []
    component_map: Dict[str, List[str]] = {}

    for cond in z:
        comp = comp_by_node.get(cond, [cond] if cond in admg.node_set and cond not in set(x) else [])
        component_map[cond] = list(comp)
        if set(comp) & y_set:
            kept.append(cond)
        else:
            pruned.append(cond)

    if not pruned:
        status = "idc_pruning_no_conditions_removed_step62"
        reason = "IDC_STEP62_NO_CONDITION_COMPONENT_DISCONNECTED_FROM_OUTCOMES"
        used = 0
    elif not kept:
        status = "idc_pruning_all_conditions_removed_step62"
        reason = "IDC_STEP62_ALL_CONDITION_COMPONENTS_DISCONNECTED_FROM_OUTCOMES_AFTER_DOX"
        used = 1
    else:
        status = "idc_pruning_partial_conditions_removed_step62"
        reason = "IDC_STEP62_SOME_CONDITION_COMPONENTS_DISCONNECTED_FROM_OUTCOMES_AFTER_DOX"
        used = 1

    return IDCPruningDiagnostic(
        version=IDC_PRUNING_VERSION,
        level=IDC_PRUNING_LEVEL,
        status=status,
        treatments=x,
        outcomes=y,
        original_conditions=z,
        kept_conditions=tuple(kept),
        pruned_conditions=tuple(pruned),
        rule="remove_condition_if_its_mixed_skeleton_component_after_removing_treatments_contains_no_outcome",
        component_map=component_map,
        reason_codes=reason,
        simplification_used=used,
    )


__all__ = [
    "IDC_PRUNING_VERSION",
    "IDC_PRUNING_LEVEL",
    "IDCPruningDiagnostic",
    "idc_pruning_diagnostic",
]
