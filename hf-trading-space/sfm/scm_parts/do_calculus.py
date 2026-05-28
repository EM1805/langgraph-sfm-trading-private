from __future__ import annotations

"""Explicit do-calculus rule diagnostics for Amantia SCM.

This module makes Pearl's three do-calculus rules visible as audited graph
checks.  It is intentionally conservative: a passed rule is evidence for a
single legal symbolic rewrite, not a complete Shpitser/Pearl ID proof by itself.
Full identification authority still lives in ``id_algorithm`` and the contract
layer.

Step 70-compatible behavior: rule diagnostics accept singleton nodes or
set-valued X/Y/Z/W.  A set-valued rule is applicable only when every required
pairwise d-separation check passes; otherwise the rule is blocked.
"""

from dataclasses import asdict, dataclass
import json
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .admg import ADMG
from .graph_criteria import d_separation_diagnostic, directed_cycle_nodes
from .do_ast import P_do

SET_VALUED_DO_CALCULUS_VERSION = "do_calculus_set_valued_v1"


def _s(value: object) -> str:
    raw = "" if value is None else str(value).strip()
    return "" if raw.lower() in {"nan", "none", "null"} else raw


def _dedupe(values: Iterable[object]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for value in values or []:
        item = _s(value)
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _nodes(value: object) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        # Preserve normal node names; split only explicit compact set syntax.
        if "|" in raw:
            return _dedupe(raw.split("|"))
        return _dedupe([raw])
    try:
        return _dedupe(list(value))  # type: ignore[arg-type]
    except TypeError:
        return _dedupe([value])


def _join(values: Iterable[object]) -> str:
    return "|".join(_dedupe(values))


def _comma(values: Sequence[str]) -> str:
    return ",".join(_dedupe(values))


def _drop_directed_edges(admg: ADMG, *, incoming_to: Sequence[str] = (), outgoing_from: Sequence[str] = ()) -> ADMG:
    incoming = set(_dedupe(incoming_to))
    outgoing = set(_dedupe(outgoing_from))
    directed: List[Tuple[str, str]] = []
    for a, b in admg.directed_edges:
        if b in incoming:
            continue
        if a in outgoing:
            continue
        directed.append((a, b))
    return ADMG(nodes=admg.nodes, directed_edges=tuple(directed), bidirected_edges=admg.bidirected_edges)


def _descendants_after_action(admg: ADMG, actions: Sequence[str]) -> Set[str]:
    """Descendants in G_{bar(actions)}: incoming arrows to actions removed."""
    acted = _dedupe(actions)
    if not acted:
        return set()
    g = _drop_directed_edges(admg, incoming_to=acted)
    return g.descendants(acted) - set(acted)


def _validate_sets(admg: ADMG, ys: Sequence[str], xs: Sequence[str], zs: Sequence[str]) -> Optional[str]:
    if not ys or not xs or not zs:
        return "MISSING_QUERY_NODE"
    all_nodes = list(ys) + list(xs) + list(zs)
    if any(v not in admg.node_set for v in all_nodes):
        return "MISSING_QUERY_NODE"
    if set(ys) & set(xs) or set(ys) & set(zs) or set(xs) & set(zs):
        return "OVERLAPPING_QUERY_SETS"
    return None


def _set_dsep(admg: ADMG, sources: Sequence[str], targets: Sequence[str], *, conditioned_on: Sequence[str]) -> Tuple[bool, str, int, int, str, str]:
    """Conservative all-pairs d-separation summary."""
    srcs = _dedupe(sources)
    tgts = _dedupe(targets)
    cond = _dedupe(conditioned_on)
    if not srcs or not tgts:
        return False, "invalid_query", 0, 0, "", "EMPTY_DSEP_SET"
    open_paths: List[str] = []
    total_open = 0
    total_checked = 0
    invalid = False
    for src in srcs:
        for dst in tgts:
            if src == dst:
                return False, "invalid_query", total_open, total_checked, "", "OVERLAPPING_DSEP_SETS"
            dsep = d_separation_diagnostic(admg, src, dst, conditioned_on=cond)
            total_open += int(dsep.open_path_count)
            total_checked += int(dsep.checked_path_count)
            if dsep.status == "invalid_query":
                invalid = True
            if not dsep.separated:
                open_paths.append(f"{src}->{dst}:{dsep.open_paths or dsep.reason_codes}")
    if invalid:
        return False, "invalid_query", total_open, total_checked, "|".join(open_paths[:16]), "MISSING_QUERY_NODE"
    if open_paths:
        return False, "not_separated", total_open, total_checked, "|".join(open_paths[:16]), "OPEN_DCONNECTING_PATHS"
    return True, "separated", total_open, total_checked, "", "D_SEPARATED"


@dataclass(frozen=True)
class DoCalculusRuleDiagnostic:
    rule: str
    applicable: bool
    status: str
    y: str = ""
    x: str = ""
    z: str = ""
    w: str = ""
    graph_variant: str = ""
    conditioned_on: str = ""
    removed_incoming_to: str = ""
    removed_outgoing_from: str = ""
    dsep_status: str = ""
    open_path_count: int = 0
    checked_path_count: int = 0
    open_paths: str = ""
    expression_before: str = ""
    expression_after: str = ""
    reason_codes: str = ""
    expression_before_ast_json: str = ""
    expression_after_ast_json: str = ""
    set_valued: int = 0
    y_set: str = ""
    x_set: str = ""
    z_set: str = ""
    w_set: str = ""
    set_rule_version: str = SET_VALUED_DO_CALCULUS_VERSION

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class DoCalculusDiagnostic:
    do_calculus_status: str
    rule1_applicable: int = 0
    rule2_applicable: int = 0
    rule3_applicable: int = 0
    applicable_rules: str = ""
    rule_trace_json: str = ""
    rule1_reason_codes: str = ""
    rule2_reason_codes: str = ""
    rule3_reason_codes: str = ""
    reason_codes: str = ""

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _expr_ast_json(*, y: object, interventions: Sequence[str] = (), observations: Sequence[str] = (), label: str = "") -> str:
    """Build a deterministic do-AST JSON mirror for a display formula."""
    return P_do(
        _nodes(y),
        interventions=interventions,
        observations=observations,
        label=label,
        metadata={"source": "do_calculus_rule_diagnostic", "authority": "audit_only"},
    ).to_json()


def _diag_common(
    *,
    rule: str,
    ok: bool,
    ys: Sequence[str],
    xs: Sequence[str],
    zs: Sequence[str],
    ww: Sequence[str],
    graph_variant: str,
    conditioned_on: Sequence[str],
    removed_incoming_to: Sequence[str],
    removed_outgoing_from: Sequence[str] = (),
    dsep_status: str = "",
    open_path_count: int = 0,
    checked_path_count: int = 0,
    open_paths: str = "",
    expression_before: str = "",
    expression_after: str = "",
    reason_codes: str = "",
) -> DoCalculusRuleDiagnostic:
    set_valued = int(len(ys) > 1 or len(xs) > 1 or len(zs) > 1 or len(ww) > 1)
    return DoCalculusRuleDiagnostic(
        rule,
        bool(ok),
        "applicable" if ok else "blocked",
        _join(ys),
        _join(xs),
        _join(zs),
        _join(ww),
        graph_variant,
        _join(conditioned_on),
        removed_incoming_to=_join(removed_incoming_to),
        removed_outgoing_from=_join(removed_outgoing_from),
        dsep_status=dsep_status,
        open_path_count=open_path_count,
        checked_path_count=checked_path_count,
        open_paths=open_paths,
        expression_before=expression_before,
        expression_after=expression_after,
        reason_codes=reason_codes,
        expression_before_ast_json=_expr_ast_json(y=ys, interventions=xs + ([] if rule == "rule1_observation_insertion_deletion" else zs), observations=(zs + ww if rule == "rule1_observation_insertion_deletion" else ww), label=f"{rule}_before"),
        expression_after_ast_json=(
            _expr_ast_json(y=ys, interventions=xs, observations=ww, label=f"{rule}_after")
            if rule != "rule2_action_observation_exchange"
            else _expr_ast_json(y=ys, interventions=xs, observations=zs + ww, label=f"{rule}_after")
        ),
        set_valued=set_valued,
        y_set=_join(ys),
        x_set=_join(xs),
        z_set=_join(zs),
        w_set=_join(ww),
    )


def rule1_insertion_deletion_observation(
    admg: ADMG,
    *,
    y: object,
    x: object,
    z: object,
    w: Optional[Sequence[object]] = None,
) -> DoCalculusRuleDiagnostic:
    """Rule 1: delete/insert observation Z if Y ⫫ Z | X,W in G_bar(X)."""
    ys, xs, zs = _nodes(y), _nodes(x), _nodes(z)
    ww = [v for v in _dedupe(w or []) if v not in set(ys + xs + zs)]
    err = _validate_sets(admg, ys, xs, zs)
    if err:
        return DoCalculusRuleDiagnostic("rule1_observation_insertion_deletion", False, "invalid_query", _join(ys), _join(xs), _join(zs), _join(ww), reason_codes=err)
    g = _drop_directed_edges(admg, incoming_to=xs)
    ok, dstatus, openc, checked, openp, dreason = _set_dsep(g, zs, ys, conditioned_on=xs + ww)
    before = f"P({_comma(ys)}|do({_comma(xs)}),{_comma(zs + ww)})" if (zs or ww) else f"P({_comma(ys)}|do({_comma(xs)}))"
    after = f"P({_comma(ys)}|do({_comma(xs)}){',' + _comma(ww) if ww else ''})"
    return _diag_common(
        rule="rule1_observation_insertion_deletion", ok=ok, ys=ys, xs=xs, zs=zs, ww=ww,
        graph_variant="G_bar_X", conditioned_on=xs + ww, removed_incoming_to=xs,
        dsep_status=dstatus, open_path_count=openc, checked_path_count=checked, open_paths=openp,
        expression_before=before, expression_after=after,
        reason_codes="RULE1_DSEP_Y_Z_G_BAR_X" if ok else dreason,
    )


def rule2_action_observation_exchange(
    admg: ADMG,
    *,
    y: object,
    x: object,
    z: object,
    w: Optional[Sequence[object]] = None,
) -> DoCalculusRuleDiagnostic:
    """Rule 2: exchange action/observation on Z if Y ⫫ Z | X,W in G_bar(X),underline(Z)."""
    ys, xs, zs = _nodes(y), _nodes(x), _nodes(z)
    ww = [v for v in _dedupe(w or []) if v not in set(ys + xs + zs)]
    err = _validate_sets(admg, ys, xs, zs)
    if err:
        return DoCalculusRuleDiagnostic("rule2_action_observation_exchange", False, "invalid_query", _join(ys), _join(xs), _join(zs), _join(ww), reason_codes=err)
    g = _drop_directed_edges(admg, incoming_to=xs, outgoing_from=zs)
    ok, dstatus, openc, checked, openp, dreason = _set_dsep(g, zs, ys, conditioned_on=xs + ww)
    before = f"P({_comma(ys)}|do({_comma(xs + zs)}){',' + _comma(ww) if ww else ''})"
    after = f"P({_comma(ys)}|do({_comma(xs)}),{_comma(zs + ww)})" if (zs or ww) else f"P({_comma(ys)}|do({_comma(xs)}))"
    return _diag_common(
        rule="rule2_action_observation_exchange", ok=ok, ys=ys, xs=xs, zs=zs, ww=ww,
        graph_variant="G_bar_X_under_Z", conditioned_on=xs + ww, removed_incoming_to=xs, removed_outgoing_from=zs,
        dsep_status=dstatus, open_path_count=openc, checked_path_count=checked, open_paths=openp,
        expression_before=before, expression_after=after,
        reason_codes="RULE2_DSEP_Y_Z_G_BAR_X_UNDER_Z" if ok else dreason,
    )


def rule3_insertion_deletion_action(
    admg: ADMG,
    *,
    y: object,
    x: object,
    z: object,
    w: Optional[Sequence[object]] = None,
) -> DoCalculusRuleDiagnostic:
    """Rule 3: delete/insert action Z if Y ⫫ Z | X,W in G_bar(X),bar(Z(W)).

    ``Z(W)`` is implemented conservatively: every Z must not be an ancestor of
    W in ``G_bar(X)``.  If any Z is an ancestor of W, the rule is blocked.
    """
    ys, xs, zs = _nodes(y), _nodes(x), _nodes(z)
    ww = [v for v in _dedupe(w or []) if v not in set(ys + xs + zs)]
    err = _validate_sets(admg, ys, xs, zs)
    if err:
        return DoCalculusRuleDiagnostic("rule3_action_insertion_deletion", False, "invalid_query", _join(ys), _join(xs), _join(zs), _join(ww), reason_codes=err)
    g_bar_x = _drop_directed_edges(admg, incoming_to=xs)
    ancestors_w = g_bar_x.ancestors(ww) if ww else set()
    z_for_bar = [zz for zz in zs if zz not in ancestors_w]
    g = _drop_directed_edges(g_bar_x, incoming_to=z_for_bar)
    ok_dsep, dstatus, openc, checked, openp, dreason = _set_dsep(g, zs, ys, conditioned_on=xs + ww)
    ok = bool(z_for_bar) and len(z_for_bar) == len(zs) and bool(ok_dsep)
    if ok:
        reason = "RULE3_DSEP_Y_Z_G_BAR_X_BAR_ZW"
    elif len(z_for_bar) != len(zs):
        reason = "Z_ANCESTOR_OF_W_RULE3_NOT_ALLOWED"
    else:
        reason = dreason
    before = f"P({_comma(ys)}|do({_comma(xs + zs)}){',' + _comma(ww) if ww else ''})"
    after = f"P({_comma(ys)}|do({_comma(xs)}){',' + _comma(ww) if ww else ''})"
    return _diag_common(
        rule="rule3_action_insertion_deletion", ok=ok, ys=ys, xs=xs, zs=zs, ww=ww,
        graph_variant="G_bar_X_bar_Z_not_ancestor_W", conditioned_on=xs + ww,
        removed_incoming_to=xs + z_for_bar, dsep_status=dstatus,
        open_path_count=openc, checked_path_count=checked, open_paths=openp,
        expression_before=before, expression_after=after, reason_codes=reason,
    )


def do_calculus_diagnostic(
    admg: ADMG,
    treatment: object,
    outcome: object,
    *,
    candidate_z: Optional[Sequence[object]] = None,
    conditioned_on: Optional[Sequence[object]] = None,
) -> DoCalculusDiagnostic:
    """Run explicit rule checks for an ID query without granting authority."""
    x, y = _s(treatment), _s(outcome)
    if not x or not y or x not in admg.node_set or y not in admg.node_set:
        return DoCalculusDiagnostic("invalid_query", reason_codes="MISSING_QUERY_NODE")
    if directed_cycle_nodes(admg):
        return DoCalculusDiagnostic("blocked_directed_cycle", reason_codes="DIRECTED_CYCLE_NOT_ADMG_DAG")

    z_values = [z for z in _dedupe(candidate_z or []) if z in admg.node_set and z not in {x, y}]
    if not z_values:
        # Prefer likely mediators/confounders so the audit is informative even without hints.
        z_values = sorted(admg.node_set - {x, y})[:8]

    trace: List[Dict[str, object]] = []
    app: List[str] = []
    r1 = r2 = r3 = 0
    r1_reason = r2_reason = r3_reason = ""
    for z in z_values:
        d1 = rule1_insertion_deletion_observation(admg, y=y, x=x, z=z, w=conditioned_on or [])
        d2 = rule2_action_observation_exchange(admg, y=y, x=x, z=z, w=conditioned_on or [])
        d3 = rule3_insertion_deletion_action(admg, y=y, x=x, z=z, w=conditioned_on or [])
        for d in (d1, d2, d3):
            trace.append(d.to_dict())
            if d.applicable:
                app.append(f"{d.rule}:{z}")
        r1 = r1 or int(d1.applicable); r2 = r2 or int(d2.applicable); r3 = r3 or int(d3.applicable)
        r1_reason = r1_reason or d1.reason_codes; r2_reason = r2_reason or d2.reason_codes; r3_reason = r3_reason or d3.reason_codes

    status = "rules_applicable_audit_only" if app else "no_rule_applicable_audit_only"
    return DoCalculusDiagnostic(
        status,
        rule1_applicable=int(r1), rule2_applicable=int(r2), rule3_applicable=int(r3),
        applicable_rules="|".join(app),
        rule_trace_json=json.dumps({"query": {"treatment": x, "outcome": y}, "rules": trace}, sort_keys=True),
        rule1_reason_codes=r1_reason, rule2_reason_codes=r2_reason, rule3_reason_codes=r3_reason,
        reason_codes="DO_CALCULUS_RULES_EXPLICIT_AUDIT_ONLY",
    )


__all__ = [
    "SET_VALUED_DO_CALCULUS_VERSION",
    "DoCalculusRuleDiagnostic",
    "DoCalculusDiagnostic",
    "rule1_insertion_deletion_observation",
    "rule2_action_observation_exchange",
    "rule3_insertion_deletion_action",
    "do_calculus_diagnostic",
]
