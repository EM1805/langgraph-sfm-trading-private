from __future__ import annotations

"""Recursive expression layer for the conservative SCM ID algorithm.

This module owns the set-valued recursive-ID expression logic.  The legacy
public entrypoints remain re-exported from ``id_algorithm.py`` so downstream
imports stay compatible.
"""

from dataclasses import asdict, dataclass
import json
import re
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from .admg import ADMG
from .graph_criteria import directed_cycle_nodes, directed_path_exists, topological_order
from .id_ast import (
    FormulaAST,
    Do,
    HedgeFail,
    P,
    Placeholder,
    Product,
    Q,
    Sum,
    ast_from_dict,
    ast_from_formula_parts,
    parse_factor_term,
    payload_to_ast,
)
from .id_algorithm_common import (
    _conditional_factor,
    _dedupe,
    _format_component,
    _format_components,
    _joint_symbol,
    _json_formula,
    _s,
)
from .id_carried_q import build_carried_q_context
from .id_ast_normalizer import ID_AST_NORMALIZER_VERSION, normalize_formula_ast


def _proof_step(name: str, status: str, **fields: object) -> Dict[str, object]:
    payload: Dict[str, object] = {"step": name, "status": status}
    for key, value in fields.items():
        if value not in (None, "", [], {}, ()): 
            payload[key] = value
    return payload

@dataclass(frozen=True)
class RecursiveIDExpressionDiagnostic:
    """Executable conservative recursive-ID expression layer.

    This is not the full Shpitser/Pearl ID algorithm yet.  It implements the
    safe recursive branches that Amantia can audit today:

    - no-intervention base case;
    - graphical-zero effect;
    - ancestor reduction;
    - W-step promotion of irrelevant post-intervention variables;
    - observed-DAG truncated factorization;
    - district decomposition over ``G[V\\X]``;
    - full-district q-factor extraction from the chain rule;
    - formal hedge certificate construction for the canonical single-district fail
      case.

    Step 5 adds a structured Q-input recursion attempt for strict subdistricts.
    It can identify additional safe recursive Q branches and can bubble formal
    hedge failures from the carried-Q subproblem; unsupported Q-input shapes
    remain blocked explicitly instead of being overclaimed.
    """

    expression_status: str
    expression_identified: bool = False
    formula: str = ""
    expression_json: str = ""
    trace_json: str = ""
    depth: int = 0
    blocker: str = ""
    blocker_class: str = ""
    pending_operator: str = ""
    reason_codes: str = ""

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _as_sorted_node_set(admg: ADMG, values: Iterable[object]) -> List[str]:
    return sorted(v for v in _dedupe(values) if v in admg.node_set)


def _no_directed_path_between(admg: ADMG, sources: Sequence[str], targets: Sequence[str]) -> bool:
    src = [s for s in _dedupe(sources) if s in admg.node_set]
    dst = [t for t in _dedupe(targets) if t in admg.node_set]
    if not src or not dst:
        return False
    return not any(directed_path_exists(admg, s, t) for s in src for t in dst if s != t)


def _remove_incoming_to(admg: ADMG, nodes: Sequence[str]) -> ADMG:
    blocked = set(_dedupe(nodes))
    return ADMG(
        nodes=admg.nodes,
        directed_edges=tuple((a, b) for a, b in admg.directed_edges if b not in blocked),
        bidirected_edges=admg.bidirected_edges,
    )


def _joint_probability_symbol(nodes: Sequence[str]) -> str:
    joint = _joint_symbol(nodes)
    return f"P({joint})" if joint else "1"


def _formula_rhs(formula: str) -> str:
    text = _s(formula)
    return text.split(" = ", 1)[1] if " = " in text else text


def _conditional_factor_from_previous(node: str, previous: Sequence[str]) -> str:
    prev = _joint_symbol(previous)
    return f"P({node} | {prev})" if prev else f"P({node})"


def _truncated_factorization_formula_for_sets(admg: ADMG, x_set: Sequence[str], y_set: Sequence[str]) -> Tuple[str, List[str], List[str], List[str]]:
    order = topological_order(admg)
    if not order:
        return "", [], [], []
    x = set(_dedupe(x_set))
    y = set(_dedupe(y_set))
    parents = admg.parents()
    eliminated = [n for n in order if n not in x and n not in y]
    retained_terms = [_conditional_factor(n, sorted(parents.get(n, set()))) for n in order if n not in x]
    removed_terms = [_conditional_factor(n, sorted(parents.get(n, set()))) for n in order if n in x]
    product = " * ".join(retained_terms) if retained_terms else "1"
    lhs_y = _joint_symbol([n for n in order if n in y]) or _joint_symbol(y_set)
    lhs_x = _joint_symbol([n for n in order if n in x]) or _joint_symbol(x_set)
    formula = f"P_{{do({lhs_x})}}({lhs_y}) = "
    formula += f"sum_{{{','.join(eliminated)}}} {product}" if eliminated else product
    return formula, eliminated, retained_terms, removed_terms



def _q_factor_formula_for_district(admg: ADMG, district: Sequence[str], y_set: Sequence[str]) -> Tuple[str, List[str], List[str]]:
    """Return the q-factor for a district that is also a full district of G."""
    order = topological_order(admg)
    if not order:
        return "", [], []
    s = set(_dedupe(district))
    y = set(_dedupe(y_set))
    previous: List[str] = []
    terms: List[str] = []
    for node in order:
        if node in s:
            terms.append(_conditional_factor_from_previous(node, previous))
        previous.append(node)
    sum_over = [node for node in order if node in s and node not in y]
    product = " * ".join(terms) if terms else "1"
    formula = f"sum_{{{','.join(sum_over)}}} {product}" if sum_over else product
    return formula, sum_over, terms


def _factor_lhs_variable(term: str) -> str:
    """Best-effort extraction of the left variable from a compact P(v | ...)."""
    text = _s(term)
    if not text.startswith("P(") or not text.endswith(")"):
        return ""
    body = text[2:-1]
    lhs = body.split("|", 1)[0].strip()
    lhs = lhs.split(",", 1)[0].strip()
    return lhs


def _terms_for_nodes_from_q_input(q_input_terms: Sequence[str], nodes: Sequence[str]) -> List[str]:
    wanted = set(_dedupe(nodes))
    out: List[str] = []
    for term in q_input_terms or []:
        clean = _s(term)
        lhs = _factor_lhs_variable(clean)
        if lhs in wanted:
            out.append(clean)
    return out


def _chain_rule_terms_for_scope(admg: ADMG, scope: Sequence[str]) -> List[str]:
    """Return original-order chain-rule terms that define Q[scope]."""
    order = topological_order(admg)
    scope_set = set(_dedupe(scope))
    previous: List[str] = []
    terms: List[str] = []
    for node in order:
        if node in scope_set:
            terms.append(_conditional_factor_from_previous(node, previous))
        previous.append(node)
    return terms


def _q_factor_formula_for_district_from_terms(
    admg: ADMG,
    district: Sequence[str],
    y_set: Sequence[str],
    q_input_terms: Sequence[str] = (),
) -> Tuple[str, List[str], List[str]]:
    """Return a district expression using carried Q-input terms when present."""
    order = topological_order(admg)
    if not order:
        return "", [], []
    s = set(_dedupe(district))
    y = set(_dedupe(y_set))
    terms = _terms_for_nodes_from_q_input(q_input_terms, sorted(s)) if q_input_terms else []
    if not terms:
        previous: List[str] = []
        for node in order:
            if node in s:
                terms.append(_conditional_factor_from_previous(node, previous))
            previous.append(node)
    sum_over = [node for node in order if node in s and node not in y]
    product = " * ".join(terms) if terms else "1"
    formula = f"sum_{{{','.join(sum_over)}}} {product}" if sum_over else product
    return formula, sum_over, terms


def _q_input_recursion_payload(
    *,
    containing_district: Sequence[str],
    subdistrict: Sequence[str],
    q_input_name: str,
    q_input_terms: Sequence[str],
    recursive_status: str,
    recursive_identified: bool,
    recursive_formula: str,
    recursive_expression: Mapping[str, object],
    recursive_reason_codes: str,
    carried_q_context: Mapping[str, object] | None = None,
) -> Dict[str, object]:
    payload = {
        "containing_district": list(_dedupe(containing_district)),
        "subdistrict": list(_dedupe(subdistrict)),
        "q_input": q_input_name,
        "q_input_terms": list(q_input_terms),
        "recursive_status": recursive_status,
        "recursive_identified": bool(recursive_identified),
        "recursive_formula": recursive_formula,
        "recursive_expression": dict(recursive_expression),
        "recursive_reason_codes": recursive_reason_codes,
        "rule": "general_q_input_recursive_subproblem_step44",
        "step51_carried_q_context_enabled": 1,
        "step53_operational_carried_q_ast_enabled": int(bool(carried_q_context and isinstance(carried_q_context.get("formula_ast") if isinstance(carried_q_context, Mapping) else None, Mapping))),
    }
    if carried_q_context:
        payload["carried_q_context"] = dict(carried_q_context)
        if isinstance(carried_q_context.get("formula_ast"), Mapping):
            payload["q_input_formula_ast"] = dict(carried_q_context["formula_ast"])
    return payload


def _primed_name(node: str) -> str:
    return f"{node}'"


def _replace_symbol_token(text: str, old: str, new: str) -> str:
    """Replace a variable token in human-readable probability formulas."""
    if not old:
        return text
    return re.sub(rf"(?<![A-Za-z0-9_]){re.escape(old)}(?![A-Za-z0-9_])", new, text)


def _prime_bound_symbols(text: str, bound_nodes: Sequence[str]) -> str:
    out = text
    for node in sorted(_dedupe(bound_nodes), key=len, reverse=True):
        out = _replace_symbol_token(out, node, _primed_name(node))
    return out


def _q_input_subdistrict_formula(
    admg: ADMG,
    subdistrict: Sequence[str],
    containing_district: Sequence[str],
    y_set: Sequence[str],
    x_set: Sequence[str],
) -> Tuple[bool, str, List[str], List[str], Dict[str, object], str]:
    """Identify a safe Step-7 style subdistrict with a carried Q input.

    This is a narrow, auditable version of the Shpitser-Pearl ID branch that
    recurses inside a larger district ``S'`` using its c-factor ``Q[S']``.  It
    is intentionally limited to the safe ancestral-reduction case where the
    active intervention variables inside ``S'`` are *not* ancestors of the
    requested outcome inside ``G[S']``.  That is enough to derive the classic
    front-door formula from recursive ID, while direct ``X <-> Y`` confounding
    remains blocked by the hedge branch.
    """
    order = topological_order(admg)
    if not order:
        return False, "", [], [], {}, "DIRECTED_CYCLE_NOT_ADMG_DAG"

    s = set(_dedupe(subdistrict))
    sp = set(_dedupe(containing_district))
    y_inside = [n for n in _dedupe(y_set) if n in sp]
    x_inside = [n for n in _dedupe(x_set) if n in sp and n not in set(y_inside)]
    if not s or not sp or not s.issubset(sp) or s == sp or not y_inside:
        return False, "", [], [], {}, "Q_INPUT_SUBDISTRICT_SHAPE_UNSUPPORTED"

    working = admg.induced_subgraph(sorted(sp))
    ancestors_y = sorted(working.ancestors(y_inside))
    active_x_ancestors = sorted(set(x_inside) & set(ancestors_y))
    if active_x_ancestors:
        return (
            False,
            "",
            [],
            [],
            {
                "containing_district": sorted(sp),
                "subdistrict": sorted(s),
                "active_intervention_ancestors": active_x_ancestors,
                "ancestors_inside_containing_district": ancestors_y,
            },
            "Q_INPUT_RECURSION_BLOCKED_ACTIVE_X_REMAINS_ANCESTOR",
        )

    previous: List[str] = []
    raw_terms: List[str] = []
    for node in order:
        if node in sp:
            raw_terms.append(_conditional_factor_from_previous(node, previous))
        previous.append(node)

    sum_over = [node for node in order if node in sp and node not in set(y_inside)]
    # Only active intervention symbols need alpha-renaming (X -> X') so
    # fixed do-values are not confused with the summed observational copy.
    # Non-intervention summation variables keep their own names; otherwise the
    # summation header and displayed factors can drift apart (e.g. sum_M terms
    # containing M'). This is display/algebra hygiene, not a new ID authority.
    primed_sum_over = [_primed_name(node) if node in set(x_inside) else node for node in sum_over]
    displayed_terms = [_prime_bound_symbols(term, x_inside) for term in raw_terms]
    product = " * ".join(displayed_terms) if displayed_terms else "1"
    formula = f"sum_{{{','.join(primed_sum_over)}}} {product}" if primed_sum_over else product
    is_multi_intervention = len(x_inside) > 1
    reason = (
        "Q_INPUT_MULTI_INTERVENTION_SUBDISTRICT_RECURSION_IDENTIFIED_STEP29"
        if is_multi_intervention
        else "Q_INPUT_SUBDISTRICT_RECURSION_IDENTIFIED_STEP23"
    )
    payload = {
        "containing_district": sorted(sp),
        "subdistrict": sorted(s),
        "q_input": f"Q[{','.join(sorted(sp))}]",
        "q_input_raw_terms": raw_terms,
        "q_input_display_terms": displayed_terms,
        "sum_over": primed_sum_over,
        "alpha_renamed_interventions": [_primed_name(node) for node in x_inside],
        "multi_intervention_q_input": bool(is_multi_intervention),
        "ancestors_inside_containing_district": ancestors_y,
        "active_interventions_inside_containing_district": x_inside,
        "rule": "safe_q_input_ancestral_reduction",
        "step29_scope": "set_valued_treatment_q_input_recursion" if is_multi_intervention else "",
    }
    return True, formula, primed_sum_over, displayed_terms, payload, reason

def _expression_payload(
    *,
    kind: str,
    y_set: Sequence[str],
    x_set: Sequence[str],
    formula: str,
    sum_over: Sequence[str] = (),
    product_terms: Sequence[str] = (),
    subexpressions: Sequence[Mapping[str, object]] = (),
    districts: Sequence[Sequence[str]] = (),
    formal_hedge: Optional[Mapping[str, object]] = None,
    reason_codes: str = "",
    formula_ast: Optional[FormulaAST] = None,
) -> str:
    payload: Dict[str, object] = {
        "type": kind,
        "estimand": {"outcome": list(_dedupe(y_set)), "intervention": list(_dedupe(x_set))},
        "formula": formula,
        "sum_over": list(_dedupe(sum_over)),
        "product_terms": list(product_terms),
        "districts": [list(_dedupe(d)) for d in districts],
        "subexpressions": list(subexpressions),
        "reason_codes": reason_codes,
    }
    if formal_hedge:
        payload["formal_hedge_candidate"] = dict(formal_hedge)

    # Step 4 toward Full ID: supported recursive branches build their AST at
    # runtime and pass it here directly.  The Step-3 compatibility parser remains
    # as a non-authoritative fallback for older/blocked payloads.
    try:
        if formula_ast is not None:
            ast = normalize_formula_ast(formula_ast)
            payload["formula_ast_source"] = "internal_recursive_builder_step43"
            payload["formula_ast_runtime_used"] = 1
        else:
            ast = ast_from_formula_parts(
                kind=kind,
                y_set=y_set,
                x_set=x_set,
                formula=formula,
                sum_over=sum_over,
                product_terms=product_terms,
                subexpressions=subexpressions,
                districts=districts,
                formal_hedge=formal_hedge,
                reason_codes=reason_codes,
            )
            ast = normalize_formula_ast(ast)
            payload["formula_ast_source"] = "compatibility_parser_step42_normalized_step54"
            payload["formula_ast_runtime_used"] = 0
        payload["formula_ast"] = ast.to_dict()
        payload["formula_ast_version"] = "id_ast_v1"
        payload["formula_ast_normalized"] = 1
        payload["formula_ast_normalizer_version"] = ID_AST_NORMALIZER_VERSION
    except Exception as exc:  # pragma: no cover - audit-only fallback
        payload["formula_ast_error"] = f"{type(exc).__name__}:{exc}"
    return _json_formula(payload)


def _wrap_do_ast(x_set: Sequence[str], expr: FormulaAST) -> FormulaAST:
    """Wrap a runtime-built AST with the requested intervention set."""
    x = _dedupe(x_set)
    return Do(x, expr, label="estimand_do") if x else expr


def _product_terms_ast(terms: Sequence[str], *, label: str = "runtime_product") -> FormulaAST:
    """Build a product AST from already-created probability term strings.

    Step 4 keeps the human-readable display formula stable, but the recursive
    engine now carries structured factors internally so later Full-ID Q inputs do
    not need to parse free text at the point where recursion happens.
    """
    factors = [parse_factor_term(term) for term in terms or []]
    return Product(factors or [Placeholder("1", metadata={"constant": 1})], label=label)


def _sum_if_needed_ast(sum_over: Sequence[str], expr: FormulaAST) -> FormulaAST:
    bound = _dedupe(sum_over)
    return Sum(bound, expr, label="sum_over") if bound else expr


def _formula_ast_from_mapping(payload: object) -> Optional[FormulaAST]:
    """Safely recover a FormulaAST carried as q-input metadata.

    Step 53 makes the carried-Q AST operational: when a recursive subproblem
    receives Q[S'] as its input distribution, supported base/full-district
    cases use that AST directly instead of rebuilding authority only from
    display strings.  Invalid payloads return None so the legacy conservative
    term path remains the fallback.
    """
    if not isinstance(payload, Mapping):
        return None
    try:
        return ast_from_dict(payload)
    except Exception:  # pragma: no cover - defensive audit fallback
        return None


def _with_operational_q_metadata(ast: FormulaAST, *, q_input: str, q_scope: Sequence[str], source: str = "") -> FormulaAST:
    metadata = dict(ast.metadata) if isinstance(ast.metadata, Mapping) else {}
    metadata.update({
        "operational_carried_q_ast_step53": 1,
        "q_input": q_input,
        "q_input_scope": list(_dedupe(q_scope)),
    })
    if source:
        metadata["operational_source"] = source
    return FormulaAST(
        ast.node_type,
        variables=ast.variables,
        conditioned_on=ast.conditioned_on,
        interventions=ast.interventions,
        bound_variables=ast.bound_variables,
        children=ast.children,
        label=ast.label or "operational_carried_q_input",
        metadata=metadata,
    )


def _operational_q_input_ast(q_input_formula_ast: object, *, q_input: str, q_scope: Sequence[str], fallback_terms: Sequence[str], fallback_label: str) -> FormulaAST:
    carried = _formula_ast_from_mapping(q_input_formula_ast)
    if carried is not None:
        return _with_operational_q_metadata(carried, q_input=q_input, q_scope=q_scope, source="carried_q_formula_ast")
    return Q(q_scope, terms=(_product_terms_ast(fallback_terms, label=f"{fallback_label}_product"),), q_input=q_input, label=fallback_label)


def _project_q_input_ast_for_scope(
    admg: ADMG,
    scope: Sequence[str],
    *,
    source_scope: Sequence[str],
    source_terms: Sequence[str],
    source_name: str,
) -> Optional[Mapping[str, object]]:
    clean_scope = [n for n in _dedupe(scope) if n in admg.node_set]
    if not clean_scope:
        return None
    ctx = build_carried_q_context(
        admg,
        clean_scope,
        source_scope=source_scope,
        source_terms=source_terms,
        source_name=source_name,
        name=f"Q[{','.join(clean_scope)}]",
    )
    return dict(ctx.formula_ast)


def _q_runtime_ast(
    district: Sequence[str],
    terms: Sequence[str],
    sum_over: Sequence[str] = (),
    *,
    q_input: str = "",
    label: str = "q_factor",
) -> FormulaAST:
    product = _product_terms_ast(terms, label=f"{label}_product")
    inner = _sum_if_needed_ast(sum_over, product)
    return Q(district, terms=(inner,), q_input=q_input, label=label)


def _subexpression_runtime_ast(expr: Mapping[str, object]) -> FormulaAST:
    """Recover a child AST from a recursive subproblem payload."""
    try:
        return payload_to_ast(expr)
    except Exception:  # pragma: no cover - defensive audit fallback
        return Placeholder("unparsed_subexpression", metadata={"raw_keys": sorted(str(k) for k in expr.keys())})


def _formal_hedge_certificate_payload(admg: ADMG, s: Sequence[str], x_set: Sequence[str], y_set: Sequence[str]) -> Dict[str, object]:
    """Return a limited formal hedge certificate for an ID fail branch.

    This remains deliberately conservative, but Step 28 makes the certificate
    local to the relevant bidirected district instead of requiring the whole
    graph to be one district.  That matches recursive ID better: a district
    decomposition subproblem may fail because a *containing* district ``F`` has a
    strict ``G[V\\X]`` district ``F'``.  We certify only the auditable shape:

    - ``F`` is a full bidirected district of the current graph;
    - ``F'`` is the current remaining district ``s`` and is a strict subset of
      ``F``;
    - ``F'`` is a district in ``G[V\\X]``;
    - a treatment node lies in ``F \\ F'``;
    - the requested outcome intersects ``F'``;
    - ``F'`` is disjoint from the treatment set.

    Unsupported shapes still return ``{}`` instead of overclaiming.
    """
    s_set = set(_dedupe(s))
    x = set(_dedupe(x_set))
    y = set(_dedupe(y_set))
    full_districts = admg.districts()
    if not s_set or s_set & x or not (y & s_set):
        return {}

    gx_nodes = sorted(admg.node_set - x)
    gx = admg.induced_subgraph(gx_nodes)
    f_prime_is_gx_district = any(set(d) == s_set for d in gx.districts())
    if not f_prime_is_gx_district:
        return {}

    candidate_fs: List[Set[str]] = []
    for district in full_districts:
        f_set = set(district)
        if not s_set < f_set:
            continue
        if x & (f_set - s_set):
            candidate_fs.append(f_set)

    if not candidate_fs:
        return {}

    # Deterministic and conservative: prefer the smallest containing district.
    f_set = sorted(candidate_fs, key=lambda fs: (len(fs), sorted(fs)))[0]
    removed = sorted(f_set - s_set)
    scope = (
        "single_full_district_strict_g_without_x_district"
        if len(full_districts) == 1
        else "localized_full_district_strict_g_without_x_district"
    )
    checks = {
        "F_is_full_district_in_G": True,
        "F_is_single_district_in_G": len(full_districts) == 1,
        "F_prime_is_district_in_G_without_X": True,
        "F_prime_strict_subset_of_F": True,
        "F_intersects_treatment": True,
        "F_prime_disjoint_from_treatment": True,
        "outcome_intersects_F_prime": True,
        "localized_certificate": len(full_districts) != 1,
    }
    return {
        "hedge_status": "formal_hedge_certificate",
        "certificate_status": "valid_limited_formal_hedge_certificate",
        "F": sorted(f_set),
        "F_prime": sorted(s_set),
        "treatment_in_F_minus_F_prime": sorted(x & set(removed)),
        "outcome_in_F_prime": sorted(y & s_set),
        "graph_without_treatment_nodes": gx_nodes,
        "all_full_districts": [sorted(d) for d in full_districts],
        "checks": checks,
        "scope": scope,
        "reason_codes": "FORMAL_HEDGE_CERTIFICATE_LOCAL_DISTRICT_FAIL_BRANCH_STEP28",
    }


# Backwards-compatible internal alias for older comments/tests that still use
# the previous candidate wording. The returned payload is now a certificate
# when the limited formal checks pass.
def _formal_hedge_candidate_payload(admg: ADMG, s: Sequence[str], x_set: Sequence[str], y_set: Sequence[str]) -> Dict[str, object]:
    return _formal_hedge_certificate_payload(admg, s, x_set, y_set)

def _recursive_id_expression(
    admg: ADMG,
    y_set: Sequence[str],
    x_set: Sequence[str],
    *,
    depth: int = 0,
    max_depth: int = 8,
    trace: Optional[List[Dict[str, object]]] = None,
    seen: Optional[Set[Tuple[Tuple[str, ...], Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]]]] = None,
    q_input_scope: Sequence[str] = (),
    q_input_terms: Sequence[str] = (),
    q_input_name: str = "",
    q_input_formula_ast: Optional[Mapping[str, object]] = None,
) -> RecursiveIDExpressionDiagnostic:
    """Execute safe recursive-ID branches for set-valued queries."""
    trace = [] if trace is None else trace
    seen = set() if seen is None else seen
    y = _as_sorted_node_set(admg, y_set)
    x = [n for n in _as_sorted_node_set(admg, x_set) if n not in set(y)]
    v = sorted(admg.node_set)
    q_scope = tuple(_dedupe(q_input_scope))
    q_terms = tuple(_s(t) for t in q_input_terms or () if _s(t))
    q_name = _s(q_input_name) or (f"Q[{','.join(q_scope)}]" if q_scope else "")
    key = (tuple(v), tuple(y), tuple(x), q_scope, tuple(q_terms))
    q_ast_enabled = int(isinstance(q_input_formula_ast, Mapping) and bool(q_input_formula_ast))
    trace.append(_proof_step("recursive_id_enter", "entered", depth=depth, y="|".join(y), x="|".join(x), nodes="|".join(v), q_input=q_name, q_input_formula_ast_enabled=q_ast_enabled))

    if depth > max_depth:
        trace.append(_proof_step("recursive_id_depth", "blocked", depth=depth, max_depth=max_depth))
        return RecursiveIDExpressionDiagnostic("blocked_max_recursive_depth", False, depth=depth, trace_json=_json_formula({"trace": trace}), blocker="MAX_RECURSIVE_DEPTH", blocker_class="max_depth", pending_operator="increase_or_debug_recursive_depth", reason_codes="MAX_RECURSIVE_DEPTH")
    if not y:
        trace.append(_proof_step("validate_query", "blocked", reason_codes="MISSING_OUTCOME_NODE"))
        return RecursiveIDExpressionDiagnostic("invalid_query", False, depth=depth, trace_json=_json_formula({"trace": trace}), blocker="MISSING_OUTCOME_NODE", blocker_class="invalid_query", pending_operator="validate_query", reason_codes="MISSING_OUTCOME_NODE")
    if key in seen:
        trace.append(_proof_step("recursive_id_cycle_guard", "blocked", reason_codes="RECURSIVE_STATE_REVISITED"))
        return RecursiveIDExpressionDiagnostic("blocked_recursive_state_revisited", False, depth=depth, trace_json=_json_formula({"trace": trace}), blocker="RECURSIVE_STATE_REVISITED", blocker_class="recursion_cycle_guard", pending_operator="debug_recursive_state", reason_codes="RECURSIVE_STATE_REVISITED")
    seen.add(key)

    cycles = directed_cycle_nodes(admg)
    if cycles:
        trace.append(_proof_step("directed_acyclicity", "blocked", directed_cycle_nodes="|".join(cycles)))
        return RecursiveIDExpressionDiagnostic("blocked_directed_cycle", False, depth=depth, trace_json=_json_formula({"trace": trace}), blocker="|".join(cycles), blocker_class="directed_cycle", pending_operator="repair_or_reject_cyclic_graph", reason_codes="DIRECTED_CYCLE_NOT_ADMG_DAG")

    if not x:
        order = topological_order(admg) or v
        sum_over = [n for n in order if n not in set(y)]
        if q_terms:
            product_terms = list(q_terms)
            product = " * ".join(product_terms) if product_terms else "1"
            formula = f"sum_{{{','.join(sum_over)}}} {product}" if sum_over else product
            q_ast = _operational_q_input_ast(q_input_formula_ast, q_input=q_name, q_scope=q_scope or tuple(v), fallback_terms=product_terms, fallback_label="q_input_no_intervention")
            formula_ast = _sum_if_needed_ast(sum_over, q_ast)
            reason = "Q_INPUT_NO_INTERVENTION_BASE_CASE_STEP53" if q_ast_enabled else "Q_INPUT_NO_INTERVENTION_BASE_CASE_STEP44"
        else:
            product_terms = [f"P({','.join(order)})"]
            formula = f"sum_{{{','.join(sum_over)}}} P({','.join(order)})" if sum_over else _joint_probability_symbol(y)
            formula_ast = _sum_if_needed_ast(sum_over, P(order, label="observational_joint"))
            reason = "NO_INTERVENTION_BASE_CASE"
        trace.append(_proof_step("id_base_no_intervention", "identified", sum_over="|".join(sum_over), q_input=q_name, formula_ast_source="operational_carried_q_ast_step53" if q_ast_enabled and q_terms else "internal_recursive_builder_step43"))
        return RecursiveIDExpressionDiagnostic("identified_no_intervention_base_case", True, formula, expression_json=_expression_payload(kind="no_intervention", y_set=y, x_set=x, formula=formula, sum_over=sum_over, product_terms=product_terms, formula_ast=formula_ast, reason_codes=reason), trace_json=_json_formula({"trace": trace}), depth=depth, pending_operator="none", reason_codes=reason)

    if _no_directed_path_between(admg, x, y):
        formula = _joint_probability_symbol(y)
        trace.append(_proof_step("graphical_zero_effect", "identified", formula=formula, formula_ast_source="internal_recursive_builder_step43"))
        formula_ast = _wrap_do_ast(x, P(y, label="zero_effect_marginal"))
        return RecursiveIDExpressionDiagnostic("identified_graphical_zero_effect", True, formula, expression_json=_expression_payload(kind="graphical_zero_effect", y_set=y, x_set=x, formula=formula, product_terms=[formula], reason_codes="NO_DIRECTED_PATH_ZERO_EFFECT", formula_ast=formula_ast), trace_json=_json_formula({"trace": trace}), depth=depth, pending_operator="none", reason_codes="NO_DIRECTED_PATH_ZERO_EFFECT")

    ancestors_y = sorted(admg.ancestors(y))
    removed_non_ancestors = sorted(set(v) - set(ancestors_y))
    if removed_non_ancestors:
        trace.append(_proof_step("ancestor_reduction", "applied", ancestral_nodes="|".join(ancestors_y), removed_non_ancestors="|".join(removed_non_ancestors)))
        reduced = admg.induced_subgraph(ancestors_y)
        reduced_x = [n for n in x if n in set(ancestors_y)]
        reduced_q_terms = _terms_for_nodes_from_q_input(q_terms, ancestors_y) if q_terms else []
        reduced_q_scope = [n for n in q_scope if n in set(ancestors_y)]
        reduced_q_ast = _project_q_input_ast_for_scope(reduced, reduced_q_scope, source_scope=q_scope, source_terms=q_terms, source_name=q_name) if reduced_q_terms else None
        return _recursive_id_expression(reduced, y, reduced_x, depth=depth + 1, max_depth=max_depth, trace=trace, seen=seen, q_input_scope=reduced_q_scope, q_input_terms=reduced_q_terms, q_input_name=(f"Q[{','.join(reduced_q_scope)}]" if reduced_q_scope else ""), q_input_formula_ast=reduced_q_ast)
    trace.append(_proof_step("ancestor_reduction", "not_needed", ancestral_nodes="|".join(ancestors_y)))

    graph_without_incoming_x = _remove_incoming_to(admg, x)
    ancestors_after_do = sorted(graph_without_incoming_x.ancestors(y))
    w = sorted((set(v) - set(x)) - set(ancestors_after_do))
    if w:
        trace.append(_proof_step("irrelevant_after_intervention_w_step", "applied", promoted_to_intervention="|".join(w), ancestors_after_do="|".join(ancestors_after_do)))
        return _recursive_id_expression(admg, y, sorted(set(x) | set(w)), depth=depth + 1, max_depth=max_depth, trace=trace, seen=seen, q_input_scope=q_scope, q_input_terms=q_terms, q_input_name=q_name, q_input_formula_ast=q_input_formula_ast)
    trace.append(_proof_step("irrelevant_after_intervention_w_step", "not_needed", ancestors_after_do="|".join(ancestors_after_do)))

    if not admg.bidirected_edges:
        formula, eliminated, retained_terms, _removed_terms = _truncated_factorization_formula_for_sets(admg, x, y)
        if formula:
            trace.append(_proof_step("observed_dag_truncated_factorization", "identified", eliminated_nodes="|".join(eliminated), formula_ast_source="internal_recursive_builder_step43"))
            formula_ast = _wrap_do_ast(x, _sum_if_needed_ast(eliminated, _product_terms_ast(retained_terms, label="truncated_factorization_product")))
            return RecursiveIDExpressionDiagnostic("identified_observed_dag_truncated_factorization_set_case", True, formula, expression_json=_expression_payload(kind="truncated_factorization", y_set=y, x_set=x, formula=formula, sum_over=eliminated, product_terms=retained_terms, reason_codes="OBSERVED_DAG_TRUNCATED_FACTORIZATION_VALID", formula_ast=formula_ast), trace_json=_json_formula({"trace": trace}), depth=depth, pending_operator="none", reason_codes="OBSERVED_DAG_TRUNCATED_FACTORIZATION_VALID")

    remaining_nodes = sorted(set(v) - set(x))
    gx = admg.induced_subgraph(remaining_nodes)
    remaining_districts = gx.districts()
    if len(remaining_districts) > 1:
        trace.append(_proof_step("district_decomposition_g_v_minus_x", "applied", districts=_format_components(remaining_districts)))
        subexpressions: List[Mapping[str, object]] = []
        formulas: List[str] = []
        blockers: List[str] = []
        blocker_classes: List[str] = []
        formal_hedges: List[Mapping[str, object]] = []
        sub_asts: List[FormulaAST] = []
        all_identified = True
        for idx, district in enumerate(remaining_districts, start=1):
            d = sorted(district)
            sub = _recursive_id_expression(admg, d, sorted(set(v) - set(d)), depth=depth + 1, max_depth=max_depth, trace=trace, seen=seen, q_input_scope=q_scope, q_input_terms=q_terms, q_input_name=q_name, q_input_formula_ast=q_input_formula_ast)
            try:
                expr = json.loads(sub.expression_json) if sub.expression_json else {}
            except Exception:
                expr = {"raw_formula": sub.formula}
            subexpressions.append({"subproblem_id": f"D{idx}", "district": d, "status": sub.expression_status, "identified": bool(sub.expression_identified), "formula": sub.formula, "expression": expr, "reason_codes": sub.reason_codes})
            if sub.expression_identified and isinstance(expr, Mapping):
                sub_asts.append(_subexpression_runtime_ast(expr))
            formulas.append(_formula_rhs(sub.formula) or f"UNRESOLVED_Q[{','.join(d)}]")
            if not sub.expression_identified:
                all_identified = False
                blockers.append(sub.blocker or sub.reason_codes or _format_component(d))
                if sub.blocker_class:
                    blocker_classes.append(sub.blocker_class)
                if sub.blocker_class == "formal_hedge_certificate" and isinstance(expr, Mapping):
                    hedge = expr.get("formal_hedge_candidate")
                    if isinstance(hedge, Mapping):
                        formal_hedges.append(hedge)
        sum_over = [n for n in remaining_nodes if n not in set(y)]
        product = " * ".join(formulas) if formulas else "1"
        formula = f"sum_{{{','.join(sum_over)}}} {product}" if sum_over else product
        has_formal_hedge = bool(formal_hedges) or "formal_hedge_certificate" in blocker_classes
        if all_identified:
            status = "identified_recursive_district_decomposition"
            reason = "RECURSIVE_DISTRICT_DECOMPOSITION_IDENTIFIED"
            blocker_class = "none"
            pending_operator = "none"
        elif has_formal_hedge:
            status = "blocked_formal_hedge_certificate"
            reason = "FORMAL_HEDGE_CERTIFICATE_RECURSIVE_DISTRICT_SUBPROBLEM_STEP28"
            blocker_class = "formal_hedge_certificate"
            pending_operator = "fail_id_or_construct_full_hedge_certificate"
        else:
            status = "blocked_recursive_district_decomposition_subproblem"
            reason = "RECURSIVE_DISTRICT_SUBPROBLEM_BLOCKED"
            blocker_class = "recursive_subproblem_blocked"
            pending_operator = "extend_subdistrict_recursion"
        formula_ast = None
        if all_identified:
            formula_ast = _wrap_do_ast(x, _sum_if_needed_ast(sum_over, Product(sub_asts or [Placeholder("1", metadata={"constant": 1})], label="district_decomposition_product")))
        trace.append(_proof_step("district_decomposition_g_v_minus_x", "identified" if all_identified else "blocked", sum_over="|".join(sum_over), blockers="|".join(blockers), blocker_class=blocker_class, formula_ast_source="internal_recursive_builder_step43" if all_identified else "compatibility_parser_step42"))
        return RecursiveIDExpressionDiagnostic(
            status,
            all_identified,
            formula if all_identified else "",
            expression_json=_expression_payload(
                kind="recursive_district_decomposition" if all_identified else status,
                y_set=y,
                x_set=x,
                formula=formula if all_identified else "",
                sum_over=sum_over,
                product_terms=formulas,
                subexpressions=subexpressions,
                districts=remaining_districts,
                formal_hedge=formal_hedges[0] if formal_hedges else None,
                reason_codes=reason,
                formula_ast=formula_ast,
            ),
            trace_json=_json_formula({"trace": trace}),
            depth=depth,
            blocker="|".join(blockers),
            blocker_class=blocker_class,
            pending_operator=pending_operator,
            reason_codes=reason,
        )

    if len(remaining_districts) == 1:
        s = sorted(remaining_districts[0])
        full_districts = admg.districts()
        if any(set(s) == set(d) for d in full_districts):
            q_formula, sum_over, terms = _q_factor_formula_for_district_from_terms(admg, s, y, q_terms)
            if q_formula:
                trace.append(_proof_step("q_factor_full_district", "identified", district=_format_component(s), sum_over="|".join(sum_over), q_input=q_name, formula_ast_source="internal_recursive_builder_step43"))
                q_label = "q_input_full_district" if q_terms else "q_factor_full_district"
                if q_terms:
                    q_base_ast = _operational_q_input_ast(q_input_formula_ast, q_input=q_name, q_scope=q_scope or s, fallback_terms=terms, fallback_label=q_label)
                    formula_ast = _wrap_do_ast(x, _sum_if_needed_ast(sum_over, q_base_ast))
                else:
                    formula_ast = _wrap_do_ast(x, _q_runtime_ast(s, terms, sum_over, q_input=q_name, label=q_label))
                reason = "Q_INPUT_FULL_DISTRICT_IDENTIFIED_FROM_OPERATIONAL_CARRIED_Q_AST_STEP53" if q_terms and q_ast_enabled else ("Q_INPUT_FULL_DISTRICT_IDENTIFIED_FROM_CARRIED_Q_STEP44" if q_terms else "Q_FACTOR_FULL_DISTRICT_IDENTIFIED_FROM_CHAIN_RULE")
                return RecursiveIDExpressionDiagnostic("identified_q_factor_full_district", True, q_formula, expression_json=_expression_payload(kind="q_factor_full_district", y_set=y, x_set=x, formula=q_formula, sum_over=sum_over, product_terms=terms, districts=[s], reason_codes=reason, formula_ast=formula_ast), trace_json=_json_formula({"trace": trace}), depth=depth, pending_operator="none", reason_codes=reason)
        containing = [d for d in full_districts if set(s).issubset(set(d)) and set(s) != set(d)]
        for containing_district in containing:
            ok, q_formula, q_sum_over, q_terms, q_payload, q_reason = _q_input_subdistrict_formula(
                admg,
                s,
                containing_district,
                y,
                x,
            )
            if ok and q_formula:
                trace.append(_proof_step(
                    "q_input_subdistrict_recursion",
                    "identified",
                    district=_format_component(s),
                    containing_district=_format_component(containing_district),
                    sum_over="|".join(q_sum_over),
                    reason_codes=q_reason,
                    formula_ast_source="internal_recursive_builder_step43",
                ))
                q_input_name = f"Q[{','.join(sorted(containing_district))}]"
                formula_ast = _wrap_do_ast(x, _q_runtime_ast(s, q_terms, q_sum_over, q_input=q_input_name, label="q_input_subdistrict_recursion"))
                return RecursiveIDExpressionDiagnostic(
                    "identified_q_input_subdistrict_recursion",
                    True,
                    q_formula,
                    expression_json=_expression_payload(
                        kind="q_input_subdistrict_recursion",
                        y_set=y,
                        x_set=x,
                        formula=q_formula,
                        sum_over=q_sum_over,
                        product_terms=q_terms,
                        subexpressions=[q_payload],
                        districts=[containing_district, s],
                        reason_codes=q_reason,
                        formula_ast=formula_ast,
                    ),
                    trace_json=_json_formula({"trace": trace}),
                    depth=depth,
                    pending_operator="none",
                    reason_codes=q_reason,
                )
        # Step 5 toward Full ID: if the narrow safe Q-input shortcut above did
        # not fire, create the actual carried-Q recursive subproblem rather than
        # returning a flat placeholder.  This mirrors ID step 7: recurse inside
        # the containing district S' with Q[S'] as the input distribution.  We
        # only attempt this from the outer graph; once inside a carried-Q scope,
        # a repeated strict-subdistrict failure falls through to the hedge branch
        # instead of cycling.
        # Step 51: use an explicit carried-Q context for the recursive ID-7
        # subproblem.  We still avoid the degenerate Q[S] -> Q[S] retry, but
        # if the active Q input is larger than the containing district we can
        # project its terms and continue with Q[S'] as the input distribution.
        if containing and (not q_scope or set(containing[0]) != set(q_scope)):
            containing_district = sorted(containing[0])
            q_ctx = build_carried_q_context(
                admg,
                containing_district,
                source_scope=q_scope,
                source_terms=q_terms,
                source_name=q_name,
                name=f"Q[{','.join(containing_district)}]",
            )
            q_input_terms_for_containing = list(q_ctx.terms)
            q_input_name_for_containing = q_ctx.name
            working = admg.induced_subgraph(containing_district)
            y_inside = [n for n in y if n in set(containing_district)]
            x_inside = [n for n in x if n in set(containing_district) and n not in set(y_inside)]
            if y_inside:
                trace.append(_proof_step("q_input_general_carried_q_recursion_step51", "entered", district=_format_component(s), containing_district=_format_component(containing_district), q_input=q_input_name_for_containing, source_q_input=q_name, source_q_scope="|".join(q_scope), carried_q_source=q_ctx.source, projected_from_source=q_ctx.projected_from_source, projection_loss=q_ctx.projection_loss, x_inside="|".join(x_inside), y_inside="|".join(y_inside)))
                trace.append(_proof_step("q_input_general_carried_q_ast_step53", "entered", district=_format_component(s), containing_district=_format_component(containing_district), q_input=q_input_name_for_containing, formula_ast_available=int(bool(q_ctx.formula_ast)), formula=q_ctx.formula))
                trace.append(_proof_step("q_input_general_recursion_step44", "entered", district=_format_component(s), containing_district=_format_component(containing_district), q_input=q_input_name_for_containing, x_inside="|".join(x_inside), y_inside="|".join(y_inside)))
                sub = _recursive_id_expression(working, y_inside, x_inside, depth=depth + 1, max_depth=max_depth, trace=trace, seen=seen, q_input_scope=containing_district, q_input_terms=q_input_terms_for_containing, q_input_name=q_input_name_for_containing, q_input_formula_ast=q_ctx.formula_ast)
                try:
                    sub_expr = json.loads(sub.expression_json) if sub.expression_json else {}
                except Exception:
                    sub_expr = {"raw_formula": sub.formula}
                sub_payload = _q_input_recursion_payload(containing_district=containing_district, subdistrict=s, q_input_name=q_input_name_for_containing, q_input_terms=q_input_terms_for_containing, recursive_status=sub.expression_status, recursive_identified=sub.expression_identified, recursive_formula=sub.formula, recursive_expression=sub_expr if isinstance(sub_expr, Mapping) else {}, recursive_reason_codes=sub.reason_codes, carried_q_context=q_ctx.to_dict())
                if sub.expression_identified:
                    formula_ast = _subexpression_runtime_ast(sub_expr) if isinstance(sub_expr, Mapping) else Placeholder("identified_general_q_input_recursion_step44")
                    trace.append(_proof_step("q_input_general_recursion_step44", "identified", district=_format_component(s), containing_district=_format_component(containing_district), reason_codes=sub.reason_codes or "GENERAL_Q_INPUT_RECURSION_IDENTIFIED_STEP44", formula_ast_source="internal_recursive_builder_step43"))
                    return RecursiveIDExpressionDiagnostic("identified_general_q_input_subdistrict_recursion", True, sub.formula, expression_json=_expression_payload(kind="general_q_input_subdistrict_recursion", y_set=y, x_set=x, formula=sub.formula, subexpressions=[sub_payload], districts=[containing_district, s], reason_codes=sub.reason_codes or "GENERAL_Q_INPUT_RECURSION_IDENTIFIED_STEP44", formula_ast=formula_ast), trace_json=_json_formula({"trace": trace}), depth=depth, pending_operator="none", reason_codes=sub.reason_codes or "GENERAL_Q_INPUT_RECURSION_IDENTIFIED_STEP44")
                if sub.blocker_class == "formal_hedge_certificate" or sub.expression_status == "blocked_formal_hedge_certificate":
                    hedge = sub_expr.get("formal_hedge_candidate") if isinstance(sub_expr, Mapping) else None
                    trace.append(_proof_step("q_input_general_recursion_step44", "blocked_formal_hedge", district=_format_component(s), containing_district=_format_component(containing_district), reason_codes=sub.reason_codes or "GENERAL_Q_INPUT_RECURSION_HEDGE_FAIL_STEP44"))
                    if isinstance(hedge, Mapping):
                        hedge = dict(hedge)
                        # If this hedge was discovered inside a carried-Q subproblem,
                        # preserve the fact that the certificate is localized in the
                        # outer graph even when the induced subgraph itself has one district.
                        if set(containing_district) < set(admg.node_set):
                            hedge["scope"] = "localized_full_district_strict_g_without_x_district"
                            checks = dict(hedge.get("checks", {})) if isinstance(hedge.get("checks"), Mapping) else {}
                            checks["localized_certificate"] = True
                            hedge["checks"] = checks
                        hedge_ast = HedgeFail(hedge.get("F", containing_district), hedge.get("F_prime", s), roots=hedge.get("roots_F", []) or hedge.get("roots", []), label="general_q_input_recursion_formal_hedge")
                    else:
                        hedge_ast = HedgeFail(containing_district, s, roots=[], label="general_q_input_recursion_formal_hedge")
                    return RecursiveIDExpressionDiagnostic("blocked_formal_hedge_certificate", False, depth=depth, expression_json=_expression_payload(kind="general_q_input_recursion_formal_hedge", y_set=y, x_set=x, formula="", subexpressions=[sub_payload], districts=[containing_district, s], formal_hedge=hedge if isinstance(hedge, Mapping) else None, reason_codes=sub.reason_codes or "GENERAL_Q_INPUT_RECURSION_HEDGE_FAIL_STEP44", formula_ast=hedge_ast), trace_json=_json_formula({"trace": trace}), blocker=sub.blocker or _format_component(s), blocker_class="formal_hedge_certificate", pending_operator="fail_id_or_construct_full_hedge_certificate", reason_codes=sub.reason_codes or "GENERAL_Q_INPUT_RECURSION_HEDGE_FAIL_STEP44")
                trace.append(_proof_step("q_input_general_recursion_step44", "blocked", district=_format_component(s), containing_district=_format_component(containing_district), sub_blocker_class=sub.blocker_class, reason_codes=sub.reason_codes or "GENERAL_Q_INPUT_RECURSION_SUBPROBLEM_BLOCKED_STEP44"))
                return RecursiveIDExpressionDiagnostic("blocked_general_q_input_subdistrict_recursion", False, depth=depth, expression_json=_expression_payload(kind="general_q_input_subdistrict_recursion_blocked", y_set=y, x_set=x, formula="", subexpressions=[sub_payload], districts=[containing_district, s], reason_codes=sub.reason_codes or "GENERAL_Q_INPUT_RECURSION_SUBPROBLEM_BLOCKED_STEP44", formula_ast=Placeholder("general_q_input_subdistrict_recursion_blocked", metadata={"containing_district": containing_district, "subdistrict": s, "sub_status": sub.expression_status, "sub_reason_codes": sub.reason_codes})), trace_json=_json_formula({"trace": trace}), blocker=sub.blocker or _format_component(s), blocker_class=sub.blocker_class or "general_q_input_subproblem_blocked", pending_operator="extend_general_q_input_recursion", reason_codes=sub.reason_codes or "GENERAL_Q_INPUT_RECURSION_SUBPROBLEM_BLOCKED_STEP44")

        hedge_candidate = _formal_hedge_certificate_payload(admg, s, x, y)
        if hedge_candidate:
            blocker = f"F={_format_component(hedge_candidate['F'])};F_prime={_format_component(hedge_candidate['F_prime'])}"
            trace.append(_proof_step("formal_hedge_certificate", "blocked", **hedge_candidate))
            return RecursiveIDExpressionDiagnostic(
                "blocked_formal_hedge_certificate",
                False,
                depth=depth,
                expression_json=_expression_payload(kind="formal_hedge_certificate", y_set=y, x_set=x, formula="", districts=[s], formal_hedge=hedge_candidate, reason_codes=hedge_candidate.get("reason_codes", "FORMAL_HEDGE_CERTIFICATE_LOCAL_DISTRICT_FAIL_BRANCH_STEP28"), formula_ast=HedgeFail(hedge_candidate.get("F", []), hedge_candidate.get("F_prime", []), roots=hedge_candidate.get("roots_F", []) or hedge_candidate.get("roots", []), label="formal_hedge_certificate")),
                trace_json=_json_formula({"trace": trace}),
                blocker=blocker,
                blocker_class="formal_hedge_certificate",
                pending_operator="fail_id_or_construct_full_hedge_certificate",
                reason_codes=str(hedge_candidate.get("reason_codes", "FORMAL_HEDGE_CERTIFICATE_LOCAL_DISTRICT_FAIL_BRANCH_STEP28")),
            )
        blocker = _format_component(s)
        reason = "SUBDISTRICT_Q_FACTOR_RECURSION_REQUIRES_UNSUPPORTED_Q_INPUT_STEP23"
        if containing:
            _ok, _formula, _sum_over, _terms, q_payload, q_reason = _q_input_subdistrict_formula(admg, s, containing[0], y, x)
            reason = q_reason or reason
        trace.append(_proof_step("q_factor_subdistrict", "blocked", district=blocker, containing_districts=_format_components(containing), reason_codes=reason))
        return RecursiveIDExpressionDiagnostic(
            "blocked_subdistrict_q_factor_input_pending",
            False,
            depth=depth,
            expression_json=_expression_payload(kind="blocked_subdistrict_q_factor_input_pending", y_set=y, x_set=x, formula="", districts=[s], reason_codes=reason, formula_ast=Placeholder("blocked_subdistrict_q_factor_input_pending", metadata={"blocked_district": s, "reason_codes": reason})),
            trace_json=_json_formula({"trace": trace}),
            blocker=blocker,
            blocker_class="subdistrict_q_factor_input_pending",
            pending_operator="extend_q_input_recursion_beyond_safe_ancestral_case",
            reason_codes=reason,
        )

    trace.append(_proof_step("recursive_id_fallback", "blocked", reason_codes="UNHANDLED_RECURSIVE_BRANCH"))
    return RecursiveIDExpressionDiagnostic("blocked_unhandled_recursive_case", False, depth=depth, trace_json=_json_formula({"trace": trace}), blocker="UNHANDLED_RECURSIVE_BRANCH", blocker_class="unhandled_recursive_case", pending_operator="extend_recursive_id_branch", reason_codes="UNHANDLED_RECURSIVE_BRANCH")


def recursive_id_expression_diagnostic(
    admg: ADMG,
    treatment: str,
    outcome: str,
    *,
    max_depth: int = 8,
) -> RecursiveIDExpressionDiagnostic:
    """Public single-treatment/single-outcome wrapper for recursive-ID expression diagnostics."""
    x = _s(treatment)
    y = _s(outcome)
    if not x or not y or x not in admg.node_set or y not in admg.node_set:
        return RecursiveIDExpressionDiagnostic("invalid_query", False, blocker="MISSING_QUERY_NODE", blocker_class="invalid_query", pending_operator="validate_query", reason_codes="MISSING_QUERY_NODE")
    return _recursive_id_expression(admg, [y], [x], max_depth=max_depth)


def recursive_id_set_expression_diagnostic(
    admg: ADMG,
    treatments: Sequence[object],
    outcomes: Sequence[object],
    *,
    max_depth: int = 8,
) -> RecursiveIDExpressionDiagnostic:
    """Public set-valued wrapper for the recursive-ID expression engine.

    Full ID is a set-valued algorithm: both ``X`` and ``Y`` may contain more
    than one observed variable.  Earlier public APIs exposed only the common
    single-edge case, even though the internal recursive engine already operated
    on sets.  This wrapper makes the set-valued path explicit and conservative:

    - requested nodes must exist in the ADMG;
    - treatment/outcome overlap is rejected instead of silently dropping nodes;
    - the returned object is the same auditable ``RecursiveIDExpressionDiagnostic``
      used by the single-edge route.
    """
    raw_x = _dedupe(treatments or [])
    raw_y = _dedupe(outcomes or [])
    missing = sorted([n for n in raw_x + raw_y if n not in admg.node_set])
    overlap = sorted(set(raw_x) & set(raw_y))
    if not raw_x or not raw_y or missing or overlap:
        reasons: List[str] = []
        if not raw_x:
            reasons.append("MISSING_TREATMENT_SET")
        if not raw_y:
            reasons.append("MISSING_OUTCOME_SET")
        if missing:
            reasons.append("QUERY_NODE_NOT_IN_GRAPH:" + "|".join(missing))
        if overlap:
            reasons.append("TREATMENT_OUTCOME_OVERLAP:" + "|".join(overlap))
        reason = ";".join(reasons) or "INVALID_SET_QUERY"
        return RecursiveIDExpressionDiagnostic(
            "invalid_set_query",
            False,
            blocker=reason,
            blocker_class="invalid_set_query",
            pending_operator="validate_set_query",
            reason_codes=reason,
        )
    x = sorted(raw_x)
    y = sorted(raw_y)
    return _recursive_id_expression(admg, y, x, max_depth=max_depth)


def _recursive_expression_fields(expr: Optional[RecursiveIDExpressionDiagnostic]) -> Dict[str, object]:
    if expr is None:
        return {}
    return {
        "recursive_expression_json": expr.expression_json,
        "recursive_trace_json": expr.trace_json,
        "recursive_formula_source": expr.expression_status,
    }
