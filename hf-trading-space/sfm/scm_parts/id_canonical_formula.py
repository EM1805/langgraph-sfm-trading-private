from __future__ import annotations

"""Step 67 canonical formula authority.

This module owns conservative canonical formulas for the ID public facade. Step
67 keeps the contextual and chain-frontdoor carried-Q families and adds a
strict parallel-mediator/frontdoor-set ID-7 family for graphs like
X -> {Z1,Z2} -> Y with X <-> Y. It still deliberately does not claim
arbitrary Full ID: fully general ID-7 carried-Q recursion and complete IDC
remain gated.
"""
from dataclasses import asdict, dataclass
import json
from typing import Dict, Iterable, List, Mapping, MutableSequence, Optional, Sequence, Set, Tuple

from .admg import ADMG
from .graph_criteria import directed_cycle_nodes, directed_path_exists, topological_order
from .id_algorithm_common import _conditional_factor, _dedupe, _joint_symbol, _json_formula, _s
from .id_ast import Do, FormulaAST, HedgeFail, P, Placeholder, Product, Sum, ast_from_dict
from .id_ast_normalizer import ID_AST_NORMALIZER_VERSION, normalize_formula_ast

ID_CANONICAL_FORMULA_VERSION = "id_canonical_formula_v6_step67"
ID_CANONICAL_FORMULA_AUTHORITY = "id_canonical_formula_step60"
ID_CANONICAL_FORMULA_LEVEL = (
    "owned_formula_authority_for_ID_1_ID_2_ID_4_ID_6_and_gated_ID_7_frontdoor_contextual_chain_and_parallel_frontdoor_set_carried_Q_no_full_id_claim"
)


def _json(payload: Mapping[str, object]) -> str:
    return json.dumps(dict(payload), sort_keys=True, separators=(",", ":"))


def _nodes(values: Sequence[object] | object) -> List[str]:
    return _dedupe([values] if isinstance(values, str) else values or [])


def _product(parts: Sequence[str]) -> str:
    clean = [_s(p) for p in parts if _s(p)]
    return " * ".join(clean) if clean else "1"


def _sum(bound: Sequence[str], body: str) -> str:
    return f"sum_{{{','.join(_dedupe(bound))}}} {body}" if bound else body


def _rhs(formula: str) -> str:
    text = _s(formula)
    return text.split(" = ", 1)[1].strip() if " = " in text else text


def _trace_steps(trace_json: str) -> List[Mapping[str, object]]:
    try:
        payload = json.loads(trace_json) if trace_json else {}
    except Exception:
        return []
    steps = payload.get("trace", []) if isinstance(payload, Mapping) else []
    return [s for s in steps if isinstance(s, Mapping)] if isinstance(steps, list) else []


def _ast_from_json(raw: str, fallback: str) -> FormulaAST:
    try:
        payload = json.loads(raw) if raw else {}
        if isinstance(payload, Mapping):
            return ast_from_dict(payload)
    except Exception:
        pass
    return Placeholder(fallback)


def _bidirected_key(a: str, b: str) -> Tuple[str, str]:
    return tuple(sorted((a, b)))


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


def _payload(
    kind: str,
    y: Sequence[str],
    x: Sequence[str],
    formula: str,
    ast: FormulaAST,
    rule: str,
    *,
    reason: str,
    terms: Sequence[str] = (),
    sum_over: Sequence[str] = (),
    subexpressions: Sequence[Mapping[str, object]] = (),
    districts: Sequence[Sequence[str]] = (),
) -> str:
    norm = normalize_formula_ast(ast)
    return _json_formula({
        "kind": kind,
        "canonical_formula_version": ID_CANONICAL_FORMULA_VERSION,
        "formula_authority": ID_CANONICAL_FORMULA_AUTHORITY,
        "formula_authority_level": ID_CANONICAL_FORMULA_LEVEL,
        "formula": formula,
        "y_set": list(_dedupe(y)),
        "x_set": list(_dedupe(x)),
        "rule": rule,
        "product_terms": list(terms),
        "sum_over": list(sum_over),
        "districts": [list(_dedupe(d)) for d in districts],
        "subexpressions": [dict(s) for s in subexpressions],
        "formula_ast": norm.to_dict(),
        "formula_ast_normalized": 1,
        "formula_ast_normalizer_version": ID_AST_NORMALIZER_VERSION,
        "reason_codes": reason,
        "full_id_claim_allowed": 0,
    })


@dataclass(frozen=True)
class CanonicalFormulaDiagnostic:
    status: str
    identified: bool
    formula: str = ""
    formula_ast_json: str = ""
    expression_json: str = ""
    trace_json: str = ""
    terminal_rule: str = ""
    terminal_status: str = ""
    authority_level: str = ID_CANONICAL_FORMULA_LEVEL
    version: str = ID_CANONICAL_FORMULA_VERSION
    blocker: str = ""
    blocker_class: str = ""
    pending_operator: str = ""
    reason_codes: str = ""
    id7_carried_q_formula_used: int = 0

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _diag(
    status: str,
    identified: bool,
    *,
    formula: str = "",
    ast: Optional[FormulaAST] = None,
    expression_json: str = "",
    trace: Sequence[Mapping[str, object]] = (),
    rule: str = "",
    terminal: str = "",
    blocker: str = "",
    blocker_class: str = "",
    pending: str = "",
    reason: str = "",
    id7: int = 0,
) -> CanonicalFormulaDiagnostic:
    norm = normalize_formula_ast(ast or Placeholder(status))
    return CanonicalFormulaDiagnostic(
        status=status,
        identified=identified,
        formula=formula if identified else "",
        formula_ast_json=_json(norm.to_dict()),
        expression_json=expression_json,
        trace_json=_json_formula({
            "trace": list(trace),
            "canonical_formula_version": ID_CANONICAL_FORMULA_VERSION,
            "formula_authority": ID_CANONICAL_FORMULA_AUTHORITY,
            "full_id_claim_allowed": 0,
        }),
        terminal_rule=rule,
        terminal_status=terminal,
        blocker=blocker,
        blocker_class=blocker_class,
        pending_operator=pending,
        reason_codes=reason,
        id7_carried_q_formula_used=id7,
    )


def _no_intervention_formula(admg: ADMG, x: Sequence[str], y: Sequence[str]) -> CanonicalFormulaDiagnostic:
    order = topological_order(admg) or sorted(admg.node_set)
    bound = [n for n in order if n not in set(y)]
    rhs = _sum(bound, f"P({_joint_symbol(order)})")
    ast: FormulaAST = P(order, label="canonical_observational_joint")
    if bound:
        ast = Sum(bound, ast, label="canonical_no_intervention_sum")
    reason = "CANONICAL_ID1_NO_INTERVENTION_STEP59"
    expr = _payload("canonical_no_intervention", y, x, rhs, ast, "ID-1", reason=reason, terms=[f"P({_joint_symbol(order)})"], sum_over=bound)
    return _diag("identified_canonical_id1_no_intervention_step59", True, formula=rhs, ast=ast, expression_json=expr, trace=[{"rule": "ID-1", "status": "identified_no_intervention"}], rule="ID-1", terminal="identified_no_intervention", reason=reason)


def _zero_effect_formula(admg: ADMG, x: Sequence[str], y: Sequence[str]) -> Optional[CanonicalFormulaDiagnostic]:
    if not _no_directed_path_between(admg, x, y):
        return None
    rhs = f"P({_joint_symbol(y)})"
    ast = Do(x, P(y, label="canonical_zero_effect_marginal"), label="canonical_id2_zero_effect_do_wrapper")
    reason = "CANONICAL_ID2_GRAPHICAL_ZERO_EFFECT_STEP59"
    expr = _payload("canonical_graphical_zero_effect", y, x, rhs, ast, "ID-2", reason=reason, terms=[rhs])
    trace = [{"rule": "ID-2", "status": "identified_graphical_zero_effect", "reason": "no_directed_path_from_treatment_to_outcome"}]
    return _diag("identified_canonical_id2_graphical_zero_effect_step59", True, formula=rhs, ast=ast, expression_json=expr, trace=trace, rule="ID-2", terminal="identified_graphical_zero_effect", reason=reason)


def _observed_dag_formula(admg: ADMG, x: Sequence[str], y: Sequence[str]) -> CanonicalFormulaDiagnostic:
    order = topological_order(admg) or sorted(admg.node_set)
    parents = admg.parents()
    x_set, y_set = set(x), set(y)
    bound = [n for n in order if n not in x_set and n not in y_set]
    terms = [_conditional_factor(n, sorted(parents.get(n, set()))) for n in order if n not in x_set]
    rhs = _sum(bound, _product(terms))
    ast: FormulaAST = Product([P([n], given=sorted(parents.get(n, set())), label="canonical_truncated_factor") for n in order if n not in x_set], label="canonical_observed_dag_product")
    if bound:
        ast = Sum(bound, ast, label="canonical_observed_dag_sum")
    ast = Do(x, ast, label="canonical_id6_observed_dag")
    reason = "CANONICAL_ID6_OBSERVED_DAG_TRUNCATED_FACTORIZATION_STEP59"
    expr = _payload("canonical_observed_dag_truncated_factorization", y, x, rhs, ast, "ID-6", reason=reason, terms=terms, sum_over=bound)
    trace = [{"rule": "ID-6", "status": "identified_observed_dag_truncated_factorization", "sum_over": "|".join(bound)}]
    return _diag("identified_canonical_id6_observed_dag_step59", True, formula=rhs, ast=ast, expression_json=expr, trace=trace, rule="ID-6", terminal="identified_observed_dag_truncated_factorization", reason=reason)


def _q_factor_formula(admg: ADMG, x: Sequence[str], y: Sequence[str]) -> Optional[CanonicalFormulaDiagnostic]:
    v = sorted(admg.node_set)
    remaining = sorted(set(v) - set(x))
    if not remaining:
        return None
    gx = admg.induced_subgraph(remaining)
    districts = gx.districts()
    if len(districts) != 1:
        return None
    s = districts[0]
    if not any(set(s) == set(d) for d in admg.districts()):
        return None
    order = topological_order(admg) or sorted(admg.node_set)
    prev: List[str] = []
    terms: List[str] = []
    ast_terms: List[FormulaAST] = []
    for n in order:
        if n in set(s):
            terms.append(_conditional_factor(n, list(prev)))
            ast_terms.append(P([n], given=list(prev), label="canonical_q_chain_factor"))
        prev.append(n)
    bound = [n for n in order if n in set(s) and n not in set(y)]
    rhs = _sum(bound, _product(terms))
    ast: FormulaAST = Product(ast_terms, label="canonical_q_factor_chain_product")
    if bound:
        ast = Sum(bound, ast, label="canonical_q_factor_sum")
    reason = "CANONICAL_ID6_FULL_DISTRICT_Q_FACTOR_STEP59"
    expr = _payload("canonical_q_factor_full_district", y, x, rhs, ast, "ID-6", reason=reason, terms=terms, sum_over=bound)
    trace = [{"rule": "ID-6", "status": "identified_full_district_q_factor", "district": "|".join(s)}]
    return _diag("identified_canonical_id6_full_district_q_factor_step59", True, formula=rhs, ast=ast, expression_json=expr, trace=trace, rule="ID-6", terminal="identified_full_district_q_factor", reason=reason)



def _simple_directed_paths(admg: ADMG, source: str, target: str, *, max_len: int = 6) -> List[List[str]]:
    """Return simple directed paths source -> ... -> target, conservatively bounded."""
    children = admg.children()
    out: List[List[str]] = []

    def dfs(cur: str, path: List[str]) -> None:
        if len(path) > max_len:
            return
        if cur == target:
            out.append(list(path))
            return
        for nxt in sorted(children.get(cur, set())):
            if nxt in path:
                continue
            dfs(nxt, [*path, nxt])

    if source in admg.node_set and target in admg.node_set:
        dfs(source, [source])
    return out


def _frontdoor_chain_formula(admg: ADMG, x: Sequence[str], y: Sequence[str]) -> Optional[CanonicalFormulaDiagnostic]:
    """Narrow ID-7 chain-frontdoor family added in Step 66.

    Supported shape is intentionally strict:
      X -> Z1 -> ... -> Zk -> Y, k >= 2, with X <-> Y and no direct X -> Y.

    Additional conservatism:
    - the directed graph may not contain extra observed nodes beyond the chain;
    - the only bidirected edge may be X <-> Y;
    - no mediator is bidirected-confounded with X, Y, or another mediator.

    Formula:
      P_x(y) = sum_z P(z1|x) * Π_i P(z_i | z_{i-1}) * sum_x' P(x') P(y | x',z)
    """
    if len(x) != 1 or len(y) != 1:
        return None
    tx, out = x[0], y[0]
    directed = set(admg.directed_edges)
    bidirected = {tuple(sorted(e)) for e in admg.bidirected_edges}
    if _bidirected_key(tx, out) not in bidirected or (tx, out) in directed:
        return None

    paths = [p for p in _simple_directed_paths(admg, tx, out, max_len=max(6, len(admg.node_set) + 1)) if len(p) >= 4]
    if len(paths) != 1:
        return None
    path = paths[0]
    mediators = path[1:-1]
    if len(mediators) < 2:
        return None
    if set(admg.node_set) != set(path):
        return None
    if bidirected != {_bidirected_key(tx, out)}:
        return None

    xp = f"{tx}_prime"
    z_joint = _joint_symbol(mediators)
    outer_terms: List[str] = []
    outer_asts: List[FormulaAST] = []
    prev = tx
    for z in mediators:
        term = f"P({z} | {prev})"
        outer_terms.append(term)
        outer_asts.append(P([z], given=[prev], label=term))
        prev = z

    outcome_given = [xp, *mediators]
    inner_terms = [f"P({xp})", f"P({out} | {','.join(outcome_given)})"]
    inner_ast = Sum(
        [xp],
        Product([
            P([xp], label=f"P({xp})"),
            P([out], given=outcome_given, label=f"P({out} | {','.join(outcome_given)})"),
        ], label="canonical_chain_frontdoor_inner_product_step66"),
        label="canonical_chain_frontdoor_sum_over_treatment_prime_step66",
    )
    rhs = f"sum_{{{z_joint}}} " + _product([*outer_terms, f"sum_{{{xp}}} " + _product(inner_terms)])
    ast = Sum(
        mediators,
        Product([*outer_asts, inner_ast], label="canonical_chain_frontdoor_product_step66"),
        label="canonical_chain_frontdoor_sum_over_mediators_step66",
    )
    terms = [*outer_terms, *inner_terms]
    reason = "CANONICAL_ID7_CHAIN_FRONTDOOR_CARRIED_Q_FORMULA_STEP66"
    expr = _payload(
        "canonical_id7_chain_frontdoor_carried_q_formula",
        y,
        x,
        rhs,
        ast,
        "ID-7",
        reason=reason,
        terms=terms,
        sum_over=[*mediators, xp],
    )
    trace = [
        {"rule": "ID-4", "status": "recognized_chain_frontdoor_district_decomposition", "mediators": "|".join(mediators), "target": out},
        {"rule": "ID-7", "status": "identified_chain_frontdoor_carried_q_formula", "mediators": "|".join(mediators), "target": out, "alpha_renamed_treatment": xp},
    ]
    return _diag(
        "identified_canonical_id7_chain_frontdoor_carried_q_formula_step66",
        True,
        formula=rhs,
        ast=ast,
        expression_json=expr,
        trace=trace,
        rule="ID-7",
        terminal="identified_chain_frontdoor_carried_q_formula",
        reason=reason,
        id7=1,
    )


def _frontdoor_parallel_formula(admg: ADMG, x: Sequence[str], y: Sequence[str]) -> Optional[CanonicalFormulaDiagnostic]:
    """Strict multi-mediator frontdoor-set family added in Step 67.

    Supported shape is deliberately narrow:
      X -> Z_i -> Y for every mediator Z_i, at least two mediators;
      X <-> Y is the only bidirected edge;
      no direct X -> Y; no extra observed directed edges.

    Formula:
      P_x(y) = sum_z P(z_1,...,z_k | x) * sum_x' P(x') P(y | x', z_1,...,z_k)

    This covers a useful ID-7 carried-Q case beyond classic singleton and
    chain-frontdoor formulas, while avoiding arbitrary frontdoor-set claims.
    """
    if len(x) != 1 or len(y) != 1:
        return None
    tx, out = x[0], y[0]
    directed = set(admg.directed_edges)
    bidirected = {tuple(sorted(e)) for e in admg.bidirected_edges}
    if _bidirected_key(tx, out) not in bidirected or (tx, out) in directed:
        return None
    if bidirected != {_bidirected_key(tx, out)}:
        return None

    mediators = sorted(set(admg.node_set) - {tx, out})
    if len(mediators) < 2:
        return None
    required_edges = {(tx, z) for z in mediators} | {(z, out) for z in mediators}
    if directed != required_edges:
        return None

    xp = f"{tx}_prime"
    z_joint = _joint_symbol(mediators)
    mediator_factor = f"P({z_joint} | {tx})"
    outcome_given = [xp, *mediators]
    outcome_factor = f"P({out} | {','.join(outcome_given)})"
    inner_terms = [f"P({xp})", outcome_factor]
    rhs = f"sum_{{{z_joint}}} {mediator_factor} * sum_{{{xp}}} " + _product(inner_terms)

    inner_ast = Sum(
        [xp],
        Product([
            P([xp], label=f"P({xp})"),
            P([out], given=outcome_given, label=outcome_factor),
        ], label="canonical_parallel_frontdoor_inner_product_step67"),
        label="canonical_parallel_frontdoor_sum_over_treatment_prime_step67",
    )
    ast = Sum(
        mediators,
        Product([
            P(mediators, given=[tx], label=mediator_factor),
            inner_ast,
        ], label="canonical_parallel_frontdoor_product_step67"),
        label="canonical_parallel_frontdoor_sum_over_mediators_step67",
    )
    terms = [mediator_factor, *inner_terms]
    reason = "CANONICAL_ID7_PARALLEL_FRONTDOOR_SET_CARRIED_Q_FORMULA_STEP67"
    expr = _payload(
        "canonical_id7_parallel_frontdoor_set_carried_q_formula",
        y,
        x,
        rhs,
        ast,
        "ID-7",
        reason=reason,
        terms=terms,
        sum_over=[*mediators, xp],
    )
    trace = [
        {"rule": "ID-4", "status": "recognized_parallel_frontdoor_set_district_decomposition", "mediators": "|".join(mediators), "target": out},
        {"rule": "ID-7", "status": "identified_parallel_frontdoor_set_carried_q_formula", "mediators": "|".join(mediators), "target": out, "alpha_renamed_treatment": xp},
    ]
    return _diag(
        "identified_canonical_id7_parallel_frontdoor_set_carried_q_formula_step67",
        True,
        formula=rhs,
        ast=ast,
        expression_json=expr,
        trace=trace,
        rule="ID-7",
        terminal="identified_parallel_frontdoor_set_carried_q_formula",
        reason=reason,
        id7=1,
    )

def _frontdoor_formula(admg: ADMG, x: Sequence[str], y: Sequence[str]) -> Optional[CanonicalFormulaDiagnostic]:
    """Canonical gated ID-7 frontdoor / contextual-frontdoor formula.

    Step 56 covered the classic singleton outcome frontdoor pattern. Step 60
    extends the same carried-Q formula family to joint outcome queries of the
    form P(Y, C | do(X)), where C is an observed pre-treatment/context outcome
    such as the condition variable later used by IDC. This lets IDC normalize
    P(Y | do(X), C) with canonical formula authority instead of falling back to
    the legacy delegate.

    This is intentionally **not** arbitrary ID-7: it is a narrow, auditable
    family with one treatment, one mediator, one frontdoor target, and optional
    context variables that are not descendants of the treatment.
    """
    chain = _frontdoor_chain_formula(admg, x, y)
    if chain is not None:
        return chain
    parallel = _frontdoor_parallel_formula(admg, x, y)
    if parallel is not None:
        return parallel

    if len(x) != 1 or not y:
        return None
    tx = x[0]
    y_set = set(y)
    directed = set(admg.directed_edges)
    bidirected = {tuple(sorted(e)) for e in admg.bidirected_edges}

    candidate_patterns: List[Tuple[str, str, List[str]]] = []
    for out in sorted(y_set):
        if _bidirected_key(tx, out) not in bidirected:
            continue
        if (tx, out) in directed:
            # A direct unmediated causal path is outside this narrow family.
            continue
        contexts = sorted(y_set - {out})
        if any(directed_path_exists(admg, tx, c) for c in contexts):
            continue
        if any(_bidirected_key(tx, c) in bidirected for c in contexts):
            continue
        mediators = []
        for z in sorted(admg.node_set - {tx, out, *contexts}):
            if (tx, z) in directed and (z, out) in directed and _bidirected_key(tx, z) not in bidirected and _bidirected_key(z, out) not in bidirected:
                mediators.append(z)
        if len(mediators) == 1:
            candidate_patterns.append((out, mediators[0], contexts))

    if len(candidate_patterns) != 1:
        return None

    out, z, contexts = candidate_patterns[0]
    xp = f"{tx}_prime"
    ctx_joint = _joint_symbol(contexts)
    ctx_given = list(contexts)
    outcome_given = list(contexts) + [xp, z]

    if contexts:
        ctx_factor = f"P({ctx_joint})"
        xp_factor = f"P({xp} | {ctx_joint})"
        outcome_factor = f"P({out} | {ctx_joint},{xp},{z})"
        rhs = f"sum_{{{z}}} {ctx_factor} * P({z} | {tx}) * sum_{{{xp}}} {xp_factor} * {outcome_factor}"
        ast_terms: List[FormulaAST] = [
            P(contexts, label=f"P({ctx_joint})"),
            P([z], given=[tx], label=f"P({z} | {tx})"),
            Sum([xp], Product([
                P([xp], given=ctx_given, label=xp_factor),
                P([out], given=outcome_given, label=outcome_factor),
            ], label="canonical_contextual_frontdoor_inner_product_step60"), label="canonical_contextual_frontdoor_sum_over_treatment_prime_step60"),
        ]
        terms = [ctx_factor, f"P({z} | {tx})", xp_factor, outcome_factor]
        kind = "canonical_id7_contextual_frontdoor_carried_q_formula"
        status = "identified_canonical_id7_contextual_frontdoor_carried_q_formula_step60"
        terminal = "identified_contextual_frontdoor_carried_q_formula"
        reason = "CANONICAL_ID7_CONTEXTUAL_FRONTDOOR_CARRIED_Q_FORMULA_STEP60"
        trace_status = "identified_contextual_frontdoor_carried_q_formula"
    else:
        rhs = f"sum_{{{z}}} P({z} | {tx}) * sum_{{{xp}}} P({xp}) * P({out} | {xp},{z})"
        ast_terms = [
            P([z], given=[tx], label=f"P({z} | {tx})"),
            Sum([xp], Product([P([xp], label=f"P({xp})"), P([out], given=[xp, z], label=f"P({out} | {xp},{z})")], label="canonical_frontdoor_inner_product"), label="canonical_frontdoor_sum_over_treatment_prime"),
        ]
        terms = [f"P({z} | {tx})", f"P({xp})", f"P({out} | {xp},{z})"]
        kind = "canonical_id7_frontdoor_carried_q_formula"
        status = "identified_canonical_id7_frontdoor_carried_q_formula_step60"
        terminal = "identified_frontdoor_carried_q_formula"
        reason = "CANONICAL_ID7_FRONTDOOR_CARRIED_Q_FORMULA_STEP60"
        trace_status = "identified_frontdoor_carried_q_formula"

    ast = Sum([z], Product(ast_terms, label="canonical_contextual_frontdoor_product_step60"), label="canonical_contextual_frontdoor_sum_over_mediator_step60")
    expr = _payload(kind, y, x, rhs, ast, "ID-7", reason=reason, terms=terms, sum_over=[z, xp])
    trace = [
        {"rule": "ID-4", "status": "recognized_frontdoor_district_decomposition", "mediator": z, "target": out, "contexts": "|".join(contexts)},
        {"rule": "ID-7", "status": trace_status, "mediator": z, "target": out, "contexts": "|".join(contexts), "alpha_renamed_treatment": xp},
    ]
    return _diag(status, True, formula=rhs, ast=ast, expression_json=expr, trace=trace, rule="ID-7", terminal=terminal, reason=reason, id7=1)


def _direct_confounding_fail(admg: ADMG, x: Sequence[str], y: Sequence[str]) -> Optional[CanonicalFormulaDiagnostic]:
    if len(x) == 1 and len(y) == 1 and _bidirected_key(x[0], y[0]) in {tuple(sorted(e)) for e in admg.bidirected_edges}:
        if len(admg.districts()) == 1:
            ast = HedgeFail(sorted(admg.node_set), list(y), roots=y, label="canonical_step59_fail_guard")
            reason = "CANONICAL_ID5_FAIL_GUARD_DIRECT_CONFOUNDING_STEP59"
            trace = [{"rule": "ID-5", "status": "blocked_single_c_component_fail_branch"}]
            return _diag("blocked_canonical_id5_fail_branch_step59", False, ast=ast, trace=trace, rule="ID-5", terminal="blocked_single_c_component_fail_branch", blocker="|".join(y), blocker_class="canonical_fail_branch", pending="formal_hedge_certificate", reason=reason)
    return None


def _district_decomposition_formula(
    admg: ADMG,
    x: Sequence[str],
    y: Sequence[str],
    *,
    depth: int,
    max_depth: int,
    seen: Set[Tuple[Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]]],
) -> Optional[CanonicalFormulaDiagnostic]:
    v = sorted(admg.node_set)
    remaining_nodes = sorted(set(v) - set(x))
    if not remaining_nodes:
        return None
    gx = admg.induced_subgraph(remaining_nodes)
    remaining_districts = gx.districts()
    if len(remaining_districts) <= 1:
        return None

    subexpressions: List[Mapping[str, object]] = []
    sub_terms: List[str] = []
    sub_asts: List[FormulaAST] = []
    sub_trace: List[Mapping[str, object]] = []
    blockers: List[str] = []

    for idx, district in enumerate(remaining_districts, start=1):
        d = sorted(district)
        sub_x = sorted(set(v) - set(d))
        sub = _canonical_id_formula_diagnostic(admg, sub_x, d, depth=depth + 1, max_depth=max_depth, seen=seen)
        sub_trace.extend(_trace_steps(sub.trace_json))
        try:
            sub_expr = json.loads(sub.expression_json) if sub.expression_json else {}
        except Exception:
            sub_expr = {"raw_formula": sub.formula}
        subexpressions.append({
            "subproblem_id": f"D{idx}",
            "district": d,
            "x_set": sub_x,
            "status": sub.status,
            "identified": bool(sub.identified),
            "formula": sub.formula,
            "expression": sub_expr if isinstance(sub_expr, Mapping) else {"raw_expression": str(sub_expr)},
            "reason_codes": sub.reason_codes,
        })
        if not sub.identified:
            blockers.append(sub.blocker or sub.reason_codes or "|".join(d))
            continue
        sub_terms.append(_rhs(sub.formula))
        sub_asts.append(_ast_from_json(sub.formula_ast_json, f"canonical_id4_subproblem_{idx}"))

    if blockers:
        reason = "CANONICAL_ID4_SUBPROBLEM_BLOCKED_STEP59"
        trace = [{"rule": "ID-4", "status": "blocked_district_decomposition_subproblem", "districts": [list(d) for d in remaining_districts], "blockers": "|".join(blockers)}] + sub_trace
        return _diag("blocked_canonical_id4_district_decomposition_subproblem_step59", False, trace=trace, rule="ID-4", terminal="blocked_district_decomposition_subproblem", blocker="|".join(blockers), blocker_class="canonical_id4_subproblem_blocked", pending="complete_subproblem_formula_authority", reason=reason)

    sum_over = [n for n in remaining_nodes if n not in set(y)]
    rhs = _sum(sum_over, _product(sub_terms))
    ast: FormulaAST = Product(sub_asts or [Placeholder("1", metadata={"constant": 1})], label="canonical_id4_district_decomposition_product_step59")
    if sum_over:
        ast = Sum(sum_over, ast, label="canonical_id4_sum_over_non_outcomes_step59")
    ast = Do(x, ast, label="canonical_id4_estimand_do_wrapper_step59")
    reason = "CANONICAL_ID4_RECURSIVE_DISTRICT_DECOMPOSITION_STEP59"
    expr = _payload(
        "canonical_id4_recursive_district_decomposition",
        y,
        x,
        rhs,
        ast,
        "ID-4",
        reason=reason,
        terms=sub_terms,
        sum_over=sum_over,
        subexpressions=subexpressions,
        districts=remaining_districts,
    )
    trace = [{"rule": "ID-4", "status": "identified_recursive_district_decomposition", "districts": [list(d) for d in remaining_districts], "sum_over": "|".join(sum_over)}] + sub_trace
    return _diag("identified_canonical_id4_recursive_district_decomposition_step59", True, formula=rhs, ast=ast, expression_json=expr, trace=trace, rule="ID-4", terminal="identified_recursive_district_decomposition", reason=reason)


def _canonical_id_formula_diagnostic(
    admg: ADMG,
    x: Sequence[str],
    y: Sequence[str],
    *,
    depth: int,
    max_depth: int,
    seen: Set[Tuple[Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]]],
) -> CanonicalFormulaDiagnostic:
    key = (tuple(sorted(admg.node_set)), tuple(sorted(_dedupe(x))), tuple(sorted(_dedupe(y))))
    if key in seen:
        reason = "CANONICAL_FORMULA_RECURSION_STATE_REVISITED_STEP59"
        return _diag("blocked_canonical_formula_recursion_cycle_step59", False, trace=[{"rule": "ID-5", "status": reason}], rule="ID-5", terminal="blocked_revisited_state", blocker=reason, blocker_class="canonical_recursion_cycle_guard", pending="debug_recursive_state", reason=reason)
    if depth > max_depth:
        reason = "CANONICAL_FORMULA_MAX_DEPTH_HIT_STEP59"
        return _diag("blocked_canonical_formula_max_depth_step59", False, trace=[{"rule": "ID-5", "status": reason}], rule="ID-5", terminal="blocked_max_depth", blocker=reason, blocker_class="canonical_max_depth", pending="increase_max_depth_or_simplify_graph", reason=reason)
    seen.add(key)

    x = _nodes(x)
    y = _nodes(y)
    if not x:
        return _no_intervention_formula(admg, x, y)

    zero = _zero_effect_formula(admg, x, y)
    if zero is not None:
        return zero

    # ID-2: ancestor reduction.  This is required for canonical ID-4
    # subproblems such as ID({Z}, {W,X,Y}) in a larger graph, where non-ancestor
    # variables should be removed before the observed-DAG/q-factor base case.
    v = sorted(admg.node_set)
    ancestors_y = sorted(admg.ancestors(y))
    removed_non_ancestors = sorted(set(v) - set(ancestors_y))
    if removed_non_ancestors:
        reduced = admg.induced_subgraph(ancestors_y)
        reduced_x = [n for n in x if n in set(ancestors_y)]
        sub = _canonical_id_formula_diagnostic(reduced, reduced_x, y, depth=depth + 1, max_depth=max_depth, seen=seen)
        if sub.identified:
            sub_ast = _ast_from_json(sub.formula_ast_json, "canonical_id2_ancestor_reduction_child")
            reason = "CANONICAL_ID2_ANCESTOR_REDUCTION_STEP59"
            expr = _payload(
                "canonical_id2_ancestor_reduction",
                y,
                x,
                sub.formula,
                sub_ast,
                "ID-2",
                reason=reason,
                terms=[_rhs(sub.formula)],
                subexpressions=[{"status": sub.status, "formula": sub.formula, "reason_codes": sub.reason_codes}],
            )
            trace = [{"rule": "ID-2", "status": "applied_ancestor_reduction", "ancestral_nodes": "|".join(ancestors_y), "removed_non_ancestors": "|".join(removed_non_ancestors)}] + _trace_steps(sub.trace_json)
            return _diag("identified_canonical_id2_ancestor_reduction_step59", True, formula=sub.formula, ast=sub_ast, expression_json=expr, trace=trace, rule=sub.terminal_rule or "ID-2", terminal=sub.terminal_status or "identified_after_ancestor_reduction", reason=reason)
        return sub

    # ID-3: conservative W-step.  It is included so ID-4 subproblems can use the
    # same canonical path as the control overlay, but it does not introduce any
    # new completeness claim.
    graph_without_incoming_x = _remove_incoming_to(admg, x)
    ancestors_after_do = sorted(graph_without_incoming_x.ancestors(y))
    w = sorted((set(v) - set(x)) - set(ancestors_after_do))
    if w:
        sub = _canonical_id_formula_diagnostic(admg, sorted(set(x) | set(w)), y, depth=depth + 1, max_depth=max_depth, seen=seen)
        if sub.identified:
            reason = "CANONICAL_ID3_W_STEP_PROMOTION_STEP59"
            sub_ast = _ast_from_json(sub.formula_ast_json, "canonical_id3_w_step_child")
            expr = _payload("canonical_id3_w_step_promotion", y, x, sub.formula, sub_ast, "ID-3", reason=reason, terms=[_rhs(sub.formula)], subexpressions=[{"promoted_to_intervention": w, "status": sub.status, "formula": sub.formula}])
            trace = [{"rule": "ID-3", "status": "applied_w_step_promote_irrelevant_variables", "promoted_to_intervention": "|".join(w), "ancestors_after_do": "|".join(ancestors_after_do)}] + _trace_steps(sub.trace_json)
            return _diag("identified_canonical_id3_w_step_promotion_step59", True, formula=sub.formula, ast=sub_ast, expression_json=expr, trace=trace, rule=sub.terminal_rule or "ID-3", terminal=sub.terminal_status or "identified_after_w_step", reason=reason)
        return sub

    fd = _frontdoor_formula(admg, x, y)
    if fd is not None:
        return fd

    if not admg.bidirected_edges:
        return _observed_dag_formula(admg, x, y)

    id4 = _district_decomposition_formula(admg, x, y, depth=depth, max_depth=max_depth, seen=seen)
    if id4 is not None:
        return id4

    qf = _q_factor_formula(admg, x, y)
    if qf is not None:
        return qf

    fail = _direct_confounding_fail(admg, x, y)
    if fail is not None:
        return fail

    return _diag("pending_canonical_formula_unhandled_step67", False, trace=[{"rule": "ID-7", "status": "pending_general_id7"}], rule="ID-7", terminal="pending_general_id7", blocker_class="canonical_id7_pending", pending="complete_arbitrary_ID7_carried_Q_recursion_beyond_contextual_chain_and_parallel_frontdoor_sets", reason="CANONICAL_ID7_GENERAL_PENDING_STEP67")


def canonical_id_formula_diagnostic(admg: ADMG, treatments: Sequence[object] | object, outcomes: Sequence[object] | object, *, max_depth: int = 8) -> CanonicalFormulaDiagnostic:
    x = _nodes(treatments)
    y = _nodes(outcomes)
    missing = sorted([n for n in x + y if n not in admg.node_set])
    if not x or not y or missing or set(x) & set(y):
        reason = "INVALID_CANONICAL_FORMULA_QUERY_STEP59"
        return _diag("invalid_canonical_formula_query_step59", False, trace=[{"rule": "ID-5", "status": reason}], rule="ID-5", terminal=reason, blocker=reason, blocker_class="invalid_query", pending="validate_query", reason=reason)
    cycles = directed_cycle_nodes(admg)
    if cycles:
        reason = "DIRECTED_CYCLE_NOT_ADMG_DAG_STEP59"
        return _diag("blocked_canonical_formula_directed_cycle_step59", False, trace=[{"rule": "ID-5", "status": reason}], rule="ID-5", terminal="blocked_directed_cycle", blocker="|".join(cycles), blocker_class="directed_cycle", pending="repair_graph", reason=reason)
    return _canonical_id_formula_diagnostic(admg, x, y, depth=0, max_depth=max_depth, seen=set())


__all__ = [
    "CanonicalFormulaDiagnostic",
    "ID_CANONICAL_FORMULA_AUTHORITY",
    "ID_CANONICAL_FORMULA_LEVEL",
    "ID_CANONICAL_FORMULA_VERSION",
    "canonical_id_formula_diagnostic",
]
