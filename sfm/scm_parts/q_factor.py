from __future__ import annotations

"""Reusable q-factor identification diagnostics for recursive ID.

Step 53 aligns this standalone q-factor module with the recursive expression
engine introduced in Step 44 and the operational carried-Q AST path.  The module still keeps the old safe branches, but
strict subdistrict queries no longer stop at the historical "active X remains an
ancestor" blocker.  When the shape matches ID step 7, it now delegates to the
carried-Q recursive expression engine and returns one of three auditable states:

- identified_general_q_input_subdistrict_recursion;
- blocked_general_q_input_recursion_formal_hedge;
- blocked_general_q_input_subdistrict_recursion.

This is intentionally still conservative: it does not claim full ID globally;
it exposes a standalone q-factor diagnostic that agrees with the recursive ID
runtime.
"""

from dataclasses import asdict, dataclass
import json
import re
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple

from .admg import ADMG
from .graph_criteria import topological_order
from .id_algorithm_common import _conditional_factor, _dedupe, _format_component, _json_formula, _joint_symbol, _s
from .id_carried_q import build_carried_q_context


def _conditional_factor_from_previous(node: str, previous: Sequence[str]) -> str:
    prev = _joint_symbol(previous)
    return f"P({node} | {prev})" if prev else f"P({node})"


def _primed_name(node: str) -> str:
    return f"{node}'"


def _replace_symbol_token(text: str, old: str, new: str) -> str:
    if not old:
        return text
    return re.sub(rf"(?<![A-Za-z0-9_]){re.escape(old)}(?![A-Za-z0-9_])", new, text)


def _prime_bound_symbols(text: str, bound_nodes: Sequence[str]) -> str:
    out = text
    for node in sorted(_dedupe(bound_nodes), key=len, reverse=True):
        out = _replace_symbol_token(out, node, _primed_name(node))
    return out


@dataclass(frozen=True)
class QFactorIDDiagnostic:
    """Result of a conservative q-factor identification attempt."""

    q_factor_status: str
    q_factor_identified: bool = False
    q_factor_formula: str = ""
    q_factor_json: str = ""
    q_factor_target: str = ""
    q_factor_containing: str = ""
    q_factor_sum_over: str = ""
    q_factor_terms: str = ""
    q_factor_blocker: str = ""
    q_factor_reason_codes: str = ""
    q_factor_recursive_expression_json: str = ""
    q_factor_recursive_trace_json: str = ""
    q_factor_ast_json: str = ""

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _node_set(admg: ADMG, nodes: Iterable[object]) -> List[str]:
    return sorted(v for v in _dedupe(nodes) if v in admg.node_set)


def _payload(
    kind: str,
    *,
    target: Sequence[str],
    containing: Sequence[str],
    formula: str,
    sum_over: Sequence[str],
    terms: Sequence[str],
    extra: Mapping[str, object] | None = None,
    reason_codes: str = "",
) -> str:
    p: Dict[str, object] = {
        "type": kind,
        "target_district": list(_dedupe(target)),
        "containing_district": list(_dedupe(containing)),
        "formula": formula,
        "sum_over": list(_dedupe(sum_over)),
        "product_terms": list(terms),
        "reason_codes": reason_codes,
    }
    if extra:
        p.update(dict(extra))
    return _json_formula(p)


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


def _json_loads_or_empty(text: str) -> Dict[str, object]:
    try:
        obj = json.loads(text) if text else {}
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _ast_json_from_expression_payload(payload: Mapping[str, object]) -> str:
    ast = payload.get("formula_ast") if isinstance(payload, Mapping) else None
    if not isinstance(ast, Mapping):
        return ""
    return _json_formula(ast)


def _general_q_input_recursion_diagnostic(
    admg: ADMG,
    *,
    target: Sequence[str],
    containing: Sequence[str],
    y: Sequence[str],
    x: Sequence[str],
    q_input_scope: Sequence[str] = (),
    q_input_terms: Sequence[str] = (),
    q_input_name: str = "",
    max_depth: int = 8,
) -> QFactorIDDiagnostic:
    """Delegate a strict subdistrict q-factor query to the recursive engine.

    This is a lazy import to keep the public module graph acyclic for older
    downstream users that import q_factor before id_recursive_expression.
    """
    from .id_recursive_expression import _recursive_id_expression

    target = list(_dedupe(target))
    containing = list(_dedupe(containing))
    containing_set = set(containing)
    y_inside = [n for n in _dedupe(y) if n in containing_set]
    x_inside = [n for n in _dedupe(x) if n in containing_set and n not in set(y_inside)]
    if not y_inside:
        return QFactorIDDiagnostic(
            "blocked_general_q_input_no_outcome_inside_containing",
            False,
            q_factor_target=_format_component(target),
            q_factor_containing=_format_component(containing),
            q_factor_blocker="NO_OUTCOME_INSIDE_CONTAINING_DISTRICT",
            q_factor_reason_codes="NO_OUTCOME_INSIDE_CONTAINING_DISTRICT_STEP45",
        )

    q_ctx = build_carried_q_context(
        admg,
        containing,
        source_scope=q_input_scope,
        source_terms=q_input_terms,
        source_name=q_input_name,
        name=_s(q_input_name) or f"Q[{','.join(containing)}]",
    )
    q_scope = list(q_ctx.scope)
    q_terms = list(q_ctx.terms)
    q_name = q_ctx.name
    working = admg.induced_subgraph(containing)
    sub = _recursive_id_expression(
        working,
        y_inside,
        x_inside,
        max_depth=max_depth,
        q_input_scope=q_scope,
        q_input_terms=q_terms,
        q_input_name=q_name,
        q_input_formula_ast=q_ctx.formula_ast,
    )
    sub_payload = _json_loads_or_empty(sub.expression_json)
    base_extra: Dict[str, object] = {
        "rule": "general_q_input_recursive_subproblem_step45",
        "q_input": q_name,
        "q_input_scope": q_scope,
        "q_input_terms": q_terms,
        "subdistrict": target,
        "step51_carried_q_context_enabled": 1,
        "step53_operational_carried_q_ast_enabled": int(bool(q_ctx.formula_ast)),
        "carried_q_context": q_ctx.to_dict(),
        "q_input_formula_ast": dict(q_ctx.formula_ast) if isinstance(q_ctx.formula_ast, Mapping) else {},
        "recursive_status": sub.expression_status,
        "recursive_identified": bool(sub.expression_identified),
        "recursive_reason_codes": sub.reason_codes,
        "recursive_expression": sub_payload,
    }
    if sub.expression_identified:
        reason = sub.reason_codes or "Q_FACTOR_GENERAL_Q_INPUT_RECURSION_IDENTIFIED_STEP45"
        return QFactorIDDiagnostic(
            "identified_general_q_input_subdistrict_recursion",
            True,
            q_factor_formula=sub.formula,
            q_factor_json=_payload(
                "general_q_input_subdistrict_recursion",
                target=target,
                containing=containing,
                formula=sub.formula,
                sum_over=sub_payload.get("sum_over", []) if isinstance(sub_payload.get("sum_over"), list) else [],
                terms=sub_payload.get("product_terms", []) if isinstance(sub_payload.get("product_terms"), list) else [],
                extra=base_extra,
                reason_codes=reason,
            ),
            q_factor_target=_format_component(target),
            q_factor_containing=_format_component(containing),
            q_factor_sum_over="|".join(sub_payload.get("sum_over", [])) if isinstance(sub_payload.get("sum_over"), list) else "",
            q_factor_terms="|".join(sub_payload.get("product_terms", [])) if isinstance(sub_payload.get("product_terms"), list) else "",
            q_factor_reason_codes=reason,
            q_factor_recursive_expression_json=sub.expression_json,
            q_factor_recursive_trace_json=sub.trace_json,
            q_factor_ast_json=_ast_json_from_expression_payload(sub_payload),
        )

    if sub.blocker_class == "formal_hedge_certificate" or sub.expression_status == "blocked_formal_hedge_certificate":
        reason = sub.reason_codes or "Q_FACTOR_GENERAL_Q_INPUT_RECURSION_HEDGE_FAIL_STEP45"
        hedge = sub_payload.get("formal_hedge_candidate") if isinstance(sub_payload, Mapping) else None
        extra = dict(base_extra)
        if isinstance(hedge, Mapping):
            extra["formal_hedge_candidate"] = dict(hedge)
        return QFactorIDDiagnostic(
            "blocked_general_q_input_recursion_formal_hedge",
            False,
            q_factor_json=_payload(
                "general_q_input_recursion_formal_hedge",
                target=target,
                containing=containing,
                formula="",
                sum_over=[],
                terms=[],
                extra=extra,
                reason_codes=reason,
            ),
            q_factor_target=_format_component(target),
            q_factor_containing=_format_component(containing),
            q_factor_blocker=sub.blocker or _format_component(target),
            q_factor_reason_codes=reason,
            q_factor_recursive_expression_json=sub.expression_json,
            q_factor_recursive_trace_json=sub.trace_json,
            q_factor_ast_json=_ast_json_from_expression_payload(sub_payload),
        )

    reason = sub.reason_codes or "Q_FACTOR_GENERAL_Q_INPUT_RECURSION_SUBPROBLEM_BLOCKED_STEP45"
    return QFactorIDDiagnostic(
        "blocked_general_q_input_subdistrict_recursion",
        False,
        q_factor_json=_payload(
            "general_q_input_subdistrict_recursion_blocked",
            target=target,
            containing=containing,
            formula="",
            sum_over=[],
            terms=[],
            extra=base_extra,
            reason_codes=reason,
        ),
        q_factor_target=_format_component(target),
        q_factor_containing=_format_component(containing),
        q_factor_blocker=sub.blocker or _format_component(target),
        q_factor_reason_codes=reason,
        q_factor_recursive_expression_json=sub.expression_json,
        q_factor_recursive_trace_json=sub.trace_json,
        q_factor_ast_json=_ast_json_from_expression_payload(sub_payload),
    )


def identify_q_factor(
    admg: ADMG,
    target_district: Sequence[object],
    *,
    containing_district: Sequence[object] | None = None,
    outcome_set: Sequence[object] = (),
    intervention_set: Sequence[object] = (),
    q_input_scope: Sequence[object] = (),
    q_input_terms: Sequence[object] = (),
    q_input_name: str = "",
    enable_general_recursion: bool = True,
    max_depth: int = 8,
) -> QFactorIDDiagnostic:
    """Identify a q-factor branch when the shape is auditable.

    Parameters mirror the ID recursion notation.  ``target_district`` is the
    district whose q-factor is needed.  ``containing_district`` is optional; if
    omitted, the function tries the full-district chain-rule branch.
    """
    target = _node_set(admg, target_district)
    containing = _node_set(admg, containing_district or target)
    y = _node_set(admg, outcome_set or target)
    x = _node_set(admg, intervention_set)
    q_scope = _node_set(admg, q_input_scope)
    q_terms = [_s(t) for t in q_input_terms or () if _s(t)]
    if not target:
        return QFactorIDDiagnostic("invalid_q_factor_query", False, q_factor_blocker="EMPTY_TARGET_DISTRICT", q_factor_reason_codes="EMPTY_TARGET_DISTRICT")
    if not set(target).issubset(set(containing)):
        return QFactorIDDiagnostic(
            "blocked_target_not_subset_of_containing_district",
            False,
            q_factor_target=_format_component(target),
            q_factor_containing=_format_component(containing),
            q_factor_blocker="TARGET_NOT_SUBSET_OF_CONTAINING_DISTRICT",
            q_factor_reason_codes="TARGET_NOT_SUBSET_OF_CONTAINING_DISTRICT",
        )
    order = topological_order(admg)
    if not order:
        return QFactorIDDiagnostic(
            "blocked_directed_cycle",
            False,
            q_factor_target=_format_component(target),
            q_factor_containing=_format_component(containing),
            q_factor_blocker="DIRECTED_CYCLE_NOT_ADMG_DAG",
            q_factor_reason_codes="DIRECTED_CYCLE_NOT_ADMG_DAG",
        )

    full_districts = [set(d) for d in admg.districts()]
    target_set = set(target)
    containing_set = set(containing)

    if target_set == containing_set and any(target_set == d for d in full_districts):
        previous: List[str] = []
        terms: List[str] = q_terms if q_terms and set(q_scope or containing) == target_set else []
        if not terms:
            for node in order:
                if node in target_set:
                    terms.append(_conditional_factor_from_previous(node, previous))
                previous.append(node)
        sum_over = [node for node in order if node in target_set and node not in set(y)]
        product = " * ".join(terms) if terms else "1"
        formula = f"sum_{{{','.join(sum_over)}}} {product}" if sum_over else product
        reason = "Q_FACTOR_FULL_DISTRICT_IDENTIFIED_FROM_CARRIED_Q_STEP45" if q_terms else "Q_FACTOR_FULL_DISTRICT_IDENTIFIED_FROM_CHAIN_RULE_STEP40"
        return QFactorIDDiagnostic(
            "identified_full_district_q_factor",
            True,
            q_factor_formula=formula,
            q_factor_json=_payload(
                "full_district_q_factor",
                target=target,
                containing=containing,
                formula=formula,
                sum_over=sum_over,
                terms=terms,
                extra={"q_input_scope": q_scope, "q_input_name": _s(q_input_name)} if q_terms else None,
                reason_codes=reason,
            ),
            q_factor_target=_format_component(target),
            q_factor_containing=_format_component(containing),
            q_factor_sum_over="|".join(sum_over),
            q_factor_terms="|".join(terms),
            q_factor_reason_codes=reason,
        )

    # Safe q-input subdistrict branch: target is a strict subdistrict inside a
    # containing full district, and no active intervention inside that containing
    # district remains an ancestor of the requested outcomes inside G[containing].
    # Step 45: if that safe condition fails, fall through to the general carried-Q
    # recursion path instead of returning a flat blocker.
    if target_set < containing_set and any(containing_set == d for d in full_districts):
        working = admg.induced_subgraph(sorted(containing_set))
        y_inside = [n for n in y if n in containing_set]
        x_inside = [n for n in x if n in containing_set and n not in set(y_inside)]
        if not y_inside:
            return QFactorIDDiagnostic(
                "blocked_subdistrict_q_factor_no_outcome_inside_containing",
                False,
                q_factor_target=_format_component(target),
                q_factor_containing=_format_component(containing),
                q_factor_blocker="NO_OUTCOME_INSIDE_CONTAINING_DISTRICT",
                q_factor_reason_codes="NO_OUTCOME_INSIDE_CONTAINING_DISTRICT",
            )
        ancestors_y = sorted(working.ancestors(y_inside))
        active_x_ancestors = sorted(set(x_inside) & set(ancestors_y))
        if not active_x_ancestors:
            previous = []
            raw_terms: List[str] = []
            if q_terms and set(q_scope or containing) == containing_set:
                raw_terms = q_terms
            else:
                for node in order:
                    if node in containing_set:
                        raw_terms.append(_conditional_factor_from_previous(node, previous))
                    previous.append(node)
            sum_over_raw = [node for node in order if node in containing_set and node not in set(y_inside)]
            sum_over = [_primed_name(node) if node in set(x_inside) else node for node in sum_over_raw]
            terms = [_prime_bound_symbols(term, x_inside) for term in raw_terms]
            product = " * ".join(terms) if terms else "1"
            formula = f"sum_{{{','.join(sum_over)}}} {product}" if sum_over else product
            reason = "Q_INPUT_SUBDISTRICT_RECURSION_IDENTIFIED_STEP40"
            return QFactorIDDiagnostic(
                "identified_safe_subdistrict_q_input",
                True,
                q_factor_formula=formula,
                q_factor_json=_payload(
                    "safe_subdistrict_q_input",
                    target=target,
                    containing=containing,
                    formula=formula,
                    sum_over=sum_over,
                    terms=terms,
                    extra={
                        "ancestors_inside_containing_district": ancestors_y,
                        "active_interventions_inside_containing_district": x_inside,
                        "q_input_scope": q_scope,
                        "q_input_name": _s(q_input_name),
                    },
                    reason_codes=reason,
                ),
                q_factor_target=_format_component(target),
                q_factor_containing=_format_component(containing),
                q_factor_sum_over="|".join(sum_over),
                q_factor_terms="|".join(terms),
                q_factor_reason_codes=reason,
            )

        if enable_general_recursion:
            return _general_q_input_recursion_diagnostic(
                admg,
                target=target,
                containing=containing,
                y=y,
                x=x,
                q_input_scope=q_scope or containing,
                q_input_terms=q_terms,
                q_input_name=q_input_name,
                max_depth=max_depth,
            )

        reason = "Q_INPUT_RECURSION_BLOCKED_ACTIVE_X_REMAINS_ANCESTOR_STEP40"
        return QFactorIDDiagnostic(
            "blocked_q_input_active_x_remains_ancestor",
            False,
            q_factor_target=_format_component(target),
            q_factor_containing=_format_component(containing),
            q_factor_json=_payload(
                "blocked_q_input_subdistrict",
                target=target,
                containing=containing,
                formula="",
                sum_over=[],
                terms=[],
                extra={"active_intervention_ancestors": active_x_ancestors, "ancestors_inside_containing_district": ancestors_y},
                reason_codes=reason,
            ),
            q_factor_blocker="|".join(active_x_ancestors),
            q_factor_reason_codes=reason,
        )

    return QFactorIDDiagnostic(
        "blocked_q_factor_shape_unsupported",
        False,
        q_factor_target=_format_component(target),
        q_factor_containing=_format_component(containing),
        q_factor_blocker="Q_FACTOR_SHAPE_UNSUPPORTED",
        q_factor_reason_codes="Q_FACTOR_SHAPE_UNSUPPORTED_STEP40",
    )
