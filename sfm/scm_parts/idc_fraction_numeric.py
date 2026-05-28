from __future__ import annotations

"""Conservative numeric-readiness helpers for IDC fraction ASTs.

Step 65 does *not* claim full numeric Full-ID evaluation.  It only recognizes
ratio/normalization ASTs produced by the IDC layer and authorizes a guarded
conditional-mean standardization route when the AST is fully resolved into
probability/sum/product/do nodes.
"""

from dataclasses import asdict, dataclass
from typing import Iterable, List, Mapping, Sequence

from .id_ast import FormulaAST, ast_from_dict
from .id_ast_normalizer import normalize_formula_ast

IDC_FRACTION_NUMERIC_VERSION = "idc_fraction_numeric_v1_step65"


def _s(value: object) -> str:
    raw = "" if value is None else str(value).strip()
    return "" if raw.lower() in {"nan", "none", "null", "nat"} else raw


def _dedupe(values: Iterable[object]) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values or []:
        item = _s(value)
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _walk(ast: FormulaAST) -> List[FormulaAST]:
    nodes = [ast]
    for child in ast.children:
        nodes.extend(_walk(child))
    return nodes


def _first_probability(ast: FormulaAST) -> FormulaAST | None:
    for node in _walk(ast):
        if node.node_type == "probability":
            return node
    return None


def _node_types(ast: FormulaAST) -> List[str]:
    return _dedupe(node.node_type for node in _walk(ast))


@dataclass(frozen=True)
class IDCFractionNumericPlan:
    status: str
    numeric_ready: int
    route: str
    version: str
    outcome_variables: List[str]
    condition_variables: List[str]
    intervention_variables: List[str]
    denominator_sum_over: List[str]
    node_types: List[str]
    blocker: str = ""
    reason_codes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


_ALLOWED_NODE_TYPES = {"probability", "product", "sum", "fraction", "do"}


def analyze_idc_fraction_ast(ast_payload: FormulaAST | Mapping[str, object] | None, *, outcome_hint: Sequence[object] = (), treatment_hint: Sequence[object] = ()) -> IDCFractionNumericPlan:
    """Return a conservative route plan for an IDC fraction AST.

    Numeric readiness is granted only when:
    - the top-level node is ``fraction`` with numerator and denominator;
    - the tree contains no q_factor, placeholder, hedge_fail, or unknown node;
    - the denominator contains a sum node, usually ``sum_Y numerator``;
    - at least one outcome and intervention can be inferred.
    """
    try:
        ast = normalize_formula_ast(ast_payload if not isinstance(ast_payload, Mapping) else ast_from_dict(ast_payload))
    except Exception as exc:  # pragma: no cover - defensive
        return IDCFractionNumericPlan(
            status="blocked_idc_fraction_ast_parse_error_step65",
            numeric_ready=0,
            route="",
            version=IDC_FRACTION_NUMERIC_VERSION,
            outcome_variables=_dedupe(outcome_hint),
            condition_variables=[],
            intervention_variables=_dedupe(treatment_hint),
            denominator_sum_over=[],
            node_types=[],
            blocker=f"IDC_FRACTION_AST_PARSE_ERROR:{type(exc).__name__}",
            reason_codes="IDC_FRACTION_AST_PARSE_ERROR",
        )

    nodes = _walk(ast)
    node_types = _node_types(ast)
    forbidden = [nt for nt in node_types if nt not in _ALLOWED_NODE_TYPES]
    if ast.node_type != "fraction":
        return IDCFractionNumericPlan(
            status="blocked_not_idc_fraction_ast_step65",
            numeric_ready=0,
            route="",
            version=IDC_FRACTION_NUMERIC_VERSION,
            outcome_variables=_dedupe(outcome_hint),
            condition_variables=[],
            intervention_variables=_dedupe(treatment_hint),
            denominator_sum_over=[],
            node_types=node_types,
            blocker="TOP_LEVEL_AST_NOT_FRACTION",
            reason_codes="IDC_NUMERIC_REQUIRES_TOP_LEVEL_FRACTION_AST",
        )
    if forbidden:
        return IDCFractionNumericPlan(
            status="blocked_idc_fraction_unresolved_ast_nodes_step65",
            numeric_ready=0,
            route="",
            version=IDC_FRACTION_NUMERIC_VERSION,
            outcome_variables=_dedupe(outcome_hint),
            condition_variables=[],
            intervention_variables=_dedupe(treatment_hint),
            denominator_sum_over=[],
            node_types=node_types,
            blocker="UNRESOLVED_IDC_FRACTION_AST_NODES=" + "|".join(forbidden),
            reason_codes="IDC_FRACTION_AST_HAS_UNSUPPORTED_OR_UNRESOLVED_NODES",
        )
    if len(ast.children) < 2:
        return IDCFractionNumericPlan(
            status="blocked_idc_fraction_missing_denominator_step65",
            numeric_ready=0,
            route="",
            version=IDC_FRACTION_NUMERIC_VERSION,
            outcome_variables=_dedupe(outcome_hint),
            condition_variables=[],
            intervention_variables=_dedupe(treatment_hint),
            denominator_sum_over=[],
            node_types=node_types,
            blocker="IDC_FRACTION_MISSING_DENOMINATOR",
            reason_codes="IDC_FRACTION_AST_REQUIRES_NUMERATOR_AND_DENOMINATOR",
        )

    numerator = ast.children[0]
    denominator = ast.children[1]
    numerator_p = _first_probability(numerator)
    if numerator_p is None:
        return IDCFractionNumericPlan(
            status="blocked_idc_fraction_no_numerator_probability_step65",
            numeric_ready=0,
            route="",
            version=IDC_FRACTION_NUMERIC_VERSION,
            outcome_variables=_dedupe(outcome_hint),
            condition_variables=[],
            intervention_variables=_dedupe(treatment_hint),
            denominator_sum_over=[],
            node_types=node_types,
            blocker="IDC_FRACTION_NUMERATOR_HAS_NO_PROBABILITY_TERM",
            reason_codes="IDC_NUMERIC_REQUIRES_PROBABILITY_NUMERATOR",
        )

    denominator_sum = None
    for node in _walk(denominator):
        if node.node_type == "sum" and node.bound_variables:
            denominator_sum = node
            break
    denominator_sum_over = _dedupe(denominator_sum.bound_variables if denominator_sum else [])
    outcomes = _dedupe(outcome_hint or denominator_sum_over or numerator_p.variables[:1])
    interventions = _dedupe(treatment_hint or numerator_p.interventions)
    numerator_vars = _dedupe(numerator_p.variables)
    conditions = _dedupe([v for v in numerator_vars if v not in outcomes])

    blockers: List[str] = []
    if not denominator_sum_over:
        blockers.append("IDC_FRACTION_DENOMINATOR_HAS_NO_SUM_NODE")
    if not outcomes:
        blockers.append("IDC_FRACTION_OUTCOME_NOT_INFERRED")
    if not interventions:
        blockers.append("IDC_FRACTION_INTERVENTION_NOT_INFERRED")

    if blockers:
        return IDCFractionNumericPlan(
            status="blocked_idc_fraction_incomplete_normalization_plan_step65",
            numeric_ready=0,
            route="",
            version=IDC_FRACTION_NUMERIC_VERSION,
            outcome_variables=outcomes,
            condition_variables=conditions,
            intervention_variables=interventions,
            denominator_sum_over=denominator_sum_over,
            node_types=node_types,
            blocker="|".join(blockers),
            reason_codes="IDC_FRACTION_NUMERIC_PLAN_INCOMPLETE",
        )

    return IDCFractionNumericPlan(
        status="idc_fraction_numeric_ready_step65",
        numeric_ready=1,
        route="symbolic_numeric_idc_fraction_ratio",
        version=IDC_FRACTION_NUMERIC_VERSION,
        outcome_variables=outcomes,
        condition_variables=conditions,
        intervention_variables=interventions,
        denominator_sum_over=denominator_sum_over,
        node_types=node_types,
        reason_codes="IDC_FRACTION_RATIO_NORMALIZATION_READY_FOR_CONTRACT_GATED_CONDITIONAL_MEAN_STANDARDIZATION",
    )


__all__ = ["IDC_FRACTION_NUMERIC_VERSION", "IDCFractionNumericPlan", "analyze_idc_fraction_ast"]
