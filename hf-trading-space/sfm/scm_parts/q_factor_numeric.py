from __future__ import annotations

"""Conservative numeric-readiness helper for resolved Q-factor ASTs.

Step 66 does not make arbitrary Q-input recursion numeric.  It only recognizes
Q-factor ASTs that are already expanded into probability/sum/product/do children
and routes them to the same contract-gated standardization family used by other
resolved ID formulas.  Bare Q[D], nested Q-inputs, placeholders, hedge failures,
and unknown nodes remain blocked.
"""

from dataclasses import asdict, dataclass
from typing import Iterable, List, Mapping, Sequence

from .id_ast import FormulaAST, ast_from_dict
from .id_ast_normalizer import normalize_formula_ast

Q_FACTOR_NUMERIC_VERSION = "q_factor_numeric_v1_step66"


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


def _node_types(ast: FormulaAST) -> List[str]:
    return _dedupe(node.node_type for node in _walk(ast))


@dataclass(frozen=True)
class QFactorNumericPlan:
    status: str
    numeric_ready: int
    route: str
    version: str
    q_factor_variables: List[str]
    probability_terms: List[str]
    bound_variables: List[str]
    node_types: List[str]
    blocker: str = ""
    reason_codes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


_ALLOWED_CHILD_NODE_TYPES = {"probability", "product", "sum", "do"}


def _prob_label(node: FormulaAST) -> str:
    left = ",".join(node.variables)
    right = ",".join(node.conditioned_on)
    do = ",".join(node.interventions)
    base = f"P({left}{' | ' + right if right else ''})"
    return f"P_{{do({do})}}({left}{' | ' + right if right else ''})" if do else base


def analyze_resolved_q_factor_ast(ast_payload: FormulaAST | Mapping[str, object] | None) -> QFactorNumericPlan:
    """Return a route plan for a resolved q_factor AST.

    Numeric readiness is granted only when the top-level node is ``q_factor``, it
    has expanded children, and all children are already probability/sum/product/do
    nodes.  This prevents a bare carried-Q symbol from being treated as an
    executable estimand.
    """
    try:
        ast = normalize_formula_ast(ast_payload if not isinstance(ast_payload, Mapping) else ast_from_dict(ast_payload))
    except Exception as exc:  # pragma: no cover - defensive
        return QFactorNumericPlan(
            status="blocked_q_factor_ast_parse_error_step66",
            numeric_ready=0,
            route="",
            version=Q_FACTOR_NUMERIC_VERSION,
            q_factor_variables=[],
            probability_terms=[],
            bound_variables=[],
            node_types=[],
            blocker=f"Q_FACTOR_AST_PARSE_ERROR:{type(exc).__name__}",
            reason_codes="Q_FACTOR_AST_PARSE_ERROR",
        )

    nodes = _walk(ast)
    node_types = _node_types(ast)
    prob_terms = [_prob_label(n) for n in nodes if n.node_type == "probability"]
    bound = []
    for n in nodes:
        bound.extend(n.bound_variables)
    q_nodes = [n for n in nodes if n.node_type == "q_factor"]

    if ast.node_type != "q_factor":
        return QFactorNumericPlan(
            status="blocked_not_q_factor_ast_step66",
            numeric_ready=0,
            route="",
            version=Q_FACTOR_NUMERIC_VERSION,
            q_factor_variables=[],
            probability_terms=prob_terms,
            bound_variables=_dedupe(bound),
            node_types=node_types,
            blocker="TOP_LEVEL_AST_NOT_Q_FACTOR",
            reason_codes="Q_FACTOR_NUMERIC_REQUIRES_TOP_LEVEL_Q_FACTOR_AST",
        )
    if not ast.children:
        return QFactorNumericPlan(
            status="blocked_bare_q_factor_ast_step66",
            numeric_ready=0,
            route="",
            version=Q_FACTOR_NUMERIC_VERSION,
            q_factor_variables=list(ast.variables),
            probability_terms=prob_terms,
            bound_variables=_dedupe(bound),
            node_types=node_types,
            blocker="BARE_Q_FACTOR_WITHOUT_EXPANDED_TERMS",
            reason_codes="Q_FACTOR_NUMERIC_REQUIRES_EXPANDED_PROBABILITY_CHILDREN",
        )
    if len(q_nodes) > 1:
        return QFactorNumericPlan(
            status="blocked_nested_q_factor_ast_step66",
            numeric_ready=0,
            route="",
            version=Q_FACTOR_NUMERIC_VERSION,
            q_factor_variables=list(ast.variables),
            probability_terms=prob_terms,
            bound_variables=_dedupe(bound),
            node_types=node_types,
            blocker="NESTED_Q_FACTOR_AST",
            reason_codes="Q_FACTOR_NUMERIC_DOES_NOT_EXECUTE_NESTED_CARRIED_Q_INPUTS",
        )
    forbidden = sorted(nt for nt in set(node_types) if nt not in _ALLOWED_CHILD_NODE_TYPES | {"q_factor"})
    if forbidden:
        return QFactorNumericPlan(
            status="blocked_q_factor_unresolved_ast_nodes_step66",
            numeric_ready=0,
            route="",
            version=Q_FACTOR_NUMERIC_VERSION,
            q_factor_variables=list(ast.variables),
            probability_terms=prob_terms,
            bound_variables=_dedupe(bound),
            node_types=node_types,
            blocker="UNRESOLVED_Q_FACTOR_AST_NODES=" + "|".join(forbidden),
            reason_codes="Q_FACTOR_AST_HAS_UNSUPPORTED_OR_UNRESOLVED_NODES",
        )
    if not prob_terms:
        return QFactorNumericPlan(
            status="blocked_q_factor_no_probability_terms_step66",
            numeric_ready=0,
            route="",
            version=Q_FACTOR_NUMERIC_VERSION,
            q_factor_variables=list(ast.variables),
            probability_terms=prob_terms,
            bound_variables=_dedupe(bound),
            node_types=node_types,
            blocker="Q_FACTOR_AST_HAS_NO_PROBABILITY_TERMS",
            reason_codes="Q_FACTOR_NUMERIC_REQUIRES_OBSERVED_PROBABILITY_TERMS",
        )

    return QFactorNumericPlan(
        status="q_factor_numeric_ready_step66",
        numeric_ready=1,
        route="symbolic_numeric_resolved_q_factor_standardization",
        version=Q_FACTOR_NUMERIC_VERSION,
        q_factor_variables=list(ast.variables),
        probability_terms=prob_terms,
        bound_variables=_dedupe(bound),
        node_types=node_types,
        reason_codes="RESOLVED_Q_FACTOR_AST_READY_FOR_CONTRACT_GATED_STANDARDIZATION_NO_ARBITRARY_Q_INPUT_CLAIM",
    )


__all__ = ["Q_FACTOR_NUMERIC_VERSION", "QFactorNumericPlan", "analyze_resolved_q_factor_ast"]
