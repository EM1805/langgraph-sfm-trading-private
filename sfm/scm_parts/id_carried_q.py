from __future__ import annotations

"""Carried-Q context utilities for recursive ID.

Step 52 upgrades the Step-51 bookkeeping object into a small symbolic object:
every carried-Q context now exposes both display terms and an ``id_ast_v1``
formula AST for the carried c-factor ``Q[S']``.  This still does not declare
arbitrary Full ID; it simply removes one more placeholder layer from ID-7 by
making the carried input executable/auditable as structured formula data.
"""

from dataclasses import asdict, dataclass, field
import json
from typing import Dict, List, Mapping, Sequence, Set

from .admg import ADMG
from .graph_criteria import topological_order
from .id_algorithm_common import _dedupe, _joint_symbol, _s
from .id_ast import FormulaAST, Product, Q, Placeholder, parse_factor_term


CARRIED_Q_CONTEXT_VERSION = "carried_q_context_v2_step52_ast"


def _conditional_factor_from_previous(node: str, previous: Sequence[str]) -> str:
    prev = _joint_symbol(previous)
    return f"P({node} | {prev})" if prev else f"P({node})"


def _factor_lhs_variable(term: str) -> str:
    text = _s(term)
    if not text.startswith("P(") or not text.endswith(")"):
        return ""
    body = text[2:-1]
    lhs = body.split("|", 1)[0].strip()
    return lhs.split(",", 1)[0].strip()


def _product_formula(terms: Sequence[object]) -> str:
    clean = [_s(t) for t in terms or () if _s(t)]
    return " * ".join(clean) if clean else "1"


def carried_q_formula_string(name: str, terms: Sequence[object]) -> str:
    return f"{_s(name) or 'Q[]'} = {_product_formula(terms)}"


def carried_q_formula_ast(
    scope: Sequence[object],
    terms: Sequence[object],
    *,
    q_input: str = "",
    label: str = "carried_q_context",
    metadata: Mapping[str, object] | None = None,
) -> FormulaAST:
    """Build an AST for a carried Q-factor from its chain-rule terms.

    Terms are parsed into probability leaves where possible.  Unparseable terms
    become placeholders, so this helper cannot accidentally create stronger ID
    authority than the display formula already had.
    """
    factors = [parse_factor_term(t) for t in terms or () if _s(t)]
    if not factors:
        factors = [Placeholder("1", metadata={"constant": 1})]
    ast = Q(_dedupe(scope), terms=(Product(factors, label="carried_q_product"),), q_input=q_input, label=label)
    if metadata:
        return FormulaAST(
            ast.node_type,
            variables=ast.variables,
            conditioned_on=ast.conditioned_on,
            interventions=ast.interventions,
            bound_variables=ast.bound_variables,
            children=ast.children,
            label=ast.label,
            metadata={**dict(ast.metadata), **dict(metadata)},
        )
    return ast


def chain_rule_terms_for_scope(admg: ADMG, scope: Sequence[object]) -> List[str]:
    order = topological_order(admg)
    scope_set = set(_dedupe(scope))
    previous: List[str] = []
    terms: List[str] = []
    for node in order:
        if node in scope_set:
            terms.append(_conditional_factor_from_previous(node, previous))
        previous.append(node)
    return terms


def project_terms_to_scope(terms: Sequence[object], scope: Sequence[object]) -> List[str]:
    wanted = set(_dedupe(scope))
    out: List[str] = []
    seen: Set[str] = set()
    for term in terms or ():
        clean = _s(term)
        lhs = _factor_lhs_variable(clean)
        if clean and lhs in wanted and clean not in seen:
            seen.add(clean)
            out.append(clean)
    return out


@dataclass(frozen=True)
class CarriedQContext:
    scope: tuple[str, ...]
    name: str
    terms: tuple[str, ...]
    source_scope: tuple[str, ...] = ()
    source_name: str = ""
    source: str = "chain_rule"
    projected_from_source: int = 0
    projection_loss: int = 0
    formula: str = ""
    formula_ast: Mapping[str, object] = field(default_factory=dict)
    formula_ast_version: str = "id_ast_v1"
    context_version: str = CARRIED_Q_CONTEXT_VERSION

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


def build_carried_q_context(
    admg: ADMG,
    scope: Sequence[object],
    *,
    source_scope: Sequence[object] = (),
    source_terms: Sequence[object] = (),
    source_name: str = "",
    name: str = "",
) -> CarriedQContext:
    clean_scope = tuple(_dedupe([n for n in scope if _s(n) in admg.node_set]))
    clean_source_scope = tuple(_dedupe([n for n in source_scope if _s(n)]))
    clean_source_terms = tuple(_s(t) for t in source_terms or () if _s(t))
    q_name = _s(name) or f"Q[{','.join(clean_scope)}]"
    src_name = _s(source_name) or (f"Q[{','.join(clean_source_scope)}]" if clean_source_scope else "")

    scope_set = set(clean_scope)
    source_scope_set = set(clean_source_scope)
    projected_terms: List[str] = []
    projected_from_source = 0
    projection_loss = 0
    source = "chain_rule"

    if clean_source_terms and scope_set and scope_set.issubset(source_scope_set or scope_set):
        projected_terms = project_terms_to_scope(clean_source_terms, clean_scope)
        if projected_terms:
            projected_from_source = 1
            source = "projected_from_carried_q_input"
            lhs_seen = {_factor_lhs_variable(t) for t in projected_terms}
            projection_loss = int(not scope_set.issubset(lhs_seen))

    terms = tuple(projected_terms or chain_rule_terms_for_scope(admg, clean_scope))
    formula = carried_q_formula_string(q_name, terms)
    ast_meta = {
        "context_version": CARRIED_Q_CONTEXT_VERSION,
        "source_scope": list(clean_source_scope),
        "source_name": src_name,
        "source": source,
        "projected_from_source": projected_from_source,
        "projection_loss": projection_loss,
        "formula": formula,
    }
    ast = carried_q_formula_ast(clean_scope, terms, q_input=src_name, metadata=ast_meta)
    return CarriedQContext(
        scope=clean_scope,
        name=q_name,
        terms=terms,
        source_scope=clean_source_scope,
        source_name=src_name,
        source=source,
        projected_from_source=projected_from_source,
        projection_loss=projection_loss,
        formula=formula,
        formula_ast=ast.to_dict(),
    )


def carried_q_context_json(context: CarriedQContext | Mapping[str, object]) -> str:
    payload = context.to_dict() if isinstance(context, CarriedQContext) else dict(context)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


__all__ = [
    "CARRIED_Q_CONTEXT_VERSION",
    "CarriedQContext",
    "build_carried_q_context",
    "carried_q_context_json",
    "carried_q_formula_ast",
    "carried_q_formula_string",
    "chain_rule_terms_for_scope",
    "project_terms_to_scope",
]
