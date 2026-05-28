from __future__ import annotations

"""Symbolic formula and factorization diagnostics for the conservative ID layer.

This module is deliberately non-authoritative: it only serializes formulas and
observed-DAG truncation evidence already authorized by ``id_algorithm``.  Keeping
these helpers outside the core recursive-ID router makes the SCM layer easier to
audit without changing public outputs.
"""

from dataclasses import asdict, dataclass
import json
from typing import Any, Dict, Mapping, Optional, Sequence

from .admg import ADMG
from .graph_criteria import topological_order
from .id_ast import ast_to_json, payload_to_ast
from .id_algorithm_common import (
    _conditional_factor,
    _dedupe,
    _joint_symbol,
    _json_formula,
    _s,
    parse_field_list,
    parse_formula_term_list,
)


@dataclass(frozen=True)
class SymbolicFormulaDiagnostic:
    """Structured symbolic representation for supported limited estimands.

    The human-readable ``estimand_formula`` string stays for reports.  This
    diagnostic adds machine-readable structure for the same formula, so tests,
    CLI/report writers, and future symbolic engines do not have to parse free
    text.  It is deliberately conservative: unresolved c-factors remain explicit
    placeholders and are never marked as identified.
    """

    symbolic_formula_status: str
    symbolic_formula_kind: str = ""
    symbolic_formula_json: str = ""
    symbolic_formula_latex: str = ""
    symbolic_sum_over: str = ""
    symbolic_product_terms: str = ""
    symbolic_removed_terms: str = ""
    symbolic_unresolved_terms: str = ""
    symbolic_reason_codes: str = ""
    symbolic_formula_ast_json: str = ""

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TruncatedFactorizationDiagnostic:
    """Observed-DAG truncated factorization diagnostic.

    This is valid only when the graph has no explicit bidirected edges and the
    directed part is acyclic. It emits an audit-friendly estimand rather than a
    fully symbolic algebra system.
    """

    factorization_status: str
    factorization_ok: bool
    topological_order: str = ""
    eliminated_nodes: str = ""
    removed_factors: str = ""
    retained_factors: str = ""
    formula: str = ""
    reason_codes: str = ""

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)




def truncated_factorization_diagnostic(admg: ADMG, treatment: str, outcome: str) -> TruncatedFactorizationDiagnostic:
    """Return the observed-DAG truncated factorization formula when valid.

    For a DAG with observed nodes V and intervention X, the supported estimand is

        P_x(Y) = sum_{V \\ {X,Y}} prod_{V_i in V \\ {X}} P(V_i | Pa_i)

    The function deliberately refuses latent/bidirected graphs; those require
    recursive ID or one of the limited backdoor/front-door routes above.
    """
    x = _s(treatment)
    y = _s(outcome)
    if not x or not y or x not in admg.node_set or y not in admg.node_set:
        return TruncatedFactorizationDiagnostic("invalid_query", False, reason_codes="MISSING_QUERY_NODE")
    if admg.bidirected_edges:
        return TruncatedFactorizationDiagnostic(
            "not_applicable_bidirected_edges_present",
            False,
            reason_codes="EXPLICIT_LATENT_CONFOUNDING_REQUIRES_ID_OR_ADJUSTMENT",
        )
    order = topological_order(admg)
    if not order:
        return TruncatedFactorizationDiagnostic(
            "blocked_directed_cycle",
            False,
            reason_codes="DIRECTED_CYCLE_NOT_A_DAG",
        )

    parents = admg.parents()
    eliminated = [n for n in order if n not in {x, y}]
    retained_terms = [_conditional_factor(n, sorted(parents.get(n, set()))) for n in order if n != x]
    removed_terms = [_conditional_factor(x, sorted(parents.get(x, set())))]
    elim_text = ",".join(eliminated)
    product_text = " * ".join(retained_terms) if retained_terms else "1"
    if eliminated:
        formula = f"P_{{do({x})}}({y}) = sum_{{{elim_text}}} {product_text}"
    else:
        formula = f"P_{{do({x})}}({y}) = {product_text}"
    return TruncatedFactorizationDiagnostic(
        "valid_observed_dag_truncated_factorization",
        True,
        topological_order="|".join(order),
        eliminated_nodes="|".join(eliminated),
        removed_factors="|".join(removed_terms),
        retained_factors="|".join(retained_terms),
        formula=formula,
        reason_codes="OBSERVED_DAG_TRUNCATED_FACTORIZATION_VALID",
    )


def _json_loads_or_empty(text: str) -> object:
    raw = _s(text)
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {"raw": raw, "parse_status": "unparsed"}


def _ast_json_for_payload(payload: Mapping[str, object]) -> str:
    """Return deterministic AST JSON for a symbolic payload.

    AST generation is audit metadata only: if conversion ever fails, keep the
    legacy symbolic payload intact and expose an empty AST field.
    """
    try:
        return ast_to_json(payload_to_ast(payload))
    except Exception:
        return ""


def symbolic_formula_diagnostic(
    *,
    treatment: str,
    outcome: str,
    identifiable: bool,
    id_strategy: str,
    estimand_formula: str,
    adjustment_set: Sequence[str] = (),
    mediators: Sequence[str] = (),
    factorization: Optional[TruncatedFactorizationDiagnostic] = None,
    cfactor: Optional[Any] = None,
    recursive: Optional[Any] = None,
) -> SymbolicFormulaDiagnostic:
    """Build a conservative machine-readable formula diagnostic.

    This is not a CAS and it does not implement algebraic simplification.  It
    records the estimand type, summation variables, product terms, and unresolved
    Q-factors using deterministic JSON.
    """
    x = _s(treatment)
    y = _s(outcome)
    strategy = _s(id_strategy)

    if not x or not y:
        return SymbolicFormulaDiagnostic(
            "not_available_invalid_query",
            symbolic_reason_codes="MISSING_QUERY_NODE",
        )

    if strategy == "no_directed_effect":
        payload = {
            "type": "graphical_zero_effect",
            "estimand": {"outcome": y, "intervention": x},
            "sum_over": [],
            "product_terms": [f"P({y})"],
            "unresolved_terms": [],
        }
        return SymbolicFormulaDiagnostic(
            "identified_symbolic_formula",
            "graphical_zero_effect",
            _json_formula(payload),
            estimand_formula,
            symbolic_product_terms=f"P({y})",
            symbolic_reason_codes="NO_DIRECTED_PATH_ZERO_EFFECT",
            symbolic_formula_ast_json=_ast_json_for_payload(payload),
        )

    if strategy == "observed_dag_truncated_factorization" and factorization and factorization.factorization_ok:
        sum_over = parse_field_list(factorization.eliminated_nodes)
        product_terms = parse_formula_term_list(factorization.retained_factors)
        removed_terms = parse_formula_term_list(factorization.removed_factors)
        payload = {
            "type": "truncated_factorization",
            "estimand": {"outcome": y, "intervention": x},
            "sum_over": sum_over,
            "product_terms": product_terms,
            "removed_intervention_factors": removed_terms,
            "topological_order": parse_field_list(factorization.topological_order),
            "unresolved_terms": [],
        }
        return SymbolicFormulaDiagnostic(
            "identified_symbolic_formula",
            "truncated_factorization",
            _json_formula(payload),
            factorization.formula or estimand_formula,
            symbolic_sum_over="|".join(sum_over),
            symbolic_product_terms="|".join(product_terms),
            symbolic_removed_terms="|".join(removed_terms),
            symbolic_reason_codes=factorization.reason_codes,
            symbolic_formula_ast_json=_ast_json_for_payload(payload),
        )

    if strategy == "backdoor_adjustment" and identifiable:
        z = _dedupe(adjustment_set)
        z_joint = _joint_symbol(z)
        product_terms = [f"P({y} | {x}, {z_joint})" if z_joint else f"P({y} | {x})"]
        if z_joint:
            product_terms.append(f"P({z_joint})")
        payload = {
            "type": "backdoor_adjustment",
            "estimand": {"outcome": y, "intervention": x},
            "sum_over": z,
            "product_terms": product_terms,
            "adjustment_set": z,
            "unresolved_terms": [],
        }
        return SymbolicFormulaDiagnostic(
            "identified_symbolic_formula",
            "backdoor_adjustment",
            _json_formula(payload),
            estimand_formula,
            symbolic_sum_over="|".join(z),
            symbolic_product_terms="|".join(product_terms),
            symbolic_reason_codes="BACKDOOR_ADJUSTMENT_SET_DSEP_VERIFIED",
            symbolic_formula_ast_json=_ast_json_for_payload(payload),
        )

    if strategy == "frontdoor_limited" and identifiable:
        meds = _dedupe(mediators)
        z_joint = _joint_symbol(meds)
        product_terms = [
            f"P({z_joint} | {x})" if z_joint else f"P(mediator | {x})",
            f"P({y} | x_prime, {z_joint})" if z_joint else f"P({y} | x_prime, mediator)",
            "P(x_prime)",
        ]
        payload = {
            "type": "frontdoor_limited",
            "estimand": {"outcome": y, "intervention": x},
            "sum_over": meds + ["x_prime"],
            "product_terms": product_terms,
            "mediators": meds,
            "unresolved_terms": [],
            "scope": "limited_frontdoor_dsep_verified",
        }
        return SymbolicFormulaDiagnostic(
            "identified_symbolic_formula",
            "frontdoor_limited",
            _json_formula(payload),
            estimand_formula,
            symbolic_sum_over="|".join(meds + ["x_prime"]),
            symbolic_product_terms="|".join(product_terms),
            symbolic_reason_codes="LIMITED_FRONTDOOR_DSEP_CRITERIA_PASSED",
            symbolic_formula_ast_json=_ast_json_for_payload(payload),
        )

    if strategy.startswith("full_recursive_id_step") and identifiable and recursive and recursive.recursive_expression_json:
        payload = _json_loads_or_empty(recursive.recursive_expression_json)
        if isinstance(payload, dict):
            payload = dict(payload)
            payload.setdefault("type", "recursive_id_expression")
            payload.setdefault("estimand", {"outcome": y, "intervention": x})
        else:
            payload = {"type": "recursive_id_expression", "raw": payload}
        sum_over = payload.get("sum_over", []) if isinstance(payload, dict) else []
        product_terms = payload.get("product_terms", []) if isinstance(payload, dict) else []
        return SymbolicFormulaDiagnostic(
            "identified_symbolic_formula",
            "recursive_id_expression",
            _json_formula(payload if isinstance(payload, Mapping) else {"payload": payload}),
            estimand_formula,
            symbolic_sum_over="|".join(_dedupe(sum_over if isinstance(sum_over, list) else [])),
            symbolic_product_terms="|".join(str(t) for t in product_terms) if isinstance(product_terms, list) else "",
            symbolic_reason_codes=recursive.reason_codes or "FULL_RECURSIVE_ID_STEP2_IDENTIFIED",
            symbolic_formula_ast_json=_ast_json_for_payload(payload if isinstance(payload, Mapping) else {"payload": payload}),
        )

    if cfactor and cfactor.c_factor_formula:
        unresolved = parse_field_list(cfactor.c_factor_unresolved_districts)
        payload = {
            "type": "c_factor_product_placeholder",
            "estimand": {"outcome": y, "intervention": x},
            "sum_over": parse_field_list(cfactor.c_factor_sum_over),
            "observed_terms": parse_formula_term_list(cfactor.c_factor_observed_terms),
            "latent_terms": parse_formula_term_list(cfactor.c_factor_latent_terms),
            "unresolved_terms": unresolved,
            "identified": bool(cfactor.c_factor_product_ok),
        }
        status = "identified_symbolic_formula" if cfactor.c_factor_product_ok and identifiable else "blocked_symbolic_formula_unresolved_c_factor"
        return SymbolicFormulaDiagnostic(
            status,
            "c_factor_product_placeholder",
            _json_formula(payload),
            cfactor.c_factor_formula,
            symbolic_sum_over=cfactor.c_factor_sum_over,
            symbolic_product_terms="|".join(parse_formula_term_list(cfactor.c_factor_observed_terms) + parse_formula_term_list(cfactor.c_factor_latent_terms)),
            symbolic_unresolved_terms=cfactor.c_factor_unresolved_districts,
            symbolic_reason_codes=cfactor.c_factor_reason_codes,
            symbolic_formula_ast_json=_ast_json_for_payload(payload),
        )

    if recursive and recursive.formula and recursive.recursive_identified:
        payload = {
            "type": "recursive_base_case",
            "estimand": {"outcome": y, "intervention": x},
            "formula": recursive.formula,
            "recursive_status": recursive.recursive_status,
            "unresolved_terms": [],
        }
        return SymbolicFormulaDiagnostic(
            "identified_symbolic_formula",
            "recursive_base_case",
            _json_formula(payload),
            recursive.formula,
            symbolic_reason_codes=recursive.reason_codes,
            symbolic_formula_ast_json=_ast_json_for_payload(payload),
        )

    return SymbolicFormulaDiagnostic(
        "not_available_for_blocked_query",
        symbolic_formula_kind="blocked_or_unsupported",
        symbolic_formula_latex=estimand_formula,
        symbolic_reason_codes="QUERY_NOT_IDENTIFIED_OR_UNSUPPORTED",
    )


__all__ = [
    "SymbolicFormulaDiagnostic",
    "TruncatedFactorizationDiagnostic",
    "_json_loads_or_empty",
    "truncated_factorization_diagnostic",
    "symbolic_formula_diagnostic",
]
