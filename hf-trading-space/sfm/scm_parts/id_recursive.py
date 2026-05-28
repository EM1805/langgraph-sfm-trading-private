from __future__ import annotations

"""Recursive ID routing diagnostics for Amantia SCM.

This module contains the conservative recursive-ID router that used to live in
``scm_parts.id_algorithm``. It is separated from the public ID facade so the
core API can stay small while Pearl/Shpitser-style recursive routing evolves
independently.
"""

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple

from .admg import ADMG
from .graph_criteria import directed_cycle_nodes, directed_path_exists, same_district
from .id_algorithm_common import _dedupe, _format_component, _format_components, _s
from .id_decomposition import c_factor_decomposition_diagnostic, _recursive_subproblem_fields
from .id_formula import _json_loads_or_empty, truncated_factorization_diagnostic
from .id_recursive_expression import (
    RecursiveIDExpressionDiagnostic,
    _recursive_expression_fields,
    recursive_id_expression_diagnostic,
)
from .q_factor import QFactorIDDiagnostic, identify_q_factor


def _subquery_label(treatment: str, district: Sequence[str]) -> str:
    d = ",".join(sorted(_dedupe(district)))
    return f"ID(Y={{{d}}}, X={{{treatment}}})"



def recursive_id_diagnostic(
    admg: ADMG,
    treatment: str,
    outcome: str,
    *,
    depth: int = 0,
    max_depth: int = 6,
) -> RecursiveIDDiagnostic:
    """Run conservative recursive-ID routing plus expression diagnostics.

    This step adds an executable expression layer and a formal hedge-candidate
    branch for the canonical single-district ID failure case.  It still refuses
    arbitrary subdistrict/q-factor recursion until Amantia can carry symbolic
    Q-inputs safely.
    """
    x = _s(treatment)
    y = _s(outcome)
    if depth > max_depth:
        return RecursiveIDDiagnostic(
            "blocked_max_recursive_depth",
            False,
            depth=depth,
            recursive_blocker="MAX_RECURSIVE_DEPTH",
            recursive_blocker_class="max_depth",
            recursive_pending_operator="increase_or_debug_recursive_depth",
            reason_codes="MAX_RECURSIVE_DEPTH",
        )
    if not x or not y or x not in admg.node_set or y not in admg.node_set:
        return RecursiveIDDiagnostic(
            "invalid_query",
            False,
            depth=depth,
            recursive_blocker="MISSING_QUERY_NODE",
            recursive_blocker_class="invalid_query",
            recursive_pending_operator="validate_query",
            reason_codes="MISSING_QUERY_NODE",
        )
    cycles = directed_cycle_nodes(admg)
    if cycles:
        return RecursiveIDDiagnostic(
            "blocked_directed_cycle",
            False,
            depth=depth,
            recursive_blocker="DIRECTED_CYCLE_NOT_ADMG_DAG",
            recursive_blocker_class="directed_cycle",
            recursive_pending_operator="repair_or_reject_cyclic_graph",
            reason_codes="DIRECTED_CYCLE_NOT_ADMG_DAG",
        )

    expr = recursive_id_expression_diagnostic(admg, x, y, max_depth=max_depth)
    expr_fields = _recursive_expression_fields(expr)

    if not directed_path_exists(admg, x, y):
        return RecursiveIDDiagnostic(
            "identified_graphical_zero_effect",
            True,
            formula=f"P({y}) ; no directed path from {x} to {y}",
            depth=depth,
            recursive_ancestral_nodes="|".join(sorted(admg.ancestors([y]))),
            **_recursive_subproblem_fields(admg, x, y, depth=depth),
            **expr_fields,
            reason_codes="NO_DIRECTED_PATH_ZERO_EFFECT",
        )

    ancestors_y = sorted(admg.ancestors([y]))
    removed = sorted(admg.node_set - set(ancestors_y))
    working = admg.induced_subgraph(ancestors_y) if removed else admg
    ancestor_reduction = int(bool(removed))

    tf = truncated_factorization_diagnostic(working, x, y)
    if tf.factorization_ok:
        return RecursiveIDDiagnostic(
            "identified_observed_dag_after_recursive_reduction" if ancestor_reduction else "identified_observed_dag_base_case",
            True,
            formula=tf.formula,
            depth=depth,
            ancestor_reduction_applied=ancestor_reduction,
            recursive_ancestral_nodes="|".join(ancestors_y),
            recursive_removed_non_ancestors="|".join(removed),
            recursive_districts=_format_components(working.districts()),
            recursive_q_factors=" * ".join(f"Q[{','.join(d)}]" for d in working.districts()),
            **_recursive_subproblem_fields(admg, x, y, depth=depth),
            **expr_fields,
            reason_codes="RECURSIVE_ID_OBSERVED_DAG_BASE_CASE",
        )

    districts = working.districts()
    q_factors = [f"Q[{','.join(d)}]" for d in districts]
    cfactor = c_factor_decomposition_diagnostic(working, x, y)
    nontrivial = [d for d in districts if len(d) > 1]
    same_xy = same_district(working, x, y)
    subqueries = [_subquery_label(x, d) for d in districts if y in d or x in d or len(d) > 1]

    if expr.expression_identified:
        qdiag = QFactorIDDiagnostic("not_applicable_expression_route")
        if districts:
            try:
                expr_payload = _json_loads_or_empty(expr.expression_json)
                expr_type = _s(expr_payload.get("type")) if isinstance(expr_payload, Mapping) else ""
                if expr_type in {"q_factor_full_district", "q_input_subdistrict_recursion"}:
                    d0 = districts[0]
                    containing = next((d for d in working.districts() if set(d0).issubset(set(d))), d0)
                    qdiag = identify_q_factor(working, d0, containing_district=containing, outcome_set=[y], intervention_set=[x])
            except Exception:
                qdiag = QFactorIDDiagnostic("q_factor_diagnostic_failed", False, q_factor_reason_codes="Q_FACTOR_DIAGNOSTIC_FAILED")
        subfields = _recursive_subproblem_fields(admg, x, y, depth=depth)
        subfields["recursive_blocker_class"] = "none"
        subfields["recursive_pending_operator"] = "none"
        recursive_step = _recursive_id_step_number(expr)
        _strategy, _level, _hedge_status, default_reason = _recursive_id_step_metadata(expr)
        recursive_status = f"identified_full_recursive_id_step{recursive_step}" if recursive_step else "identified_full_recursive_id_step2"
        recursive_reason = expr.reason_codes or default_reason
        return RecursiveIDDiagnostic(
            recursive_status,
            True,
            formula=expr.formula,
            depth=depth,
            ancestor_reduction_applied=ancestor_reduction,
            recursive_ancestral_nodes="|".join(ancestors_y),
            recursive_removed_non_ancestors="|".join(removed),
            recursive_districts=_format_components(districts),
            recursive_q_factors=" * ".join(q_factors),
            recursive_c_factor_formula=cfactor.c_factor_formula,
            recursive_c_factor_unresolved_districts=cfactor.c_factor_unresolved_districts,
            recursive_subqueries="|".join(subqueries),
            recursive_blocker="",
            q_factor_id_status=qdiag.q_factor_status,
            q_factor_id_json=qdiag.q_factor_json,
            q_factor_id_formula=qdiag.q_factor_formula,
            q_factor_id_reason_codes=qdiag.q_factor_reason_codes,
            **subfields,
            **expr_fields,
            reason_codes=recursive_reason,
        )


    if expr.expression_status in {"blocked_formal_hedge_candidate", "blocked_formal_hedge_certificate"}:
        subfields = _recursive_subproblem_fields(admg, x, y, depth=depth)
        subfields["recursive_blocker_class"] = expr.blocker_class
        subfields["recursive_pending_operator"] = expr.pending_operator
        return RecursiveIDDiagnostic(
            "blocked_formal_hedge_certificate",
            False,
            depth=depth,
            ancestor_reduction_applied=ancestor_reduction,
            recursive_ancestral_nodes="|".join(ancestors_y),
            recursive_removed_non_ancestors="|".join(removed),
            recursive_districts=_format_components(districts),
            recursive_q_factors=" * ".join(q_factors),
            recursive_c_factor_formula=cfactor.c_factor_formula,
            recursive_c_factor_unresolved_districts=cfactor.c_factor_unresolved_districts,
            recursive_subqueries="|".join(subqueries),
            recursive_blocker=expr.blocker,
            **subfields,
            **expr_fields,
            reason_codes=expr.reason_codes,
        )

    if same_xy:
        district = next((d for d in districts if x in d and y in d), [])
        return RecursiveIDDiagnostic(
            "blocked_possible_hedge_same_recursive_district",
            False,
            depth=depth,
            ancestor_reduction_applied=ancestor_reduction,
            recursive_ancestral_nodes="|".join(ancestors_y),
            recursive_removed_non_ancestors="|".join(removed),
            recursive_districts=_format_components(districts),
            recursive_q_factors=" * ".join(q_factors),
            recursive_c_factor_formula=cfactor.c_factor_formula,
            recursive_c_factor_unresolved_districts=cfactor.c_factor_unresolved_districts,
            recursive_subqueries="|".join(subqueries),
            recursive_blocker=_format_component(district),
            **_recursive_subproblem_fields(admg, x, y, depth=depth),
            **expr_fields,
            reason_codes="RECURSIVE_ID_BLOCKED_POSSIBLE_HEDGE_SAME_DISTRICT",
        )

    if nontrivial:
        return RecursiveIDDiagnostic(
            "blocked_requires_symbolic_c_factor_identification",
            False,
            depth=depth,
            ancestor_reduction_applied=ancestor_reduction,
            recursive_ancestral_nodes="|".join(ancestors_y),
            recursive_removed_non_ancestors="|".join(removed),
            recursive_districts=_format_components(districts),
            recursive_q_factors=" * ".join(q_factors),
            recursive_c_factor_formula=cfactor.c_factor_formula,
            recursive_c_factor_unresolved_districts=cfactor.c_factor_unresolved_districts,
            recursive_subqueries="|".join(subqueries),
            recursive_blocker=expr.blocker or _format_components(nontrivial),
            **_recursive_subproblem_fields(admg, x, y, depth=depth),
            **expr_fields,
            reason_codes=expr.reason_codes or "RECURSIVE_ID_C_FACTOR_PRODUCT_DECOMPOSED_BUT_SYMBOLIC_ID_PENDING",
        )

    return RecursiveIDDiagnostic(
        "blocked_unhandled_recursive_case",
        False,
        depth=depth,
        ancestor_reduction_applied=ancestor_reduction,
        recursive_ancestral_nodes="|".join(ancestors_y),
        recursive_removed_non_ancestors="|".join(removed),
        recursive_districts=_format_components(districts),
        recursive_q_factors=" * ".join(q_factors),
        recursive_subqueries="|".join(subqueries),
        recursive_blocker=expr.blocker or "UNHANDLED_RECURSIVE_BRANCH",
        **_recursive_subproblem_fields(admg, x, y, depth=depth),
        **expr_fields,
        reason_codes=expr.reason_codes or "UNHANDLED_RECURSIVE_BRANCH",
    )





@dataclass(frozen=True)
class RecursiveIDDiagnostic:
    """Conservative recursive-ID routing diagnostic.

    This is intentionally an audit skeleton, not a full Shpitser/Pearl symbolic
    ID engine.  It performs safe graph reductions and explains why a query is
    solved by an already-supported base case or blocked for future full-ID work.
    """

    recursive_status: str
    recursive_identified: bool = False
    formula: str = ""
    depth: int = 0
    ancestor_reduction_applied: int = 0
    recursive_ancestral_nodes: str = ""
    recursive_removed_non_ancestors: str = ""
    recursive_districts: str = ""
    recursive_q_factors: str = ""
    recursive_c_factor_formula: str = ""
    recursive_c_factor_unresolved_districts: str = ""
    recursive_subqueries: str = ""
    recursive_blocker: str = ""
    recursive_subproblem_plan_json: str = ""
    recursive_blocker_class: str = ""
    recursive_pending_operator: str = ""
    recursive_reduction_chain_json: str = ""
    recursive_subproblem_count: int = 0
    recursive_expression_json: str = ""
    recursive_trace_json: str = ""
    recursive_formula_source: str = ""
    q_factor_id_status: str = ""
    q_factor_id_json: str = ""
    q_factor_id_formula: str = ""
    q_factor_id_reason_codes: str = ""
    reason_codes: str = ""



def _recursive_id_step_number(expr: Optional[RecursiveIDExpressionDiagnostic]) -> int:
    """Map an expression branch to Amantia's incremental full-ID step label."""
    if expr is None or not expr.expression_identified:
        return 0
    status = _s(expr.expression_status)
    payload = _json_loads_or_empty(expr.expression_json)
    payload_type = _s(payload.get("type")) if isinstance(payload, Mapping) else ""
    text = expr.expression_json or ""
    if "general_q_input_subdistrict_recursion" in text or payload_type == "general_q_input_subdistrict_recursion":
        return 5
    if status == "identified_q_factor_full_district" or payload_type == "q_factor_full_district":
        return 4
    if "q_input_subdistrict_recursion" in text or payload_type == "q_input_subdistrict_recursion":
        return 3
    if status.startswith("identified_"):
        return 2
    return 0


def _recursive_id_step_metadata(expr: Optional[RecursiveIDExpressionDiagnostic]) -> Tuple[str, str, str, str]:
    """Return strategy, level, hedge status, and default reason for an ID branch."""
    step = _recursive_id_step_number(expr)
    if step == 5:
        return (
            "full_recursive_id_step5",
            "recursive_id_step5_general_q_input_recursion",
            "not_blocking_full_recursive_id_step5",
            "FULL_RECURSIVE_ID_STEP5_GENERAL_Q_INPUT_RECURSION_IDENTIFIED",
        )
    if step == 4:
        return (
            "full_recursive_id_step4",
            "recursive_id_step4_full_district_q_factor",
            "not_blocking_full_recursive_id_step4",
            "FULL_RECURSIVE_ID_STEP4_FULL_DISTRICT_Q_FACTOR_IDENTIFIED",
        )
    if step == 3:
        return (
            "full_recursive_id_step3",
            "recursive_id_step3_q_input_subdistrict_expression",
            "not_blocking_full_recursive_id_step3",
            "FULL_RECURSIVE_ID_STEP3_Q_INPUT_IDENTIFIED",
        )
    return (
        "full_recursive_id_step2",
        "recursive_id_step2_expression_and_hedge_diagnostic",
        "not_blocking_full_recursive_id_step2",
        "FULL_RECURSIVE_ID_STEP2_IDENTIFIED",
    )

__all__ = [
    "RecursiveIDDiagnostic",
    "recursive_id_diagnostic",
    "_recursive_id_step_number",
    "_recursive_id_step_metadata",
    "_subquery_label",
]

