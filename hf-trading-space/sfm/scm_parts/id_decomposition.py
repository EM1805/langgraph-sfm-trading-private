from __future__ import annotations

"""C-component, c-factor, and recursive subproblem decomposition for SCM ID.

This module is intentionally diagnostic/conservative. It exposes the graph
reductions and unresolved Q-factor subproblems that the core ID engine uses,
without upgrading unresolved latent-variable cases into authorized do-claims.
"""

from dataclasses import asdict, dataclass
from typing import Dict, List

from .admg import ADMG
from .graph_criteria import (
    directed_cycle_nodes,
    directed_path_exists,
    same_district,
    topological_order,
)
from .id_algorithm_common import (
    _conditional_factor,
    _dedupe,
    _format_component,
    _format_components,
    _json_formula,
    _s,
)
from .id_proof import _proof_step


@dataclass(frozen=True)
class CComponentDiagnostic:
    """C-component / district decomposition diagnostic for ADMG ID routing.

    This is not the recursive ID algorithm.  It exposes the decomposition facts
    that a recursive ID implementation will need and prevents the limited
    scaffold from silently treating latent districts as observed DAG cases.
    """

    district_status: str
    requires_recursive_id: bool
    districts: str = ""
    ancestral_nodes: str = ""
    ancestral_districts: str = ""
    treatment_district: str = ""
    outcome_district: str = ""
    nontrivial_districts: str = ""
    district_factor_placeholders: str = ""
    reason_codes: str = ""

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CFactorDecompositionDiagnostic:
    """Conservative c-factor product decomposition for recursive-ID audit."""
    c_factor_status: str
    c_factor_product_ok: bool = False
    c_factor_formula: str = ""
    c_factor_sum_over: str = ""
    c_factor_observed_terms: str = ""
    c_factor_latent_terms: str = ""
    c_factor_unresolved_districts: str = ""
    c_factor_districts: str = ""
    c_factor_reason_codes: str = ""
    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def c_factor_decomposition_diagnostic(admg: ADMG, treatment: str, outcome: str) -> CFactorDecompositionDiagnostic:
    """Emit sum_{An(Y) minus {X,Y}} prod_D Q[D] with unresolved Q placeholders."""
    x = _s(treatment); y = _s(outcome)
    if not x or not y or x not in admg.node_set or y not in admg.node_set:
        return CFactorDecompositionDiagnostic("invalid_query", False, c_factor_reason_codes="MISSING_QUERY_NODE")
    if directed_cycle_nodes(admg):
        return CFactorDecompositionDiagnostic("blocked_directed_cycle", False, c_factor_reason_codes="DIRECTED_CYCLE_NOT_ADMG_DAG")
    ancestral_nodes = sorted(admg.ancestors([y]))
    working = admg.induced_subgraph(ancestral_nodes)
    parents = working.parents(); districts = working.districts()
    observed_terms: List[str] = []; latent_terms: List[str] = []; unresolved: List[List[str]] = []
    for district in districts:
        if len(district) == 1:
            node = district[0]
            observed_terms.append(_conditional_factor(node, sorted(parents.get(node, set()))))
        else:
            latent_terms.append(f"Q[{','.join(district)}]"); unresolved.append(district)
    sum_over = [n for n in ancestral_nodes if n not in {x, y}]
    product = " * ".join(observed_terms + latent_terms) if (observed_terms or latent_terms) else "1"
    formula = f"P_{{do({x})}}({y}) = " + (f"sum_{{{','.join(sum_over)}}} {product}" if sum_over else product)
    return CFactorDecompositionDiagnostic(
        "decomposed_requires_recursive_c_factor_id" if unresolved else "identified_singleton_c_factor_product",
        not bool(unresolved),
        c_factor_formula=formula,
        c_factor_sum_over="|".join(sum_over),
        c_factor_observed_terms="|".join(observed_terms),
        c_factor_latent_terms="|".join(latent_terms),
        c_factor_unresolved_districts=_format_components(unresolved),
        c_factor_districts=_format_components(districts),
        c_factor_reason_codes="NONTRIVIAL_C_FACTORS_REQUIRE_SYMBOLIC_RECURSIVE_ID" if unresolved else "ALL_C_FACTORS_SINGLETON_OBSERVED",
    )



@dataclass(frozen=True)
class IDSubproblemDiagnostic:
    """Machine-readable recursive-ID subproblem plan.

    This is a planning/audit layer only. It decomposes the current query into
    ancestor-reduction and district/Q-factor subproblems, classifies the blocker,
    and records the next symbolic operator that a full recursive-ID engine would
    need. It never upgrades a blocked query into an authorized causal claim.
    """

    subproblem_status: str
    subproblem_count: int = 0
    subproblem_plan_json: str = ""
    reduction_chain_json: str = ""
    blocker_class: str = ""
    pending_operator: str = ""
    reason_codes: str = ""

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def recursive_subproblem_diagnostic(
    admg: ADMG,
    treatment: str,
    outcome: str,
    *,
    depth: int = 0,
) -> IDSubproblemDiagnostic:
    """Plan the next recursive-ID subproblems without claiming identification."""
    x = _s(treatment)
    y = _s(outcome)
    if not x or not y or x not in admg.node_set or y not in admg.node_set:
        chain = [_proof_step("validate_query", "blocked", treatment=x, outcome=y)]
        return IDSubproblemDiagnostic(
            "invalid_query",
            0,
            subproblem_plan_json=_json_formula({"subproblems": []}),
            reduction_chain_json=_json_formula({"chain": chain}),
            blocker_class="invalid_query",
            pending_operator="validate_query",
            reason_codes="MISSING_QUERY_NODE",
        )

    cycles = directed_cycle_nodes(admg)
    if cycles:
        chain = [
            _proof_step("validate_query", "passed", treatment=x, outcome=y),
            _proof_step("directed_acyclicity", "blocked", directed_cycle_nodes="|".join(cycles)),
        ]
        return IDSubproblemDiagnostic(
            "blocked_directed_cycle",
            0,
            subproblem_plan_json=_json_formula({"subproblems": []}),
            reduction_chain_json=_json_formula({"chain": chain}),
            blocker_class="directed_cycle",
            pending_operator="repair_or_reject_cyclic_graph",
            reason_codes="DIRECTED_CYCLE_NOT_ADMG_DAG",
        )

    ancestors_y = sorted(admg.ancestors([y]))
    removed = sorted(admg.node_set - set(ancestors_y))
    working = admg.induced_subgraph(ancestors_y) if removed else admg
    districts = working.districts()
    nontrivial = [d for d in districts if len(d) > 1]
    same_xy = same_district(working, x, y)
    treatment_latent_district = next((d for d in nontrivial if x in d), [])
    treatment_latent_reaches_outcome = bool(treatment_latent_district and y in working.descendants(treatment_latent_district))
    no_directed_path = not directed_path_exists(admg, x, y)
    observed_dag = not bool(working.bidirected_edges) and bool(topological_order(working))

    subproblems: List[Dict[str, object]] = []
    parents = working.parents()
    for idx, district in enumerate(districts, start=1):
        d = sorted(_dedupe(district))
        contains_x = x in d
        contains_y = y in d
        has_latent = len(d) > 1
        if has_latent and (contains_x and (contains_y or treatment_latent_reaches_outcome)):
            status = "blocked_possible_hedge"
            operator = "construct_hedge_or_fail_id"
        elif has_latent:
            status = "pending_symbolic_c_factor_id"
            operator = "identify_q_factor_for_district"
        else:
            status = "observed_singleton_factor"
            operator = "observed_conditional_factor"
        node = d[0] if len(d) == 1 else ""
        subproblems.append(
            {
                "subproblem_id": f"D{idx}",
                "district": d,
                "contains_treatment": bool(contains_x),
                "contains_outcome": bool(contains_y),
                "has_latent_confounding": bool(has_latent),
                "parents": sorted(parents.get(node, set())) if node else [],
                "status": status,
                "pending_operator": operator,
                "q_factor": f"Q[{','.join(d)}]",
                "query": {"outcome_or_district": d, "intervention": [x]},
            }
        )

    if no_directed_path:
        blocker_class = "none"
        pending_operator = "graphical_zero_effect"
        status = "identified_graphical_zero_effect"
        reason = "NO_DIRECTED_PATH_ZERO_EFFECT"
    elif observed_dag:
        blocker_class = "none"
        pending_operator = "observed_dag_truncated_factorization"
        status = "identified_observed_dag_base_case"
        reason = "OBSERVED_DAG_TRUNCATED_FACTORIZATION_AVAILABLE"
    elif same_xy:
        blocker_class = "possible_hedge"
        pending_operator = "construct_hedge_or_fail_id"
        status = "blocked_possible_hedge_same_recursive_district"
        reason = "RECURSIVE_ID_BLOCKED_POSSIBLE_HEDGE_SAME_DISTRICT"
    elif treatment_latent_reaches_outcome:
        blocker_class = "possible_hedge"
        pending_operator = "construct_hedge_or_fail_id"
        status = "blocked_possible_hedge_treatment_district_reaches_outcome"
        reason = "RECURSIVE_ID_BLOCKED_POSSIBLE_HEDGE_TREATMENT_DISTRICT_REACHES_OUTCOME"
    elif nontrivial:
        blocker_class = "unresolved_c_factor"
        pending_operator = "recursive_q_factor_identification"
        status = "pending_recursive_c_factor_subproblems"
        reason = "RECURSIVE_ID_C_FACTOR_PRODUCT_DECOMPOSED_BUT_SYMBOLIC_ID_PENDING"
    else:
        blocker_class = "unhandled_recursive_case"
        pending_operator = "extend_recursive_id_branch"
        status = "blocked_unhandled_recursive_case"
        reason = "UNHANDLED_RECURSIVE_BRANCH"

    chain = [
        _proof_step("validate_query", "passed", treatment=x, outcome=y),
        _proof_step("directed_acyclicity", "passed"),
        _proof_step(
            "ancestor_reduction",
            "applied" if removed else "not_needed",
            ancestral_nodes="|".join(ancestors_y),
            removed_non_ancestors="|".join(removed),
        ),
        _proof_step(
            "district_decomposition",
            "decomposed",
            districts=_format_components(districts),
            nontrivial_districts=_format_components(nontrivial),
        ),
        _proof_step(
            "subproblem_classification",
            status,
            blocker_class=blocker_class,
            pending_operator=pending_operator,
            reason_codes=reason,
        ),
    ]
    return IDSubproblemDiagnostic(
        status,
        len(subproblems),
        subproblem_plan_json=_json_formula({"subproblems": subproblems}),
        reduction_chain_json=_json_formula({"chain": chain}),
        blocker_class=blocker_class,
        pending_operator=pending_operator,
        reason_codes=reason,
    )


def _recursive_subproblem_fields(admg: ADMG, treatment: str, outcome: str, *, depth: int = 0) -> Dict[str, object]:
    diag = recursive_subproblem_diagnostic(admg, treatment, outcome, depth=depth)
    return {
        "recursive_subproblem_plan_json": diag.subproblem_plan_json,
        "recursive_blocker_class": diag.blocker_class,
        "recursive_pending_operator": diag.pending_operator,
        "recursive_reduction_chain_json": diag.reduction_chain_json,
        "recursive_subproblem_count": int(diag.subproblem_count),
    }


def c_component_diagnostic(admg: ADMG, treatment: str, outcome: str) -> CComponentDiagnostic:
    """Audit c-components/districts relevant to ``P(Y | do(X))``.

    The diagnostic is deliberately conservative:
    - singleton districts in observed DAGs are marked as not requiring recursive ID;
    - nontrivial ancestral districts are marked as requiring recursive ID unless a
      higher-priority limited route (verified backdoor/frontdoor) handles them;
    - it emits ``Q[D]`` placeholders rather than pretending to symbolically
      identify arbitrary latent-variable functionals.
    """
    x = _s(treatment)
    y = _s(outcome)
    if not x or not y or x not in admg.node_set or y not in admg.node_set:
        return CComponentDiagnostic("invalid_query", True, reason_codes="MISSING_QUERY_NODE")

    all_districts = admg.districts()
    ancestral_nodes = sorted(admg.ancestors([y]))
    ancestral = admg.induced_subgraph(ancestral_nodes)
    ancestral_districts = ancestral.districts()
    nontrivial = [d for d in ancestral_districts if len(d) > 1]

    treatment_district = next((d for d in ancestral_districts if x in d), [])
    outcome_district = next((d for d in ancestral_districts if y in d), [])

    district_factors = [f"Q[{','.join(d)}]" for d in ancestral_districts]
    if not admg.bidirected_edges:
        status = "observed_dag_singleton_districts"
        requires = False
        reason = "NO_BIDIRECTED_EDGES_SINGLETON_DISTRICTS"
    elif not nontrivial:
        status = "no_nontrivial_ancestral_districts"
        requires = False
        reason = "NO_NONTRIVIAL_ANCESTRAL_C_COMPONENTS"
    elif treatment_district and outcome_district and set(treatment_district) == set(outcome_district):
        status = "same_ancestral_district_possible_hedge"
        requires = True
        reason = "X_Y_SAME_ANCESTRAL_DISTRICT_REQUIRES_HEDGE_OR_FULL_ID"
    else:
        status = "latent_districts_require_recursive_id_or_limited_route"
        requires = True
        reason = "NONTRIVIAL_ANCESTRAL_C_COMPONENTS_REQUIRE_RECURSIVE_ID"

    return CComponentDiagnostic(
        status,
        requires,
        districts=_format_components(all_districts),
        ancestral_nodes="|".join(ancestral_nodes),
        ancestral_districts=_format_components(ancestral_districts),
        treatment_district=_format_component(treatment_district) if treatment_district else "",
        outcome_district=_format_component(outcome_district) if outcome_district else "",
        nontrivial_districts=_format_components(nontrivial),
        district_factor_placeholders=" * ".join(district_factors),
        reason_codes=reason,
    )





__all__ = [
    "CComponentDiagnostic",
    "CFactorDecompositionDiagnostic",
    "IDSubproblemDiagnostic",
    "c_component_diagnostic",
    "c_factor_decomposition_diagnostic",
    "recursive_subproblem_diagnostic",
    "_recursive_subproblem_fields",
]
