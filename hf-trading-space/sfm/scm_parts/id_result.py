from __future__ import annotations

"""Result objects and payload assembly for conservative SCM identification.

This module intentionally contains no graph-routing decisions. It only turns
diagnostics produced by the ID core into the public audit-safe ``IDResult`` /
``IDSetResult`` contract.
"""

from dataclasses import asdict, dataclass
from typing import Dict, Optional, Sequence

from .do_calculus import DoCalculusDiagnostic
from .hedge import FormalHedgeDiagnostic
from .id_authority import IDAuthorityDiagnostic, id_authority_diagnostic
from .id_contract import IDContractDiagnostic, id_contract_diagnostic
from .id_decomposition import CComponentDiagnostic, CFactorDecompositionDiagnostic
from .id_formula import (
    SymbolicFormulaDiagnostic,
    TruncatedFactorizationDiagnostic,
    symbolic_formula_diagnostic,
)
from .id_proof import IDProofDiagnostic, id_proof_diagnostic
from .id_routes import BackdoorDiagnostic, FrontdoorDiagnostic, HedgeDiagnostic

@dataclass(frozen=True)
class IDResult:
    treatment: str
    outcome: str
    identifiable: bool
    id_strategy: str
    id_algorithm_level: str
    estimand_formula: str
    adjustment_set: str = ""
    mediators: str = ""
    c_components: str = ""
    hedge_status: str = "not_checked_full_id_not_implemented"
    hedge_witness: str = ""
    ancestral_c_components: str = ""
    treatment_ancestral_district: str = ""
    outcome_ancestral_district: str = ""
    possible_hedge: int = 0
    formal_hedge_status: str = "not_checked"
    formal_hedge_certified: int = 0
    hedge_F: str = ""
    hedge_F_prime: str = ""
    hedge_roots_F: str = ""
    hedge_roots_F_prime: str = ""
    hedge_treatment_in_F_minus_F_prime: str = ""
    hedge_outcome_witness: str = ""
    hedge_graph_without_treatment_nodes: str = ""
    hedge_checks_json: str = ""
    hedge_certificate_json: str = ""
    hedge_reason_codes: str = ""
    directed_acyclic: int = 1
    directed_cycle_nodes: str = ""
    backdoor_status: str = "not_checked"
    backdoor_open_paths: str = ""
    backdoor_descendant_controls: str = ""
    frontdoor_status: str = "not_checked"
    frontdoor_active_mediators: str = ""
    frontdoor_unmediated_paths: str = ""
    frontdoor_x_to_mediator_open_paths: str = ""
    frontdoor_mediator_to_y_open_paths: str = ""
    frontdoor_witness_paths: str = ""
    factorization_status: str = "not_checked"
    factorization_topological_order: str = ""
    factorization_eliminated_nodes: str = ""
    factorization_removed_factors: str = ""
    factorization_retained_factors: str = ""
    district_status: str = "not_checked"
    district_requires_recursive_id: int = 0
    districts_all: str = ""
    ancestral_nodes: str = ""
    ancestral_districts: str = ""
    district_treatment: str = ""
    district_outcome: str = ""
    nontrivial_ancestral_districts: str = ""
    district_factor_placeholders: str = ""
    recursive_status: str = "not_checked"
    recursive_identified: int = 0
    recursive_depth: int = 0
    recursive_ancestor_reduction_applied: int = 0
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
    c_factor_status: str = "not_checked"
    c_factor_product_ok: int = 0
    c_factor_formula: str = ""
    c_factor_sum_over: str = ""
    c_factor_observed_terms: str = ""
    c_factor_latent_terms: str = ""
    c_factor_unresolved_districts: str = ""
    symbolic_formula_status: str = "not_checked"
    symbolic_formula_kind: str = ""
    symbolic_formula_json: str = ""
    symbolic_formula_latex: str = ""
    symbolic_sum_over: str = ""
    symbolic_product_terms: str = ""
    symbolic_removed_terms: str = ""
    symbolic_unresolved_terms: str = ""
    symbolic_reason_codes: str = ""
    symbolic_formula_ast_json: str = ""
    do_calculus_status: str = "not_checked"
    do_calculus_rule1_applicable: int = 0
    do_calculus_rule2_applicable: int = 0
    do_calculus_rule3_applicable: int = 0
    do_calculus_applicable_rules: str = ""
    do_calculus_rule_trace_json: str = ""
    do_calculus_rule1_reason_codes: str = ""
    do_calculus_rule2_reason_codes: str = ""
    do_calculus_rule3_reason_codes: str = ""
    do_calculus_reason_codes: str = ""
    id_proof_status: str = "not_checked"
    id_proof_steps_json: str = ""
    formula_tree_json: str = ""
    proof_blocker: str = ""
    proof_reason_codes: str = ""
    id_contract_status: str = "not_checked"
    id_contract_ok: int = 0
    identification_certificate_json: str = ""
    nonidentification_certificate_json: str = ""
    id_contract_reason_codes: str = ""
    authority_status: str = "not_checked"
    identification_authority: str = ""
    authority_basis: str = ""
    authority_reason_codes: str = ""
    diagnostic_roles_json: str = ""
    failure_reason: str = ""
    reason_codes: str = ""

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class IDSetResult:
    """Public set-valued recursive-ID result.

    This is intentionally narrower than ``IDResult`` because backdoor/frontdoor
    wrappers are single-edge APIs.  It exposes the recursive expression branch
    for full-ID style ``P(Y_set | do(X_set))`` queries.
    """

    treatments: str
    outcomes: str
    identifiable: bool
    id_strategy: str
    id_algorithm_level: str
    estimand_formula: str = ""
    recursive_status: str = "not_checked"
    recursive_expression_json: str = ""
    recursive_trace_json: str = ""
    recursive_formula_source: str = ""
    blocker: str = ""
    blocker_class: str = ""
    pending_operator: str = ""
    reason_codes: str = ""

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _with_hedge_fields(base: Dict[str, object], hedge: HedgeDiagnostic, cycles: Sequence[str], formal_hedge: Optional[FormalHedgeDiagnostic] = None) -> Dict[str, object]:
    base.update(
        {
            "hedge_status": hedge.hedge_status,
            "hedge_witness": hedge.hedge_witness,
            "ancestral_c_components": hedge.ancestral_c_components,
            "treatment_ancestral_district": hedge.treatment_ancestral_district,
            "outcome_ancestral_district": hedge.outcome_ancestral_district,
            "possible_hedge": int(bool(hedge.possible_hedge)),
            "directed_acyclic": int(len(cycles) == 0),
            "directed_cycle_nodes": "|".join(cycles),
        }
    )
    if formal_hedge is not None:
        base.update({
            "formal_hedge_status": formal_hedge.formal_hedge_status,
            "formal_hedge_certified": int(bool(formal_hedge.formal_hedge_certified)),
            "hedge_F": formal_hedge.hedge_F,
            "hedge_F_prime": formal_hedge.hedge_F_prime,
            "hedge_roots_F": formal_hedge.hedge_roots_F,
            "hedge_roots_F_prime": formal_hedge.hedge_roots_F_prime,
            "hedge_treatment_in_F_minus_F_prime": formal_hedge.hedge_treatment_in_F_minus_F_prime,
            "hedge_outcome_witness": formal_hedge.hedge_outcome_witness,
            "hedge_graph_without_treatment_nodes": formal_hedge.hedge_graph_without_treatment_nodes,
            "hedge_checks_json": formal_hedge.hedge_checks_json,
            "hedge_certificate_json": formal_hedge.hedge_certificate_json,
            "hedge_reason_codes": formal_hedge.hedge_reason_codes,
        })
    return base

def _with_backdoor_fields(base: Dict[str, object], backdoor: Optional[BackdoorDiagnostic]) -> Dict[str, object]:
    if backdoor is None:
        return base
    base.update(
        {
            "backdoor_status": backdoor.backdoor_status,
            "backdoor_open_paths": backdoor.open_paths,
            "backdoor_descendant_controls": backdoor.descendant_controls,
        }
    )
    return base


def _with_frontdoor_fields(base: Dict[str, object], frontdoor: Optional[FrontdoorDiagnostic]) -> Dict[str, object]:
    if frontdoor is None:
        return base
    base.update(
        {
            "frontdoor_status": frontdoor.frontdoor_status,
            "frontdoor_active_mediators": frontdoor.active_mediators,
            "frontdoor_unmediated_paths": frontdoor.unmediated_directed_paths,
            "frontdoor_x_to_mediator_open_paths": frontdoor.x_to_mediator_open_paths,
            "frontdoor_mediator_to_y_open_paths": frontdoor.mediator_to_y_open_paths,
            "frontdoor_witness_paths": frontdoor.witness_paths,
        }
    )
    return base


def _with_factorization_fields(
    base: Dict[str, object],
    factorization: Optional[TruncatedFactorizationDiagnostic],
) -> Dict[str, object]:
    if factorization is None:
        return base
    base.update(
        {
            "factorization_status": factorization.factorization_status,
            "factorization_topological_order": factorization.topological_order,
            "factorization_eliminated_nodes": factorization.eliminated_nodes,
            "factorization_removed_factors": factorization.removed_factors,
            "factorization_retained_factors": factorization.retained_factors,
        }
    )
    return base



def _with_c_component_fields(base: Dict[str, object], cdiag: Optional[CComponentDiagnostic]) -> Dict[str, object]:
    if cdiag is None:
        return base
    base.update(
        {
            "district_status": cdiag.district_status,
            "district_requires_recursive_id": int(bool(cdiag.requires_recursive_id)),
            "districts_all": cdiag.districts,
            "ancestral_nodes": cdiag.ancestral_nodes,
            "ancestral_districts": cdiag.ancestral_districts,
            "district_treatment": cdiag.treatment_district,
            "district_outcome": cdiag.outcome_district,
            "nontrivial_ancestral_districts": cdiag.nontrivial_districts,
            "district_factor_placeholders": cdiag.district_factor_placeholders,
        }
    )
    return base




def _with_c_factor_fields(base: Dict[str, object], cfactor: Optional[CFactorDecompositionDiagnostic]) -> Dict[str, object]:
    if cfactor is None:
        return base
    base.update({
        "c_factor_status": cfactor.c_factor_status,
        "c_factor_product_ok": int(bool(cfactor.c_factor_product_ok)),
        "c_factor_formula": cfactor.c_factor_formula,
        "c_factor_sum_over": cfactor.c_factor_sum_over,
        "c_factor_observed_terms": cfactor.c_factor_observed_terms,
        "c_factor_latent_terms": cfactor.c_factor_latent_terms,
        "c_factor_unresolved_districts": cfactor.c_factor_unresolved_districts,
    })
    return base


def _with_symbolic_fields(base: Dict[str, object], symbolic: Optional[SymbolicFormulaDiagnostic]) -> Dict[str, object]:
    if symbolic is None:
        return base
    base.update(
        {
            "symbolic_formula_status": symbolic.symbolic_formula_status,
            "symbolic_formula_kind": symbolic.symbolic_formula_kind,
            "symbolic_formula_json": symbolic.symbolic_formula_json,
            "symbolic_formula_latex": symbolic.symbolic_formula_latex,
            "symbolic_sum_over": symbolic.symbolic_sum_over,
            "symbolic_product_terms": symbolic.symbolic_product_terms,
            "symbolic_removed_terms": symbolic.symbolic_removed_terms,
            "symbolic_unresolved_terms": symbolic.symbolic_unresolved_terms,
            "symbolic_reason_codes": symbolic.symbolic_reason_codes,
            "symbolic_formula_ast_json": symbolic.symbolic_formula_ast_json,
        }
    )
    return base




def _with_do_calculus_fields(base: Dict[str, object], do_calc: Optional[DoCalculusDiagnostic]) -> Dict[str, object]:
    if do_calc is None:
        return base
    base.update({
        "do_calculus_status": do_calc.do_calculus_status,
        "do_calculus_rule1_applicable": int(do_calc.rule1_applicable),
        "do_calculus_rule2_applicable": int(do_calc.rule2_applicable),
        "do_calculus_rule3_applicable": int(do_calc.rule3_applicable),
        "do_calculus_applicable_rules": do_calc.applicable_rules,
        "do_calculus_rule_trace_json": do_calc.rule_trace_json,
        "do_calculus_rule1_reason_codes": do_calc.rule1_reason_codes,
        "do_calculus_rule2_reason_codes": do_calc.rule2_reason_codes,
        "do_calculus_rule3_reason_codes": do_calc.rule3_reason_codes,
        "do_calculus_reason_codes": do_calc.reason_codes,
    })
    return base


def _with_proof_fields(base: Dict[str, object], proof: Optional[IDProofDiagnostic]) -> Dict[str, object]:
    if proof is None:
        return base
    base.update(
        {
            "id_proof_status": proof.id_proof_status,
            "id_proof_steps_json": proof.id_proof_steps_json,
            "formula_tree_json": proof.formula_tree_json,
            "proof_blocker": proof.proof_blocker,
            "proof_reason_codes": proof.proof_reason_codes,
        }
    )
    return base


def _with_authority_fields(base: Dict[str, object], authority: Optional[IDAuthorityDiagnostic]) -> Dict[str, object]:
    if authority is None:
        return base
    base.update({
        "authority_status": authority.authority_status,
        "identification_authority": authority.identification_authority,
        "authority_basis": authority.authority_basis,
        "authority_reason_codes": authority.authority_reason_codes,
        "diagnostic_roles_json": authority.diagnostic_roles_json,
    })
    return base


def _with_id_contract_fields(base: Dict[str, object], contract: Optional[IDContractDiagnostic]) -> Dict[str, object]:
    if contract is None:
        return base
    base.update({
        "id_contract_status": contract.id_contract_status,
        "id_contract_ok": int(bool(contract.id_contract_ok)),
        "identification_certificate_json": contract.identification_certificate_json,
        "nonidentification_certificate_json": contract.nonidentification_certificate_json,
        "id_contract_reason_codes": contract.id_contract_reason_codes,
    })
    return base


def _with_recursive_fields(base: Dict[str, object], recursive: Optional[RecursiveIDDiagnostic]) -> Dict[str, object]:
    if recursive is None:
        return base
    base.update(
        {
            "recursive_status": recursive.recursive_status,
            "recursive_identified": int(bool(recursive.recursive_identified)),
            "recursive_depth": int(recursive.depth),
            "recursive_ancestor_reduction_applied": int(bool(recursive.ancestor_reduction_applied)),
            "recursive_ancestral_nodes": recursive.recursive_ancestral_nodes,
            "recursive_removed_non_ancestors": recursive.recursive_removed_non_ancestors,
            "recursive_districts": recursive.recursive_districts,
            "recursive_q_factors": recursive.recursive_q_factors,
            "recursive_c_factor_formula": recursive.recursive_c_factor_formula,
            "recursive_c_factor_unresolved_districts": recursive.recursive_c_factor_unresolved_districts,
            "recursive_subqueries": recursive.recursive_subqueries,
            "recursive_blocker": recursive.recursive_blocker,
            "recursive_subproblem_plan_json": recursive.recursive_subproblem_plan_json,
            "recursive_blocker_class": recursive.recursive_blocker_class,
            "recursive_pending_operator": recursive.recursive_pending_operator,
            "recursive_reduction_chain_json": recursive.recursive_reduction_chain_json,
            "recursive_subproblem_count": int(recursive.recursive_subproblem_count),
            "recursive_expression_json": recursive.recursive_expression_json,
            "recursive_trace_json": recursive.recursive_trace_json,
            "recursive_formula_source": recursive.recursive_formula_source,
            "q_factor_id_status": recursive.q_factor_id_status,
            "q_factor_id_json": recursive.q_factor_id_json,
            "q_factor_id_formula": recursive.q_factor_id_formula,
            "q_factor_id_reason_codes": recursive.q_factor_id_reason_codes,
        }
    )
    if not base.get("reason_codes") and recursive.reason_codes:
        base["reason_codes"] = recursive.reason_codes
    return base


def _nonblocking_formal_hedge_for_identified_result(
    identifiable: bool,
    formal_hedge: Optional[FormalHedgeDiagnostic],
) -> Optional[FormalHedgeDiagnostic]:
    """Never export a formal non-identification certificate on an identified result.

    A formal hedge is an impossibility certificate.  If a positive ID route has
    already produced a formula/proof, carrying ``formal_hedge_certified=1`` in
    the public row is contradictory and unsafe.  Keep a non-certifying audit
    status instead so downstream contracts cannot treat it as a hedge proof.
    """
    if identifiable and formal_hedge is not None and formal_hedge.formal_hedge_certified:
        return FormalHedgeDiagnostic(
            "not_blocking_identified_result",
            False,
            hedge_reason_codes="FORMAL_HEDGE_SUPPRESSED_FOR_IDENTIFIED_RESULT",
        )
    return formal_hedge


def _result(
    treatment: str,
    outcome: str,
    identifiable: bool,
    id_strategy: str,
    id_algorithm_level: str,
    estimand_formula: str,
    *,
    adjustment_set: Sequence[str],
    mediators: Sequence[str],
    c_components: str,
    hedge: HedgeDiagnostic,
    cycles: Sequence[str],
    backdoor: Optional[BackdoorDiagnostic] = None,
    frontdoor: Optional[FrontdoorDiagnostic] = None,
    factorization: Optional[TruncatedFactorizationDiagnostic] = None,
    cdiag: Optional[CComponentDiagnostic] = None,
    cfactor: Optional[CFactorDecompositionDiagnostic] = None,
    recursive: Optional[RecursiveIDDiagnostic] = None,
    symbolic: Optional[SymbolicFormulaDiagnostic] = None,
    do_calc: Optional[DoCalculusDiagnostic] = None,
    formal_hedge: Optional[FormalHedgeDiagnostic] = None,
    failure_reason: str = "",
    reason_codes: str = "",
) -> IDResult:
    formal_hedge = _nonblocking_formal_hedge_for_identified_result(identifiable, formal_hedge)

    payload = _with_hedge_fields(
        {
            "treatment": treatment,
            "outcome": outcome,
            "identifiable": bool(identifiable),
            "id_strategy": id_strategy,
            "id_algorithm_level": id_algorithm_level,
            "estimand_formula": estimand_formula,
            "adjustment_set": "|".join(adjustment_set),
            "mediators": "|".join(mediators),
            "c_components": c_components,
            "failure_reason": failure_reason,
            "reason_codes": reason_codes,
        },
        hedge,
        cycles,
        formal_hedge,
    )
    payload = _with_backdoor_fields(payload, backdoor)
    payload = _with_frontdoor_fields(payload, frontdoor)
    payload = _with_factorization_fields(payload, factorization)
    payload = _with_c_component_fields(payload, cdiag)
    payload = _with_c_factor_fields(payload, cfactor)
    payload = _with_recursive_fields(payload, recursive)
    if symbolic is None:
        symbolic = symbolic_formula_diagnostic(
            treatment=treatment,
            outcome=outcome,
            identifiable=identifiable,
            id_strategy=id_strategy,
            estimand_formula=estimand_formula,
            adjustment_set=adjustment_set,
            mediators=mediators,
            factorization=factorization,
            cfactor=cfactor,
            recursive=recursive,
        )
    proof = id_proof_diagnostic(
        treatment=treatment,
        outcome=outcome,
        identifiable=identifiable,
        id_strategy=id_strategy,
        id_algorithm_level=id_algorithm_level,
        estimand_formula=estimand_formula,
        hedge=hedge,
        cycles=cycles,
        backdoor=backdoor,
        frontdoor=frontdoor,
        factorization=factorization,
        cdiag=cdiag,
        cfactor=cfactor,
        recursive=recursive,
        symbolic=symbolic,
        failure_reason=failure_reason,
        reason_codes=reason_codes,
    )
    authority = id_authority_diagnostic(
        identifiable=identifiable,
        id_strategy=id_strategy,
        id_algorithm_level=id_algorithm_level,
        recursive=recursive,
        backdoor=backdoor,
        frontdoor=frontdoor,
        factorization=factorization,
        hedge=hedge,
        formal_hedge=formal_hedge,
        reason_codes=reason_codes,
        failure_reason=failure_reason,
    )
    contract = id_contract_diagnostic(
        treatment=treatment,
        outcome=outcome,
        identifiable=identifiable,
        id_strategy=id_strategy,
        id_algorithm_level=id_algorithm_level,
        estimand_formula=estimand_formula,
        id_proof_status=proof.id_proof_status,
        id_proof_steps_json=proof.id_proof_steps_json,
        formula_tree_json=proof.formula_tree_json,
        formal_hedge=formal_hedge,
        failure_reason=failure_reason,
        reason_codes=reason_codes,
        authority_status=authority.authority_status,
        identification_authority=authority.identification_authority,
        authority_basis=authority.authority_basis,
    )
    payload = _with_symbolic_fields(payload, symbolic)
    payload = _with_do_calculus_fields(payload, do_calc)
    payload = _with_proof_fields(payload, proof)
    payload = _with_id_contract_fields(payload, contract)
    payload = _with_authority_fields(payload, authority)
    return IDResult(**payload)


__all__ = [
    "IDResult",
    "IDSetResult",
    "_result",
]
