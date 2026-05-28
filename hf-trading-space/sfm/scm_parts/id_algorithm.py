from __future__ import annotations

"""Conservative ID-algorithm scaffold for ADMG causal queries.

This module is intentionally *not* a full Pearl/Shpitser ID implementation yet.
It provides an audit-friendly layer above :mod:`scm_parts.admg`:

- represents an ID query ``P(Y | do(X))``;
- detects easy identifiable cases already supported by Amantia;
- verifies limited backdoor and front-door claims with graph diagnostics;
- emits symbolic formulas/statuses for DAG/backdoor/front-door-limited cases;
- refuses to overclaim when latent c-components make the query unsupported;
- emits conservative hedge diagnostics from ancestral c-components.

A conservative recursive-ID skeleton is implemented for audit routing
(ancestor reduction, observed-DAG base case, and district decomposition), while
formal hedge construction and arbitrary c-factor formulas remain future steps. Downstream code should treat ``id_algorithm_level`` and
``hedge_status`` as the source of truth for how strong the result is.
"""

import re
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

from .admg import ADMG, admg_from_scm_graph
from .graph_criteria import (
    DSeparationDiagnostic,
    d_separation_diagnostic,
    directed_cycle_nodes,
    directed_path_exists,
    directed_paths,
    nodes_on_directed_paths,
    same_district,
    topological_order,
)

DirectedEdge = Tuple[str, str]


from .do_calculus import DoCalculusDiagnostic, do_calculus_diagnostic
from .hedge import FormalHedgeDiagnostic, formal_hedge_diagnostic
from .id_algorithm_common import (
    _conditional_factor,
    _dedupe,
    _format_component,
    _format_components,
    _format_path,
    _format_paths,
    _joint_symbol,
    _json_formula,
    _s,
    parse_field_list,
    parse_formula_term_list,
)
from .id_recursive_expression import (
    RecursiveIDExpressionDiagnostic,
    _recursive_expression_fields,
    recursive_id_expression_diagnostic,
    recursive_id_set_expression_diagnostic,
)
from .id_status import id_capability_flags
from .id_formula import (
    SymbolicFormulaDiagnostic,
    TruncatedFactorizationDiagnostic,
    symbolic_formula_diagnostic,
    truncated_factorization_diagnostic,
)
from .id_proof import IDProofDiagnostic, id_proof_diagnostic
from .id_decomposition import (
    CComponentDiagnostic,
    CFactorDecompositionDiagnostic,
    IDSubproblemDiagnostic,
    c_component_diagnostic,
    c_factor_decomposition_diagnostic,
    recursive_subproblem_diagnostic,
    _recursive_subproblem_fields,
)
from .id_routes import (
    BackdoorDiagnostic,
    FrontdoorDiagnostic,
    HedgeDiagnostic,
    _frontdoor_limited_ok,
    backdoor_diagnostic,
    frontdoor_diagnostic,
    hedge_diagnostic,
)
from .id_result import IDResult, IDSetResult, _result
from .id_recursive import (
    RecursiveIDDiagnostic,
    _recursive_id_step_metadata,
    _recursive_id_step_number,
    _subquery_label,
    recursive_id_diagnostic,
)














def identify_effect(
    admg: ADMG,
    treatment: str,
    outcome: str,
    *,
    adjustment_set: Optional[Sequence[str]] = None,
    mediators: Optional[Sequence[str]] = None,
    strategy_hint: str = "",
) -> IDResult:
    """Conservatively identify ``P(outcome | do(treatment))`` where supported.

    Supported positive results:
    - ``no_directed_effect`` when no directed path exists from X to Y;
    - ``observed_dag_truncated_factorization`` when there is no explicit latent
      confounding and the directed graph is acyclic;
    - ``backdoor_adjustment`` when a supplied adjustment set passes the limited
      graphical backdoor diagnostic;
    - ``frontdoor_limited`` when supplied mediators pass directed-path and
      d-separation diagnostics.
    """
    x = _s(treatment)
    y = _s(outcome)
    z = [v for v in _dedupe(adjustment_set or []) if v in admg.node_set and v not in {x, y}]
    meds = [v for v in _dedupe(mediators or []) if v in admg.node_set and v not in {x, y}]
    districts = [_format_component(d) for d in admg.districts()]
    c_components = "|".join(districts)
    cycles = directed_cycle_nodes(admg)
    hedge = hedge_diagnostic(admg, x, y)
    formal_hedge = FormalHedgeDiagnostic(
        "not_run_requires_recursive_id_fail_branch",
        False,
        hedge_reason_codes="FORMAL_HEDGE_ONLY_RUN_FROM_RECURSIVE_ID_FAIL_BRANCH_STEP46",
    )
    backdoor = backdoor_diagnostic(admg, x, y, z) if z else None
    frontdoor = frontdoor_diagnostic(admg, x, y, meds) if meds else None
    factorization = truncated_factorization_diagnostic(admg, x, y)
    cdiag = c_component_diagnostic(admg, x, y)
    cfactor = c_factor_decomposition_diagnostic(admg, x, y)
    recursive = recursive_id_diagnostic(admg, x, y)
    if recursive.recursive_status == "blocked_formal_hedge_certificate" or recursive.recursive_blocker_class == "formal_hedge_certificate":
        _fh = formal_hedge_diagnostic(admg, [x], [y])
        if _fh.formal_hedge_certified:
            formal_hedge = FormalHedgeDiagnostic(
                "formal_hedge_certified_recursive_fail_branch",
                True,
                hedge_F=_fh.hedge_F,
                hedge_F_prime=_fh.hedge_F_prime,
                hedge_roots_F=_fh.hedge_roots_F,
                hedge_roots_F_prime=_fh.hedge_roots_F_prime,
                hedge_treatment_in_F_minus_F_prime=_fh.hedge_treatment_in_F_minus_F_prime,
                hedge_outcome_witness=_fh.hedge_outcome_witness,
                hedge_graph_without_treatment_nodes=_fh.hedge_graph_without_treatment_nodes,
                hedge_checks_json=_fh.hedge_checks_json,
                hedge_certificate_json=_fh.hedge_certificate_json,
                hedge_reason_codes="FORMAL_HEDGE_CERTIFIED_RECURSIVE_FAIL_BRANCH_STEP46",
            )
            hedge = HedgeDiagnostic(
                "formal_hedge_certified_recursive_fail_branch",
                True,
                hedge_witness=f"F={formal_hedge.hedge_F};F_prime={formal_hedge.hedge_F_prime}",
                ancestral_c_components=hedge.ancestral_c_components,
                treatment_ancestral_district=hedge.treatment_ancestral_district,
                outcome_ancestral_district=hedge.outcome_ancestral_district,
                reason_codes=formal_hedge.hedge_reason_codes,
            )
    do_calc = do_calculus_diagnostic(admg, x, y, candidate_z=list(z) + list(meds))

    if not x or not y or x not in admg.node_set or y not in admg.node_set:
        return _result(
            x,
            y,
            False,
            "invalid_query",
            "blocked",
            "",
            adjustment_set=z,
            mediators=meds,
            c_components=c_components,
            hedge=hedge,
            cycles=cycles,
            backdoor=backdoor,
            frontdoor=frontdoor,
            factorization=factorization,
            cdiag=cdiag,
            cfactor=cfactor,
            recursive=recursive,
            do_calc=do_calc,
            formal_hedge=formal_hedge,
            failure_reason="MISSING_QUERY_NODE",
            reason_codes="MISSING_QUERY_NODE",
        )

    if not directed_path_exists(admg, x, y):
        return _result(
            x,
            y,
            True,
            "no_directed_effect",
            "graphical_zero_effect",
            f"P({y}) ; no directed path from {x} to {y}",
            adjustment_set=z,
            mediators=meds,
            c_components=c_components,
            hedge=hedge,
            cycles=cycles,
            backdoor=backdoor,
            frontdoor=frontdoor,
            factorization=factorization,
            cdiag=cdiag,
            cfactor=cfactor,
            recursive=recursive,
            do_calc=do_calc,
            formal_hedge=formal_hedge,
            reason_codes="NO_DIRECTED_PATH_ZERO_EFFECT",
        )

    if cycles:
        return _result(
            x,
            y,
            False,
            "blocked_directed_cycle",
            "blocked_not_a_dag",
            "",
            adjustment_set=z,
            mediators=meds,
            c_components=c_components,
            hedge=hedge,
            cycles=cycles,
            backdoor=backdoor,
            frontdoor=frontdoor,
            factorization=factorization,
            cdiag=cdiag,
            cfactor=cfactor,
            recursive=recursive,
            do_calc=do_calc,
            formal_hedge=formal_hedge,
            failure_reason="DIRECTED_CYCLE_NOT_ADMG_DAG",
            reason_codes="DIRECTED_CYCLE_NOT_ADMG_DAG",
        )

    # Front-door is checked before hedge blocking because front-door can handle
    # explicit X<->Y confounding when the limited mediator criteria pass.
    if frontdoor and frontdoor.frontdoor_ok:
        active_meds = parse_field_list(frontdoor.active_mediators)
        ztxt = ",".join(active_meds)
        formula = f"sum_{{{ztxt}}} P({ztxt} | {x}) sum_{{x'}} P({y} | x', {ztxt}) P(x')"
        return _result(
            x,
            y,
            True,
            "frontdoor_limited",
            "limited_id_supported_frontdoor_dsep_verified",
            formula,
            adjustment_set=z,
            mediators=active_meds,
            c_components=c_components,
            hedge=HedgeDiagnostic(
                "not_blocking_frontdoor_limited",
                False,
                hedge.hedge_witness,
                hedge.ancestral_c_components,
                hedge.treatment_ancestral_district,
                hedge.outcome_ancestral_district,
                hedge.reason_codes,
            ),
            cycles=cycles,
            backdoor=backdoor,
            frontdoor=frontdoor,
            factorization=factorization,
            cdiag=cdiag,
            cfactor=cfactor,
            recursive=recursive,
            do_calc=do_calc,
            formal_hedge=formal_hedge,
            reason_codes="LIMITED_FRONTDOOR_DSEP_CRITERIA_PASSED",
        )

    if recursive.recursive_identified and recursive.recursive_status.startswith("identified_full_recursive_id_step"):
        recursive_formula = recursive.formula
        if recursive_formula and not recursive_formula.startswith("P_{"):
            recursive_formula = f"P_{{do({x})}}({y}) = {recursive_formula}"
        recursive_expr = None
        if recursive.recursive_expression_json:
            # Reuse the expression metadata encoded in the recursive diagnostic.
            recursive_expr = RecursiveIDExpressionDiagnostic(
                recursive.recursive_formula_source,
                True,
                recursive.formula,
                expression_json=recursive.recursive_expression_json,
                trace_json=recursive.recursive_trace_json,
                reason_codes=recursive.reason_codes,
            )
        recursive_strategy, recursive_level, recursive_hedge_status, default_reason = _recursive_id_step_metadata(recursive_expr)
        recursive_reason = recursive.reason_codes or default_reason
        return _result(
            x,
            y,
            True,
            recursive_strategy,
            recursive_level,
            recursive_formula,
            adjustment_set=z,
            mediators=meds,
            c_components=c_components,
            hedge=HedgeDiagnostic(
                recursive_hedge_status,
                False,
                hedge.hedge_witness,
                hedge.ancestral_c_components,
                hedge.treatment_ancestral_district,
                hedge.outcome_ancestral_district,
                hedge.reason_codes,
            ),
            cycles=cycles,
            backdoor=backdoor,
            frontdoor=frontdoor,
            factorization=factorization,
            cdiag=cdiag,
            cfactor=cfactor,
            recursive=recursive,
            do_calc=do_calc,
            formal_hedge=formal_hedge,
            reason_codes=recursive_reason,
        )


    # Step 46/48: possible hedge diagnostics are advisory only.
    # A non-identification verdict must come from the recursive ID FAIL branch
    # below, where a formal hedge certificate can be attached exactly.


    hint = _s(strategy_hint).lower()
    if z and backdoor and backdoor.backdoor_ok and ("backdoor" in hint or not admg.bidirected_edges):
        formula = f"sum_{{{','.join(z)}}} P({y} | {x}, {','.join(z)}) P({','.join(z)})"
        return _result(
            x,
            y,
            True,
            "backdoor_adjustment",
            "limited_id_supported_backdoor_dsep_verified",
            formula,
            adjustment_set=z,
            mediators=meds,
            c_components=c_components,
            hedge=hedge,
            cycles=cycles,
            backdoor=backdoor,
            frontdoor=frontdoor,
            factorization=factorization,
            cdiag=cdiag,
            cfactor=cfactor,
            recursive=recursive,
            do_calc=do_calc,
            formal_hedge=formal_hedge,
            reason_codes="BACKDOOR_ADJUSTMENT_SET_DSEP_VERIFIED",
        )

    if z and backdoor and not backdoor.backdoor_ok and "backdoor" in hint:
        return _result(
            x,
            y,
            False,
            "blocked_invalid_backdoor_adjustment",
            "blocked_backdoor_diagnostic_failed",
            "",
            adjustment_set=z,
            mediators=meds,
            c_components=c_components,
            hedge=hedge,
            cycles=cycles,
            backdoor=backdoor,
            frontdoor=frontdoor,
            factorization=factorization,
            cdiag=cdiag,
            cfactor=cfactor,
            recursive=recursive,
            do_calc=do_calc,
            formal_hedge=formal_hedge,
            failure_reason=backdoor.reason_codes,
            reason_codes=backdoor.reason_codes,
        )

    if meds and frontdoor and not frontdoor.frontdoor_ok and "frontdoor" in hint:
        return _result(
            x,
            y,
            False,
            "blocked_invalid_frontdoor",
            "blocked_frontdoor_diagnostic_failed",
            "",
            adjustment_set=z,
            mediators=meds,
            c_components=c_components,
            hedge=hedge,
            cycles=cycles,
            backdoor=backdoor,
            frontdoor=frontdoor,
            factorization=factorization,
            cdiag=cdiag,
            cfactor=cfactor,
            recursive=recursive,
            do_calc=do_calc,
            formal_hedge=formal_hedge,
            failure_reason=frontdoor.reason_codes,
            reason_codes=frontdoor.reason_codes,
        )

    if factorization.factorization_ok:
        return _result(
            x,
            y,
            True,
            "observed_dag_truncated_factorization",
            "limited_id_supported_observed_dag_truncated_factorization",
            factorization.formula,
            adjustment_set=z,
            mediators=meds,
            c_components=c_components,
            hedge=hedge,
            cycles=cycles,
            backdoor=backdoor,
            frontdoor=frontdoor,
            factorization=factorization,
            cdiag=cdiag,
            cfactor=cfactor,
            recursive=recursive,
            do_calc=do_calc,
            formal_hedge=formal_hedge,
            reason_codes=factorization.reason_codes,
        )

    if recursive.recursive_status in {
        "blocked_formal_hedge_candidate",
        "blocked_formal_hedge_certificate",
        "blocked_possible_hedge_same_recursive_district",
        "blocked_requires_symbolic_c_factor_identification",
        "blocked_unhandled_recursive_case",
    }:
        return _result(
            x,
            y,
            False,
            recursive.recursive_status,
            "recursive_id_skeleton_blocked",
            recursive.formula,
            adjustment_set=z,
            mediators=meds,
            c_components=c_components,
            hedge=hedge,
            cycles=cycles,
            backdoor=backdoor,
            frontdoor=frontdoor,
            factorization=factorization,
            cdiag=cdiag,
            cfactor=cfactor,
            recursive=recursive,
            do_calc=do_calc,
            formal_hedge=formal_hedge,
            failure_reason=recursive.reason_codes,
            reason_codes=recursive.reason_codes,
        )

    reason = "FULL_ID_REQUIRED_EXPLICIT_C_COMPONENTS"
    if same_district(admg, x, y):
        reason = "POSSIBLE_HEDGE_TREATMENT_OUTCOME_SAME_C_COMPONENT"
    elif hedge.hedge_status == "no_ancestral_hedge_witness_found":
        reason = "FULL_ID_REQUIRED_NO_SIMPLE_HEDGE_WITNESS"
    return _result(
        x,
        y,
        False,
        "unsupported_requires_full_id",
        "full_id_not_implemented",
        "",
        adjustment_set=z,
        mediators=meds,
        c_components=c_components,
        hedge=hedge,
        cycles=cycles,
        backdoor=backdoor,
        frontdoor=frontdoor,
        factorization=factorization,
        cdiag=cdiag,
        failure_reason=reason,
        reason_codes=reason,
    )


def identify_effect_set(
    admg: ADMG,
    treatments: Sequence[object],
    outcomes: Sequence[object],
    *,
    max_depth: int = 8,
) -> IDSetResult:
    """Identify a set-valued recursive-ID query where currently supported.

    This wrapper is the public Step-26/27 bridge from single-edge ID toward full
    Shpitser/Pearl ID. It does not run limited backdoor/frontdoor shortcuts; it
    delegates to the audited recursive expression engine and classifies the
    exact branch that produced the formula.
    """
    raw_x = _dedupe(treatments or [])
    raw_y = _dedupe(outcomes or [])
    expr = recursive_id_set_expression_diagnostic(admg, raw_x, raw_y, max_depth=max_depth)
    treatments_text = "|".join(raw_x)
    outcomes_text = "|".join(raw_y)
    if not expr.expression_identified:
        return IDSetResult(
            treatments_text,
            outcomes_text,
            False,
            "blocked_recursive_id_set_query",
            "recursive_id_set_query_blocked",
            recursive_status=expr.expression_status,
            recursive_expression_json=expr.expression_json,
            recursive_trace_json=expr.trace_json,
            recursive_formula_source=expr.expression_status,
            blocker=expr.blocker,
            blocker_class=expr.blocker_class,
            pending_operator=expr.pending_operator,
            reason_codes=expr.reason_codes,
        )
    strategy, level, _hedge_status, default_reason = _recursive_id_step_metadata(expr)
    formula = expr.formula
    if formula and not formula.startswith("P_{"):
        formula = f"P_{{do({','.join(raw_x)})}}({','.join(raw_y)}) = {formula}"
    return IDSetResult(
        treatments_text,
        outcomes_text,
        True,
        strategy,
        level,
        estimand_formula=formula,
        recursive_status=f"identified_{strategy}",
        recursive_expression_json=expr.expression_json,
        recursive_trace_json=expr.trace_json,
        recursive_formula_source=expr.expression_status,
        reason_codes=expr.reason_codes or default_reason,
    )


def identify_effect_from_scm_graph(scm_graph: Mapping[str, object], treatment: str, outcome: str, **kwargs: object) -> IDResult:
    return identify_effect(admg_from_scm_graph(scm_graph), treatment, outcome, **kwargs)


def id_audit_rows_from_scm_graph(scm_graph: Mapping[str, object]) -> List[Dict[str, object]]:
    """Build conservative ID audit rows for directed SCM edges and explicit SCM queries."""
    admg = admg_from_scm_graph(scm_graph)
    rows: List[Dict[str, object]] = []
    seen: set[tuple[str, str, str]] = set()

    def _append_result(x: str, y: str, *, source_kind: str, payload: Mapping[str, object]) -> None:
        x = _s(x)
        y = _s(y)
        if not x or not y:
            return
        query_id = _s(payload.get("id") or payload.get("insight_id"))
        key = (x, y, query_id or source_kind)
        if key in seen:
            return
        seen.add(key)
        result = identify_effect(
            admg,
            x,
            y,
            adjustment_set=parse_field_list(payload.get("candidate_covariates") or payload.get("adjustment_set")),
            mediators=parse_field_list(payload.get("post_treatment_columns") or payload.get("mediators")),
            strategy_hint=payload.get("identification_strategy", ""),
        ).to_dict()
        result.update(
            {
                "edge_authority_level": _s(payload.get("edge_authority_level")) or ("domain_scm_query" if source_kind == "scm_query" else ""),
                "edge_source_artifact": _s(payload.get("edge_source_artifact")) or ("scm_input.query" if source_kind == "scm_query" else ""),
                "insight_id": query_id,
                "audit_source": source_kind,
                "query_type": _s(payload.get("type") or payload.get("query_type")),
                "declared_estimand": _s(payload.get("estimand")),
            }
        )
        rows.append(result)

    for edge in scm_graph.get("edges", []) if isinstance(scm_graph, Mapping) else []:
        if not isinstance(edge, Mapping):
            continue
        kind = _s(edge.get("edge_kind", edge.get("edge_type", ""))).lower()
        if kind in {"bidirected", "latent_confounding", "unobserved_confounding", "exogenous_noise", "noise_input", "latent_noise"}:
            continue
        _append_result(
            edge.get("source", edge.get("from", "")),
            edge.get("target", edge.get("to", "")),
            source_kind="directed_edge",
            payload=edge,
        )

    for query in scm_graph.get("queries", []) if isinstance(scm_graph, Mapping) else []:
        if not isinstance(query, Mapping):
            continue
        _append_result(
            query.get("treatment", query.get("treatment_col", query.get("source", ""))),
            query.get("outcome", query.get("outcome_col", query.get("target", ""))),
            source_kind="scm_query",
            payload=query,
        )
    return rows


def id_algorithm_summary(admg: ADMG, audit_rows: Optional[Sequence[Mapping[str, object]]] = None) -> Dict[str, object]:
    rows = list(audit_rows or [])
    identifiable = sum(1 for r in rows if bool(r.get("identifiable")))
    possible_hedges = sum(1 for r in rows if int(r.get("possible_hedge") or 0) == 1)
    cycle_blocked = sum(1 for r in rows if _s(r.get("id_strategy")) == "blocked_directed_cycle")
    frontdoor_ok = sum(1 for r in rows if _s(r.get("id_strategy")) == "frontdoor_limited")
    frontdoor_blocked = sum(1 for r in rows if _s(r.get("id_strategy")) == "blocked_invalid_frontdoor")
    backdoor_ok = sum(1 for r in rows if _s(r.get("id_strategy")) == "backdoor_adjustment")
    backdoor_blocked = sum(1 for r in rows if _s(r.get("id_strategy")) == "blocked_invalid_backdoor_adjustment")
    observed_dag_tf = sum(1 for r in rows if _s(r.get("id_strategy")) == "observed_dag_truncated_factorization")
    district_recursive = sum(1 for r in rows if int(r.get("district_requires_recursive_id") or 0) == 1)
    nontrivial_ancestral = sum(1 for r in rows if _s(r.get("nontrivial_ancestral_districts")))
    recursive_blocked = sum(1 for r in rows if _s(r.get("id_algorithm_level")) == "recursive_id_skeleton_blocked")
    c_factor_decomposed = sum(1 for r in rows if _s(r.get("c_factor_status")) == "decomposed_requires_recursive_c_factor_id")
    c_factor_product_ok = sum(1 for r in rows if int(r.get("c_factor_product_ok") or 0) == 1)
    recursive_identified = sum(1 for r in rows if int(r.get("recursive_identified") or 0) == 1)
    full_recursive_step2 = sum(1 for r in rows if _s(r.get("id_strategy")) == "full_recursive_id_step2")
    full_recursive_step3 = sum(1 for r in rows if _s(r.get("id_strategy")) == "full_recursive_id_step3")
    full_recursive_step4 = sum(1 for r in rows if _s(r.get("id_strategy")) == "full_recursive_id_step4")
    full_recursive_step5 = sum(1 for r in rows if _s(r.get("id_strategy")) == "full_recursive_id_step5")
    q_factor_general_recursion = sum(1 for r in rows if _s(r.get("q_factor_id_status")) in {"identified_general_q_input_subdistrict_recursion", "blocked_general_q_input_recursion_formal_hedge", "blocked_general_q_input_subdistrict_recursion"})
    q_input_multi_intervention = sum(1 for r in rows if "Q_INPUT_MULTI_INTERVENTION_SUBDISTRICT_RECURSION_IDENTIFIED_STEP29" in _s(r.get("recursive_expression_json")) or "Q_INPUT_MULTI_INTERVENTION_SUBDISTRICT_RECURSION_IDENTIFIED_STEP29" in _s(r.get("recursive_trace_json")))
    recursive_expr_identified = sum(1 for r in rows if _s(r.get("recursive_formula_source")).startswith("identified_"))
    formal_hedge_candidates = sum(1 for r in rows if _s(r.get("recursive_formula_source")) in {"blocked_formal_hedge_candidate", "blocked_formal_hedge_certificate"} or _s(r.get("recursive_blocker_class")) in {"formal_hedge_candidate", "formal_hedge_certificate"})
    subdistrict_q_pending = sum(1 for r in rows if _s(r.get("recursive_blocker_class")) == "subdistrict_q_factor_input_pending")
    ancestor_reduced = sum(1 for r in rows if int(r.get("recursive_ancestor_reduction_applied") or 0) == 1)
    symbolic_identified = sum(1 for r in rows if _s(r.get("symbolic_formula_status")) == "identified_symbolic_formula")
    symbolic_unresolved = sum(1 for r in rows if _s(r.get("symbolic_formula_status")) == "blocked_symbolic_formula_unresolved_c_factor")
    proof_identified = sum(1 for r in rows if _s(r.get("id_proof_status")) == "identified_proof_trace")
    proof_blocked = sum(1 for r in rows if _s(r.get("id_proof_status")) == "blocked_proof_trace")
    subproblem_plans = sum(1 for r in rows if _s(r.get("recursive_subproblem_plan_json")))
    unresolved_cfactor_plans = sum(1 for r in rows if _s(r.get("recursive_blocker_class")) == "unresolved_c_factor")
    possible_hedge_plans = sum(1 for r in rows if _s(r.get("recursive_blocker_class")) == "possible_hedge")
    summary = id_capability_flags()
    summary.update({
        "n_id_audit_rows": int(len(rows)),
        "n_identifiable_limited": int(identifiable),
        "n_not_identifiable_limited": int(len(rows) - identifiable),
        "n_possible_hedge_witnesses": int(possible_hedges),
        "n_blocked_directed_cycles": int(cycle_blocked),
        "n_backdoor_dsep_verified": int(backdoor_ok),
        "n_backdoor_dsep_blocked": int(backdoor_blocked),
        "n_observed_dag_truncated_factorization": int(observed_dag_tf),
        "n_frontdoor_dsep_verified": int(frontdoor_ok),
        "n_frontdoor_dsep_blocked": int(frontdoor_blocked),
        "n_rows_requiring_recursive_id_by_district": int(district_recursive),
        "n_rows_with_nontrivial_ancestral_districts": int(nontrivial_ancestral),
        "n_c_factor_product_decomposed": int(c_factor_decomposed),
        "n_c_factor_product_ok": int(c_factor_product_ok),
        "n_recursive_skeleton_identified": int(recursive_identified),
        "n_full_recursive_step2_identified": int(full_recursive_step2),
        "n_full_recursive_step3_identified": int(full_recursive_step3),
        "n_full_recursive_step4_identified": int(full_recursive_step4),
        "n_full_recursive_step5_identified": int(full_recursive_step5),
        "n_q_factor_general_q_input_recursion_step45": int(q_factor_general_recursion),
        "n_q_input_multi_intervention_step29": int(q_input_multi_intervention),
        "n_recursive_expression_identified": int(recursive_expr_identified),
        "n_formal_hedge_candidates": int(formal_hedge_candidates),
        "n_subdistrict_q_factor_pending": int(subdistrict_q_pending),
        "n_recursive_skeleton_blocked": int(recursive_blocked),
        "n_recursive_ancestor_reductions": int(ancestor_reduced),
        "n_symbolic_formulas_identified": int(symbolic_identified),
        "n_symbolic_formulas_unresolved_c_factor": int(symbolic_unresolved),
        "n_id_proof_traces_identified": int(proof_identified),
        "n_id_proof_traces_blocked": int(proof_blocked),
        "n_recursive_subproblem_plans": int(subproblem_plans),
        "n_recursive_subproblem_unresolved_c_factor": int(unresolved_cfactor_plans),
        "n_recursive_subproblem_possible_hedge": int(possible_hedge_plans),
        "directed_acyclic": int(len(directed_cycle_nodes(admg)) == 0),
        "directed_cycle_nodes": "|".join(directed_cycle_nodes(admg)),
        "n_c_components": int(len(admg.districts())),
        "n_nontrivial_c_components": int(sum(1 for d in admg.districts() if len(d) > 1)),
    })
    return summary


__all__ = [
    "BackdoorDiagnostic",
    "DSeparationDiagnostic",
    "FrontdoorDiagnostic",
    "HedgeDiagnostic",
    "IDResult",
    "IDSetResult",
    "IDProofDiagnostic",
    "TruncatedFactorizationDiagnostic",
    "SymbolicFormulaDiagnostic",
    "CComponentDiagnostic",
    "CFactorDecompositionDiagnostic",
    "RecursiveIDDiagnostic",
    "RecursiveIDExpressionDiagnostic",
    "IDSubproblemDiagnostic",
    "backdoor_diagnostic",
    "c_component_diagnostic",
    "c_factor_decomposition_diagnostic",
    "d_separation_diagnostic",
    "directed_cycle_nodes",
    "directed_path_exists",
    "directed_paths",
    "frontdoor_diagnostic",
    "hedge_diagnostic",
    "nodes_on_directed_paths",
    "topological_order",
    "truncated_factorization_diagnostic",
    "symbolic_formula_diagnostic",
    "same_district",
    "parse_field_list",
    "parse_formula_term_list",
    "recursive_id_diagnostic",
    "recursive_id_expression_diagnostic",
    "recursive_id_set_expression_diagnostic",
    "recursive_subproblem_diagnostic",
    "id_proof_diagnostic",
    "identify_effect",
    "identify_effect_set",
    "identify_effect_from_scm_graph",
    "id_audit_rows_from_scm_graph",
    "id_algorithm_summary",
]
