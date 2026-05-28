from __future__ import annotations

"""Step 68 Full-ID readiness matrix for Amantia SCM-ID.

This module is a **gate**, not a proof of arbitrary Shpitser/Pearl ID
completeness.  It exercises the public ``full_id`` and conservative IDC APIs
across the families that must stay green while the implementation migrates from
partial recursive ID to full recursive ID.

The matrix deliberately includes successful branches, certified failures, input
rejections, and delegated branches. Step 60 additionally checks that the first non-trivial ID-4 district decomposition family is owned by the canonical formula layer. Step 61 additionally checks that blocked/invalid public outputs expose failure certificates. Step 62 adds conservative IDC condition-pruning cases. Step 68 separates failure/rejection authority from formula authority so certified or explicit non-identification is not counted as delegated formula construction. A green matrix means the current supported
surface is internally coherent.  It does **not** permit raising
``full_recursive_id_implemented`` or ``full_id_claim_allowed``.
"""

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple
import json

from .admg import ADMG, admg_from_edges
from .id_full import full_id, identify_conditional_effect
from .id_algorithm_common import _s

ID_FULL_READINESS_MATRIX_VERSION = "id_full_readiness_matrix_v8_step68"


@dataclass(frozen=True)
class IDFullReadinessCase:
    case_id: str
    description: str
    query_kind: str  # "id" or "idc"
    graph: ADMG
    treatments: Sequence[str]
    outcomes: Sequence[str]
    conditions: Sequence[str] = ()
    expected_identified: bool = False
    expected_status: str = ""
    expected_status_contains: str = ""
    expected_primary_formula_authority: str = ""
    expected_joint_primary_formula_authority: str = ""
    expected_canonical_rules: Sequence[str] = ()
    forbidden_canonical_rules: Sequence[str] = ()
    expected_blocker_class: str = ""
    expected_pending_operator: str = ""
    expected_idc_pruning_status: str = ""
    expected_idc_effective_conditions: str = ""
    expected_idc_pruned_conditions: str = ""
    expected_formula_contains: Sequence[str] = ()
    expected_canonical_formula_used: int | None = None
    expected_canonical_id7_used: int | None = None
    expected_full_id_claim_allowed: int = 0


@dataclass(frozen=True)
class IDFullReadinessRow:
    case_id: str
    description: str
    query_kind: str
    passed: bool
    identified: bool
    expected_identified: bool
    identification_status: str
    expected_status: str
    expected_status_contains: str
    primary_formula_authority: str
    expected_primary_formula_authority: str
    joint_primary_formula_authority: str
    expected_joint_primary_formula_authority: str
    canonical_rules: str
    expected_canonical_rules: str
    forbidden_canonical_rules: str
    canonical_formula_used_for_output: str
    expected_canonical_formula_used: str
    canonical_id7_carried_q_formula_used: str
    expected_canonical_id7_used: str
    blocker_class: str
    expected_blocker_class: str
    pending_operator: str
    expected_pending_operator: str
    idc_pruning_status: str
    expected_idc_pruning_status: str
    idc_effective_conditions: str
    expected_idc_effective_conditions: str
    idc_pruned_conditions: str
    expected_idc_pruned_conditions: str
    full_id_claim_allowed: int
    expected_full_id_claim_allowed: int
    formula: str
    expected_formula_contains: str
    reason_codes: str
    failure_certificate_status: str = ""
    failure_certified: int = 0
    formal_hedge_certificate_present: int = 0
    failure: str = ""

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _json_loads(raw: object) -> Dict[str, object]:
    if not raw:
        return {}
    try:
        parsed = json.loads(str(raw))
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _rules_from_result(result: Mapping[str, object]) -> List[str]:
    rules = _s(result.get("canonical_rules"))
    if rules:
        return [r for r in rules.split("|") if r]
    joint = _json_loads(result.get("joint_full_id_json"))
    joint_rules = _s(joint.get("canonical_rules"))
    return [r for r in joint_rules.split("|") if r] if joint_rules else []


def _joint_primary_authority(result: Mapping[str, object]) -> str:
    joint = _json_loads(result.get("joint_full_id_json"))
    return _s(joint.get("primary_formula_authority"))


def _case_graph(nodes: Sequence[str], directed: Sequence[Tuple[str, str]], bidirected: Sequence[Tuple[str, str]] = ()) -> ADMG:
    return admg_from_edges(nodes, directed, bidirected)


def full_id_readiness_cases() -> List[IDFullReadinessCase]:
    """Return the Step-58 public Full-ID/IDC readiness matrix.

    These cases are intentionally named by branch family.  The expected output
    captures current safe behavior: identified where Amantia has authority,
    rejected where queries/graphs are invalid, and blocked where the hedge/fail
    branch is the correct conservative result.
    """
    return [
        IDFullReadinessCase(
            case_id="id_observed_dag_single_edge",
            description="Observed DAG X -> Y identifies by canonical ID-6 truncated factorization.",
            query_kind="id",
            graph=_case_graph(["X", "Y"], [("X", "Y")]),
            treatments=("X",),
            outcomes=("Y",),
            expected_identified=True,
            expected_status="identified_observed_dag_truncated_factorization_set_case",
            expected_primary_formula_authority="id_canonical_formula_step60",
            expected_canonical_rules=("ID-6",),
            expected_formula_contains=("P_{do(X)}(Y)", "P(Y | X)"),
            expected_canonical_formula_used=1,
        ),
        IDFullReadinessCase(
            case_id="id_observed_dag_chain_marginalization",
            description="Observed chain X -> Z -> Y identifies and sums out non-outcome Z.",
            query_kind="id",
            graph=_case_graph(["X", "Z", "Y"], [("X", "Z"), ("Z", "Y")]),
            treatments=("X",),
            outcomes=("Y",),
            expected_identified=True,
            expected_status="identified_observed_dag_truncated_factorization_set_case",
            expected_primary_formula_authority="id_canonical_formula_step60",
            expected_canonical_rules=("ID-6",),
            expected_formula_contains=("sum_{Z}", "P(Z | X)", "P(Y | Z)"),
            expected_canonical_formula_used=1,
        ),
        IDFullReadinessCase(
            case_id="id_observed_dag_multi_treatment",
            description="Set-valued intervention {X1, X2} -> Y is supported for observed DAGs.",
            query_kind="id",
            graph=_case_graph(["X1", "X2", "Y"], [("X1", "Y"), ("X2", "Y")]),
            treatments=("X1", "X2"),
            outcomes=("Y",),
            expected_identified=True,
            expected_status="identified_observed_dag_truncated_factorization_set_case",
            expected_primary_formula_authority="id_canonical_formula_step60",
            expected_canonical_rules=("ID-6",),
            expected_formula_contains=("P_{do(X1,X2)}(Y)", "P(Y | X1,X2)"),
            expected_canonical_formula_used=1,
        ),
        IDFullReadinessCase(
            case_id="id_observed_dag_multi_outcome",
            description="Set-valued outcome {Y1, Y2} is supported for observed DAGs.",
            query_kind="id",
            graph=_case_graph(["X", "Y1", "Y2"], [("X", "Y1"), ("X", "Y2")]),
            treatments=("X",),
            outcomes=("Y1", "Y2"),
            expected_identified=True,
            expected_status="identified_observed_dag_truncated_factorization_set_case",
            expected_primary_formula_authority="id_canonical_formula_step60",
            expected_canonical_rules=("ID-6",),
            expected_formula_contains=("P_{do(X)}(Y1,Y2)", "P(Y1 | X)", "P(Y2 | X)"),
            expected_canonical_formula_used=1,
        ),
        IDFullReadinessCase(
            case_id="id_graphical_zero_effect_with_latent_association",
            description="X <-> Y without a directed path identifies as zero-effect P(Y) with canonical ID-2 authority.",
            query_kind="id",
            graph=_case_graph(["X", "Y"], [], [("X", "Y")]),
            treatments=("X",),
            outcomes=("Y",),
            expected_identified=True,
            expected_status="identified_graphical_zero_effect",
            expected_primary_formula_authority="id_canonical_formula_step60",
            expected_canonical_rules=("ID-1", "ID-2"),
            expected_formula_contains=("P_{do(X)}(Y)", "P(Y)"),
            expected_canonical_formula_used=1,
        ),
        IDFullReadinessCase(
            case_id="id_frontdoor_canonical_id7",
            description="Classic frontdoor X -> Z -> Y with X <-> Y uses the gated canonical ID-7 formula.",
            query_kind="id",
            graph=_case_graph(["X", "Z", "Y"], [("X", "Z"), ("Z", "Y")], [("X", "Y")]),
            treatments=("X",),
            outcomes=("Y",),
            expected_identified=True,
            expected_primary_formula_authority="id_canonical_formula_step60",
            expected_canonical_rules=("ID-4", "ID-7"),
            expected_formula_contains=("sum_{Z}", "X_prime", "P(Z | X)", "P(Y | X_prime,Z)"),
            expected_canonical_formula_used=1,
            expected_canonical_id7_used=1,
        ),
        IDFullReadinessCase(
            case_id="id_contextual_frontdoor_canonical_id7_step60",
            description="Contextual frontdoor joint P(Y,W | do(X)) uses canonical Step-60 carried-Q ID-7 authority instead of the legacy delegate.",
            query_kind="id",
            graph=_case_graph(["X", "Z", "Y", "W"], [("X", "Z"), ("Z", "Y"), ("W", "Y")], [("X", "Y")]),
            treatments=("X",),
            outcomes=("Y", "W"),
            expected_identified=True,
            expected_primary_formula_authority="id_canonical_formula_step60",
            expected_canonical_rules=("ID-4", "ID-7"),
            expected_formula_contains=("P_{do(X)}(Y,W)", "sum_{Z}", "P(W)", "P(Z | X)", "P(X_prime | W)", "P(Y | W,X_prime,Z)"),
            expected_canonical_formula_used=1,
            expected_canonical_id7_used=1,
        ),
        IDFullReadinessCase(
            case_id="id_chain_frontdoor_canonical_id7_step66",
            description="Two-mediator frontdoor chain X -> Z1 -> Z2 -> Y with X <-> Y uses the Step-66 canonical carried-Q ID-7 family.",
            query_kind="id",
            graph=_case_graph(["X", "Z1", "Z2", "Y"], [("X", "Z1"), ("Z1", "Z2"), ("Z2", "Y")], [("X", "Y")]),
            treatments=("X",),
            outcomes=("Y",),
            expected_identified=True,
            expected_primary_formula_authority="id_canonical_formula_step60",
            expected_canonical_rules=("ID-4", "ID-7"),
            expected_formula_contains=("P_{do(X)}(Y)", "sum_{Z1,Z2}", "P(Z1 | X)", "P(Z2 | Z1)", "P(X_prime)", "P(Y | X_prime,Z1,Z2)"),
            expected_canonical_formula_used=1,
            expected_canonical_id7_used=1,
        ),
        IDFullReadinessCase(
            case_id="id_parallel_frontdoor_set_canonical_id7_step67",
            description="Parallel frontdoor-set mediators X -> {Z1,Z2} -> Y with X <-> Y use the Step-67 canonical carried-Q ID-7 family.",
            query_kind="id",
            graph=_case_graph(["X", "Z1", "Z2", "Y"], [("X", "Z1"), ("X", "Z2"), ("Z1", "Y"), ("Z2", "Y")], [("X", "Y")]),
            treatments=("X",),
            outcomes=("Y",),
            expected_identified=True,
            expected_primary_formula_authority="id_canonical_formula_step60",
            expected_canonical_rules=("ID-4", "ID-7"),
            expected_formula_contains=("P_{do(X)}(Y)", "sum_{Z1,Z2}", "P(Z1,Z2 | X)", "P(X_prime)", "P(Y | X_prime,Z1,Z2)"),
            expected_canonical_formula_used=1,
            expected_canonical_id7_used=1,
        ),
        IDFullReadinessCase(
            case_id="id_direct_confounding_hedge_fail",
            description="Direct X -> Y plus X <-> Y must block with formal hedge authority.",
            query_kind="id",
            graph=_case_graph(["X", "Y"], [("X", "Y")], [("X", "Y")]),
            treatments=("X",),
            outcomes=("Y",),
            expected_identified=False,
            expected_status="blocked_formal_hedge_certificate",
            expected_canonical_rules=("ID-5", "ID-7"),
            expected_blocker_class="formal_hedge_certificate",
            expected_pending_operator="fail_id_or_construct_full_hedge_certificate",
            expected_canonical_formula_used=0,
        ),
        IDFullReadinessCase(
            case_id="id_q_factor_full_district",
            description="Full-district Q-factor branch remains identified and canonical-authority backed.",
            query_kind="id",
            graph=_case_graph(["X", "Z", "Y"], [("X", "Y"), ("Z", "Y")], [("Z", "Y")]),
            treatments=("X",),
            outcomes=("Y",),
            expected_identified=True,
            expected_status="identified_q_factor_full_district",
            expected_primary_formula_authority="id_canonical_formula_step60",
            expected_canonical_rules=("ID-6",),
            expected_formula_contains=("sum_{Z}", "P(Z | X)", "P(Y | X,Z)"),
            expected_canonical_formula_used=1,
        ),
        IDFullReadinessCase(
            case_id="id_canonical_district_decomposition_id4",
            description="Non-trivial district decomposition is identified by canonical ID-4 formula authority.",
            query_kind="id",
            graph=_case_graph(["X", "Y", "Z", "W"], [("X", "Y"), ("Z", "Y"), ("W", "Z")], [("X", "Z")]),
            treatments=("X",),
            outcomes=("Y",),
            expected_identified=True,
            expected_status="identified_recursive_district_decomposition",
            expected_primary_formula_authority="id_canonical_formula_step60",
            expected_canonical_rules=("ID-4", "ID-6"),
            expected_formula_contains=("P_{do(X)}(Y)", "sum_{W,Z}", "P(W)", "P(Y | W,X,Z)", "P(Z | W)"),
            expected_canonical_formula_used=1,
        ),
        IDFullReadinessCase(
            case_id="id_multi_node_hedge_fail",
            description="A larger confounded mediator graph still blocks through the formal hedge fail branch.",
            query_kind="id",
            graph=_case_graph(["X", "M", "Y"], [("X", "M"), ("M", "Y")], [("X", "Y"), ("M", "Y")]),
            treatments=("X",),
            outcomes=("Y",),
            expected_identified=False,
            expected_status="blocked_formal_hedge_certificate",
            expected_canonical_rules=("ID-5", "ID-7"),
            expected_blocker_class="formal_hedge_certificate",
            expected_pending_operator="fail_id_or_construct_full_hedge_certificate",
            expected_canonical_formula_used=0,
        ),
        IDFullReadinessCase(
            case_id="id_directed_cycle_rejected",
            description="Directed cycles are rejected before ID authority is considered.",
            query_kind="id",
            graph=_case_graph(["X", "Y"], [("X", "Y"), ("Y", "X")]),
            treatments=("X",),
            outcomes=("Y",),
            expected_identified=False,
            expected_status="blocked_directed_cycle",
            expected_primary_formula_authority="id_failure_certificate_step68",
            expected_blocker_class="directed_cycle",
            expected_pending_operator="repair_or_reject_cyclic_directed_graph",
            expected_canonical_formula_used=0,
        ),
        IDFullReadinessCase(
            case_id="id_invalid_overlap_rejected",
            description="Treatment/outcome overlap is rejected as an invalid query.",
            query_kind="id",
            graph=_case_graph(["X", "Y"], [("X", "Y")]),
            treatments=("X",),
            outcomes=("X",),
            expected_identified=False,
            expected_status="invalid_full_id_query",
            expected_primary_formula_authority="id_failure_certificate_step68",
            expected_blocker_class="invalid_query",
            expected_pending_operator="validate_full_id_query",
            expected_canonical_formula_used=0,
        ),
        IDFullReadinessCase(
            case_id="id_missing_node_rejected",
            description="Queries containing nodes outside the graph are rejected.",
            query_kind="id",
            graph=_case_graph(["X", "Y"], [("X", "Y")]),
            treatments=("X",),
            outcomes=("Y_missing",),
            expected_identified=False,
            expected_status="invalid_full_id_query",
            expected_primary_formula_authority="id_failure_certificate_step68",
            expected_blocker_class="invalid_query",
            expected_pending_operator="validate_full_id_query",
            expected_canonical_formula_used=0,
        ),
        IDFullReadinessCase(
            case_id="idc_observed_dag_single_condition",
            description="IDC ratio over identified P(Y,Z | do(X)) succeeds in an observed DAG.",
            query_kind="idc",
            graph=_case_graph(["X", "Y", "Z"], [("X", "Y"), ("Z", "Y")]),
            treatments=("X",),
            outcomes=("Y",),
            conditions=("Z",),
            expected_identified=True,
            expected_status="identified_idc_ratio_over_identified_joint_step57",
            expected_joint_primary_formula_authority="id_canonical_formula_step60",
            expected_formula_contains=("P_{do(X)}(Y | Z)", "sum_{Y}"),
        ),
        IDFullReadinessCase(
            case_id="idc_observed_dag_multi_condition",
            description="IDC accepts multiple conditions when the joint is identified.",
            query_kind="idc",
            graph=_case_graph(["X", "Y", "Z1", "Z2"], [("X", "Y"), ("Z1", "Y"), ("Z2", "Y")]),
            treatments=("X",),
            outcomes=("Y",),
            conditions=("Z1", "Z2"),
            expected_identified=True,
            expected_status="identified_idc_ratio_over_identified_joint_step57",
            expected_joint_primary_formula_authority="id_canonical_formula_step60",
            expected_formula_contains=("P_{do(X)}(Y | Z1,Z2)", "sum_{Y}"),
        ),
        IDFullReadinessCase(
            case_id="idc_frontdoor_joint_condition",
            description="IDC can normalize a frontdoor-shaped identified joint query.",
            query_kind="idc",
            graph=_case_graph(["X", "Z", "Y", "W"], [("X", "Z"), ("Z", "Y"), ("W", "Y")], [("X", "Y")]),
            treatments=("X",),
            outcomes=("Y",),
            conditions=("W",),
            expected_identified=True,
            expected_status="identified_idc_ratio_over_identified_joint_step57",
            expected_joint_primary_formula_authority="id_canonical_formula_step60",
            expected_formula_contains=("P_{do(X)}(Y | W)", "sum_{Y}", "P(W)", "P(X_prime | W)"),
        ),
        IDFullReadinessCase(
            case_id="idc_prunes_isolated_condition_step62",
            description="IDC Step-62 prunes an isolated condition Z and reduces P(Y | do(X), Z) to P(Y | do(X)).",
            query_kind="idc",
            graph=_case_graph(["X", "Y", "Z"], [("X", "Y")]),
            treatments=("X",),
            outcomes=("Y",),
            conditions=("Z",),
            expected_identified=True,
            expected_status="identified_idc_pruned_to_marginal_effect_step62",
            expected_idc_pruning_status="idc_pruning_all_conditions_removed_step62",
            expected_idc_effective_conditions="",
            expected_idc_pruned_conditions="Z",
            expected_formula_contains=("P_{do(X)}(Y | Z) = P_{do(X)}(Y)", "P(Y | X)"),
        ),
        IDFullReadinessCase(
            case_id="idc_partially_prunes_condition_step62",
            description="IDC Step-62 prunes isolated Z while keeping W because W remains connected to Y.",
            query_kind="idc",
            graph=_case_graph(["X", "Y", "W", "Z"], [("X", "Y"), ("W", "Y")]),
            treatments=("X",),
            outcomes=("Y",),
            conditions=("W", "Z"),
            expected_identified=True,
            expected_status="identified_idc_ratio_over_identified_joint_step57",
            expected_joint_primary_formula_authority="id_canonical_formula_step60",
            expected_idc_pruning_status="idc_pruning_partial_conditions_removed_step62",
            expected_idc_effective_conditions="W",
            expected_idc_pruned_conditions="Z",
            expected_formula_contains=("P_{do(X)}(Y | W,Z) = P_{do(X)}(Y | W)", "sum_{Y}"),
        ),
        IDFullReadinessCase(
            case_id="idc_joint_hedge_blocked",
            description="IDC blocks when the required joint P(Y,Z | do(X)) is not identified.",
            query_kind="idc",
            graph=_case_graph(["X", "Y", "Z"], [("X", "Y"), ("Z", "Y")], [("X", "Y")]),
            treatments=("X",),
            outcomes=("Y",),
            conditions=("Z",),
            expected_identified=False,
            expected_status="blocked_idc_joint_not_identified_step57",
            expected_joint_primary_formula_authority="id_failure_certificate_step68",
            expected_blocker_class="formal_hedge_certificate",
            expected_pending_operator="identify_joint_yz_under_do_x_before_idc_ratio",
        ),
        IDFullReadinessCase(
            case_id="idc_invalid_overlap_rejected",
            description="IDC rejects overlapping outcome/condition sets before doing any ratio construction.",
            query_kind="idc",
            graph=_case_graph(["X", "Y", "Z"], [("X", "Y")]),
            treatments=("X",),
            outcomes=("Y",),
            conditions=("Y",),
            expected_identified=False,
            expected_status="invalid_idc_query",
            expected_blocker_class="invalid_query",
            expected_pending_operator="validate_idc_query",
        ),
    ]


def evaluate_full_id_readiness_case(case: IDFullReadinessCase) -> IDFullReadinessRow:
    if case.query_kind == "id":
        result = full_id(case.graph, case.treatments, case.outcomes).to_dict()
    elif case.query_kind == "idc":
        result = identify_conditional_effect(case.graph, case.treatments, case.outcomes, case.conditions).to_dict()
    else:
        raise ValueError(f"Unsupported query_kind: {case.query_kind}")

    rules = _rules_from_result(result)
    primary = _s(result.get("primary_formula_authority"))
    joint_primary = _joint_primary_authority(result)
    canonical_used = result.get("canonical_formula_used_for_output", "")
    id7_used = result.get("canonical_id7_carried_q_formula_used", "")
    status = _s(result.get("identification_status"))
    formula = _s(result.get("formula"))
    blocker_class = _s(result.get("blocker_class"))
    pending_operator = _s(result.get("pending_operator"))
    idc_pruning_status = _s(result.get("idc_pruning_status"))
    idc_effective_conditions = _s(result.get("idc_effective_conditions"))
    idc_pruned_conditions = _s(result.get("idc_pruned_conditions"))
    full_id_claim_allowed = int(result.get("full_id_claim_allowed") or 0)
    failure_certificate_status = _s(result.get("failure_certificate_status"))
    failure_certified = int(result.get("failure_certified") or 0)
    formal_hedge_certificate_present = int(bool(_s(result.get("formal_hedge_certificate_json"))))

    failures: List[str] = []
    if bool(result.get("identified")) != bool(case.expected_identified):
        failures.append(f"identified={result.get('identified')} expected={case.expected_identified}")
    if case.expected_status and status != case.expected_status:
        failures.append(f"status={status} expected={case.expected_status}")
    if case.expected_status_contains and case.expected_status_contains not in status:
        failures.append(f"status_missing={case.expected_status_contains}")
    if case.expected_primary_formula_authority and primary != case.expected_primary_formula_authority:
        failures.append(f"primary_formula_authority={primary} expected={case.expected_primary_formula_authority}")
    if case.expected_joint_primary_formula_authority and joint_primary != case.expected_joint_primary_formula_authority:
        failures.append(f"joint_primary_formula_authority={joint_primary} expected={case.expected_joint_primary_formula_authority}")
    for rule in case.expected_canonical_rules:
        if rule not in rules:
            failures.append(f"missing_canonical_rule={rule}")
    for rule in case.forbidden_canonical_rules:
        if rule in rules:
            failures.append(f"forbidden_canonical_rule={rule}")
    if case.expected_blocker_class and blocker_class != case.expected_blocker_class:
        failures.append(f"blocker_class={blocker_class} expected={case.expected_blocker_class}")
    if case.expected_pending_operator and pending_operator != case.expected_pending_operator:
        failures.append(f"pending_operator={pending_operator} expected={case.expected_pending_operator}")
    if case.expected_idc_pruning_status and idc_pruning_status != case.expected_idc_pruning_status:
        failures.append(f"idc_pruning_status={idc_pruning_status} expected={case.expected_idc_pruning_status}")
    if case.expected_idc_effective_conditions and idc_effective_conditions != case.expected_idc_effective_conditions:
        failures.append(f"idc_effective_conditions={idc_effective_conditions} expected={case.expected_idc_effective_conditions}")
    if case.expected_idc_pruned_conditions and idc_pruned_conditions != case.expected_idc_pruned_conditions:
        failures.append(f"idc_pruned_conditions={idc_pruned_conditions} expected={case.expected_idc_pruned_conditions}")
    for part in case.expected_formula_contains:
        if part not in formula:
            failures.append(f"formula_missing={part}")
    if case.expected_canonical_formula_used is not None and int(canonical_used or 0) != int(case.expected_canonical_formula_used):
        failures.append(f"canonical_formula_used={canonical_used} expected={case.expected_canonical_formula_used}")
    if case.expected_canonical_id7_used is not None and int(id7_used or 0) != int(case.expected_canonical_id7_used):
        failures.append(f"canonical_id7_used={id7_used} expected={case.expected_canonical_id7_used}")
    if full_id_claim_allowed != int(case.expected_full_id_claim_allowed):
        failures.append(f"full_id_claim_allowed={full_id_claim_allowed} expected={case.expected_full_id_claim_allowed}")
    if not bool(result.get("identified")) and not failure_certificate_status:
        failures.append("missing_failure_certificate_status_for_nonidentified_output")
    if blocker_class == "formal_hedge_certificate" and not formal_hedge_certificate_present:
        failures.append("formal_hedge_block_without_certificate_json")

    return IDFullReadinessRow(
        case_id=case.case_id,
        description=case.description,
        query_kind=case.query_kind,
        passed=not failures,
        identified=bool(result.get("identified")),
        expected_identified=bool(case.expected_identified),
        identification_status=status,
        expected_status=case.expected_status,
        expected_status_contains=case.expected_status_contains,
        primary_formula_authority=primary,
        expected_primary_formula_authority=case.expected_primary_formula_authority,
        joint_primary_formula_authority=joint_primary,
        expected_joint_primary_formula_authority=case.expected_joint_primary_formula_authority,
        canonical_rules="|".join(rules),
        expected_canonical_rules="|".join(case.expected_canonical_rules),
        forbidden_canonical_rules="|".join(case.forbidden_canonical_rules),
        canonical_formula_used_for_output=str(canonical_used),
        expected_canonical_formula_used="" if case.expected_canonical_formula_used is None else str(case.expected_canonical_formula_used),
        canonical_id7_carried_q_formula_used=str(id7_used),
        expected_canonical_id7_used="" if case.expected_canonical_id7_used is None else str(case.expected_canonical_id7_used),
        blocker_class=blocker_class,
        expected_blocker_class=case.expected_blocker_class,
        pending_operator=pending_operator,
        expected_pending_operator=case.expected_pending_operator,
        idc_pruning_status=idc_pruning_status,
        expected_idc_pruning_status=case.expected_idc_pruning_status,
        idc_effective_conditions=idc_effective_conditions,
        expected_idc_effective_conditions=case.expected_idc_effective_conditions,
        idc_pruned_conditions=idc_pruned_conditions,
        expected_idc_pruned_conditions=case.expected_idc_pruned_conditions,
        full_id_claim_allowed=full_id_claim_allowed,
        expected_full_id_claim_allowed=int(case.expected_full_id_claim_allowed),
        formula=formula,
        expected_formula_contains="|".join(case.expected_formula_contains),
        reason_codes=_s(result.get("reason_codes")),
        failure_certificate_status=failure_certificate_status,
        failure_certified=failure_certified,
        formal_hedge_certificate_present=formal_hedge_certificate_present,
        failure="; ".join(failures),
    )


def run_full_id_readiness_matrix(cases: Iterable[IDFullReadinessCase] | None = None) -> Dict[str, object]:
    rows = [evaluate_full_id_readiness_case(c) for c in (list(cases) if cases is not None else full_id_readiness_cases())]
    passed = sum(1 for r in rows if r.passed)
    identified = sum(1 for r in rows if r.identified)
    blocked = sum(1 for r in rows if not r.identified and "blocked" in r.identification_status)
    invalid = sum(1 for r in rows if not r.identified and "invalid" in r.identification_status)
    delegated = sum(1 for r in rows if r.identified and (r.primary_formula_authority == "recursive_id_set_expression_diagnostic" or r.joint_primary_formula_authority == "recursive_id_set_expression_diagnostic"))
    canonical = sum(1 for r in rows if r.identified and (r.primary_formula_authority == "id_canonical_formula_step60" or r.joint_primary_formula_authority == "id_canonical_formula_step60"))
    idc = sum(1 for r in rows if r.query_kind == "idc")
    failure_certs = sum(1 for r in rows if (not r.identified) and bool(r.failure_certificate_status))
    formal_hedge_certs = sum(1 for r in rows if bool(r.formal_hedge_certificate_present))
    failure_authority = sum(1 for r in rows if (r.primary_formula_authority == "id_failure_certificate_step68" or r.joint_primary_formula_authority == "id_failure_certificate_step68"))
    return {
        "matrix_version": ID_FULL_READINESS_MATRIX_VERSION,
        "n_cases": len(rows),
        "n_passed": passed,
        "n_failed": len(rows) - passed,
        "all_passed": int(passed == len(rows)),
        "n_identified_or_normalized": identified,
        "n_blocked": blocked,
        "n_invalid_rejected": invalid,
        "n_delegated_formula_authority": delegated,
        "n_canonical_formula_authority": canonical,
        "n_idc_cases": idc,
        "n_failure_certificates": failure_certs,
        "n_formal_hedge_certificates": formal_hedge_certs,
        "n_failure_certificate_authority": failure_authority,
        "full_id_claim_allowed": 0,
        "full_id_claim_reason": (
            "Step 68 is a readiness/regression matrix. It confirms supported branches, "
            "certified blocks, IDC normalization, conservative IDC pruning, chain-frontdoor ID-7, and parallel-frontdoor-set ID-7 and failure-authority separation stay coherent, "
            "but it is not a proof of arbitrary recursive ID/IDC completeness. Keep full_recursive_id_implemented=0."
        ),
        "remaining_required_before_full_id_claim": [
            "arbitrary ID-7 carried-Q recursion beyond classic/contextual/chain/parallel frontdoor-shaped and currently supported subdistrict cases",
            "complete formal hedge certificate for every ID-5 FAIL branch",
            "reference/oracle parity matrix for non-trivial ADMG examples",
            "random ADMG fuzzing with conservative no-overclaim invariants",
            "full IDC simplification rules beyond conservative disconnectivity pruning and ratio over identified joint",
            "numeric/symbolic evaluator coverage for every emitted AST node, beyond Step-66 resolved-Q-factor readiness",
        ],
        "rows": [r.to_dict() for r in rows],
    }


__all__ = [
    "ID_FULL_READINESS_MATRIX_VERSION",
    "IDFullReadinessCase",
    "IDFullReadinessRow",
    "full_id_readiness_cases",
    "evaluate_full_id_readiness_case",
    "run_full_id_readiness_matrix",
]
