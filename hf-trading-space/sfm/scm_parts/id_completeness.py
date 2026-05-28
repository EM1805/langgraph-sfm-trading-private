from __future__ import annotations

"""Canonical completeness-gate cases for Amantia's SCM ID layer.

This module is intentionally small and deterministic.  It does not claim that
Amantia implements full Shpitser/Pearl ID; it defines the regression matrix that
must stay green before the final ``full_recursive_id_implemented`` flag can ever
be raised.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Mapping, Sequence
import json

from .admg import ADMG, admg_from_edges
from .id_algorithm import identify_effect
from .id_algorithm_common import _s


@dataclass(frozen=True)
class IDCompletenessCase:
    case_id: str
    description: str
    graph: ADMG
    treatment: str
    outcome: str
    adjustment_set: Sequence[str] = ()
    mediators: Sequence[str] = ()
    strategy_hint: str = ""
    expected_identifiable: bool = False
    expected_strategy: str = ""
    expected_authority: str = ""
    expected_formal_hedge: int = 0
    expected_canonical_rules: Sequence[str] = ()
    forbidden_canonical_rules: Sequence[str] = ()


@dataclass(frozen=True)
class IDCompletenessRow:
    case_id: str
    description: str
    passed: bool
    identifiable: bool
    expected_identifiable: bool
    id_strategy: str
    expected_strategy: str
    identification_authority: str
    expected_authority: str
    formal_hedge_certified: int
    expected_formal_hedge: int
    canonical_rules: str
    expected_canonical_rules: str
    forbidden_canonical_rules: str
    reason_codes: str
    failure: str = ""

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def canonical_id_completeness_cases() -> List[IDCompletenessCase]:
    """Return the required Step-48 ID regression matrix.

    The matrix covers the minimum practical ID gates used by the package today:
    observed-DAG factorization, backdoor, frontdoor, direct hedge FAIL,
    zero-effect, and district/Q-factor decomposition.  These are not a proof of
    full ID completeness; they are the mandatory smoke/regression gates before
    implementing the remaining hard cases.
    """
    return [
        IDCompletenessCase(
            case_id="observed_dag_truncated_factorization",
            description="Observed DAG X -> Y should identify by truncated factorization.",
            graph=admg_from_edges(["X", "Y"], [("X", "Y")], []),
            treatment="X",
            outcome="Y",
            expected_identifiable=True,
            expected_strategy="observed_dag_truncated_factorization",
            expected_authority="recursive_id",
            expected_canonical_rules=("ID-6",),
        ),
        IDCompletenessCase(
            case_id="backdoor_adjustment",
            description="Classic observed confounder Z should identify with supplied backdoor adjustment.",
            graph=admg_from_edges(["X", "Y", "Z"], [("Z", "X"), ("Z", "Y"), ("X", "Y")], []),
            treatment="X",
            outcome="Y",
            adjustment_set=("Z",),
            strategy_hint="backdoor",
            expected_identifiable=True,
            expected_strategy="backdoor_adjustment",
            expected_authority="recursive_id",
            expected_canonical_rules=("ID-6",),
            forbidden_canonical_rules=("ID-5",),
        ),
        IDCompletenessCase(
            case_id="frontdoor_limited",
            description="Classic frontdoor X -> Z -> Y with X <-> Y should identify, not hedge-block.",
            graph=admg_from_edges(["X", "Y", "Z"], [("X", "Z"), ("Z", "Y")], [("X", "Y")]),
            treatment="X",
            outcome="Y",
            mediators=("Z",),
            strategy_hint="frontdoor",
            expected_identifiable=True,
            expected_strategy="frontdoor_limited",
            expected_authority="recursive_id",
            expected_canonical_rules=("ID-4", "ID-7"),
            forbidden_canonical_rules=("ID-5",),
        ),
        IDCompletenessCase(
            case_id="direct_confounding_hedge_fail",
            description="Direct confounded X -> Y plus X <-> Y should fail with formal hedge from recursive FAIL branch.",
            graph=admg_from_edges(["X", "Y"], [("X", "Y")], [("X", "Y")]),
            treatment="X",
            outcome="Y",
            expected_identifiable=False,
            expected_strategy="blocked_formal_hedge_certificate",
            expected_authority="recursive_id_fail_branch",
            expected_formal_hedge=1,
            expected_canonical_rules=("ID-5", "ID-7"),
        ),
        IDCompletenessCase(
            case_id="graphical_zero_effect",
            description="No directed path from X to Y should identify as zero-effect P(Y).",
            graph=admg_from_edges(["X", "Y", "Z"], [("Z", "Y")], []),
            treatment="X",
            outcome="Y",
            expected_identifiable=True,
            expected_strategy="no_directed_effect",
            expected_authority="recursive_id",
            expected_canonical_rules=("ID-6",),
            forbidden_canonical_rules=("ID-5",),
        ),
        IDCompletenessCase(
            case_id="district_decomposition_q_factor",
            description="District decomposition after removing X should identify via Q-factor/product structure.",
            graph=admg_from_edges(["X", "Y", "Z"], [("X", "Y"), ("Z", "Y")], [("X", "Z")]),
            treatment="X",
            outcome="Y",
            expected_identifiable=True,
            expected_strategy="full_recursive_id_step2",
            expected_authority="recursive_id",
            expected_canonical_rules=("ID-4", "ID-6"),
            forbidden_canonical_rules=("ID-5",),
        ),
    ]


def _canonical_rules_from_result(result: Mapping[str, object]) -> List[str]:
    raw_tree = _s(result.get("formula_tree_json"))
    if not raw_tree:
        return []
    try:
        tree = json.loads(raw_tree)
    except Exception:
        return []
    trace = tree.get("canonical_id_trace", {}) if isinstance(tree, Mapping) else {}
    rules = trace.get("applied_rules", []) if isinstance(trace, Mapping) else []
    if not isinstance(rules, list):
        return []
    return [str(r) for r in rules]


def evaluate_id_completeness_case(case: IDCompletenessCase) -> IDCompletenessRow:
    result = identify_effect(
        case.graph,
        case.treatment,
        case.outcome,
        adjustment_set=list(case.adjustment_set),
        mediators=list(case.mediators),
        strategy_hint=case.strategy_hint,
    ).to_dict()
    rules = _canonical_rules_from_result(result)
    failures: List[str] = []
    if bool(result.get("identifiable")) != bool(case.expected_identifiable):
        failures.append(f"identifiable={result.get('identifiable')} expected={case.expected_identifiable}")
    if case.expected_strategy and _s(result.get("id_strategy")) != case.expected_strategy:
        failures.append(f"id_strategy={result.get('id_strategy')} expected={case.expected_strategy}")
    if case.expected_authority and _s(result.get("identification_authority")) != case.expected_authority:
        failures.append(f"authority={result.get('identification_authority')} expected={case.expected_authority}")
    hedge_value = int(result.get("formal_hedge_certified") or 0)
    if hedge_value != int(case.expected_formal_hedge):
        failures.append(f"formal_hedge_certified={hedge_value} expected={case.expected_formal_hedge}")
    for rule in case.expected_canonical_rules:
        if rule not in rules:
            failures.append(f"missing_canonical_rule={rule}")
    for rule in case.forbidden_canonical_rules:
        if rule in rules:
            failures.append(f"forbidden_canonical_rule={rule}")
    return IDCompletenessRow(
        case_id=case.case_id,
        description=case.description,
        passed=not failures,
        identifiable=bool(result.get("identifiable")),
        expected_identifiable=bool(case.expected_identifiable),
        id_strategy=_s(result.get("id_strategy")),
        expected_strategy=case.expected_strategy,
        identification_authority=_s(result.get("identification_authority")),
        expected_authority=case.expected_authority,
        formal_hedge_certified=hedge_value,
        expected_formal_hedge=int(case.expected_formal_hedge),
        canonical_rules="|".join(rules),
        expected_canonical_rules="|".join(case.expected_canonical_rules),
        forbidden_canonical_rules="|".join(case.forbidden_canonical_rules),
        reason_codes=_s(result.get("reason_codes")),
        failure="; ".join(failures),
    )


def run_id_completeness_matrix(cases: Iterable[IDCompletenessCase] | None = None) -> Dict[str, object]:
    rows = [evaluate_id_completeness_case(c) for c in (list(cases) if cases is not None else canonical_id_completeness_cases())]
    passed = sum(1 for r in rows if r.passed)
    return {
        "matrix_version": "id_completeness_matrix_v1_step48",
        "n_cases": len(rows),
        "n_passed": passed,
        "n_failed": len(rows) - passed,
        "all_passed": int(passed == len(rows)),
        "full_id_claim_allowed": 0,
        "full_id_claim_reason": "Completeness matrix is a gate, not a proof of arbitrary full recursive ID. Keep full_recursive_id_implemented=0 until the remaining general ID cases pass.",
        "rows": [r.to_dict() for r in rows],
    }


__all__ = [
    "IDCompletenessCase",
    "IDCompletenessRow",
    "canonical_id_completeness_cases",
    "evaluate_id_completeness_case",
    "run_id_completeness_matrix",
]
