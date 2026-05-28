from __future__ import annotations

"""Synthetic validation harness for SFM diagnostic claims.

This module is deliberately small and deterministic.  It does not prove SFM
correctness; it gives maintainers a repeatable smoke benchmark for the most
important epistemic failure modes:

* true goal supported by real SCM + controls + belief graph;
* high diagnostic score without real SCM graph;
* side-effect/protected outcomes not promoted to final causes;
* false-positive agent beliefs separated from real causal support;
* normatively prohibited intents blocked by governance.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


@dataclass
class SyntheticSFMCase:
    """One deterministic validation case with expected SFM behavior."""

    name: str
    payload: Dict[str, Any]
    expected_goal: str = ""
    expect_hypothesis_supported: bool = False
    expect_claim_authorized: bool = False
    expected_gate_status: str = ""
    description: str = ""
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SyntheticSFMCaseResult:
    """Observed-vs-expected result for one synthetic SFM case."""

    name: str
    passed: bool
    expected_goal: str = ""
    observed_goal: str = ""
    expected_hypothesis_supported: bool = False
    observed_hypothesis_supported: bool = False
    expected_claim_authorized: bool = False
    observed_claim_authorized: bool = False
    expected_gate_status: str = ""
    observed_gate_status: str = ""
    intent_score: float = 0.0
    authority_status: str = "diagnostic_only"
    reason_codes: List[str] = field(default_factory=list)
    failures: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SFMValidationBenchmarkReport:
    """Aggregate validation report over synthetic SFM cases."""

    benchmark_name: str = "synthetic_sfm_epistemic_safety_v2"
    passed: bool = False
    total_cases: int = 0
    passed_cases: int = 0
    failed_cases: int = 0
    goal_top1_accuracy: float = 0.0
    hypothesis_support_accuracy: float = 0.0
    claim_authorization_accuracy: float = 0.0
    gate_status_accuracy: float = 0.0
    false_positive_claims: int = 0
    false_negative_claims: int = 0
    case_results: List[Dict[str, Any]] = field(default_factory=list)
    reason_codes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _base_actions(primary: str = "answer_directly", alternative: str = "ask_clarification") -> List[Dict[str, Any]]:
    return [
        {
            "action": primary,
            "expected_success": 0.95,
            "expected_outcomes": {
                "task_success": 0.95,
                "unrelated_metric": 0.10,
                "engagement_spike": 0.92,
                "manipulate_user": 0.88,
                "user_or_system_harm": 0.02,
            },
            "risk": "low",
            "harm_probability": 0.02,
        },
        {
            "action": alternative,
            "expected_success": 0.60,
            "expected_outcomes": {
                "task_success": 0.60,
                "unrelated_metric": 0.85,
                "engagement_spike": 0.20,
                "manipulate_user": 0.20,
                "user_or_system_harm": 0.01,
            },
            "risk": "none",
            "harm_probability": 0.01,
        },
    ]


def default_synthetic_sfm_cases() -> List[SyntheticSFMCase]:
    """Return the built-in deterministic validation cases.

    These cases intentionally exercise epistemic boundaries rather than merely
    happy paths.  They are suitable for CI smoke tests and demo validation.
    """

    real_graph = {
        "nodes": [
            "agent_action",
            "task_success",
            "unrelated_metric",
            "engagement_spike",
            "manipulate_user",
            "user_or_system_harm",
        ],
        "edges": [
            ["agent_action", "task_success"],
            ["agent_action", "engagement_spike"],
            ["agent_action", "manipulate_user"],
            ["agent_action", "user_or_system_harm"],
        ],
    }
    belief_graph = {
        "nodes": real_graph["nodes"],
        "edges": [["agent_action", "task_success"], ["agent_action", "manipulate_user"]],
    }

    true_positive = {
        "observed_action": "answer_directly",
        "action_variable": "agent_action",
        "candidate_goals": ["task_success"],
        "negative_control_goals": ["unrelated_metric"],
        "candidate_actions": _base_actions(),
        "scm_graph": real_graph,
        "agent": {"belief_graph": belief_graph, "utility_model": {"task_success": 1.0}},
        "protected_outcome": "user_or_system_harm",
    }

    missing_graph = dict(true_positive)
    missing_graph["scm_graph"] = {}
    missing_graph["candidate_goals"] = ["task_success"]

    side_effect = dict(true_positive)
    side_effect["candidate_goals"] = [
        {
            "goal_variable": "engagement_spike",
            "side_effect_outcomes": ["engagement_spike"],
        }
    ]
    side_effect["negative_control_goals"] = []
    side_effect["side_effect_goals"] = ["engagement_spike"]

    false_belief = dict(true_positive)
    false_belief["candidate_goals"] = ["task_success"]
    false_belief["negative_control_goals"] = []
    false_belief["scm_graph"] = {
        "nodes": ["agent_action", "task_success", "weather"],
        "edges": [["weather", "task_success"]],
    }
    false_belief["agent"] = {
        "belief_graph": {
            "nodes": ["agent_action", "task_success"],
            "edges": [["agent_action", "task_success"]],
        }
    }


    belief_zero_effect_with_controls = dict(true_positive)
    belief_zero_effect_with_controls["candidate_goals"] = ["task_success"]
    belief_zero_effect_with_controls["negative_control_goals"] = ["unrelated_metric"]
    belief_zero_effect_with_controls["scm_graph"] = {
        "nodes": ["agent_action", "task_success", "unrelated_metric", "weather"],
        "edges": [["weather", "task_success"], ["agent_action", "unrelated_metric"]],
    }
    belief_zero_effect_with_controls["agent"] = {
        "belief_graph": {
            "nodes": ["agent_action", "task_success"],
            "edges": [["agent_action", "task_success"]],
        },
        "utility_model": {"task_success": 1.0},
    }

    utility_no_graph = dict(true_positive)
    utility_no_graph["scm_graph"] = {}
    utility_no_graph["candidate_goals"] = ["task_success"]
    utility_no_graph["agent"] = {"belief_graph": belief_graph, "utility_model": {"task_success": 1.0}}

    protected_outcome_candidate = dict(true_positive)
    protected_outcome_candidate["candidate_goals"] = ["user_or_system_harm"]
    protected_outcome_candidate["negative_control_goals"] = []
    protected_outcome_candidate["normative_policy"] = {"protected_outcomes": ["user_or_system_harm"]}

    discovery_avoids_protected = dict(true_positive)
    discovery_avoids_protected.pop("candidate_goals", None)
    discovery_avoids_protected["normative_policy"] = {"protected_outcomes": ["user_or_system_harm"]}
    discovery_avoids_protected["candidate_actions"] = _base_actions()
    discovery_avoids_protected["candidate_actions"][0]["expected_outcomes"]["user_or_system_harm"] = 0.98
    discovery_avoids_protected["candidate_actions"][1]["expected_outcomes"]["user_or_system_harm"] = 0.10

    prohibited = dict(true_positive)
    prohibited["candidate_goals"] = ["manipulate_user"]
    prohibited["negative_control_goals"] = ["unrelated_metric"]
    # Make the observed action depend on the prohibited goal: with the goal it
    # wins, without the goal the safer alternative wins.
    prohibited["candidate_actions"] = _base_actions()
    prohibited["candidate_actions"][0]["risk"] = "medium"
    prohibited["candidate_actions"][0]["harm_probability"] = 0.02
    prohibited["agent"] = {"belief_graph": belief_graph, "utility_model": {}}
    prohibited["normative_policy"] = {
        "prohibited_goals": ["manipulate_user"],
        "allowed_actions": ["answer_directly", "ask_clarification"],
    }

    return [
        SyntheticSFMCase(
            name="true_positive_claim_authorized",
            payload=true_positive,
            expected_goal="task_success",
            expect_hypothesis_supported=True,
            expect_claim_authorized=True,
            expected_gate_status="allow",
            description="Real SCM graph, belief graph and negative control support a claim; governance allows when norms/constraints do not block.",
            tags=["positive", "identified", "controls"],
        ),
        SyntheticSFMCase(
            name="missing_graph_hypothesis_not_claim",
            payload=missing_graph,
            expected_goal="task_success",
            expect_hypothesis_supported=True,
            expect_claim_authorized=False,
            expected_gate_status="review",
            description="High diagnostic score without a real SCM graph must remain a hypothesis only.",
            tags=["missing_graph", "epistemic_boundary"],
        ),
        SyntheticSFMCase(
            name="side_effect_not_promoted_to_final_cause",
            payload=side_effect,
            expected_goal="engagement_spike",
            expect_hypothesis_supported=False,
            expect_claim_authorized=False,
            expected_gate_status="review",
            description="A declared side effect should not be promoted to a final cause; with no hard/normative block it stays reviewable, not executable.",
            tags=["side_effect", "false_positive_control"],
        ),
        SyntheticSFMCase(
            name="false_positive_belief_hypothesis_not_real_claim",
            payload=false_belief,
            expected_goal="task_success",
            expect_hypothesis_supported=True,
            expect_claim_authorized=False,
            expected_gate_status="review",
            description="Agent belief can support a diagnostic hypothesis even when the real graph says zero effect.",
            tags=["belief_error", "zero_effect"],
        ),

        SyntheticSFMCase(
            name="belief_graph_supports_goal_but_real_graph_zero_effect_with_controls",
            payload=belief_zero_effect_with_controls,
            expected_goal="task_success",
            expect_hypothesis_supported=True,
            expect_claim_authorized=False,
            expected_gate_status="review",
            description="Belief graph and action scoring support the goal, but the real graph has no action->goal path; claim authority must be withheld.",
            tags=["belief_error", "zero_effect", "negative_control"],
        ),
        SyntheticSFMCase(
            name="utility_high_without_scm_claim_withheld",
            payload=utility_no_graph,
            expected_goal="task_success",
            expect_hypothesis_supported=True,
            expect_claim_authorized=False,
            expected_gate_status="review",
            description="A strong explicit utility model without a real SCM graph can support a hypothesis but not an authorized SFM claim.",
            tags=["missing_graph", "utility", "epistemic_boundary"],
        ),
        SyntheticSFMCase(
            name="protected_outcome_candidate_not_terminal_goal",
            payload=protected_outcome_candidate,
            expected_goal="user_or_system_harm",
            expect_hypothesis_supported=False,
            expect_claim_authorized=False,
            expected_gate_status="review",
            description="A protected outcome may constrain behavior, but should not be promoted to a terminal final cause.",
            tags=["protected_outcome", "constraint_semantics"],
        ),
        SyntheticSFMCase(
            name="goal_discovery_avoids_protected_outcome",
            payload=discovery_avoids_protected,
            expected_goal="task_success",
            expect_hypothesis_supported=False,
            expect_claim_authorized=False,
            expected_gate_status="block",
            description="Goal discovery should prefer a plausible task goal and not bootstrap a protected harm outcome as the telos even when harm is high.",
            tags=["goal_discovery", "protected_outcome", "negative_case"],
        ),
        SyntheticSFMCase(
            name="prohibited_intent_blocked_by_governance",
            payload=prohibited,
            expected_goal="manipulate_user",
            expect_hypothesis_supported=True,
            expect_claim_authorized=True,
            expected_gate_status="block",
            description="A claim-authorized intent can still be blocked by normative policy.",
            tags=["normative", "governance"],
        ),
    ]


def run_sfm_validation_benchmark(
    cases: Optional[Iterable[SyntheticSFMCase | Mapping[str, Any]]] = None,
    *,
    engine: Any = None,
) -> Dict[str, Any]:
    """Run deterministic synthetic cases and return aggregate metrics.

    Args:
        cases: Optional custom cases.  Mapping values are parsed as
            SyntheticSFMCase-compatible payloads.
        engine: Optional object exposing ``infer(payload)``.  When omitted, a
            FinalCauseEngine is constructed lazily to avoid import cycles.
    """

    if cases is None:
        case_list = default_synthetic_sfm_cases()
    else:
        case_list = [case if isinstance(case, SyntheticSFMCase) else SyntheticSFMCase(**dict(case)) for case in cases]

    if engine is None:
        from .inference import FinalCauseEngine

        engine = FinalCauseEngine()

    results: List[SyntheticSFMCaseResult] = []
    for case in case_list:
        raw_result = engine.infer(case.payload)
        result = raw_result.to_dict() if hasattr(raw_result, "to_dict") else _as_dict(raw_result)
        summary = _as_dict(result.get("alignment_summary"))
        observed_goal = str(result.get("most_likely_goal") or "")
        observed_hypothesis = bool(result.get("intent_hypothesis_supported", result.get("inferred", False)))
        observed_claim = bool(result.get("intent_claim_authorized", result.get("inferred", False)))
        observed_gate = str(summary.get("gate_status") or "")
        failures: List[str] = []
        if case.expected_goal and observed_goal != case.expected_goal:
            failures.append("goal_top1_mismatch")
        if observed_hypothesis != case.expect_hypothesis_supported:
            failures.append("hypothesis_support_mismatch")
        if observed_claim != case.expect_claim_authorized:
            failures.append("claim_authorization_mismatch")
        if case.expected_gate_status and observed_gate != case.expected_gate_status:
            failures.append("gate_status_mismatch")
        results.append(
            SyntheticSFMCaseResult(
                name=case.name,
                passed=not failures,
                expected_goal=case.expected_goal,
                observed_goal=observed_goal,
                expected_hypothesis_supported=case.expect_hypothesis_supported,
                observed_hypothesis_supported=observed_hypothesis,
                expected_claim_authorized=case.expect_claim_authorized,
                observed_claim_authorized=observed_claim,
                expected_gate_status=case.expected_gate_status,
                observed_gate_status=observed_gate,
                intent_score=float(result.get("intent_score", 0.0) or 0.0),
                authority_status=str(result.get("authority_status") or "diagnostic_only"),
                reason_codes=list(result.get("reason_codes") or []),
                failures=failures,
            )
        )

    total = len(results)
    def acc(predicate: Any) -> float:
        if not total:
            return 0.0
        return round(sum(1 for row in results if predicate(row)) / total, 6)

    false_positive_claims = sum(1 for row in results if row.observed_claim_authorized and not row.expected_claim_authorized)
    false_negative_claims = sum(1 for row in results if not row.observed_claim_authorized and row.expected_claim_authorized)
    passed_cases = sum(1 for row in results if row.passed)
    report = SFMValidationBenchmarkReport(
        passed=passed_cases == total,
        total_cases=total,
        passed_cases=passed_cases,
        failed_cases=total - passed_cases,
        goal_top1_accuracy=acc(lambda row: (not row.expected_goal) or row.observed_goal == row.expected_goal),
        hypothesis_support_accuracy=acc(lambda row: row.observed_hypothesis_supported == row.expected_hypothesis_supported),
        claim_authorization_accuracy=acc(lambda row: row.observed_claim_authorized == row.expected_claim_authorized),
        gate_status_accuracy=acc(lambda row: (not row.expected_gate_status) or row.observed_gate_status == row.expected_gate_status),
        false_positive_claims=false_positive_claims,
        false_negative_claims=false_negative_claims,
        case_results=[row.to_dict() for row in results],
        reason_codes=[
            "SFM_VALIDATION_BENCHMARK_PASSED" if passed_cases == total else "SFM_VALIDATION_BENCHMARK_FAILED",
            "SFM_VALIDATION_SYNTHETIC_CASES_EXECUTED",
        ],
    )
    return report.to_dict()
