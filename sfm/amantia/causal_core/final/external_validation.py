from __future__ import annotations

"""Panel-backed validation harness for Structural Final Models.

The synthetic benchmark in :mod:`validation_benchmark` protects SFM against
known epistemic failure modes.  This module adds a second, data-backed harness:
it converts a longitudinal action-event panel into deterministic SFM cases and
checks whether claim authority changes in the right direction under ablations.

The harness is conservative by design.  It does not treat panel correlations as
scientific proof of final causality.  Instead, it verifies three properties that
matter before using SFM on real operational logs:

* a domain graph plus repeated decision records can authorize a final-cause
  hypothesis when the observed action is goal-dependent;
* removing the real SCM graph withholds claim authority while preserving a
  diagnostic hypothesis;
* protected/side-effect outcomes and negative controls are not promoted to
  terminal final causes.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .inference import infer_final_cause


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _clean_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return default
    return text if text else default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if out == out else default


def _clip01(value: Any, default: float = 0.0) -> float:
    return max(0.0, min(1.0, _safe_float(value, default)))


def _mean(values: Iterable[float], default: float = 0.0) -> float:
    vals = [float(v) for v in values]
    if not vals:
        return default
    return sum(vals) / len(vals)


def _find_default_panel_path() -> Optional[Path]:
    """Find the bundled development panel when callers do not pass a path."""

    candidates = [
        Path.cwd() / "data" / "action_event_panel.csv",
        Path.cwd() / "action_event_panel.csv",
    ]
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        candidates.append(parent / "data" / "action_event_panel.csv")
        candidates.append(parent.parent / "data" / "action_event_panel.csv")
    for path in candidates:
        if path.exists():
            return path
    return None


def _import_pandas():
    try:
        import pandas as pd  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency contract guard
        raise RuntimeError("run_sfm_external_panel_benchmark requires pandas to read CSV panels") from exc
    return pd


@dataclass
class ExternalSFMPanelCase:
    """One panel-backed SFM validation case."""

    name: str
    payload: Dict[str, Any]
    expected_goal: str = ""
    expect_hypothesis_supported: bool = False
    expect_claim_authorized: bool = False
    expected_gate_status: str = ""
    description: str = ""
    tags: List[str] = field(default_factory=list)
    panel_summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExternalSFMPanelCaseResult:
    """Observed-vs-expected result for one panel-backed case."""

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
class ExternalSFMPanelBenchmarkReport:
    """Aggregate report over a longitudinal, panel-backed SFM benchmark."""

    benchmark_name: str = "external_sfm_panel_epistemic_safety_v1"
    source_path: str = ""
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
    panel_summary: Dict[str, Any] = field(default_factory=dict)
    reason_codes: List[str] = field(default_factory=list)
    limits: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _boolean_series(df: Any, column: str, default: bool = False) -> Any:
    if column not in df.columns:
        return df.index.to_series().map(lambda _: default)
    series = df[column]
    if str(series.dtype) == "bool":
        return series.fillna(default).astype(bool)
    return series.map(lambda value: str(value).strip().lower() in {"1", "true", "yes", "y", "peer"})


def _incident_rate(df: Any) -> float:
    if len(df) == 0:
        return 0.0
    if "incident_detected" in df.columns:
        return round(float(_boolean_series(df, "incident_detected").mean()), 6)
    if "outcome" in df.columns:
        return round(float(df["outcome"].astype(str).str.lower().eq("incident").mean()), 6)
    return 0.0


def _panel_summary(df: Any, source_path: str = "") -> Dict[str, Any]:
    review_gate = _boolean_series(df, "approval_present")
    if "review_type" in df.columns:
        review_gate = review_gate | df["review_type"].astype(str).str.lower().eq("peer")
    direct = df.loc[~review_gate]
    review = df.loc[review_gate]
    action_counts = {}
    if "action_name" in df.columns:
        action_counts = {str(k): int(v) for k, v in df["action_name"].value_counts().to_dict().items()}
    return {
        "source_path": source_path,
        "records": int(len(df)),
        "review_gate_records": int(len(review)),
        "direct_execute_records": int(len(direct)),
        "incident_rate_review_gate": _incident_rate(review),
        "incident_rate_direct_execute": _incident_rate(direct),
        "action_counts": action_counts,
    }


def _panel_records(df: Any, *, max_records: int = 80) -> List[Dict[str, Any]]:
    """Convert rows into policy-learning records with candidate options.

    The conversion keeps only a bounded sample to make CI deterministic and fast.
    Rows are sorted by event time when the column is available.
    """

    if "event_time" in df.columns:
        df = df.sort_values("event_time")
    summary = _panel_summary(df)
    review_safe = 1.0 - float(summary["incident_rate_review_gate"])
    direct_safe = 1.0 - float(summary["incident_rate_direct_execute"])
    # The panel records an operational action together with whether a peer or
    # approval gate was used.  We model that gate as the policy choice to test
    # repeated goal consistency, not as a causal estimate by itself.
    records: List[Dict[str, Any]] = []
    for idx, row in df.head(max_records).iterrows():
        has_review = bool(_boolean_series(df.loc[[idx]], "approval_present").iloc[0])
        if "review_type" in df.columns:
            has_review = has_review or str(row.get("review_type", "")).strip().lower() == "peer"
        selected = "peer_review_gate" if has_review else "direct_execute"
        records.append(
            {
                "record_id": _clean_str(row.get("event_id"), str(idx)),
                "time": _clean_str(row.get("event_time")),
                "selected_action": selected,
                "context": {
                    "environment": _clean_str(row.get("environment")),
                    "action_name": _clean_str(row.get("action_name")),
                    "resource_sensitivity": _clean_str(row.get("resource_sensitivity")),
                },
                "candidate_actions": [
                    {
                        "action": "direct_execute",
                        "expected_success": 0.92,
                        "expected_outcomes": {
                            "task_throughput": 0.92,
                            "operational_latency": 0.10,
                            "incident_avoidance": direct_safe,
                            "audit_noise": 0.50,
                        },
                        "risk": "medium",
                        "harm_probability": min(0.08, float(summary["incident_rate_direct_execute"]) / 8.0),
                    },
                    {
                        "action": "peer_review_gate",
                        "expected_success": 0.68,
                        "expected_outcomes": {
                            "task_throughput": 0.68,
                            "operational_latency": 0.35,
                            "incident_avoidance": review_safe,
                            "audit_noise": 0.50,
                        },
                        "risk": "low",
                        "harm_probability": min(0.08, float(summary["incident_rate_review_gate"]) / 8.0),
                    },
                ],
            }
        )
    return records


def _base_payload(summary: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    direct_incident = float(summary.get("incident_rate_direct_execute") or 0.0)
    review_incident = float(summary.get("incident_rate_review_gate") or 0.0)
    direct_option = {
        "action": "direct_execute",
        "expected_success": 0.92,
        "expected_outcomes": {
            "task_throughput": 0.92,
            "operational_latency": 0.10,
            "incident_avoidance": round(1.0 - direct_incident, 6),
            "audit_noise": 0.50,
        },
        "risk": "medium",
        "harm_probability": min(0.08, direct_incident / 8.0),
        "evidence_quality": "medium",
        "uncertainty": 0.08,
    }
    review_option = {
        "action": "peer_review_gate",
        "expected_success": 0.68,
        "expected_outcomes": {
            "task_throughput": 0.68,
            "operational_latency": 0.35,
            "incident_avoidance": round(1.0 - review_incident, 6),
            "audit_noise": 0.50,
        },
        "risk": "low",
        "harm_probability": min(0.08, review_incident / 8.0),
        "evidence_quality": "medium",
        "uncertainty": 0.08,
    }
    return {
        "observed_action": "direct_execute",
        "action_variable": "agent_action",
        "candidate_actions": [direct_option, review_option],
        "negative_control_goals": ["audit_noise"],
        "decision_records": [dict(record) for record in records],
        "min_policy_records": 8,
        "min_temporal_window_records": 8,
        "temporal_window_size": 16,
        "scm_graph": {
            "nodes": [
                "agent_action",
                "task_throughput",
                "operational_latency",
                "incident_avoidance",
                "audit_noise",
            ],
            "edges": [
                ["agent_action", "task_throughput"],
                ["agent_action", "operational_latency"],
                ["agent_action", "incident_avoidance"],
            ],
        },
        "agent": {
            "belief_graph": {
                "nodes": ["agent_action", "task_throughput", "operational_latency", "incident_avoidance"],
                "edges": [["agent_action", "task_throughput"], ["agent_action", "operational_latency"]],
            },
            "utility_model": {"task_throughput": 1.0, "incident_avoidance": 0.25},
        },
        "protected_outcome": "user_or_system_harm",
        "source": "external_validation.action_event_panel",
        "raw_panel_summary": dict(summary),
    }


def build_external_sfm_panel_cases(path: Optional[str] = None) -> List[ExternalSFMPanelCase]:
    """Build deterministic SFM validation cases from an action-event panel CSV.

    The default adapter expects the development ``action_event_panel.csv``
    schema, but the generated payloads are ordinary ``infer_final_cause``
    inputs and can be replaced with a domain-specific adapter for production
    datasets.
    """

    panel_path = Path(path) if path else _find_default_panel_path()
    if panel_path is None:
        return []
    pd = _import_pandas()
    df = pd.read_csv(panel_path)
    summary = _panel_summary(df, str(panel_path))
    records = _panel_records(df)
    base = _base_payload(summary, records)

    throughput_claim = dict(base)
    throughput_claim["candidate_goals"] = ["task_throughput"]

    missing_graph = dict(throughput_claim)
    missing_graph["scm_graph"] = {}

    negative_control = dict(base)
    negative_control["candidate_goals"] = ["audit_noise"]
    negative_control["negative_control_goals"] = []

    protected_outcome = dict(base)
    protected_outcome["candidate_goals"] = ["user_or_system_harm"]
    protected_outcome["candidate_actions"] = [
        {
            **dict(base["candidate_actions"][0]),
            "expected_outcomes": {**dict(base["candidate_actions"][0]["expected_outcomes"]), "user_or_system_harm": min(0.08, float(summary["incident_rate_direct_execute"]) / 8.0)},
        },
        {
            **dict(base["candidate_actions"][1]),
            "expected_outcomes": {**dict(base["candidate_actions"][1]["expected_outcomes"]), "user_or_system_harm": min(0.08, float(summary["incident_rate_review_gate"]) / 8.0)},
        },
    ]
    protected_outcome["scm_graph"] = {
        "nodes": ["agent_action", "user_or_system_harm", "audit_noise"],
        "edges": [["agent_action", "user_or_system_harm"]],
    }
    protected_outcome["agent"] = {
        "belief_graph": {
            "nodes": ["agent_action", "user_or_system_harm"],
            "edges": [["agent_action", "user_or_system_harm"]],
        },
        "utility_model": {"user_or_system_harm": 1.0},
    }

    return [
        ExternalSFMPanelCase(
            name="panel_throughput_claim_authorized_with_domain_graph",
            payload=throughput_claim,
            expected_goal="task_throughput",
            expect_hypothesis_supported=True,
            expect_claim_authorized=True,
            expected_gate_status="allow",
            description=(
                "Longitudinal panel records plus a supplied domain graph authorize a throughput-oriented "
                "final-cause claim when the observed action changes under goal removal."
            ),
            tags=["panel", "longitudinal", "positive", "domain_graph"],
            panel_summary=summary,
        ),
        ExternalSFMPanelCase(
            name="panel_missing_graph_withholds_claim_authority",
            payload=missing_graph,
            expected_goal="task_throughput",
            expect_hypothesis_supported=True,
            expect_claim_authorized=False,
            expected_gate_status="review",
            description="The same panel evidence remains diagnostic only when the real SCM graph is ablated.",
            tags=["panel", "graph_ablation", "epistemic_boundary"],
            panel_summary=summary,
        ),
        ExternalSFMPanelCase(
            name="panel_negative_control_not_promoted_to_final_cause",
            payload=negative_control,
            expected_goal="audit_noise",
            expect_hypothesis_supported=True,
            expect_claim_authorized=False,
            expected_gate_status="review",
            description="A panel negative-control outcome may remain diagnostic under action scoring, but must not become an authorized terminal goal.",
            tags=["panel", "negative_control", "false_positive_control"],
            panel_summary=summary,
        ),
        ExternalSFMPanelCase(
            name="panel_protected_harm_not_terminal_goal",
            payload=protected_outcome,
            expected_goal="user_or_system_harm",
            expect_hypothesis_supported=False,
            expect_claim_authorized=False,
            expected_gate_status="review",
            description="Observed harmful side effects in the panel are not promoted to a terminal final cause.",
            tags=["panel", "protected_outcome", "false_positive_control"],
            panel_summary=summary,
        ),
    ]


def _evaluate_case(case: ExternalSFMPanelCase) -> ExternalSFMPanelCaseResult:
    result = infer_final_cause(case.payload)
    observed_goal = str(result.get("most_likely_goal") or "")
    observed_hypothesis_supported = bool(result.get("intent_hypothesis_supported"))
    observed_claim_authorized = bool(result.get("intent_claim_authorized"))
    alignment = _as_dict(result.get("alignment_summary"))
    observed_gate_status = str(alignment.get("gate_status") or "")
    failures: List[str] = []
    if case.expected_goal and observed_goal != case.expected_goal:
        failures.append(f"expected_goal={case.expected_goal!r} observed_goal={observed_goal!r}")
    if observed_hypothesis_supported != case.expect_hypothesis_supported:
        failures.append(
            f"expected_hypothesis_supported={case.expect_hypothesis_supported!r} "
            f"observed={observed_hypothesis_supported!r}"
        )
    if observed_claim_authorized != case.expect_claim_authorized:
        failures.append(
            f"expected_claim_authorized={case.expect_claim_authorized!r} "
            f"observed={observed_claim_authorized!r}"
        )
    if case.expected_gate_status and observed_gate_status != case.expected_gate_status:
        failures.append(f"expected_gate_status={case.expected_gate_status!r} observed={observed_gate_status!r}")
    return ExternalSFMPanelCaseResult(
        name=case.name,
        passed=not failures,
        expected_goal=case.expected_goal,
        observed_goal=observed_goal,
        expected_hypothesis_supported=case.expect_hypothesis_supported,
        observed_hypothesis_supported=observed_hypothesis_supported,
        expected_claim_authorized=case.expect_claim_authorized,
        observed_claim_authorized=observed_claim_authorized,
        expected_gate_status=case.expected_gate_status,
        observed_gate_status=observed_gate_status,
        intent_score=float(result.get("intent_score") or 0.0),
        authority_status=str(result.get("authority_status") or "diagnostic_only"),
        reason_codes=list(result.get("reason_codes") or []),
        failures=failures,
    )


def run_sfm_external_panel_benchmark(path: Optional[str] = None) -> Dict[str, Any]:
    """Run panel-backed SFM validation and return a serializable report."""

    cases = build_external_sfm_panel_cases(path)
    if not cases:
        return ExternalSFMPanelBenchmarkReport(
            passed=False,
            reason_codes=["SFM_EXTERNAL_PANEL_BENCHMARK_NO_PANEL_FOUND"],
            limits=["action_event_panel_csv_required"],
        ).to_dict()
    results = [_evaluate_case(case) for case in cases]
    total = len(results)

    def acc(predicate: Any) -> float:
        if not total:
            return 0.0
        return round(sum(1 for row in results if predicate(row)) / total, 6)

    false_positive_claims = sum(1 for row in results if row.observed_claim_authorized and not row.expected_claim_authorized)
    false_negative_claims = sum(1 for row in results if not row.observed_claim_authorized and row.expected_claim_authorized)
    passed_cases = sum(1 for row in results if row.passed)
    first_summary = cases[0].panel_summary if cases else {}
    report = ExternalSFMPanelBenchmarkReport(
        source_path=str(first_summary.get("source_path") or path or ""),
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
        panel_summary=dict(first_summary),
        reason_codes=[
            "SFM_EXTERNAL_PANEL_BENCHMARK_PASSED" if passed_cases == total else "SFM_EXTERNAL_PANEL_BENCHMARK_FAILED",
            "SFM_EXTERNAL_PANEL_CASES_EXECUTED",
            "SFM_EXTERNAL_PANEL_INCLUDES_GRAPH_ABLATION",
            "SFM_EXTERNAL_PANEL_INCLUDES_NEGATIVE_CONTROL",
            "SFM_EXTERNAL_PANEL_INCLUDES_PROTECTED_OUTCOME_CONTROL",
        ],
    )
    return report.to_dict()
