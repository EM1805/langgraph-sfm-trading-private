from __future__ import annotations

"""SFM Agent Monitor for LangGraph-style agent runs.

The monitor is intentionally dependency-light.  It consumes ``sfm_analysis``
objects produced by :class:`sfm_langgraph.SFMIntentAnalyzerNode` and returns a
partial state update that can be used as a LangGraph node, a post-step hook, or a
plain Python run reporter.

Unlike a classifier dashboard, this monitor keeps the SFM epistemic boundary in
view: supported hypotheses, authorized final-cause claims, claim-withholding
reasons, side-effect risk and deception risk are tracked separately.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

StateMapping = Mapping[str, Any]


@dataclass(frozen=True)
class SFMAgentMonitorConfig:
    """Configuration for the SFM Agent Monitor.

    The defaults match the keys returned by ``SFMIntentAnalyzerNode``.  The
    monitor can be used directly as a LangGraph node because it is callable and
    returns only a partial state update.
    """

    analysis_key: str = "sfm_analysis"
    output_key: str = "sfm_monitor"
    events_key: str = "sfm_monitor_events"
    review_key: str = "requires_human_review"
    gate_key: str = "sfm_gate_status"
    run_id_key: str = "run_id"
    default_run_id: str = "sfm-agent-run"
    source: str = "sfm_langgraph.SFMAgentMonitor"
    review_on_medium_deception: bool = True
    review_on_high_side_effect: bool = True
    review_on_claim_withheld: bool = True
    block_on_node_failure: bool = True
    block_on_high_deception_with_authorized_claim: bool = False
    max_events: int = 500


@dataclass(frozen=True)
class SFMRiskEvent:
    """One monitor event derived from an SFM analysis object."""

    event_id: str
    step_index: int
    event_type: str
    observed_action: str = ""
    primary_intent: str = ""
    intent_score: float = 0.0
    claim_level: str = "diagnostic_only"
    intent_hypothesis_supported: bool = False
    intent_claim_authorized: bool = False
    deception_risk: str = "unknown"
    side_effect_risk: str = "unknown"
    gate_status: str = "review"
    risk_score: float = 0.0
    requires_human_review: bool = False
    blocked_claim_reason: str = ""
    rationale: str = ""
    reason_codes: List[str] = field(default_factory=list)
    limits: List[str] = field(default_factory=list)
    intended_effects: List[str] = field(default_factory=list)
    side_effects: List[str] = field(default_factory=list)
    source: str = "sfm_langgraph.SFMAgentMonitor"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SFMRunReport:
    """Governance-facing report over all SFM monitor events in one run."""

    run_id: str
    total_events: int
    final_gate_status: str
    human_review_required: bool
    max_risk_score: float = 0.0
    mean_risk_score: float = 0.0
    high_risk_events: int = 0
    deception_alerts: int = 0
    side_effect_alerts: int = 0
    claim_withheld_events: int = 0
    authorized_claim_events: int = 0
    blocked_events: int = 0
    review_events: int = 0
    claim_level_counts: Dict[str, int] = field(default_factory=dict)
    primary_intent_counts: Dict[str, int] = field(default_factory=dict)
    intent_timeline: List[Dict[str, Any]] = field(default_factory=list)
    risk_events: List[Dict[str, Any]] = field(default_factory=list)
    reason_codes: List[str] = field(default_factory=list)
    limits: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _clean_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if out == out else default


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _count_by(values: Iterable[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for value in values:
        key = _clean_str(value, "unknown")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _risk_score(analysis: Mapping[str, Any]) -> float:
    score = 0.0
    deception = _clean_str(analysis.get("deception_risk"), "unknown").lower()
    side_effect = _clean_str(analysis.get("side_effect_risk"), "unknown").lower()
    gate = _clean_str(analysis.get("gate_status"), "review").lower()
    blocked_reason = _clean_str(analysis.get("blocked_claim_reason"))
    claim_authorized = bool(analysis.get("intent_claim_authorized"))
    hypothesis_supported = bool(analysis.get("intent_hypothesis_supported"))

    if deception == "high":
        score += 0.50
    elif deception == "medium":
        score += 0.30
    elif deception == "unknown":
        score += 0.05

    if side_effect == "high":
        score += 0.35
    elif side_effect == "monitored":
        score += 0.15
    elif side_effect == "unknown":
        score += 0.05

    if gate == "block":
        score += 0.40
    elif gate == "review":
        score += 0.15

    if hypothesis_supported and not claim_authorized and blocked_reason:
        score += 0.15

    for code in _as_list(analysis.get("reason_codes")):
        text = str(code).upper()
        if "FAILED_CLOSED" in text or "PROTECTED" in text or "FALSIFICATION_FAILED" in text:
            score += 0.20
            break
    return round(_clip01(score), 6)


def _event_type(analysis: Mapping[str, Any], risk_score: float) -> str:
    gate = _clean_str(analysis.get("gate_status"), "review").lower()
    deception = _clean_str(analysis.get("deception_risk"), "unknown").lower()
    side_effect = _clean_str(analysis.get("side_effect_risk"), "unknown").lower()
    if gate == "block":
        return "blocked"
    if deception == "high":
        return "deception_alert"
    if side_effect == "high":
        return "side_effect_alert"
    if bool(analysis.get("intent_hypothesis_supported")) and not bool(analysis.get("intent_claim_authorized")):
        return "claim_withheld"
    if risk_score >= 0.50:
        return "risk_alert"
    if gate == "review":
        return "review"
    return "allow"


def _requires_review(analysis: Mapping[str, Any], event_type: str, config: SFMAgentMonitorConfig) -> bool:
    gate = _clean_str(analysis.get("gate_status"), "review").lower()
    deception = _clean_str(analysis.get("deception_risk"), "unknown").lower()
    side_effect = _clean_str(analysis.get("side_effect_risk"), "unknown").lower()
    reason_codes = "|".join(str(code).upper() for code in _as_list(analysis.get("reason_codes")))
    claim_withheld = bool(analysis.get("intent_hypothesis_supported")) and not bool(analysis.get("intent_claim_authorized"))

    if gate in {"block", "review"}:
        return True
    if deception == "high":
        return True
    if config.review_on_medium_deception and deception == "medium":
        return True
    if config.review_on_high_side_effect and side_effect == "high":
        return True
    if config.review_on_claim_withheld and claim_withheld:
        return True
    if config.block_on_node_failure and "SFM_LANGGRAPH_NODE_FAILED_CLOSED" in reason_codes:
        return True
    return event_type in {"blocked", "deception_alert", "side_effect_alert", "risk_alert"}


def _to_event(analysis: Mapping[str, Any], *, step_index: int, run_id: str, config: SFMAgentMonitorConfig) -> SFMRiskEvent:
    risk = _risk_score(analysis)
    event_type = _event_type(analysis, risk)
    requires_review = _requires_review(analysis, event_type, config)
    return SFMRiskEvent(
        event_id=f"{run_id}:{step_index}",
        step_index=step_index,
        event_type=event_type,
        observed_action=_clean_str(analysis.get("observed_action")),
        primary_intent=_clean_str(analysis.get("primary_intent")),
        intent_score=round(_safe_float(analysis.get("intentionality_score")), 6),
        claim_level=_clean_str(analysis.get("final_cause_claim_level"), "diagnostic_only"),
        intent_hypothesis_supported=bool(analysis.get("intent_hypothesis_supported")),
        intent_claim_authorized=bool(analysis.get("intent_claim_authorized")),
        deception_risk=_clean_str(analysis.get("deception_risk"), "unknown"),
        side_effect_risk=_clean_str(analysis.get("side_effect_risk"), "unknown"),
        gate_status=_clean_str(analysis.get("gate_status"), "review"),
        risk_score=risk,
        requires_human_review=requires_review,
        blocked_claim_reason=_clean_str(analysis.get("blocked_claim_reason")),
        rationale=_clean_str(analysis.get("deception_rationale")),
        reason_codes=[str(code) for code in _as_list(analysis.get("reason_codes"))],
        limits=[str(limit) for limit in _as_list(analysis.get("limits"))],
        intended_effects=[str(item) for item in _as_list(analysis.get("intended_effects"))],
        side_effects=[str(item) for item in _as_list(analysis.get("side_effects"))],
        source=config.source,
    )


def _final_gate(events: Sequence[Mapping[str, Any]], config: SFMAgentMonitorConfig) -> str:
    if not events:
        return "review"
    if any(_clean_str(event.get("gate_status"), "review").lower() == "block" for event in events):
        return "block"
    if config.block_on_node_failure and any(
        "SFM_LANGGRAPH_NODE_FAILED_CLOSED" in "|".join(str(code).upper() for code in _as_list(event.get("reason_codes")))
        for event in events
    ):
        return "block"
    if config.block_on_high_deception_with_authorized_claim and any(
        _clean_str(event.get("deception_risk"), "unknown").lower() == "high"
        and bool(event.get("intent_claim_authorized"))
        for event in events
    ):
        return "block"
    if any(bool(event.get("requires_human_review")) for event in events):
        return "review"
    return "allow"


def build_sfm_run_report(
    events: Sequence[Mapping[str, Any]],
    *,
    run_id: str = "sfm-agent-run",
    config: Optional[SFMAgentMonitorConfig] = None,
) -> Dict[str, Any]:
    """Build a serializable run report from SFM monitor events."""

    cfg = config or SFMAgentMonitorConfig()
    materialized = [dict(event) for event in events][-cfg.max_events :]
    total = len(materialized)
    scores = [_safe_float(event.get("risk_score")) for event in materialized]
    max_score = round(max(scores), 6) if scores else 0.0
    mean_score = round(sum(scores) / len(scores), 6) if scores else 0.0
    final_gate = _final_gate(materialized, cfg)
    human_review_required = final_gate in {"block", "review"} or any(
        bool(event.get("requires_human_review")) for event in materialized
    )
    timeline = [
        {
            "step_index": event.get("step_index"),
            "observed_action": event.get("observed_action", ""),
            "primary_intent": event.get("primary_intent", ""),
            "claim_level": event.get("claim_level", "diagnostic_only"),
            "risk_score": event.get("risk_score", 0.0),
            "event_type": event.get("event_type", "review"),
        }
        for event in materialized
    ]
    reason_codes = sorted({str(code) for event in materialized for code in _as_list(event.get("reason_codes"))})
    limits = sorted({str(limit) for event in materialized for limit in _as_list(event.get("limits"))})
    report = SFMRunReport(
        run_id=run_id,
        total_events=total,
        final_gate_status=final_gate,
        human_review_required=human_review_required,
        max_risk_score=max_score,
        mean_risk_score=mean_score,
        high_risk_events=sum(1 for event in materialized if _safe_float(event.get("risk_score")) >= 0.50),
        deception_alerts=sum(1 for event in materialized if _clean_str(event.get("deception_risk"), "unknown").lower() == "high"),
        side_effect_alerts=sum(1 for event in materialized if _clean_str(event.get("side_effect_risk"), "unknown").lower() == "high"),
        claim_withheld_events=sum(
            1
            for event in materialized
            if bool(event.get("intent_hypothesis_supported")) and not bool(event.get("intent_claim_authorized"))
        ),
        authorized_claim_events=sum(1 for event in materialized if bool(event.get("intent_claim_authorized"))),
        blocked_events=sum(1 for event in materialized if _clean_str(event.get("gate_status"), "review").lower() == "block"),
        review_events=sum(1 for event in materialized if bool(event.get("requires_human_review"))),
        claim_level_counts=_count_by(_clean_str(event.get("claim_level"), "diagnostic_only") for event in materialized),
        primary_intent_counts=_count_by(_clean_str(event.get("primary_intent"), "unknown") for event in materialized),
        intent_timeline=timeline,
        risk_events=materialized,
        reason_codes=reason_codes,
        limits=limits,
    )
    return report.to_dict()


class SFMAgentMonitor:
    """Callable LangGraph-compatible monitor for SFM intent analyses.

    Typical graph shape:

    ``agent_step -> sfm_intent_analyzer -> sfm_agent_monitor -> conditional_edge``

    The monitor consumes the current ``sfm_analysis`` state key, appends an event
    to ``sfm_monitor_events``, and returns a run-level ``sfm_monitor`` report plus
    top-level routing keys such as ``requires_human_review`` and
    ``sfm_gate_status``.
    """

    def __init__(self, config: Optional[SFMAgentMonitorConfig] = None) -> None:
        self.config = config or SFMAgentMonitorConfig()

    def event_from_state(self, state: StateMapping) -> SFMRiskEvent:
        analysis = _as_dict(state.get(self.config.analysis_key))
        run_id = _clean_str(state.get(self.config.run_id_key), self.config.default_run_id)
        prior = _as_list(state.get(self.config.events_key))[-self.config.max_events :]
        return _to_event(analysis, step_index=len(prior) + 1, run_id=run_id, config=self.config)

    def report(self, events: Sequence[Mapping[str, Any]], *, run_id: str = "") -> Dict[str, Any]:
        return build_sfm_run_report(
            events,
            run_id=_clean_str(run_id, self.config.default_run_id),
            config=self.config,
        )

    def __call__(self, state: StateMapping) -> Dict[str, Any]:
        run_id = _clean_str(state.get(self.config.run_id_key), self.config.default_run_id)
        prior = [dict(item) for item in _as_list(state.get(self.config.events_key))][-self.config.max_events :]
        event = self.event_from_state(state).to_dict()
        events = [*prior, event][-self.config.max_events :]
        report = self.report(events, run_id=run_id)
        return {
            self.config.events_key: events,
            self.config.output_key: report,
            self.config.review_key: bool(report.get("human_review_required")),
            self.config.gate_key: _clean_str(report.get("final_gate_status"), "review"),
        }


def build_sfm_agent_monitor(
    *,
    analysis_key: str = "sfm_analysis",
    output_key: str = "sfm_monitor",
    events_key: str = "sfm_monitor_events",
) -> SFMAgentMonitor:
    """Factory helper for examples and app configuration."""

    return SFMAgentMonitor(
        SFMAgentMonitorConfig(
            analysis_key=analysis_key,
            output_key=output_key,
            events_key=events_key,
        )
    )


def add_sfm_agent_monitor_node(
    graph_builder: Any,
    *,
    name: str = "sfm_agent_monitor",
    node: Optional[SFMAgentMonitor] = None,
    **config_overrides: Any,
) -> SFMAgentMonitor:
    """Attach the monitor to a LangGraph-like builder and return it."""

    if node is None:
        config = SFMAgentMonitorConfig(**config_overrides) if config_overrides else SFMAgentMonitorConfig()
        node = SFMAgentMonitor(config)
    graph_builder.add_node(name, node)
    return node
