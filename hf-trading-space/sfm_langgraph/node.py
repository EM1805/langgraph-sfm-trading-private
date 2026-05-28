from __future__ import annotations

"""LangGraph-compatible SFM intent analysis node.

The node is intentionally dependency-light: LangGraph treats nodes as callables
that receive a state object and return a partial state update, so this module can
be imported and tested without installing ``langgraph``.  When LangGraph is
installed, pass ``SFMIntentAnalyzerNode(...)`` directly to ``StateGraph.add_node``.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from sfm import infer_final_cause_compact

StateMapping = Mapping[str, Any]
AnalyzerFn = Callable[[Dict[str, Any]], Dict[str, Any]]


@dataclass(frozen=True)
class SFMIntentAnalyzerConfig:
    """Configuration for extracting SFM inputs from a LangGraph state.

    ``query_key`` lets callers pass a fully formed ``FinalCauseQuery`` payload in
    the graph state.  Otherwise the node builds a query from common state keys
    such as ``observed_action``, ``candidate_actions``, ``candidate_goals`` and
    ``scm_graph``.
    """

    query_key: str = "sfm_query"
    output_key: str = "sfm_analysis"
    trace_key: str = "sfm_trace_events"
    append_trace: bool = True
    compact: bool = True
    default_action_variable: str = "agent_action"
    default_protected_outcome: str = "user_or_system_harm"
    default_min_intent_score: float = 0.6
    include_raw_result: bool = False
    fail_closed: bool = True
    source: str = "sfm_langgraph.SFMIntentAnalyzerNode"
    state_action_keys: Sequence[str] = (
        "observed_action",
        "selected_action",
        "last_action",
        "action",
        "agent_action",
        "tool_name",
    )
    stated_goal_keys: Sequence[str] = (
        "stated_goal",
        "declared_goal",
        "declared_intent",
        "requested_goal",
        "user_goal",
    )


@dataclass(frozen=True)
class SFMNodeAnalysis:
    """Governance-facing SFM analysis returned into LangGraph state."""

    primary_intent: str = ""
    intentionality_score: float = 0.0
    final_cause_claim_level: str = "diagnostic_only"
    intent_hypothesis_supported: bool = False
    intent_claim_authorized: bool = False
    governance_execution_allowed: bool = False
    intended_effects: List[str] = field(default_factory=list)
    side_effects: List[str] = field(default_factory=list)
    side_effect_risk: str = "unknown"
    deception_risk: str = "unknown"
    deception_rationale: str = ""
    observed_action: str = ""
    recommended_action: str = ""
    recommendation_status: str = "unassessed"
    robustness_status: str = "unassessed"
    authority_status: str = "diagnostic_only"
    gate_status: str = "review"
    blocked_claim_reason: str = ""
    reason_codes: List[str] = field(default_factory=list)
    limits: List[str] = field(default_factory=list)
    raw_result: Dict[str, Any] = field(default_factory=dict)

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


def _first_text(state: StateMapping, keys: Iterable[str]) -> str:
    for key in keys:
        text = _clean_str(state.get(key))
        if text:
            return text
    return ""


def _goal_name(value: Any) -> str:
    if isinstance(value, str):
        return _clean_str(value)
    data = _as_dict(value)
    return _clean_str(data.get("goal_variable") or data.get("outcome") or data.get("name") or data.get("goal"))


def _extract_side_effects(query: Mapping[str, Any], result: Mapping[str, Any]) -> List[str]:
    effects = set()
    protected = _clean_str(query.get("protected_outcome"))
    if protected:
        effects.add(protected)
    for key in ("candidate_goals", "goals", "side_effect_goals", "side_effect_outcomes"):
        for item in _as_list(query.get(key)):
            data = _as_dict(item)
            if isinstance(item, str):
                if key in {"side_effect_goals", "side_effect_outcomes"}:
                    effects.add(item)
                continue
            for field_name in ("side_effect_outcomes", "protected_outcomes"):
                for effect in _as_list(data.get(field_name)):
                    text = _clean_str(effect)
                    if text:
                        effects.add(text)
    alignment = _as_dict(result.get("alignment_summary"))
    for warning in _as_list(alignment.get("warnings")):
        text = _clean_str(warning)
        if text and "side" in text.lower():
            effects.add(text)
    primary = _clean_str(result.get("most_likely_goal"))
    return sorted(effect for effect in effects if effect and effect != primary)


def _side_effect_risk(result: Mapping[str, Any], side_effects: Sequence[str]) -> str:
    codes = "|".join(str(code) for code in _as_list(result.get("reason_codes"))).upper()
    limits = "|".join(str(limit) for limit in _as_list(result.get("limits"))).upper()
    if (
        "GOAL_OVERLAPS_PROTECTED" in codes
        or "CANDIDATE_IS_PROTECTED" in codes
        or "PROTECTED_OUTCOME_NOT_TERMINAL" in codes
        or "PROTECTED" in limits
    ):
        return "high"
    if side_effects:
        return "monitored"
    if bool(result.get("intent_claim_authorized")):
        return "low"
    return "unknown"


def _claim_level(result: Mapping[str, Any]) -> str:
    if bool(result.get("intent_claim_authorized")):
        return _clean_str(result.get("authority_status"), "partially_identifiable")
    if bool(result.get("intent_hypothesis_supported")):
        return "falsifiable_diagnostic"
    return "diagnostic_only"


def _blocked_claim_reason(result: Mapping[str, Any]) -> str:
    if bool(result.get("intent_claim_authorized")):
        return ""
    codes = [str(code) for code in _as_list(result.get("reason_codes"))]
    limits = [str(limit) for limit in _as_list(result.get("limits"))]
    joined = "|".join(codes + limits).upper()
    if (
        "MISSING_SCM_GRAPH" in joined
        or "MISSING_GRAPH" in joined
        or "REAL_GRAPH_MISSING" in joined
        or "REAL_SCM_GRAPH_NOT_SUPPLIED" in joined
        or "SCM_GRAPH_NOT_SUPPLIED" in joined
        or "REAL_SCM_GRAPH_REQUIRED" in joined
    ):
        return "missing validated SCM graph"
    if "INDEPENDENT_VALIDATION" in joined:
        return "missing independent validation channel"
    if "FALSIFICATION" in joined and "FAILED" in joined:
        return "falsification failed"
    if "PROTECTED" in joined or "SIDE_EFFECT" in joined:
        return "candidate goal overlaps protected outcome or side effect"
    if not bool(result.get("intent_hypothesis_supported")):
        return "intent hypothesis not supported above threshold"
    return "SFM identifiability layer withheld claim authority"


def _deception_assessment(state: StateMapping, result: Mapping[str, Any], config: SFMIntentAnalyzerConfig) -> tuple[str, str]:
    stated = _first_text(state, config.stated_goal_keys)
    primary = _clean_str(result.get("most_likely_goal"))
    hypothesis_supported = bool(result.get("intent_hypothesis_supported"))
    claim_authorized = bool(result.get("intent_claim_authorized"))
    if not stated:
        return ("unknown" if not hypothesis_supported else "low", "no stated goal found in graph state")
    if not primary:
        return "unknown", f"stated goal '{stated}' present, but SFM found no primary intent"
    if stated == primary:
        return "low", "stated goal matches the SFM primary-intent candidate"
    if claim_authorized:
        return "high", f"stated goal '{stated}' differs from authorized SFM intent '{primary}'"
    if hypothesis_supported:
        return "medium", f"stated goal '{stated}' differs from diagnostic SFM intent '{primary}'"
    return "low", "SFM did not support an alternative intent hypothesis"


def _gate_status(result: Mapping[str, Any]) -> str:
    alignment = _as_dict(result.get("alignment_summary"))
    return _clean_str(alignment.get("gate_status"), "allow" if result.get("governance_execution_allowed") else "review")


class SFMIntentAnalyzerNode:
    """Callable node for LangGraph agentic workflows.

    Example:
        ``builder.add_node("sfm_intent", SFMIntentAnalyzerNode())``

    The returned update is small and governance-facing; full SFM output can be
    included by setting ``include_raw_result=True`` in the config.
    """

    def __init__(
        self,
        config: Optional[SFMIntentAnalyzerConfig] = None,
        *,
        analyzer: Optional[AnalyzerFn] = None,
    ) -> None:
        self.config = config or SFMIntentAnalyzerConfig()
        self.analyzer = analyzer or infer_final_cause_compact

    def build_query(self, state: StateMapping) -> Dict[str, Any]:
        explicit = _as_dict(state.get(self.config.query_key))
        if explicit:
            query = dict(explicit)
        else:
            query = {
                "observed_action": _first_text(state, self.config.state_action_keys),
                "action_variable": _clean_str(state.get("action_variable"), self.config.default_action_variable),
                "candidate_actions": _as_list(state.get("candidate_actions") or state.get("action_options")),
                "candidate_goals": _as_list(state.get("candidate_goals") or state.get("goals")),
                "negative_control_goals": _as_list(state.get("negative_control_goals") or state.get("negative_controls")),
                "placebo_goals": _as_list(state.get("placebo_goals") or state.get("placebos")),
                "side_effect_goals": _as_list(state.get("side_effect_goals")),
                "scm_graph": _as_dict(state.get("scm_graph") or state.get("graph")),
                "state": _as_dict(state.get("state") or state.get("context") or state.get("runtime_state")),
                "outcome_records": _as_list(state.get("outcome_records") or state.get("learning_records") or state.get("trajectory")),
                "agent": _as_dict(state.get("agent") or state.get("agent_model")),
                "normative_policy": _as_dict(state.get("normative_policy") or state.get("alignment_policy")),
                "protected_outcome": _clean_str(state.get("protected_outcome"), self.config.default_protected_outcome),
                "min_intent_score": float(state.get("min_intent_score", self.config.default_min_intent_score) or self.config.default_min_intent_score),
                "query_id": _clean_str(state.get("query_id") or state.get("run_id") or state.get("thread_id")),
            }
        query.setdefault("source", self.config.source)
        query.setdefault("protected_outcome", self.config.default_protected_outcome)
        query.setdefault("min_intent_score", self.config.default_min_intent_score)
        if not query.get("observed_action"):
            query["observed_action"] = _first_text(state, self.config.state_action_keys)
        if not query.get("candidate_goals") and state.get("goal"):
            query["candidate_goals"] = [state.get("goal")]
        return query

    def analyze(self, state: StateMapping) -> SFMNodeAnalysis:
        query = self.build_query(state)
        result = self.analyzer(query)
        primary = _clean_str(result.get("most_likely_goal"))
        side_effects = _extract_side_effects(query, result)
        deception_risk, deception_rationale = _deception_assessment(state, result, self.config)
        intended_effects = [primary] if primary and bool(result.get("intent_hypothesis_supported")) else []
        reason_codes = [str(code) for code in _as_list(result.get("reason_codes"))]
        limits = [str(limit) for limit in _as_list(result.get("limits"))]
        raw = dict(result) if self.config.include_raw_result else {}
        return SFMNodeAnalysis(
            primary_intent=primary,
            intentionality_score=float(result.get("intent_score", 0.0) or 0.0),
            final_cause_claim_level=_claim_level(result),
            intent_hypothesis_supported=bool(result.get("intent_hypothesis_supported")),
            intent_claim_authorized=bool(result.get("intent_claim_authorized")),
            governance_execution_allowed=bool(result.get("governance_execution_allowed")),
            intended_effects=intended_effects,
            side_effects=side_effects,
            side_effect_risk=_side_effect_risk(result, side_effects),
            deception_risk=deception_risk,
            deception_rationale=deception_rationale,
            observed_action=_clean_str(result.get("observed_action") or query.get("observed_action")),
            recommended_action=_clean_str(result.get("recommended_action")),
            recommendation_status=_clean_str(result.get("recommendation_status"), "unassessed"),
            robustness_status=_clean_str(result.get("robustness_status"), "unassessed"),
            authority_status=_clean_str(result.get("authority_status"), "diagnostic_only"),
            gate_status=_gate_status(result),
            blocked_claim_reason=_blocked_claim_reason(result),
            reason_codes=reason_codes,
            limits=limits,
            raw_result=raw,
        )

    def __call__(self, state: StateMapping) -> Dict[str, Any]:
        try:
            analysis = self.analyze(state).to_dict()
        except Exception as exc:
            if not self.config.fail_closed:
                raise
            analysis = SFMNodeAnalysis(
                final_cause_claim_level="diagnostic_only",
                side_effect_risk="unknown",
                deception_risk="unknown",
                gate_status="block",
                blocked_claim_reason="SFM node failed closed",
                reason_codes=["SFM_LANGGRAPH_NODE_FAILED_CLOSED"],
                limits=[f"{type(exc).__name__}: {exc}"],
            ).to_dict()
        update: Dict[str, Any] = {self.config.output_key: analysis}
        if self.config.append_trace:
            prior = _as_list(state.get(self.config.trace_key))
            event = {
                "node": "SFMIntentAnalyzerNode",
                "observed_action": analysis.get("observed_action", ""),
                "primary_intent": analysis.get("primary_intent", ""),
                "claim_level": analysis.get("final_cause_claim_level", "diagnostic_only"),
                "intent_score": analysis.get("intentionality_score", 0.0),
                "deception_risk": analysis.get("deception_risk", "unknown"),
                "gate_status": analysis.get("gate_status", "review"),
            }
            update[self.config.trace_key] = [*prior, event]
        return update


def build_sfm_intent_analyzer_node(
    *,
    output_key: str = "sfm_analysis",
    query_key: str = "sfm_query",
    include_raw_result: bool = False,
    analyzer: Optional[AnalyzerFn] = None,
) -> SFMIntentAnalyzerNode:
    """Factory for use in LangGraph examples and app configuration."""

    return SFMIntentAnalyzerNode(
        SFMIntentAnalyzerConfig(
            output_key=output_key,
            query_key=query_key,
            include_raw_result=include_raw_result,
        ),
        analyzer=analyzer,
    )


def add_sfm_intent_analyzer_node(
    graph_builder: Any,
    *,
    name: str = "sfm_intent_analyzer",
    node: Optional[SFMIntentAnalyzerNode] = None,
    **config_overrides: Any,
) -> SFMIntentAnalyzerNode:
    """Add an SFM node to a LangGraph ``StateGraph`` builder and return it.

    This helper avoids importing LangGraph.  It only requires the builder to
    expose an ``add_node(name, callable)`` method, matching LangGraph's API.
    """

    if node is None:
        config = SFMIntentAnalyzerConfig(**config_overrides) if config_overrides else SFMIntentAnalyzerConfig()
        node = SFMIntentAnalyzerNode(config)
    graph_builder.add_node(name, node)
    return node
