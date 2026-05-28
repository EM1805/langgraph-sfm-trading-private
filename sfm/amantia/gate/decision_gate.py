from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Set

from amantia.contracts import ActionPackage, DecisionPackage, decision_package_from_runtime, normalize_action_package


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_IDENTIFIED_TIERS = {"identified", "identified_graphical", "identified_recursive"}
_HIGH_RISK = {"high", "critical"}
_IRREVERSIBLE = {"irreversible", "not_reversible", "none"}


def _resolve_project_path(path_value: str) -> str:
    """Resolve default runtime file paths from either cwd or project root.

    Online callers often import Amantia from another working directory. The
    legacy runtime expects files such as action_registry.yaml and
    dangerous_paths.yaml to be relative to the repository root. This helper
    keeps that behavior while making the new agentic facade import-safe.
    """

    path = Path(str(path_value))
    if path.is_absolute():
        return str(path)
    if path.exists():
        return str(path)
    candidate = _PROJECT_ROOT / path
    if candidate.exists():
        return str(candidate)
    return str(candidate)


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _clean_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _clean_lower(value: Any, default: str = "unknown") -> str:
    return _clean_str(value, default).lower() or default


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray, Mapping)):
        return list(value)
    return [value]


def _split_adjustment(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw = value
        for sep in ["|", ";", ","]:
            raw = raw.replace(sep, ",")
        return [part.strip() for part in raw.split(",") if part.strip()]
    return [str(v).strip() for v in _as_list(value) if str(v).strip()]


def _graph_nodes(graph: Mapping[str, Any]) -> Set[str]:
    nodes: Set[str] = set()
    for node in _as_list(graph.get("nodes")):
        if isinstance(node, Mapping):
            node_id = _clean_str(node.get("id") or node.get("node_id") or node.get("name"))
        else:
            node_id = _clean_str(node)
        if node_id:
            nodes.add(node_id)

    for edge in _as_list(graph.get("edges")):
        if isinstance(edge, Mapping):
            src = _clean_str(edge.get("source") or edge.get("from"))
            tgt = _clean_str(edge.get("target") or edge.get("to"))
        else:
            parts = _as_list(edge)
            src = _clean_str(parts[0]) if len(parts) >= 1 else ""
            tgt = _clean_str(parts[1]) if len(parts) >= 2 else ""
        if src:
            nodes.add(src)
        if tgt:
            nodes.add(tgt)

    return nodes


def _has_graph(graph: Mapping[str, Any]) -> bool:
    return bool(_as_list(graph.get("nodes")) or _as_list(graph.get("edges")))


def _pick_from_nodes(preferred: Iterable[str], nodes: Set[str], fallback: str = "") -> str:
    for item in preferred:
        text = _clean_str(item)
        if text and text in nodes:
            return text
    for item in preferred:
        text = _clean_str(item)
        if text:
            return text
    return fallback


class DecisionGate:
    """Agentic facade over the existing Amantia runtime veto gateway.

    The old runtime returns PASS / PASS_WITH_WARNING / REVIEW / HARD_BLOCK.
    This facade converts that into the product-level decisions expected by
    LLMs and agents: allow / warn / ask_clarification / abstain / veto.

    Step 8 adds an optional SCM-ID online check. It runs only when the action
    package contains an ``scm_graph`` or ``causal_query``. Missing graphs do not
    block normal runtime safety routing.

    Step 85 chains the online checks: when SCM-ID identifies the effect, that
    result is injected into the estimation query so ``data_path`` backends can
    run only behind an explicit ID-derived authorization boundary.
    """

    def __init__(
        self,
        *,
        registry_path: str = "action_registry.yaml",
        path_library_path: str = "dangerous_paths.yaml",
        graph_path: str = "operational_causal_graph.yaml",
        event_log_path: str = "historical_action_events.jsonl",
        validation_plan_path: str = "out/validation_plan_level2.csv",
        authority_cards_path: str = "out/veto/causal_authority_cards.jsonl",
        enable_identification: bool = True,
        identification_engine: Any = None,
        enable_estimation: bool = True,
        estimation_engine: Any = None,
        enable_counterfactual: bool = True,
        counterfactual_engine: Any = None,
        enable_risk_policy: bool = True,
        risk_policy: Any = None,
        enable_recommender: bool = True,
        action_recommender: Any = None,
    ) -> None:
        self.registry_path = _resolve_project_path(registry_path)
        self.path_library_path = _resolve_project_path(path_library_path)
        self.graph_path = _resolve_project_path(graph_path)
        self.event_log_path = _resolve_project_path(event_log_path)
        self.validation_plan_path = _resolve_project_path(validation_plan_path)
        self.authority_cards_path = _resolve_project_path(authority_cards_path)
        self.enable_identification = enable_identification
        self.identification_engine = identification_engine
        self.enable_estimation = enable_estimation
        self.estimation_engine = estimation_engine
        self.enable_counterfactual = enable_counterfactual
        self.counterfactual_engine = counterfactual_engine
        self.enable_risk_policy = enable_risk_policy
        self.risk_policy = risk_policy
        self.enable_recommender = enable_recommender
        self.action_recommender = action_recommender

    def _causal_query_from_action(self, action: ActionPackage) -> Optional[Dict[str, Any]]:
        """Build an IdentificationEngine payload when graph evidence exists."""

        causal_query = _as_dict(action.causal_query)
        graph = _as_dict(
            action.scm_graph
            or causal_query.get("scm_graph")
            or causal_query.get("graph")
            or action.context.get("scm_graph")
            or action.context.get("graph")
            or action.params.get("scm_graph")
            or action.params.get("graph")
        )
        if not _has_graph(graph):
            return None

        nodes = _graph_nodes(graph)
        treatment = _clean_str(
            action.treatment
            or causal_query.get("treatment")
            or action.context.get("treatment")
            or action.params.get("treatment")
        )
        if not treatment:
            treatment = _pick_from_nodes(
                [
                    "agent_action",
                    "action_active",
                    action.action_name,
                    action.candidate_action,
                    "treatment",
                ],
                nodes,
                action.action_name or action.candidate_action,
            )

        outcome = _clean_str(
            action.outcome
            or causal_query.get("outcome")
            or action.context.get("outcome")
            or action.params.get("outcome")
        )
        if not outcome:
            outcome = _pick_from_nodes(
                [
                    action.intended_outcome,
                    "task_success",
                    action.protected_outcome,
                    "user_or_system_harm",
                    "harm_event",
                    "outcome",
                ],
                nodes,
                action.intended_outcome or action.protected_outcome,
            )

        adjustment_set = (
            list(action.adjustment_set or [])
            or _split_adjustment(causal_query.get("adjustment_set"))
            or _split_adjustment(action.context.get("adjustment_set"))
            or _split_adjustment(action.context.get("candidate_confounders"))
            or _split_adjustment(action.params.get("adjustment_set"))
        )

        return {
            "scm_graph": graph,
            "treatment": treatment,
            "outcome": outcome,
            "adjustment_set": adjustment_set,
            "strategy_hint": causal_query.get("strategy_hint") or action.context.get("strategy_hint", ""),
            "query_id": causal_query.get("query_id") or action.request_id,
            "source": "gate.decision_gate.online_identification",
        }

    def _run_identification(self, action: ActionPackage) -> Optional[Dict[str, Any]]:
        if not self.enable_identification:
            return None

        query = self._causal_query_from_action(action)
        if not query:
            return None

        try:
            engine = self.identification_engine
            if engine is None:
                from amantia.causal_core.identification import IdentificationEngine

                engine = IdentificationEngine()

            result = engine.identify(query)
            if hasattr(result, "to_dict"):
                return result.to_dict()
            return dict(result or {})
        except Exception as exc:  # pragma: no cover - defensive boundary
            return {
                "identified": False,
                "identification_strategy": "adapter_runtime_error",
                "identification_tier": "unidentified",
                "authority_status": "error",
                "reason": f"SCM-ID adapter failed safely: {type(exc).__name__}: {exc}",
                "reason_codes": ["SCM_ID_ADAPTER_ERROR"],
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }


    def _estimation_query_from_action(self, action: ActionPackage) -> Optional[Dict[str, Any]]:
        """Build an EstimationEngine payload only when estimation evidence exists."""

        query = _as_dict(getattr(action, "estimation_query", {}))
        query.update(_as_dict(action.context.get("estimation_query")))
        query.update(_as_dict(action.params.get("estimation_query")))

        # Allow direct shorthand fields in context/params without polluting the
        # core ActionPackage with every possible estimator input.
        for source in (action.context, action.params):
            for key in (
                "data_path",
                "data_csv",
                "effect_estimates_path",
                "effects_path",
                "effect_estimate",
                "ci_low",
                "ci_high",
                "support_n",
                "treated_n",
                "control_n",
                "estimator_used",
                "robustness_status",
                "negative_control_status",
                "placebo_status",
                "sensitivity_status",
                "expected_direction",
            ):
                if key in source and key not in query:
                    query[key] = source[key]

        if action.treatment and "treatment" not in query:
            query["treatment"] = action.treatment
        if action.outcome and "outcome" not in query:
            query["outcome"] = action.outcome
        if action.adjustment_set and "adjustment_set" not in query:
            query["adjustment_set"] = list(action.adjustment_set)

        evidence_keys = {
            "effect_estimate",
            "effect_estimates_path",
            "effects_path",
            "data_path",
            "data_csv",
            "csv_path",
        }
        if not any(key in query and query.get(key) not in (None, "") for key in evidence_keys):
            return None

        query.setdefault("source", "gate.decision_gate.online_estimation")
        return query

    def _estimation_query_with_identification(
        self,
        query: Dict[str, Any],
        identification: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        """Attach ID output to an estimation query without weakening the gate.

        The estimation adapter intentionally requires explicit ID/contract
        authorization before running a causal backend over raw CSV data.  This
        helper is the DecisionGate bridge: a positive online SCM-ID result can
        authorize estimation, while a failed or absent ID result leaves the
        estimation adapter in diagnostic-only mode.
        """

        out = dict(query or {})
        id_result = dict(identification or {}) if isinstance(identification, Mapping) else {}
        if not id_result:
            return out

        out.setdefault("identification_result", id_result)
        identified = bool(id_result.get("identified"))
        tier = _clean_str(id_result.get("identification_tier"), "")
        strategy = _clean_str(id_result.get("identification_strategy"), "")

        if strategy and "identification_strategy" not in out:
            out["identification_strategy"] = strategy
        if tier and "identification_status" not in out:
            out["identification_status"] = tier

        if identified:
            out.setdefault("identified", True)
            out.setdefault("allowed_for_estimation", True)
            # ``identified_estimable`` is an adapter-level authorization token.
            # It does not claim arbitrary full-ID completeness; the raw ID
            # payload remains attached for audits and full_id_claim_allowed.
            out.setdefault("authority_level", "identified_estimable")
            out.setdefault("estimation_enabled", True)
        else:
            out.setdefault("identified", False)

        return out

    def _run_estimation(
        self,
        action: ActionPackage,
        identification: Optional[Mapping[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        if not self.enable_estimation:
            return None
        query = self._estimation_query_from_action(action)
        if not query:
            return None
        query = self._estimation_query_with_identification(query, identification)
        try:
            engine = self.estimation_engine
            if engine is None:
                from amantia.causal_core.estimation import EstimationEngine

                engine = EstimationEngine()
            result = engine.estimate(query)
            if hasattr(result, "to_dict"):
                return result.to_dict()
            return dict(result or {})
        except Exception as exc:  # pragma: no cover - defensive boundary
            return {
                "estimated": False,
                "estimation_status": "adapter_runtime_error",
                "authority_status": "error",
                "reason": f"Estimation adapter failed safely: {type(exc).__name__}: {exc}",
                "reason_codes": ["ESTIMATION_ADAPTER_ERROR"],
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }


    def _counterfactual_query_from_action(self, action: ActionPackage) -> Optional[Dict[str, Any]]:
        """Build a CounterfactualEngine payload when alternatives/evidence exist."""

        query = _as_dict(getattr(action, "counterfactual_query", {}))
        query.update(_as_dict(action.context.get("counterfactual_query")))
        query.update(_as_dict(action.params.get("counterfactual_query")))

        # Direct shorthand fields accepted by the adapter.
        for source in (action.context, action.params):
            for key in (
                "action_options",
                "alternatives",
                "options",
                "action_scores",
                "risk_scores",
                "harm_scores",
                "harm_probabilities",
                "effect_scores",
                "effect_estimates",
                "min_margin",
                "risk_weight",
            ):
                if key in source and key not in query:
                    query[key] = source[key]

        if action.candidate_actions and not any(k in query for k in ("candidate_actions", "action_options", "alternatives", "options")):
            query["candidate_actions"] = list(action.candidate_actions)

        current_action = action.action_name or action.candidate_action
        if current_action and "current_action" not in query:
            query["current_action"] = current_action

        if action.intended_outcome and "outcome" not in query:
            query["outcome"] = action.intended_outcome
        if action.protected_outcome and "protected_outcome" not in query:
            query["protected_outcome"] = action.protected_outcome

        evidence_keys = {
            "candidate_actions",
            "action_options",
            "alternatives",
            "options",
            "action_scores",
            "risk_scores",
            "harm_scores",
            "harm_probabilities",
            "effect_scores",
            "effect_estimates",
        }
        if not any(key in query and query.get(key) not in (None, "", []) for key in evidence_keys):
            return None

        query.setdefault("source", "gate.decision_gate.online_counterfactual")
        return query

    def _run_counterfactual(self, action: ActionPackage) -> Optional[Dict[str, Any]]:
        if not self.enable_counterfactual:
            return None
        query = self._counterfactual_query_from_action(action)
        if not query:
            return None
        try:
            engine = self.counterfactual_engine
            if engine is None:
                from amantia.causal_core.counterfactual import CounterfactualEngine

                engine = CounterfactualEngine()
            result = engine.compare(query)
            if hasattr(result, "to_dict"):
                return result.to_dict()
            return dict(result or {})
        except Exception as exc:  # pragma: no cover - defensive boundary
            return {
                "compared": False,
                "comparison_status": "adapter_runtime_error",
                "authority_status": "error",
                "reason": f"Counterfactual adapter failed safely: {type(exc).__name__}: {exc}",
                "reason_codes": ["COUNTERFACTUAL_ADAPTER_ERROR"],
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }

    def _enrich_with_counterfactual(
        self,
        decision: DecisionPackage,
        action: ActionPackage,
        counterfactual: Optional[Mapping[str, Any]],
    ) -> DecisionPackage:
        if not counterfactual:
            return decision
        counterfactual = dict(counterfactual)
        decision.causal_counterfactual = counterfactual
        decision.audit_payload["causal_counterfactual"] = counterfactual
        decision.raw_runtime_result["causal_counterfactual"] = counterfactual

        if counterfactual.get("compared"):
            if "COUNTERFACTUAL_COMPARED" not in decision.reason_codes:
                decision.reason_codes.append("COUNTERFACTUAL_COMPARED")
        else:
            if "COUNTERFACTUAL_NOT_COMPARED" not in decision.reason_codes:
                decision.reason_codes.append("COUNTERFACTUAL_NOT_COMPARED")
            return decision

        recommended = _clean_str(counterfactual.get("recommended_action"))
        current = _clean_str(counterfactual.get("current_action") or decision.selected_action)
        status = _clean_str(counterfactual.get("comparison_status"))
        margin = counterfactual.get("score_margin")

        if recommended and current and recommended != current and status == "alternative_recommended":
            if "COUNTERFACTUAL_ALTERNATIVE_RECOMMENDED" not in decision.reason_codes:
                decision.reason_codes.append("COUNTERFACTUAL_ALTERNATIVE_RECOMMENDED")

            # A hard veto is never softened. For allowed/warned actions, the
            # counterfactual comparison can make the language more cautious.
            if decision.decision == "allow":
                decision.decision = "warn"
                decision.reason = (
                    f"Runtime allowed {current}, but counterfactual comparison recommends "
                    f"{recommended} with margin {margin}."
                )
                decision.llm_instruction = (
                    f"Prefer {recommended} over {current} unless the user explicitly requires {current}. "
                    "State that Amantia selected the safer/higher-utility alternative."
                )
                if decision.confidence == "high":
                    decision.confidence = "medium"
            elif decision.decision == "warn":
                decision.reason = f"{decision.reason} Counterfactual comparison recommends {recommended} over {current}."

        return decision

    def _enrich_with_estimation(
        self,
        decision: DecisionPackage,
        action: ActionPackage,
        estimation: Optional[Mapping[str, Any]],
    ) -> DecisionPackage:
        if not estimation:
            return decision
        estimation = dict(estimation)
        decision.causal_estimation = estimation
        decision.audit_payload["causal_estimation"] = estimation
        decision.raw_runtime_result["causal_estimation"] = estimation

        if estimation.get("estimated") or estimation.get("causal_estimate_available"):
            if "ESTIMATION_AVAILABLE" not in decision.reason_codes:
                decision.reason_codes.append("ESTIMATION_AVAILABLE")
            if decision.confidence in {"unknown", "low"} and decision.decision in {"allow", "warn"}:
                decision.confidence = "medium"
        elif estimation.get("association_estimate_available"):
            if "DIAGNOSTIC_ASSOCIATION_AVAILABLE" not in decision.reason_codes:
                decision.reason_codes.append("DIAGNOSTIC_ASSOCIATION_AVAILABLE")
            if "ESTIMATION_UNAVAILABLE" not in decision.reason_codes:
                decision.reason_codes.append("ESTIMATION_UNAVAILABLE")
        else:
            if "ESTIMATION_UNAVAILABLE" not in decision.reason_codes:
                decision.reason_codes.append("ESTIMATION_UNAVAILABLE")
        return decision

    def _enrich_with_identification(
        self,
        decision: DecisionPackage,
        action: ActionPackage,
        identification: Optional[Mapping[str, Any]],
    ) -> DecisionPackage:
        if not identification:
            return decision

        identification = dict(identification)
        identified = bool(identification.get("identified"))
        identification_tier = _clean_str(identification.get("identification_tier"), "unidentified")

        decision.causal_identification = identification
        decision.identification_tier = identification_tier
        decision.audit_payload["causal_identification"] = identification
        decision.raw_runtime_result["causal_identification"] = identification

        if identified or identification_tier in _IDENTIFIED_TIERS:
            if "SCM_ID_IDENTIFIED" not in decision.reason_codes:
                decision.reason_codes.append("SCM_ID_IDENTIFIED")
            decision.audit_payload["identification_tier"] = identification_tier
            if decision.confidence in {"unknown", "low"} and decision.decision in {"allow", "warn"}:
                decision.confidence = "medium"
            return decision

        if "SCM_ID_UNIDENTIFIED" not in decision.reason_codes:
            decision.reason_codes.append("SCM_ID_UNIDENTIFIED")

        # Do not soften hard vetoes. A runtime veto remains veto even if the
        # graph check is inconclusive.
        if decision.decision == "veto":
            return decision

        risk = _clean_lower(action.risk_level)
        reversibility = _clean_lower(action.reversibility)
        id_reason = _clean_str(identification.get("reason"), "SCM-ID did not identify the action effect.")

        if decision.decision == "allow":
            if risk in _HIGH_RISK or reversibility in _IRREVERSIBLE:
                decision.decision = "abstain"
                decision.reason = f"SCM-ID check was available but did not identify the effect: {id_reason}"
                decision.llm_instruction = (
                    "Do not execute the action yet. Explain that Amantia lacks identifiable causal support "
                    "for this higher-risk action and route to review, clarification, or a safer alternative."
                )
                decision.confidence = "low"
            else:
                decision.decision = "warn"
                decision.reason = f"Runtime allowed the action, but SCM-ID did not identify the effect: {id_reason}"
                decision.llm_instruction = (
                    "Proceed only as a low-risk communication/tool step; state uncertainty and avoid claiming causal support."
                )
                if decision.confidence == "high":
                    decision.confidence = "medium"

        elif decision.decision == "warn":
            decision.reason = f"{decision.reason} SCM-ID did not identify the effect: {id_reason}"
            if decision.confidence == "high":
                decision.confidence = "medium"

        return decision

    def _apply_risk_policy(self, decision: DecisionPackage, action: ActionPackage) -> DecisionPackage:
        """Strengthen the decision according to causal-evidence-by-risk rules.

        This policy never softens a veto. It can turn allow -> warn, allow/warn
        -> abstain, or any non-veto critical-risk action -> veto when required
        trusted evidence is missing.
        """

        if not self.enable_risk_policy:
            return decision

        try:
            policy = self.risk_policy
            if policy is None:
                from amantia.risk_policy import CausalEvidenceByRiskPolicy

                policy = CausalEvidenceByRiskPolicy()

            result_obj = policy.evaluate(action, decision)
            result = result_obj.to_dict() if hasattr(result_obj, "to_dict") else dict(result_obj or {})
            old_decision = decision.decision
            new_decision = _clean_lower(result.get("policy_decision"), old_decision or "abstain")

            decision.risk_policy = result
            decision.policy_decision = new_decision
            decision.evidence_required = list(result.get("evidence_required", []) or [])
            decision.evidence_present = list(result.get("evidence_present", []) or [])
            decision.evidence_missing = list(result.get("evidence_missing", []) or [])
            decision.audit_payload["risk_policy"] = result
            decision.raw_runtime_result["risk_policy"] = result

            for code in result.get("reason_codes", []) or []:
                code = str(code)
                if code and code not in decision.reason_codes:
                    decision.reason_codes.append(code)

            if new_decision != old_decision:
                decision.decision = new_decision
                policy_reason = _clean_str(result.get("reason"), "Risk policy strengthened the decision because evidence was missing.")
                decision.reason = policy_reason
                if new_decision == "veto":
                    decision.llm_instruction = (
                        "Do not execute this action. Amantia's risk policy requires stronger trusted evidence "
                        "before any tool call can run. Offer the safer recommended action or route to approval."
                    )
                    decision.confidence = "high"
                elif new_decision == "abstain":
                    decision.llm_instruction = (
                        "Do not execute yet. Amantia needs more causal support, trusted approval, or mitigation evidence "
                        "for this higher-risk action."
                    )
                    decision.confidence = "low"
                elif new_decision == "warn":
                    decision.llm_instruction = (
                        "Proceed only with warning, mitigation, and audit because Amantia's risk policy found missing evidence."
                    )
                    if decision.confidence == "high":
                        decision.confidence = "medium"

            decision.short_for_llm = {}
            decision.short_for_llm = decision.to_dict().get("short_for_llm", {})
            return decision
        except Exception as exc:  # pragma: no cover - defensive boundary
            decision.audit_payload["risk_policy_error"] = {"type": type(exc).__name__, "message": str(exc)}
            if "RISK_POLICY_ERROR" not in decision.reason_codes:
                decision.reason_codes.append("RISK_POLICY_ERROR")
            # Fail safely: if policy itself fails on a high/critical action, do not allow silently.
            if decision.decision == "allow" and _clean_lower(action.risk_level) in {"high", "critical"}:
                decision.decision = "abstain"
                decision.reason = f"Risk policy failed safely: {type(exc).__name__}: {exc}"
                decision.llm_instruction = "Do not execute until the risk policy can evaluate this high-risk action."
                decision.confidence = "low"
            decision.short_for_llm = {}
            decision.short_for_llm = decision.to_dict().get("short_for_llm", {})
            return decision

    def _attach_recommendations(self, decision: DecisionPackage, action: ActionPackage) -> DecisionPackage:
        """Attach proposal-only safer-action recommendations.

        Recommendations never soften a veto and never execute tools. They are
        compact next-action candidates for the LLM/agent planner and must pass
        ToolGuard before execution.
        """

        if not self.enable_recommender:
            return decision

        try:
            recommender = self.action_recommender
            if recommender is None:
                from amantia.action_recommender import CausalActionRecommender

                recommender = CausalActionRecommender()

            package = recommender.recommend(action, decision)
            rec = package.to_dict() if hasattr(package, "to_dict") else dict(package or {})
            decision.recommended_action = dict(rec.get("recommended_action", {}) or {})
            decision.recommended_actions = list(rec.get("recommended_actions", []) or [])
            decision.recommendation_summary = _clean_str(rec.get("recommendation_summary"))
            decision.recommendation_status = _clean_str(rec.get("execution_status"), "no_recommendation")
            decision.audit_payload["action_recommendation"] = rec
            decision.raw_runtime_result["action_recommendation"] = rec
            if decision.recommended_actions and "ACTION_RECOMMENDATION_AVAILABLE" not in decision.reason_codes:
                decision.reason_codes.append("ACTION_RECOMMENDATION_AVAILABLE")
            decision.short_for_llm = {}
            decision.short_for_llm = decision.to_dict().get("short_for_llm", {})
            return decision
        except Exception as exc:  # pragma: no cover - defensive boundary
            decision.audit_payload["action_recommendation_error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
            if "ACTION_RECOMMENDER_ERROR" not in decision.reason_codes:
                decision.reason_codes.append("ACTION_RECOMMENDER_ERROR")
            decision.short_for_llm = {}
            decision.short_for_llm = decision.to_dict().get("short_for_llm", {})
            return decision

    def evaluate(self, payload: Any) -> DecisionPackage:
        action = normalize_action_package(payload)
        runtime_payload = action.to_runtime_payload()

        try:
            from runtime.veto_gateway import evaluate_action_request

            runtime_result = evaluate_action_request(
                runtime_payload,
                registry_path=self.registry_path,
                path_library_path=self.path_library_path,
                graph_path=self.graph_path,
                event_log_path=self.event_log_path,
                validation_plan_path=self.validation_plan_path,
                authority_cards_path=self.authority_cards_path,
            )
            decision = decision_package_from_runtime(action, runtime_result)
            identification = self._run_identification(action)
            decision = self._enrich_with_identification(decision, action, identification)
            decision = self._enrich_with_estimation(decision, action, self._run_estimation(action, identification))
            decision = self._enrich_with_counterfactual(decision, action, self._run_counterfactual(action))
            decision = self._apply_risk_policy(decision, action)
            return self._attach_recommendations(decision, action)
        except Exception as exc:  # pragma: no cover - defensive runtime boundary
            # Safety-first behavior: runtime failure does not become allow.
            error_result: Dict[str, Any] = {
                "decision": {
                    "decision": "UNKNOWN",
                    "reason_codes": ["DECISION_GATE_RUNTIME_ERROR"],
                    "notes": [f"Decision gate failed safely: {type(exc).__name__}: {exc}"],
                    "evidence_tier": "unknown",
                    "identification_tier": "unknown",
                    "structural_tier": "unknown",
                },
                "runtime_error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
                "action_intent": runtime_payload,
            }
            decision = decision_package_from_runtime(action, error_result)
            identification = self._run_identification(action)
            decision = self._enrich_with_identification(decision, action, identification)
            decision = self._enrich_with_estimation(decision, action, self._run_estimation(action, identification))
            decision = self._enrich_with_counterfactual(decision, action, self._run_counterfactual(action))
            decision = self._apply_risk_policy(decision, action)
            return self._attach_recommendations(decision, action)


def evaluate_action_package(payload: Any, **kwargs: Any) -> Dict[str, Any]:
    """Convenience function returning a plain dict for API/CLI callers."""
    return DecisionGate(**kwargs).evaluate(payload).to_dict()


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate an ActionPackage through Amantia's agentic Decision Gate.")
    parser.add_argument("--input", required=True, help="Path to ActionPackage JSON or legacy runtime action JSON.")
    parser.add_argument("--out", default="out/decision_package.json", help="Where to write DecisionPackage JSON.")
    parser.add_argument("--registry-path", default="action_registry.yaml")
    parser.add_argument("--path-library-path", default="dangerous_paths.yaml")
    parser.add_argument("--graph-path", default="operational_causal_graph.yaml")
    parser.add_argument("--event-log-path", default="historical_action_events.jsonl")
    parser.add_argument("--validation-plan-path", default="out/validation_plan_level2.csv")
    parser.add_argument("--authority-cards-path", default="out/veto/causal_authority_cards.jsonl")
    parser.add_argument("--disable-identification", action="store_true", help="Skip optional online SCM-ID enrichment.")
    parser.add_argument("--disable-estimation", action="store_true", help="Skip optional online estimation enrichment.")
    parser.add_argument("--disable-counterfactual", action="store_true", help="Skip optional online counterfactual enrichment.")
    parser.add_argument("--disable-risk-policy", action="store_true", help="Skip causal-evidence-by-risk policy enforcement.")
    parser.add_argument("--disable-recommender", action="store_true", help="Skip proposal-only causal action recommendations.")
    args = parser.parse_args(argv)

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    gate = DecisionGate(
        registry_path=args.registry_path,
        path_library_path=args.path_library_path,
        graph_path=args.graph_path,
        event_log_path=args.event_log_path,
        validation_plan_path=args.validation_plan_path,
        authority_cards_path=args.authority_cards_path,
        enable_identification=not args.disable_identification,
        enable_estimation=not args.disable_estimation,
        enable_counterfactual=not args.disable_counterfactual,
        enable_risk_policy=not args.disable_risk_policy,
        enable_recommender=not args.disable_recommender,
    )
    result = gate.evaluate(payload).to_dict()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "status": "ok",
        "out": str(out_path),
        "decision": result.get("decision"),
        "runtime_decision": result.get("runtime_decision"),
        "selected_action": result.get("selected_action"),
        "identification_tier": result.get("identification_tier"),
        "has_causal_identification": bool(result.get("causal_identification")),
        "has_causal_estimation": bool(result.get("causal_estimation")),
        "has_causal_counterfactual": bool(result.get("causal_counterfactual")),
        "has_risk_policy": bool(result.get("risk_policy")),
        "has_recommended_actions": bool(result.get("recommended_actions")),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
