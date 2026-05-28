from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from amantia.contracts import ActionPackage, DecisionPackage, normalize_action_package

from .types import RecommendedAction, RecommendedActionPackage


_TRADING_ACTIONS = {
    "connect_brokerage_api",
    "place_market_order",
    "place_limit_order",
    "cancel_order",
    "modify_order",
    "close_position",
    "open_margin_position",
    "set_stop_loss",
    "change_trading_risk_limits",
    "rebalance_portfolio",
}
_FINANCE_ACTIONS = {
    "view_account_balance",
    "access_financial_data",
    "initiate_bank_transfer",
    "withdraw_funds",
    "approve_invoice_payment",
    "charge_customer",
    "refund_payment",
    "issue_payout",
}
_DESTRUCTIVE_ACTIONS = {"delete_resource", "delete_file", "erase_memory"}
_EXTERNAL_COMM_ACTIONS = {"send_email_external", "share_file_external"}
_CODE_OR_OPS_ACTIONS = {
    "run_shell_command",
    "execute_code",
    "modify_database",
    "deploy_code",
    "install_package",
    "deploy_config_change",
}


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _clean_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _clean_lower(value: Any, default: str = "unknown") -> str:
    return _clean_str(value, default).lower() or default


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _as_list(value: Any) -> List[Any]:
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


def _dedupe(items: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        text = _clean_str(item)
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _decision_to_mapping(decision: Any) -> Dict[str, Any]:
    if isinstance(decision, DecisionPackage):
        return decision.to_dict()
    if isinstance(decision, Mapping):
        return dict(decision)
    return {}


class CausalActionRecommender:
    """Rule-bounded causal action recommender for AI-agent integration.

    The recommender does not execute tools and does not override vetoes. It only
    proposes a safer next ActionPackage candidate after DecisionGate, SCM-ID,
    estimation, and counterfactual evidence have been attached to the decision.
    """

    def recommend(self, action_payload: Any, decision_package: Any) -> RecommendedActionPackage:
        action = normalize_action_package(action_payload)
        decision = _decision_to_mapping(decision_package)
        original = _clean_str(decision.get("selected_action") or action.action_name or action.candidate_action)
        gate_decision = _clean_lower(decision.get("decision"), "abstain")

        proposals: List[RecommendedAction] = []
        proposals.extend(self._counterfactual_proposals(action, decision, original))
        proposals.extend(self._domain_proposals(action, decision, original, gate_decision))
        proposals.extend(self._causal_evidence_proposals(action, decision, original, gate_decision))

        if not proposals and gate_decision in {"allow", "warn"}:
            proposals.append(
                RecommendedAction(
                    action_name=original or "proceed_with_guarded_action",
                    recommendation_type="guarded_execution",
                    rationale="The original action was not blocked; execute only through the Tool Guard and keep audit enabled.",
                    parameters={"original_action": original},
                    safety_constraints=["tool_guard_required", "audit_required"],
                    execution_status="proposal_only_requires_gate_review",
                    priority=25,
                )
            )

        proposals = self._rank_and_dedupe(proposals)
        top = proposals[0].to_dict() if proposals else {}
        constraints = _dedupe([c for p in proposals for c in p.safety_constraints])
        summary = self._summary(original, gate_decision, top, proposals)
        status = top.get("execution_status", "no_recommendation") if top else "no_recommendation"

        return RecommendedActionPackage(
            original_action=original,
            decision=gate_decision,
            recommended_action=top,
            recommended_actions=[p.to_dict() for p in proposals],
            recommendation_summary=summary,
            safety_constraints=constraints,
            execution_status=status,
            notes=[
                "Recommendations are proposal-only and must be re-submitted to DecisionGate/ToolGuard before execution.",
                "The recommender may reduce risk or request evidence, but it cannot bypass veto, abstain, or approval gates.",
            ],
            causal_inputs=self._causal_inputs(decision),
        )

    def _counterfactual_proposals(self, action: ActionPackage, decision: Mapping[str, Any], original: str) -> List[RecommendedAction]:
        cf = _as_dict(decision.get("causal_counterfactual"))
        recommended = _clean_str(cf.get("recommended_action"))
        current = _clean_str(cf.get("current_action") or original)
        status = _clean_str(cf.get("comparison_status"))
        if not recommended or recommended == current:
            return []
        if status and status != "alternative_recommended":
            return []
        return [
            RecommendedAction(
                action_name=recommended,
                recommendation_type="counterfactual_safer_alternative",
                rationale=(
                    f"Counterfactual comparison preferred {recommended} over {current}. "
                    "Use it as a new candidate action, not as an automatic execution."
                ),
                parameters={
                    "current_action": current,
                    "score_margin": cf.get("score_margin"),
                    "comparison_status": status or "alternative_recommended",
                },
                safety_constraints=["tool_guard_required", "audit_required"],
                priority=92,
            )
        ]

    def _domain_proposals(
        self,
        action: ActionPackage,
        decision: Mapping[str, Any],
        original: str,
        gate_decision: str,
    ) -> List[RecommendedAction]:
        name = (original or action.action_name or action.candidate_action or "").strip().lower()
        trusted = _as_dict(action.trusted_runtime_context)
        params = _as_dict(action.params)
        ctx = trusted or _as_dict(action.context)
        is_trading = name in _TRADING_ACTIONS or _as_bool(ctx.get("trading_action") or params.get("trading_action"))
        is_finance = name in _FINANCE_ACTIONS or is_trading or _as_bool(ctx.get("financial_action") or params.get("financial_action"))
        live_or_real_money = _as_bool(ctx.get("live_trading") or params.get("live_trading")) or _as_bool(ctx.get("real_money") or params.get("real_money"))
        leverage_or_margin = _as_bool(ctx.get("leverage_used") or params.get("leverage_used")) or _as_bool(ctx.get("margin_used") or params.get("margin_used")) or name == "open_margin_position"
        risk_limits_present = _as_bool(ctx.get("risk_limits_present") or params.get("risk_limits_present"))
        approval_present = _as_bool(ctx.get("approval_present") or params.get("approval_present"))

        proposals: List[RecommendedAction] = []
        blocked_or_cautious = gate_decision in {"veto", "abstain", "ask_clarification", "warn"}

        if is_trading and blocked_or_cautious:
            constraints = [
                "paper_trading_only_until_approved",
                "risk_limits_required",
                "max_notional_must_be_trusted_runtime_value",
                "audit_required",
                "tool_guard_required",
            ]
            if leverage_or_margin:
                constraints.append("no_margin_or_leverage_without_human_approval")
            if live_or_real_money:
                constraints.append("no_live_trade_without_trusted_approval")
            if not risk_limits_present:
                constraints.append("define_stop_loss_and_position_limit_first")
            proposals.append(
                RecommendedAction(
                    action_name="paper_trade_limit_order",
                    recommendation_type="trading_safer_alternative",
                    rationale=(
                        "Live trading/real-money action is high risk. Prefer a paper-trading or limit-order simulation "
                        "with explicit risk limits before any live execution."
                    ),
                    parameters={
                        "original_action": original,
                        "paper_trading": True,
                        "live_trading": False,
                        "order_type": "limit",
                        "approval_required_before_live_execution": True,
                    },
                    safety_constraints=constraints,
                    execution_status="proposal_only_requires_gate_review",
                    priority=98,
                )
            )

        if is_finance and not is_trading and blocked_or_cautious:
            proposals.append(
                RecommendedAction(
                    action_name="prepare_financial_action_for_human_review",
                    recommendation_type="financial_review_action",
                    rationale="Financial action should be prepared for review instead of executed directly without trusted approval.",
                    parameters={
                        "original_action": original,
                        "approval_present": approval_present,
                        "execute_funds_movement": False,
                    },
                    safety_constraints=[
                        "trusted_approval_required",
                        "two_party_review_required_for_value_transfer",
                        "audit_required",
                        "tool_guard_required",
                    ],
                    execution_status="requires_trusted_human_approval",
                    priority=96,
                )
            )

        if name in _DESTRUCTIVE_ACTIONS and blocked_or_cautious:
            proposals.append(
                RecommendedAction(
                    action_name="backup_or_archive_then_request_approval",
                    recommendation_type="destructive_action_mitigation",
                    rationale="The original action can cause irreversible data loss; use a reversible archive/backup path first.",
                    parameters={"original_action": original, "delete_now": False, "backup_required": True},
                    safety_constraints=["backup_required", "rollback_required", "trusted_approval_required", "tool_guard_required"],
                    execution_status="requires_trusted_human_approval",
                    priority=90,
                )
            )

        if name in _CODE_OR_OPS_ACTIONS and blocked_or_cautious:
            proposals.append(
                RecommendedAction(
                    action_name="run_in_sandbox_or_staging_with_rollback",
                    recommendation_type="ops_mitigation",
                    rationale="Execute code/ops changes only in a bounded sandbox or staging environment with rollback evidence.",
                    parameters={"original_action": original, "environment": "sandbox_or_staging", "production_execution": False},
                    safety_constraints=["sandbox_required", "rollback_required", "blast_radius_limit_required", "tool_guard_required"],
                    execution_status="proposal_only_requires_gate_review",
                    priority=84,
                )
            )

        if name in _EXTERNAL_COMM_ACTIONS and blocked_or_cautious:
            proposals.append(
                RecommendedAction(
                    action_name="draft_for_review_before_external_send",
                    recommendation_type="external_communication_mitigation",
                    rationale="External communication/share action should be drafted and reviewed before sending outside the trusted boundary.",
                    parameters={"original_action": original, "send_now": False, "review_required": True},
                    safety_constraints=["recipient_verification_required", "content_review_required", "tool_guard_required"],
                    execution_status="requires_review",
                    priority=78,
                )
            )

        return proposals

    def _causal_evidence_proposals(
        self,
        action: ActionPackage,
        decision: Mapping[str, Any],
        original: str,
        gate_decision: str,
    ) -> List[RecommendedAction]:
        proposals: List[RecommendedAction] = []
        risk = _clean_lower(decision.get("risk_level") or action.risk_level)
        ambiguity = _clean_lower(action.ambiguity)
        reason_codes = {str(c) for c in _as_list(decision.get("reason_codes"))}
        identification = _as_dict(decision.get("causal_identification"))
        estimation = _as_dict(decision.get("causal_estimation"))
        unidentified = (
            "SCM_ID_UNIDENTIFIED" in reason_codes
            or _clean_str(identification.get("identification_tier"), "").lower() in {"unidentified", "none", "unknown"}
            or identification.get("identified") is False
        )
        estimation_unavailable = "ESTIMATION_UNAVAILABLE" in reason_codes or (estimation and not estimation.get("estimated"))

        if ambiguity in {"high", "critical"} or gate_decision == "ask_clarification":
            proposals.append(
                RecommendedAction(
                    action_name="ask_clarification",
                    recommendation_type="clarification_action",
                    rationale="The action has ambiguous intent or missing runtime facts; clarify before tool execution.",
                    parameters={"original_action": original, "clarify_missing_fields": True},
                    safety_constraints=["no_tool_execution_until_clarified", "tool_guard_required"],
                    execution_status="requires_user_clarification",
                    priority=88,
                )
            )

        if gate_decision in {"abstain", "warn"} and risk in {"medium", "high", "critical"} and (unidentified or estimation_unavailable):
            proposals.append(
                RecommendedAction(
                    action_name="run_small_safe_experiment_or_collect_evidence",
                    recommendation_type="evidence_collection_action",
                    rationale="Causal evidence is insufficient for the original action; collect bounded evidence before scaling execution.",
                    parameters={
                        "original_action": original,
                        "exposure": "small_segment",
                        "rollback_required": True,
                        "success_metric_required": True,
                        "harm_metric_required": True,
                    },
                    safety_constraints=[
                        "small_exposure_only",
                        "predefined_success_metric_required",
                        "predefined_harm_metric_required",
                        "rollback_required",
                        "tool_guard_required",
                    ],
                    execution_status="proposal_only_requires_gate_review",
                    priority=74,
                )
            )

        if gate_decision == "veto" and not proposals:
            proposals.append(
                RecommendedAction(
                    action_name="request_human_review_with_causal_audit",
                    recommendation_type="human_review_action",
                    rationale="The original action was vetoed; route the audit and missing evidence to review instead of executing.",
                    parameters={"original_action": original, "execute_original": False},
                    safety_constraints=["no_original_tool_execution", "human_review_required", "tool_guard_required"],
                    execution_status="requires_human_review",
                    priority=70,
                )
            )

        return proposals

    def _rank_and_dedupe(self, proposals: List[RecommendedAction]) -> List[RecommendedAction]:
        by_name: Dict[str, RecommendedAction] = {}
        for p in proposals:
            key = p.action_name.strip().lower()
            if not key:
                continue
            if key not in by_name or p.priority > by_name[key].priority:
                by_name[key] = p
        return sorted(by_name.values(), key=lambda p: p.priority, reverse=True)

    def _summary(self, original: str, decision: str, top: Mapping[str, Any], proposals: Sequence[RecommendedAction]) -> str:
        if not proposals:
            return "No safer alternative was generated. Keep the original decision and ask for more evidence or review."
        name = _clean_str(top.get("action_name"), "the recommended action")
        if decision in {"veto", "abstain"}:
            return f"Do not execute {original or 'the original action'}; propose {name} as a safer next candidate."
        if decision == "ask_clarification":
            return f"Clarify before execution; propose {name} as the next safe step."
        if decision == "warn":
            return f"Original action is risky; prefer {name} or execute only with listed mitigations."
        return f"Original action is allowed, but {name} is the guarded recommendation for execution planning."

    def _causal_inputs(self, decision: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "has_scm_id": bool(decision.get("causal_identification")),
            "has_estimation": bool(decision.get("causal_estimation")),
            "has_counterfactual": bool(decision.get("causal_counterfactual")),
            "identification_tier": decision.get("identification_tier", "unknown"),
            "evidence_tier": decision.get("evidence_tier", "unknown"),
            "reason_codes": list(decision.get("reason_codes", []) or []),
        }


def recommend_actions(action_payload: Any, decision_package: Any) -> Dict[str, Any]:
    """Convenience function returning a plain recommendation dict."""

    return CausalActionRecommender().recommend(action_payload, decision_package).to_dict()
