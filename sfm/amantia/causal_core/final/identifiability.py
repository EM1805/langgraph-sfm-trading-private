from __future__ import annotations

"""Conservative identifiability diagnostics for Structural Final Models.

The SFM layer should not treat an intent score as proof of teleology.  This
module classifies a final-cause query by the evidence that makes the claim
empirically constrained:

* diagnostic_only: useful scoring, but not enough structure to falsify intent.
* falsifiable_diagnostic: controls/twin comparisons make the claim testable.
* partially_identifiable: a real SCM graph identifies a non-zero action-goal
  effect, the twin-policy comparison is goal-dependent, side effects are
  excluded, and at least one independent validation channel is present.
* strongly_supported: partial identifiability plus agent-belief / utility / do*
  support.  This is still not metaphysical proof of final causation.

Step 23 makes this module deliberately stricter: a high diagnostic score without
a real SCM graph can still support a hypothesis, but it cannot authorize an SFM
intent claim.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from .schema import FinalCauseQuery, GoalSpec


def _as_bool(value: Any) -> bool:
    return bool(value)


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _has_control_goals(query: FinalCauseQuery) -> bool:
    return bool(query.negative_control_goals or query.placebo_goals or query.side_effect_goals)


@dataclass
class SFMIdentifiabilityAssessment:
    """Epistemic-status assessment for a candidate final cause.

    `can_claim_intent` means the diagnostic layer has enough structural,
    counterfactual, and falsification support to report an intent inference.  It
    does not mean full teleological identification.
    """

    assessed: bool = True
    tier: str = "diagnostic_only"
    authority_status: str = "diagnostic_only"
    can_claim_intent: bool = False
    is_falsifiable: bool = False
    partially_identifiable: bool = False
    strongly_supported: bool = False
    independent_validation_present: bool = False
    evidence_matrix: Dict[str, bool] = field(default_factory=dict)
    required_assumptions: List[str] = field(default_factory=list)
    failed_conditions: List[str] = field(default_factory=list)
    reason_codes: List[str] = field(default_factory=list)
    limits: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SFMIdentifiabilityEvaluator:
    """Classify the epistemic strength of an SFM final-cause claim."""

    def assess(
        self,
        query: FinalCauseQuery | Mapping[str, Any],
        goal: GoalSpec | Mapping[str, Any] | str,
        *,
        causal: Optional[Mapping[str, Any]] = None,
        counterfactual: Optional[Mapping[str, Any]] = None,
        twin: Optional[Mapping[str, Any]] = None,
        belief: Optional[Mapping[str, Any]] = None,
        falsification: Optional[Mapping[str, Any]] = None,
        utility: Optional[Mapping[str, Any]] = None,
        empirical_utility: Optional[Mapping[str, Any]] = None,
        multi_goal: Optional[Mapping[str, Any]] = None,
        do_star: Optional[Mapping[str, Any]] = None,
        side_effects_excluded: bool = False,
        zero_effect_identification: bool = False,
        observed_action_supported_by_score: bool = False,
    ) -> SFMIdentifiabilityAssessment:
        if not isinstance(query, FinalCauseQuery):
            query = FinalCauseQuery.from_payload(query)
        if not isinstance(goal, GoalSpec):
            goal = GoalSpec.from_payload(goal)

        causal = _as_dict(causal)
        counterfactual = _as_dict(counterfactual)
        twin = _as_dict(twin)
        belief = _as_dict(belief)
        falsification = _as_dict(falsification)
        utility = _as_dict(utility)
        empirical_utility = _as_dict(empirical_utility)
        multi_goal = _as_dict(multi_goal)
        do_star = _as_dict(do_star)

        causal_identified = (
            bool(causal.get("identified"))
            or str(causal.get("identification_tier") or "")
            in {"identified", "identified_graphical", "identified_recursive", "identified_canonical"}
        ) and not zero_effect_identification
        action_cf_supported = bool(counterfactual.get("compared")) and str(counterfactual.get("recommended_action") or "") == query.observed_action
        twin_compared = bool(twin.get("compared"))
        twin_selects_observed = twin_compared and bool(twin.get("observed_selected_with_goal"))
        twin_goal_dependent = twin_compared and bool(twin.get("action_changes_when_goal_removed"))
        belief_model_supplied = bool(belief.get("belief_model_supplied")) or bool(query.agent.belief_graph)
        belief_supports_intent = bool(belief.get("intent_under_agent_beliefs"))
        falsification_passed = bool(falsification.get("passed", True))
        falsification_assessed = bool(falsification.get("assessed")) or _has_control_goals(query)
        falsification_controls_supplied = _has_control_goals(query)
        utility_support = bool(utility.get("assessed")) and bool(utility.get("selected_action_matches_observed"))
        empirical_support = bool(empirical_utility.get("assessed")) and bool(empirical_utility.get("selected_action_matches_observed"))
        multi_goal_support = bool(multi_goal.get("assessed")) and bool(multi_goal.get("selected_action_matches_observed"))
        do_star_support = bool(do_star.get("evaluated")) and bool(do_star.get("selected_action_matches_observed"))
        has_action_alternatives = len(query.candidate_actions) >= 2
        has_real_graph = bool(query.scm_graph)
        empirical_channel_present = bool(empirical_support or query.outcome_records or query.outcome_log_path)
        independent_validation_present = bool(
            falsification_controls_supplied
            or (belief_model_supplied and belief_supports_intent)
            or empirical_channel_present
        )

        evidence_matrix = {
            "real_scm_graph_supplied": has_real_graph,
            "causal_action_goal_effect_identified": causal_identified,
            "zero_effect_identification": bool(zero_effect_identification),
            "candidate_action_alternatives_supplied": has_action_alternatives,
            "action_counterfactual_selects_observed": action_cf_supported,
            "twin_policy_compared": twin_compared,
            "twin_policy_selects_observed_with_goal": twin_selects_observed,
            "twin_policy_changes_when_goal_removed": twin_goal_dependent,
            "agent_belief_model_supplied": belief_model_supplied,
            "agent_beliefs_support_intent": belief_supports_intent,
            "falsification_assessed": falsification_assessed,
            "falsification_controls_supplied": falsification_controls_supplied,
            "independent_validation_present": independent_validation_present,
            "falsification_passed": falsification_passed,
            "side_effects_excluded": bool(side_effects_excluded),
            "explicit_utility_selects_observed": utility_support,
            "empirical_utility_selects_observed": empirical_support,
            "multi_goal_policy_selects_observed": multi_goal_support,
            "do_star_policy_selects_observed": do_star_support,
            "intent_score_threshold_passed": bool(observed_action_supported_by_score),
        }

        failed_conditions = [name for name, ok in evidence_matrix.items() if name not in {"zero_effect_identification"} and not ok]
        reason_codes: List[str] = []
        limits: List[str] = []
        required_assumptions = [
            "candidate_goal_set_is_complete_enough_for_comparison",
            "candidate_actions_represent_relevant_agent_options",
            "observed_action_was_available_to_the_agent",
            "outcome_scores_are_comparable_across_actions",
            "scm_graph_and_agent_belief_graph_are_not_silently_misspecified",
        ]

        if zero_effect_identification:
            reason_codes.append("SFM_IDENT_ZERO_EFFECT_BLOCKS_PARTIAL_IDENTIFICATION")
            limits.append("real_graph_suggests_zero_action_goal_effect")
        if not has_real_graph:
            reason_codes.append("SFM_IDENT_MISSING_REAL_SCM_GRAPH")
            limits.append("real_scm_graph_required_for_partial_identification")
        if not has_action_alternatives:
            reason_codes.append("SFM_IDENT_MISSING_ACTION_ALTERNATIVES")
            limits.append("action_alternatives_required_for_goal_dependence_test")
        if not twin_compared:
            reason_codes.append("SFM_IDENT_TWIN_POLICY_NOT_COMPARED")
            limits.append("twin_policy_comparison_required_for_goal_dependence")
        if not falsification_controls_supplied:
            limits.append("negative_placebo_or_side_effect_controls_recommended")
        if not independent_validation_present:
            reason_codes.append("SFM_IDENT_MISSING_INDEPENDENT_VALIDATION_CHANNEL")
            limits.append("partial_identification_requires_controls_beliefs_or_empirical_history")
        if not belief_model_supplied:
            limits.append("agent_belief_graph_recommended_for_intent_attribution")

        is_falsifiable = bool(has_action_alternatives and (twin_compared or falsification_controls_supplied or falsification_assessed))
        partially_identifiable = bool(
            has_real_graph
            and causal_identified
            and twin_selects_observed
            and twin_goal_dependent
            and side_effects_excluded
            and falsification_passed
            and has_action_alternatives
            and independent_validation_present
        )
        strong_auxiliary_support = bool(
            partially_identifiable
            and (belief_model_supplied and belief_supports_intent)
            and (utility_support or multi_goal_support or do_star_support)
            and (empirical_support or falsification_controls_supplied or do_star_support)
        )
        strongly_supported = bool(partially_identifiable and strong_auxiliary_support)

        if strongly_supported:
            tier = "strongly_supported"
            authority_status = "strong_diagnostic_sfm_support"
            can_claim_intent = True
            reason_codes.append("SFM_IDENT_STRONGLY_SUPPORTED")
        elif partially_identifiable:
            tier = "partially_identifiable"
            authority_status = "partial_sfm_identification"
            can_claim_intent = True
            reason_codes.append("SFM_IDENT_PARTIALLY_IDENTIFIABLE")
        elif is_falsifiable:
            tier = "falsifiable_diagnostic"
            authority_status = "falsifiable_diagnostic_only"
            # Falsifiability is valuable, but by itself it is not enough to
            # authorize an SFM intent claim after Step 23.  The engine will still
            # expose intent_hypothesis_supported=True when the score passes.
            can_claim_intent = False
            reason_codes.append("SFM_IDENT_FALSIFIABLE_DIAGNOSTIC")
            reason_codes.append("SFM_IDENT_FALSIFIABLE_BUT_NOT_CLAIM_AUTHORIZED")
        else:
            tier = "diagnostic_only"
            authority_status = "diagnostic_only"
            can_claim_intent = False
            reason_codes.append("SFM_IDENT_DIAGNOSTIC_ONLY")

        if not has_real_graph:
            can_claim_intent = False
            reason_codes.append("SFM_IDENT_MISSING_REAL_GRAPH_BLOCKS_CLAIM_AUTHORITY")
        if zero_effect_identification and not belief_supports_intent:
            can_claim_intent = False
        if can_claim_intent:
            reason_codes.append("SFM_IDENT_CAN_REPORT_INTENT_DIAGNOSTICALLY")
        else:
            reason_codes.append("SFM_IDENT_CANNOT_REPORT_INTENT")
            limits.append("insufficient_epistemic_support_for_intent_claim")
        if not falsification_passed:
            reason_codes.append("SFM_IDENT_FALSIFICATION_FAILED")
            limits.append("failed_falsification_blocks_intent_claim")
            can_claim_intent = False
        if not side_effects_excluded:
            reason_codes.append("SFM_IDENT_SIDE_EFFECT_NOT_EXCLUDED")
            limits.append("side_effect_or_protected_outcome_not_excluded")
            can_claim_intent = False

        return SFMIdentifiabilityAssessment(
            assessed=True,
            tier=tier,
            authority_status=authority_status,
            can_claim_intent=can_claim_intent,
            is_falsifiable=is_falsifiable,
            partially_identifiable=partially_identifiable,
            strongly_supported=strongly_supported,
            independent_validation_present=independent_validation_present,
            evidence_matrix=evidence_matrix,
            required_assumptions=required_assumptions,
            failed_conditions=failed_conditions,
            reason_codes=reason_codes,
            limits=sorted(set(limits)),
        )


def assess_sfm_identifiability(payload: Any) -> Dict[str, Any]:
    """Standalone lightweight assessment for a raw SFM query.

    Without engine-generated evidence this reports only whether the query has
    enough surface structure to become falsifiable.  Full assessment is exposed
    through `FinalCauseEngine.infer`, which passes SCM-ID/twin/utility evidence.
    """

    query = FinalCauseQuery.from_payload(payload)
    goal = query.candidate_goals[0] if query.candidate_goals else GoalSpec(goal_variable="")
    return SFMIdentifiabilityEvaluator().assess(query, goal).to_dict()
