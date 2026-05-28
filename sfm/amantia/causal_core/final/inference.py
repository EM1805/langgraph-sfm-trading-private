from __future__ import annotations

"""First-pass final-cause inference scaffold.

This layer composes existing Amantia SCM-ID and counterfactual adapters.  It is
meant as a development bridge toward SFM, not a claim of complete teleological
identification.
"""

from typing import Any, Dict, Mapping, Optional

from amantia.causal_core.counterfactual import CounterfactualEngine
from amantia.causal_core.identification import IdentificationEngine

from .alignment_summary import SFMAlignmentSummarizer
from .belief_model import AgentBeliefEvaluator
from .constraint import ConstraintAwareEvaluator
from .context_conditioning import ContextConditioningEvaluator
from .do_star import DoStarOperator
from .empirical_utility import EmpiricalUtilityLearner
from .execution import SFMExecutionPlan
from .runner import SFMExecutionRunner
from .falsification import SFMFalsificationAuditor
from .goal_discovery import GoalDiscoveryEngine
from .identifiability import SFMIdentifiabilityEvaluator
from .hierarchical import HierarchicalGoalEvaluator
from .intentional_intervention import build_intentional_intervention
from .multi_goal import MultiGoalUtilityEvaluator
from .normative import NormativeSFMEvaluator
from .recommendation import SFMActionRecommender
from .reporting import SFMAuditReportGenerator
from .robustness import RobustSFMEvaluator
from .policy_learning import PolicyLearningEngine
from .temporal import TemporalGoalDriftDetector
from .schema import FinalCauseQuery, FinalCauseResult, GoalSpec
from .twin_model import TwinPolicyComparator
from .utility import UtilityFunctionEvaluator


def _action_name(row: Mapping[str, Any]) -> str:
    return str(row.get("action") or row.get("action_name") or row.get("candidate_action") or row.get("name") or "").strip()


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _reason_code_text(value: Any) -> str:
    if isinstance(value, (list, tuple, set)):
        return "|".join(str(item) for item in value)
    return str(value or "")


def _is_zero_effect_identification(causal: Mapping[str, Any], belief: Mapping[str, Any]) -> bool:
    text = "|".join([
        _reason_code_text(causal.get("reason_codes")),
        _reason_code_text(causal.get("reason")),
        _reason_code_text(causal.get("identification_strategy")),
        _reason_code_text(causal.get("identification_status")),
    ]).upper()
    return "NO_DIRECTED_PATH" in text or "ZERO_EFFECT" in text or belief.get("real_has_path") is False


class FinalCauseEngine:
    """Diagnostic SFM development facade.

    Current evidence signals:
    1. SCM-ID support that the action can affect the candidate goal.
    2. Counterfactual support that the observed action is preferred for that
       goal relative to alternatives.
    3. A simple side-effect exclusion check: protected/side-effect outcomes are
       not treated as the most likely goal.
    """

    def __init__(
        self,
        *,
        identification_engine: Optional[Any] = None,
        counterfactual_engine: Optional[Any] = None,
    ) -> None:
        self.identification_engine = identification_engine or IdentificationEngine()
        self.counterfactual_engine = counterfactual_engine or CounterfactualEngine()
        self.twin_policy_comparator = TwinPolicyComparator()
        self.alignment_summarizer = SFMAlignmentSummarizer()
        self.agent_belief_evaluator = AgentBeliefEvaluator()
        self.falsification_auditor = SFMFalsificationAuditor(twin_policy_comparator=self.twin_policy_comparator)
        self.utility_evaluator = UtilityFunctionEvaluator()
        self.empirical_utility_learner = EmpiricalUtilityLearner()
        self.multi_goal_evaluator = MultiGoalUtilityEvaluator()
        self.do_star_operator = DoStarOperator(
            multi_goal_evaluator=self.multi_goal_evaluator,
            utility_evaluator=self.utility_evaluator,
        )
        self.sfm_identifiability_evaluator = SFMIdentifiabilityEvaluator()
        self.goal_discovery_engine = GoalDiscoveryEngine()
        self.policy_learning_engine = PolicyLearningEngine()
        self.temporal_goal_drift_detector = TemporalGoalDriftDetector(policy_learning_engine=self.policy_learning_engine)
        self.context_conditioning_evaluator = ContextConditioningEvaluator(policy_learning_engine=self.policy_learning_engine)
        self.hierarchical_goal_evaluator = HierarchicalGoalEvaluator()
        self.constraint_evaluator = ConstraintAwareEvaluator()
        self.normative_evaluator = NormativeSFMEvaluator()
        self.action_recommender = SFMActionRecommender()
        self.robustness_evaluator = RobustSFMEvaluator()
        self.audit_report_generator = SFMAuditReportGenerator()

    def _identify_goal_effect(self, query: FinalCauseQuery, goal: GoalSpec) -> Dict[str, Any]:
        if not query.scm_graph:
            return {
                "identified": False,
                "identification_tier": "missing_graph",
                "reason_codes": ["SFM_MISSING_SCM_GRAPH"],
            }
        payload = {
            "scm_graph": query.scm_graph,
            "treatment": query.action_variable,
            "outcome": goal.goal_variable,
            "query_id": query.query_id,
            "source": "final_cause.identify_goal_effect",
        }
        result = self.identification_engine.identify(payload)
        return result.to_dict() if hasattr(result, "to_dict") else dict(result or {})

    def _compare_goal_actions(self, query: FinalCauseQuery, goal: GoalSpec) -> Dict[str, Any]:
        if len(query.candidate_actions) < 2:
            return {
                "compared": False,
                "comparison_status": "insufficient_alternatives",
                "reason_codes": ["SFM_INSUFFICIENT_ACTION_ALTERNATIVES"],
            }
        payload = {
            "current_action": query.observed_action,
            "candidate_actions": query.candidate_actions,
            "outcome": goal.goal_variable,
            "protected_outcome": query.protected_outcome,
            "source": "final_cause.compare_goal_actions",
        }
        result = self.counterfactual_engine.compare(payload)
        return result.to_dict() if hasattr(result, "to_dict") else dict(result or {})

    def infer(self, payload: Any) -> FinalCauseResult:
        query = FinalCauseQuery.from_payload(payload)
        execution_plan = SFMExecutionPlan.from_query(query)

        runner = SFMExecutionRunner(execution_plan)
        run_layer = runner.run

        discovery_used_for_inference = False
        goal_discovery = run_layer(
            "goal_discovery",
            lambda: self.goal_discovery_engine.discover(query, used_for_inference=False).to_dict(),
        )
        if not query.candidate_goals:
            candidate_goal_payloads = goal_discovery.get("selected_goals") or []
            if candidate_goal_payloads:
                query.candidate_goals = [GoalSpec.from_payload(item) for item in candidate_goal_payloads]
                discovery_used_for_inference = True
                goal_discovery = run_layer("goal_discovery", lambda: self.goal_discovery_engine.discover(query, used_for_inference=True).to_dict())
            else:
                return FinalCauseResult(
                    inferred=False,
                    observed_action=query.observed_action,
                    goal_discovery_support=goal_discovery,
                    reason="No candidate goals were supplied and goal discovery could not produce a usable candidate.",
                    reason_codes=["SFM_NO_CANDIDATE_GOALS", *(goal_discovery.get("reason_codes") or [])],
                    limits=["candidate_goal_set_required", *(goal_discovery.get("limits") or [])],
                    raw=query.to_dict(),
                    execution_profile_support=execution_plan.to_dict(),
                )

        best: FinalCauseResult | None = None
        multi_goal = run_layer("multi_goal", lambda: self.multi_goal_evaluator.evaluate(query).to_dict())
        do_star = run_layer("do_star", lambda: self.do_star_operator.evaluate(query).to_dict())
        policy_learning = run_layer("policy_learning", lambda: self.policy_learning_engine.evaluate(query).to_dict())
        temporal_goal_drift = run_layer("temporal_drift", lambda: self.temporal_goal_drift_detector.evaluate(query).to_dict())
        context_conditioning = run_layer("context_conditioning", lambda: self.context_conditioning_evaluator.evaluate(query).to_dict())
        hierarchical_goal = run_layer("hierarchical_goal", lambda: self.hierarchical_goal_evaluator.evaluate(query).to_dict())
        constraint_support = run_layer("constraint", lambda: self.constraint_evaluator.evaluate(query).to_dict())
        action_recommendation = run_layer("recommendation", lambda: self.action_recommender.recommend(query).to_dict())
        recommendation_selects_observed = bool(action_recommendation.get("assessed")) and bool(action_recommendation.get("recommendation_matches_observed"))
        recommendation_status = str(action_recommendation.get("recommendation_status") or "")
        recommendation_goal_bundle = set(action_recommendation.get("goal_bundle") or [])
        recommendation_blocked_actions = set(action_recommendation.get("blocked_actions") or [])
        multi_goal_selects_observed = bool(multi_goal.get("assessed")) and bool(multi_goal.get("selected_action_matches_observed"))
        do_star_selects_observed = bool(do_star.get("evaluated")) and bool(do_star.get("selected_action_matches_observed"))
        constraint_selects_observed = bool(constraint_support.get("assessed")) and bool(constraint_support.get("selected_action_matches_observed"))
        for goal in query.candidate_goals:
            causal = run_layer("identification", lambda: self._identify_goal_effect(query, goal))
            cf = run_layer("counterfactual", lambda: self._compare_goal_actions(query, goal))
            twin = run_layer("twin_model", lambda: self.twin_policy_comparator.compare(query, goal).to_dict())
            belief = run_layer("belief_model", lambda: self.agent_belief_evaluator.assess(query, goal).to_dict())
            falsification = run_layer("falsification", lambda: self.falsification_auditor.audit(query, goal).to_dict())
            utility = run_layer("utility", lambda: self.utility_evaluator.evaluate(query, goal).to_dict())
            empirical_utility = run_layer("empirical_utility", lambda: self.empirical_utility_learner.evaluate(query, goal).to_dict())
            identified_raw = bool(causal.get("identified")) or str(causal.get("identification_tier")) in {
                "identified",
                "identified_graphical",
                "identified_recursive",
                "identified_canonical",
            }
            zero_effect_identification = _is_zero_effect_identification(causal, belief)
            identified = identified_raw and not zero_effect_identification
            current_preferred = (
                bool(cf.get("compared"))
                and str(cf.get("recommended_action") or "") == query.observed_action
                and str(cf.get("comparison_status") or "") == "current_action_preferred"
            )
            twin_selects_observed = bool(twin.get("compared")) and bool(twin.get("observed_selected_with_goal"))
            twin_goal_dependent = bool(twin.get("compared")) and bool(twin.get("action_changes_when_goal_removed"))
            belief_goal_supported = bool(belief.get("intent_under_agent_beliefs"))
            utility_selects_observed = bool(utility.get("assessed")) and bool(utility.get("selected_action_matches_observed"))
            empirical_selects_observed = bool(empirical_utility.get("assessed")) and bool(empirical_utility.get("selected_action_matches_observed"))
            policy_goal_evidence = next(
                (row for row in (policy_learning.get("goal_evidence") or []) if str(row.get("goal_variable") or "") == goal.goal_variable),
                {},
            )
            policy_learning_supports_goal = bool(policy_learning.get("assessed")) and (
                str(policy_learning.get("most_likely_goal") or "") == goal.goal_variable
                or int(policy_goal_evidence.get("rank") or 9999) == 1
            )
            temporal_drift_assessed = bool(temporal_goal_drift.get("assessed"))
            temporal_final_goal_matches = temporal_drift_assessed and str(temporal_goal_drift.get("final_goal") or "") == goal.goal_variable
            temporal_dominant_goal_matches = temporal_drift_assessed and str(temporal_goal_drift.get("dominant_goal") or "") == goal.goal_variable
            temporal_drift_detected = bool(temporal_goal_drift.get("drift_detected"))
            context_conditioning_assessed = bool(context_conditioning.get("assessed"))
            context_conditioning_detected = bool(context_conditioning.get("context_conditioning_detected"))
            context_current_goal_matches = context_conditioning_assessed and str(context_conditioning.get("current_context_goal") or "") == goal.goal_variable
            context_goal_matches_some_bucket = context_conditioning_assessed and goal.goal_variable in set((context_conditioning.get("dominant_goal_by_context") or {}).values())
            hierarchical_assessed = bool(hierarchical_goal.get("assessed"))
            hierarchical_profile = next(
                (row for row in (hierarchical_goal.get("goal_profiles") or []) if str(row.get("goal_variable") or "") == goal.goal_variable),
                {},
            )
            hierarchical_role = str(hierarchical_profile.get("role") or "")
            hierarchical_selected_ultimate_matches = hierarchical_assessed and str(hierarchical_goal.get("selected_ultimate_goal") or "") == goal.goal_variable
            hierarchical_selected_goal_matches = hierarchical_assessed and str(hierarchical_goal.get("selected_hierarchical_goal") or "") == goal.goal_variable
            hierarchical_instrumental_for_selected_ultimate = hierarchical_assessed and goal.goal_variable in set(
                (hierarchical_goal.get("ultimate_goal_by_instrument") or {}).keys()
            )
            constraint_assessed = bool(constraint_support.get("assessed"))
            observed_constraint_feasible = bool(constraint_support.get("observed_feasible")) if constraint_assessed else True
            constraint_like_candidate = goal.goal_variable in set(constraint_support.get("constraint_like_candidate_goals") or [])
            normative = run_layer(
                "normative",
                lambda: self.normative_evaluator.evaluate(
                    query,
                    goal,
                    constraint_support=constraint_support,
                    hierarchical_goal=hierarchical_goal,
                ).to_dict(),
            )
            normative_assessed = bool(normative.get("assessed"))
            normative_aligned = bool(normative.get("normatively_aligned"))
            normative_prohibited = bool(normative.get("prohibited"))
            normative_requires_escalation = bool(normative.get("requires_escalation"))
            normative_protected_goal_like = bool(normative.get("protected_goal_like"))
            protection_policy = _as_dict(constraint_support.get("normalized_protection_policy"))
            protected = {query.protected_outcome, *goal.protected_outcomes, *goal.side_effect_outcomes}
            protected.update(constraint_support.get("protected_constraints") or [])
            protected.update(constraint_support.get("hard_constraints") or [])
            protected.update(constraint_support.get("side_effect_outcomes") or [])
            protected.update(protection_policy.get("protected_outcomes") or [])
            protected.update(protection_policy.get("hard_constraints") or [])
            protected.update(protection_policy.get("side_effect_outcomes") or [])
            protected.update(protection_policy.get("protected_goals") or [])
            side_effects_excluded = goal.goal_variable not in protected and not constraint_like_candidate

            score = 0.0
            if identified:
                score += 0.30
            if current_preferred:
                score += 0.20
            if twin_selects_observed:
                score += 0.20
            if twin_goal_dependent:
                score += 0.20
            if belief_goal_supported:
                score += 0.10
            if utility_selects_observed:
                score += 0.10
            if empirical_selects_observed:
                # Historical action/outcome evidence is weaker than structural support,
                # but it is useful as an implicit-utility consistency check.
                score += 0.10 * float(empirical_utility.get("support_strength", 1.0) or 1.0)
            if multi_goal_selects_observed:
                # Multi-goal support is an extra policy-level signal: the action
                # may be selected by a bundle of ends even when no single goal
                # fully explains it.
                score += 0.10 * float(multi_goal.get("support_strength", 1.0) or 1.0)
            if do_star_selects_observed:
                # The formal do* operator is primarily a serialization and
                # policy-surface check, so it gets a bounded diagnostic bump.
                score += 0.08 * float(do_star.get("support_strength", 1.0) or 1.0)
            if constraint_selects_observed:
                # Constraint-aware support says the observed action is selected
                # after hard/protected constraints are enforced, not merely by
                # unconstrained goal maximization.
                score += 0.08 * float(constraint_support.get("support_strength", 1.0) or 1.0)
            if normative_assessed and normative_aligned:
                # Normative support is not evidence that the goal was pursued; it
                # is a value-alignment classification.  Give only a tiny bounded
                # bump when the pursued-goal hypothesis is also allowed/required.
                score += 0.03 * float(normative.get("support_strength", 1.0) or 1.0)
            if discovery_used_for_inference:
                for discovered_goal in goal_discovery.get("selected_goals") or []:
                    if str(discovered_goal.get("goal_variable") or "") == goal.goal_variable:
                        score += 0.05 * float(discovered_goal.get("metadata", {}).get("discovery_score", 1.0) or 1.0)
                        break
            if policy_learning_supports_goal:
                # Sequence-level inverse goal inference is diagnostic evidence: it
                # says the same candidate goal explains repeated choices over time.
                score += 0.10 * float(policy_goal_evidence.get("support_strength", policy_learning.get("support_strength", 1.0)) or 1.0)
            elif bool(policy_learning.get("assessed")) and policy_goal_evidence:
                # A non-top but still bundle-compatible goal gets a smaller bump.
                if float(policy_goal_evidence.get("support_strength", 0.0) or 0.0) >= 0.5:
                    score += 0.04 * float(policy_goal_evidence.get("support_strength", 0.0) or 0.0)
            if temporal_drift_assessed and (temporal_final_goal_matches or temporal_dominant_goal_matches):
                # Temporal drift evidence is weak but useful: it tells us whether
                # the current candidate goal is the current/dominant inferred telos.
                score += 0.06 * float(temporal_goal_drift.get("support_strength", 1.0) or 1.0)
            if context_current_goal_matches:
                # Context-conditioned SFM can explain an apparent temporal drift
                # as a stable policy whose telos depends on observed context.
                score += 0.08 * float(context_conditioning.get("current_context_support", context_conditioning.get("support_strength", 1.0)) or 1.0)
            elif context_goal_matches_some_bucket:
                score += 0.03 * float(context_conditioning.get("support_strength", 1.0) or 1.0)
            if hierarchical_selected_ultimate_matches:
                # Hierarchical SFM distinguishes terminal ends from means.  A
                # terminal goal selected by a means-end hierarchy gets a small
                # diagnostic bump because the observed action may optimize an
                # instrument in service of this higher-level telos.
                score += 0.08 * float(hierarchical_goal.get("support_strength", 1.0) or 1.0)
            elif hierarchical_selected_goal_matches:
                score += 0.04 * float(hierarchical_profile.get("support_strength", hierarchical_goal.get("support_strength", 1.0)) or 1.0)
            elif hierarchical_instrumental_for_selected_ultimate:
                score += 0.02 * float(hierarchical_profile.get("support_strength", 1.0) or 1.0)
            if hierarchical_role in {"instrumental_goal", "intermediate_goal"} and not hierarchical_selected_ultimate_matches:
                # Means can be intentional, but this layer should not promote a
                # means over its stated terminal end without extra evidence.
                score = min(score, 0.87)
            recommendation_contains_goal = goal.goal_variable in recommendation_goal_bundle
            if (
                bool(action_recommendation.get("assessed"))
                and recommendation_contains_goal
                and hierarchical_role not in {"instrumental_goal", "intermediate_goal"}
            ):
                if recommendation_selects_observed:
                    # Step 17 turns the SFM evidence stack into a forward action
                    # recommendation.  If that policy selects the observed action,
                    # it is weak but useful diagnostic support for the candidate telos.
                    score += 0.08 * float(action_recommendation.get("support_strength", 1.0) or 1.0)
                else:
                    score += 0.02 * float(action_recommendation.get("support_strength", 0.0) or 0.0)
            if constraint_assessed and not observed_constraint_feasible:
                # A hard/protected violation blocks strong claims that the
                # observed action was selected by a constraint-respecting telos.
                score = min(score, 0.39)
            if (
                bool(action_recommendation.get("assessed"))
                and query.observed_action in recommendation_blocked_actions
                and not normative_prohibited
                and not normative_requires_escalation
            ):
                # The forward SFM recommender treats the observed intervention as
                # infeasible under non-normative constraints.  Normative prohibition
                # is handled by the governance summary and must not erase evidence
                # that a prohibited goal was pursued.
                score = min(score, 0.39)
            if constraint_like_candidate:
                # Outcomes declared as constraints or side effects are not
                # promoted to final causes by this layer.
                score = min(score, 0.49)
            if normative_protected_goal_like:
                # A protected normative outcome can be intended as a constraint,
                # but should not be promoted to a terminal final cause here.
                score = min(score, 0.49)
            if temporal_drift_detected and not temporal_final_goal_matches and not context_current_goal_matches:
                # If a drift audit suggests the agent ended elsewhere, cap this
                # candidate unless context conditioning explains why this goal is
                # still the right telos for the current context.
                score = min(score, 0.79)
            if side_effects_excluded:
                score += 0.10
            else:
                # Protected outcomes and declared side effects can be effects of
                # an action, but this diagnostic layer must not promote them to
                # final causes without a stronger agent-belief model.
                score = min(score, 0.49)
            falsification_multiplier = float(falsification.get("intent_score_multiplier", 1.0) or 1.0)
            score = round(min(score, 1.0) * falsification_multiplier, 6)

            reason_codes = list(execution_plan.reason_codes)
            if identified:
                reason_codes.append("SFM_GOAL_EFFECT_IDENTIFIED")
            elif zero_effect_identification:
                reason_codes.append("SFM_REAL_GRAPH_SUGGESTS_ZERO_GOAL_EFFECT")
            else:
                reason_codes.append("SFM_GOAL_EFFECT_NOT_IDENTIFIED")
            if current_preferred:
                reason_codes.append("SFM_ACTION_PREFERS_GOAL")
            else:
                reason_codes.append("SFM_ACTION_NOT_SHOWN_GOAL_OPTIMAL")
            reason_codes.extend(twin.get("reason_codes") or [])
            reason_codes.extend(belief.get("reason_codes") or [])
            reason_codes.extend(falsification.get("reason_codes") or [])
            reason_codes.extend(utility.get("reason_codes") or [])
            reason_codes.extend(empirical_utility.get("reason_codes") or [])
            reason_codes.extend(multi_goal.get("reason_codes") or [])
            reason_codes.extend(do_star.get("reason_codes") or [])
            reason_codes.extend(policy_learning.get("reason_codes") or [])
            reason_codes.extend(temporal_goal_drift.get("reason_codes") or [])
            reason_codes.extend(context_conditioning.get("reason_codes") or [])
            reason_codes.extend(hierarchical_goal.get("reason_codes") or [])
            reason_codes.extend(constraint_support.get("reason_codes") or [])
            reason_codes.extend(normative.get("reason_codes") or [])
            reason_codes.extend(action_recommendation.get("reason_codes") or [])
            if bool(action_recommendation.get("assessed")):
                if recommendation_contains_goal and recommendation_selects_observed:
                    reason_codes.append("SFM_ACTION_RECOMMENDATION_SUPPORTS_OBSERVED_ACTION_FOR_CANDIDATE_GOAL")
                elif recommendation_contains_goal:
                    reason_codes.append("SFM_ACTION_RECOMMENDATION_FAVORS_DIFFERENT_ACTION_FOR_CANDIDATE_GOAL")
            if context_current_goal_matches:
                reason_codes.append("SFM_CONTEXT_CURRENT_BUCKET_SUPPORTS_CANDIDATE_GOAL")
            elif context_goal_matches_some_bucket:
                reason_codes.append("SFM_CONTEXT_OTHER_BUCKET_SUPPORTS_CANDIDATE_GOAL")
            if context_conditioning_detected and temporal_drift_detected:
                reason_codes.append("SFM_CONTEXT_CONDITIONING_MAY_EXPLAIN_TEMPORAL_DRIFT")
            if hierarchical_assessed:
                if hierarchical_selected_ultimate_matches:
                    reason_codes.append("SFM_HIERARCHY_SUPPORTS_CANDIDATE_AS_FINAL_GOAL")
                elif hierarchical_selected_goal_matches:
                    reason_codes.append("SFM_HIERARCHY_SUPPORTS_CANDIDATE_AS_SELECTED_GOAL")
                elif hierarchical_role in {"instrumental_goal", "intermediate_goal"}:
                    reason_codes.append("SFM_HIERARCHY_CANDIDATE_IS_INSTRUMENTAL")
            if constraint_assessed:
                if constraint_selects_observed:
                    reason_codes.append("SFM_CONSTRAINT_AWARE_SUPPORTS_OBSERVED_ACTION")
                if not observed_constraint_feasible:
                    reason_codes.append("SFM_CONSTRAINT_AWARE_OBSERVED_ACTION_INFEASIBLE")
                if constraint_like_candidate:
                    reason_codes.append("SFM_CANDIDATE_GOAL_CLASSIFIED_AS_CONSTRAINT_NOT_FINAL_GOAL")
            if normative_assessed:
                if normative_aligned:
                    reason_codes.append("SFM_NORMATIVE_VALUE_ALIGNMENT_SUPPORTS_CANDIDATE_GOAL")
                if normative_prohibited:
                    reason_codes.append("SFM_NORMATIVE_POLICY_FLAGS_CANDIDATE_GOAL_OR_ACTION")
                if normative_requires_escalation:
                    reason_codes.append("SFM_NORMATIVE_POLICY_REQUIRES_ESCALATION")
                if normative_protected_goal_like:
                    reason_codes.append("SFM_NORMATIVE_CANDIDATE_IS_PROTECTED_OUTCOME_NOT_TERMINAL_GOAL")
            if policy_learning_supports_goal:
                reason_codes.append("SFM_POLICY_LEARNING_SUPPORTS_CANDIDATE_GOAL")
            elif bool(policy_learning.get("assessed")) and policy_goal_evidence:
                reason_codes.append("SFM_POLICY_LEARNING_DOES_NOT_SELECT_CANDIDATE_GOAL")
            if bool(temporal_goal_drift.get("assessed")):
                if str(temporal_goal_drift.get("final_goal") or "") == goal.goal_variable:
                    reason_codes.append("SFM_TEMPORAL_FINAL_WINDOW_SUPPORTS_CANDIDATE_GOAL")
                elif bool(temporal_goal_drift.get("drift_detected")):
                    reason_codes.append("SFM_TEMPORAL_DRIFT_FAVORS_DIFFERENT_FINAL_GOAL")
            if side_effects_excluded:
                reason_codes.append("SFM_SIDE_EFFECT_EXCLUDED")
            else:
                reason_codes.append("SFM_GOAL_OVERLAPS_PROTECTED_OR_SIDE_EFFECT")

            support_level = "high" if score >= 0.8 else "moderate" if score >= 0.6 else "low" if score > 0 else "none"
            falsification_passed = bool(falsification.get("passed", True))
            preliminary_inferred = (
                score >= query.min_intent_score
                and side_effects_excluded
                and falsification_passed
                and (not constraint_assessed or observed_constraint_feasible)
            )
            sfm_identifiability = run_layer(
                "identifiability",
                lambda: self.sfm_identifiability_evaluator.assess(
                    query,
                    goal,
                    causal=causal,
                    counterfactual=cf,
                    twin=twin,
                    belief=belief,
                    falsification=falsification,
                    utility=utility,
                    empirical_utility=empirical_utility,
                    multi_goal=multi_goal,
                    do_star=do_star,
                    side_effects_excluded=side_effects_excluded,
                    zero_effect_identification=zero_effect_identification,
                    observed_action_supported_by_score=preliminary_inferred,
                ).to_dict(),
            )
            intent_hypothesis_supported = bool(preliminary_inferred)
            if execution_plan.is_enabled("identifiability"):
                intent_claim_authorized = bool(
                    intent_hypothesis_supported
                    and sfm_identifiability.get("can_claim_intent", False)
                )
            else:
                # Without the identifiability layer, Step 23 keeps the diagnostic
                # hypothesis but does not authorize a causal-intent claim.
                intent_claim_authorized = False
                reason_codes.append("SFM_IDENTIFIABILITY_LAYER_DISABLED_CLAIM_AUTHORITY_WITHHELD")
            inferred = intent_claim_authorized
            robustness = run_layer(
                "robustness",
                lambda: self.robustness_evaluator.evaluate(
                    query,
                    goal,
                    intent_score=score,
                    intent_supported=inferred,
                    falsification_passed=falsification_passed,
                    sfm_identifiability_support=sfm_identifiability,
                    constraint_support=constraint_support,
                    normative_support=normative,
                    action_recommendation_support=action_recommendation,
                ).to_dict(),
            )
            reason_codes.extend(sfm_identifiability.get("reason_codes") or [])
            reason_codes.extend(robustness.get("reason_codes") or [])
            reason_codes.extend(goal_discovery.get("reason_codes") or [])
            if intent_hypothesis_supported:
                reason_codes.append("SFM_INTENT_HYPOTHESIS_SUPPORTED_DIAGNOSTICALLY")
            else:
                reason_codes.append("SFM_INTENT_HYPOTHESIS_NOT_SUPPORTED")
            if intent_claim_authorized:
                reason_codes.append("SFM_INTENT_CLAIM_AUTHORIZED")
            else:
                reason_codes.append("SFM_INTENT_CLAIM_NOT_AUTHORIZED")
            if discovery_used_for_inference:
                reason_codes.append("SFM_GOAL_DISCOVERY_BOOTSTRAPPED_CANDIDATE_GOALS")
            limits = [
                "diagnostic_only_not_full_sfm_identification",
                "twin_policy_comparison_is_diagnostic_not_full_structural_counterfactual",
            ]
            if intent_hypothesis_supported and not intent_claim_authorized:
                limits.append("intent_hypothesis_supported_but_claim_not_authorized")
            if not query.scm_graph:
                limits.append("real_scm_graph_not_supplied_claim_authority_withheld")
            if not belief.get("belief_model_supplied"):
                limits.append("agent_belief_graph_not_supplied")
            if belief.get("belief_error_type") in {"false_positive_belief", "false_negative_belief"}:
                limits.append("agent_belief_graph_diverges_from_real_graph")
            if zero_effect_identification:
                limits.append("real_graph_suggests_zero_action_goal_effect")
            limits.extend(falsification.get("limits") or [])
            limits.extend(utility.get("limits") or [])
            limits.extend(empirical_utility.get("limits") or [])
            limits.extend(multi_goal.get("limits") or [])
            limits.extend(do_star.get("limits") or [])
            limits.extend(policy_learning.get("limits") or [])
            limits.extend(temporal_goal_drift.get("limits") or [])
            limits.extend(context_conditioning.get("limits") or [])
            limits.extend(hierarchical_goal.get("limits") or [])
            limits.extend(constraint_support.get("limits") or [])
            limits.extend(normative.get("limits") or [])
            limits.extend(action_recommendation.get("limits") or [])
            limits.extend(robustness.get("limits") or [])
            limits.extend(sfm_identifiability.get("limits") or [])
            limits.extend(goal_discovery.get("limits") or [])
            if discovery_used_for_inference:
                limits.append("candidate_goals_were_discovered_not_user_supplied")
            if empirical_utility.get("assessed") and not empirical_selects_observed:
                limits.append("empirical_utility_does_not_select_observed_action")
            if multi_goal.get("assessed") and not multi_goal_selects_observed:
                limits.append("multi_goal_utility_does_not_select_observed_action")
            if do_star.get("evaluated") and not do_star_selects_observed:
                limits.append("do_star_policy_does_not_select_observed_action")
            if policy_learning.get("assessed") and policy_goal_evidence and not policy_learning_supports_goal:
                limits.append("policy_learning_sequence_favors_different_goal")
            if temporal_goal_drift.get("assessed") and temporal_goal_drift.get("drift_detected"):
                limits.append("temporal_goal_drift_detected")
                if context_conditioning_detected:
                    limits.append("temporal_drift_may_be_context_conditioned_policy")
            if context_conditioning.get("assessed") and not context_conditioning_detected:
                limits.append("context_conditioning_does_not_explain_goal_pattern")
            if utility.get("assessed") and not utility_selects_observed:
                limits.append("explicit_utility_function_does_not_select_observed_action")
            if hierarchical_assessed and hierarchical_role in {"instrumental_goal", "intermediate_goal"}:
                limits.append("candidate_goal_may_be_instrumental_not_terminal_final_cause")
            if constraint_assessed and not observed_constraint_feasible:
                limits.append("observed_action_violates_hard_or_protected_constraints")
            if constraint_like_candidate:
                limits.append("candidate_goal_classified_as_constraint_or_side_effect_not_final_goal")
            if normative_assessed and normative_prohibited:
                limits.append("normative_policy_flags_goal_or_action_as_not_allowed")
            if normative_assessed and normative_requires_escalation:
                limits.append("normative_policy_requires_escalation_for_goal_or_action")
            if normative_assessed and normative_protected_goal_like:
                limits.append("normative_policy_classifies_candidate_as_protected_outcome")
            if bool(action_recommendation.get("assessed")) and recommendation_contains_goal and not recommendation_selects_observed:
                limits.append("sfm_action_recommendation_favors_different_action")
            if bool(action_recommendation.get("assessed")) and recommendation_status == "requires_escalation":
                limits.append("sfm_recommended_action_requires_escalation")
            if bool(action_recommendation.get("assessed")) and query.observed_action in recommendation_blocked_actions:
                limits.append("observed_action_blocked_by_sfm_recommendation_layer")
            if bool(robustness.get("assessed")) and robustness.get("uncertainty_review_required"):
                limits.append("sfm_robustness_audit_requires_uncertainty_review")
            if bool(robustness.get("assessed")) and not robustness.get("robust_to_uncertainty"):
                limits.append("sfm_claim_not_robust_under_pessimistic_uncertainty")
            if not falsification_passed:
                limits.append("sfm_falsification_failed")
            intervention = build_intentional_intervention(
                action_variable=query.action_variable,
                selected_action=query.observed_action,
                goal=goal,
                agent=query.agent,
            ).to_dict()
            if do_star.get("expression"):
                intervention["formal_expression"] = do_star.get("expression")
                intervention["policy_signature"] = do_star.get("policy_signature")
                intervention["selected_by_policy"] = do_star.get("selected_action")
            alignment_summary = run_layer(
                "alignment_summary",
                lambda: self.alignment_summarizer.summarize(
                    query,
                    goal,
                    intent_supported=inferred,
                    intent_score=score,
                    support_level=support_level,
                    authority_status=str(sfm_identifiability.get("authority_status") or "diagnostic_only"),
                    falsification_passed=falsification_passed,
                    side_effects_excluded=side_effects_excluded,
                    constraint_support=constraint_support,
                    normative_support=normative,
                    sfm_identifiability_support=sfm_identifiability,
                    action_recommendation_support=action_recommendation,
                    robustness_support=robustness,
                    reason_codes=reason_codes,
                    limits=limits,
                ).to_dict(),
            )
            report_payload = {
                "inferred": inferred,
                "intent_hypothesis_supported": intent_hypothesis_supported,
                "intent_claim_authorized": intent_claim_authorized,
                "most_likely_goal": goal.goal_variable,
                "observed_action": query.observed_action,
                "intent_score": score,
                "support_level": support_level,
                "authority_status": str(sfm_identifiability.get("authority_status") or "diagnostic_only"),
                "intentional_intervention": intervention,
                "causal_support": causal,
                "counterfactual_support": cf,
                "twin_support": twin,
                "belief_support": belief,
                "falsification_support": falsification,
                "utility_support": utility,
                "empirical_utility_support": empirical_utility,
                "multi_goal_support": multi_goal,
                "do_star_support": do_star,
                "sfm_identifiability_support": sfm_identifiability,
                "goal_discovery_support": goal_discovery,
                "policy_learning_support": policy_learning,
                "temporal_goal_drift_support": temporal_goal_drift,
                "context_conditioning_support": context_conditioning,
                "hierarchical_goal_support": hierarchical_goal,
                "constraint_support": constraint_support,
                "normative_support": normative,
                "action_recommendation_support": action_recommendation,
                "robustness_support": robustness,
                "alignment_summary": alignment_summary,
                "falsification_passed": falsification_passed,
                "side_effects_excluded": side_effects_excluded,
                "reason_codes": reason_codes,
                "limits": limits,
            }
            audit_report = run_layer(
                "audit_report",
                lambda: self.audit_report_generator.generate(report_payload).to_dict(),
            )
            reason_codes.extend(alignment_summary.get("reason_codes") or [])
            reason_codes.extend(audit_report.get("reason_codes") or [])
            result = FinalCauseResult(
                inferred=inferred,
                intent_hypothesis_supported=intent_hypothesis_supported,
                intent_claim_authorized=intent_claim_authorized,
                governance_execution_allowed=bool(alignment_summary.get("allow_execution", False)),
                most_likely_goal=goal.goal_variable,
                observed_action=query.observed_action,
                intent_score=score,
                intentional_intervention=intervention,
                causal_support=causal,
                counterfactual_support=cf,
                twin_support=twin,
                belief_support=belief,
                falsification_support=falsification,
                utility_support=utility,
                empirical_utility_support=empirical_utility,
                multi_goal_support=multi_goal,
                do_star_support=do_star,
                sfm_identifiability_support=sfm_identifiability,
                goal_discovery_support=goal_discovery,
                policy_learning_support=policy_learning,
                temporal_goal_drift_support=temporal_goal_drift,
                context_conditioning_support=context_conditioning,
                hierarchical_goal_support=hierarchical_goal,
                constraint_support=constraint_support,
                normative_support=normative,
                action_recommendation_support=action_recommendation,
                robustness_support=robustness,
                alignment_summary=alignment_summary,
                audit_report=audit_report,
                execution_profile_support=execution_plan.to_dict(),
                falsification_passed=falsification_passed,
                side_effects_excluded=side_effects_excluded,
                support_level=support_level,
                authority_status=str(sfm_identifiability.get("authority_status") or "diagnostic_only"),
                reason=(
                    "Diagnostic final-cause score computed from real SCM-ID support, "
                    "agent-belief action-goal support, action-counterfactual support, "
                    "twin-policy support, explicit utility-function support, empirical implicit-utility support, "
                    "multi-goal utility support, formal do-star policy support, "
                    "goal-discovery diagnostics, sequence-level policy-learning support, "
                    "temporal goal-drift diagnostics, context-conditioned policy diagnostics, "
                    "hierarchical means-end goal diagnostics, "
                    "constraint-aware goal/constraint separation diagnostics, "
                    "normative/value-alignment classification, "
                    "SFM intervention recommendation under goals/constraints/norms/uncertainty, "
                    "uncertainty-aware robustness stress testing, "
                    "a governance-facing alignment summary, "
                    "SFM falsification checks, side-effect exclusion, "
                    "and SFM identifiability classification."
                ),
                reason_codes=reason_codes,
                limits=limits,
                raw=query.to_dict(),
            )
            if best is None or result.intent_score > best.intent_score:
                best = result
        return best or FinalCauseResult(raw=query.to_dict(), execution_profile_support=execution_plan.to_dict())


def infer_final_cause(payload: Any) -> Dict[str, Any]:
    return FinalCauseEngine().infer(payload).to_dict()


def infer_final_cause_compact(payload: Any) -> Dict[str, Any]:
    """Compact governance-facing SFM result.

    The full ``infer_final_cause`` output keeps every diagnostic layer.  This
    helper preserves the same inference power internally, but returns only the
    fields that a safety gate or product integration usually needs.
    """

    full = infer_final_cause(payload)
    recommendation = _as_dict(full.get("action_recommendation_support"))
    robustness = _as_dict(full.get("robustness_support"))
    audit_report = _as_dict(full.get("audit_report"))
    return {
        "inferred": bool(full.get("inferred")),
        "intent_hypothesis_supported": bool(full.get("intent_hypothesis_supported")),
        "intent_claim_authorized": bool(full.get("intent_claim_authorized")),
        "governance_execution_allowed": bool(full.get("governance_execution_allowed")),
        "most_likely_goal": full.get("most_likely_goal", ""),
        "observed_action": full.get("observed_action", ""),
        "intent_score": full.get("intent_score", 0.0),
        "support_level": full.get("support_level", "none"),
        "authority_status": full.get("authority_status", "diagnostic_only"),
        "execution_profile_support": _as_dict(full.get("execution_profile_support")),
        "alignment_summary": _as_dict(full.get("alignment_summary")),
        "recommended_action": recommendation.get("recommended_action", ""),
        "recommendation_status": recommendation.get("recommendation_status", "unassessed"),
        "robustness_status": robustness.get("robustness_status", "unassessed"),
        "robust_to_uncertainty": bool(robustness.get("robust_to_uncertainty", False)),
        "intentional_intervention": _as_dict(full.get("intentional_intervention")),
        "audit_report_summary": audit_report.get("executive_summary", ""),
        "audit_report_markdown": audit_report.get("markdown", ""),
        "reason_codes": list(full.get("alignment_summary", {}).get("reason_codes", [])),
        "limits": list(full.get("alignment_summary", {}).get("warnings", [])),
    }
