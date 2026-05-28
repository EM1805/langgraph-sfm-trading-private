"""Structural Final Model development layer for Amantia.

This is a conservative scaffold: it exposes contracts and a diagnostic facade
that composes existing SCM-ID and counterfactual adapters.  It is intended as a
starting point for implementing intentional interventions and twin models.
"""

from .alignment_summary import SFMAlignmentSummary, SFMAlignmentSummarizer, summarize_sfm_alignment
from .belief_model import AgentBeliefEvaluator, BeliefCausalAssessment, assess_agent_beliefs
from .constraint import (
    ConstraintActionAssessment,
    ConstraintAwareAudit,
    ConstraintAwareEvaluator,
    ConstraintEvaluation,
    SFMConstraintSpec,
    evaluate_constraint_aware_sfm,
)
from .context_conditioning import (
    ContextConditioningAudit,
    ContextConditioningEvaluator,
    ContextGoalProfile,
    evaluate_context_conditioning,
)
from .do_star import DoStarOperator, DoStarOperatorResult, DoStarPolicyInputAudit, evaluate_do_star_intervention
from .execution import (
    ALL_SFM_LAYERS,
    EXECUTION_PROFILES,
    SFMExecutionPlan,
    normalize_layer_name,
    resolve_sfm_execution_plan,
)
from .empirical_utility import (
    EmpiricalActionEvidence,
    EmpiricalUtilityAudit,
    EmpiricalUtilityLearner,
    evaluate_empirical_utility,
)
from .goal_discovery import (
    DiscoveredGoalCandidate,
    GoalDiscoveryEngine,
    GoalDiscoveryReport,
    discover_candidate_goals,
)
from .identifiability import SFMIdentifiabilityAssessment, SFMIdentifiabilityEvaluator, assess_sfm_identifiability
from .hierarchical import (
    GoalHierarchyEdge,
    HierarchicalGoalAudit,
    HierarchicalGoalEvaluator,
    HierarchicalGoalProfile,
    evaluate_hierarchical_goals,
)
from .falsification import (
    FalsificationGoalAudit,
    FalsificationReport,
    SFMFalsificationAuditor,
    audit_sfm_falsification,
)
from .intentional_intervention import build_intentional_intervention, do_star_expression
from .normative import (
    NormalizedNormativePolicy,
    NormativeRule,
    NormativeSFMAudit,
    NormativeSFMEvaluator,
    evaluate_normative_sfm,
    normalize_normative_policy,
    normative_status_for_target,
)
from .layer_protocol import SFMLayerEvaluator, SFMLayerResult, layer_result_to_dict
from .protection import SFMProtectionPolicy, SFMProtectionSpec, normalize_protection_policy
from .runner import SFMExecutionRunner
from .recommendation import (
    RecommendedInterventionAction,
    SFMActionRecommendationAudit,
    SFMActionRecommender,
    recommend_sfm_action,
)
from .reporting import (
    SFMAuditReport,
    SFMAuditReportGenerator,
    SFMReportSection,
    render_sfm_audit_report,
)
from .robustness import (
    RobustnessScenario,
    RobustSFMAudit,
    RobustSFMEvaluator,
    evaluate_sfm_robustness,
)

from .external_validation import (
    ExternalSFMPanelBenchmarkReport,
    ExternalSFMPanelCase,
    ExternalSFMPanelCaseResult,
    build_external_sfm_panel_cases,
    run_sfm_external_panel_benchmark,
)
from .validation_benchmark import (
    SFMValidationBenchmarkReport,
    SyntheticSFMCase,
    SyntheticSFMCaseResult,
    default_synthetic_sfm_cases,
    run_sfm_validation_benchmark,
)
from .policy_learning import (
    PolicyGoalEvidence,
    PolicyLearningAudit,
    PolicyLearningEngine,
    evaluate_policy_learning,
)
from .temporal import (
    GoalDriftEvent,
    TemporalGoalDriftAudit,
    TemporalGoalDriftDetector,
    TemporalGoalWindow,
    evaluate_temporal_goal_drift,
)
from .multi_goal import (
    MultiGoalActionScore,
    MultiGoalContribution,
    MultiGoalUtilityAudit,
    MultiGoalUtilityEvaluator,
    evaluate_multi_goal_utility,
)
from .inference import FinalCauseEngine, infer_final_cause, infer_final_cause_compact
from .schema import AgentModel, FinalCauseQuery, FinalCauseResult, GoalSpec, IntentionalIntervention
from .twin_model import TwinPolicyComparator, TwinPolicyComparison, compare_twin_policies
from .utility import (
    ActionUtilityBreakdown,
    UtilityComponent,
    UtilityFunctionAudit,
    UtilityFunctionEvaluator,
    evaluate_utility_function,
)

__all__ = [
    "SFMAlignmentSummary",
    "ALL_SFM_LAYERS",
    "EXECUTION_PROFILES",
    "SFMExecutionPlan",
    "SFMExecutionRunner",
    "SFMProtectionPolicy",
    "SFMProtectionSpec",
    "SFMLayerEvaluator",
    "SFMLayerResult",
    "SFMAlignmentSummarizer",
    "summarize_sfm_alignment",
    "AgentBeliefEvaluator",
    "AgentModel",
    "BeliefCausalAssessment",
    "ContextConditioningAudit",
    "ConstraintActionAssessment",
    "ConstraintAwareAudit",
    "ConstraintAwareEvaluator",
    "ConstraintEvaluation",
    "SFMConstraintSpec",
    "evaluate_constraint_aware_sfm",
    "ContextConditioningEvaluator",
    "ContextGoalProfile",
    "DoStarOperator",
    "DoStarOperatorResult",
    "DoStarPolicyInputAudit",
    "EmpiricalActionEvidence",
    "EmpiricalUtilityAudit",
    "EmpiricalUtilityLearner",
    "FalsificationGoalAudit",
    "FalsificationReport",
    "FinalCauseEngine",
    "FinalCauseQuery",
    "DiscoveredGoalCandidate",
    "GoalDiscoveryEngine",
    "GoalDiscoveryReport",
    "FinalCauseResult",
    "GoalSpec",
    "IntentionalIntervention",
    "HierarchicalGoalProfile",
    "HierarchicalGoalEvaluator",
    "HierarchicalGoalAudit",
    "GoalHierarchyEdge",
    "MultiGoalActionScore",
    "MultiGoalContribution",
    "MultiGoalUtilityAudit",
    "MultiGoalUtilityEvaluator",
    "NormalizedNormativePolicy",
    "NormativeRule",
    "NormativeSFMAudit",
    "NormativeSFMEvaluator",
    "RecommendedInterventionAction",
    "SFMActionRecommendationAudit",
    "SFMActionRecommender",
    "SFMValidationBenchmarkReport",
    "ExternalSFMPanelBenchmarkReport",
    "ExternalSFMPanelCase",
    "ExternalSFMPanelCaseResult",
    "SyntheticSFMCase",
    "SyntheticSFMCaseResult",
    "SFMAuditReport",
    "SFMAuditReportGenerator",
    "SFMReportSection",
    "RobustnessScenario",
    "RobustSFMAudit",
    "RobustSFMEvaluator",
    "PolicyGoalEvidence",
    "PolicyLearningAudit",
    "PolicyLearningEngine",
    "GoalDriftEvent",
    "TemporalGoalDriftAudit",
    "TemporalGoalDriftDetector",
    "TemporalGoalWindow",
    "SFMFalsificationAuditor",
    "SFMIdentifiabilityAssessment",
    "SFMIdentifiabilityEvaluator",
    "TwinPolicyComparator",
    "TwinPolicyComparison",
    "ActionUtilityBreakdown",
    "UtilityComponent",
    "UtilityFunctionAudit",
    "UtilityFunctionEvaluator",
    "assess_agent_beliefs",
    "assess_sfm_identifiability",
    "audit_sfm_falsification",
    "build_intentional_intervention",
    "compare_twin_policies",
    "discover_candidate_goals",
    "do_star_expression",
    "evaluate_context_conditioning",
    "evaluate_do_star_intervention",
    "normalize_layer_name",
    "layer_result_to_dict",
    "normalize_protection_policy",
    "resolve_sfm_execution_plan",
    "evaluate_empirical_utility",
    "evaluate_hierarchical_goals",
    "evaluate_multi_goal_utility",
    "evaluate_normative_sfm",
    "normalize_normative_policy",
    "normative_status_for_target",
    "recommend_sfm_action",
    "default_synthetic_sfm_cases",
    "render_sfm_audit_report",
    "run_sfm_validation_benchmark",
    "build_external_sfm_panel_cases",
    "run_sfm_external_panel_benchmark",
    "evaluate_sfm_robustness",
    "evaluate_policy_learning",
    "evaluate_temporal_goal_drift",
    "evaluate_utility_function",
    "infer_final_cause",
    "infer_final_cause_compact",
]
