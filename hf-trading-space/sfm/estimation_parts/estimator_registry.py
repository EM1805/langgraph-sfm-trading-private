"""Estimator registry for Amantia Estimation.

This module is a declarative catalog of the estimators Estimation is allowed to
recommend or run.  It does not create causal authority: authority comes from the
SCM/ID contract layer.  The registry only answers questions such as:

- Which estimator name is canonical for this contract row?
- Is the estimator decision-safe or diagnostic-only?
- Does it require formal identification, an adjustment set, or only review?

Keeping this metadata in one place avoids scattering strings such as
``backdoor_ridge_adjustment`` and ``diagnostic_lagged_regression`` throughout the
Estimation codebase.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    from runtime_env import configure_scientific_runtime
    configure_scientific_runtime()
except Exception:
    pass

try:
    import pandas as pd
except Exception:  # pragma: no cover - pandas may be unavailable in tiny runtimes
    pd = None  # type: ignore

from . import _utils as U


@dataclass(frozen=True)
class EstimatorSpec:
    """Declarative metadata for an estimation method.

    ``decision_safe`` means the estimator may support downstream decision/veto
    artifacts, but only when the upstream contract row is identified and
    enabled. ``diagnostic_only`` means it must never be promoted to causal
    authority.
    """

    name: str
    family: str
    label: str
    status: str
    claim_level: str
    estimator_authority: str
    requires_identification: bool
    requires_contract_authority: bool
    requires_adjustment_set: bool
    allows_empty_adjustment_set: bool
    decision_safe: bool
    diagnostic_only: bool
    implementation_module: str = ""
    implementation_function: str = ""
    notes: str = ""

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


_ESTIMABLE_LEVELS = {"identified_estimable"}
_REVIEW_LEVELS = {"identified_needs_estimation", "graph_review"}
_DIAGNOSTIC_MCI = {"diagnostic_support", "pass"}
_VALID_ADJUSTMENT_STATUSES = {"valid_empty", "valid_nonempty"}


REGISTRY: Dict[str, EstimatorSpec] = {
    "backdoor_ridge_adjustment": EstimatorSpec(
        name="backdoor_ridge_adjustment",
        family="backdoor",
        label="Backdoor ridge adjustment",
        status="enabled",
        claim_level="identified_estimable",
        estimator_authority="formal_identification_required",
        requires_identification=True,
        requires_contract_authority=True,
        requires_adjustment_set=False,
        allows_empty_adjustment_set=True,
        decision_safe=True,
        diagnostic_only=False,
        implementation_module="estimation_parts.pearl_backdoor",
        implementation_function="estimate_backdoor_effect",
        notes="Authorized only for SCM/ID identified rows with backdoor/adjustment evidence.",
    ),
    "lagged_backdoor_ols_bootstrap": EstimatorSpec(
        name="lagged_backdoor_ols_bootstrap",
        family="backdoor",
        label="Lagged backdoor OLS bootstrap",
        status="enabled",
        claim_level="identified_estimable",
        estimator_authority="formal_identification_required",
        requires_identification=True,
        requires_contract_authority=True,
        requires_adjustment_set=False,
        allows_empty_adjustment_set=True,
        decision_safe=True,
        diagnostic_only=False,
        implementation_module="estimation_parts.effect_estimates",
        implementation_function="estimate_plan_row",
        notes="Main compact effect-estimates path; numerical primitive lives in stat_core.",
    ),
    "matched_aipw": EstimatorSpec(
        name="matched_aipw",
        family="matching_doubly_robust",
        label="Matched AIPW-style counterfactual effect",
        status="experimental",
        claim_level="counterfactual_trial_diagnostic",
        estimator_authority="requires_trial_or_counterfactual_design",
        requires_identification=True,
        requires_contract_authority=True,
        requires_adjustment_set=False,
        allows_empty_adjustment_set=True,
        decision_safe=False,
        diagnostic_only=True,
        implementation_module="estimation_parts.effects",
        implementation_function="estimate_effect_bundle",
        notes="Useful for Level 3.2 trial evaluation; not a standalone SCM/ID effect claim.",
    ),
    "diagnostic_lagged_regression": EstimatorSpec(
        name="diagnostic_lagged_regression",
        family="diagnostic_regression",
        label="Diagnostic lagged regression",
        status="enabled",
        claim_level="diagnostic_only",
        estimator_authority="identified_but_estimator_choice_uncertain",
        requires_identification=True,
        requires_contract_authority=True,
        requires_adjustment_set=False,
        allows_empty_adjustment_set=True,
        decision_safe=False,
        diagnostic_only=True,
        implementation_module="estimation_parts.effect_estimates",
        implementation_function="estimate_plan_row",
        notes="Fallback for identified rows where the formal estimator choice is not clear.",
    ),
    "diagnostic_lagged_regression_only": EstimatorSpec(
        name="diagnostic_lagged_regression_only",
        family="diagnostic_regression",
        label="Diagnostic-only lagged regression",
        status="enabled",
        claim_level="diagnostic_only",
        estimator_authority="diagnostic_only",
        requires_identification=False,
        requires_contract_authority=False,
        requires_adjustment_set=False,
        allows_empty_adjustment_set=True,
        decision_safe=False,
        diagnostic_only=True,
        implementation_module="estimation_parts.effect_estimates",
        implementation_function="estimate_plan_row",
        notes="May inspect temporal parents but must not promote Discovery signals to causal claims.",
    ),
    "frontdoor_limited_estimator": EstimatorSpec(
        name="frontdoor_limited_estimator",
        family="frontdoor",
        label="Limited frontdoor estimator",
        status="planned",
        claim_level="identified_needs_estimation",
        estimator_authority="formal_identification_required",
        requires_identification=True,
        requires_contract_authority=True,
        requires_adjustment_set=False,
        allows_empty_adjustment_set=True,
        decision_safe=False,
        diagnostic_only=True,
        implementation_module="",
        implementation_function="",
        notes="Placeholder: frontdoor logic is recognized by the planner but not enabled as a decision-safe estimator.",
    ),
    "frontdoor_estimator_not_enabled_yet": EstimatorSpec(
        name="frontdoor_estimator_not_enabled_yet",
        family="frontdoor",
        label="Frontdoor estimator not enabled",
        status="not_enabled",
        claim_level="plan_only",
        estimator_authority="plan_only",
        requires_identification=True,
        requires_contract_authority=True,
        requires_adjustment_set=False,
        allows_empty_adjustment_set=True,
        decision_safe=False,
        diagnostic_only=True,
        notes="Plan-only marker for frontdoor rows until estimator implementation is added.",
    ),
    "backdoor_or_matching_estimator_review": EstimatorSpec(
        name="backdoor_or_matching_estimator_review",
        family="review",
        label="Backdoor or matching estimator review",
        status="review_only",
        claim_level="plan_only",
        estimator_authority="plan_only",
        requires_identification=True,
        requires_contract_authority=True,
        requires_adjustment_set=False,
        allows_empty_adjustment_set=True,
        decision_safe=False,
        diagnostic_only=True,
        notes="The graph says this may be identifiable, but Estimation should not claim an effect yet.",
    ),
    "no_effect_estimate_until_graph_review": EstimatorSpec(
        name="no_effect_estimate_until_graph_review",
        family="review",
        label="No effect estimate until graph review",
        status="review_only",
        claim_level="review_only",
        estimator_authority="review_only",
        requires_identification=True,
        requires_contract_authority=True,
        requires_adjustment_set=False,
        allows_empty_adjustment_set=True,
        decision_safe=False,
        diagnostic_only=True,
        notes="Graph/SCM review must happen before any estimator is selected.",
    ),
    "skip": EstimatorSpec(
        name="skip",
        family="none",
        label="Skip estimation",
        status="blocked",
        claim_level="not_authorized",
        estimator_authority="not_authorized",
        requires_identification=True,
        requires_contract_authority=True,
        requires_adjustment_set=False,
        allows_empty_adjustment_set=True,
        decision_safe=False,
        diagnostic_only=True,
        notes="No estimator is allowed for this row.",
    ),
    "none": EstimatorSpec(
        name="none",
        family="none",
        label="No estimator",
        status="blocked",
        claim_level="not_estimated",
        estimator_authority="not_authorized",
        requires_identification=False,
        requires_contract_authority=False,
        requires_adjustment_set=False,
        allows_empty_adjustment_set=True,
        decision_safe=False,
        diagnostic_only=True,
        notes="Compatibility marker used in empty/unestimated output rows.",
    ),
}


def _as_str(value) -> str:
    return U.as_str(value).strip()


def _boolish(value) -> bool:
    return _as_str(value).lower() in {"1", "true", "yes", "y", "on"}


def _has_text(value) -> bool:
    text = _as_str(value)
    return bool(text and text.lower() not in {"nan", "none", "null", "[]"})


def list_estimators(*, include_blocked: bool = True) -> List[EstimatorSpec]:
    """Return registered estimators in stable order."""

    specs = list(REGISTRY.values())
    if not include_blocked:
        specs = [s for s in specs if s.status not in {"blocked", "not_enabled"}]
    return specs


def get_estimator(name: str, default: Optional[EstimatorSpec] = None) -> Optional[EstimatorSpec]:
    """Fetch one estimator spec by canonical name."""

    return REGISTRY.get(_as_str(name), default)


def estimator_names(*, decision_safe_only: bool = False, diagnostic_only: Optional[bool] = None) -> List[str]:
    """Return canonical estimator names filtered by high-level safety flags."""

    out: List[str] = []
    for spec in list_estimators():
        if decision_safe_only and not spec.decision_safe:
            continue
        if diagnostic_only is not None and bool(spec.diagnostic_only) != bool(diagnostic_only):
            continue
        out.append(spec.name)
    return out


def estimator_catalog_frame():
    """Return the registry as a DataFrame when pandas is available."""

    rows = [s.to_dict() for s in list_estimators()]
    if pd is None:  # pragma: no cover
        return rows
    return pd.DataFrame(rows)


def is_estimator_decision_safe(name: str) -> bool:
    spec = get_estimator(name)
    return bool(spec and spec.decision_safe and not spec.diagnostic_only)


def is_estimator_diagnostic_only(name: str) -> bool:
    spec = get_estimator(name)
    return True if spec is None else bool(spec.diagnostic_only)


def _contract_flags(row: Mapping[str, object]) -> Tuple[str, bool, bool, bool]:
    authority = _as_str(row.get("authority_level", "")).lower()
    enabled = _boolish(row.get("estimation_enabled", ""))
    allowed_text = _as_str(row.get("allowed_for_estimation", ""))
    allowed = _boolish(allowed_text) if allowed_text else (authority == "identified_estimable" and enabled)
    identified_text = _as_str(row.get("identified", ""))
    identified = _boolish(identified_text) if identified_text else authority == "identified_estimable"
    return authority, enabled, allowed, identified


def select_estimator_for_row(row: Mapping[str, object]) -> Tuple[str, str]:
    """Select the planner's recommended estimator and authority label.

    This preserves the previous handoff_reader behavior while centralizing the
    estimator string catalog.  It is intentionally conservative: non-identified
    Discovery/MCI support can only receive diagnostic-only estimator labels.
    """

    authority, enabled, allowed, identified = _contract_flags(row)
    strategy = _as_str(row.get("identification_strategy", "")).lower()
    estimand = _as_str(row.get("estimand_type", "")).lower()
    adj_status = _as_str(row.get("adjustment_set_status", "")).lower()
    adj = _as_str(row.get("adjustment_set", row.get("total_adjustment_set", "")))
    backdoor_status = _as_str(row.get("backdoor_status", "")).lower()
    role = _as_str(row.get("scm_role_hint", "")).lower()
    mci_status = _as_str(row.get("mci_status", "")).lower()

    if authority in _ESTIMABLE_LEVELS and enabled and allowed and identified:
        if "frontdoor" in strategy or "frontdoor" in estimand:
            spec = REGISTRY["frontdoor_limited_estimator"]
            return spec.name, spec.estimator_authority
        if (
            "backdoor" in strategy
            or backdoor_status in {"backdoor_adjustment_candidate", "adjustment_candidate"}
            or adj_status in _VALID_ADJUSTMENT_STATUSES
            or _has_text(adj)
        ):
            spec = REGISTRY["backdoor_ridge_adjustment"]
            return spec.name, spec.estimator_authority
        spec = REGISTRY["diagnostic_lagged_regression"]
        return spec.name, spec.estimator_authority

    if authority == "identified_needs_estimation":
        if "frontdoor" in strategy or "frontdoor" in estimand:
            spec = REGISTRY["frontdoor_estimator_not_enabled_yet"]
            return spec.name, spec.estimator_authority
        spec = REGISTRY["backdoor_or_matching_estimator_review"]
        return spec.name, spec.estimator_authority

    if authority == "graph_review":
        spec = REGISTRY["no_effect_estimate_until_graph_review"]
        return spec.name, spec.estimator_authority

    if "temporal_parent" in role or mci_status in _DIAGNOSTIC_MCI:
        spec = REGISTRY["diagnostic_lagged_regression_only"]
        return spec.name, spec.estimator_authority

    spec = REGISTRY["skip"]
    return spec.name, spec.estimator_authority



def resolve_effect_estimator_for_row(row: Mapping[str, object]) -> Tuple[str, str]:
    """Return the estimator implementation used by effect_estimates.py.

    The planner may recommend high-level estimators such as
    ``backdoor_ridge_adjustment``. The compact effect-estimates artifact uses a
    conservative lagged OLS/bootstrap implementation for those rows unless a
    dedicated estimator is wired into this module. Planned, review-only, blocked,
    and unknown estimators are not run.
    """
    requested = _as_str(row.get("recommended_estimator", row.get("estimator_used", "")))
    if not requested:
        requested, _ = select_estimator_for_row(row)
    ok, reason = validate_estimator_for_row(requested, row)
    if not ok:
        return "none", reason
    spec = get_estimator(requested)
    if spec is None:
        return "none", "UNKNOWN_ESTIMATOR"
    if spec.status != "enabled":
        return "none", f"ESTIMATOR_STATUS_NOT_RUNNABLE:{spec.status}"
    if requested == "backdoor_ridge_adjustment":
        return "lagged_backdoor_ols_bootstrap", "OK_BACKDOOR_COMPACT_EFFECT_ESTIMATOR"
    if requested in {"lagged_backdoor_ols_bootstrap", "diagnostic_lagged_regression"}:
        return requested, "OK"
    if spec.implementation_module == "estimation_parts.effect_estimates":
        return requested, "OK"
    return "none", f"ESTIMATOR_NOT_RUNNABLE_BY_EFFECT_ESTIMATES:{requested}"


def validate_estimator_for_row(name: str, row: Mapping[str, object]) -> Tuple[bool, str]:
    """Check whether an estimator name is compatible with one contract row."""

    spec = get_estimator(name)
    if spec is None:
        return False, "UNKNOWN_ESTIMATOR"
    authority, enabled, allowed, identified = _contract_flags(row)
    if spec.status in {"blocked", "not_enabled"}:
        return False, "ESTIMATOR_NOT_ENABLED"
    if spec.requires_identification and not identified:
        return False, "ESTIMATOR_REQUIRES_IDENTIFICATION"
    if spec.requires_contract_authority and not (authority in _ESTIMABLE_LEVELS and enabled and allowed):
        if spec.diagnostic_only and spec.estimator_authority in {"diagnostic_only", "review_only", "plan_only"}:
            return True, "DIAGNOSTIC_OR_REVIEW_ONLY"
        return False, "ESTIMATOR_REQUIRES_CONTRACT_AUTHORITY"
    if spec.requires_adjustment_set and not _has_text(row.get("adjustment_set", row.get("total_adjustment_set", ""))):
        return False, "ESTIMATOR_REQUIRES_ADJUSTMENT_SET"
    return True, "OK"


__all__ = [
    "EstimatorSpec",
    "REGISTRY",
    "estimator_catalog_frame",
    "estimator_names",
    "get_estimator",
    "is_estimator_decision_safe",
    "is_estimator_diagnostic_only",
    "list_estimators",
    "select_estimator_for_row",
    "resolve_effect_estimator_for_row",
    "validate_estimator_for_row",
]
