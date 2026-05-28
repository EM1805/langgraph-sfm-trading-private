from __future__ import annotations

"""Execution profiles for the SFM diagnostic stack.

The SFM stack deliberately exposes many independent diagnostic layers.  This
module keeps that power while avoiding a monolithic always-on execution path:
callers can choose a named profile or pass explicit enabled/disabled layer lists.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Set


ALL_SFM_LAYERS: Set[str] = {
    "goal_discovery",
    "identification",
    "counterfactual",
    "twin_model",
    "belief_model",
    "falsification",
    "utility",
    "empirical_utility",
    "multi_goal",
    "do_star",
    "policy_learning",
    "temporal_drift",
    "context_conditioning",
    "hierarchical_goal",
    "constraint",
    "normative",
    "recommendation",
    "robustness",
    "identifiability",
    "alignment_summary",
    "audit_report",
}

# Useful aliases for external callers and docs.
LAYER_ALIASES: Dict[str, str] = {
    "scm_id": "identification",
    "id": "identification",
    "cf": "counterfactual",
    "counterfactuals": "counterfactual",
    "twin": "twin_model",
    "belief": "belief_model",
    "beliefs": "belief_model",
    "empirical": "empirical_utility",
    "do*": "do_star",
    "do_star_operator": "do_star",
    "policy": "policy_learning",
    "temporal": "temporal_drift",
    "context": "context_conditioning",
    "hierarchy": "hierarchical_goal",
    "hierarchical": "hierarchical_goal",
    "constraints": "constraint",
    "norms": "normative",
    "alignment": "alignment_summary",
    "summary": "alignment_summary",
    "report": "audit_report",
    "audit": "audit_report",
    "human_report": "audit_report",
    "action_recommendation": "recommendation",
    "robust": "robustness",
    "uncertainty": "robustness",
    "uncertainty_aware": "robustness",
    "sensitivity": "robustness",
}

EXECUTION_PROFILES: Dict[str, Set[str]] = {
    # Preserve the pre-step19 behavior.
    "full": set(ALL_SFM_LAYERS),
    # Low-latency intent screening: structural, counterfactual, twin, belief,
    # falsification, identifiability, summary.
    "fast": {
        "goal_discovery",
        "identification",
        "counterfactual",
        "twin_model",
        "belief_model",
        "falsification",
        "robustness",
        "identifiability",
        "alignment_summary",
        "audit_report",
    },
    # Smallest useful core for development or smoke tests.
    "minimal": {
        "identification",
        "counterfactual",
        "twin_model",
        "robustness",
        "identifiability",
        "alignment_summary",
        "audit_report",
    },
    # Safety/governance integration: prioritize falsification, constraints,
    # norms, recommendation and a compact summary.
    "governance": {
        "goal_discovery",
        "identification",
        "counterfactual",
        "twin_model",
        "belief_model",
        "falsification",
        "utility",
        "multi_goal",
        "do_star",
        "hierarchical_goal",
        "constraint",
        "normative",
        "recommendation",
        "robustness",
        "identifiability",
        "alignment_summary",
        "audit_report",
    },
    # Historical/observational telos discovery.
    "discovery": {
        "goal_discovery",
        "identification",
        "counterfactual",
        "belief_model",
        "empirical_utility",
        "policy_learning",
        "temporal_drift",
        "context_conditioning",
        "robustness",
        "identifiability",
        "alignment_summary",
        "audit_report",
    },
    # Forward policy/recommendation under goals, constraints and norms.
    "recommendation": {
        "goal_discovery",
        "identification",
        "counterfactual",
        "twin_model",
        "utility",
        "multi_goal",
        "do_star",
        "hierarchical_goal",
        "constraint",
        "normative",
        "recommendation",
        "robustness",
        "identifiability",
        "alignment_summary",
        "audit_report",
    },
}


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    return [value]


def normalize_layer_name(name: Any) -> str:
    text = str(name or "").strip().lower().replace("-", "_").replace(" ", "_")
    return LAYER_ALIASES.get(text, text)


def normalize_layer_set(values: Iterable[Any]) -> Set[str]:
    return {layer for layer in (normalize_layer_name(v) for v in values) if layer}


@dataclass
class SFMExecutionPlan:
    """Resolved layer-selection contract for one SFM inference call."""

    profile: str = "full"
    enabled_layers: List[str] = field(default_factory=list)
    disabled_layers: List[str] = field(default_factory=list)
    unknown_layers: List[str] = field(default_factory=list)
    reason_codes: List[str] = field(default_factory=list)

    @classmethod
    def resolve(
        cls,
        *,
        profile: Any = "full",
        enabled_layers: Any = None,
        disabled_layers: Any = None,
    ) -> "SFMExecutionPlan":
        profile_name = str(profile or "full").strip().lower().replace("-", "_").replace(" ", "_") or "full"
        if profile_name not in EXECUTION_PROFILES:
            base = set(EXECUTION_PROFILES["full"])
            unknown = [profile_name]
            reason_codes = ["SFM_EXECUTION_PROFILE_UNKNOWN_USING_FULL"]
            profile_name = "full"
        else:
            base = set(EXECUTION_PROFILES[profile_name])
            unknown = []
            reason_codes = [f"SFM_EXECUTION_PROFILE_{profile_name.upper()}"]

        explicit_enabled = normalize_layer_set(_as_list(enabled_layers))
        explicit_disabled = normalize_layer_set(_as_list(disabled_layers))
        unknown.extend(sorted((explicit_enabled | explicit_disabled) - ALL_SFM_LAYERS))

        # Explicit enabled layers narrow the profile; explicit disabled layers
        # subtract from the final plan.  Alignment summary is kept on by default
        # unless disabled explicitly, because it is the external gate contract.
        if explicit_enabled:
            enabled = explicit_enabled & ALL_SFM_LAYERS
            if "alignment_summary" not in explicit_disabled:
                enabled.add("alignment_summary")
            reason_codes.append("SFM_EXECUTION_ENABLED_LAYERS_OVERRIDE")
        else:
            enabled = base
        if explicit_disabled:
            enabled -= (explicit_disabled & ALL_SFM_LAYERS)
            reason_codes.append("SFM_EXECUTION_DISABLED_LAYERS_OVERRIDE")

        return cls(
            profile=profile_name,
            enabled_layers=sorted(enabled),
            disabled_layers=sorted(ALL_SFM_LAYERS - enabled),
            unknown_layers=sorted(set(unknown)),
            reason_codes=reason_codes,
        )

    @classmethod
    def from_query(cls, query: Any) -> "SFMExecutionPlan":
        return cls.resolve(
            profile=getattr(query, "execution_profile", "full"),
            enabled_layers=getattr(query, "enabled_layers", []),
            disabled_layers=getattr(query, "disabled_layers", []),
        )

    def is_enabled(self, layer: str) -> bool:
        return normalize_layer_name(layer) in set(self.enabled_layers)

    def disabled_report(self, layer: str) -> Dict[str, Any]:
        normalized = normalize_layer_name(layer)
        return {
            "assessed": False,
            "evaluated": False,
            "compared": False,
            "disabled": True,
            "layer": normalized,
            "reason_codes": [f"SFM_LAYER_DISABLED_{normalized.upper()}"],
            "limits": [f"{normalized}_layer_disabled_by_execution_profile"],
        }

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def resolve_sfm_execution_plan(payload: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    """Public helper for callers that want to inspect a profile before running."""

    data = dict(payload or {})
    return SFMExecutionPlan.resolve(
        profile=data.get("execution_profile") or data.get("profile") or "full",
        enabled_layers=data.get("enabled_layers"),
        disabled_layers=data.get("disabled_layers"),
    ).to_dict()
