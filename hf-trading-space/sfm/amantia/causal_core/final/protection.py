from __future__ import annotations

"""Canonical protection/constraint/normative semantics for SFM.

Before Step 24, similar ideas appeared in multiple places:

* ``FinalCauseQuery.protected_outcome`` and ``GoalSpec.protected_outcomes``;
* constraint-aware hard/protected/soft/side-effect constraints;
* normative rules such as protected/prohibited goals and actions.

This module does not replace those layers.  It provides a single normalized view
that they and external gates can inspect to avoid semantic drift.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .schema import FinalCauseQuery
from .normative import normalize_normative_policy


@dataclass
class SFMProtectionSpec:
    """One canonical protection-related declaration."""

    target: str = ""
    target_type: str = "outcome"
    protection_type: str = "protected_outcome"
    status: str = "protected"
    severity: float = 1.0
    source: str = "derived"
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SFMProtectionPolicy:
    """Canonical protection view spanning constraints and normative policy."""

    assessed: bool = False
    specs: List[SFMProtectionSpec] = field(default_factory=list)
    protected_outcomes: List[str] = field(default_factory=list)
    hard_constraints: List[str] = field(default_factory=list)
    soft_constraints: List[str] = field(default_factory=list)
    side_effect_outcomes: List[str] = field(default_factory=list)
    protected_goals: List[str] = field(default_factory=list)
    prohibited_goals: List[str] = field(default_factory=list)
    prohibited_actions: List[str] = field(default_factory=list)
    escalation_goals: List[str] = field(default_factory=list)
    escalation_actions: List[str] = field(default_factory=list)
    reason_codes: List[str] = field(default_factory=list)
    limits: List[str] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)

    def targets(self, *, target_type: Optional[str] = None, protection_type: Optional[str] = None, status: Optional[str] = None) -> List[str]:
        out: List[str] = []
        for spec in self.specs:
            if target_type and spec.target_type != target_type:
                continue
            if protection_type and spec.protection_type != protection_type:
                continue
            if status and spec.status != status:
                continue
            if spec.target and spec.target not in out:
                out.append(spec.target)
        return out

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["specs"] = [spec.to_dict() for spec in self.specs]
        return data


def _clean_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


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


def _unique(items: Iterable[Any]) -> List[str]:
    out: List[str] = []
    for item in items:
        text = _clean_str(item)
        if text and text not in out:
            out.append(text)
    return out


def _safe_float(value: Any, default: float = 1.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _policy_from_query(query: FinalCauseQuery) -> Dict[str, Any]:
    raw = _as_dict(query.raw)
    return _as_dict(raw.get("normative_policy") or raw.get("value_policy") or raw.get("alignment_policy") or raw.get("policy"))


def _spec(target: Any, *, target_type: str, protection_type: str, status: str, source: str, severity: float = 1.0, reason: str = "", metadata: Optional[Mapping[str, Any]] = None) -> Optional[SFMProtectionSpec]:
    text = _clean_str(target)
    if not text:
        return None
    return SFMProtectionSpec(
        target=text,
        target_type=target_type,
        protection_type=protection_type,
        status=status,
        severity=float(severity),
        source=source,
        reason=reason,
        metadata=_as_dict(metadata),
    )


def _add(out: List[SFMProtectionSpec], maybe: Optional[SFMProtectionSpec]) -> None:
    if maybe is not None:
        out.append(maybe)


def _specs_from_constraint_support(constraint_support: Optional[Mapping[str, Any]]) -> List[SFMProtectionSpec]:
    support = _as_dict(constraint_support)
    specs: List[SFMProtectionSpec] = []
    for target in _as_list(support.get("hard_constraints")):
        _add(specs, _spec(target, target_type="outcome", protection_type="hard_constraint", status="protected", source="constraint_support.hard_constraints"))
    for target in _as_list(support.get("protected_constraints")):
        _add(specs, _spec(target, target_type="outcome", protection_type="protected_outcome", status="protected", source="constraint_support.protected_constraints"))
    for target in _as_list(support.get("soft_constraints")):
        _add(specs, _spec(target, target_type="outcome", protection_type="soft_constraint", status="monitored", source="constraint_support.soft_constraints", severity=0.5))
    for target in _as_list(support.get("side_effect_outcomes")):
        _add(specs, _spec(target, target_type="outcome", protection_type="side_effect", status="monitored", source="constraint_support.side_effect_outcomes", severity=0.25))
    return specs


def _specs_from_raw_constraints(query: FinalCauseQuery) -> List[SFMProtectionSpec]:
    raw = _as_dict(query.raw)
    model = raw.get("constraint_model") or raw.get("constraints") or raw.get("constraint_function") or {}
    model = _as_dict(model)
    specs: List[SFMProtectionSpec] = []
    groups = [
        ("hard_constraints", "hard_constraint", "protected", 1.0),
        ("hard", "hard_constraint", "protected", 1.0),
        ("protected_constraints", "protected_outcome", "protected", 1.0),
        ("protected", "protected_outcome", "protected", 1.0),
        ("soft_constraints", "soft_constraint", "monitored", 0.5),
        ("soft", "soft_constraint", "monitored", 0.5),
        ("side_effect_constraints", "side_effect", "monitored", 0.25),
        ("side_effects", "side_effect", "monitored", 0.25),
        ("side_effect", "side_effect", "monitored", 0.25),
    ]
    for key, ptype, status, severity in groups:
        for source_prefix, container in [(key, raw.get(key)), (f"constraint_model.{key}", model.get(key))]:
            if isinstance(container, Mapping):
                for target, payload in container.items():
                    sev = _safe_float(_as_dict(payload).get("severity", severity), severity) if isinstance(payload, Mapping) else severity
                    _add(specs, _spec(target, target_type="outcome", protection_type=ptype, status=status, source=source_prefix, severity=sev))
            else:
                for item in _as_list(container):
                    if isinstance(item, Mapping):
                        target = item.get("outcome") or item.get("target") or item.get("name")
                        sev = _safe_float(item.get("severity", severity), severity)
                    else:
                        target = item
                        sev = severity
                    _add(specs, _spec(target, target_type="outcome", protection_type=ptype, status=status, source=source_prefix, severity=sev))
    return specs


def _specs_from_normative_policy(query: FinalCauseQuery) -> List[SFMProtectionSpec]:
    policy = normalize_normative_policy(_policy_from_query(query))
    specs: List[SFMProtectionSpec] = []
    if not policy.assessed:
        return specs
    for rule in policy.rules:
        if rule.status == "protected":
            ptype = "normative_protected"
        elif rule.status == "prohibited":
            ptype = "normative_prohibited"
        elif rule.status == "escalation_required":
            ptype = "normative_escalation"
        elif rule.status in {"discouraged", "monitored"}:
            ptype = f"normative_{rule.status}"
        else:
            continue
        _add(
            specs,
            _spec(
                rule.target,
                target_type=rule.target_type,
                protection_type=ptype,
                status=rule.status,
                source=f"normative.{rule.source}",
                severity=rule.severity,
                reason=rule.reason,
                metadata=rule.metadata,
            ),
        )
    return specs


def _dedupe(specs: Iterable[SFMProtectionSpec]) -> List[SFMProtectionSpec]:
    out: List[SFMProtectionSpec] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    for spec in specs:
        key = (spec.target, spec.target_type, spec.protection_type, spec.status, spec.source)
        if spec.target and key not in seen:
            seen.add(key)
            out.append(spec)
    return out


def normalize_protection_policy(
    payload: Any,
    *,
    constraint_support: Optional[Mapping[str, Any]] = None,
) -> SFMProtectionPolicy:
    """Build one canonical protection policy from an SFM query and layer outputs."""

    query = FinalCauseQuery.from_payload(payload)
    specs: List[SFMProtectionSpec] = []
    _add(specs, _spec(query.protected_outcome, target_type="outcome", protection_type="protected_outcome", status="protected", source="query.protected_outcome"))
    for goal in query.candidate_goals:
        for outcome in goal.protected_outcomes:
            _add(specs, _spec(outcome, target_type="outcome", protection_type="protected_outcome", status="protected", source=f"candidate_goal.{goal.goal_variable}.protected_outcomes"))
        for outcome in goal.side_effect_outcomes:
            _add(specs, _spec(outcome, target_type="outcome", protection_type="side_effect", status="monitored", source=f"candidate_goal.{goal.goal_variable}.side_effect_outcomes", severity=0.25))
    for goal in query.side_effect_goals:
        _add(specs, _spec(goal.goal_variable, target_type="outcome", protection_type="side_effect", status="monitored", source="query.side_effect_goals", severity=0.25))

    specs.extend(_specs_from_raw_constraints(query))
    specs.extend(_specs_from_constraint_support(constraint_support))
    specs.extend(_specs_from_normative_policy(query))
    specs = _dedupe(specs)

    policy = SFMProtectionPolicy(
        assessed=bool(specs),
        specs=specs,
        protected_outcomes=_unique(
            spec.target
            for spec in specs
            if spec.target_type in {"outcome", "goal"}
            and spec.protection_type in {"protected_outcome", "hard_constraint", "normative_protected"}
        ),
        hard_constraints=_unique(spec.target for spec in specs if spec.protection_type == "hard_constraint"),
        soft_constraints=_unique(spec.target for spec in specs if spec.protection_type == "soft_constraint"),
        side_effect_outcomes=_unique(spec.target for spec in specs if spec.protection_type == "side_effect"),
        protected_goals=_unique(spec.target for spec in specs if spec.target_type == "goal" and spec.status == "protected"),
        prohibited_goals=_unique(spec.target for spec in specs if spec.target_type == "goal" and spec.status == "prohibited"),
        prohibited_actions=_unique(spec.target for spec in specs if spec.target_type == "action" and spec.status == "prohibited"),
        escalation_goals=_unique(spec.target for spec in specs if spec.target_type == "goal" and spec.status == "escalation_required"),
        escalation_actions=_unique(spec.target for spec in specs if spec.target_type == "action" and spec.status == "escalation_required"),
        reason_codes=["SFM_PROTECTION_POLICY_NORMALIZED"] if specs else ["SFM_PROTECTION_POLICY_EMPTY"],
        limits=[] if specs else ["no_protection_constraints_or_normative_rules_supplied"],
        raw={"query": query.to_dict(), "constraint_support": dict(constraint_support or {})},
    )
    return policy
