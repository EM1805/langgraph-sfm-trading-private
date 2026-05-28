from __future__ import annotations

"""Normative / value-alignment diagnostics for Structural Final Models.

This layer asks whether an inferred or candidate final cause is permitted by an
explicit normative/value policy.  It is deliberately separate from the telic
claim: a prohibited goal can still be the goal the agent appears to pursue.

Step 20 consolidates the policy contract.  Simple inline JSON policies such as
``allowed_goals`` and rich ``NormativeRule`` objects are normalized into the same
internal rule list before evaluation.  Downstream modules can reuse that
normalizer, so governance semantics are not split across multiple ad-hoc parsers.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .schema import FinalCauseQuery, GoalSpec
from .utility import _is_protected_like


@dataclass
class NormativeRule:
    """One normalized rule in a normative/value policy.

    target_type is normally one of: ``goal``, ``action``, ``outcome``,
    ``constraint``.  status is normally one of: ``allowed``, ``prohibited``,
    ``required``, ``protected``, ``escalation_required``, ``monitored``, or
    ``discouraged``.
    """

    target: str = ""
    target_type: str = "goal"
    status: str = "allowed"
    severity: float = 1.0
    reason: str = ""
    source: str = "explicit"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(
        cls,
        payload: Any,
        *,
        default_target_type: str = "goal",
        default_status: str = "allowed",
        default_source: str = "explicit",
        default_severity: float = 1.0,
    ) -> Optional["NormativeRule"]:
        return _rule_from_payload(
            payload,
            default_target_type=default_target_type,
            default_status=default_status,
            default_source=default_source,
            default_severity=default_severity,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class NormalizedNormativePolicy:
    """Canonical policy representation shared by normative and recommendation layers."""

    assessed: bool = False
    rules: List[NormativeRule] = field(default_factory=list)
    strict_goal_allowlist: bool = False
    strict_action_allowlist: bool = False
    raw: Dict[str, Any] = field(default_factory=dict)
    reason_codes: List[str] = field(default_factory=list)
    limits: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["rules"] = [rule.to_dict() for rule in self.rules]
        data["allowed_goals"] = self.targets("goal", "allowed")
        data["prohibited_goals"] = self.targets("goal", "prohibited")
        data["required_goals"] = self.targets("goal", "required")
        data["protected_goals"] = self.targets("goal", "protected")
        data["escalation_goals"] = self.targets("goal", "escalation_required")
        data["monitored_goals"] = self.targets("goal", "monitored")
        data["discouraged_goals"] = self.targets("goal", "discouraged")
        data["allowed_actions"] = self.targets("action", "allowed")
        data["prohibited_actions"] = self.targets("action", "prohibited")
        data["escalation_actions"] = self.targets("action", "escalation_required")
        data["discouraged_actions"] = self.targets("action", "discouraged")
        return data

    def matching_rules(self, target: str, target_type: str) -> List[NormativeRule]:
        target = _clean_str(target)
        target_type = _normalise_target_type(target_type)
        return [rule for rule in self.rules if rule.target == target and rule.target_type == target_type]

    def status_for_target(self, target: str, target_type: str, *, allowlist_mode: Optional[bool] = None) -> str:
        if allowlist_mode is None:
            allowlist_mode = self.strict_goal_allowlist if _normalise_target_type(target_type) == "goal" else self.strict_action_allowlist
        return _status_from_rules(self.rules, target, target_type, allowlist_mode=bool(allowlist_mode))

    def dominant_rule(self, target: str, target_type: str) -> Optional[NormativeRule]:
        return _dominant_rule(self.matching_rules(target, target_type))

    def severity_for_target(self, target: str, target_type: str) -> float:
        rule = self.dominant_rule(target, target_type)
        return float(rule.severity) if rule is not None else 0.0

    def targets(self, target_type: str, status: Optional[str] = None) -> List[str]:
        target_type = _normalise_target_type(target_type)
        status = _normalise_status(status) if status else None
        items = [rule.target for rule in self.rules if rule.target_type == target_type and (status is None or rule.status == status)]
        return _unique(items)


@dataclass
class NormativeSFMAudit:
    """Normative classification for one candidate final cause."""

    assessed: bool = False
    goal_variable: str = ""
    observed_action: str = ""
    goal_status: str = "unspecified"
    action_status: str = "unspecified"
    alignment_status: str = "unassessed"
    normatively_aligned: bool = False
    prohibited: bool = False
    requires_escalation: bool = False
    protected_goal_like: bool = False
    allowlist_mode: bool = False
    support_strength: float = 0.0
    goal_rule_severity: float = 0.0
    action_rule_severity: float = 0.0
    max_applicable_severity: float = 0.0
    normalized_rule_count: int = 0
    allowed_goals: List[str] = field(default_factory=list)
    prohibited_goals: List[str] = field(default_factory=list)
    required_goals: List[str] = field(default_factory=list)
    protected_goals: List[str] = field(default_factory=list)
    escalation_goals: List[str] = field(default_factory=list)
    monitored_goals: List[str] = field(default_factory=list)
    prohibited_actions: List[str] = field(default_factory=list)
    allowed_actions: List[str] = field(default_factory=list)
    applicable_rules: List[Dict[str, Any]] = field(default_factory=list)
    normalized_policy: Dict[str, Any] = field(default_factory=dict)
    authority_status: str = "diagnostic_only"
    reason: str = ""
    reason_codes: List[str] = field(default_factory=list)
    limits: List[str] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Low-level coercion helpers


def _clean_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _clean_lower(value: Any, default: str = "") -> str:
    return _clean_str(value, default).lower()


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
    policy = raw.get("normative_policy") or raw.get("value_policy") or raw.get("alignment_policy") or raw.get("policy")
    return _as_dict(policy)


# ---------------------------------------------------------------------------
# Rule normalization


def _normalise_status(status: Any) -> str:
    value = _clean_lower(status, "allowed")
    aliases = {
        "allow": "allowed",
        "permit": "allowed",
        "permitted": "allowed",
        "ok": "allowed",
        "safe": "allowed",
        "deny": "prohibited",
        "forbid": "prohibited",
        "forbidden": "prohibited",
        "blocked": "prohibited",
        "block": "prohibited",
        "ban": "prohibited",
        "banned": "prohibited",
        "must": "required",
        "mandatory": "required",
        "protect": "protected",
        "protected_outcome": "protected",
        "escalate": "escalation_required",
        "escalation": "escalation_required",
        "requires_escalation": "escalation_required",
        "review_required": "escalation_required",
        "monitor": "monitored",
        "watch": "monitored",
        "discourage": "discouraged",
        "avoid": "discouraged",
    }
    return aliases.get(value, value)


def _normalise_target_type(target_type: Any) -> str:
    value = _clean_lower(target_type, "goal")
    aliases = {
        "goals": "goal",
        "candidate_goal": "goal",
        "final_goal": "goal",
        "outcome_goal": "goal",
        "actions": "action",
        "candidate_action": "action",
        "agent_action": "action",
        "intervention": "action",
        "outcomes": "outcome",
        "protected_outcome": "outcome",
        "constraints": "constraint",
    }
    return aliases.get(value, value)


def _rule_target(payload: Mapping[str, Any]) -> str:
    return _clean_str(
        payload.get("target")
        or payload.get("goal")
        or payload.get("outcome")
        or payload.get("action")
        or payload.get("constraint")
        or payload.get("name")
    )


def _rule_from_payload(
    payload: Any,
    *,
    default_target_type: str = "goal",
    default_status: str = "allowed",
    default_source: str = "explicit",
    default_severity: float = 1.0,
) -> Optional[NormativeRule]:
    if isinstance(payload, NormativeRule):
        return NormativeRule(
            target=payload.target,
            target_type=_normalise_target_type(payload.target_type),
            status=_normalise_status(payload.status),
            severity=max(0.0, _safe_float(payload.severity, default_severity)),
            reason=payload.reason,
            source=payload.source or default_source,
            metadata=dict(payload.metadata or {}),
        )
    if isinstance(payload, str):
        target = _clean_str(payload)
        if not target:
            return None
        return NormativeRule(
            target=target,
            target_type=_normalise_target_type(default_target_type),
            status=_normalise_status(default_status),
            severity=max(0.0, float(default_severity)),
            source=default_source,
        )
    if not isinstance(payload, Mapping):
        return None
    target = _rule_target(payload)
    if not target:
        return None
    target_type = _normalise_target_type(payload.get("target_type") or payload.get("type") or default_target_type)
    status = _normalise_status(payload.get("status") or payload.get("normative_status") or default_status)
    severity = max(0.0, _safe_float(payload.get("severity", payload.get("weight", default_severity)), default_severity))
    metadata = _as_dict(payload.get("metadata"))
    # Preserve unknown fields as metadata for auditability without polluting the
    # first-class rule contract.
    for key, value in payload.items():
        if key not in {
            "target",
            "goal",
            "outcome",
            "action",
            "constraint",
            "name",
            "target_type",
            "type",
            "status",
            "normative_status",
            "severity",
            "weight",
            "reason",
            "rationale",
            "source",
            "metadata",
        }:
            metadata.setdefault(key, value)
    return NormativeRule(
        target=target,
        target_type=target_type,
        status=status,
        severity=severity,
        reason=_clean_str(payload.get("reason") or payload.get("rationale")),
        source=_clean_str(payload.get("source"), default_source),
        metadata=metadata,
    )


def _severity_default(policy: Mapping[str, Any], source: str, status: str) -> float:
    status = _normalise_status(status)
    candidates = [
        f"{source}_severity",
        f"{status}_severity",
        f"default_{status}_severity",
        "default_severity",
    ]
    for key in candidates:
        if key in policy:
            return max(0.0, _safe_float(policy.get(key), 1.0))
    return 1.0


def _rules_from_key_group(
    policy: Mapping[str, Any],
    keys: Sequence[str],
    *,
    target_type: str,
    status: str,
    source: str,
) -> List[NormativeRule]:
    rules: List[NormativeRule] = []
    default_severity = _severity_default(policy, source, status)
    for key in keys:
        for item in _as_list(policy.get(key)):
            rule = _rule_from_payload(
                item,
                default_target_type=target_type,
                default_status=status,
                default_source=key,
                default_severity=default_severity,
            )
            if rule is not None:
                rules.append(rule)
    return rules


def _explicit_rules_from_policy(policy: Mapping[str, Any]) -> List[NormativeRule]:
    rules: List[NormativeRule] = []
    for item in _as_list(policy.get("rules") or policy.get("goal_rules") or policy.get("normative_rules")):
        rule = _rule_from_payload(item, default_target_type="goal", default_source="rules")
        if rule is not None:
            rules.append(rule)
    for item in _as_list(policy.get("action_rules")):
        rule = _rule_from_payload(item, default_target_type="action", default_source="action_rules")
        if rule is not None:
            rules.append(rule)
    for item in _as_list(policy.get("outcome_rules")):
        rule = _rule_from_payload(item, default_target_type="outcome", default_source="outcome_rules")
        if rule is not None:
            rules.append(rule)
    for item in _as_list(policy.get("constraint_rules")):
        rule = _rule_from_payload(item, default_target_type="constraint", default_source="constraint_rules")
        if rule is not None:
            rules.append(rule)
    return rules


def _dedupe_rules(rules: Iterable[NormativeRule]) -> List[NormativeRule]:
    out: List[NormativeRule] = []
    seen: set[Tuple[str, str, str, str, float]] = set()
    for rule in rules:
        if not rule.target:
            continue
        key = (rule.target, rule.target_type, rule.status, rule.source, round(float(rule.severity), 12))
        if key not in seen:
            seen.add(key)
            out.append(rule)
    return out


def normalize_normative_policy(
    policy: Any,
    *,
    derived_rules: Optional[Iterable[Any]] = None,
) -> NormalizedNormativePolicy:
    """Normalize inline policy lists and rich rules into one canonical contract.

    The helper accepts the legacy compact JSON shape:

    ``{"allowed_goals": ["task_success"], "prohibited_actions": ["unsafe"]}``

    and the richer rule shape:

    ``{"rules": [{"target": "task_success", "target_type": "goal", "status": "allowed", "severity": 0.8}]}``

    Both become ``NormativeRule`` rows and are evaluated with the same precedence
    and allowlist logic.
    """

    policy_dict = _as_dict(policy)
    if not policy_dict and not derived_rules:
        return NormalizedNormativePolicy(
            assessed=False,
            raw=policy_dict,
            reason_codes=["SFM_NORMATIVE_POLICY_NOT_SUPPLIED"],
            limits=["normative_policy_required_for_value_alignment_layer"],
        )

    rules: List[NormativeRule] = []
    rules.extend(_explicit_rules_from_policy(policy_dict))
    rules.extend(_rules_from_key_group(policy_dict, ("allowed_goals", "permitted_goals"), target_type="goal", status="allowed", source="allowed_goals"))
    rules.extend(_rules_from_key_group(policy_dict, ("prohibited_goals", "forbidden_goals", "blocked_goals"), target_type="goal", status="prohibited", source="prohibited_goals"))
    rules.extend(_rules_from_key_group(policy_dict, ("required_goals", "mandatory_goals"), target_type="goal", status="required", source="required_goals"))
    rules.extend(_rules_from_key_group(policy_dict, ("protected_goals", "protected_outcomes"), target_type="goal", status="protected", source="protected_goals"))
    rules.extend(_rules_from_key_group(policy_dict, ("escalation_goals", "escalate_goals", "goals_requiring_escalation"), target_type="goal", status="escalation_required", source="escalation_goals"))
    rules.extend(_rules_from_key_group(policy_dict, ("monitored_goals", "monitor_goals"), target_type="goal", status="monitored", source="monitored_goals"))
    rules.extend(_rules_from_key_group(policy_dict, ("discouraged_goals", "avoid_goals"), target_type="goal", status="discouraged", source="discouraged_goals"))
    rules.extend(_rules_from_key_group(policy_dict, ("allowed_actions", "permitted_actions"), target_type="action", status="allowed", source="allowed_actions"))
    rules.extend(_rules_from_key_group(policy_dict, ("prohibited_actions", "forbidden_actions", "blocked_actions"), target_type="action", status="prohibited", source="prohibited_actions"))
    rules.extend(_rules_from_key_group(policy_dict, ("escalation_actions", "actions_requiring_escalation"), target_type="action", status="escalation_required", source="escalation_actions"))
    rules.extend(_rules_from_key_group(policy_dict, ("discouraged_actions", "avoid_actions"), target_type="action", status="discouraged", source="discouraged_actions"))

    for item in derived_rules or []:
        rule = _rule_from_payload(item, default_target_type="goal", default_status="protected", default_source="derived")
        if rule is not None:
            rules.append(rule)

    rules = _dedupe_rules(rules)
    allowed_goals = [rule.target for rule in rules if rule.target_type == "goal" and rule.status == "allowed"]
    strict_goal_allowlist = bool(policy_dict.get("strict_goal_allowlist", policy_dict.get("allowlist_mode", False))) or bool(allowed_goals)
    strict_action_allowlist = bool(policy_dict.get("strict_action_allowlist", False))

    return NormalizedNormativePolicy(
        assessed=True,
        rules=rules,
        strict_goal_allowlist=strict_goal_allowlist,
        strict_action_allowlist=strict_action_allowlist,
        raw=policy_dict,
        reason_codes=["SFM_NORMATIVE_POLICY_NORMALIZED"],
        limits=[] if rules else ["normative_policy_contains_no_rules_after_normalization"],
    )


def _dominant_rule(rules: List[NormativeRule]) -> Optional[NormativeRule]:
    if not rules:
        return None
    precedence = {
        "prohibited": 0,
        "escalation_required": 1,
        "protected": 2,
        "required": 3,
        "allowed": 4,
        "discouraged": 5,
        "monitored": 6,
    }
    return sorted(rules, key=lambda r: (precedence.get(r.status, 99), -float(r.severity)))[0]


def _status_from_rules(rules: List[NormativeRule], target: str, target_type: str, *, allowlist_mode: bool = False) -> str:
    target = _clean_str(target)
    target_type = _normalise_target_type(target_type)
    applicable = [r for r in rules if r.target == target and r.target_type == target_type]
    dominant = _dominant_rule(applicable)
    if dominant is None:
        return "not_on_allowlist" if allowlist_mode else "unspecified"
    return dominant.status


def normative_status_for_target(policy: Any, target: str, target_type: str = "goal") -> str:
    """Convenience API for downstream modules needing the canonical status."""

    normalized = normalize_normative_policy(policy)
    return normalized.status_for_target(target, target_type)


# ---------------------------------------------------------------------------
# Evaluator


class NormativeSFMEvaluator:
    """Classify candidate final causes against a normalized normative/value policy."""

    def evaluate(
        self,
        query: FinalCauseQuery,
        goal: GoalSpec,
        *,
        constraint_support: Optional[Mapping[str, Any]] = None,
        hierarchical_goal: Optional[Mapping[str, Any]] = None,
    ) -> NormativeSFMAudit:
        policy = _policy_from_query(query)
        derived_protected = _unique(
            [
                query.protected_outcome,
                *list((constraint_support or {}).get("protected_constraints") or []),
                *list((constraint_support or {}).get("hard_constraints") or []),
            ]
        )
        derived_rules = [
            NormativeRule(target=item, target_type="goal", status="protected", source="derived_protected_constraint")
            for item in derived_protected
            if item
        ]
        normalized = normalize_normative_policy(policy, derived_rules=derived_rules)
        if not policy:
            return NormativeSFMAudit(
                assessed=False,
                goal_variable=goal.goal_variable,
                observed_action=query.observed_action,
                normalized_policy=normalized.to_dict(),
                reason="No normative/value policy was supplied for SFM value-alignment classification.",
                reason_codes=["SFM_NORMATIVE_POLICY_NOT_SUPPLIED"],
                limits=["normative_policy_required_for_value_alignment_layer"],
            )

        goal_status = normalized.status_for_target(goal.goal_variable, "goal", allowlist_mode=normalized.strict_goal_allowlist)
        action_status = normalized.status_for_target(query.observed_action, "action", allowlist_mode=normalized.strict_action_allowlist)
        applicable = [
            rule
            for rule in normalized.rules
            if (rule.target == goal.goal_variable and rule.target_type == "goal")
            or (rule.target == query.observed_action and rule.target_type == "action")
        ]

        protected_goals = _unique([*normalized.targets("goal", "protected"), *derived_protected])
        protected_goal_like = goal_status == "protected" or goal.goal_variable in protected_goals or _is_protected_like(goal.goal_variable)
        prohibited = goal_status in {"prohibited", "not_on_allowlist"} or action_status in {"prohibited", "not_on_allowlist"}
        requires_escalation = goal_status == "escalation_required" or action_status == "escalation_required"
        goal_severity = normalized.severity_for_target(goal.goal_variable, "goal")
        action_severity = normalized.severity_for_target(query.observed_action, "action")
        max_severity = max([goal_severity, action_severity, *[float(rule.severity) for rule in applicable]] or [0.0])

        if goal_status == "required" and not prohibited and not requires_escalation:
            alignment_status = "required_goal_supported"
            support_strength = 1.0
            normatively_aligned = True
        elif goal_status == "allowed" and not prohibited and not requires_escalation:
            alignment_status = "normatively_aligned"
            support_strength = 0.95
            normatively_aligned = True
        elif goal_status == "monitored" and not prohibited:
            alignment_status = "monitored_goal"
            support_strength = 0.65
            normatively_aligned = not requires_escalation
        elif goal_status == "discouraged" and not prohibited:
            alignment_status = "discouraged_goal"
            support_strength = 0.35
            normatively_aligned = False
        elif goal_status == "protected":
            alignment_status = "protected_outcome_not_final_goal"
            support_strength = 0.30
            normatively_aligned = False
        elif requires_escalation:
            alignment_status = "normative_escalation_required"
            support_strength = 0.40
            normatively_aligned = False
        elif prohibited:
            alignment_status = "normatively_prohibited"
            support_strength = 0.0
            normatively_aligned = False
        else:
            alignment_status = "normatively_unclassified"
            support_strength = 0.25
            normatively_aligned = False

        # Low-severity allow/monitor rules should not look as strong as
        # high-severity permissions.  Blocking statuses remain blocking;
        # severity is reported rather than used to weaken the block.
        if normatively_aligned and max_severity > 0:
            support_strength = min(1.0, support_strength * (0.5 + 0.5 * min(max_severity, 1.0)))
        elif goal_status in {"monitored", "discouraged", "protected", "escalation_required"} and max_severity > 0:
            support_strength = min(1.0, support_strength * (0.5 + 0.5 * min(max_severity, 1.0)))

        reason_codes: List[str] = ["SFM_NORMATIVE_POLICY_ASSESSED", "SFM_NORMATIVE_POLICY_NORMALIZED"]
        if goal_status == "allowed":
            reason_codes.append("SFM_NORMATIVE_GOAL_ALLOWED")
        elif goal_status == "required":
            reason_codes.append("SFM_NORMATIVE_GOAL_REQUIRED")
        elif goal_status == "prohibited":
            reason_codes.append("SFM_NORMATIVE_GOAL_PROHIBITED")
        elif goal_status == "protected":
            reason_codes.append("SFM_NORMATIVE_GOAL_PROTECTED_NOT_FINAL")
        elif goal_status == "escalation_required":
            reason_codes.append("SFM_NORMATIVE_GOAL_ESCALATION_REQUIRED")
        elif goal_status == "not_on_allowlist":
            reason_codes.append("SFM_NORMATIVE_GOAL_NOT_ON_ALLOWLIST")
        else:
            reason_codes.append("SFM_NORMATIVE_GOAL_UNSPECIFIED")

        if action_status == "allowed":
            reason_codes.append("SFM_NORMATIVE_ACTION_ALLOWED")
        elif action_status == "prohibited":
            reason_codes.append("SFM_NORMATIVE_ACTION_PROHIBITED")
        elif action_status == "escalation_required":
            reason_codes.append("SFM_NORMATIVE_ACTION_ESCALATION_REQUIRED")
        elif action_status == "discouraged":
            reason_codes.append("SFM_NORMATIVE_ACTION_DISCOURAGED")
        elif action_status == "not_on_allowlist":
            reason_codes.append("SFM_NORMATIVE_ACTION_NOT_ON_ALLOWLIST")

        if normatively_aligned:
            reason_codes.append("SFM_NORMATIVE_ALIGNMENT_PASS")
        else:
            reason_codes.append("SFM_NORMATIVE_ALIGNMENT_NOT_CONFIRMED")
        if prohibited:
            reason_codes.append("SFM_NORMATIVE_ALIGNMENT_FAIL")
        if requires_escalation:
            reason_codes.append("SFM_NORMATIVE_ESCALATION_REQUIRED")
        if protected_goal_like:
            reason_codes.append("SFM_NORMATIVE_PROTECTED_OUTCOME_NOT_PROMOTED_TO_FINAL_GOAL")
        if max_severity:
            reason_codes.append("SFM_NORMATIVE_RULE_SEVERITY_RECORDED")

        limits: List[str] = []
        if alignment_status == "normatively_unclassified":
            limits.append("candidate_goal_not_classified_by_normative_policy")
        if prohibited:
            limits.append("candidate_goal_or_action_normatively_prohibited")
        if requires_escalation:
            limits.append("normative_escalation_required_before_acting_on_goal")
        if protected_goal_like:
            limits.append("candidate_goal_is_protected_outcome_or_constraint")
        limits.extend(normalized.limits)

        return NormativeSFMAudit(
            assessed=True,
            goal_variable=goal.goal_variable,
            observed_action=query.observed_action,
            goal_status=goal_status,
            action_status=action_status,
            alignment_status=alignment_status,
            normatively_aligned=normatively_aligned,
            prohibited=prohibited,
            requires_escalation=requires_escalation,
            protected_goal_like=protected_goal_like,
            allowlist_mode=normalized.strict_goal_allowlist,
            support_strength=round(support_strength, 6),
            goal_rule_severity=round(goal_severity, 6),
            action_rule_severity=round(action_severity, 6),
            max_applicable_severity=round(max_severity, 6),
            normalized_rule_count=len(normalized.rules),
            allowed_goals=normalized.targets("goal", "allowed"),
            prohibited_goals=normalized.targets("goal", "prohibited"),
            required_goals=normalized.targets("goal", "required"),
            protected_goals=protected_goals,
            escalation_goals=normalized.targets("goal", "escalation_required"),
            monitored_goals=normalized.targets("goal", "monitored"),
            prohibited_actions=normalized.targets("action", "prohibited"),
            allowed_actions=normalized.targets("action", "allowed"),
            applicable_rules=[rule.to_dict() for rule in applicable],
            normalized_policy=normalized.to_dict(),
            authority_status="value_alignment_diagnostic",
            reason="Normative SFM classification uses a normalized rule policy and separates the pursued goal from whether that goal/action is permitted.",
            reason_codes=reason_codes,
            limits=_unique(limits),
            raw={"policy": dict(policy), "normalized_policy": normalized.to_dict(), "hierarchical_goal": dict(hierarchical_goal or {})},
        )


def evaluate_normative_sfm(payload: Any, goal: Any = None) -> Dict[str, Any]:
    """Convenience wrapper for normative SFM diagnostics.

    If goal is omitted, the first candidate goal from the query is evaluated.
    """

    query = FinalCauseQuery.from_payload(payload)
    target_goal = GoalSpec.from_payload(goal) if goal is not None else (query.candidate_goals[0] if query.candidate_goals else GoalSpec(goal_variable=""))
    return NormativeSFMEvaluator().evaluate(query, target_goal).to_dict()
