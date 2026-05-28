from __future__ import annotations

"""Structured AST objects for Amantia do-calculus expressions.

Step 1 for a real do-calculus proof layer: represent expressions such as
``P(Y | do(X), Z)`` as deterministic data, not only as display strings.

This module is intentionally small and conservative.  It does not grant ID or
veto authority; it only gives later rewrite/proof modules a stable object model
for Pearl-rule transformations.
"""

from dataclasses import asdict, dataclass, field
import json
import re
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple


DO_AST_VERSION = "do_ast_v1"


def _s(value: object) -> str:
    raw = "" if value is None else str(value).strip()
    return "" if raw.lower() in {"nan", "none", "null"} else raw


def _dedupe(values: Iterable[object]) -> Tuple[str, ...]:
    out: List[str] = []
    seen = set()
    for value in values or []:
        item = _s(value)
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return tuple(out)


def _split_vars(value: object) -> Tuple[str, ...]:
    text = _s(value)
    if not text:
        return tuple()
    for ch in "{}[]'\"":
        text = text.replace(ch, "")
    return _dedupe(part.strip() for part in re.split(r"[,|]", text) if part.strip())


def _split_top_level_commas(text: str) -> List[str]:
    """Split on commas that are not inside parentheses.

    Needed for conditions like ``do(x,z),w`` where the comma inside ``do`` must
    not create a separate condition token.
    """
    tokens: List[str] = []
    depth = 0
    start = 0
    for idx, ch in enumerate(text):
        if ch == "(":
            depth += 1
        elif ch == ")" and depth > 0:
            depth -= 1
        elif ch == "," and depth == 0:
            tokens.append(text[start:idx].strip())
            start = idx + 1
    tail = text[start:].strip()
    if tail:
        tokens.append(tail)
    return [tok for tok in tokens if tok]


@dataclass(frozen=True)
class DoCondition:
    """Observed variables in the conditioning bar of a probability term."""

    variables: Tuple[str, ...] = tuple()

    def __post_init__(self) -> None:
        object.__setattr__(self, "variables", _dedupe(self.variables))

    def to_dict(self) -> dict:
        return {"variables": list(self.variables)}


@dataclass(frozen=True)
class DoIntervention:
    """Variables under the do-operator in a probability term."""

    variables: Tuple[str, ...] = tuple()

    def __post_init__(self) -> None:
        object.__setattr__(self, "variables", _dedupe(self.variables))

    def to_dict(self) -> dict:
        return {"variables": list(self.variables)}


@dataclass(frozen=True)
class DoExpression:
    """AST for a do-calculus probability expression.

    ``outcomes`` are the left side of ``P(...)``.
    ``interventions`` are variables inside ``do(...)`` terms.
    ``observations`` are ordinary conditioning variables.
    ``summations`` is included now so frontdoor/backdoor derivations can later
    carry ``sum_z`` without inventing a second AST format.
    """

    outcomes: Tuple[str, ...] = tuple()
    interventions: Tuple[str, ...] = tuple()
    observations: Tuple[str, ...] = tuple()
    summations: Tuple[str, ...] = tuple()
    label: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "outcomes", _dedupe(self.outcomes))
        object.__setattr__(self, "interventions", _dedupe(self.interventions))
        object.__setattr__(self, "observations", _dedupe(self.observations))
        object.__setattr__(self, "summations", _dedupe(self.summations))
        object.__setattr__(self, "label", _s(self.label))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def is_observational(self) -> bool:
        return not self.interventions

    @property
    def has_do(self) -> bool:
        return bool(self.interventions)

    def to_dict(self) -> dict:
        payload = {
            "ast_version": DO_AST_VERSION,
            "node_type": "do_probability",
            "outcomes": list(self.outcomes),
            "interventions": list(self.interventions),
            "observations": list(self.observations),
            "summations": list(self.summations),
            "is_observational": self.is_observational,
        }
        if self.label:
            payload["label"] = self.label
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        payload["formula"] = self.to_formula()
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def to_formula(self) -> str:
        lhs = ",".join(self.outcomes)
        cond_parts: List[str] = []
        if self.interventions:
            cond_parts.append("do(" + ",".join(self.interventions) + ")")
        cond_parts.extend(self.observations)
        base = f"P({lhs}|{','.join(cond_parts)})" if cond_parts else f"P({lhs})"
        if self.summations:
            return "sum_{" + ",".join(self.summations) + "} " + base
        return base

    def with_observations(self, observations: Sequence[object], *, label: Optional[str] = None) -> "DoExpression":
        return DoExpression(
            outcomes=self.outcomes,
            interventions=self.interventions,
            observations=observations,
            summations=self.summations,
            label=self.label if label is None else _s(label),
            metadata=self.metadata,
        )

    def with_interventions(self, interventions: Sequence[object], *, label: Optional[str] = None) -> "DoExpression":
        return DoExpression(
            outcomes=self.outcomes,
            interventions=interventions,
            observations=self.observations,
            summations=self.summations,
            label=self.label if label is None else _s(label),
            metadata=self.metadata,
        )

    def remove_observations(self, variables: Sequence[object], *, label: Optional[str] = None) -> "DoExpression":
        remove = set(_dedupe(variables))
        return self.with_observations([v for v in self.observations if v not in remove], label=label)

    def add_observations(self, variables: Sequence[object], *, label: Optional[str] = None) -> "DoExpression":
        return self.with_observations(list(self.observations) + list(_dedupe(variables)), label=label)

    def remove_interventions(self, variables: Sequence[object], *, label: Optional[str] = None) -> "DoExpression":
        remove = set(_dedupe(variables))
        return self.with_interventions([v for v in self.interventions if v not in remove], label=label)

    def add_interventions(self, variables: Sequence[object], *, label: Optional[str] = None) -> "DoExpression":
        return self.with_interventions(list(self.interventions) + list(_dedupe(variables)), label=label)

    def exchange_intervention_for_observation(self, variables: Sequence[object], *, label: Optional[str] = None) -> "DoExpression":
        move = set(_dedupe(variables))
        remaining_do = [v for v in self.interventions if v not in move]
        moved_obs = [v for v in self.interventions if v in move]
        new_obs = moved_obs + list(self.observations)
        return DoExpression(
            outcomes=self.outcomes,
            interventions=remaining_do,
            observations=new_obs,
            summations=self.summations,
            label=self.label if label is None else _s(label),
            metadata=self.metadata,
        )


@dataclass(frozen=True)
class DoRewriteStep:
    """Single Pearl-rule rewrite candidate, independent of authority."""

    rule: str
    before: DoExpression
    after: DoExpression
    applicable: bool = False
    premise: str = ""
    graph_variant: str = ""
    reason_codes: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "rule", _s(self.rule))
        object.__setattr__(self, "premise", _s(self.premise))
        object.__setattr__(self, "graph_variant", _s(self.graph_variant))
        object.__setattr__(self, "reason_codes", _s(self.reason_codes))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def to_dict(self) -> dict:
        payload = {
            "ast_version": DO_AST_VERSION,
            "node_type": "do_rewrite_step",
            "rule": self.rule,
            "applicable": bool(self.applicable),
            "before": self.before.to_dict(),
            "after": self.after.to_dict(),
            "premise": self.premise,
            "graph_variant": self.graph_variant,
            "reason_codes": self.reason_codes,
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class DoProof:
    """Container for a future do-calculus derivation trace.

    In Step 1 this is only a structured audit/proof shell.  Later steps can add
    search results without changing contract shape.
    """

    query: DoExpression
    steps: Tuple[DoRewriteStep, ...] = tuple()
    status: str = "audit_only"
    authority: str = "audit_only"
    terminal: Optional[DoExpression] = None
    reason_codes: str = "DO_AST_ONLY_NO_ID_AUTHORITY"
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "steps", tuple(self.steps or ()))
        object.__setattr__(self, "status", _s(self.status) or "audit_only")
        object.__setattr__(self, "authority", _s(self.authority) or "audit_only")
        object.__setattr__(self, "reason_codes", _s(self.reason_codes))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def terminal_expression(self) -> DoExpression:
        return self.terminal or (self.steps[-1].after if self.steps else self.query)

    def to_dict(self) -> dict:
        payload = {
            "ast_version": DO_AST_VERSION,
            "node_type": "do_proof",
            "status": self.status,
            "authority": self.authority,
            "query": self.query.to_dict(),
            "steps": [step.to_dict() for step in self.steps],
            "terminal": self.terminal_expression.to_dict(),
            "reason_codes": self.reason_codes,
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


def P_do(
    outcomes: Sequence[object],
    *,
    interventions: Sequence[object] = (),
    observations: Sequence[object] = (),
    summations: Sequence[object] = (),
    label: str = "",
    metadata: Optional[Mapping[str, object]] = None,
) -> DoExpression:
    return DoExpression(
        outcomes=_dedupe(outcomes),
        interventions=_dedupe(interventions),
        observations=_dedupe(observations),
        summations=_dedupe(summations),
        label=label,
        metadata=dict(metadata or {}),
    )


def expression_from_dict(payload: Mapping[str, object]) -> DoExpression:
    return DoExpression(
        outcomes=_dedupe(payload.get("outcomes", [])),
        interventions=_dedupe(payload.get("interventions", [])),
        observations=_dedupe(payload.get("observations", [])),
        summations=_dedupe(payload.get("summations", [])),
        label=_s(payload.get("label")),
        metadata=payload.get("metadata", {}) if isinstance(payload.get("metadata", {}), Mapping) else {},
    )


def rewrite_step_from_dict(payload: Mapping[str, object]) -> DoRewriteStep:
    before = payload.get("before", {})
    after = payload.get("after", {})
    return DoRewriteStep(
        rule=_s(payload.get("rule")),
        before=expression_from_dict(before if isinstance(before, Mapping) else {}),
        after=expression_from_dict(after if isinstance(after, Mapping) else {}),
        applicable=bool(payload.get("applicable", False)),
        premise=_s(payload.get("premise")),
        graph_variant=_s(payload.get("graph_variant")),
        reason_codes=_s(payload.get("reason_codes")),
        metadata=payload.get("metadata", {}) if isinstance(payload.get("metadata", {}), Mapping) else {},
    )


def proof_from_dict(payload: Mapping[str, object]) -> DoProof:
    query = payload.get("query", {})
    terminal = payload.get("terminal")
    steps: List[DoRewriteStep] = []
    for step_payload in payload.get("steps", []) if isinstance(payload.get("steps", []), list) else []:
        if isinstance(step_payload, Mapping):
            steps.append(rewrite_step_from_dict(step_payload))
    return DoProof(
        query=expression_from_dict(query if isinstance(query, Mapping) else {}),
        steps=tuple(steps),
        status=_s(payload.get("status")) or "audit_only",
        authority=_s(payload.get("authority")) or "audit_only",
        terminal=expression_from_dict(terminal) if isinstance(terminal, Mapping) else None,
        reason_codes=_s(payload.get("reason_codes")) or "DO_AST_ONLY_NO_ID_AUTHORITY",
        metadata=payload.get("metadata", {}) if isinstance(payload.get("metadata", {}), Mapping) else {},
    )


def parse_do_expression(formula: object) -> DoExpression:
    """Best-effort parser for compact probability strings.

    Supported examples:
    - ``P(y|do(x),z)``
    - ``P(y|do(x),do(z),w)``
    - ``sum_{z} P(y|do(x),z)``

    Unknown or malformed strings are preserved as metadata/label rather than
    converted into causal authority.
    """
    text = _s(formula)
    if not text:
        return DoExpression(label="empty", metadata={"parse_status": "empty"})

    summations: Tuple[str, ...] = tuple()
    body = text
    sum_match = re.match(r"^sum_\{(?P<vars>[^}]*)\}\s*(?P<body>.*)$", body)
    if sum_match:
        summations = _split_vars(sum_match.group("vars"))
        body = _s(sum_match.group("body"))

    m = re.match(r"^P\((?P<body>.*)\)$", body)
    if not m:
        return DoExpression(label=text, metadata={"parse_status": "unsupported_formula"})

    inside = m.group("body") or ""
    if "|" in inside:
        left, right = inside.split("|", 1)
    else:
        left, right = inside, ""
    outcomes = _split_vars(left)
    interventions: List[str] = []
    observations: List[str] = []
    for token in _split_top_level_commas(right):
        do_match = re.match(r"^do\((?P<vars>[^)]*)\)$", token.strip())
        if do_match:
            interventions.extend(_split_vars(do_match.group("vars")))
        else:
            observations.extend(_split_vars(token))

    return DoExpression(
        outcomes=outcomes,
        interventions=interventions,
        observations=observations,
        summations=summations,
        label=text,
        metadata={"parse_status": "ok"},
    )


def expression_to_dict(expr: DoExpression) -> dict:
    return expr.to_dict()


def expression_to_json(expr: DoExpression) -> str:
    return expr.to_json()


def expression_latex(expr: DoExpression) -> str:
    lhs = ",".join(expr.outcomes)
    cond_parts: List[str] = []
    if expr.interventions:
        cond_parts.append("do(" + ",".join(expr.interventions) + ")")
    cond_parts.extend(expr.observations)
    prob = f"P({lhs} \\mid {','.join(cond_parts)})" if cond_parts else f"P({lhs})"
    if expr.summations:
        return f"\\sum_{{{','.join(expr.summations)}}} {prob}"
    return prob


__all__ = [
    "DO_AST_VERSION",
    "DoCondition",
    "DoIntervention",
    "DoExpression",
    "DoRewriteStep",
    "DoProof",
    "P_do",
    "expression_from_dict",
    "rewrite_step_from_dict",
    "proof_from_dict",
    "parse_do_expression",
    "expression_to_dict",
    "expression_to_json",
    "expression_latex",
]
