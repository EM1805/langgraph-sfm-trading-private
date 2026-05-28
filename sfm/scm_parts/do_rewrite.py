from __future__ import annotations

"""Conservative AST rewrite layer for Amantia do-calculus.

Step 71/72 bridge: convert audited Pearl-rule diagnostics from
``do_calculus.py`` into structured ``DoRewriteStep`` objects from
``do_ast.py``.  This module does not run proof search and grants no
identification/veto authority.  It only makes single symbolic moves executable
and auditable for the bounded proof engine.
"""

from dataclasses import asdict, dataclass, field
import json
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from .admg import ADMG
from .do_ast import DoExpression, DoProof, DoRewriteStep, expression_from_dict, parse_do_expression
from .do_calculus import (
    DoCalculusRuleDiagnostic,
    rule1_insertion_deletion_observation,
    rule2_action_observation_exchange,
    rule3_insertion_deletion_action,
)

DO_REWRITE_VERSION = "do_rewrite_v1"
DO_REWRITE_AUTHORITY = "audit_only"


def _s(value: object) -> str:
    raw = "" if value is None else str(value).strip()
    return "" if raw.lower() in {"nan", "none", "null"} else raw


def _dedupe(values: Iterable[object]) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values or []:
        item = _s(value)
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _expr_from_ast_json_or_formula(ast_json: object, formula: object) -> DoExpression:
    text = _s(ast_json)
    if text:
        try:
            payload = json.loads(text)
            if isinstance(payload, Mapping):
                return expression_from_dict(payload)
        except Exception:
            pass
    return parse_do_expression(formula)


def _premise_for_rule(rule: str) -> str:
    if rule == "rule1_observation_insertion_deletion":
        return "Y_dsep_Z_given_XW_in_G_bar_X"
    if rule == "rule2_action_observation_exchange":
        return "Y_dsep_Z_given_XW_in_G_bar_X_under_Z"
    if rule == "rule3_action_insertion_deletion":
        return "Y_dsep_Z_given_XW_in_G_bar_X_bar_ZW"
    return "UNKNOWN_DO_CALCULUS_PREMISE"


@dataclass(frozen=True)
class DoRewriteAudit:
    """Stable wrapper around one AST rewrite candidate.

    ``authority`` remains ``audit_only`` even when the graph premise passes.
    Later proof engines may consume applicable steps, but this module itself
    never claims that an effect is identified.
    """

    step: DoRewriteStep
    status: str
    authority: str = DO_REWRITE_AUTHORITY
    rewrite_version: str = DO_REWRITE_VERSION
    source_rule_status: str = ""
    reason_codes: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", _s(self.status) or "blocked")
        object.__setattr__(self, "authority", _s(self.authority) or DO_REWRITE_AUTHORITY)
        object.__setattr__(self, "rewrite_version", _s(self.rewrite_version) or DO_REWRITE_VERSION)
        object.__setattr__(self, "source_rule_status", _s(self.source_rule_status))
        object.__setattr__(self, "reason_codes", _s(self.reason_codes))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def applicable(self) -> bool:
        return bool(self.step.applicable)

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "node_type": "do_rewrite_audit",
            "rewrite_version": self.rewrite_version,
            "status": self.status,
            "authority": self.authority,
            "applicable": bool(self.applicable),
            "source_rule_status": self.source_rule_status,
            "reason_codes": self.reason_codes,
            "step": self.step.to_dict(),
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


def rewrite_step_from_diagnostic(diag: DoCalculusRuleDiagnostic) -> DoRewriteStep:
    """Convert one graph diagnostic into an AST rewrite candidate."""
    before = _expr_from_ast_json_or_formula(diag.expression_before_ast_json, diag.expression_before)
    after = _expr_from_ast_json_or_formula(diag.expression_after_ast_json, diag.expression_after)
    metadata = {
        "source": "do_rewrite_from_do_calculus_diagnostic",
        "authority": DO_REWRITE_AUTHORITY,
        "rewrite_version": DO_REWRITE_VERSION,
        "rule_status": diag.status,
        "dsep_status": diag.dsep_status,
        "open_path_count": int(diag.open_path_count),
        "checked_path_count": int(diag.checked_path_count),
        "open_paths": diag.open_paths,
        "conditioned_on": diag.conditioned_on,
        "removed_incoming_to": diag.removed_incoming_to,
        "removed_outgoing_from": diag.removed_outgoing_from,
    }
    return DoRewriteStep(
        rule=diag.rule,
        before=before,
        after=after,
        applicable=bool(diag.applicable),
        premise=_premise_for_rule(diag.rule),
        graph_variant=diag.graph_variant,
        reason_codes=diag.reason_codes,
        metadata=metadata,
    )


def rewrite_audit_from_diagnostic(diag: DoCalculusRuleDiagnostic) -> DoRewriteAudit:
    step = rewrite_step_from_diagnostic(diag)
    return DoRewriteAudit(
        step=step,
        status="applicable" if step.applicable else "blocked",
        source_rule_status=diag.status,
        reason_codes=diag.reason_codes,
        metadata={"diagnostic": diag.to_dict()},
    )


def rule1_rewrite(admg: ADMG, *, y: object, x: object, z: object, w: Optional[Sequence[object]] = None) -> DoRewriteAudit:
    return rewrite_audit_from_diagnostic(rule1_insertion_deletion_observation(admg, y=y, x=x, z=z, w=w))


def rule2_rewrite(admg: ADMG, *, y: object, x: object, z: object, w: Optional[Sequence[object]] = None) -> DoRewriteAudit:
    return rewrite_audit_from_diagnostic(rule2_action_observation_exchange(admg, y=y, x=x, z=z, w=w))


def rule3_rewrite(admg: ADMG, *, y: object, x: object, z: object, w: Optional[Sequence[object]] = None) -> DoRewriteAudit:
    return rewrite_audit_from_diagnostic(rule3_insertion_deletion_action(admg, y=y, x=x, z=z, w=w))


def candidate_rewrites(
    admg: ADMG,
    *,
    y: object,
    x: object,
    candidate_z: Optional[Sequence[object]] = None,
    w: Optional[Sequence[object]] = None,
) -> List[DoRewriteAudit]:
    """Return conservative rule1/2/3 rewrite candidates for each Z.

    This is candidate generation only.  The bounded proof engine additionally
    checks that the candidate's ``before`` expression exactly matches the
    current AST state before applying it.
    """
    yy, xx = _s(y), _s(x)
    zs = [z for z in _dedupe(candidate_z or []) if z not in {yy, xx}]
    out: List[DoRewriteAudit] = []
    for zz in zs:
        out.append(rule1_rewrite(admg, y=yy, x=xx, z=zz, w=w))
        out.append(rule2_rewrite(admg, y=yy, x=xx, z=zz, w=w))
        out.append(rule3_rewrite(admg, y=yy, x=xx, z=zz, w=w))
    return out


def proof_shell_from_rewrites(
    query: DoExpression,
    rewrites: Sequence[DoRewriteAudit],
    *,
    status: str = "rewrite_shell_audit_only",
    reason_codes: str = "DO_REWRITE_SHELL_NO_SEARCH_AUTHORITY",
) -> DoProof:
    steps = tuple(a.step for a in rewrites if a.applicable)
    terminal = steps[-1].after if steps else query
    return DoProof(
        query=query,
        steps=steps,
        status=status,
        authority=DO_REWRITE_AUTHORITY,
        terminal=terminal,
        reason_codes=reason_codes,
        metadata={
            "source": "do_rewrite.proof_shell_from_rewrites",
            "rewrite_version": DO_REWRITE_VERSION,
            "candidate_count": len(rewrites),
            "applicable_count": len(steps),
        },
    )


__all__ = [
    "DO_REWRITE_VERSION",
    "DO_REWRITE_AUTHORITY",
    "DoRewriteAudit",
    "rewrite_step_from_diagnostic",
    "rewrite_audit_from_diagnostic",
    "rule1_rewrite",
    "rule2_rewrite",
    "rule3_rewrite",
    "candidate_rewrites",
    "proof_shell_from_rewrites",
]
