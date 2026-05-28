from __future__ import annotations

"""Independent structural verifier for Amantia do-calculus proof traces.

Step 75 adds a second conservative audit layer. It checks whether a proof trace
is internally coherent, but it does not grant identification, estimation, or
veto authority.
"""

from dataclasses import dataclass, field
import hashlib
import json
from typing import Iterable, Mapping, Optional, Sequence

from .do_ast import DoExpression, DoProof, DoRewriteStep, expression_from_dict, proof_from_dict

DO_PROOF_VERIFIER_VERSION = "do_proof_verifier_v1_step75_structural"
DO_PROOF_VERIFIER_AUTHORITY = "audit_only"
_ALLOWED_AUTHORITIES = {"", "audit_only"}


def _s(value: object) -> str:
    raw = "" if value is None else str(value).strip()
    return "" if raw.lower() in {"nan", "none", "null"} else raw


def _expr_key(expr: DoExpression) -> str:
    return json.dumps({
        "outcomes": list(expr.outcomes),
        "interventions": list(expr.interventions),
        "observations": list(expr.observations),
        "summations": list(expr.summations),
    }, sort_keys=True, separators=(",", ":"))


def _expr_nodes(expr: DoExpression) -> Sequence[str]:
    return tuple(expr.outcomes) + tuple(expr.interventions) + tuple(expr.observations) + tuple(expr.summations)


def _dedupe_reasons(reasons: Iterable[str]) -> str:
    out = []
    seen = set()
    for reason in reasons:
        item = _s(reason)
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return "|".join(out)


@dataclass(frozen=True)
class DoProofVerification:
    valid: int
    verification_status: str
    authority: str = DO_PROOF_VERIFIER_AUTHORITY
    verifier_version: str = DO_PROOF_VERIFIER_VERSION
    reason_codes: str = ""
    step_count: int = 0
    terminal_observational: int = 0
    terminal_formula: str = ""
    proof_hash: str = ""
    failed_step_index: Optional[int] = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "valid", int(bool(self.valid)))
        object.__setattr__(self, "verification_status", _s(self.verification_status) or "proof_trace_invalid_audit_only")
        object.__setattr__(self, "authority", _s(self.authority) or DO_PROOF_VERIFIER_AUTHORITY)
        object.__setattr__(self, "verifier_version", _s(self.verifier_version) or DO_PROOF_VERIFIER_VERSION)
        object.__setattr__(self, "reason_codes", _s(self.reason_codes))
        object.__setattr__(self, "step_count", int(self.step_count))
        object.__setattr__(self, "terminal_observational", int(bool(self.terminal_observational)))
        object.__setattr__(self, "terminal_formula", _s(self.terminal_formula))
        object.__setattr__(self, "proof_hash", _s(self.proof_hash))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def to_dict(self) -> dict:
        payload = {
            "node_type": "do_proof_verification",
            "verifier_version": self.verifier_version,
            "verification_status": self.verification_status,
            "valid": int(self.valid),
            "authority": self.authority,
            "reason_codes": self.reason_codes,
            "step_count": int(self.step_count),
            "terminal_observational": int(self.terminal_observational),
            "terminal_formula": self.terminal_formula,
            "proof_hash": self.proof_hash,
        }
        if self.failed_step_index is not None:
            payload["failed_step_index"] = int(self.failed_step_index)
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


def _normalise_proof(proof: object) -> DoProof:
    if isinstance(proof, DoProof):
        return proof
    if isinstance(proof, str):
        try:
            payload = json.loads(proof)
        except Exception:
            return DoProof(query=DoExpression(label="unparseable_proof"), status="invalid_proof_payload", reason_codes="UNPARSEABLE_PROOF_JSON")
        if isinstance(payload, Mapping):
            return proof_from_dict(payload)
    if isinstance(proof, Mapping):
        return proof_from_dict(proof)
    return DoProof(query=DoExpression(label="unsupported_proof_payload"), status="invalid_proof_payload", reason_codes="UNSUPPORTED_PROOF_PAYLOAD")


def _admg_node_set(admg: object) -> Optional[set]:
    nodes = getattr(admg, "node_set", None)
    if nodes is None:
        nodes = getattr(admg, "nodes", None)
    if nodes is None:
        return None
    return {_s(v) for v in nodes if _s(v)}


def verify_do_proof(proof: object, *, admg: object = None, require_audit_only: bool = True) -> DoProofVerification:
    """Verify structural integrity of a do-calculus proof trace."""
    p = _normalise_proof(proof)
    reasons = []
    failed_step_index = None

    if not p.query.outcomes:
        reasons.append("MISSING_QUERY_OUTCOME")
    if require_audit_only and _s(p.authority) not in _ALLOWED_AUTHORITIES:
        reasons.append("NON_AUDIT_PROOF_AUTHORITY")

    graph_nodes = _admg_node_set(admg)
    if graph_nodes is not None:
        missing = []
        for expr in [p.query, p.terminal_expression]:
            missing.extend([node for node in _expr_nodes(expr) if node and node not in graph_nodes])
        for step in p.steps:
            missing.extend([node for node in _expr_nodes(step.before) if node and node not in graph_nodes])
            missing.extend([node for node in _expr_nodes(step.after) if node and node not in graph_nodes])
        if missing:
            reasons.append("PROOF_REFERENCES_MISSING_GRAPH_NODE")

    current = p.query
    seen_states = {_expr_key(current)}
    for idx, step in enumerate(p.steps):
        if not isinstance(step, DoRewriteStep):
            reasons.append("MALFORMED_REWRITE_STEP")
            failed_step_index = idx
            break
        if require_audit_only:
            step_authority = _s(step.metadata.get("authority", "audit_only")) if isinstance(step.metadata, Mapping) else "audit_only"
            if step_authority not in _ALLOWED_AUTHORITIES:
                reasons.append("NON_AUDIT_STEP_AUTHORITY")
                failed_step_index = idx
                break
        if not step.applicable:
            reasons.append("INAPPLICABLE_STEP_IN_PROOF")
            failed_step_index = idx
            break
        if _expr_key(step.before) != _expr_key(current):
            reasons.append("BROKEN_STEP_CHAIN")
            failed_step_index = idx
            break
        current = step.after
        key = _expr_key(current)
        if key in seen_states:
            reasons.append("REPEATED_PROOF_STATE")
            failed_step_index = idx
            break
        seen_states.add(key)

    if _expr_key(p.terminal_expression) != _expr_key(current):
        reasons.append("TERMINAL_MISMATCH")
    if "observational" in _s(p.status) and not p.terminal_expression.is_observational:
        reasons.append("OBSERVATIONAL_STATUS_WITH_DO_TERMINAL")
    if dict(p.metadata or {}).get("full_id_claim_allowed") not in (None, 0, "0", False):
        reasons.append("CLAIM_ALLOWED_IN_AUDIT_PROOF")

    proof_dict = p.to_dict()
    proof_hash = hashlib.sha256(json.dumps(proof_dict, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    reason_codes = _dedupe_reasons(reasons) or "PROOF_TRACE_STRUCTURALLY_VALID_AUDIT_ONLY"
    valid = 0 if reasons else 1
    return DoProofVerification(
        valid=valid,
        verification_status="proof_trace_valid_audit_only" if valid else "proof_trace_invalid_audit_only",
        reason_codes=reason_codes,
        step_count=len(p.steps),
        terminal_observational=int(p.terminal_expression.is_observational),
        terminal_formula=p.terminal_expression.to_formula(),
        proof_hash=proof_hash,
        failed_step_index=failed_step_index,
        metadata={
            "require_audit_only": int(bool(require_audit_only)),
            "graph_node_check": int(graph_nodes is not None),
            "seen_states": len(seen_states),
        },
    )


def verify_do_proof_dict(payload: Mapping[str, object], *, admg: object = None, require_audit_only: bool = True) -> DoProofVerification:
    return verify_do_proof(payload, admg=admg, require_audit_only=require_audit_only)


def verify_do_expression_terminal(payload: Mapping[str, object]) -> DoProofVerification:
    query = payload.get("query") if isinstance(payload, Mapping) else None
    terminal = payload.get("terminal") if isinstance(payload, Mapping) else None
    p = DoProof(
        query=expression_from_dict(query if isinstance(query, Mapping) else {}),
        terminal=expression_from_dict(terminal if isinstance(terminal, Mapping) else {}) if isinstance(terminal, Mapping) else None,
        status=_s(payload.get("status")) if isinstance(payload, Mapping) else "audit_only",
        authority=_s(payload.get("authority")) if isinstance(payload, Mapping) else "audit_only",
    )
    return verify_do_proof(p)


__all__ = [
    "DO_PROOF_VERIFIER_VERSION",
    "DO_PROOF_VERIFIER_AUTHORITY",
    "DoProofVerification",
    "verify_do_proof",
    "verify_do_proof_dict",
    "verify_do_expression_terminal",
]
