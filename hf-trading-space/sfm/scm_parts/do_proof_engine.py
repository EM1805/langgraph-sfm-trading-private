from __future__ import annotations

"""Bounded, conservative proof search for Amantia do-calculus.

Step 72 adds the first proof-search layer over the AST/rewrite machinery.  This
is intentionally limited and safety-first:

- finite BFS with max_depth/max_states;
- a rewrite applies only when its ``before`` AST exactly matches the current
  proof state;
- no effect-estimation, no Full-ID authority, no veto authority;
- successful termination means "observational AST reached inside this bounded
  search", not a global completeness claim.
"""

from dataclasses import dataclass, field
import json
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .admg import ADMG
from .do_ast import DoExpression, DoProof, DoRewriteStep, P_do, parse_do_expression
from .do_rewrite import DO_REWRITE_AUTHORITY, rule1_rewrite, rule2_rewrite, rule3_rewrite
from .do_proof_verifier import verify_do_proof
from .graph_criteria import directed_cycle_nodes
from .do_canonical_proofs import canonical_do_proof_template

DO_PROOF_ENGINE_VERSION = "do_proof_engine_bounded_v2_step74_canonical_templates"
DO_PROOF_ENGINE_AUTHORITY = "audit_only"


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


def _state_key(expr: DoExpression) -> str:
    payload = {
        "outcomes": list(expr.outcomes),
        "interventions": list(expr.interventions),
        "observations": list(expr.observations),
        "summations": list(expr.summations),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class DoProofEngineResult:
    proof: DoProof
    status: str
    authority: str = DO_PROOF_ENGINE_AUTHORITY
    proof_engine_version: str = DO_PROOF_ENGINE_VERSION
    explored_states: int = 0
    max_depth: int = 0
    max_states: int = 0
    terminal_observational: int = 0
    reason_codes: str = ""
    rejected_rewrite_count: int = 0
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", _s(self.status) or "search_exhausted_audit_only")
        object.__setattr__(self, "authority", _s(self.authority) or DO_PROOF_ENGINE_AUTHORITY)
        object.__setattr__(self, "proof_engine_version", _s(self.proof_engine_version) or DO_PROOF_ENGINE_VERSION)
        object.__setattr__(self, "reason_codes", _s(self.reason_codes))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def terminal(self) -> DoExpression:
        return self.proof.terminal_expression

    def to_dict(self) -> Dict[str, object]:
        verification = verify_do_proof(self.proof)

        payload: Dict[str, object] = {
            "node_type": "do_proof_engine_result",
            "proof_engine_version": self.proof_engine_version,
            "status": self.status,
            "authority": self.authority,
            "terminal_observational": int(self.terminal_observational),
            "explored_states": int(self.explored_states),
            "max_depth": int(self.max_depth),
            "max_states": int(self.max_states),
            "rejected_rewrite_count": int(self.rejected_rewrite_count),
            "reason_codes": self.reason_codes,
            "terminal_formula": self.terminal.to_formula(),
            "proof_verification": verification.to_dict(),
            "proof_trace_valid": int(verification.valid),
            "proof": self.proof.to_dict(),
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


def _candidate_steps_for_state(admg: ADMG, expr: DoExpression) -> Tuple[List[DoRewriteStep], int]:
    """Generate state-local candidate steps and count blocked candidates.

    Current v1 support is intentionally narrow:
    - Rule 1 can remove an observed Z from P(Y | do(X), Z, W).
    - Rules 2/3 can exchange/remove an extra intervention Z from
      P(Y | do(X), do(Z), W).

    Multi-outcome/multi-treatment formulas are preserved but not over-expanded.
    """
    if not expr.outcomes:
        return [], 0
    y = expr.outcomes[0]
    steps: List[DoRewriteStep] = []
    rejected = 0

    # Rule 1: remove observed variables one at a time when there is a base do(X).
    if len(expr.interventions) == 1 and expr.observations:
        x = expr.interventions[0]
        for z in expr.observations:
            w = [v for v in expr.observations if v != z]
            audit = rule1_rewrite(admg, y=y, x=x, z=z, w=w)
            if audit.applicable and _state_key(audit.step.before) == _state_key(expr):
                steps.append(audit.step)
            else:
                rejected += 1

    # Rules 2/3: exchange/remove additional interventions Z relative to base X.
    if len(expr.interventions) >= 2:
        x = expr.interventions[0]
        for z in expr.interventions[1:]:
            for audit in (
                rule2_rewrite(admg, y=y, x=x, z=z, w=expr.observations),
                rule3_rewrite(admg, y=y, x=x, z=z, w=expr.observations),
            ):
                if audit.applicable and _state_key(audit.step.before) == _state_key(expr):
                    steps.append(audit.step)
                else:
                    rejected += 1

    return steps, rejected


def bounded_do_proof_from_expression(
    admg: ADMG,
    query: DoExpression,
    *,
    max_depth: int = 3,
    max_states: int = 64,
) -> DoProofEngineResult:
    """Run bounded AST rewrite search from an explicit query expression."""
    max_depth = max(0, int(max_depth))
    max_states = max(1, int(max_states))

    if not query.outcomes:
        proof = DoProof(query=query, status="invalid_query", authority=DO_PROOF_ENGINE_AUTHORITY, reason_codes="MISSING_OUTCOME")
        return DoProofEngineResult(proof, "invalid_query", max_depth=max_depth, max_states=max_states, reason_codes="MISSING_OUTCOME")

    missing = [v for v in list(query.outcomes) + list(query.interventions) + list(query.observations) if v not in admg.node_set]
    if missing:
        proof = DoProof(query=query, status="invalid_query", authority=DO_PROOF_ENGINE_AUTHORITY, reason_codes="MISSING_QUERY_NODE")
        return DoProofEngineResult(proof, "invalid_query", max_depth=max_depth, max_states=max_states, reason_codes="MISSING_QUERY_NODE", metadata={"missing_nodes": missing})

    cycles = directed_cycle_nodes(admg)
    if cycles:
        proof = DoProof(query=query, status="blocked_directed_cycle", authority=DO_PROOF_ENGINE_AUTHORITY, reason_codes="DIRECTED_CYCLE_NOT_ADMG_DAG")
        return DoProofEngineResult(proof, "blocked_directed_cycle", max_depth=max_depth, max_states=max_states, reason_codes="DIRECTED_CYCLE_NOT_ADMG_DAG", metadata={"cycle_nodes": cycles})

    if query.is_observational:
        proof = DoProof(
            query=query,
            status="proof_found_observational_audit_only",
            authority=DO_PROOF_ENGINE_AUTHORITY,
            terminal=query,
            reason_codes="QUERY_ALREADY_OBSERVATIONAL",
            metadata={"proof_engine_version": DO_PROOF_ENGINE_VERSION},
        )
        return DoProofEngineResult(
            proof,
            "proof_found_observational_audit_only",
            explored_states=1,
            max_depth=max_depth,
            max_states=max_states,
            terminal_observational=1,
            reason_codes="QUERY_ALREADY_OBSERVATIONAL",
        )

    canonical = canonical_do_proof_template(admg, query)
    if canonical is not None:
        proof = DoProof(
            query=canonical.proof.query,
            steps=canonical.proof.steps,
            status=canonical.status,
            authority=DO_PROOF_ENGINE_AUTHORITY,
            terminal=canonical.proof.terminal_expression,
            reason_codes=canonical.reason_codes,
            metadata={
                "proof_engine_version": DO_PROOF_ENGINE_VERSION,
                "canonical_template": canonical.to_metadata(),
                "full_id_claim_allowed": 0,
            },
        )
        return DoProofEngineResult(
            proof,
            canonical.status,
            explored_states=1,
            max_depth=max_depth,
            max_states=max_states,
            terminal_observational=int(canonical.terminal_observational),
            reason_codes=canonical.reason_codes,
            rejected_rewrite_count=0,
            metadata={"canonical_template": canonical.to_metadata()},
        )

    queue: List[Tuple[DoExpression, Tuple[DoRewriteStep, ...], int]] = [(query, tuple(), 0)]
    seen = {_state_key(query)}
    explored = 0
    rejected_total = 0
    best_expr = query
    best_steps: Tuple[DoRewriteStep, ...] = tuple()
    stopped_by_limit = False

    while queue and explored < max_states:
        expr, steps_so_far, depth = queue.pop(0)
        explored += 1
        best_expr, best_steps = expr, steps_so_far
        if expr.is_observational:
            proof = DoProof(
                query=query,
                steps=steps_so_far,
                status="proof_found_observational_audit_only",
                authority=DO_PROOF_ENGINE_AUTHORITY,
                terminal=expr,
                reason_codes="BOUNDED_SEARCH_REACHED_OBSERVATIONAL_AST",
                metadata={"proof_engine_version": DO_PROOF_ENGINE_VERSION},
            )
            return DoProofEngineResult(
                proof,
                "proof_found_observational_audit_only",
                explored_states=explored,
                max_depth=max_depth,
                max_states=max_states,
                terminal_observational=1,
                reason_codes="BOUNDED_SEARCH_REACHED_OBSERVATIONAL_AST",
                rejected_rewrite_count=rejected_total,
            )
        if depth >= max_depth:
            continue

        next_steps, rejected = _candidate_steps_for_state(admg, expr)
        rejected_total += rejected
        for step in next_steps:
            key = _state_key(step.after)
            if key in seen:
                rejected_total += 1
                continue
            seen.add(key)
            queue.append((step.after, steps_so_far + (step,), depth + 1))

    if queue:
        stopped_by_limit = True

    if best_steps:
        status = "bounded_rewrite_progress_audit_only"
        reason = "BOUNDED_REWRITE_PROGRESS_NO_OBSERVATIONAL_TERMINAL"
    else:
        status = "search_exhausted_audit_only"
        reason = "NO_APPLICABLE_STATE_MATCHED_REWRITE"
    if stopped_by_limit or explored >= max_states:
        reason = reason + "|SEARCH_LIMIT_REACHED"

    proof = DoProof(
        query=query,
        steps=best_steps,
        status=status,
        authority=DO_PROOF_ENGINE_AUTHORITY,
        terminal=best_expr,
        reason_codes=reason,
        metadata={
            "proof_engine_version": DO_PROOF_ENGINE_VERSION,
            "search_complete": int(not stopped_by_limit and explored < max_states),
            "seen_states": len(seen),
        },
    )
    return DoProofEngineResult(
        proof,
        status,
        explored_states=explored,
        max_depth=max_depth,
        max_states=max_states,
        terminal_observational=int(best_expr.is_observational),
        reason_codes=reason,
        rejected_rewrite_count=rejected_total,
    )


def bounded_do_proof(
    admg: ADMG,
    treatment: object,
    outcome: object,
    *,
    conditioned_on: Optional[Sequence[object]] = None,
    max_depth: int = 3,
    max_states: int = 64,
) -> DoProofEngineResult:
    """Convenience wrapper for the standard query P(Y | do(X), W)."""
    x, y = _s(treatment), _s(outcome)
    query = P_do(
        [y],
        interventions=[x],
        observations=_dedupe(conditioned_on or []),
        label="bounded_do_proof_query",
        metadata={"source": "bounded_do_proof", "authority": DO_PROOF_ENGINE_AUTHORITY},
    )
    return bounded_do_proof_from_expression(admg, query, max_depth=max_depth, max_states=max_states)


def bounded_do_proof_from_formula(
    admg: ADMG,
    formula: object,
    *,
    max_depth: int = 3,
    max_states: int = 64,
) -> DoProofEngineResult:
    """Parse a compact formula and run bounded proof search."""
    return bounded_do_proof_from_expression(admg, parse_do_expression(formula), max_depth=max_depth, max_states=max_states)


__all__ = [
    "DO_PROOF_ENGINE_VERSION",
    "DO_PROOF_ENGINE_AUTHORITY",
    "DoProofEngineResult",
    "bounded_do_proof",
    "bounded_do_proof_from_expression",
    "bounded_do_proof_from_formula",
]
