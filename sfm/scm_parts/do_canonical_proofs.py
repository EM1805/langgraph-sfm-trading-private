from __future__ import annotations

"""Canonical Pearl-template proof traces for Amantia do-calculus.

Step 74 adds a conservative template layer for common textbook cases.  These
traces are deliberately *audit only*: they make the algebraic route visible for
backdoor, frontdoor, and zero-effect patterns, but they do not grant arbitrary
Full-ID, estimator, or veto authority.
"""

from dataclasses import dataclass, field
from itertools import combinations
import json
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .admg import ADMG
from .do_ast import DoExpression, DoProof, DoRewriteStep, P_do
from .graph_criteria import directed_path_exists, directed_paths
from .id_routes import BackdoorDiagnostic, FrontdoorDiagnostic, backdoor_diagnostic, frontdoor_diagnostic

DO_CANONICAL_PROOF_VERSION = "do_canonical_proof_templates_v1_step74"
DO_CANONICAL_PROOF_AUTHORITY = "audit_only"


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


def _join(values: Sequence[str], sep: str = ",") -> str:
    return sep.join(_dedupe(values))


def _bar(values: Sequence[str]) -> str:
    return "|".join(_dedupe(values))


def _p(vars_: Sequence[str], given: Sequence[str] = ()) -> str:
    v = _join(vars_)
    g = _join(given)
    return f"P({v}|{g})" if g else f"P({v})"


def _backdoor_formula(x: str, y: str, z: Sequence[str]) -> str:
    zz = _dedupe(z)
    if zz:
        return f"sum_{{{_join(zz)}}} P({y}|{x},{_join(zz)}) * P({_join(zz)})"
    return f"P({y}|{x})"


def _frontdoor_formula(x: str, y: str, z: Sequence[str]) -> str:
    zz = _dedupe(z)
    if not zz:
        return ""
    z_text = _join(zz)
    return f"sum_{{{z_text}}} P({z_text}|{x}) * sum_{{{x}'}} P({y}|{x}',{z_text}) * P({x}')"


@dataclass(frozen=True)
class CanonicalDoProofTemplate:
    """Audit-only canonical proof template result."""

    proof: DoProof
    status: str
    proof_family: str
    observational_formula: str = ""
    adjustment_set: Tuple[str, ...] = tuple()
    mediators: Tuple[str, ...] = tuple()
    authority: str = DO_CANONICAL_PROOF_AUTHORITY
    version: str = DO_CANONICAL_PROOF_VERSION
    reason_codes: str = ""
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", _s(self.status) or "template_not_applicable")
        object.__setattr__(self, "proof_family", _s(self.proof_family))
        object.__setattr__(self, "observational_formula", _s(self.observational_formula))
        object.__setattr__(self, "adjustment_set", tuple(_dedupe(self.adjustment_set)))
        object.__setattr__(self, "mediators", tuple(_dedupe(self.mediators)))
        object.__setattr__(self, "authority", _s(self.authority) or DO_CANONICAL_PROOF_AUTHORITY)
        object.__setattr__(self, "version", _s(self.version) or DO_CANONICAL_PROOF_VERSION)
        object.__setattr__(self, "reason_codes", _s(self.reason_codes))
        object.__setattr__(self, "diagnostics", dict(self.diagnostics or {}))

    @property
    def terminal_observational(self) -> int:
        return int(self.proof.terminal_expression.is_observational)

    def to_metadata(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "source": "do_canonical_proofs",
            "template_version": self.version,
            "template_authority": self.authority,
            "proof_family": self.proof_family,
            "observational_formula": self.observational_formula,
            "adjustment_set": list(self.adjustment_set),
            "mediators": list(self.mediators),
            "full_id_claim_allowed": 0,
            "reason_codes": self.reason_codes,
        }
        if self.diagnostics:
            payload["diagnostics"] = dict(self.diagnostics)
        return payload

    def to_dict(self) -> Dict[str, object]:
        return {
            "node_type": "canonical_do_proof_template",
            "status": self.status,
            "authority": self.authority,
            "version": self.version,
            "proof_family": self.proof_family,
            "terminal_observational": self.terminal_observational,
            "observational_formula": self.observational_formula,
            "adjustment_set": list(self.adjustment_set),
            "mediators": list(self.mediators),
            "reason_codes": self.reason_codes,
            "proof": self.proof.to_dict(),
            "diagnostics": dict(self.diagnostics),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


def _step(rule: str, before: DoExpression, after: DoExpression, *, premise: str, reason: str, metadata: Optional[Mapping[str, object]] = None) -> DoRewriteStep:
    base = {
        "source": "do_canonical_proof_template",
        "template_version": DO_CANONICAL_PROOF_VERSION,
        "authority": DO_CANONICAL_PROOF_AUTHORITY,
        "full_id_claim_allowed": 0,
    }
    if metadata:
        base.update(dict(metadata))
    return DoRewriteStep(
        rule=rule,
        before=before,
        after=after,
        applicable=True,
        premise=premise,
        graph_variant="canonical_template_graph_audit",
        reason_codes=reason,
        metadata=base,
    )


def _standard_singleton_query(query: DoExpression) -> Optional[Tuple[str, str]]:
    if len(query.outcomes) != 1 or len(query.interventions) != 1 or query.observations:
        return None
    return query.interventions[0], query.outcomes[0]


def _candidate_backdoor_set(admg: ADMG, x: str, y: str, *, max_set_size: int = 3) -> Tuple[List[str], Optional[BackdoorDiagnostic]]:
    descendants_x = admg.descendants([x]) - {x}
    candidates = sorted(n for n in admg.node_set if n not in {x, y} and n not in descendants_x)
    # Prefer smaller sets, and let the diagnostic decide whether empty is valid.
    for k in range(0, min(int(max_set_size), len(candidates)) + 1):
        for combo in combinations(candidates, k):
            diag = backdoor_diagnostic(admg, x, y, list(combo))
            if diag.backdoor_ok:
                return list(combo), diag
    return [], None


def _candidate_frontdoor_set(admg: ADMG, x: str, y: str) -> Tuple[List[str], Optional[FrontdoorDiagnostic]]:
    paths = directed_paths(admg, x, y)
    if not paths:
        return [], None
    on_paths: List[str] = []
    for path in paths:
        for node in path[1:-1]:
            if node not in on_paths:
                on_paths.append(node)
    if not on_paths:
        return [], None
    # Try the full mediator cover first, then singleton mediators for simple cases.
    candidates: List[List[str]] = [_dedupe(on_paths)] + [[m] for m in on_paths]
    seen = set()
    for meds in candidates:
        key = tuple(meds)
        if key in seen:
            continue
        seen.add(key)
        diag = frontdoor_diagnostic(admg, x, y, meds)
        if diag.frontdoor_ok:
            return meds, diag
    return [], None


def zero_effect_template(admg: ADMG, query: DoExpression) -> Optional[CanonicalDoProofTemplate]:
    pair = _standard_singleton_query(query)
    if pair is None:
        return None
    x, y = pair
    if directed_path_exists(admg, x, y):
        return None
    terminal = P_do([y], label="canonical_zero_effect_terminal", metadata={"observational_formula": f"P({y})"})
    step = _step(
        "canonical_zero_effect_do_deletion_template",
        query,
        terminal,
        premise="no_directed_path_from_treatment_to_outcome",
        reason="CANONICAL_ZERO_EFFECT_TEMPLATE_NO_DIRECTED_PATH_AUDIT_ONLY",
    )
    proof = DoProof(
        query=query,
        steps=(step,),
        status="canonical_zero_effect_template_proof_audit_only",
        authority=DO_CANONICAL_PROOF_AUTHORITY,
        terminal=terminal,
        reason_codes="CANONICAL_ZERO_EFFECT_TEMPLATE_NO_DIRECTED_PATH_AUDIT_ONLY",
        metadata={
            "proof_family": "canonical_zero_effect",
            "observational_formula": f"P({y})",
            "template_version": DO_CANONICAL_PROOF_VERSION,
            "full_id_claim_allowed": 0,
        },
    )
    return CanonicalDoProofTemplate(
        proof=proof,
        status="canonical_zero_effect_template_proof_audit_only",
        proof_family="canonical_zero_effect",
        observational_formula=f"P({y})",
        reason_codes="CANONICAL_ZERO_EFFECT_TEMPLATE_NO_DIRECTED_PATH_AUDIT_ONLY",
    )


def backdoor_template(admg: ADMG, query: DoExpression, *, max_set_size: int = 3) -> Optional[CanonicalDoProofTemplate]:
    pair = _standard_singleton_query(query)
    if pair is None:
        return None
    x, y = pair
    z, diag = _candidate_backdoor_set(admg, x, y, max_set_size=max_set_size)
    if diag is None or not diag.backdoor_ok:
        return None
    formula = _backdoor_formula(x, y, z)
    terminal = P_do(
        [y],
        observations=[x] + z,
        summations=z,
        label="canonical_backdoor_terminal_observational_formula",
        metadata={
            "observational_formula": formula,
            "adjustment_set": z,
            "template_version": DO_CANONICAL_PROOF_VERSION,
            "note": "DoExpression carries the main conditional term; product P(Z) is recorded in observational_formula metadata.",
        },
    )
    expand = P_do(
        [y],
        interventions=[x],
        observations=z,
        summations=z,
        label="canonical_backdoor_total_probability_shell",
        metadata={"observational_formula_component": f"sum_{{{_join(z)}}} P({y}|do({x}),{_join(z)}) * P({_join(z)}|do({x}))" if z else f"P({y}|do({x}))"},
    )
    steps: List[DoRewriteStep] = []
    if z:
        steps.append(_step(
            "canonical_law_total_probability_expand_adjustment",
            query,
            expand,
            premise="standardize_over_valid_backdoor_adjustment_set",
            reason="BACKDOOR_TOTAL_PROBABILITY_EXPANSION_AUDIT_ONLY",
            metadata={"adjustment_set": z, "algebraic_factor": f"P({_join(z)}|do({x}))"},
        ))
        steps.append(_step(
            "rule2_action_observation_exchange_backdoor_template",
            expand,
            terminal,
            premise="backdoor_diagnostic_valid_adjustment_set",
            reason="BACKDOOR_RULE2_EXCHANGE_AND_RULE3_DELETE_ADJUSTMENT_DISTRIBUTION_AUDIT_ONLY",
            metadata={"adjustment_set": z, "observational_formula": formula},
        ))
    else:
        steps.append(_step(
            "rule2_empty_backdoor_exchange_template",
            query,
            terminal,
            premise="empty_backdoor_set_valid_no_open_backdoor_paths",
            reason="EMPTY_BACKDOOR_EXCHANGE_TO_OBSERVATIONAL_CONDITIONAL_AUDIT_ONLY",
            metadata={"adjustment_set": [], "observational_formula": formula},
        ))
    proof = DoProof(
        query=query,
        steps=tuple(steps),
        status="canonical_backdoor_template_proof_audit_only",
        authority=DO_CANONICAL_PROOF_AUTHORITY,
        terminal=terminal,
        reason_codes="CANONICAL_BACKDOOR_TEMPLATE_DSEP_VERIFIED_AUDIT_ONLY",
        metadata={
            "proof_family": "canonical_backdoor",
            "observational_formula": formula,
            "adjustment_set": z,
            "backdoor_status": diag.backdoor_status,
            "template_version": DO_CANONICAL_PROOF_VERSION,
            "full_id_claim_allowed": 0,
        },
    )
    return CanonicalDoProofTemplate(
        proof=proof,
        status="canonical_backdoor_template_proof_audit_only",
        proof_family="canonical_backdoor",
        observational_formula=formula,
        adjustment_set=tuple(z),
        reason_codes="CANONICAL_BACKDOOR_TEMPLATE_DSEP_VERIFIED_AUDIT_ONLY",
        diagnostics={"backdoor": diag.to_dict()},
    )


def frontdoor_template(admg: ADMG, query: DoExpression) -> Optional[CanonicalDoProofTemplate]:
    pair = _standard_singleton_query(query)
    if pair is None:
        return None
    x, y = pair
    meds, diag = _candidate_frontdoor_set(admg, x, y)
    if diag is None or not diag.frontdoor_ok:
        return None
    formula = _frontdoor_formula(x, y, meds)
    terminal = P_do(
        [y],
        summations=meds,
        label="canonical_frontdoor_terminal_observational_formula",
        metadata={
            "observational_formula": formula,
            "mediators": meds,
            "template_version": DO_CANONICAL_PROOF_VERSION,
            "note": "DoExpression cannot encode nested products yet; full formula is recorded in observational_formula metadata.",
        },
    )
    shell = P_do(
        [y],
        interventions=[x],
        observations=meds,
        summations=meds,
        label="canonical_frontdoor_mediator_expansion_shell",
        metadata={"mediators": meds},
    )
    steps = (
        _step(
            "canonical_frontdoor_total_probability_expand_mediators",
            query,
            shell,
            premise="frontdoor_condition_1_all_directed_paths_intercepted",
            reason="FRONTDOOR_TOTAL_PROBABILITY_EXPANSION_AUDIT_ONLY",
            metadata={"mediators": meds},
        ),
        _step(
            "rule2_rule3_frontdoor_mediator_formula_template",
            shell,
            terminal,
            premise="frontdoor_conditions_2_and_3_dsep_verified",
            reason="FRONTDOOR_RULE2_RULE3_OBSERVATIONAL_FORMULA_AUDIT_ONLY",
            metadata={"mediators": meds, "observational_formula": formula},
        ),
    )
    proof = DoProof(
        query=query,
        steps=steps,
        status="canonical_frontdoor_template_proof_audit_only",
        authority=DO_CANONICAL_PROOF_AUTHORITY,
        terminal=terminal,
        reason_codes="CANONICAL_FRONTDOOR_TEMPLATE_DSEP_VERIFIED_AUDIT_ONLY",
        metadata={
            "proof_family": "canonical_frontdoor",
            "observational_formula": formula,
            "mediators": meds,
            "frontdoor_status": diag.frontdoor_status,
            "template_version": DO_CANONICAL_PROOF_VERSION,
            "full_id_claim_allowed": 0,
        },
    )
    return CanonicalDoProofTemplate(
        proof=proof,
        status="canonical_frontdoor_template_proof_audit_only",
        proof_family="canonical_frontdoor",
        observational_formula=formula,
        mediators=tuple(meds),
        reason_codes="CANONICAL_FRONTDOOR_TEMPLATE_DSEP_VERIFIED_AUDIT_ONLY",
        diagnostics={"frontdoor": diag.to_dict()},
    )


def canonical_do_proof_template(admg: ADMG, query: DoExpression, *, max_backdoor_set_size: int = 3) -> Optional[CanonicalDoProofTemplate]:
    """Return the first applicable audit-only canonical template.

    Order is important and conservative:
    1. zero-effect/no directed path;
    2. valid backdoor adjustment, including empty adjustment;
    3. valid limited frontdoor.
    """
    for maker in (
        zero_effect_template,
        lambda g, q: backdoor_template(g, q, max_set_size=max_backdoor_set_size),
        frontdoor_template,
    ):
        result = maker(admg, query)
        if result is not None:
            return result
    return None


__all__ = [
    "DO_CANONICAL_PROOF_VERSION",
    "DO_CANONICAL_PROOF_AUTHORITY",
    "CanonicalDoProofTemplate",
    "canonical_do_proof_template",
    "zero_effect_template",
    "backdoor_template",
    "frontdoor_template",
]
