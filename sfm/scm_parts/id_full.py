from __future__ import annotations

"""Full-ID public scaffold and canonical control-flow overlay for Amantia.

Step 68 keeps the canonical control-flow shell, owned canonical formula authority, operational carried-Q AST, a first conservative IDC layer, canonical ID-4/ID-7 special-family formulas, and standardized failure certificates. It adds conservative IDC condition pruning for graph-disconnected conditions. It still does **not** claim arbitrary Full ID/IDC: formula authority remains gated until arbitrary ID-7, full hedge coverage, and a broad IDC matrix are complete.

What this module now owns:
- set-valued public entrypoints ``full_id`` and ``full_id_from_scm_graph``;
- input validation and ADMG directed-cycle rejection;
- a machine-readable canonical ID-1 -> ID-7 control trace;
- stable formula/AST/proof payloads from the audited delegate runtime;
- a permanent ``full_id_claim_allowed`` gate that stays 0 until the complete
  arbitrary-ID matrix and IDC matrix are green.
- carried-Q context formula AST extraction and operational propagation for ID-7 auditability.
"""

from dataclasses import asdict, dataclass, field
import json
from typing import Dict, Iterable, List, Mapping, MutableSequence, Optional, Sequence, Set, Tuple

from .admg import ADMG, admg_from_scm_graph
from .graph_criteria import directed_cycle_nodes, topological_order
from .id_algorithm_common import _dedupe, _format_components, _joint_symbol, _s
from .id_ast import ast_from_dict, Fraction, Placeholder, Sum
from .id_ast_normalizer import ID_AST_NORMALIZER_VERSION, normalize_formula_ast
from .id_canonical_formula import (
    ID_CANONICAL_FORMULA_AUTHORITY,
    ID_CANONICAL_FORMULA_LEVEL,
    ID_CANONICAL_FORMULA_VERSION,
    canonical_id_formula_diagnostic,
)
from .id_failure_certificate import (
    ID_FAILURE_CERTIFICATE_AUTHORITY,
    ID_FAILURE_CERTIFICATE_LEVEL,
    ID_FAILURE_CERTIFICATE_VERSION,
    failure_certificate_for_query,
    rejection_certificate,
)
from .idc_rules import (
    IDC_PRUNING_LEVEL,
    IDC_PRUNING_VERSION,
    idc_pruning_diagnostic,
)
from .id_recursive_expression import (
    RecursiveIDExpressionDiagnostic,
    recursive_id_set_expression_diagnostic,
)
from .id_status import id_capability_flags
from .do_ast import P_do
from .do_proof_engine import (
    DO_PROOF_ENGINE_AUTHORITY,
    DO_PROOF_ENGINE_VERSION,
    bounded_do_proof_from_expression,
)

ID_FULL_INTERFACE_VERSION = "id_full_interface_v2_step74"
ID_FULL_IMPLEMENTATION_LEVEL = (
    "canonical_id_1_to_7_control_flow_overlay_with_owned_formula_authority_for_ID_1_ID_2_ID_4_ID_6_gated_ID_7_frontdoor_contextual_chain_parallel_frontdoor_failure_certificate_authority_step68_idc_pruning_step62_do_proof_metadata_step73_canonical_templates_step74_no_full_id_claim"
)

_CANONICAL_ID_RULES: Dict[str, str] = {
    "ID-1": "No intervention / marginalization base case",
    "ID-2": "Ancestor reduction",
    "ID-3": "W-step promotion of irrelevant variables into the intervention set",
    "ID-4": "C-component / district decomposition in G[V\\X]",
    "ID-5": "FAIL with formal hedge certificate or non-identifiable district witness",
    "ID-6": "Q-factor / observed-DAG truncated factorization",
    "ID-7": "Recursive subdistrict Q-input identification",
}


# ---------------------------------------------------------------------------
# Small normalization / JSON helpers
# ---------------------------------------------------------------------------


def _join(values: Iterable[object]) -> str:
    return "|".join(_dedupe(values or []))


def _normalise_nodes(values: Sequence[object] | object) -> Tuple[str, ...]:
    if isinstance(values, str):
        return tuple(_dedupe([values]))
    return tuple(_dedupe(values or []))


def _json(payload: Mapping[str, object]) -> str:
    return json.dumps(dict(payload), sort_keys=True, separators=(",", ":"))


def _list(values: Iterable[object]) -> List[str]:
    return list(_dedupe(values or []))


def _sorted_set(values: Iterable[object]) -> List[str]:
    return sorted(_dedupe(values or []))


def _is_same_component(a: Sequence[str], b: Sequence[str]) -> bool:
    return set(a) == set(b)


def _remove_incoming_to(admg: ADMG, nodes: Sequence[str]) -> ADMG:
    blocked = set(_dedupe(nodes))
    return ADMG(
        nodes=admg.nodes,
        directed_edges=tuple((a, b) for a, b in admg.directed_edges if b not in blocked),
        bidirected_edges=admg.bidirected_edges,
    )


def _load_expression_payload(expr: RecursiveIDExpressionDiagnostic) -> Dict[str, object]:
    if not expr.expression_json:
        return {}
    try:
        payload = json.loads(expr.expression_json)
    except Exception as exc:  # pragma: no cover - defensive audit fallback
        return {"payload_parse_error": f"{type(exc).__name__}:{exc}"}
    return payload if isinstance(payload, dict) else {"payload_non_mapping": True}


def _extract_formula_ast_json(expr: RecursiveIDExpressionDiagnostic) -> str:
    payload = _load_expression_payload(expr)
    ast = payload.get("formula_ast") if isinstance(payload, Mapping) else None
    if not isinstance(ast, Mapping):
        return ""
    return _json(ast)


def _extract_carried_q_context(expr: RecursiveIDExpressionDiagnostic) -> Optional[Mapping[str, object]]:
    """Find the first Step-52 carried-Q context in the delegate payload."""
    payload = _load_expression_payload(expr)

    def walk(obj: object) -> Optional[Mapping[str, object]]:
        if isinstance(obj, Mapping):
            ctx = obj.get("carried_q_context")
            if isinstance(ctx, Mapping):
                return ctx
            for value in obj.values():
                found = walk(value)
                if found is not None:
                    return found
        elif isinstance(obj, list):
            for value in obj:
                found = walk(value)
                if found is not None:
                    return found
        return None

    return walk(payload)


def _extract_carried_q_context_json(expr: RecursiveIDExpressionDiagnostic) -> str:
    found = _extract_carried_q_context(expr)
    return _json(found) if isinstance(found, Mapping) else ""


def _extract_carried_q_formula_ast_json(expr: RecursiveIDExpressionDiagnostic) -> str:
    found = _extract_carried_q_context(expr)
    if not isinstance(found, Mapping):
        return ""
    ast = found.get("formula_ast")
    return _json(ast) if isinstance(ast, Mapping) else ""


def _extract_operational_carried_q_ast_enabled(expr: RecursiveIDExpressionDiagnostic) -> int:
    payload = _load_expression_payload(expr)

    def walk(obj: object) -> bool:
        if isinstance(obj, Mapping):
            if obj.get("step53_operational_carried_q_ast_enabled") == 1:
                return True
            ast = obj.get("formula_ast")
            if isinstance(ast, Mapping):
                meta = ast.get("metadata")
                if isinstance(meta, Mapping) and meta.get("operational_carried_q_ast_step53") == 1:
                    return True
            return any(walk(v) for v in obj.values())
        if isinstance(obj, list):
            return any(walk(v) for v in obj)
        return False

    return int(walk(payload))


def _extract_canonical_rules(trace_json: str) -> str:
    """Best-effort rule extraction from the delegate recursive trace.

    Step 51 keeps this for backwards compatibility, but the new authoritative
    control overlay is exposed separately in ``canonical_control_json``.
    """
    if not trace_json:
        return ""
    try:
        payload = json.loads(trace_json)
    except Exception:
        return ""
    steps = payload.get("trace", []) if isinstance(payload, Mapping) else []
    if not isinstance(steps, list):
        return ""
    mapping = {
        "id_base_no_intervention": "ID-1",
        "no_intervention": "ID-1",
        "ancestor_reduction": "ID-2",
        "irrelevant_after_intervention_w_step": "ID-3",
        "w_step_irrelevant_interventions": "ID-3",
        "district_decomposition_g_v_minus_x": "ID-4",
        "formal_hedge_certificate": "ID-5",
        "q_factor_full_district": "ID-6",
        "observed_dag_truncated_factorization": "ID-6",
        "graphical_zero_effect": "ID-6",
        "q_input_subdistrict_recursion": "ID-7",
        "q_input_general_recursion_step44": "ID-7",
        "q_input_general_carried_q_recursion_step51": "ID-7",
    }
    out: List[str] = []
    seen: Set[str] = set()
    for step in steps:
        if not isinstance(step, Mapping):
            continue
        status = _s(step.get("status"))
        if status in {"not_needed", "skipped", "entered"}:
            continue
        rule = mapping.get(_s(step.get("step")))
        if rule and rule not in seen:
            seen.add(rule)
            out.append(rule)
    return "|".join(out)


def _estimand_formula(treatments: Sequence[str], outcomes: Sequence[str], rhs: str) -> str:
    y = _joint_symbol(outcomes) or ",".join(outcomes)
    x = _joint_symbol(treatments) or ",".join(treatments)
    if not rhs:
        return ""
    if rhs.startswith("P_{"):
        return rhs
    return f"P_{{do({x})}}({y}) = {rhs}"


def _do_proof_metadata_for_query(
    admg: ADMG,
    treatments: Sequence[str],
    outcomes: Sequence[str],
    *,
    max_depth: int,
) -> Dict[str, object]:
    """Run Step-72 bounded do-calculus proof as non-authoritative metadata.

    The result is attached to Full-ID diagnostics only for auditability.  It
    never changes ``identified``, ``formula``, ``primary_formula_authority``,
    or ``full_id_claim_allowed``.
    """
    query = P_do(
        outcomes,
        interventions=treatments,
        label="full_id_step74_do_proof_metadata_query",
        metadata={
            "source": "id_full_step74_metadata",
            "authority": DO_PROOF_ENGINE_AUTHORITY,
            "full_id_claim_allowed": 0,
        },
    )
    depth = max(0, min(int(max_depth), 3))
    result = bounded_do_proof_from_expression(admg, query, max_depth=depth, max_states=64)
    payload = result.to_dict()
    proof_payload = payload.get("proof") if isinstance(payload, Mapping) else {}
    terminal_ast = proof_payload.get("terminal", {}) if isinstance(proof_payload, Mapping) else {}
    return {
        "do_proof_status": result.status,
        "do_proof_json": result.to_json(),
        "do_proof_terminal_formula": _s(payload.get("terminal_formula")),
        "do_proof_terminal_ast_json": _json(terminal_ast) if isinstance(terminal_ast, Mapping) else "",
        "do_proof_authority": result.authority,
        "do_proof_engine_version": result.proof_engine_version,
        "do_proof_terminal_observational": int(result.terminal_observational),
        "do_proof_explored_states": int(result.explored_states),
        "do_proof_reason_codes": result.reason_codes,
    }


def _empty_do_proof_metadata(status: str = "not_run_invalid_or_blocked_query") -> Dict[str, object]:
    return {
        "do_proof_status": status,
        "do_proof_json": "",
        "do_proof_terminal_formula": "",
        "do_proof_terminal_ast_json": "",
        "do_proof_authority": DO_PROOF_ENGINE_AUTHORITY,
        "do_proof_engine_version": DO_PROOF_ENGINE_VERSION,
        "do_proof_terminal_observational": 0,
        "do_proof_explored_states": 0,
        "do_proof_reason_codes": "DO_PROOF_METADATA_NOT_RUN",
    }


# ---------------------------------------------------------------------------
# Canonical ID-1 -> ID-7 control-flow overlay
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CanonicalIDStep:
    depth: int
    rule_id: str
    rule_name: str
    status: str
    y: str
    x: str
    v: str
    q_input_scope: str = ""
    details: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["details"] = dict(self.details or {})
        return payload


@dataclass(frozen=True)
class CanonicalControlDiagnostic:
    status: str
    terminal_rule: str
    terminal_status: str
    applied_rules: str
    n_steps: int
    max_depth_hit: int
    steps: Tuple[CanonicalIDStep, ...]

    def to_dict(self) -> Dict[str, object]:
        return {
            "control_version": "canonical_id_control_flow_v1_step56",
            "status": self.status,
            "terminal_rule": self.terminal_rule,
            "terminal_status": self.terminal_status,
            "applied_rules": self.applied_rules.split("|") if self.applied_rules else [],
            "n_steps": self.n_steps,
            "max_depth_hit": self.max_depth_hit,
            "rule_names": _CANONICAL_ID_RULES,
            "steps": [s.to_dict() for s in self.steps],
            "full_id_claim_allowed": 0,
            "claim_note": "control-flow overlay only; formula authority still delegated until general carried-Q and IDC are complete",
        }


def _step(
    steps: MutableSequence[CanonicalIDStep],
    *,
    depth: int,
    rule_id: str,
    status: str,
    admg: ADMG,
    y: Sequence[str],
    x: Sequence[str],
    q_input_scope: Sequence[str] = (),
    **details: object,
) -> None:
    steps.append(
        CanonicalIDStep(
            depth=depth,
            rule_id=rule_id,
            rule_name=_CANONICAL_ID_RULES.get(rule_id, rule_id),
            status=status,
            y=_join(y),
            x=_join(x),
            v=_join(sorted(admg.node_set)),
            q_input_scope=_join(q_input_scope),
            details={k: v for k, v in details.items() if v not in (None, "", [], {}, ())},
        )
    )


def _control_key(admg: ADMG, y: Sequence[str], x: Sequence[str], q_input_scope: Sequence[str]) -> Tuple[Tuple[str, ...], Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]]:
    return (tuple(sorted(admg.node_set)), tuple(_sorted_set(y)), tuple(_sorted_set(x)), tuple(_sorted_set(q_input_scope)))


def _canonical_id_control_walk(
    admg: ADMG,
    y_set: Sequence[str],
    x_set: Sequence[str],
    *,
    depth: int,
    max_depth: int,
    steps: MutableSequence[CanonicalIDStep],
    seen: Set[Tuple[Tuple[str, ...], Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]]],
    q_input_scope: Sequence[str] = (),
) -> Tuple[str, str, str, int]:
    """Build a conservative canonical ID control trace.

    This mirrors the order of the Shpitser/Pearl ID cases.  It intentionally
    does not construct formulas; formula construction is still delegated.  The
    point of this walker is to make the eventual full replacement path explicit
    and testable without changing downstream callers again later.
    """
    y = _sorted_set([n for n in y_set if n in admg.node_set])
    x = _sorted_set([n for n in x_set if n in admg.node_set and n not in set(y)])
    v = sorted(admg.node_set)
    q_scope = _sorted_set(q_input_scope)
    max_depth_hit = depth

    if depth > max_depth:
        _step(steps, depth=depth, rule_id="ID-5", status="blocked_max_depth", admg=admg, y=y, x=x, q_input_scope=q_scope, max_depth=max_depth)
        return "blocked", "ID-5", "blocked_max_depth", max_depth_hit

    key = _control_key(admg, y, x, q_scope)
    if key in seen:
        _step(steps, depth=depth, rule_id="ID-5", status="blocked_revisited_state", admg=admg, y=y, x=x, q_input_scope=q_scope)
        return "blocked", "ID-5", "blocked_revisited_state", max_depth_hit
    seen.add(key)

    if not y:
        _step(steps, depth=depth, rule_id="ID-5", status="blocked_missing_outcome", admg=admg, y=y, x=x, q_input_scope=q_scope)
        return "blocked", "ID-5", "blocked_missing_outcome", max_depth_hit

    cycles = directed_cycle_nodes(admg)
    if cycles:
        _step(steps, depth=depth, rule_id="ID-5", status="blocked_directed_cycle", admg=admg, y=y, x=x, q_input_scope=q_scope, directed_cycle_nodes="|".join(cycles))
        return "blocked", "ID-5", "blocked_directed_cycle", max_depth_hit

    # ID-1: no intervention.
    if not x:
        _step(steps, depth=depth, rule_id="ID-1", status="terminal_no_intervention", admg=admg, y=y, x=x, q_input_scope=q_scope)
        return "identified_or_delegated", "ID-1", "terminal_no_intervention", max_depth_hit

    # ID-2: ancestor reduction.
    ancestors_y = sorted(admg.ancestors(y))
    removed_non_ancestors = sorted(set(v) - set(ancestors_y))
    if removed_non_ancestors:
        reduced = admg.induced_subgraph(ancestors_y)
        reduced_x = [n for n in x if n in set(ancestors_y)]
        reduced_q_scope = [n for n in q_scope if n in set(ancestors_y)]
        _step(
            steps,
            depth=depth,
            rule_id="ID-2",
            status="applied_ancestor_reduction",
            admg=admg,
            y=y,
            x=x,
            q_input_scope=q_scope,
            ancestral_nodes="|".join(ancestors_y),
            removed_non_ancestors="|".join(removed_non_ancestors),
        )
        status, rule, terminal, hit = _canonical_id_control_walk(reduced, y, reduced_x, depth=depth + 1, max_depth=max_depth, steps=steps, seen=seen, q_input_scope=reduced_q_scope)
        return status, rule, terminal, max(max_depth_hit, hit)

    # ID-3: W-step / irrelevant intervention promotion.
    graph_without_incoming_x = _remove_incoming_to(admg, x)
    ancestors_after_do = sorted(graph_without_incoming_x.ancestors(y))
    w = sorted((set(v) - set(x)) - set(ancestors_after_do))
    if w:
        _step(
            steps,
            depth=depth,
            rule_id="ID-3",
            status="applied_w_step_promote_irrelevant_variables",
            admg=admg,
            y=y,
            x=x,
            q_input_scope=q_scope,
            promoted_to_intervention="|".join(w),
            ancestors_after_do="|".join(ancestors_after_do),
        )
        status, rule, terminal, hit = _canonical_id_control_walk(admg, y, sorted(set(x) | set(w)), depth=depth + 1, max_depth=max_depth, steps=steps, seen=seen, q_input_scope=q_scope)
        return status, rule, terminal, max(max_depth_hit, hit)

    # Observed DAG shortcut is ID-6 in the canonical trace.
    if not admg.bidirected_edges:
        _step(steps, depth=depth, rule_id="ID-6", status="terminal_observed_dag_truncated_factorization", admg=admg, y=y, x=x, q_input_scope=q_scope)
        return "identified_or_delegated", "ID-6", "terminal_observed_dag_truncated_factorization", max_depth_hit

    # ID-4: district decomposition in G[V\X].
    remaining_nodes = sorted(set(v) - set(x))
    gx = admg.induced_subgraph(remaining_nodes)
    remaining_districts = gx.districts()
    if len(remaining_districts) > 1:
        _step(
            steps,
            depth=depth,
            rule_id="ID-4",
            status="applied_district_decomposition_g_v_minus_x",
            admg=admg,
            y=y,
            x=x,
            q_input_scope=q_scope,
            districts=_format_components(remaining_districts),
        )
        final_status = "identified_or_delegated"
        terminal_rule = "ID-4"
        terminal_status = "applied_district_decomposition_g_v_minus_x"
        for district in remaining_districts:
            d = sorted(district)
            # This mirrors ID's product over c-factors for districts in G[V\X].
            sub_status, sub_rule, sub_terminal, hit = _canonical_id_control_walk(
                admg,
                d,
                sorted(set(v) - set(d)),
                depth=depth + 1,
                max_depth=max_depth,
                steps=steps,
                seen=seen,
                q_input_scope=q_scope,
            )
            max_depth_hit = max(max_depth_hit, hit)
            if sub_status == "blocked":
                final_status = "blocked"
                terminal_rule = sub_rule
                terminal_status = sub_terminal
                break
            terminal_rule = sub_rule
            terminal_status = sub_terminal
        return final_status, terminal_rule, terminal_status, max_depth_hit

    # Single district in G[V\X]: either direct Q-factor, recursive Q-input, or FAIL.
    if len(remaining_districts) == 1:
        s = sorted(remaining_districts[0])
        full_districts = admg.districts()
        if any(_is_same_component(s, d) for d in full_districts):
            _step(
                steps,
                depth=depth,
                rule_id="ID-6",
                status="terminal_q_factor_full_district",
                admg=admg,
                y=y,
                x=x,
                q_input_scope=q_scope,
                district="|".join(s),
            )
            return "identified_or_delegated", "ID-6", "terminal_q_factor_full_district", max_depth_hit

        containing = [sorted(d) for d in full_districts if set(s).issubset(set(d)) and set(s) != set(d)]
        if containing and not q_scope:
            containing_district = containing[0]
            working = admg.induced_subgraph(containing_district)
            y_inside = [n for n in y if n in set(containing_district)] or s
            x_inside = [n for n in x if n in set(containing_district) and n not in set(y_inside)]
            _step(
                steps,
                depth=depth,
                rule_id="ID-7",
                status="applied_recursive_q_input_subdistrict",
                admg=admg,
                y=y,
                x=x,
                q_input_scope=q_scope,
                subdistrict="|".join(s),
                containing_district="|".join(containing_district),
            )
            sub_status, sub_rule, sub_terminal, hit = _canonical_id_control_walk(
                working,
                y_inside,
                x_inside,
                depth=depth + 1,
                max_depth=max_depth,
                steps=steps,
                seen=seen,
                q_input_scope=containing_district,
            )
            return sub_status, sub_rule, sub_terminal, max(max_depth_hit, hit)

        # If already inside Q[S'] and still stuck on a strict subdistrict, the
        # current implementation should surface a hedge/block.  Keep this as ID-5
        # until full general Q-input recursion replaces the delegate.
        _step(
            steps,
            depth=depth,
            rule_id="ID-5",
            status="terminal_fail_or_pending_strict_subdistrict",
            admg=admg,
            y=y,
            x=x,
            q_input_scope=q_scope,
            district="|".join(s),
            containing_districts=_format_components(containing),
        )
        return "blocked", "ID-5", "terminal_fail_or_pending_strict_subdistrict", max_depth_hit

    _step(steps, depth=depth, rule_id="ID-5", status="blocked_unhandled_control_branch", admg=admg, y=y, x=x, q_input_scope=q_scope)
    return "blocked", "ID-5", "blocked_unhandled_control_branch", max_depth_hit


def _canonical_control_flow(admg: ADMG, treatments: Sequence[str], outcomes: Sequence[str], *, max_depth: int = 8) -> CanonicalControlDiagnostic:
    steps: List[CanonicalIDStep] = []
    status, terminal_rule, terminal_status, max_depth_hit = _canonical_id_control_walk(
        admg,
        outcomes,
        treatments,
        depth=0,
        max_depth=max_depth,
        steps=steps,
        seen=set(),
        q_input_scope=(),
    )
    applied: List[str] = []
    seen_rules: Set[str] = set()
    for step in steps:
        if step.rule_id not in seen_rules:
            seen_rules.add(step.rule_id)
            applied.append(step.rule_id)
    return CanonicalControlDiagnostic(
        status=status,
        terminal_rule=terminal_rule,
        terminal_status=terminal_status,
        applied_rules="|".join(applied),
        n_steps=len(steps),
        max_depth_hit=max_depth_hit,
        steps=tuple(steps),
    )


# ---------------------------------------------------------------------------
# Public diagnostics / entrypoints
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FullIDDiagnostic:
    """Stable return object for Full-ID-shaped queries.

    ``identified`` reports what the current delegated recursive runtime could
    identify. ``full_id_claim_allowed`` is deliberately separate and remains 0
    until the canonical arbitrary-ID matrix is implemented.
    """

    interface_version: str
    implementation_level: str
    treatments: str
    outcomes: str
    identified: bool
    identification_status: str
    formula: str = ""
    formula_ast_json: str = ""
    expression_json: str = ""
    trace_json: str = ""
    canonical_rules: str = ""
    canonical_control_json: str = ""
    canonical_terminal_rule: str = ""
    canonical_terminal_status: str = ""
    carried_q_context_json: str = ""
    carried_q_formula_ast_json: str = ""
    carried_q_context_enabled: int = 0
    carried_q_operational_ast_enabled: int = 0
    canonical_formula_status: str = ""
    canonical_formula_json: str = ""
    canonical_formula_ast_json: str = ""
    canonical_formula_authority_level: str = ""
    canonical_formula_version: str = ""
    canonical_formula_used_for_output: int = 0
    canonical_id7_carried_q_formula_used: int = 0
    primary_formula_authority: str = "recursive_id_set_expression_diagnostic"
    delegate_runtime: str = "recursive_id_set_expression_diagnostic"
    blocker: str = ""
    blocker_class: str = ""
    pending_operator: str = ""
    reason_codes: str = ""
    failure_certificate_status: str = ""
    failure_certificate_json: str = ""
    failure_certified: int = 0
    formal_hedge_certificate_json: str = ""
    failure_ast_json: str = ""
    failure_trace_json: str = ""
    failure_certificate_version: str = ID_FAILURE_CERTIFICATE_VERSION
    failure_certificate_level: str = ID_FAILURE_CERTIFICATE_LEVEL
    full_id_claim_allowed: int = 0
    full_id_claim_reason: str = "general_ID_and_IDC_not_complete_yet"
    do_proof_status: str = "not_run"
    do_proof_json: str = ""
    do_proof_terminal_formula: str = ""
    do_proof_terminal_ast_json: str = ""
    do_proof_authority: str = DO_PROOF_ENGINE_AUTHORITY
    do_proof_engine_version: str = DO_PROOF_ENGINE_VERSION
    do_proof_terminal_observational: int = 0
    do_proof_explored_states: int = 0
    do_proof_reason_codes: str = ""

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)



@dataclass(frozen=True)
class ConditionalIDDiagnostic:
    """Stable return object for conservative IDC-shaped queries.

    Step 62 adds conservative graph-disconnectivity pruning before falling back
    to the Step-57 ratio-over-joint pattern.

    Step 57 implements the safe IDC normalization pattern:
    identify P(Y,Z | do(X)) first, then return
    P(Y | do(X), Z) = P(Y,Z | do(X)) / sum_Y P(Y,Z | do(X)).
    It does not create new identification authority beyond ``full_id``.
    """

    interface_version: str
    implementation_level: str
    treatments: str
    outcomes: str
    conditions: str
    identified: bool
    identification_status: str
    formula: str = ""
    formula_ast_json: str = ""
    expression_json: str = ""
    trace_json: str = ""
    joint_full_id_json: str = ""
    numerator_formula: str = ""
    denominator_formula: str = ""
    blocker: str = ""
    blocker_class: str = ""
    pending_operator: str = ""
    reason_codes: str = ""
    failure_certificate_status: str = ""
    failure_certificate_json: str = ""
    failure_certified: int = 0
    formal_hedge_certificate_json: str = ""
    failure_ast_json: str = ""
    failure_trace_json: str = ""
    failure_certificate_version: str = ID_FAILURE_CERTIFICATE_VERSION
    failure_certificate_level: str = ID_FAILURE_CERTIFICATE_LEVEL
    idc_pruning_status: str = ""
    idc_pruning_json: str = ""
    idc_pruning_version: str = IDC_PRUNING_VERSION
    idc_pruning_level: str = IDC_PRUNING_LEVEL
    idc_simplification_used: int = 0
    idc_original_conditions: str = ""
    idc_effective_conditions: str = ""
    idc_pruned_conditions: str = ""
    full_id_claim_allowed: int = 0
    full_id_claim_reason: str = "IDC_step62_pruning_plus_step57_ratio_over_identified_joint_only_no_arbitrary_full_idc_claim"

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _rhs_from_estimand_formula(formula: str) -> str:
    text = _s(formula)
    if " = " in text:
        return text.split(" = ", 1)[1].strip()
    return text


def _conditional_formula_text(treatments: Sequence[str], outcomes: Sequence[str], conditions: Sequence[str], numerator_rhs: str) -> Tuple[str, str, str]:
    x = _joint_symbol(treatments) or ",".join(treatments)
    y = _joint_symbol(outcomes) or ",".join(outcomes)
    z = _joint_symbol(conditions) or ",".join(conditions)
    denominator = f"sum_{{{','.join(_dedupe(outcomes))}}} ({numerator_rhs})"
    formula = f"P_{{do({x})}}({y} | {z}) = ({numerator_rhs}) / ({denominator})"
    return formula, numerator_rhs, denominator


def _conditional_formula_text_with_pruning(
    treatments: Sequence[str],
    outcomes: Sequence[str],
    original_conditions: Sequence[str],
    effective_conditions: Sequence[str],
    numerator_rhs: str,
) -> Tuple[str, str, str]:
    x = _joint_symbol(treatments) or ",".join(treatments)
    y = _joint_symbol(outcomes) or ",".join(outcomes)
    z0 = _joint_symbol(original_conditions) or ",".join(original_conditions)
    ze = _joint_symbol(effective_conditions) or ",".join(effective_conditions)
    if effective_conditions:
        denominator = f"sum_{{{','.join(_dedupe(outcomes))}}} ({numerator_rhs})"
        formula = f"P_{{do({x})}}({y} | {z0}) = P_{{do({x})}}({y} | {ze}) = ({numerator_rhs}) / ({denominator})"
    else:
        denominator = ""
        formula = f"P_{{do({x})}}({y} | {z0}) = P_{{do({x})}}({y}) = {numerator_rhs}"
    return formula, numerator_rhs, denominator


def _conditional_pruned_marginal_expression_json(
    *,
    treatments: Sequence[str],
    outcomes: Sequence[str],
    original_conditions: Sequence[str],
    formula: str,
    numerator: str,
    ast_json: str,
    marginal: FullIDDiagnostic,
    pruning_json: str,
) -> str:
    ast_payload: Mapping[str, object] = {}
    try:
        raw = json.loads(ast_json) if ast_json else {}
        ast_payload = raw if isinstance(raw, Mapping) else {}
    except Exception:
        ast_payload = {}
    return _json({
        "kind": "idc_pruned_to_marginal_effect_step62",
        "idc_version": "idc_pruning_v1_step62_plus_idc_normalization_v1_step57",
        "formula": formula,
        "numerator_formula": numerator,
        "denominator_formula": "",
        "y_set": list(_dedupe(outcomes)),
        "x_set": list(_dedupe(treatments)),
        "z_set_original": list(_dedupe(original_conditions)),
        "z_set_effective": [],
        "marginal_full_id_status": marginal.identification_status,
        "marginal_primary_formula_authority": marginal.primary_formula_authority,
        "formula_ast": dict(ast_payload),
        "idc_pruning": json.loads(pruning_json) if pruning_json else {},
        "formula_ast_normalized": 1,
        "formula_ast_normalizer_version": ID_AST_NORMALIZER_VERSION,
        "full_id_claim_allowed": 0,
        "reason_codes": "IDC_STEP62_ALL_CONDITIONS_PRUNED_TO_MARGINAL_EFFECT",
    })


def _conditional_ast_from_joint(joint: FullIDDiagnostic, outcomes: Sequence[str]) -> str:
    try:
        payload = json.loads(joint.formula_ast_json) if joint.formula_ast_json else {}
        numerator_ast = ast_from_dict(payload) if isinstance(payload, Mapping) else Placeholder("missing_joint_formula_ast")
    except Exception as exc:  # pragma: no cover - defensive audit fallback
        numerator_ast = Placeholder("unparseable_joint_formula_ast", metadata={"error": f"{type(exc).__name__}:{exc}"})
    denominator_ast = Sum(outcomes, numerator_ast, label="idc_step57_normalization_sum_over_outcome")
    frac = Fraction(numerator_ast, denominator_ast, label="idc_step57_joint_over_normalizer")
    norm = normalize_formula_ast(frac)
    return _json(norm.to_dict())


def _conditional_expression_json(
    *,
    treatments: Sequence[str],
    outcomes: Sequence[str],
    conditions: Sequence[str],
    formula: str,
    numerator: str,
    denominator: str,
    ast_json: str,
    joint: FullIDDiagnostic,
    original_conditions: Sequence[str] = (),
    pruning_json: str = "",
) -> str:
    ast_payload: Mapping[str, object] = {}
    try:
        raw = json.loads(ast_json) if ast_json else {}
        ast_payload = raw if isinstance(raw, Mapping) else {}
    except Exception:
        ast_payload = {}
    return _json({
        "kind": "idc_conditional_effect_ratio_step57",
        "idc_version": "idc_normalization_v1_step57",
        "formula": formula,
        "numerator_formula": numerator,
        "denominator_formula": denominator,
        "y_set": list(_dedupe(outcomes)),
        "x_set": list(_dedupe(treatments)),
        "z_set": list(_dedupe(conditions)),
        "z_set_original": list(_dedupe(original_conditions or conditions)),
        "z_set_effective": list(_dedupe(conditions)),
        "joint_query_yz": list(_dedupe(list(outcomes) + list(conditions))),
        "joint_full_id_status": joint.identification_status,
        "joint_primary_formula_authority": joint.primary_formula_authority,
        "formula_ast": dict(ast_payload),
        "idc_pruning": json.loads(pruning_json) if pruning_json else {},
        "formula_ast_normalized": 1,
        "formula_ast_normalizer_version": ID_AST_NORMALIZER_VERSION,
        "full_id_claim_allowed": 0,
        "reason_codes": "IDC_STEP57_RATIO_OVER_IDENTIFIED_JOINT_NO_NEW_ID_AUTHORITY",
    })


def _conditional_trace_json(joint: FullIDDiagnostic) -> str:
    return _json({
        "trace_version": "idc_trace_v1_step57",
        "trace": [
            {"rule": "IDC-1", "status": "validated_disjoint_y_x_z"},
            {"rule": "IDC-2", "status": "identified_joint_yz_under_do_x", "joint_status": joint.identification_status},
            {"rule": "IDC-3", "status": "normalized_joint_by_summing_over_outcome"},
        ],
        "joint_full_id_trace_json": joint.trace_json,
        "full_id_claim_allowed": 0,
    })

def _invalid_control_payload(reason: str) -> str:
    return _json(
        {
            "control_version": "canonical_id_control_flow_v1_step56",
            "status": "invalid_query",
            "terminal_rule": "ID-5",
            "terminal_status": reason,
            "applied_rules": ["ID-5"],
            "n_steps": 1,
            "max_depth_hit": 0,
            "rule_names": _CANONICAL_ID_RULES,
            "steps": [],
            "full_id_claim_allowed": 0,
        }
    )


def full_id(
    admg: ADMG,
    treatments: Sequence[object] | object,
    outcomes: Sequence[object] | object,
    *,
    max_depth: int = 8,
) -> FullIDDiagnostic:
    """Run Amantia's Full-ID-shaped set-valued identification entrypoint.

    Step 53 adds a canonical ID-1 -> ID-7 control trace plus carried-Q context with operational formula AST before delegating formula
    authority.  This makes the migration path to true Full ID explicit without
    overclaiming completeness.
    """
    x = _normalise_nodes(treatments)
    y = _normalise_nodes(outcomes)
    caps = id_capability_flags()
    claim_allowed = int(caps.get("full_recursive_id_implemented") == 1)

    missing = sorted([n for n in list(x) + list(y) if n not in admg.node_set])
    overlap = sorted(set(x) & set(y))
    if not x or not y or missing or overlap:
        reasons: List[str] = []
        if not x:
            reasons.append("MISSING_TREATMENT_SET")
        if not y:
            reasons.append("MISSING_OUTCOME_SET")
        if missing:
            reasons.append("QUERY_NODE_NOT_IN_GRAPH:" + "|".join(missing))
        if overlap:
            reasons.append("TREATMENT_OUTCOME_OVERLAP:" + "|".join(overlap))
        reason = ";".join(reasons) or "INVALID_FULL_ID_QUERY"
        failure = rejection_certificate(
            treatments=x,
            outcomes=y,
            failure_kind="invalid_query",
            blocker_class="invalid_query",
            pending_operator="validate_full_id_query",
            blocker=reason,
            reason_codes=reason,
            source_status="invalid_full_id_query",
        )
        return FullIDDiagnostic(
            ID_FULL_INTERFACE_VERSION,
            ID_FULL_IMPLEMENTATION_LEVEL,
            _join(x),
            _join(y),
            False,
            "invalid_full_id_query",
            canonical_control_json=_invalid_control_payload(reason),
            canonical_terminal_rule="ID-5",
            canonical_terminal_status=reason,
            blocker=reason,
            blocker_class="invalid_query",
            pending_operator="validate_full_id_query",
            reason_codes=reason,
            failure_certificate_status=failure.certificate_status,
            failure_certificate_json=failure.certificate_json,
            failure_certified=int(failure.certified),
            formal_hedge_certificate_json=failure.formal_hedge_certificate_json,
            failure_ast_json=failure.failure_ast_json,
            failure_trace_json=failure.failure_trace_json,
            formula_ast_json=failure.failure_ast_json,
            trace_json=failure.failure_trace_json,
            primary_formula_authority=ID_FAILURE_CERTIFICATE_AUTHORITY,
            full_id_claim_allowed=claim_allowed,
            **_empty_do_proof_metadata("not_run_invalid_query"),
        )

    cycles = directed_cycle_nodes(admg)
    if cycles:
        reason = "DIRECTED_CYCLE_NOT_ADMG_DAG"
        failure = rejection_certificate(
            treatments=x,
            outcomes=y,
            failure_kind="directed_cycle",
            blocker_class="directed_cycle",
            pending_operator="repair_or_reject_cyclic_directed_graph",
            blocker="|".join(cycles),
            reason_codes=reason,
            source_status="blocked_directed_cycle",
        )
        return FullIDDiagnostic(
            ID_FULL_INTERFACE_VERSION,
            ID_FULL_IMPLEMENTATION_LEVEL,
            _join(x),
            _join(y),
            False,
            "blocked_directed_cycle",
            canonical_control_json=_invalid_control_payload(reason),
            canonical_terminal_rule="ID-5",
            canonical_terminal_status="blocked_directed_cycle",
            blocker="|".join(cycles),
            blocker_class="directed_cycle",
            pending_operator="repair_or_reject_cyclic_directed_graph",
            reason_codes=reason,
            failure_certificate_status=failure.certificate_status,
            failure_certificate_json=failure.certificate_json,
            failure_certified=int(failure.certified),
            formal_hedge_certificate_json=failure.formal_hedge_certificate_json,
            failure_ast_json=failure.failure_ast_json,
            failure_trace_json=failure.failure_trace_json,
            formula_ast_json=failure.failure_ast_json,
            trace_json=failure.failure_trace_json,
            primary_formula_authority=ID_FAILURE_CERTIFICATE_AUTHORITY,
            full_id_claim_allowed=claim_allowed,
            **_empty_do_proof_metadata("not_run_blocked_directed_cycle"),
        )

    control = _canonical_control_flow(admg, x, y, max_depth=max_depth)
    control_json = _json(control.to_dict())
    canonical_formula = canonical_id_formula_diagnostic(admg, x, y, max_depth=max_depth)

    expr = recursive_id_set_expression_diagnostic(admg, x, y, max_depth=max_depth)
    delegate_identified = bool(expr.expression_identified)
    canonical_identified = bool(canonical_formula.identified)
    use_canonical_formula = canonical_identified
    identified = bool(canonical_identified or delegate_identified)
    chosen_rhs = canonical_formula.formula if use_canonical_formula else expr.formula
    formula = _estimand_formula(x, y, chosen_rhs) if identified else ""
    expression_payload = _load_expression_payload(expr)
    delegate_rules = _extract_canonical_rules(expr.trace_json)
    merged_rules: List[str] = []
    for rule in list(control.applied_rules.split("|") if control.applied_rules else []) + list(delegate_rules.split("|") if delegate_rules else []):
        if rule and rule not in merged_rules:
            merged_rules.append(rule)
    source_status = expr.expression_status if delegate_identified or not canonical_identified else canonical_formula.status
    source_blocker = expr.blocker if not canonical_identified else canonical_formula.blocker
    source_blocker_class = expr.blocker_class if not canonical_identified else canonical_formula.blocker_class
    source_pending_operator = expr.pending_operator if not canonical_identified else canonical_formula.pending_operator
    source_reason_codes = expr.reason_codes or _s(expression_payload.get("reason_codes")) or canonical_formula.reason_codes
    failure = None
    if not identified:
        failure = failure_certificate_for_query(
            admg,
            x,
            y,
            source_status=source_status,
            source_blocker=source_blocker,
            source_blocker_class=source_blocker_class,
            source_pending_operator=source_pending_operator,
            source_reason_codes=source_reason_codes,
        )
        if failure.formal_hedge_certified:
            source_status = "blocked_formal_hedge_certificate"
            source_blocker = failure.blocker
            source_blocker_class = failure.blocker_class
            source_pending_operator = failure.pending_operator
            source_reason_codes = failure.reason_codes
    claim_reason = (
        "full_recursive_id_implemented_flag_enabled"
        if claim_allowed
        else "general_ID_and_IDC_not_complete_yet;canonical_ID_1_ID_2_ID_4_ID_6_gated_ID7_frontdoor_contextual_and_failure_certificates_ready"
    )
    do_proof_metadata = _do_proof_metadata_for_query(admg, x, y, max_depth=max_depth)
    return FullIDDiagnostic(
        ID_FULL_INTERFACE_VERSION,
        ID_FULL_IMPLEMENTATION_LEVEL,
        _join(x),
        _join(y),
        identified,
        source_status,
        formula=formula,
        formula_ast_json=(canonical_formula.formula_ast_json if use_canonical_formula else (failure.failure_ast_json if failure else _extract_formula_ast_json(expr))),
        expression_json=canonical_formula.expression_json if use_canonical_formula else expr.expression_json,
        trace_json=(canonical_formula.trace_json if use_canonical_formula else (failure.failure_trace_json if failure else expr.trace_json)),
        canonical_rules="|".join(merged_rules),
        canonical_control_json=control_json,
        canonical_terminal_rule=control.terminal_rule,
        canonical_terminal_status=control.terminal_status,
        carried_q_context_json=_extract_carried_q_context_json(expr),
        carried_q_formula_ast_json=_extract_carried_q_formula_ast_json(expr),
        carried_q_context_enabled=int(bool(_extract_carried_q_context_json(expr))),
        carried_q_operational_ast_enabled=_extract_operational_carried_q_ast_enabled(expr),
        canonical_formula_status=canonical_formula.status,
        canonical_formula_json=canonical_formula.expression_json,
        canonical_formula_ast_json=canonical_formula.formula_ast_json,
        canonical_formula_authority_level=ID_CANONICAL_FORMULA_LEVEL,
        canonical_formula_version=ID_CANONICAL_FORMULA_VERSION,
        canonical_formula_used_for_output=int(use_canonical_formula),
        canonical_id7_carried_q_formula_used=int(canonical_formula.id7_carried_q_formula_used),
        primary_formula_authority=(ID_CANONICAL_FORMULA_AUTHORITY if use_canonical_formula else (ID_FAILURE_CERTIFICATE_AUTHORITY if failure else "recursive_id_set_expression_diagnostic")),
        blocker=source_blocker if not canonical_identified or not identified else "",
        blocker_class=source_blocker_class if not canonical_identified or not identified else "",
        pending_operator=source_pending_operator if not canonical_identified or not identified else "none",
        reason_codes=source_reason_codes,
        failure_certificate_status=failure.certificate_status if failure else "",
        failure_certificate_json=failure.certificate_json if failure else "",
        failure_certified=int(failure.certified) if failure else 0,
        formal_hedge_certificate_json=failure.formal_hedge_certificate_json if failure else "",
        failure_ast_json=failure.failure_ast_json if failure else "",
        failure_trace_json=failure.failure_trace_json if failure else "",
        full_id_claim_allowed=claim_allowed,
        full_id_claim_reason=claim_reason,
        **do_proof_metadata,
    )


def full_id_from_scm_graph(
    scm_graph: Mapping[str, object],
    treatments: Sequence[object] | object,
    outcomes: Sequence[object] | object,
    *,
    max_depth: int = 8,
) -> FullIDDiagnostic:
    """Build an ADMG from an SCM graph payload and run ``full_id``."""
    return full_id(admg_from_scm_graph(scm_graph), treatments, outcomes, max_depth=max_depth)



def identify_conditional_effect(
    admg: ADMG,
    treatments: Sequence[object] | object,
    outcomes: Sequence[object] | object,
    conditions: Sequence[object] | object,
    *,
    max_depth: int = 8,
) -> ConditionalIDDiagnostic:
    """Conservative IDC-shaped entrypoint for P(Y | do(X), Z).

    Step 62 first applies a conservative graph-disconnectivity pruning rule to
    remove conditioning variables that are structurally irrelevant after do(X).
    It then falls back to the Step-57 safe pattern: identify the effective joint
    interventional query P(Y,Z_eff | do(X)) and return the normalized ratio.
    No arbitrary conditional-ID completeness claim is made here.
    """
    x = _normalise_nodes(treatments)
    y = _normalise_nodes(outcomes)
    z = _normalise_nodes(conditions)
    caps = id_capability_flags()
    claim_allowed = int(caps.get("full_recursive_id_implemented") == 1)
    level = ID_FULL_IMPLEMENTATION_LEVEL + "+idc_pruning_step62+idc_normalization_step57"

    missing = sorted([n for n in list(x) + list(y) + list(z) if n not in admg.node_set])
    overlap_xy = sorted(set(x) & set(y))
    overlap_xz = sorted(set(x) & set(z))
    overlap_yz = sorted(set(y) & set(z))
    if not x or not y or not z or missing or overlap_xy or overlap_xz or overlap_yz:
        reasons: List[str] = []
        if not x:
            reasons.append("MISSING_TREATMENT_SET")
        if not y:
            reasons.append("MISSING_OUTCOME_SET")
        if not z:
            reasons.append("MISSING_CONDITION_SET_FOR_IDC")
        if missing:
            reasons.append("QUERY_NODE_NOT_IN_GRAPH:" + "|".join(missing))
        if overlap_xy:
            reasons.append("TREATMENT_OUTCOME_OVERLAP:" + "|".join(overlap_xy))
        if overlap_xz:
            reasons.append("TREATMENT_CONDITION_OVERLAP:" + "|".join(overlap_xz))
        if overlap_yz:
            reasons.append("OUTCOME_CONDITION_OVERLAP:" + "|".join(overlap_yz))
        reason = ";".join(reasons) or "INVALID_IDC_QUERY"
        failure = rejection_certificate(
            treatments=x,
            outcomes=list(y) + list(z),
            failure_kind="invalid_idc_query",
            blocker_class="invalid_query",
            pending_operator="validate_idc_query",
            blocker=reason,
            reason_codes=reason,
            source_status="invalid_idc_query",
        )
        return ConditionalIDDiagnostic(
            ID_FULL_INTERFACE_VERSION,
            level,
            _join(x),
            _join(y),
            _join(z),
            False,
            "invalid_idc_query",
            blocker=reason,
            blocker_class="invalid_query",
            pending_operator="validate_idc_query",
            reason_codes=reason,
            failure_certificate_status=failure.certificate_status,
            failure_certificate_json=failure.certificate_json,
            failure_certified=int(failure.certified),
            formal_hedge_certificate_json=failure.formal_hedge_certificate_json,
            failure_ast_json=failure.failure_ast_json,
            failure_trace_json=failure.failure_trace_json,
            full_id_claim_allowed=claim_allowed,
        )

    pruning = idc_pruning_diagnostic(admg, x, y, z)
    pruning_json = _json(pruning.to_dict())
    effective_z = tuple(pruning.kept_conditions)

    if pruning.simplification_used and not effective_z:
        marginal = full_id(admg, x, y, max_depth=max_depth)
        marginal_json = _json(marginal.to_dict())
        if not marginal.identified:
            reason = "IDC_STEP62_PRUNED_CONDITIONS_BUT_MARGINAL_Y_UNDER_DO_X_NOT_IDENTIFIED"
            return ConditionalIDDiagnostic(
                ID_FULL_INTERFACE_VERSION,
                level,
                _join(x),
                _join(y),
                _join(z),
                False,
                "blocked_idc_pruned_marginal_not_identified_step62",
                joint_full_id_json=marginal_json,
                blocker=marginal.blocker or reason,
                blocker_class=marginal.blocker_class or "marginal_full_id_not_identified",
                pending_operator="identify_marginal_y_under_do_x_after_idc_pruning",
                reason_codes=(marginal.reason_codes + ";" if marginal.reason_codes else "") + reason,
                failure_certificate_status=marginal.failure_certificate_status,
                failure_certificate_json=marginal.failure_certificate_json,
                failure_certified=marginal.failure_certified,
                formal_hedge_certificate_json=marginal.formal_hedge_certificate_json,
                failure_ast_json=marginal.failure_ast_json,
                failure_trace_json=marginal.failure_trace_json,
                idc_pruning_status=pruning.status,
                idc_pruning_json=pruning_json,
                idc_simplification_used=1,
                idc_original_conditions=_join(z),
                idc_effective_conditions="",
                idc_pruned_conditions=_join(pruning.pruned_conditions),
                full_id_claim_allowed=claim_allowed,
            )
        numerator_rhs = _rhs_from_estimand_formula(marginal.formula)
        formula, numerator, denominator = _conditional_formula_text_with_pruning(x, y, z, (), numerator_rhs)
        ast_json = marginal.formula_ast_json
        expr_json = _conditional_pruned_marginal_expression_json(
            treatments=x,
            outcomes=y,
            original_conditions=z,
            formula=formula,
            numerator=numerator,
            ast_json=ast_json,
            marginal=marginal,
            pruning_json=pruning_json,
        )
        return ConditionalIDDiagnostic(
            ID_FULL_INTERFACE_VERSION,
            level,
            _join(x),
            _join(y),
            _join(z),
            True,
            "identified_idc_pruned_to_marginal_effect_step62",
            formula=formula,
            formula_ast_json=ast_json,
            expression_json=expr_json,
            trace_json=_json({
                "trace_version": "idc_trace_v2_step62",
                "trace": [
                    {"rule": "IDC-1", "status": "validated_disjoint_y_x_z"},
                    {"rule": "IDC-PRUNE-1", "status": pruning.status, "pruned_conditions": list(pruning.pruned_conditions)},
                    {"rule": "IDC-PRUNE-2", "status": "identified_marginal_y_under_do_x_after_all_conditions_pruned", "marginal_status": marginal.identification_status},
                ],
                "marginal_full_id_trace_json": marginal.trace_json,
                "idc_pruning_json": pruning_json,
                "full_id_claim_allowed": 0,
            }),
            joint_full_id_json=marginal_json,
            numerator_formula=numerator,
            denominator_formula=denominator,
            pending_operator="none",
            reason_codes="IDC_STEP62_ALL_CONDITIONS_PRUNED_TO_MARGINAL_EFFECT",
            idc_pruning_status=pruning.status,
            idc_pruning_json=pruning_json,
            idc_simplification_used=1,
            idc_original_conditions=_join(z),
            idc_effective_conditions="",
            idc_pruned_conditions=_join(pruning.pruned_conditions),
            full_id_claim_allowed=claim_allowed,
        )

    joint_outcomes = tuple(_dedupe(list(y) + list(effective_z)))
    joint = full_id(admg, x, joint_outcomes, max_depth=max_depth)
    joint_json = _json(joint.to_dict())
    if not joint.identified:
        reason = "IDC_STEP57_JOINT_YZ_UNDER_DO_X_NOT_IDENTIFIED"
        return ConditionalIDDiagnostic(
            ID_FULL_INTERFACE_VERSION,
            level,
            _join(x),
            _join(y),
            _join(z),
            False,
            "blocked_idc_joint_not_identified_step57",
            joint_full_id_json=joint_json,
            blocker=joint.blocker or reason,
            blocker_class=joint.blocker_class or "joint_full_id_not_identified",
            pending_operator="identify_joint_yz_under_do_x_before_idc_ratio",
            reason_codes=(joint.reason_codes + ";" if joint.reason_codes else "") + reason,
            failure_certificate_status=joint.failure_certificate_status,
            failure_certificate_json=joint.failure_certificate_json,
            failure_certified=joint.failure_certified,
            formal_hedge_certificate_json=joint.formal_hedge_certificate_json,
            failure_ast_json=joint.failure_ast_json,
            failure_trace_json=joint.failure_trace_json,
            idc_pruning_status=pruning.status,
            idc_pruning_json=pruning_json,
            idc_simplification_used=int(pruning.simplification_used),
            idc_original_conditions=_join(z),
            idc_effective_conditions=_join(effective_z),
            idc_pruned_conditions=_join(pruning.pruned_conditions),
            full_id_claim_allowed=claim_allowed,
        )

    numerator_rhs = _rhs_from_estimand_formula(joint.formula)
    if pruning.simplification_used:
        formula, numerator, denominator = _conditional_formula_text_with_pruning(x, y, z, effective_z, numerator_rhs)
    else:
        formula, numerator, denominator = _conditional_formula_text(x, y, effective_z, numerator_rhs)
    ast_json = _conditional_ast_from_joint(joint, y)
    expr_json = _conditional_expression_json(
        treatments=x,
        outcomes=y,
        conditions=effective_z,
        original_conditions=z,
        formula=formula,
        numerator=numerator,
        denominator=denominator,
        ast_json=ast_json,
        joint=joint,
        pruning_json=pruning_json,
    )
    return ConditionalIDDiagnostic(
        ID_FULL_INTERFACE_VERSION,
        level,
        _join(x),
        _join(y),
        _join(z),
        True,
        "identified_idc_ratio_over_identified_joint_step57",
        formula=formula,
        formula_ast_json=ast_json,
        expression_json=expr_json,
        trace_json=_conditional_trace_json(joint),
        joint_full_id_json=joint_json,
        numerator_formula=numerator,
        denominator_formula=denominator,
        pending_operator="none",
        reason_codes=("IDC_STEP62_CONDITION_PRUNING_APPLIED;" if pruning.simplification_used else "") + "IDC_STEP57_RATIO_OVER_IDENTIFIED_JOINT_NO_NEW_ID_AUTHORITY",
        idc_pruning_status=pruning.status,
        idc_pruning_json=pruning_json,
        idc_simplification_used=int(pruning.simplification_used),
        idc_original_conditions=_join(z),
        idc_effective_conditions=_join(effective_z),
        idc_pruned_conditions=_join(pruning.pruned_conditions),
        full_id_claim_allowed=claim_allowed,
    )


def identify_conditional_effect_from_scm_graph(
    scm_graph: Mapping[str, object],
    treatments: Sequence[object] | object,
    outcomes: Sequence[object] | object,
    conditions: Sequence[object] | object,
    *,
    max_depth: int = 8,
) -> ConditionalIDDiagnostic:
    """Build an ADMG from an SCM graph payload and run Step-62 IDC."""
    return identify_conditional_effect(
        admg_from_scm_graph(scm_graph),
        treatments,
        outcomes,
        conditions,
        max_depth=max_depth,
    )


__all__ = [
    "CanonicalControlDiagnostic",
    "CanonicalIDStep",
    "FullIDDiagnostic",
    "ConditionalIDDiagnostic",
    "ID_FULL_IMPLEMENTATION_LEVEL",
    "ID_FULL_INTERFACE_VERSION",
    "full_id",
    "full_id_from_scm_graph",
    "identify_conditional_effect",
    "identify_conditional_effect_from_scm_graph",
]
