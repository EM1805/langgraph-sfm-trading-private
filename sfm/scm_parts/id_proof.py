from __future__ import annotations
"""Machine-readable proof traces for conservative SCM ID results."""
from dataclasses import asdict, dataclass
import json
from typing import Dict, List, Optional, Sequence


def _s(value: object) -> str:
    return "" if value is None else str(value).strip()


def _json_formula(payload: object) -> str:
    try:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))
    except Exception:
        return json.dumps({"serialization_status": "failed"}, sort_keys=True)


def _json_loads_or_empty(text: str) -> object:
    raw = _s(text)
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {"raw": raw, "parse_status": "unparsed"}


@dataclass(frozen=True)
class IDProofDiagnostic:
    id_proof_status: str
    id_proof_steps_json: str = ""
    formula_tree_json: str = ""
    proof_blocker: str = ""
    proof_reason_codes: str = ""

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _proof_step(name: str, status: str, **fields: object) -> Dict[str, object]:
    payload: Dict[str, object] = {"step": name, "status": status}
    for key, value in fields.items():
        if value not in (None, "", [], {}, ()):
            payload[key] = value
    return payload


_CANONICAL_ID_RULES = {
    "ID-1": "No intervention / marginalization base case",
    "ID-2": "Ancestor reduction",
    "ID-3": "W-step promotion of irrelevant variables into the intervention set",
    "ID-4": "C-component / district decomposition in G[V\\X]",
    "ID-5": "FAIL with formal hedge certificate",
    "ID-6": "Q-factor or observed DAG factorization",
    "ID-7": "Recursive subdistrict Q-input identification",
}

_TRACE_TO_CANONICAL_RULE = {
    "id_base_no_intervention": "ID-1",
    "ancestor_reduction": "ID-2",
    "irrelevant_after_intervention_w_step": "ID-3",
    "district_decomposition_g_v_minus_x": "ID-4",
    "formal_hedge_certificate": "ID-5",
    "q_factor_full_district": "ID-6",
    "observed_dag_truncated_factorization": "ID-6",
    "graphical_zero_effect": "ID-6",
    "q_input_subdistrict_recursion": "ID-7",
    "q_input_general_recursion_step44": "ID-7",
}


def _canonical_status(rule: str, raw_status: str) -> str:
    status = _s(raw_status) or "observed"
    if rule == "ID-5" and status in {"blocked", "blocked_formal_hedge"}:
        return "fail_hedge"
    if status in {"identified", "passed", "applied", "entered"}:
        return status
    if status in {"not_needed", "passed_or_not_applicable"}:
        return "not_needed"
    if "blocked" in status:
        return "blocked"
    return status


def _canonical_step_from_raw(raw: object) -> Optional[Dict[str, object]]:
    if not isinstance(raw, dict):
        return None
    raw_name = _s(raw.get("step"))
    rule = _TRACE_TO_CANONICAL_RULE.get(raw_name)
    if not rule:
        return None
    payload: Dict[str, object] = {
        "id_rule": rule,
        "rule_name": _CANONICAL_ID_RULES[rule],
        "raw_step": raw_name,
        "status": _canonical_status(rule, _s(raw.get("status"))),
    }
    for key in ("depth", "y", "x", "nodes", "q_input", "ancestral_nodes", "removed_non_ancestors", "ancestors_after_do", "promoted_to_intervention", "districts", "district", "containing_district", "sum_over", "blockers", "blocker_class", "formula_ast_source", "reason_codes", "F", "F_prime", "roots_F", "roots_F_prime", "treatment_in_F_minus_F_prime", "outcome_witness"):
        value = raw.get(key)
        if value not in (None, "", [], {}, ()):
            payload[key] = value
    return payload


def canonical_id_proof_trace(*, recursive_trace_json: str = "", recursive_expression_json: str = "", identifiable: bool = False, id_strategy: str = "", estimand_formula: str = "", treatment: str = "", outcome: str = "", reason_codes: str = "") -> Dict[str, object]:
    raw_payload = _json_loads_or_empty(recursive_trace_json)
    raw_trace = raw_payload.get("trace", []) if isinstance(raw_payload, dict) else []
    canonical_steps: List[Dict[str, object]] = []
    seen = set()
    for raw in raw_trace if isinstance(raw_trace, list) else []:
        step = _canonical_step_from_raw(raw)
        if step is None:
            continue
        key = _json_formula(step)
        if key in seen:
            continue
        seen.add(key)
        canonical_steps.append(step)
    expr_payload = _json_loads_or_empty(recursive_expression_json)
    if isinstance(expr_payload, dict) and _s(expr_payload.get("kind")) == "formal_hedge_certificate" and not any(s.get("id_rule") == "ID-5" for s in canonical_steps):
        hedge = expr_payload.get("formal_hedge_candidate") if isinstance(expr_payload.get("formal_hedge_candidate"), dict) else {}
        canonical_steps.append(_proof_step("canonical_id_fail_hedge", "fail_hedge", id_rule="ID-5", rule_name=_CANONICAL_ID_RULES["ID-5"], raw_step="formal_hedge_certificate", F=hedge.get("F", ""), F_prime=hedge.get("F_prime", ""), roots_F=hedge.get("roots_F", ""), roots_F_prime=hedge.get("roots_F_prime", ""), reason_codes=hedge.get("reason_codes", reason_codes)))
    applied_rules = [str(s.get("id_rule")) for s in canonical_steps if s.get("status") != "not_needed"]
    fail_rules = [str(s.get("id_rule")) for s in canonical_steps if str(s.get("status")) in {"fail_hedge", "blocked"}]
    return {"type": "canonical_id_proof_trace", "version": "id_proof_trace_v2_step47", "identified": bool(identifiable), "query": {"treatment": _s(treatment), "outcome": _s(outcome)}, "strategy": _s(id_strategy), "formula": _s(estimand_formula), "rules": _CANONICAL_ID_RULES, "applied_rules": applied_rules, "fail_rules": fail_rules, "steps": canonical_steps, "source": "recursive_trace_json_overlay_no_new_authority", "reason_codes": _s(reason_codes)}


def id_proof_diagnostic(*, treatment: str, outcome: str, identifiable: bool, id_strategy: str, id_algorithm_level: str, estimand_formula: str, hedge: object, cycles: Sequence[str], backdoor: Optional[object] = None, frontdoor: Optional[object] = None, factorization: Optional[object] = None, cdiag: Optional[object] = None, cfactor: Optional[object] = None, recursive: Optional[object] = None, symbolic: Optional[object] = None, do_calc: Optional[object] = None, formal_hedge: Optional[object] = None, failure_reason: str = "", reason_codes: str = "") -> IDProofDiagnostic:
    del do_calc, formal_hedge
    x = _s(treatment); y = _s(outcome)
    strategy = _s(id_strategy); level = _s(id_algorithm_level)
    reasons = _s(reason_codes) or _s(failure_reason)
    steps: List[Dict[str, object]] = []
    steps.append(_proof_step("query", "passed" if x and y else "blocked", treatment=x, outcome=y))
    steps.append(_proof_step("directed_acyclicity", "passed" if not cycles else "blocked", directed_cycle_nodes="|".join(cycles)))
    if recursive is not None:
        steps.append(_proof_step("recursive_routing", "identified" if bool(getattr(recursive, "recursive_identified", False)) else "blocked_or_pending", recursive_status=getattr(recursive, "recursive_status", ""), ancestral_nodes=getattr(recursive, "recursive_ancestral_nodes", ""), districts=getattr(recursive, "recursive_districts", ""), blocker=getattr(recursive, "recursive_blocker", ""), blocker_class=getattr(recursive, "recursive_blocker_class", ""), pending_operator=getattr(recursive, "recursive_pending_operator", ""), reason_codes=getattr(recursive, "reason_codes", "")))
    hedge_possible = bool(getattr(hedge, "possible_hedge", False))
    steps.append(_proof_step("hedge_diagnostic", "blocked" if hedge_possible and strategy != "frontdoor_limited" else "passed_or_not_applicable", hedge_status=getattr(hedge, "hedge_status", ""), hedge_witness=getattr(hedge, "hedge_witness", ""), reason_codes=getattr(hedge, "reason_codes", "")))
    if backdoor is not None:
        steps.append(_proof_step("backdoor_criterion", "passed" if bool(getattr(backdoor, "backdoor_ok", False)) else "blocked", backdoor_status=getattr(backdoor, "backdoor_status", ""), adjustment_set=getattr(backdoor, "adjustment_set", ""), open_paths=getattr(backdoor, "open_paths", ""), reason_codes=getattr(backdoor, "reason_codes", "")))
    if frontdoor is not None:
        steps.append(_proof_step("frontdoor_criterion", "passed" if bool(getattr(frontdoor, "frontdoor_ok", False)) else "blocked", frontdoor_status=getattr(frontdoor, "frontdoor_status", ""), active_mediators=getattr(frontdoor, "active_mediators", ""), witness_paths=getattr(frontdoor, "witness_paths", ""), reason_codes=getattr(frontdoor, "reason_codes", "")))
    if factorization is not None:
        steps.append(_proof_step("observed_dag_truncated_factorization", "passed" if bool(getattr(factorization, "factorization_ok", False)) else "not_applicable_or_blocked", factorization_status=getattr(factorization, "factorization_status", ""), topological_order=getattr(factorization, "topological_order", ""), eliminated_nodes=getattr(factorization, "eliminated_nodes", ""), reason_codes=getattr(factorization, "reason_codes", "")))
    if cfactor is not None:
        steps.append(_proof_step("c_factor_decomposition", "identified" if bool(getattr(cfactor, "c_factor_product_ok", False)) else "unresolved", c_factor_status=getattr(cfactor, "c_factor_status", ""), sum_over=getattr(cfactor, "c_factor_sum_over", ""), unresolved_districts=getattr(cfactor, "c_factor_unresolved_districts", ""), reason_codes=getattr(cfactor, "c_factor_reason_codes", "")))
    symbolic_payload = _json_loads_or_empty(getattr(symbolic, "symbolic_formula_json", "")) if symbolic is not None else {}
    if symbolic is not None:
        steps.append(_proof_step("symbolic_formula", "identified" if getattr(symbolic, "symbolic_formula_status", "") == "identified_symbolic_formula" else "blocked_or_unavailable", symbolic_formula_status=getattr(symbolic, "symbolic_formula_status", ""), symbolic_formula_kind=getattr(symbolic, "symbolic_formula_kind", ""), reason_codes=getattr(symbolic, "symbolic_reason_codes", "")))
    canonical_trace = canonical_id_proof_trace(recursive_trace_json=getattr(recursive, "recursive_trace_json", "") if recursive is not None else "", recursive_expression_json=getattr(recursive, "recursive_expression_json", "") if recursive is not None else "", identifiable=identifiable, id_strategy=strategy, estimand_formula=estimand_formula, treatment=x, outcome=y, reason_codes=reasons)
    if canonical_trace.get("steps"):
        steps.append(_proof_step("canonical_id_trace", "identified" if identifiable else "blocked_or_failed", canonical_version=canonical_trace.get("version"), applied_rules="|".join(canonical_trace.get("applied_rules", [])), fail_rules="|".join(canonical_trace.get("fail_rules", []))))
    proof_blocker = getattr(recursive, "recursive_blocker", "") if recursive is not None else ""
    if not proof_blocker and cfactor is not None:
        proof_blocker = getattr(cfactor, "c_factor_unresolved_districts", "")
    if not proof_blocker:
        proof_blocker = getattr(hedge, "hedge_witness", "") or _s(failure_reason)
    formula_tree = {"type": "id_proof_tree", "identified": bool(identifiable), "strategy": strategy, "id_algorithm_level": level, "estimand": {"intervention": x, "outcome": y}, "formula": estimand_formula, "symbolic_formula": symbolic_payload, "recursive_subproblem_plan": _json_loads_or_empty(getattr(recursive, "recursive_subproblem_plan_json", "")) if recursive is not None else {}, "recursive_reduction_chain": _json_loads_or_empty(getattr(recursive, "recursive_reduction_chain_json", "")) if recursive is not None else {}, "recursive_expression": _json_loads_or_empty(getattr(recursive, "recursive_expression_json", "")) if recursive is not None else {}, "recursive_trace": _json_loads_or_empty(getattr(recursive, "recursive_trace_json", "")) if recursive is not None else {}, "canonical_id_trace": canonical_trace, "blocker": proof_blocker, "blocker_class": getattr(recursive, "recursive_blocker_class", "") if recursive is not None else "", "pending_operator": getattr(recursive, "recursive_pending_operator", "") if recursive is not None else "", "reason_codes": reasons}
    return IDProofDiagnostic("identified_proof_trace" if identifiable else "blocked_proof_trace", id_proof_steps_json=_json_formula({"steps": steps}), formula_tree_json=_json_formula(formula_tree), proof_blocker=proof_blocker, proof_reason_codes=reasons)


__all__ = ["IDProofDiagnostic", "id_proof_diagnostic", "canonical_id_proof_trace", "_proof_step"]
