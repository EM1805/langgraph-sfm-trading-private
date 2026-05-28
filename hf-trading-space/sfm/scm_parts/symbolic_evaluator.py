from __future__ import annotations

"""Conservative evaluator for SCM ID symbolic formula JSON.

This module is intentionally not a numeric causal effect estimator.  It turns
``id_algorithm.py`` symbolic JSON into a stable machine-readable evaluation plan
that downstream estimators can consume safely.

Policy:
- formulas with unresolved Q/c-factor terms are blocked;
- backdoor and limited-frontdoor formulas are marked numeric-estimator ready
  because Amantia already has corresponding gated do-estimator routes;
- observed-DAG truncated factorization is formula-evaluable and routed to
  ``symbolic_numeric.py`` for contract-gated sequential standardization;
- graphical-zero effects are evaluable without fitting a statistical effect.
"""

from dataclasses import asdict, dataclass
import csv
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .id_ast import FormulaAST, ast_from_dict
from .id_ast_normalizer import ID_AST_NORMALIZER_VERSION, normalize_formula_ast
from .idc_fraction_numeric import IDC_FRACTION_NUMERIC_VERSION, analyze_idc_fraction_ast
from .q_factor_numeric import Q_FACTOR_NUMERIC_VERSION, analyze_resolved_q_factor_ast

SYMBOLIC_EVALUATOR_VERSION = 5

SYMBOLIC_EVALUATION_COLUMNS: List[str] = [
    "insight_id",
    "treatment",
    "outcome",
    "symbolic_formula_status",
    "symbolic_formula_kind",
    "symbolic_evaluator_status",
    "formula_evaluable",
    "numeric_estimator_ready",
    "estimator_route",
    "estimator_family",
    "effect_estimate_semantics",
    "required_columns",
    "sum_over",
    "product_terms",
    "removed_terms",
    "unresolved_terms",
    "formula_type",
    "formula_json_valid",
    "formula_ast_present",
    "formula_ast_node_types",
    "formula_ast_bound_variables",
    "formula_ast_probability_terms",
    "formula_ast_q_factors",
    "formula_ast_placeholders",
    "formula_ast_evaluator_version",
    "formula_ast_normalizer_version",
    "blocker",
    "reason_codes",
]


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


def _as_list(value: object) -> List[str]:
    if isinstance(value, list):
        return _dedupe(value)
    if isinstance(value, tuple):
        return _dedupe(value)
    text = _s(value)
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return _dedupe(parsed)
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
    return _dedupe(part.strip() for part in text.replace(",", "|").split("|"))


def _read_csv(path: str | os.PathLike) -> List[Dict[str, str]]:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return []
    try:
        with p.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return []
            return [{k: ("" if v is None else str(v)) for k, v in row.items()} for row in reader]
    except (OSError, csv.Error, UnicodeDecodeError, ValueError, TypeError):
        return []


def _write_csv(path: str | os.PathLike, rows: Iterable[Mapping[str, object]], columns: Sequence[str]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})


def parse_symbolic_formula_json(value: object) -> Tuple[Optional[Dict[str, object]], str]:
    """Return ``(payload, error_code)`` for symbolic_formula_json."""
    text = _s(value)
    if not text:
        return None, "MISSING_SYMBOLIC_FORMULA_JSON"
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None, "INVALID_SYMBOLIC_FORMULA_JSON"
    if not isinstance(parsed, dict):
        return None, "SYMBOLIC_FORMULA_JSON_NOT_OBJECT"
    return parsed, ""

def parse_formula_ast_json(value: object) -> Tuple[Optional[FormulaAST], str]:
    """Return ``(FormulaAST, error_code)`` for Step-54+ formula AST JSON.

    This parser is deliberately strict about the top-level object but tolerant of
    unknown AST nodes: ``normalize_formula_ast`` will turn unknown constructs into
    placeholders instead of granting execution authority.
    """
    text = _s(value)
    if not text:
        return None, "MISSING_FORMULA_AST_JSON"
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None, "INVALID_FORMULA_AST_JSON"
    if not isinstance(parsed, dict):
        return None, "FORMULA_AST_JSON_NOT_OBJECT"
    try:
        return normalize_formula_ast(ast_from_dict(parsed)), ""
    except Exception as exc:  # pragma: no cover - defensive fallback
        return None, f"FORMULA_AST_PARSE_ERROR:{type(exc).__name__}"


def _ast_label(ast: FormulaAST) -> str:
    return _s(ast.label) or ast.node_type


def _probability_label(ast: FormulaAST) -> str:
    if ast.node_type != "probability":
        return ""
    left = ",".join(ast.variables)
    right = ",".join(ast.conditioned_on)
    do = ",".join(ast.interventions)
    base = f"P({left}{' | ' + right if right else ''})"
    return f"P_{{do({do})}}({left}{' | ' + right if right else ''})" if do else base


def _collect_ast_features(ast: FormulaAST) -> Dict[str, List[str]]:
    features: Dict[str, List[str]] = {
        "node_types": [],
        "variables": [],
        "interventions": [],
        "conditioned_on": [],
        "bound_variables": [],
        "probability_terms": [],
        "q_factors": [],
        "placeholders": [],
        "hedge_failures": [],
    }

    def add(key: str, values: Iterable[object]) -> None:
        features[key] = _dedupe([*features.get(key, []), *list(values or [])])

    def walk(node: FormulaAST) -> None:
        add("node_types", [node.node_type])
        add("variables", node.variables)
        add("interventions", node.interventions)
        add("conditioned_on", node.conditioned_on)
        add("bound_variables", node.bound_variables)
        if node.node_type == "probability":
            add("probability_terms", [_probability_label(node) or _ast_label(node)])
        elif node.node_type == "q_factor":
            add("q_factors", [_ast_label(node)])
        elif node.node_type == "placeholder":
            add("placeholders", [_ast_label(node)])
        elif node.node_type == "hedge_fail":
            add("hedge_failures", [_ast_label(node)])
        for child in node.children:
            walk(child)

    walk(ast)
    return features


def _ast_common(
    ast: FormulaAST,
    *,
    row: Optional[Mapping[str, object]] = None,
    symbolic_status: str = "identified_formula_ast",
    symbolic_kind: str = "formula_ast",
) -> Dict[str, object]:
    row = row or {}
    features = _collect_ast_features(ast)
    treatment = _s(row.get("treatment")) or _s(row.get("source")) or "|".join(features.get("interventions", []))
    outcome = _s(row.get("outcome")) or _s(row.get("target")) or "|".join(features.get("variables", []))
    required = _dedupe([treatment, outcome, *features.get("variables", []), *features.get("conditioned_on", []), *features.get("interventions", [])])
    return {
        "insight_id": _s(row.get("insight_id")),
        "treatment": treatment,
        "outcome": outcome,
        "symbolic_formula_status": symbolic_status,
        "symbolic_formula_kind": symbolic_kind,
        "required_columns": "|".join(required),
        "sum_over": "|".join(features.get("bound_variables", [])),
        "product_terms": "|".join(features.get("probability_terms", [])),
        "removed_terms": "",
        "unresolved_terms": "|".join([*features.get("q_factors", []), *features.get("placeholders", [])]),
        "formula_type": ast.node_type,
        "formula_json_valid": 1,
        "formula_ast_present": 1,
        "formula_ast_node_types": "|".join(features.get("node_types", [])),
        "formula_ast_bound_variables": "|".join(features.get("bound_variables", [])),
        "formula_ast_probability_terms": "|".join(features.get("probability_terms", [])),
        "formula_ast_q_factors": "|".join(features.get("q_factors", [])),
        "formula_ast_placeholders": "|".join(features.get("placeholders", [])),
        "formula_ast_evaluator_version": SYMBOLIC_EVALUATOR_VERSION,
        "formula_ast_normalizer_version": ID_AST_NORMALIZER_VERSION,
    }


def evaluate_formula_ast_payload(
    ast_payload: Mapping[str, object] | FormulaAST,
    *,
    row: Optional[Mapping[str, object]] = None,
    symbolic_status: str = "identified_formula_ast",
    symbolic_kind: str = "formula_ast",
) -> SymbolicEvaluationDiagnostic:
    """Evaluate a normalized ID formula AST into a conservative route plan.

    Step 64 makes the evaluator understand the canonical AST operators produced
    by the Full-ID facade.  It does not treat arbitrary Q-factors or placeholders
    as numerically executable; those remain auditable symbolic plans until the
    corresponding numeric estimators are implemented.
    """
    ast = normalize_formula_ast(ast_payload)
    features = _collect_ast_features(ast)
    common = _ast_common(ast, row=row, symbolic_status=symbolic_status, symbolic_kind=symbolic_kind)
    node_types = set(features.get("node_types", []))

    if features.get("hedge_failures"):
        return SymbolicEvaluationDiagnostic(
            **common,
            symbolic_evaluator_status="blocked_formula_ast_hedge_failure",
            formula_evaluable=0,
            numeric_estimator_ready=0,
            blocker="FORMAL_HEDGE_OR_ID_FAILURE_AST",
            reason_codes="FORMULA_AST_CONTAINS_HEDGE_FAIL_NODE",
        )

    placeholders = [p for p in features.get("placeholders", []) if p and p not in {"1"}]
    if placeholders:
        return SymbolicEvaluationDiagnostic(
            **common,
            symbolic_evaluator_status="blocked_formula_ast_placeholders",
            formula_evaluable=0,
            numeric_estimator_ready=0,
            blocker="FORMULA_AST_PLACEHOLDER_NODE",
            reason_codes="FORMULA_AST_HAS_UNRESOLVED_PLACEHOLDERS",
        )

    if features.get("q_factors"):
        q_plan = analyze_resolved_q_factor_ast(ast)
        if q_plan.numeric_ready:
            ready_common = {**common, "unresolved_terms": ""}
            return SymbolicEvaluationDiagnostic(
                **ready_common,
                symbolic_evaluator_status="evaluable_formula_ast_resolved_q_factor_numeric_ready",
                formula_evaluable=1,
                numeric_estimator_ready=1,
                estimator_route=q_plan.route,
                estimator_family="resolved_q_factor_chain_standardization",
                effect_estimate_semantics="resolved_Q_factor_AST_with_observed_probability_children_contract_gated_standardization",
                reason_codes=q_plan.reason_codes + f"|{Q_FACTOR_NUMERIC_VERSION}",
            )
        return SymbolicEvaluationDiagnostic(
            **common,
            symbolic_evaluator_status="symbolic_formula_ast_q_factor_plan_only",
            formula_evaluable=1,
            numeric_estimator_ready=0,
            estimator_route="symbolic_ast_q_factor_plan",
            estimator_family="q_factor_or_carried_q_symbolic_ast",
            effect_estimate_semantics="symbolic_q_factor_formula_requires_specialized_q_input_numeric_evaluator_or_expanded_children",
            blocker=q_plan.blocker,
            reason_codes=q_plan.reason_codes or "Q_FACTOR_AST_SYMBOLIC_ONLY_NUMERIC_EVALUATOR_PENDING",
        )

    if node_types.issubset({"probability", "product", "sum", "do"}):
        return SymbolicEvaluationDiagnostic(
            **common,
            symbolic_evaluator_status="evaluable_formula_ast_probability_sum_product_do",
            formula_evaluable=1,
            numeric_estimator_ready=1,
            estimator_route="symbolic_numeric_ast_standardization",
            estimator_family="ast_sum_product_probability_standardization",
            effect_estimate_semantics="ID_formula_ast_with_probability_sum_product_do_operators",
            reason_codes="FORMULA_AST_SUM_PRODUCT_PROBABILITY_DO_READY_FOR_CONTRACT_GATED_STANDARDIZATION",
        )

    if node_types.issubset({"probability", "product", "sum", "fraction", "do"}):
        plan = analyze_idc_fraction_ast(
            ast,
            outcome_hint=_as_list((row or {}).get("outcomes") or (row or {}).get("outcome") or (row or {}).get("target")),
            treatment_hint=_as_list((row or {}).get("treatments") or (row or {}).get("treatment") or (row or {}).get("source")),
        )
        if plan.numeric_ready:
            return SymbolicEvaluationDiagnostic(
                **common,
                symbolic_evaluator_status="evaluable_formula_ast_idc_fraction_numeric_ready",
                formula_evaluable=1,
                numeric_estimator_ready=1,
                estimator_route=plan.route,
                estimator_family="idc_fraction_ratio_conditional_mean_standardization",
                effect_estimate_semantics="IDC_fraction_ratio_AST_contract_gated_conditional_mean_standardization",
                reason_codes=plan.reason_codes + f"|{IDC_FRACTION_NUMERIC_VERSION}",
            )
        return SymbolicEvaluationDiagnostic(
            **common,
            symbolic_evaluator_status="evaluable_formula_ast_fraction_ratio_plan",
            formula_evaluable=1,
            numeric_estimator_ready=0,
            estimator_route="symbolic_ast_fraction_normalization_plan",
            estimator_family="idc_ratio_or_normalized_interventional_distribution",
            effect_estimate_semantics="symbolic_fraction_ratio_needs_normalization_evaluator_before_numeric_execution",
            blocker=plan.blocker,
            reason_codes=plan.reason_codes or "FORMULA_AST_FRACTION_EVALUABLE_AS_SYMBOLIC_PLAN_NUMERIC_RATIO_EVALUATOR_PENDING",
        )

    return SymbolicEvaluationDiagnostic(
        **common,
        symbolic_evaluator_status="unsupported_formula_ast_operator_set",
        formula_evaluable=0,
        numeric_estimator_ready=0,
        blocker="UNSUPPORTED_FORMULA_AST_OPERATOR_SET",
        reason_codes="UNSUPPORTED_FORMULA_AST_NODE_TYPES=" + "|".join(sorted(node_types)),
    )


def _estimand_nodes(payload: Mapping[str, object], row: Optional[Mapping[str, object]] = None) -> Tuple[str, str]:
    row = row or {}
    estimand = payload.get("estimand", {})
    if not isinstance(estimand, Mapping):
        estimand = {}
    x = _s(estimand.get("intervention")) or _s(row.get("treatment")) or _s(row.get("source"))
    y = _s(estimand.get("outcome")) or _s(row.get("outcome")) or _s(row.get("target"))
    return x, y


def _required_columns(treatment: str, outcome: str, sum_over: Sequence[str], product_terms: Sequence[str]) -> List[str]:
    # Keep this deterministic and conservative.  Formula terms are not parsed as
    # algebra; the observed-variable requirements come from the estimand and the
    # explicit summation/adjustment variables.
    return _dedupe([treatment, outcome, *sum_over])


@dataclass(frozen=True)
class SymbolicEvaluationDiagnostic:
    insight_id: str
    treatment: str
    outcome: str
    symbolic_formula_status: str
    symbolic_formula_kind: str
    symbolic_evaluator_status: str
    formula_evaluable: int = 0
    numeric_estimator_ready: int = 0
    estimator_route: str = ""
    estimator_family: str = ""
    effect_estimate_semantics: str = ""
    required_columns: str = ""
    sum_over: str = ""
    product_terms: str = ""
    removed_terms: str = ""
    unresolved_terms: str = ""
    formula_type: str = ""
    formula_json_valid: int = 0
    formula_ast_present: int = 0
    formula_ast_node_types: str = ""
    formula_ast_bound_variables: str = ""
    formula_ast_probability_terms: str = ""
    formula_ast_q_factors: str = ""
    formula_ast_placeholders: str = ""
    formula_ast_evaluator_version: object = ""
    formula_ast_normalizer_version: str = ""
    blocker: str = ""
    reason_codes: str = ""

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def evaluate_symbolic_formula_payload(
    payload: Mapping[str, object],
    *,
    row: Optional[Mapping[str, object]] = None,
) -> SymbolicEvaluationDiagnostic:
    """Evaluate one parsed symbolic formula into a conservative route plan."""
    row = row or {}
    formula_type = _s(payload.get("type"))
    treatment, outcome = _estimand_nodes(payload, row)
    insight_id = _s(row.get("insight_id"))
    symbolic_status = _s(row.get("symbolic_formula_status")) or "identified_symbolic_formula"
    symbolic_kind = _s(row.get("symbolic_formula_kind")) or formula_type
    sum_over = _as_list(payload.get("sum_over"))
    product_terms = _as_list(payload.get("product_terms")) or _as_list(payload.get("observed_terms"))
    removed_terms = _as_list(payload.get("removed_intervention_factors"))
    unresolved_terms = _as_list(payload.get("unresolved_terms"))
    if not unresolved_terms:
        unresolved_terms = _as_list(row.get("symbolic_unresolved_terms"))

    required = _required_columns(treatment, outcome, sum_over, product_terms)
    common = {
        "insight_id": insight_id,
        "treatment": treatment,
        "outcome": outcome,
        "symbolic_formula_status": symbolic_status,
        "symbolic_formula_kind": symbolic_kind,
        "required_columns": "|".join(required),
        "sum_over": "|".join(sum_over),
        "product_terms": "|".join(product_terms),
        "removed_terms": "|".join(removed_terms),
        "unresolved_terms": "|".join(unresolved_terms),
        "formula_type": formula_type,
        "formula_json_valid": 1,
    }

    if unresolved_terms:
        return SymbolicEvaluationDiagnostic(
            **common,
            symbolic_evaluator_status="blocked_unresolved_symbolic_terms",
            formula_evaluable=0,
            numeric_estimator_ready=0,
            blocker="UNRESOLVED_SYMBOLIC_TERMS",
            reason_codes="UNRESOLVED_Q_OR_C_FACTOR_TERMS_BLOCK_NUMERIC_EVALUATION",
        )

    if formula_type == "graphical_zero_effect":
        return SymbolicEvaluationDiagnostic(
            **common,
            symbolic_evaluator_status="evaluable_graphical_zero_effect",
            formula_evaluable=1,
            numeric_estimator_ready=1,
            estimator_route="zero_effect_contract",
            estimator_family="graphical_zero_no_fit",
            effect_estimate_semantics="zero_by_no_directed_path_not_statistical_effect_size",
            reason_codes="NO_DIRECTED_PATH_ZERO_EFFECT_EVALUABLE_WITHOUT_MODEL_FIT",
        )

    if formula_type == "backdoor_adjustment":
        return SymbolicEvaluationDiagnostic(
            **common,
            symbolic_evaluator_status="evaluable_backdoor_formula",
            formula_evaluable=1,
            numeric_estimator_ready=1,
            estimator_route="do_backdoor_outcome_regression",
            estimator_family="backdoor_standardization",
            effect_estimate_semantics="total_effect_do_x_via_backdoor_adjustment",
            reason_codes="BACKDOOR_SYMBOLIC_FORMULA_READY_FOR_GATED_DO_BACKDOOR_ESTIMATOR",
        )

    if formula_type == "frontdoor_limited":
        return SymbolicEvaluationDiagnostic(
            **common,
            symbolic_evaluator_status="evaluable_frontdoor_limited_formula",
            formula_evaluable=1,
            numeric_estimator_ready=1,
            estimator_route="do_frontdoor_limited_standardization",
            estimator_family="frontdoor_limited_standardization",
            effect_estimate_semantics="limited_frontdoor_total_effect_requires_mediator_models",
            reason_codes="LIMITED_FRONTDOOR_SYMBOLIC_FORMULA_READY_FOR_GATED_FRONTDOOR_ESTIMATOR",
        )

    if formula_type == "truncated_factorization":
        return SymbolicEvaluationDiagnostic(
            **common,
            symbolic_evaluator_status="evaluable_truncated_factorization_numeric_ready",
            formula_evaluable=1,
            numeric_estimator_ready=1,
            estimator_route="symbolic_numeric_truncated_factorization",
            estimator_family="observed_dag_g_formula_sequential_standardization",
            effect_estimate_semantics="observed_dag_total_effect_via_contract_gated_sequential_standardization",
            reason_codes="TRUNCATED_FACTORIZATION_SYMBOLIC_NUMERIC_EVALUATOR_READY",
        )

    if formula_type == "c_factor_product_placeholder":
        # A c-factor placeholder with no unresolved terms is rare in this scaffold;
        # keep it as a symbolic plan rather than enabling a numeric estimator.
        return SymbolicEvaluationDiagnostic(
            **common,
            symbolic_evaluator_status="symbolic_c_factor_product_plan_only",
            formula_evaluable=1,
            numeric_estimator_ready=0,
            estimator_route="symbolic_c_factor_plan",
            estimator_family="c_factor_product",
            effect_estimate_semantics="symbolic_only_no_numeric_c_factor_evaluator",
            reason_codes="C_FACTOR_PRODUCT_SYMBOLIC_ONLY_NUMERIC_EVALUATOR_PENDING",
        )

    return SymbolicEvaluationDiagnostic(
        **common,
        symbolic_evaluator_status="unsupported_symbolic_formula_type",
        formula_evaluable=0,
        numeric_estimator_ready=0,
        blocker="UNSUPPORTED_SYMBOLIC_FORMULA_TYPE",
        reason_codes=f"UNSUPPORTED_SYMBOLIC_FORMULA_TYPE={formula_type or 'missing'}",
    )


def evaluate_symbolic_formula_row(row: Mapping[str, object]) -> SymbolicEvaluationDiagnostic:
    # Step 64: rows from the Full-ID/IDC facade may carry formula_ast_json
    # instead of the older symbolic_formula_json contract.  Prefer the explicit
    # old contract when present for backwards compatibility, otherwise route the
    # normalized AST through the conservative AST evaluator.
    payload, err = parse_symbolic_formula_json(row.get("symbolic_formula_json"))
    if payload is None:
        ast, ast_err = parse_formula_ast_json(row.get("formula_ast_json"))
        if ast is not None:
            return evaluate_formula_ast_payload(
                ast,
                row=row,
                symbolic_status=_s(row.get("symbolic_formula_status")) or "identified_formula_ast",
                symbolic_kind=_s(row.get("symbolic_formula_kind")) or "formula_ast",
            )
        treatment = _s(row.get("treatment")) or _s(row.get("source"))
        outcome = _s(row.get("outcome")) or _s(row.get("target"))
        combined_err = err if err != "MISSING_SYMBOLIC_FORMULA_JSON" else ast_err or err
        return SymbolicEvaluationDiagnostic(
            insight_id=_s(row.get("insight_id")),
            treatment=treatment,
            outcome=outcome,
            symbolic_formula_status=_s(row.get("symbolic_formula_status")),
            symbolic_formula_kind=_s(row.get("symbolic_formula_kind")),
            symbolic_evaluator_status="invalid_or_missing_symbolic_json",
            formula_json_valid=0,
            blocker=combined_err,
            reason_codes=combined_err,
        )
    return evaluate_symbolic_formula_payload(payload, row=row)


def evaluate_id_audit_rows(rows: Iterable[Mapping[str, object]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for row in rows or []:
        # Rows without any symbolic JSON are still audited as invalid/missing so
        # downstream users can see why no evaluator route exists.
        out.append(evaluate_symbolic_formula_row(row).to_dict())
    return out


def symbolic_evaluation_summary(rows: Iterable[Mapping[str, object]]) -> Dict[str, object]:
    rows = list(rows or [])
    status_counts: Dict[str, int] = {}
    route_counts: Dict[str, int] = {}
    n_evaluable = 0
    n_ready = 0
    n_blocked = 0
    for row in rows:
        status = _s(row.get("symbolic_evaluator_status")) or "missing"
        route = _s(row.get("estimator_route")) or "none"
        status_counts[status] = status_counts.get(status, 0) + 1
        route_counts[route] = route_counts.get(route, 0) + 1
        if _s(row.get("formula_evaluable")) in {"1", "true", "True"}:
            n_evaluable += 1
        if _s(row.get("numeric_estimator_ready")) in {"1", "true", "True"}:
            n_ready += 1
        if _s(row.get("blocker")):
            n_blocked += 1
    return {
        "symbolic_evaluator_version": SYMBOLIC_EVALUATOR_VERSION,
        "n_rows": len(rows),
        "n_formula_evaluable": n_evaluable,
        "n_numeric_estimator_ready": n_ready,
        "n_blocked_or_invalid": n_blocked,
        "symbolic_evaluator_status_counts": status_counts,
        "estimator_route_counts": route_counts,
        "policy": "symbolic formulas and Step-54+ formula ASTs are routed into conservative evaluator plans; unresolved Q/c-factor/placeholder terms remain blocked or symbolic-only; AST sum/product/probability/do formulas and resolved IDC fraction-ratio formulas are numeric-ready only through contract-gated symbolic standardization/conditional-mean standardization.",
    }


def write_symbolic_evaluation(
    *,
    id_audit_csv: str | os.PathLike,
    out_csv: str | os.PathLike,
    out_summary_json: Optional[str | os.PathLike] = None,
) -> Dict[str, str]:
    rows = _read_csv(id_audit_csv)
    evaluated = evaluate_id_audit_rows(rows)
    _write_csv(out_csv, evaluated, SYMBOLIC_EVALUATION_COLUMNS)
    summary = symbolic_evaluation_summary(evaluated)
    if out_summary_json is None:
        out_summary_json = str(Path(out_csv).with_suffix(".summary.json"))
    p = Path(out_summary_json)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return {
        "symbolic_evaluation_csv": str(out_csv),
        "symbolic_evaluation_summary_json": str(out_summary_json),
    }


__all__ = [
    "SYMBOLIC_EVALUATOR_VERSION",
    "SYMBOLIC_EVALUATION_COLUMNS",
    "SymbolicEvaluationDiagnostic",
    "parse_symbolic_formula_json",
    "parse_formula_ast_json",
    "evaluate_symbolic_formula_payload",
    "evaluate_formula_ast_payload",
    "evaluate_symbolic_formula_row",
    "evaluate_id_audit_rows",
    "symbolic_evaluation_summary",
    "write_symbolic_evaluation",
]
