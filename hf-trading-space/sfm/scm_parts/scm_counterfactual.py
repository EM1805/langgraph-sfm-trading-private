from __future__ import annotations

"""SCM-native counterfactual authority layer.

This module is the canonical SCM entrypoint for counterfactual requests.
It deliberately separates:

* path-level diagnostic counterfactual reasoning, which belongs outside SCM; and
* SCM counterfactual authority, which must be gated by graphical ID.

Rule: no ID -> no counterfactual authority.

The current implementation authorizes only interventional/do-style estimands
that the SCM ID layer already marks as identifiable. Individual/nested
counterfactual distributions remain blocked until a formal counterfactual-ID
layer and exogenous-noise model are implemented.
"""

from dataclasses import asdict, dataclass
import csv
import json
import os
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

try:  # package import
    from .id_algorithm import IDResult, identify_effect_from_scm_graph
except Exception:  # pragma: no cover - direct script compatibility
    IDResult = Any  # type: ignore
    identify_effect_from_scm_graph = None  # type: ignore

SCM_COUNTERFACTUAL_VERSION = "scm_counterfactual_step20"
NO_ID_NO_COUNTERFACTUAL_AUTHORITY = "NO_ID_NO_COUNTERFACTUAL_AUTHORITY"
FORMAL_INDIVIDUAL_COUNTERFACTUAL_STATUS = "not_implemented_counterfactual_id_required"

SCM_COUNTERFACTUAL_COLUMNS = [
    "treatment",
    "outcome",
    "requested_treatment_value",
    "case_index",
    "counterfactual_authority",
    "counterfactual_kind",
    "id_identifiable",
    "id_strategy",
    "id_algorithm_level",
    "estimand_formula",
    "formula_tree_json",
    "formal_individual_counterfactual_authorized",
    "formal_individual_counterfactual_status",
    "numeric_evaluation_status",
    "diagnostic_proxy_available",
    "reason_codes",
]


def _s(value: object) -> str:
    raw = "" if value is None else str(value).strip()
    return "" if raw.lower() in {"nan", "none", "null"} else raw


def _join_codes(*items: object) -> str:
    out = []
    for item in items:
        if isinstance(item, (list, tuple, set)):
            values = item
        else:
            values = str(item or "").replace(",", "|").split("|")
        for value in values:
            code = _s(value)
            if code and code not in out:
                out.append(code)
    return "|".join(out)


def _get(obj: Any, key: str, default: Any = "") -> Any:
    """Read dataclass/object and dict-like ID results through one stable API."""
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _as_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _graph_id_rows(scm_graph: Optional[Mapping[str, object]]) -> List[Mapping[str, object]]:
    if not scm_graph:
        return []
    rows = scm_graph.get("id_algorithm_audit", [])
    if isinstance(rows, list):
        return [r for r in rows if isinstance(r, Mapping)]
    return []


@dataclass(frozen=True)
class SCMCounterfactualResult:
    treatment: str
    outcome: str
    requested_treatment_value: float = 0.0
    case_index: int = 0
    counterfactual_authority: str = "blocked"
    counterfactual_kind: str = "interventional_do_counterfactual"
    id_identifiable: int = 0
    id_strategy: str = ""
    id_algorithm_level: str = ""
    estimand_formula: str = ""
    formula_tree_json: str = ""
    formal_individual_counterfactual_authorized: int = 0
    formal_individual_counterfactual_status: str = FORMAL_INDIVIDUAL_COUNTERFACTUAL_STATUS
    numeric_evaluation_status: str = "not_requested"
    diagnostic_proxy_available: int = 0
    reason_codes: str = NO_ID_NO_COUNTERFACTUAL_AUTHORITY

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _id_result_from_input(
    *,
    scm_graph: Optional[Mapping[str, object]],
    treatment: str,
    outcome: str,
    id_result: Optional[Any] = None,
    id_kwargs: Optional[Mapping[str, object]] = None,
) -> Optional[Any]:
    if id_result is not None:
        return id_result
    if scm_graph is None or identify_effect_from_scm_graph is None:
        return None
    return identify_effect_from_scm_graph(scm_graph, treatment, outcome, **dict(id_kwargs or {}))


def evaluate_scm_counterfactual(
    *,
    treatment: str,
    outcome: str,
    intervention_value: float = 0.0,
    case_index: int = 0,
    scm_graph: Optional[Mapping[str, object]] = None,
    id_result: Optional[Any] = None,
    data: Optional[Any] = None,
    evaluate_numeric: bool = False,
    id_kwargs: Optional[Mapping[str, object]] = None,
) -> SCMCounterfactualResult:
    """Evaluate SCM counterfactual authority under the ID gate.

    Parameters are intentionally lightweight so this module can be used without
    importing pandas/numpy.  ``data`` is accepted for future numeric evaluation;
    currently numeric execution is delegated to the symbolic/do-estimation layer
    elsewhere in SCM and remains diagnostic unless explicitly implemented.
    """
    x = _s(treatment)
    y = _s(outcome)
    idr = _id_result_from_input(scm_graph=scm_graph, treatment=x, outcome=y, id_result=id_result, id_kwargs=id_kwargs)
    if idr is None:
        return SCMCounterfactualResult(
            treatment=x,
            outcome=y,
            requested_treatment_value=float(intervention_value),
            case_index=int(case_index),
            counterfactual_authority="blocked",
            numeric_evaluation_status="not_available_no_id_result",
            diagnostic_proxy_available=1,
            reason_codes="ID_RESULT_REQUIRED|" + NO_ID_NO_COUNTERFACTUAL_AUTHORITY,
        )

    identifiable = bool(_as_int(_get(idr, "identifiable", _get(idr, "id_identifiable", 0))))
    strategy = _s(_get(idr, "id_strategy", ""))
    level = _s(_get(idr, "id_algorithm_level", ""))
    formula = _s(_get(idr, "estimand_formula", ""))
    formula_tree = _s(_get(idr, "formula_tree_json", ""))
    id_reasons = _s(_get(idr, "reason_codes", ""))

    if not identifiable:
        return SCMCounterfactualResult(
            treatment=x,
            outcome=y,
            requested_treatment_value=float(intervention_value),
            case_index=int(case_index),
            counterfactual_authority="blocked",
            id_identifiable=0,
            id_strategy=strategy,
            id_algorithm_level=level,
            estimand_formula=formula,
            formula_tree_json=formula_tree,
            numeric_evaluation_status="blocked_by_id",
            diagnostic_proxy_available=1,
            reason_codes=_join_codes(NO_ID_NO_COUNTERFACTUAL_AUTHORITY, id_reasons or "ID_NOT_IDENTIFIED"),
        )

    numeric_status = "not_requested"
    if evaluate_numeric:
        # This is deliberately conservative: the SCM authority layer only marks
        # the estimand as ID-authorized.  Numeric evaluation must be performed by
        # symbolic_numeric/do_outputs where support, overlap, and robustness gates
        # already exist.
        numeric_status = "requires_symbolic_numeric_or_do_outputs_evaluation"

    return SCMCounterfactualResult(
        treatment=x,
        outcome=y,
        requested_treatment_value=float(intervention_value),
        case_index=int(case_index),
        counterfactual_authority="authorized_interventional_estimand",
        counterfactual_kind="id_authorized_do_estimand_not_individual_nested_cf",
        id_identifiable=1,
        id_strategy=strategy,
        id_algorithm_level=level,
        estimand_formula=formula,
        formula_tree_json=formula_tree,
        formal_individual_counterfactual_authorized=0,
        formal_individual_counterfactual_status=FORMAL_INDIVIDUAL_COUNTERFACTUAL_STATUS,
        numeric_evaluation_status=numeric_status,
        diagnostic_proxy_available=1,
        reason_codes=_join_codes("ID_AUTHORIZED_INTERVENTIONAL_ESTIMAND", "INDIVIDUAL_COUNTERFACTUAL_ID_NOT_IMPLEMENTED", id_reasons),
    )


def run_scm_counterfactual_proxy(
    *,
    treatment: str,
    outcome: str,
    intervention_value: float = 0.0,
    case_index: int = 0,
    scm_graph: Optional[Mapping[str, object]] = None,
    id_result: Optional[Any] = None,
    out_dir: str = "out",
    structural_models_path: Optional[str] = None,
    data_path: Optional[str] = None,
) -> Dict[str, object]:
    """Return SCM counterfactual authority metadata without legacy simulation.

    Step 35 physically removes the old diagnostic ``intervention.py`` facade.
    This compatibility entrypoint remains so callers get the same authority
    fields, but no removed diagnostic proxy is imported or executed.
    """
    authority = evaluate_scm_counterfactual(
        treatment=treatment,
        outcome=outcome,
        intervention_value=intervention_value,
        case_index=case_index,
        scm_graph=scm_graph,
        id_result=id_result,
    ).to_dict()
    proxy = {
        "diagnostic_proxy_status": "removed",
        "diagnostic_proxy_error": "legacy intervention diagnostic removed in SCM Step 35",
    }
    return {**proxy, **authority, "scm_counterfactual_version": SCM_COUNTERFACTUAL_VERSION}


def evaluate_counterfactual_audit_rows(
    *,
    scm_graph: Optional[Mapping[str, object]] = None,
    id_rows: Optional[Sequence[Mapping[str, object]]] = None,
    intervention_value: float = 0.0,
    case_index: int = 0,
    evaluate_numeric: bool = False,
) -> List[Dict[str, object]]:
    """Evaluate SCM counterfactual authority for every ID audit row.

    This is a pipeline/export helper: it never upgrades ID decisions. Rows that
    are not identified remain blocked with ``NO_ID_NO_COUNTERFACTUAL_AUTHORITY``.
    """
    rows = list(id_rows or _graph_id_rows(scm_graph))
    out: List[Dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        x = _s(row.get("treatment"))
        y = _s(row.get("outcome"))
        if not x or not y:
            continue
        key = (x, y)
        if key in seen:
            continue
        seen.add(key)
        result = evaluate_scm_counterfactual(
            treatment=x,
            outcome=y,
            intervention_value=intervention_value,
            case_index=case_index,
            scm_graph=scm_graph,
            id_result=row,
            evaluate_numeric=evaluate_numeric,
        ).to_dict()
        result["scm_counterfactual_version"] = SCM_COUNTERFACTUAL_VERSION
        out.append(result)
    return out


def write_scm_counterfactual_audit(
    *,
    scm_graph: Optional[Mapping[str, object]] = None,
    scm_graph_path: Optional[str] = None,
    out_dir: str = "out",
    intervention_value: float = 0.0,
    case_index: int = 0,
    evaluate_numeric: bool = False,
) -> Dict[str, str]:
    """Write a batch authority table from ``scm_graph.id_algorithm_audit``.

    Output is intentionally authority-only: numeric effects remain delegated to
    symbolic_numeric/do_outputs and final action authority remains in the gate.
    """
    graph: Optional[Mapping[str, object]] = scm_graph
    if graph is None and scm_graph_path:
        with open(scm_graph_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, Mapping):
            graph = payload
    rows = evaluate_counterfactual_audit_rows(
        scm_graph=graph,
        intervention_value=intervention_value,
        case_index=case_index,
        evaluate_numeric=evaluate_numeric,
    )
    scm_dir = os.path.join(out_dir, "scm")
    os.makedirs(scm_dir, exist_ok=True)
    csv_path = os.path.join(scm_dir, "scm_counterfactual_authority.csv")
    fieldnames = list(SCM_COUNTERFACTUAL_COLUMNS) + ["scm_counterfactual_version"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "layer": "scm_counterfactual",
        "version": SCM_COUNTERFACTUAL_VERSION,
        "policy": "No ID -> no counterfactual authority; individual/nested counterfactual ID remains blocked.",
        "n_rows": len(rows),
        "n_authorized_interventional_estimands": sum(1 for r in rows if r.get("counterfactual_authority") == "authorized_interventional_estimand"),
        "n_blocked": sum(1 for r in rows if r.get("counterfactual_authority") == "blocked"),
        "output_csv": csv_path,
    }
    summary_path = os.path.join(scm_dir, "scm_counterfactual_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return {"scm_counterfactual_authority_csv": csv_path, "scm_counterfactual_summary_json": summary_path}


def write_scm_counterfactual_outputs(
    *,
    treatment: str,
    outcome: str,
    intervention_value: float = 0.0,
    case_index: int = 0,
    scm_graph: Optional[Mapping[str, object]] = None,
    id_result: Optional[Any] = None,
    out_dir: str = "out",
    structural_models_path: Optional[str] = None,
    data_path: Optional[str] = None,
) -> Dict[str, str]:
    """Write SCM counterfactual authority CSV and JSON proxy report."""
    scm_dir = os.path.join(out_dir, "scm")
    os.makedirs(scm_dir, exist_ok=True)
    result = run_scm_counterfactual_proxy(
        treatment=treatment,
        outcome=outcome,
        intervention_value=intervention_value,
        case_index=case_index,
        scm_graph=scm_graph,
        id_result=id_result,
        out_dir=out_dir,
        structural_models_path=structural_models_path,
        data_path=data_path,
    )
    csv_path = os.path.join(scm_dir, "scm_counterfactual_authority.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SCM_COUNTERFACTUAL_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerow(result)
    json_path = os.path.join(scm_dir, "scm_counterfactual_case.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return {"scm_counterfactual_authority": csv_path, "scm_counterfactual_case": json_path}


__all__ = [
    "SCM_COUNTERFACTUAL_VERSION",
    "NO_ID_NO_COUNTERFACTUAL_AUTHORITY",
    "FORMAL_INDIVIDUAL_COUNTERFACTUAL_STATUS",
    "SCM_COUNTERFACTUAL_COLUMNS",
    "SCMCounterfactualResult",
    "evaluate_scm_counterfactual",
    "run_scm_counterfactual_proxy",
    "evaluate_counterfactual_audit_rows",
    "write_scm_counterfactual_audit",
    "write_scm_counterfactual_outputs",
]
