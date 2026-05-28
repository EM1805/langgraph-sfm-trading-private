from __future__ import annotations

"""Validation for explicit SCM-first JSON inputs.

The validator is intentionally stdlib-only so users can run:

    amantia scm-validate --scm-input scm_input.json

before importing pandas/numpy or running the heavier SCM/estimation pipeline.
"""

import csv
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


REQUIRED_BLOCKS = {
    "nodes",
    "edges",
    "queries",
    "assumptions",
    "structural_equations",
    "exogenous",
    "data_path",
    "safety_policy",
}

EXOGENOUS_ROLES = {"exogenous", "latent", "unobserved", "noise"}
NON_STRUCTURAL_EDGE_KINDS = {"exogenous_noise", "noise_input", "latent_noise"}


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    code: str
    message: str
    path: str = ""
    hint: str = ""

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


def _as_str(value: object) -> str:
    if value is None:
        return ""
    try:
        if value != value:  # NaN-like
            return ""
    except Exception:
        pass
    text = str(value).strip()
    return "" if text.lower() in {"none", "nan", "null", "nat"} else text


def _issue(severity: str, code: str, message: str, path: str = "", hint: str = "") -> ValidationIssue:
    return ValidationIssue(severity=severity, code=code, message=message, path=path, hint=hint)


def _node_id(raw: object) -> str:
    if not isinstance(raw, dict):
        return ""
    return _as_str(raw.get("id") or raw.get("node_id") or raw.get("name"))


def _node_role(raw: object) -> str:
    if not isinstance(raw, dict):
        return ""
    return _as_str(raw.get("role") or raw.get("node_role") or raw.get("type")).lower()


def _parse_edge(raw: object) -> Tuple[str, str, Dict[str, object]]:
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        meta = raw[2] if len(raw) >= 3 and isinstance(raw[2], dict) else {}
        return _as_str(raw[0]), _as_str(raw[1]), dict(meta)
    if isinstance(raw, dict):
        return _as_str(raw.get("source") or raw.get("from")), _as_str(raw.get("target") or raw.get("to")), dict(raw)
    return "", "", {}


def _parse_query(raw: object) -> Tuple[str, str, str]:
    if not isinstance(raw, dict):
        return "", "", ""
    qid = _as_str(raw.get("id") or raw.get("query_id"))
    treatment = _as_str(raw.get("treatment") or raw.get("treatment_col") or raw.get("source"))
    outcome = _as_str(raw.get("outcome") or raw.get("outcome_col") or raw.get("target"))
    return qid, treatment, outcome


def _resolve_data_path(scm_input_path: str | os.PathLike | None, payload: Dict[str, object], data_path: Optional[str]) -> str:
    raw = data_path or _as_str(payload.get("data_path", ""))
    if not raw:
        return ""
    p = Path(raw)
    if p.is_absolute() or p.exists():
        return str(p)
    if scm_input_path:
        beside = Path(scm_input_path).resolve().parent / p
        if beside.exists():
            return str(beside)
    return str(p)


def _read_csv_header(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            return [_as_str(c) for c in row]
    return []


def _detect_cycle(edges: Sequence[Tuple[str, str]]) -> List[str]:
    graph: Dict[str, List[str]] = {}
    for src, tgt in edges:
        graph.setdefault(src, []).append(tgt)
        graph.setdefault(tgt, [])

    visiting: set[str] = set()
    visited: set[str] = set()
    stack: List[str] = []

    def dfs(node: str) -> Optional[List[str]]:
        visiting.add(node)
        stack.append(node)
        for nxt in graph.get(node, []):
            if nxt in visiting:
                idx = stack.index(nxt) if nxt in stack else 0
                return stack[idx:] + [nxt]
            if nxt not in visited:
                found = dfs(nxt)
                if found:
                    return found
        visiting.remove(node)
        visited.add(node)
        stack.pop()
        return None

    for n in list(graph):
        if n not in visited:
            found = dfs(n)
            if found:
                return found
    return []


def validate_scm_input_payload(
    payload: object,
    *,
    scm_input_path: str | os.PathLike | None = None,
    data_path: Optional[str] = None,
    strict_data: bool = False,
) -> Dict[str, object]:
    """Validate an SCM-first JSON payload and return a JSON-serializable report."""
    issues: List[ValidationIssue] = []

    if not isinstance(payload, dict):
        issues.append(_issue("error", "payload_not_object", "SCM input must be a JSON object.", "$"))
        return _summary(payload, issues, data_path="", data_columns=[])

    missing = sorted(REQUIRED_BLOCKS - set(payload.keys()))
    for key in missing:
        issues.append(_issue("error", "missing_required_block", f"Missing required block: {key}", f"$.{key}"))

    for key in ["nodes", "edges", "queries", "assumptions"]:
        if key in payload and not isinstance(payload.get(key), list):
            issues.append(_issue("error", "block_wrong_type", f"{key} must be a list.", f"$.{key}"))
    for key in ["structural_equations", "exogenous", "safety_policy"]:
        if key in payload and not isinstance(payload.get(key), dict):
            issues.append(_issue("error", "block_wrong_type", f"{key} must be an object.", f"$.{key}"))

    nodes = payload.get("nodes", []) if isinstance(payload.get("nodes", []), list) else []
    node_roles: Dict[str, str] = {}
    duplicate_nodes: set[str] = set()
    for i, raw in enumerate(nodes):
        if not isinstance(raw, dict):
            issues.append(_issue("error", "node_not_object", "Every node must be an object.", f"$.nodes[{i}]"))
            continue
        nid = _node_id(raw)
        if not nid:
            issues.append(_issue("error", "node_missing_id", "Node is missing id/node_id/name.", f"$.nodes[{i}]"))
            continue
        if nid in node_roles:
            duplicate_nodes.add(nid)
            issues.append(_issue("error", "duplicate_node_id", f"Duplicate node id: {nid}", f"$.nodes[{i}]"))
        role = _node_role(raw)
        if not role:
            issues.append(_issue("warning", "node_missing_role", f"Node has no role: {nid}", f"$.nodes[{i}].role"))
        node_roles[nid] = role

    exogenous = payload.get("exogenous", {}) if isinstance(payload.get("exogenous", {}), dict) else {}
    exog_ids = {_as_str(k) for k in exogenous.keys() if _as_str(k)}
    declared_ids = set(node_roles) | exog_ids

    edges = payload.get("edges", []) if isinstance(payload.get("edges", []), list) else []
    seen_edges: set[Tuple[str, str]] = set()
    structural_edges: List[Tuple[str, str]] = []
    edge_count = 0
    for i, raw in enumerate(edges):
        src, tgt, meta = _parse_edge(raw)
        if not src or not tgt:
            issues.append(_issue("error", "edge_missing_endpoint", "Edge must define source/from and target/to.", f"$.edges[{i}]"))
            continue
        edge_count += 1
        if src == tgt:
            issues.append(_issue("error", "self_loop_edge", f"Self-loop edge is not supported: {src} -> {tgt}", f"$.edges[{i}]"))
        if (src, tgt) in seen_edges:
            issues.append(_issue("warning", "duplicate_edge", f"Duplicate edge: {src} -> {tgt}", f"$.edges[{i}]"))
        seen_edges.add((src, tgt))

        src_is_exog = node_roles.get(src, "") in EXOGENOUS_ROLES or src in exog_ids
        edge_kind = _as_str(meta.get("edge_kind") or meta.get("edge_type")).lower()
        is_noise_edge = src_is_exog or edge_kind in NON_STRUCTURAL_EDGE_KINDS
        if src not in declared_ids:
            issues.append(_issue("error", "edge_source_unknown", f"Edge source is not declared as node/exogenous: {src}", f"$.edges[{i}][0]"))
        if tgt not in declared_ids:
            issues.append(_issue("error", "edge_target_unknown", f"Edge target is not declared as node/exogenous: {tgt}", f"$.edges[{i}][1]"))
        if tgt in exog_ids or node_roles.get(tgt, "") in EXOGENOUS_ROLES:
            issues.append(_issue("error", "edge_targets_exogenous", f"Structural edge cannot target exogenous/noise node: {tgt}", f"$.edges[{i}]"))
        if not is_noise_edge:
            structural_edges.append((src, tgt))

    if not edges:
        issues.append(_issue("error", "no_edges", "SCM input must contain at least one edge.", "$.edges"))

    cycle = _detect_cycle(structural_edges)
    if cycle:
        issues.append(_issue("error", "cycle_detected", "Structural graph contains a directed cycle: " + " -> ".join(cycle), "$.edges", "Use explicit time-lagged variables if the relationship is dynamic."))

    queries = payload.get("queries", []) if isinstance(payload.get("queries", []), list) else []
    query_ids: List[str] = []
    seen_queries: set[Tuple[str, str]] = set()
    for i, raw in enumerate(queries):
        if not isinstance(raw, dict):
            issues.append(_issue("error", "query_not_object", "Every query must be an object.", f"$.queries[{i}]"))
            continue
        qid, treatment, outcome = _parse_query(raw)
        if not qid:
            qid = f"query_{i + 1:04d}"
        query_ids.append(qid)
        if not treatment or not outcome:
            issues.append(_issue("error", "query_missing_endpoint", "Query must define treatment/source and outcome/target.", f"$.queries[{i}]"))
            continue
        if treatment == outcome:
            issues.append(_issue("error", "query_self_effect", f"Query treatment and outcome are identical: {treatment}", f"$.queries[{i}]"))
        if (treatment, outcome) in seen_queries:
            issues.append(_issue("warning", "duplicate_query", f"Duplicate query: {treatment} -> {outcome}", f"$.queries[{i}]"))
        seen_queries.add((treatment, outcome))
        if treatment not in declared_ids:
            issues.append(_issue("error", "query_treatment_unknown", f"Query treatment is not declared: {treatment}", f"$.queries[{i}].treatment"))
        if outcome not in declared_ids:
            issues.append(_issue("error", "query_outcome_unknown", f"Query outcome is not declared: {outcome}", f"$.queries[{i}].outcome"))
        if treatment in exog_ids or node_roles.get(treatment, "") in EXOGENOUS_ROLES:
            issues.append(_issue("error", "query_treatment_exogenous", f"Query treatment cannot be exogenous/noise: {treatment}", f"$.queries[{i}].treatment"))

    if not queries:
        issues.append(_issue("error", "no_queries", "SCM input must contain at least one causal query.", "$.queries"))

    assumptions = payload.get("assumptions", []) if isinstance(payload.get("assumptions", []), list) else []
    if assumptions:
        has_required_assumption = any(isinstance(a, dict) and _as_str(a.get("status")).lower() == "required" for a in assumptions)
        if not has_required_assumption:
            issues.append(_issue("warning", "no_required_assumptions", "No assumption is marked as required.", "$.assumptions"))

    structural_equations = payload.get("structural_equations", {}) if isinstance(payload.get("structural_equations", {}), dict) else {}
    for eq_node in structural_equations.keys():
        if _as_str(eq_node) not in declared_ids:
            issues.append(_issue("warning", "equation_for_unknown_node", f"Structural equation references undeclared node: {eq_node}", "$.structural_equations"))
    non_exog_nodes = [nid for nid, role in node_roles.items() if role not in EXOGENOUS_ROLES]
    for nid in non_exog_nodes:
        if nid not in structural_equations:
            issues.append(_issue("warning", "missing_structural_equation", f"No structural equation declared for node: {nid}", f"$.structural_equations.{nid}"))

    safety_policy = payload.get("safety_policy", {}) if isinstance(payload.get("safety_policy", {}), dict) else {}
    if safety_policy.get("require_identification") is not True:
        issues.append(_issue("error", "require_identification_not_true", "safety_policy.require_identification must be true for SCM-first safety runs.", "$.safety_policy.require_identification"))

    resolved_data_path = _resolve_data_path(scm_input_path, payload, data_path)
    data_columns: List[str] = []
    missing_data_nodes: List[str] = []
    if resolved_data_path:
        if Path(resolved_data_path).exists():
            try:
                data_columns = _read_csv_header(resolved_data_path)
                data_set = set(data_columns)
                for raw in nodes:
                    if not isinstance(raw, dict):
                        continue
                    nid = _node_id(raw)
                    role = _node_role(raw)
                    observed = bool(raw.get("observed", True))
                    if observed and role not in EXOGENOUS_ROLES and nid and nid not in data_set:
                        missing_data_nodes.append(nid)
                if missing_data_nodes:
                    severity = "error" if strict_data else "warning"
                    issues.append(_issue(severity, "observed_node_missing_from_data", "Observed SCM nodes missing from data columns: " + ", ".join(sorted(missing_data_nodes)), "$.nodes", "Pass the matching --data file or set observed=false for latent/template-only nodes."))
            except (OSError, csv.Error, UnicodeDecodeError) as exc:
                severity = "error" if strict_data else "warning"
                issues.append(_issue(severity, "data_header_unreadable", f"Could not read data header: {type(exc).__name__}: {exc}", "$.data_path"))
        else:
            severity = "error" if strict_data else "warning"
            issues.append(_issue(severity, "data_path_not_found", f"Data path does not exist: {resolved_data_path}", "$.data_path"))

    return _summary(
        payload,
        issues,
        data_path=resolved_data_path,
        data_columns=data_columns,
        node_ids=sorted(node_roles),
        exogenous_ids=sorted(exog_ids),
        query_ids=query_ids,
        edge_count=edge_count,
        missing_observed_data_columns=sorted(set(missing_data_nodes)),
    )


def _summary(
    payload: object,
    issues: Sequence[ValidationIssue],
    *,
    data_path: str,
    data_columns: Sequence[str],
    node_ids: Sequence[str] = (),
    exogenous_ids: Sequence[str] = (),
    query_ids: Sequence[str] = (),
    edge_count: int = 0,
    missing_observed_data_columns: Sequence[str] = (),
) -> Dict[str, object]:
    errors = [i for i in issues if i.severity == "error"]
    warnings = [i for i in issues if i.severity == "warning"]
    return {
        "status": "ok" if not errors else "failed",
        "ok": not errors,
        "schema_version": _as_str(payload.get("schema_version", "")) if isinstance(payload, dict) else "",
        "name": _as_str(payload.get("name", "")) if isinstance(payload, dict) else "",
        "errors_count": len(errors),
        "warnings_count": len(warnings),
        "node_count": len(node_ids),
        "edge_count": edge_count,
        "query_count": len(query_ids),
        "exogenous_count": len(exogenous_ids),
        "node_ids": list(node_ids),
        "query_ids": list(query_ids),
        "data_path": data_path,
        "data_columns_checked": list(data_columns),
        "missing_observed_data_columns": list(missing_observed_data_columns),
        "issues": [i.to_dict() for i in issues],
    }


def validate_scm_input_file(
    scm_input_path: str | os.PathLike,
    *,
    data_path: Optional[str] = None,
    strict_data: bool = False,
) -> Dict[str, object]:
    path = Path(scm_input_path)
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except FileNotFoundError:
        issue = _issue("error", "file_not_found", f"SCM input file not found: {path}", "$")
        return _summary({}, [issue], data_path=str(data_path or ""), data_columns=[])
    except json.JSONDecodeError as exc:
        issue = _issue("error", "invalid_json", f"Invalid JSON: {exc}", "$")
        return _summary({}, [issue], data_path=str(data_path or ""), data_columns=[])
    return validate_scm_input_payload(payload, scm_input_path=path, data_path=data_path, strict_data=strict_data)


def raise_if_invalid_scm_input(
    payload: object,
    *,
    scm_input_path: str | os.PathLike | None = None,
    data_path: Optional[str] = None,
    strict_data: bool = False,
) -> Dict[str, object]:
    report = validate_scm_input_payload(payload, scm_input_path=scm_input_path, data_path=data_path, strict_data=strict_data)
    if not report.get("ok"):
        problems = []
        for issue in report.get("issues", []):
            if issue.get("severity") == "error":
                loc = issue.get("path") or "$"
                problems.append(f"{issue.get('code')}: {issue.get('message')} ({loc})")
        raise ValueError("SCM input validation failed: " + "; ".join(problems[:8]))
    return report
