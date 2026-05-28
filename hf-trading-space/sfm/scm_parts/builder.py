from __future__ import annotations

from runtime_env import configure_scientific_runtime
configure_scientific_runtime()


"""SCM builder on top of Discovery outputs.

Builds a data-aware structural causal model registry from Discovery/PCMCI-style
links and bridge metadata. The goal remains pragmatic rather than claiming a
fully identified Pearl SCM, but this version is closer to a fitted SCM layer:
- node roles come from offline prior graph/bridge/discovery plus empirical data hints
- structural families are inferred from variable type + support + role
- artifacts expose variable family, support, and noise assumptions explicitly
"""

import json
import os
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from ._utils import (
    SOURCE_FIELD_CANDIDATES,
    TARGET_FIELD_CANDIDATES,
    edge_endpoints_from_row,
)
from ._io import load_data as _io_load_data, load_dag as _io_load_dag, read_discovery_frames
from .edge_authority import build_scm_edge_row, dedupe_edges_by_authority
from .admg import admg_report_from_scm_graph
from .id_algorithm import id_algorithm_summary, id_audit_rows_from_scm_graph
from .input_validator import raise_if_invalid_scm_input, validate_scm_input_file
from .symbolic_evaluator import (
    SYMBOLIC_EVALUATION_COLUMNS,
    evaluate_id_audit_rows,
    symbolic_evaluation_summary,
)

from runtime_compat import assert_scientific_stack
assert_scientific_stack()

import numpy as np
import pandas as pd


SCM_DIRNAME = "scm"


def _as_str(x) -> str:
    """Return a graph-safe string; drop NaN/null sentinels before node creation."""
    if x is None:
        return ""
    try:
        if x != x:  # NaN without importing math/pandas
            return ""
    except Exception:
        pass
    text = str(x).strip()
    return "" if text.lower() in {"nan", "none", "null", "nat"} else text


def _safe_float(x, default=0.0) -> float:
    try:
        v = float(x)
        return v if v == v else float(default)
    except (TypeError, ValueError, OverflowError):
        return float(default)



def _first_present(row, names, default=""):
    """Return the first non-empty value from a row-like object."""
    for name in names:
        try:
            val = row.get(name, "")
        except (TypeError, ValueError, AttributeError):
            val = ""
        text = _as_str(val)
        if text:
            return text
    return default


def _normalize_edge_contract(df):
    """Normalize legacy and agentic edge columns to a common SCM contract."""
    df = df.copy() if df is not None else pd.DataFrame()
    if df.empty:
        return df
    if "source" not in df.columns:
        for c in SOURCE_FIELD_CANDIDATES:
            if c in df.columns:
                df["source"] = df[c]
                break
    if "target" not in df.columns:
        for c in TARGET_FIELD_CANDIDATES:
            if c in df.columns:
                df["target"] = df[c]
                break
    if "treatment_col" not in df.columns and "source" in df.columns:
        df["treatment_col"] = df["source"]
    if "outcome_col" not in df.columns and "target" in df.columns:
        df["outcome_col"] = df["target"]
    return df

def _split_pipe(value) -> List[str]:
    raw = _as_str(value)
    if not raw:
        return []
    cleaned: List[str] = []
    for part in raw.split("|"):
        item = _as_str(part)
        if item:
            cleaned.append(item)
    return cleaned


def _dedupe(seq: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in seq:
        item = _as_str(item).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out



def _looks_binary(name: str) -> bool:
    s = name.lower()
    return any(k in s for k in ["harm", "incident", "failure", "rollback", "unsafe", "alert", "approved", "blocked"])


def _looks_count(name: str) -> bool:
    s = name.lower()
    return any(k in s for k in ["count", "n_", "num_", "events", "errors", "alerts"])


def _looks_rate(name: str) -> bool:
    s = name.lower()
    return any(k in s for k in ["rate", "ratio", "share", "prob", "likelihood"])


def _role_from_name(name: str) -> str:
    s = name.lower()
    if any(k in s for k in ["action", "policy", "intervention", "treatment", "override", "approval", "guardrail", "review"]):
        return "action"
    if any(k in s for k in ["harm", "incident", "rollback", "failure", "unsafe", "target", "outcome", "risk"]):
        return "outcome"
    if any(k in s for k in ["mediat", "latency", "queue", "retry", "memory", "blast_radius", "recovery"]):
        return "mediator"
    if any(k in s for k in ["context", "load", "traffic", "season", "time", "weekday", "dow", "trend"]):
        return "context"
    return "state"


def _load_data(data_path: Optional[str], out_dir: str) -> pd.DataFrame:
    """Compatibility wrapper; disk I/O lives in scm_parts._io."""
    return _io_load_data(data_path, out_dir)


def _infer_variable_profile(series: pd.Series) -> Dict[str, object]:
    s = series.dropna()
    out = {
        "dtype_family": "unknown",
        "support_min": None,
        "support_max": None,
        "n_unique": int(s.nunique(dropna=True)) if len(s) else 0,
        "missing_frac": float(series.isna().mean()) if len(series) else 1.0,
        "value_family_source": "data",
    }
    if len(s) == 0:
        return out
    if pd.api.types.is_bool_dtype(series):
        out["dtype_family"] = "binary"
        out["support_min"] = 0.0
        out["support_max"] = 1.0
        return out

    if pd.api.types.is_numeric_dtype(series):
        sn = pd.to_numeric(s, errors="coerce").dropna()
        if len(sn) == 0:
            out["dtype_family"] = "unknown"
            return out
        uniq = sorted(set(float(v) for v in sn.unique().tolist()))
        out["support_min"] = float(np.min(sn))
        out["support_max"] = float(np.max(sn))
        if len(uniq) <= 2 and set(uniq).issubset({0.0, 1.0}):
            out["dtype_family"] = "binary"
        elif all(abs(v - round(v)) < 1e-9 for v in uniq) and min(uniq) >= 0.0 and max(uniq) >= 2.0:
            out["dtype_family"] = "count"
        elif min(uniq) >= 0.0 and max(uniq) <= 1.0 and len(uniq) >= 8:
            out["dtype_family"] = "rate"
        else:
            out["dtype_family"] = "continuous"
        return out

    out["dtype_family"] = "categorical"
    return out


def _merge_profile(name: str, role: str, df: pd.DataFrame) -> Dict[str, object]:
    if df is None or df.empty or name not in df.columns:
        # fallback to lexical hints when data is missing
        if _looks_binary(name):
            return {"dtype_family": "binary", "support_min": 0.0, "support_max": 1.0, "n_unique": 2, "missing_frac": 1.0, "value_family_source": "heuristic"}
        if _looks_count(name):
            return {"dtype_family": "count", "support_min": 0.0, "support_max": None, "n_unique": 0, "missing_frac": 1.0, "value_family_source": "heuristic"}
        if _looks_rate(name):
            return {"dtype_family": "rate", "support_min": 0.0, "support_max": 1.0, "n_unique": 0, "missing_frac": 1.0, "value_family_source": "heuristic"}
        return {"dtype_family": "continuous", "support_min": None, "support_max": None, "n_unique": 0, "missing_frac": 1.0, "value_family_source": "heuristic"}
    prof = _infer_variable_profile(df[name])
    if role in {"action", "guardrail"} and prof.get("dtype_family") == "continuous":
        # preserve action semantics even when logged numerically
        prof["dtype_family"] = "policy_signal"
    return prof


def _structural_family(node_name: str, node_role: str, profile: Dict[str, object]) -> Tuple[str, str, str]:
    dtype_family = _as_str(profile.get("dtype_family", "continuous")) or "continuous"
    if dtype_family == "binary":
        return "binary_structural", "sigmoid(sum(parent_effects) + U)", "bernoulli_logit_noise"
    if dtype_family == "count":
        return "count_structural", "exp(sum(parent_effects) + U)", "poisson_like_noise"
    if dtype_family == "rate":
        return "rate_structural", "clip(sigmoid(sum(parent_effects) + U), 0, 1)", "bounded_logit_noise"
    if dtype_family == "categorical":
        return "categorical_structural", "softmax(parent_effects + U)", "categorical_noise"
    if node_role in {"action", "guardrail"} or dtype_family == "policy_signal":
        return "policy_structural", "policy(context, state, U)", "decision_noise"
    return "additive_structural", "sum(parent_effects) + U", "gaussian_like_noise"


def _load_dag(dag_path: Optional[str], out_dir: str):
    """Compatibility wrapper; optional offline prior graph loading lives in scm_parts._io."""
    return _io_load_dag(dag_path, out_dir)


def _role_from_dag(dag, node_name: str) -> Optional[Dict[str, object]]:
    if dag is None:
        return None
    node = dag.get_node(node_name)
    if node is None:
        return None
    node_type = _as_str(getattr(node, "node_type", ""))
    role = "state"
    if node_type in {"action", "decision"}:
        role = "action"
    elif node_type in {"guardrail", "review_gate"}:
        role = "guardrail"
    elif node_type in {"risk_outcome", "sensitive_risk", "outcome"}:
        role = "outcome"
    elif node_type in {"mediator", "mechanism"}:
        role = "mediator"
    elif node_type in {"context", "confounder", "environment", "covariate", "control"}:
        role = "context"
    elif node_type in {"policy", "constraint", "invariant"}:
        role = "guardrail"
    elif node_type in {"exogenous", "latent", "unobserved", "noise"}:
        role = "exogenous"
    elif node_type in {"treatment", "treatment_modifier", "intervention"}:
        role = "action"
    elif node_type in {"secondary_outcome", "downstream_outcome", "business_outcome"}:
        role = "outcome"
    return {
        "node_role": role,
        "observed": bool(getattr(node, "observed", True)),
        "intervenable": bool(getattr(node, "intervenable", False)),
        "adjustable": bool(getattr(node, "adjustable", False)),
        "sensitive": bool(getattr(node, "sensitive", False)),
        "time_role": _as_str(getattr(node, "time_role", "concurrent")) or "concurrent",
        "role_source": "offline_prior",
    }


class _SCMInputNode:
    def __init__(self, node_id: str, meta: Dict[str, object]):
        self.node_id = node_id
        self.node_type = _as_str(meta.get("role") or meta.get("node_role") or meta.get("type") or "state")
        self.observed = bool(meta.get("observed", True))
        self.intervenable = bool(meta.get("intervenable", self.node_type in {"treatment", "action", "intervention"}))
        self.adjustable = bool(meta.get("adjustable", self.node_type in {"covariate", "context", "confounder", "control"}))
        self.sensitive = bool(meta.get("sensitive", False))
        self.time_role = _as_str(meta.get("time_role", "concurrent")) or "concurrent"


class _SCMInputDAG:
    """Minimal get_node adapter so SCM-first templates reuse the data-aware builder."""

    def __init__(self, nodes: Iterable[Dict[str, object]]):
        self._nodes: Dict[str, _SCMInputNode] = {}
        for raw in nodes or []:
            if not isinstance(raw, dict):
                continue
            node_id = _as_str(raw.get("node_id") or raw.get("id") or raw.get("name")).strip()
            if node_id:
                self._nodes[node_id] = _SCMInputNode(node_id, raw)

    def get_node(self, node_name: str):
        return self._nodes.get(_as_str(node_name).strip())


def _load_scm_input_payload(scm_input_path: str | os.PathLike, data_path: Optional[str] = None) -> Dict[str, object]:
    path = Path(scm_input_path)
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    raise_if_invalid_scm_input(payload, scm_input_path=path, data_path=data_path, strict_data=False)
    if not isinstance(payload, dict):
        raise ValueError("SCM input must be a JSON object")
    return payload


def _resolve_scm_input_data_path(scm_input_path: str | os.PathLike, payload: Dict[str, object], data_path: Optional[str]) -> Optional[str]:
    raw = data_path or _as_str(payload.get("data_path", ""))
    if not raw:
        return data_path
    p = Path(raw)
    if p.is_absolute() or p.exists():
        return str(p)
    beside_input = Path(scm_input_path).resolve().parent / p
    if beside_input.exists():
        return str(beside_input)
    return str(p)


def _normalize_scm_input_edges(payload: Dict[str, object], scm_input_path: str | os.PathLike) -> pd.DataFrame:
    node_roles: Dict[str, str] = {}
    for raw in payload.get("nodes", []) or []:
        if isinstance(raw, dict):
            node_id = _as_str(raw.get("id") or raw.get("node_id") or raw.get("name")).strip()
            if node_id:
                node_roles[node_id] = _as_str(raw.get("role") or raw.get("node_role") or raw.get("type")).strip().lower()
    rows: List[Dict[str, object]] = []
    for idx, raw in enumerate(payload.get("edges", []) or [], start=1):
        if isinstance(raw, (list, tuple)) and len(raw) >= 2:
            src, tgt = raw[0], raw[1]
            meta = raw[2] if len(raw) >= 3 and isinstance(raw[2], dict) else {}
        elif isinstance(raw, dict):
            src = raw.get("source", raw.get("from"))
            tgt = raw.get("target", raw.get("to"))
            meta = raw
        else:
            continue
        src = _as_str(src).strip()
        tgt = _as_str(tgt).strip()
        if not src or not tgt:
            continue
        src_role = node_roles.get(src, "")
        tgt_role = node_roles.get(tgt, "")
        edge_kind = _as_str(meta.get("edge_kind") or meta.get("edge_type") or "").strip()
        if not edge_kind:
            edge_kind = "exogenous_noise" if src_role in {"exogenous", "latent", "unobserved", "noise"} else "domain_structural_prior"
        eligible = edge_kind not in {"exogenous_noise", "noise_input", "latent_noise"}
        rows.append({
            "source": src,
            "target": tgt,
            "treatment_col": src,
            "outcome_col": tgt,
            "lag": int(_safe_float(meta.get("lag", 0), 0)),
            "edge_kind": edge_kind,
            "scm_input_edge_kind": edge_kind,
            "__source_artifact": str(scm_input_path),
            "bridge_version": _as_str(payload.get("schema_version", "amantia.scm_input.v1")) or "amantia.scm_input.v1",
            "insight_id": _as_str(meta.get("id") or meta.get("edge_id") or f"scm_input_edge_{idx:04d}"),
            "confidence_tier": _as_str(meta.get("confidence_tier") or "domain_prior"),
            "domain_review_status": _as_str(meta.get("domain_review_status") or "required"),
            "candidate_covariates": _as_str(meta.get("candidate_covariates") or meta.get("adjustment_set") or ""),
            "conditioning_set_used": _as_str(meta.get("conditioning_set_used") or meta.get("adjustment_set") or ""),
            "post_treatment_columns": _as_str(meta.get("post_treatment_columns") or meta.get("mediators") or ""),
            "forbidden_adjustment_set": _as_str(meta.get("forbidden_adjustment_set") or ""),
            "scm_role_hint": "exogenous_noise_not_do_query" if not eligible else "domain_reviewed_prior",
        })
    return _normalize_edge_contract(pd.DataFrame(rows))


def _normalize_scm_input_queries(payload: Dict[str, object], scm_input_path: str | os.PathLike) -> List[Dict[str, object]]:
    queries: List[Dict[str, object]] = []
    for idx, raw in enumerate(payload.get("queries", []) or [], start=1):
        if not isinstance(raw, dict):
            continue
        treatment = _as_str(raw.get("treatment") or raw.get("treatment_col") or raw.get("source")).strip()
        outcome = _as_str(raw.get("outcome") or raw.get("outcome_col") or raw.get("target")).strip()
        if not treatment or not outcome:
            continue
        q = dict(raw)
        q.setdefault("id", f"scm_input_query_{idx:04d}")
        q["treatment"] = treatment
        q["outcome"] = outcome
        q.setdefault("type", "total_effect")
        q.setdefault("edge_authority_level", "domain_scm_query")
        q.setdefault("edge_source_artifact", str(scm_input_path))
        queries.append(q)
    return queries


def _rewrite_scm_input_augmented_artifacts(paths: Dict[str, str], payload: Dict[str, object], scm_input_path: str | os.PathLike, validation_report: Optional[Dict[str, object]] = None) -> None:
    graph_path = paths.get("scm_graph_json")
    if not graph_path or not os.path.exists(graph_path):
        return
    with open(graph_path, "r", encoding="utf-8") as f:
        scm_graph = json.load(f)
    scm_graph["scm_version"] = max(int(scm_graph.get("scm_version", 0) or 0), 6)
    scm_graph["type"] = "scm_input_domain_prior"
    scm_graph["scm_input"] = {
        "schema_version": _as_str(payload.get("schema_version", "amantia.scm_input.v1")),
        "name": _as_str(payload.get("name", "")),
        "source_path": str(scm_input_path),
        "discovery_policy": _as_str(payload.get("discovery_policy", "optional_suggestion_only")),
    }
    scm_graph["queries"] = _normalize_scm_input_queries(payload, scm_input_path)
    if validation_report is not None:
        scm_graph["scm_input_validation"] = validation_report
    scm_graph["declared_assumptions"] = payload.get("assumptions", [])
    scm_graph["declared_structural_equations"] = payload.get("structural_equations", {})
    scm_graph["declared_exogenous"] = payload.get("exogenous", {})
    scm_graph["safety_policy"] = payload.get("safety_policy", {})
    scm_graph.setdefault("assumptions", {})["scm_input_policy"] = "Explicit SCM input is a domain prior; identification and causal_contract.csv still gate estimation authority."

    _admg, admg_summary, admg_components = admg_report_from_scm_graph(scm_graph)
    id_audit_rows = id_audit_rows_from_scm_graph(scm_graph)
    id_summary = id_algorithm_summary(_admg, id_audit_rows)
    symbolic_evaluation_rows = evaluate_id_audit_rows(id_audit_rows)
    symbolic_eval_summary = symbolic_evaluation_summary(symbolic_evaluation_rows)
    scm_graph["admg_summary"] = admg_summary
    scm_graph["c_components"] = admg_components
    scm_graph["id_algorithm_summary"] = id_summary
    scm_graph["id_algorithm_audit"] = id_audit_rows
    scm_graph["symbolic_evaluation_summary"] = symbolic_eval_summary

    with open(paths["scm_graph_json"], "w", encoding="utf-8") as f:
        json.dump(scm_graph, f, ensure_ascii=False, indent=2)
    pd.DataFrame(admg_components).to_csv(paths["admg_c_components_csv"], index=False)
    pd.DataFrame(id_audit_rows).to_csv(paths["id_algorithm_audit_csv"], index=False)
    pd.DataFrame(symbolic_evaluation_rows, columns=SYMBOLIC_EVALUATION_COLUMNS).to_csv(paths["symbolic_evaluation_csv"], index=False)
    with open(paths["symbolic_evaluation_summary_json"], "w", encoding="utf-8") as f:
        json.dump(symbolic_eval_summary, f, ensure_ascii=False, indent=2)
    if validation_report is not None:
        validation_path = str(Path(paths["scm_graph_json"]).with_name("scm_input_validation.json"))
        with open(validation_path, "w", encoding="utf-8") as f:
            json.dump(validation_report, f, ensure_ascii=False, indent=2)
        paths["scm_input_validation_json"] = validation_path
    manifest_path = paths.get("manifest")
    if manifest_path and os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        manifest.update({
            "scm_type": "scm_input_domain_prior",
            "scm_version": 6,
            "scm_input_path": str(scm_input_path),
            "scm_input_name": _as_str(payload.get("name", "")),
            "discovery_policy": _as_str(payload.get("discovery_policy", "optional_suggestion_only")),
            "n_queries": int(len(scm_graph.get("queries", []))),
            "admg_summary": admg_summary,
            "id_algorithm_summary": id_summary,
            "symbolic_evaluation_summary": symbolic_eval_summary,
        })
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)


def build_from_scm_input(scm_input_path: str, out_dir: str = "out", data_path: Optional[str] = None) -> Dict[str, str]:
    """Build SCM assets from an explicit SCM-first JSON contract.

    Discovery remains optional. The explicit graph/query template becomes the
    structural prior, while ID, symbolic evaluation, causal_contract.csv and
    estimation gates remain conservative downstream authorities.
    """
    payload = _load_scm_input_payload(scm_input_path, data_path=data_path)
    validation_report = validate_scm_input_file(scm_input_path, data_path=data_path, strict_data=False)
    resolved_data_path = _resolve_scm_input_data_path(scm_input_path, payload, data_path)
    dag = _SCMInputDAG(payload.get("nodes", []) or [])
    edges = _normalize_scm_input_edges(payload, scm_input_path)
    data = _load_data(resolved_data_path, out_dir)
    paths = write_scm_assets(
        proposals=edges,
        insights=pd.DataFrame(columns=edges.columns),
        out_dir=out_dir,
        bridge=edges,
        dag_path=None,
        data_path=resolved_data_path,
        dag_obj=dag,
    )
    _rewrite_scm_input_augmented_artifacts(paths, payload, scm_input_path, validation_report=validation_report)
    try:
        from contracts.causal_contract import write_causal_contract
        paths.update(write_causal_contract(out_dir=out_dir))
    except (OSError, ValueError, TypeError, RuntimeError, KeyError, ImportError) as exc:
        warnings.warn(f"[amantia][warning] causal contract sync after scm-input build failed: {type(exc).__name__}: {exc}", RuntimeWarning)
    return paths


def build_scm_assets(
    proposals: pd.DataFrame,
    insights: pd.DataFrame,
    bridge: Optional[pd.DataFrame] = None,
    dag=None,
    data: Optional[pd.DataFrame] = None,
) -> Dict[str, pd.DataFrame | dict]:
    proposals = _normalize_edge_contract(proposals.copy() if proposals is not None else pd.DataFrame())
    insights = _normalize_edge_contract(insights.copy() if insights is not None else pd.DataFrame())
    bridge = _normalize_edge_contract(bridge.copy() if bridge is not None else pd.DataFrame())
    data = data.copy() if data is not None else pd.DataFrame()

    if len(proposals) > 0:
        kept_edges = proposals[proposals.get("discovery_track", "") != "dropped"].copy() if "discovery_track" in proposals.columns else proposals.copy()
    else:
        kept_edges = pd.DataFrame(columns=["source", "target", "lag"])

    nodes: Dict[str, Dict[str, object]] = {}
    data_columns = set(str(c) for c in getattr(data, "columns", []))

    def _preferred_role(name: str, fallback: str) -> str:
        dag_info = _role_from_dag(dag, name)
        if dag_info and _as_str(dag_info.get("node_role", "")):
            return _as_str(dag_info.get("node_role"))
        return fallback

    def ensure_node(name: str, updates: Optional[Dict[str, object]] = None) -> None:
        name = _as_str(name).strip()
        if not name:
            return
        data_column_present = name in data_columns
        default_observation_status = "observed_in_data" if data_column_present else "not_observed_in_current_data"
        base = nodes.get(name, {
            "node_id": name,
            "node_role": _role_from_name(name),
            "observed": data_column_present,
            "data_column_present": data_column_present,
            "declared_observed_in_graph": None,
            "observation_status": default_observation_status,
            "fit_eligible": data_column_present,
            "intervenable": False,
            "adjustable": False,
            "sensitive": False,
            "time_role": "lagged_or_state",
            "role_source": "heuristic",
        })
        # Refresh observability from the current dataset on every call. An offline prior can
        # declare a node observed, but if the column is absent in this run the node
        # is not fit-eligible and must be explicit in the audit artifacts.
        base["data_column_present"] = data_column_present
        dag_info = _role_from_dag(dag, name)
        if dag_info:
            declared = bool(dag_info.get("observed", True))
            dag_info = dict(dag_info)
            dag_info.pop("observed", None)
            base.update(dag_info)
            base["declared_observed_in_graph"] = declared
        if updates:
            for k, v in updates.items():
                if v is None or v == "":
                    continue
                if k == "observed":
                    base["declared_observed_in_graph"] = bool(v)
                elif k in {"intervenable", "adjustable", "sensitive"}:
                    base[k] = bool(v)
                else:
                    base[k] = v
        declared = base.get("declared_observed_in_graph")
        if data_column_present:
            base["observed"] = True
            base["fit_eligible"] = True
            base["observation_status"] = "observed_in_data"
        elif declared is True:
            base["observed"] = False
            base["fit_eligible"] = False
            base["observation_status"] = "declared_observed_but_missing_from_data"
        elif declared is False:
            base["observed"] = False
            base["fit_eligible"] = False
            base["observation_status"] = "latent_or_conceptual_not_observed"
        else:
            base["observed"] = False
            base["fit_eligible"] = False
            base["observation_status"] = default_observation_status
        if base.get("node_role") in {"action", "guardrail"}:
            base["intervenable"] = True
        prof = _merge_profile(name, _as_str(base.get("node_role", "state")), data)
        base.update({
            "dtype_family": prof.get("dtype_family", "unknown"),
            "support_min": prof.get("support_min", None),
            "support_max": prof.get("support_max", None),
            "n_unique": prof.get("n_unique", 0),
            "missing_frac": prof.get("missing_frac", 1.0),
            "value_family_source": prof.get("value_family_source", "data"),
        })
        if not data_column_present and _as_str(base.get("value_family_source")) == "data":
            base["value_family_source"] = "not_in_current_data"
        nodes[name] = base

    for _, row in bridge.iterrows() if len(bridge) > 0 else []:
        tr, out = edge_endpoints_from_row(row)
        tk = _as_str(row.get("treatment_kind", ""))
        ensure_node(tr, {
            "node_role": _preferred_role(tr, "guardrail" if tk == "guardrail" else ("action" if tk in {"decision", "action"} else _role_from_name(tr))),
            "role_source": "bridge",
        })
        ensure_node(out, {"node_role": _preferred_role(out, "outcome"), "role_source": "bridge"})
        for c in _dedupe(_split_pipe(row.get("candidate_covariates", "")) + _split_pipe(row.get("parent_set", "")) + _split_pipe(row.get("pc1_parent_set_all", ""))):
            ensure_node(c, {"node_role": "context", "adjustable": True, "role_source": "bridge_or_pc1_parent"})
        for c in _dedupe(_split_pipe(row.get("conditioning_set_used", "")) + _split_pipe(row.get("mci_conditioning_set_used", ""))):
            ensure_node(c, {"node_role": "context", "adjustable": True, "role_source": "mci_conditioning_set"})
        for c in _split_pipe(row.get("post_treatment_columns", "")):
            ensure_node(c, {"node_role": "mediator", "adjustable": False, "role_source": "bridge_post_treatment"})

    for frame in [kept_edges, insights]:
        if len(frame) == 0:
            continue
        for _, row in frame.iterrows():
            src, tgt = edge_endpoints_from_row(row)
            ensure_node(src, {"node_role": _preferred_role(src, _role_from_name(src)) if src else None})
            ensure_node(tgt, {"node_role": _preferred_role(tgt, "outcome") if tgt else None})
            for c in _dedupe(_split_pipe(row.get("pc1_parent_set", "")) + _split_pipe(row.get("pc1_parent_set_all", "")) + _split_pipe(row.get("parent_set", ""))):
                ensure_node(c, {"node_role": "context", "adjustable": True, "role_source": "pc1_parent"})
            for c in _split_pipe(row.get("mci_conditioning_set_used", row.get("conditioning_set_used", ""))):
                ensure_node(c, {"node_role": "context", "adjustable": True, "role_source": "mci_conditioning_set"})
            for c in _split_pipe(row.get("mediators", row.get("mediator_candidates", row.get("post_treatment_columns", "")))):
                ensure_node(c, {"node_role": "mediator", "role_source": "insight_mediator"})
            for c in _split_pipe(row.get("forbidden_adjustment_set", "")):
                ensure_node(c, {"node_role": "mediator", "adjustable": False, "role_source": "forbidden_adjustment"})

    node_rows = []
    family_rows = []
    for name, meta in sorted(nodes.items()):
        fam, eq, noise_family = _structural_family(name, _as_str(meta.get("node_role", "state")), meta)
        node_rows.append(meta)
        family_rows.append({
            "node_id": name,
            "node_role": meta.get("node_role", "state"),
            "dtype_family": meta.get("dtype_family", "unknown"),
            "structural_family": fam,
            "structural_equation_template": f"{name}_t := {eq}",
            "noise_symbol": f"U_{name}",
            "noise_family": noise_family,
            "family_source": "data_aware_scm_builder",
        })
    node_df = pd.DataFrame(node_rows)
    family_df = pd.DataFrame(family_rows)

    edge_rows: List[Dict[str, object]] = []
    # Authority order is explicit: bridge candidates outrank ranked insights,
    # and both outrank raw Discovery seeds. No SCM edge is estimation-authorized
    # until the causal contract says so.
    for default_origin, frame in [
        ("out/discovery_estimation_bridge.csv", bridge),
        ("out/insights_level2.csv", insights),
        ("out/edges.csv", kept_edges),
    ]:
        if len(frame) == 0:
            continue
        for _, row in frame.iterrows():
            edge = build_scm_edge_row(row, default_origin=default_origin)
            if edge is not None:
                edge_rows.append(edge)
    edge_df = pd.DataFrame(dedupe_edges_by_authority(edge_rows))

    has_discovery_material = bool(len(proposals) or len(insights) or len(bridge))
    scm_graph = {
        "scm_version": 4,
        "type": "scm_data_aware_lite",
        "assumptions": {
            "discovery_backbone": (
                "optional PCMCI-like discovery with PC1 + MCI + BH-FDR"
                if has_discovery_material
                else "Discovery not required for this run; no Discovery/bridge material was used"
            ),
            "candidate_source_policy": "Discovery/PCMCI is an optional seed generator; SCM/ID/contract remain downstream authority gates.",
            "structural_status": "data-aware structural families with explicit support/noise metadata, but not a fully identified SCM",
            "edge_authority_policy": "SCM edges are candidates. Only causal_contract.csv can authorize estimation claims.",
            "intended_use": "offline-prior-to-SCM bridge for identification, estimation, and simple interventional simulation",
        },
        "nodes": node_df.to_dict(orient="records"),
        "edges": edge_df.to_dict(orient="records"),
        "structural_families": family_df.to_dict(orient="records"),
        "observability_summary": {
            "n_nodes": int(len(node_df)),
            "n_observed_in_data": int(node_df.get("data_column_present", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if len(node_df) else 0,
            "n_not_fit_eligible": int((~node_df.get("fit_eligible", pd.Series(dtype=bool)).fillna(False).astype(bool)).sum()) if len(node_df) else 0,
        },
    }
    _admg, admg_summary, admg_components = admg_report_from_scm_graph(scm_graph)
    id_audit_rows = id_audit_rows_from_scm_graph(scm_graph)
    id_summary = id_algorithm_summary(_admg, id_audit_rows)
    symbolic_evaluation_rows = evaluate_id_audit_rows(id_audit_rows)
    symbolic_eval_summary = symbolic_evaluation_summary(symbolic_evaluation_rows)
    scm_graph["admg_summary"] = admg_summary
    scm_graph["c_components"] = admg_components
    scm_graph["id_algorithm_summary"] = id_summary
    scm_graph["id_algorithm_audit"] = id_audit_rows
    scm_graph["symbolic_evaluation_summary"] = symbolic_eval_summary
    return {
        "node_roles": node_df,
        "structural_families": family_df,
        "scm_edges": edge_df,
        "scm_graph": scm_graph,
        "admg_components": pd.DataFrame(admg_components),
        "admg_summary": admg_summary,
        "id_algorithm_audit": pd.DataFrame(id_audit_rows),
        "id_algorithm_summary": id_summary,
        "symbolic_evaluation": pd.DataFrame(symbolic_evaluation_rows, columns=SYMBOLIC_EVALUATION_COLUMNS),
        "symbolic_evaluation_summary": symbolic_eval_summary,
        "node_observability": node_df[[c for c in [
            "node_id", "node_role", "observed", "data_column_present",
            "declared_observed_in_graph", "observation_status", "fit_eligible",
            "role_source", "value_family_source", "missing_frac", "n_unique"
        ] if c in node_df.columns]].copy() if len(node_df) else pd.DataFrame(columns=[
            "node_id", "node_role", "observed", "data_column_present",
            "declared_observed_in_graph", "observation_status", "fit_eligible",
            "role_source", "value_family_source", "missing_frac", "n_unique"
        ]),
    }



def _empty_scm_reason(proposals: pd.DataFrame, insights: pd.DataFrame, bridge: Optional[pd.DataFrame], assets: Dict[str, object]) -> str:
    """Return a stable reason code when the SCM builder writes an empty graph.

    Empty SCM assets are allowed in Amantia because Discovery is deliberately
    conservative. The reason code makes this explicit in manifests instead of
    leaving users to infer whether the builder crashed or simply had no
    authorized material to promote into an SCM graph.
    """
    try:
        n_nodes = len(assets.get("node_roles", []))
        n_edges = len(assets.get("scm_edges", []))
    except (AttributeError, TypeError):
        n_nodes = 0
        n_edges = 0
    if n_nodes > 0 or n_edges > 0:
        return ""
    if bridge is None or len(bridge) == 0:
        if insights is None or len(insights) == 0:
            if proposals is None or len(proposals) == 0:
                return "no_discovery_proposals_or_bridge_rows"
            return "no_kept_discovery_insights"
        return "missing_or_empty_discovery_bridge"
    if insights is None or len(insights) == 0:
        return "bridge_present_but_no_kept_discovery_insights"
    return "no_valid_edges_for_scm_after_authority_filter"

def write_scm_assets(
    proposals: pd.DataFrame,
    insights: pd.DataFrame,
    out_dir: str = "out",
    bridge: Optional[pd.DataFrame] = None,
    dag_path: Optional[str] = None,
    data_path: Optional[str] = None,
    dag_obj=None,
) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    scm_dir = os.path.join(out_dir, SCM_DIRNAME)
    os.makedirs(scm_dir, exist_ok=True)
    dag = dag_obj if dag_obj is not None else _load_dag(dag_path, out_dir)
    data = _load_data(data_path, out_dir)
    assets = build_scm_assets(proposals, insights, bridge=bridge, dag=dag, data=data)

    # Step 180: keep downstream SCM CSVs machine-readable even when Discovery
    # produces no kept insights. Empty files should still carry their schema.
    empty_schemas = {
        "node_roles": ["node_id", "node_role", "observed", "data_column_present", "declared_observed_in_graph", "observation_status", "fit_eligible", "intervenable", "adjustable", "sensitive", "time_role", "role_source", "dtype_family", "support_min", "support_max", "n_unique", "missing_frac", "value_family_source"],
        "structural_families": ["node_id", "node_role", "dtype_family", "structural_family", "structural_equation_template", "noise_symbol", "noise_family", "family_source"],
        "scm_edges": ["source", "target", "lag", "edge_authority_level", "edge_source_artifact", "edge_source_layer", "bridge_version", "raw_seed_only", "eligible_for_identification", "eligible_for_estimation", "is_formally_identified", "authority_reason_codes", "insight_id", "confidence_tier", "parent_set", "conditioning_set_used", "conditioning_set_size", "conditioning_quality", "mci_status", "scm_role_hint", "identification_priority", "candidate_covariates", "post_treatment_columns", "forbidden_adjustment_set"],
        "admg_components": ["component_id", "nodes", "reason_codes"],
        "id_algorithm_audit": ["treatment", "outcome", "identifiable", "id_strategy", "id_algorithm_level", "estimand_formula", "symbolic_formula_status", "symbolic_formula_kind", "symbolic_formula_json", "symbolic_formula_ast_json", "symbolic_formula_latex", "symbolic_sum_over", "symbolic_product_terms", "symbolic_removed_terms", "symbolic_unresolved_terms", "adjustment_set", "mediators", "c_components", "hedge_status", "hedge_witness", "ancestral_c_components", "treatment_ancestral_district", "outcome_ancestral_district", "possible_hedge", "directed_acyclic", "directed_cycle_nodes", "backdoor_status", "frontdoor_status", "recursive_status", "c_factor_status", "district_status", "failure_reason", "reason_codes", "edge_authority_level", "edge_source_artifact", "insight_id"],
        "symbolic_evaluation": SYMBOLIC_EVALUATION_COLUMNS,
        "node_observability": ["node_id", "node_role", "observed", "data_column_present", "declared_observed_in_graph", "observation_status", "fit_eligible", "role_source", "value_family_source", "missing_frac", "n_unique"],
    }
    for key, cols in empty_schemas.items():
        frame = assets.get(key)
        if frame is None or not hasattr(frame, "columns") or (len(frame) == 0 and len(frame.columns) == 0):
            assets[key] = pd.DataFrame(columns=cols)

    paths = {
        "scm_graph_json": os.path.join(scm_dir, "scm_graph.json"),
        "node_roles_csv": os.path.join(scm_dir, "node_roles.csv"),
        "structural_families_csv": os.path.join(scm_dir, "structural_families.csv"),
        "scm_edges_csv": os.path.join(scm_dir, "scm_edges.csv"),
        "edge_authority_report_csv": os.path.join(scm_dir, "edge_authority_report.csv"),
        "admg_c_components_csv": os.path.join(scm_dir, "admg_c_components.csv"),
        "id_algorithm_audit_csv": os.path.join(scm_dir, "id_algorithm_audit.csv"),
        "symbolic_evaluation_csv": os.path.join(scm_dir, "symbolic_evaluation.csv"),
        "symbolic_evaluation_summary_json": os.path.join(scm_dir, "symbolic_evaluation_summary.json"),
        "node_observability_csv": os.path.join(scm_dir, "node_observability.csv"),
        "manifest": os.path.join(scm_dir, "scm_manifest.json"),
    }
    assets["node_roles"].to_csv(paths["node_roles_csv"], index=False)
    assets["structural_families"].to_csv(paths["structural_families_csv"], index=False)
    assets["scm_edges"].to_csv(paths["scm_edges_csv"], index=False)
    assets.get("admg_components", pd.DataFrame()).to_csv(paths["admg_c_components_csv"], index=False)
    assets.get("id_algorithm_audit", pd.DataFrame()).to_csv(paths["id_algorithm_audit_csv"], index=False)
    assets.get("symbolic_evaluation", pd.DataFrame(columns=SYMBOLIC_EVALUATION_COLUMNS)).to_csv(paths["symbolic_evaluation_csv"], index=False)
    with open(paths["symbolic_evaluation_summary_json"], "w", encoding="utf-8") as f:
        json.dump(assets.get("symbolic_evaluation_summary", symbolic_evaluation_summary([])), f, ensure_ascii=False, indent=2)
    assets.get("node_observability", pd.DataFrame()).to_csv(paths["node_observability_csv"], index=False)
    authority_cols = [
        c for c in [
            "source", "target", "lag", "edge_authority_level", "edge_source_artifact",
            "edge_source_layer", "bridge_version", "raw_seed_only",
            "eligible_for_identification", "eligible_for_estimation",
            "is_formally_identified", "authority_reason_codes",
            "insight_id", "confidence_tier", "parent_set", "conditioning_set_used",
            "conditioning_set_size", "conditioning_quality", "mci_status", "scm_role_hint",
            "identification_priority", "candidate_covariates",
            "post_treatment_columns", "forbidden_adjustment_set",
        ] if c in assets["scm_edges"].columns
    ]
    assets["scm_edges"][authority_cols].to_csv(paths["edge_authority_report_csv"], index=False)
    with open(paths["scm_graph_json"], "w", encoding="utf-8") as f:
        json.dump(assets["scm_graph"], f, ensure_ascii=False, indent=2)

    edge_authority_counts = {}
    if "edge_authority_level" in assets["scm_edges"].columns:
        edge_authority_counts = {
            str(k): int(v) for k, v in assets["scm_edges"]["edge_authority_level"].value_counts(dropna=False).to_dict().items()
        }
    empty_scm_reason = _empty_scm_reason(proposals, insights, bridge, assets)
    manifest = {
        "layer": "scm",
        "scm_type": "scm_data_aware_lite",
        "scm_version": 5,
        "authority_policy": "SCM edges are candidates; causal_contract.csv is the only estimation authority.",
        "n_nodes": int(len(assets["node_roles"])),
        "n_edges": int(len(assets["scm_edges"])),
        "empty_scm_reason": empty_scm_reason,
        "edge_authority_counts": edge_authority_counts,
        "n_structural_families": int(len(assets["structural_families"])),
        "n_observed_nodes_in_data": int(assets.get("node_observability", pd.DataFrame()).get("data_column_present", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if len(assets.get("node_observability", pd.DataFrame())) else 0,
        "n_not_fit_eligible_nodes": int((~assets.get("node_observability", pd.DataFrame()).get("fit_eligible", pd.Series(dtype=bool)).fillna(False).astype(bool)).sum()) if len(assets.get("node_observability", pd.DataFrame())) else 0,
        "admg_summary": assets.get("admg_summary", {}),
        "id_algorithm_summary": assets.get("id_algorithm_summary", {}),
        "symbolic_evaluation_summary": assets.get("symbolic_evaluation_summary", {}),
        "data_source_used": data_path or os.path.join(out_dir, "data_clean.csv"),
        "files": paths,
    }
    with open(paths["manifest"], "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return paths


def build_from_out_dir(out_dir: str = "out", dag_path: Optional[str] = None, data_path: Optional[str] = None) -> Dict[str, str]:
    proposals, insights, bridge = read_discovery_frames(out_dir, _normalize_edge_contract)
    paths = write_scm_assets(proposals, insights, out_dir=out_dir, bridge=bridge, dag_path=dag_path, data_path=data_path)
    try:
        from contracts.causal_contract import write_causal_contract
        paths.update(write_causal_contract(out_dir=out_dir))
    except (OSError, ValueError, TypeError, RuntimeError, KeyError, ImportError) as exc:
        warnings.warn(f"[amantia][warning] causal contract sync after scm-build failed: {type(exc).__name__}: {exc}", RuntimeWarning)
    return paths


def cli(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Build data-aware SCM assets from Discovery outputs")
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--data-path", default=None)
    args = ap.parse_args(argv)
    paths = build_from_out_dir(out_dir=args.out_dir, data_path=args.data_path)
    print("Saved SCM graph:", paths["scm_graph_json"])
    print("Saved node roles:", paths["node_roles_csv"])
    print("Saved structural families:", paths["structural_families_csv"])
    if "edge_authority_report_csv" in paths:
        print("Saved edge authority report:", paths["edge_authority_report_csv"])
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
