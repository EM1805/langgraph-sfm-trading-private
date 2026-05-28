from __future__ import annotations

from runtime_env import configure_scientific_runtime
configure_scientific_runtime()


"""Fit structural equations for SCM nodes.

This module upgrades the SCM bridge with estimated node-wise structural models.
The fit remains dependency-light (numpy/pandas only) but supports a compact
nonlinear basis registry and exports richer exogenous-noise summaries.
"""

import json
import os
import warnings
from collections import defaultdict, deque
from typing import Dict, List, Optional, Set, Tuple

from runtime_compat import assert_scientific_stack
assert_scientific_stack()

import numpy as np
import pandas as pd


SCM_DIRNAME = "scm"
INTERACTION_GAIN_MIN = 0.015
QUADRATIC_GAIN_MIN = 0.02


def _sigmoid(x):
    x = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_data(data_path: Optional[str], out_dir: str) -> pd.DataFrame:
    for path in [data_path, os.path.join(out_dir, "data_clean.csv"), "data.csv", os.path.join(out_dir, "demo_data.csv")]:
        if path and os.path.exists(path):
            try:
                return pd.read_csv(path)
            except (OSError, ValueError, TypeError, pd.errors.ParserError) as exc:
                warnings.warn(f"[amantia][warning] SCM fit could not read data file {path}: {type(exc).__name__}: {exc}", RuntimeWarning)
                continue
    return pd.DataFrame()


def _build_graph(edges: List[dict]) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    children: Dict[str, Set[str]] = defaultdict(set)
    parents: Dict[str, Set[str]] = defaultdict(set)
    for e in edges:
        s = str(e.get("source", "")).strip()
        t = str(e.get("target", "")).strip()
        if s and t:
            children[s].add(t)
            parents[t].add(s)
    return children, parents


def _topological_hint(nodes: List[str], children: Dict[str, Set[str]], parents: Dict[str, Set[str]]) -> List[str]:
    indeg = {n: len(parents.get(n, set())) for n in nodes}
    dq = deque(sorted([n for n in nodes if indeg.get(n, 0) == 0]))
    out: List[str] = []
    seen: Set[str] = set()
    while dq:
        cur = dq.popleft()
        if cur in seen:
            continue
        seen.add(cur)
        out.append(cur)
        for nxt in sorted(children.get(cur, set())):
            indeg[nxt] = max(0, indeg.get(nxt, 0) - 1)
            if indeg[nxt] == 0:
                dq.append(nxt)
    for n in sorted(nodes):
        if n not in seen:
            out.append(n)
    return out


def _fit_linear(X: np.ndarray, y: np.ndarray, ridge: float = 1e-6):
    XtX = X.T.dot(X) + ridge * np.eye(X.shape[1])
    Xty = X.T.dot(y)
    beta = np.linalg.solve(XtX, Xty)
    pred = X.dot(beta)
    resid = y - pred
    return beta, pred, resid


def _continuous_parent(df: pd.DataFrame, col: str) -> bool:
    if col not in df.columns:
        return False
    s = pd.to_numeric(df[col], errors="coerce")
    s = s[np.isfinite(s)]
    if len(s) < 8:
        return False
    uniq = np.unique(s)
    if len(uniq) <= 3 and set(np.round(uniq, 8)).issubset({0.0, 1.0, -1.0}):
        return False
    return True


def _candidate_designs(df: pd.DataFrame, parents: List[str]):
    base_cols = [pd.Series(1.0, index=df.index, name="intercept")]
    base_feats = ["intercept"]
    continuous = []
    for p in parents:
        s = pd.to_numeric(df[p], errors="coerce").rename(p)
        base_cols.append(s)
        base_feats.append(p)
        if _continuous_parent(df, p):
            continuous.append(p)

    yield pd.concat(base_cols, axis=1), list(base_feats), "linear_basis"

    quad_cols = list(base_cols)
    quad_feats = list(base_feats)
    for p in continuous:
        s = pd.to_numeric(df[p], errors="coerce")
        quad_cols.append((s.astype(float) ** 2).rename(f"{p}__sq"))
        quad_feats.append(f"{p}__sq")
    if len(quad_feats) > len(base_feats):
        yield pd.concat(quad_cols, axis=1), quad_feats, "quadratic_basis"

    inter_cols = list(quad_cols)
    inter_feats = list(quad_feats)
    for i in range(len(continuous)):
        for j in range(i + 1, len(continuous)):
            a, b = continuous[i], continuous[j]
            name = f"{a}__x__{b}"
            sa = pd.to_numeric(df[a], errors="coerce").astype(float)
            sb = pd.to_numeric(df[b], errors="coerce").astype(float)
            inter_cols.append((sa * sb).rename(name))
            inter_feats.append(name)
            if len(inter_feats) >= len(base_feats) + 4:
                break
        if len(inter_feats) >= len(base_feats) + 4:
            break
    if len(inter_feats) > len(quad_feats):
        yield pd.concat(inter_cols, axis=1), inter_feats, "quadratic_interaction_basis"


def _score_prediction(dtype_family: str, yv: np.ndarray, pred_latent: np.ndarray):
    if dtype_family in {"binary", "rate"}:
        pred_mean = _sigmoid(pred_latent)
        resid_eval = yv - pred_mean
        score = float(1.0 - np.mean((yv - pred_mean) ** 2) / max(np.var(yv), 1e-9))
        prediction_link = "sigmoid"
    elif dtype_family == "count":
        pred_mean = np.maximum(0.0, np.exp(np.clip(pred_latent, -20.0, 10.0)) - 1.0)
        resid_eval = yv - pred_mean
        score = float(1.0 - np.mean((yv - pred_mean) ** 2) / max(np.var(yv), 1e-9))
        prediction_link = "exp_count"
    else:
        pred_mean = pred_latent
        resid_eval = yv - pred_mean
        score = float(1.0 - np.var(resid_eval) / max(np.var(yv), 1e-9))
        prediction_link = "identity"
    return prediction_link, pred_mean, resid_eval, float(max(-1.0, min(1.0, score)))




def _term_to_expr(term: str, coef: float) -> str:
    coef_s = f"{float(coef):.6g}"
    if term == "intercept":
        return coef_s
    if term.endswith("__sq"):
        base = term[:-4]
        return f"({coef_s} * ({base}^2))"
    if "__x__" in term:
        a, b = term.split("__x__", 1)
        return f"({coef_s} * {a} * {b})"
    return f"({coef_s} * {term})"


def _link_inverse_expr(dtype_family: str, latent_expr: str, prediction_link: str) -> str:
    expr = latent_expr
    if dtype_family in {"binary", "rate"} or prediction_link == "sigmoid":
        expr = f"sigmoid({expr})"
    elif dtype_family == "count" or prediction_link == "exp_count":
        expr = f"max(0, exp(clip({expr}, -20, 10)) - 1)"
    if dtype_family == "rate":
        expr = f"clip({expr}, 0, 1)"
    return expr


def _equation_payload(node: str, dtype_family: str, prediction_link: str, coefficients: Dict[str, float], parents: List[str], model_class: str) -> Dict[str, object]:
    ordered_terms = [t for t in coefficients.keys()]
    terms = [_term_to_expr(t, coefficients[t]) for t in ordered_terms if abs(float(coefficients[t])) > 0.0]
    latent_rhs = " + ".join(terms) if terms else "0"
    latent_with_noise = f"{latent_rhs} + U_{node}"
    response_rhs = _link_inverse_expr(dtype_family, latent_with_noise, prediction_link)
    return {
        "node_id": node,
        "exogenous_symbol": f"U_{node}",
        "parents": list(parents),
        "model_class": model_class,
        "latent_equation": f"eta_{node} := {latent_rhs}",
        "structural_equation": f"{node} := {response_rhs}",
        "structural_equation_without_noise": f"{node} := {_link_inverse_expr(dtype_family, latent_rhs, prediction_link)}",
        "response_function": _link_inverse_expr(dtype_family, f"eta_{node} + U_{node}", prediction_link),
        "coefficient_order": ordered_terms,
        "coefficients": {k: float(v) for k, v in coefficients.items()},
    }
def _noise_family(dtype_family: str, residuals: np.ndarray) -> str:
    residuals = np.asarray(residuals, dtype=float)
    if residuals.size == 0:
        return "unknown_noise"
    if dtype_family in {"binary", "rate"}:
        return "bounded_response_noise"
    if dtype_family == "count":
        return "count_latent_noise"
    q10, q50, q90 = np.quantile(residuals, [0.1, 0.5, 0.9])
    skew_hint = abs((q90 + q10) - 2.0 * q50)
    return "gaussian_like_noise" if skew_hint <= max(0.15, 0.35 * np.std(residuals)) else "asymmetric_continuous_noise"


def fit_scm_models(out_dir: str = "out", scm_graph_path: Optional[str] = None, data_path: Optional[str] = None) -> Dict[str, str]:
    scm_dir = os.path.join(out_dir, SCM_DIRNAME)
    os.makedirs(scm_dir, exist_ok=True)
    if scm_graph_path is None:
        scm_graph_path = os.path.join(scm_dir, "scm_graph.json")
        if not os.path.exists(scm_graph_path):
            scm_graph_path = os.path.join(out_dir, "scm_graph.json")
    scm_graph = _load_json(scm_graph_path)
    df = _load_data(data_path, out_dir)

    nodes = list(scm_graph.get("nodes", []))
    edges = list(scm_graph.get("edges", []))
    families = {str(x.get("node_id", "")).strip(): x for x in scm_graph.get("structural_families", [])}
    node_meta = {str(x.get("node_id", "")).strip(): x for x in nodes}
    children, parents = _build_graph(edges)
    ordering = _topological_hint(sorted(node_meta.keys()), children, parents)

    model_rows: List[dict] = []
    summary_rows: List[dict] = []
    skipped_rows: List[dict] = []
    registry = {
        "scm_fit_version": 4,
        "fit_authority_policy": "diagnostic_simulation_only_not_identification_authority",
        "models": [],
        "topological_hint": ordering,
    }

    def _skip_row(node: str, reason: str) -> dict:
        meta = node_meta.get(node, {})
        return {
            "node_id": node,
            "reason": reason,
            "node_role": str(meta.get("node_role", "state")),
            "observed": bool(meta.get("observed", False)),
            "data_column_present": bool(meta.get("data_column_present", node in df.columns)),
            "declared_observed_in_graph": meta.get("declared_observed_in_graph", ""),
            "observation_status": str(meta.get("observation_status", "observed_in_data" if node in df.columns else "not_observed_in_current_data")),
            "fit_eligible": bool(meta.get("fit_eligible", node in df.columns)),
            "value_family_source": str(meta.get("value_family_source", "")),
        }

    for node in ordering:
        meta = node_meta.get(node, {})
        if node not in df.columns:
            skipped_rows.append(_skip_row(node, str(meta.get("observation_status") or "missing_from_data")))
            continue
        if not bool(meta.get("fit_eligible", True)):
            skipped_rows.append(_skip_row(node, "not_fit_eligible_by_scm_metadata"))
            continue
        fam = families.get(node, {})
        dtype_family = str(meta.get("dtype_family", fam.get("dtype_family", "continuous")) or "continuous")
        pa = [p for p in ordering if p in parents.get(node, set()) and p in df.columns and p != node]
        y = pd.to_numeric(df[node], errors="coerce")

        best = None
        for Xdf, used_features, model_class in _candidate_designs(df, pa):
            fit_df = pd.concat([Xdf, y.rename("y")], axis=1).dropna()
            if len(fit_df) < max(5, len(used_features) + 2):
                continue
            X = fit_df[used_features].to_numpy(dtype=float)
            yv = fit_df["y"].to_numpy(dtype=float)
            beta, pred_latent, resid_latent = _fit_linear(X, yv)
            prediction_link, pred_mean, resid_eval, score = _score_prediction(dtype_family, yv, pred_latent)
            cand = {
                "fit_df": fit_df,
                "used_features": used_features,
                "beta": beta,
                "pred_latent": pred_latent,
                "pred_mean": pred_mean,
                "resid_eval": resid_eval,
                "score": score,
                "prediction_link": prediction_link,
                "model_class": model_class,
            }
            if best is None:
                best = cand
                continue
            gain = cand["score"] - best["score"]
            if cand["model_class"] == "quadratic_basis":
                if gain > QUADRATIC_GAIN_MIN:
                    best = cand
            elif cand["model_class"] == "quadratic_interaction_basis":
                if gain > INTERACTION_GAIN_MIN:
                    best = cand
            elif gain > 0.0:
                best = cand
        if best is None:
            skipped_rows.append({"node_id": node, "reason": "insufficient_fit_data_or_features"})
            continue

        resid_sd = float(np.std(best["resid_eval"]))
        resid_q10, resid_q50, resid_q90 = [float(x) for x in np.quantile(best["resid_eval"], [0.1, 0.5, 0.9])]
        noise_family = _noise_family(dtype_family, best["resid_eval"])
        node_support_min = node_meta.get(node, {}).get("support_min")
        node_support_max = node_meta.get(node, {}).get("support_max")
        coefficients = {best["used_features"][i]: float(best["beta"][i]) for i in range(len(best["used_features"]))}
        eq_payload = _equation_payload(node, dtype_family, best["prediction_link"], coefficients, pa, best["model_class"])
        row = {
            "node_id": node,
            "fit_authority_level": "diagnostic_simulation_only",
            "fit_authority_reason": "node_wise_predictive_fit_does_not_identify_causal_effects",
            "causal_authority_from_fit": 0,
            "node_role": str(node_meta.get(node, {}).get("node_role", "state")),
            "dtype_family": dtype_family,
            "structural_family": str(fam.get("structural_family", "additive_structural")),
            "prediction_link": best["prediction_link"],
            "model_class": best["model_class"],
            "parents": "|".join(pa),
            "feature_terms": "|".join(best["used_features"]),
            "coefficients_json": json.dumps(coefficients, sort_keys=True),
            "n_fit": int(len(best["fit_df"])),
            "fit_score": round(float(best["score"]), 6),
            "residual_sd": round(resid_sd, 6),
            "residual_q10": round(resid_q10, 6),
            "residual_q50": round(resid_q50, 6),
            "residual_q90": round(resid_q90, 6),
            "noise_family": noise_family,
            "support_min": None if node_support_min is None else float(node_support_min),
            "support_max": None if node_support_max is None else float(node_support_max),
            "n_unique": int(node_meta.get(node, {}).get("n_unique", 0) or 0),
            "exogenous_symbol": eq_payload["exogenous_symbol"],
            "latent_equation": eq_payload["latent_equation"],
            "structural_equation": eq_payload["structural_equation"],
            "structural_equation_without_noise": eq_payload["structural_equation_without_noise"],
            "response_function": eq_payload["response_function"],
        }
        model_rows.append(row)
        summary_rows.append({
            "node_id": node,
            "fit_authority_level": "diagnostic_simulation_only",
            "fit_authority_reason": "node_wise_predictive_fit_does_not_identify_causal_effects",
            "causal_authority_from_fit": 0,
            "fit_score": round(float(best["score"]), 6),
            "residual_sd": round(resid_sd, 6),
            "residual_q10": round(resid_q10, 6),
            "residual_q50": round(resid_q50, 6),
            "residual_q90": round(resid_q90, 6),
            "n_fit": int(len(best["fit_df"])),
            "model_class": best["model_class"],
            "noise_family": noise_family,
        })
        registry["models"].append({
            "node_id": node,
            "fit_authority_level": "diagnostic_simulation_only",
            "fit_authority_reason": "node_wise_predictive_fit_does_not_identify_causal_effects",
            "causal_authority_from_fit": False,
            "parents": pa,
            "feature_terms": best["used_features"],
            "coefficients": coefficients,
            "dtype_family": dtype_family,
            "structural_family": str(fam.get("structural_family", "additive_structural")),
            "prediction_link": best["prediction_link"],
            "model_class": best["model_class"],
            "fit_score": round(float(best["score"]), 6),
            "residual_sd": round(resid_sd, 6),
            "residual_quantiles": {"q10": round(resid_q10, 6), "q50": round(resid_q50, 6), "q90": round(resid_q90, 6)},
            "noise_family": noise_family,
            "exogenous_symbol": eq_payload["exogenous_symbol"],
            "latent_equation": eq_payload["latent_equation"],
            "structural_equation": eq_payload["structural_equation"],
            "structural_equation_without_noise": eq_payload["structural_equation_without_noise"],
            "response_function": eq_payload["response_function"],
        })

    models_df = pd.DataFrame(model_rows)
    summary_df = pd.DataFrame(summary_rows)
    skipped_df = pd.DataFrame(skipped_rows)
    equation_rows = models_df[[
        "node_id", "node_role", "dtype_family", "model_class", "parents", "feature_terms",
        "exogenous_symbol", "latent_equation", "structural_equation",
        "structural_equation_without_noise", "response_function", "fit_score", "noise_family",
        "fit_authority_level", "fit_authority_reason", "causal_authority_from_fit"
    ]].copy() if len(models_df) else pd.DataFrame(columns=[
        "node_id", "node_role", "dtype_family", "model_class", "parents", "feature_terms",
        "exogenous_symbol", "latent_equation", "structural_equation",
        "structural_equation_without_noise", "response_function", "fit_score", "noise_family",
        "fit_authority_level", "fit_authority_reason", "causal_authority_from_fit"
    ])
    paths = {
        "structural_models_csv": os.path.join(scm_dir, "structural_models.csv"),
        "structural_models_json": os.path.join(scm_dir, "structural_models.json"),
        "structural_equations_csv": os.path.join(scm_dir, "structural_equations.csv"),
        "structural_equations_json": os.path.join(scm_dir, "structural_equations.json"),
        "exogenous_noise_summary_json": os.path.join(scm_dir, "exogenous_noise_summary.json"),
        "fit_manifest_json": os.path.join(scm_dir, "fit_manifest.json"),
        "skipped_nodes_csv": os.path.join(scm_dir, "skipped_nodes.csv"),
    }
    models_df.to_csv(paths["structural_models_csv"], index=False)
    equation_rows.to_csv(paths["structural_equations_csv"], index=False)
    skipped_df.to_csv(paths["skipped_nodes_csv"], index=False)
    with open(paths["structural_models_json"], "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)
    with open(paths["structural_equations_json"], "w", encoding="utf-8") as f:
        json.dump({"scm_equations_version": 1, "equations": equation_rows.to_dict(orient="records")}, f, ensure_ascii=False, indent=2)
    with open(paths["exogenous_noise_summary_json"], "w", encoding="utf-8") as f:
        json.dump({"nodes": summary_df.to_dict(orient="records")}, f, ensure_ascii=False, indent=2)
    empty_scm_reason = ""
    if len(ordering) == 0:
        empty_scm_reason = "no_scm_graph_nodes"
    elif len(models_df) == 0:
        empty_scm_reason = "no_structural_models_fit_from_current_graph_or_data"
    manifest = {
        "scm_fit_version": 4,
        "fit_authority_policy": "diagnostic_simulation_only_not_identification_authority",
        "downstream_rule": "structural fit may support simulation diagnostics but must not authorize estimation without causal_contract identification",
        "empty_scm_reason": empty_scm_reason,
        "n_graph_nodes": int(len(ordering)),
        "n_fitted_nodes": int(len(models_df)),
        "n_skipped_nodes": int(len(skipped_df)),
        "skipped_nodes": skipped_df.to_dict(orient="records"),
    }
    with open(paths["fit_manifest_json"], "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    if len(skipped_df):
        warnings.warn(f"[amantia][warning] SCM fit skipped {len(skipped_df)} node(s); see {paths['fit_manifest_json']}", RuntimeWarning)
    return paths


if __name__ == "__main__":
    out = fit_scm_models()
    print("Saved structural models:", out["structural_models_csv"])
