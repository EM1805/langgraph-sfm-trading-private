from __future__ import annotations

from runtime_env import configure_scientific_runtime
configure_scientific_runtime()

"""Numeric evaluator for safe symbolic ID formulas.

Step 12 scope
-------------
This module evaluates only formulas that are already authorized by
``causal_contract.csv`` and routed by ``scm_parts.symbolic_evaluator``.

Supported routes:
- ``symbolic_numeric_truncated_factorization``: observed-DAG g-formula via
  sequential linear standardization over the formula topological order.
- ``zero_effect_contract``: graphical zero effect, no fitted causal model.

It deliberately does not evaluate unresolved Q/c-factor formulas or any row that
is not contract-enabled.  Backdoor/frontdoor estimators remain in
``do_backdoor.py`` and ``do_frontdoor.py``.
"""

import csv
import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .do_backdoor import DO_DIAGNOSTIC_COLUMNS, DO_ESTIMATE_COLUMNS, _write_csv
from .do_contract import has_canonical_id_authority, load_causal_contract
from .symbolic_evaluator import parse_formula_ast_json, parse_symbolic_formula_json
from .idc_fraction_numeric import IDC_FRACTION_NUMERIC_VERSION, analyze_idc_fraction_ast

SYMBOLIC_NUMERIC_VERSION = 3

SUPPORTED_NUMERIC_ROUTES = {
    "symbolic_numeric_truncated_factorization",
    "zero_effect_contract",
    "symbolic_numeric_idc_fraction_ratio",
}

MIN_SUPPORT_PER_SIDE = 5
MIN_SUPPORT_RATIO = 0.08
MIN_BOOTSTRAP_SUCCESS = 20
WIDE_CI_TO_EFFECT_RATIO = 5.0
WIDE_CI_TO_OUTCOME_SD_RATIO = 2.5


SYMBOLIC_NUMERIC_ESTIMATE_COLUMNS: List[str] = DO_ESTIMATE_COLUMNS + [
    "symbolic_numeric_version",
    "symbolic_formula_kind",
    "symbolic_estimator_route",
    "symbolic_formula_type",
    "topological_order",
    "symbolic_sum_over",
    "symbolic_product_terms",
]

SYMBOLIC_NUMERIC_DIAGNOSTIC_COLUMNS: List[str] = DO_DIAGNOSTIC_COLUMNS + [
    "symbolic_numeric_version",
    "symbolic_estimator_route",
    "symbolic_formula_type",
    "topological_order",
    "symbolic_gate_pass",
    "symbolic_gate_reasons",
    "model_fit_nodes",
    "model_skipped_nodes",
]


def _s(value: object) -> str:
    raw = "" if value is None else str(value).strip()
    return "" if raw.lower() in {"nan", "none", "null", "nat"} else raw


def _truthy(value: object) -> bool:
    text = _s(value).lower()
    if text in {"1", "true", "yes", "y", "on", "enabled", "identified_estimable"}:
        return True
    if text in {"", "0", "false", "no", "n", "off", "disabled"}:
        return False
    try:
        return float(text) != 0.0
    except (TypeError, ValueError, OverflowError):
        return False


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
    if isinstance(value, (list, tuple)):
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


def _load_data(data_path: Optional[str] = None, out_dir: str = "out") -> pd.DataFrame:
    candidates = [data_path, os.path.join(out_dir, "data_clean.csv"), "data.csv", os.path.join(out_dir, "demo_data.csv")]
    for path in candidates:
        if path and os.path.exists(path):
            try:
                return pd.read_csv(path)
            except (OSError, ValueError, TypeError, pd.errors.ParserError):
                continue
    return pd.DataFrame()


def _numeric_frame(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in columns:
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce")
    return out


def _treatment_values(series: pd.Series) -> Tuple[float, float]:
    x = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if x.empty:
        return (float("nan"), float("nan"))
    unique = sorted(set(float(v) for v in x.tolist()))
    if len(unique) == 1:
        return (unique[0], unique[0])
    if len(unique) <= 5:
        return (float(unique[0]), float(unique[-1]))
    return (float(x.quantile(0.25)), float(x.quantile(0.75)))


def _support_counts(series: pd.Series, low: float, high: float) -> Tuple[int, int]:
    diag = _support_diagnostics(series, low, high)
    return int(diag["support_n_low"]), int(diag["support_n_high"])


def _support_diagnostics(series: pd.Series, low: float, high: float) -> Dict[str, object]:
    support = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if support.empty or not np.isfinite(low) or not np.isfinite(high):
        return {
            "support_n_low": 0,
            "support_n_high": 0,
            "support_n_mid": 0,
            "support_min": 0,
            "support_ratio": 0.0,
            "overlap_score": 0.0,
            "treatment_unique_n": 0,
            "extrapolation_risk": "high_no_numeric_treatment_support",
        }
    n = int(len(support))
    unique_n = int(len(set(support.round(8))))
    if unique_n <= 5:
        support_low = int((support == low).sum())
        support_high = int((support == high).sum())
        support_mid = int(((support >= min(low, high)) & (support <= max(low, high))).sum())
    else:
        support_low = int((support <= low).sum())
        support_high = int((support >= high).sum())
        support_mid = int(((support >= min(low, high)) & (support <= max(low, high))).sum())
    support_min = int(min(support_low, support_high))
    support_ratio = float(support_min / n) if n else 0.0
    overlap_score = float(max(0.0, min(1.0, support_ratio / max(MIN_SUPPORT_RATIO, 1e-9))))
    min_obs = float(support.min())
    max_obs = float(support.max())
    if low < min_obs or high > max_obs:
        extrap = "high_outside_observed_range"
    elif support_min < MIN_SUPPORT_PER_SIDE:
        extrap = "medium_low_tail_support"
    elif support_ratio < MIN_SUPPORT_RATIO:
        extrap = "medium_weak_overlap_ratio"
    else:
        extrap = "low"
    return {
        "support_n_low": support_low,
        "support_n_high": support_high,
        "support_n_mid": support_mid,
        "support_min": support_min,
        "support_ratio": round(support_ratio, 6),
        "overlap_score": round(overlap_score, 6),
        "treatment_unique_n": unique_n,
        "extrapolation_risk": extrap,
    }


def _robustness_diagnostics(
    *,
    effect: float,
    ci_low: float,
    ci_high: float,
    bootstrap_draws: int,
    bootstrap_success_n: int,
    support_diag: Mapping[str, object],
    outcome_sd: float,
) -> Dict[str, object]:
    ci_width = float(ci_high - ci_low) if np.isfinite(ci_low) and np.isfinite(ci_high) else float("nan")
    effect_abs = abs(float(effect)) if np.isfinite(effect) else float("nan")
    ratio_to_effect = float(ci_width / max(effect_abs, 1e-9)) if np.isfinite(ci_width) and np.isfinite(effect_abs) else float("nan")
    ratio_to_sd = float(ci_width / max(abs(float(outcome_sd)), 1e-9)) if np.isfinite(ci_width) and np.isfinite(outcome_sd) and outcome_sd > 0 else float("nan")
    support_min = int(support_diag.get("support_min") or 0)
    support_ratio = float(support_diag.get("support_ratio") or 0.0)
    required_bootstrap_success = min(MIN_BOOTSTRAP_SUCCESS, max(2, min(int(max(1, bootstrap_draws)), max(3, int(max(1, bootstrap_draws) * 0.25)))))
    warnings: List[str] = []
    if not np.isfinite(effect):
        warnings.append("nonfinite_effect")
    if bootstrap_success_n < required_bootstrap_success:
        warnings.append("bootstrap_low_success")
    if not np.isfinite(ci_width):
        warnings.append("bootstrap_ci_unavailable")
    elif ratio_to_effect > WIDE_CI_TO_EFFECT_RATIO and effect_abs > 1e-9:
        warnings.append("wide_ci_relative_to_effect")
    elif np.isfinite(ratio_to_sd) and ratio_to_sd > WIDE_CI_TO_OUTCOME_SD_RATIO:
        warnings.append("wide_ci_relative_to_outcome_sd")
    if support_min < MIN_SUPPORT_PER_SIDE:
        warnings.append("low_tail_support")
    if support_ratio < MIN_SUPPORT_RATIO:
        warnings.append("weak_overlap_ratio")
    extrap = _s(support_diag.get("extrapolation_risk"))
    if extrap and extrap != "low":
        warnings.append(extrap)
    if "nonfinite_effect" in warnings or "bootstrap_ci_unavailable" in warnings or "low_tail_support" in warnings:
        status = "blocked_low_support_or_nonfinite"
    elif warnings:
        status = "diagnostic_only_robustness_warning"
    else:
        status = "ok"
    bootstrap_status = "ok" if bootstrap_success_n >= required_bootstrap_success and np.isfinite(ci_width) else "weak"
    return {
        "bootstrap_draws": int(bootstrap_draws),
        "bootstrap_success_n": int(bootstrap_success_n),
        "bootstrap_status": bootstrap_status,
        "ci_width": ci_width,
        "ci_width_to_effect_ratio": ratio_to_effect,
        "robustness_status": status,
        "overlap_score": support_diag.get("overlap_score", 0.0),
        "support_min": support_min,
        "support_ratio": support_ratio,
        "support_n_mid": support_diag.get("support_n_mid", 0),
        "treatment_unique_n": support_diag.get("treatment_unique_n", 0),
        "extrapolation_risk": extrap or "unknown",
        "sensitivity_warning": "|".join(_dedupe(warnings)) or "",
    }


_FACTOR_RE = re.compile(r"P\s*\(\s*([^|)]+?)\s*(?:\|\s*([^)]*?))?\s*\)")


def _parse_factor_parent_map(product_terms: Sequence[str]) -> Dict[str, List[str]]:
    parents: Dict[str, List[str]] = {}
    for term in product_terms or []:
        text = _s(term)
        match = _FACTOR_RE.search(text)
        if not match:
            continue
        node = _s(match.group(1))
        parent_text = _s(match.group(2))
        if not node or node.lower() == "x_prime":
            continue
        parent_tokens: List[str] = []
        if parent_text:
            for raw in parent_text.replace(",", "|").split("|"):
                item = _s(raw)
                if item and item.lower() != "x_prime":
                    parent_tokens.append(item)
        parents[node] = _dedupe(parent_tokens)
    return parents


def _fit_linear_node(df: pd.DataFrame, node: str, parents: Sequence[str]) -> Tuple[str, np.ndarray, List[str], int]:
    cols = [*parents, node]
    work = _numeric_frame(df, cols).dropna()
    if work.empty:
        return "missing_data", np.array([]), list(parents), 0
    y = work[node].to_numpy(dtype=float)
    if not parents:
        return "constant_mean", np.array([float(np.mean(y))]), [], int(len(work))
    X = work[list(parents)].to_numpy(dtype=float)
    X_design = np.column_stack([np.ones(len(X)), X])
    ridge = np.eye(X_design.shape[1]) * 1e-8
    ridge[0, 0] = 0.0
    try:
        beta = np.linalg.solve(X_design.T @ X_design + ridge, X_design.T @ y)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(X_design) @ y
    return "linear", beta, list(parents), int(len(work))


def _predict_node(values: pd.DataFrame, model: Tuple[str, np.ndarray, List[str], int]) -> np.ndarray:
    kind, beta, parents, _n = model
    if beta.size == 0:
        return np.full(len(values), np.nan)
    if kind == "constant_mean" or not parents:
        return np.full(len(values), float(beta[0]))
    if any(p not in values.columns for p in parents):
        return np.full(len(values), np.nan)
    X = values[list(parents)].to_numpy(dtype=float)
    X_design = np.column_stack([np.ones(len(X)), X])
    return X_design @ beta


def _fit_sequential_models(df: pd.DataFrame, order: Sequence[str], parent_map: Mapping[str, Sequence[str]], treatment: str) -> Tuple[Dict[str, Tuple[str, np.ndarray, List[str], int]], List[str], List[str]]:
    models: Dict[str, Tuple[str, np.ndarray, List[str], int]] = {}
    fit_nodes: List[str] = []
    skipped: List[str] = []
    for node in order:
        if node == treatment:
            continue
        parents = [p for p in parent_map.get(node, []) if p != node]
        if node not in df.columns:
            skipped.append(f"{node}:missing_node_column")
            continue
        missing_parents = [p for p in parents if p not in df.columns and p != treatment]
        if missing_parents:
            skipped.append(f"{node}:missing_parents={','.join(missing_parents)}")
            continue
        model = _fit_linear_node(df, node, parents)
        if model[0] == "missing_data":
            skipped.append(f"{node}:insufficient_numeric_data")
            continue
        models[node] = model
        fit_nodes.append(node)
    return models, fit_nodes, skipped


def _simulate_sequential_mean(
    df: pd.DataFrame,
    *,
    order: Sequence[str],
    parent_map: Mapping[str, Sequence[str]],
    treatment: str,
    outcome: str,
    value: float,
    models: Mapping[str, Tuple[str, np.ndarray, List[str], int]],
) -> float:
    involved = _dedupe([*order, treatment, outcome, *[p for ps in parent_map.values() for p in ps]])
    work = _numeric_frame(df, [c for c in involved if c in df.columns]).dropna()
    if work.empty or treatment not in work.columns:
        return float("nan")
    values = work.copy()
    values[treatment] = float(value)
    for node in order:
        if node == treatment:
            continue
        if node in models:
            pred = _predict_node(values, models[node])
            values[node] = pred
    if outcome not in values.columns:
        return float("nan")
    out = pd.to_numeric(values[outcome], errors="coerce").dropna()
    return float(out.mean()) if len(out) else float("nan")


def _bootstrap_sequential_ci(
    df: pd.DataFrame,
    *,
    order: Sequence[str],
    parent_map: Mapping[str, Sequence[str]],
    treatment: str,
    outcome: str,
    low: float,
    high: float,
    draws: int,
) -> Tuple[float, float, int]:
    if df is None or df.empty or len(df) < 8:
        return (float("nan"), float("nan"), 0)
    rng = np.random.default_rng(1805)
    vals: List[float] = []
    n = len(df)
    for _ in range(int(draws)):
        idx = rng.integers(0, n, size=n)
        sample = df.iloc[idx].reset_index(drop=True)
        models, _fit_nodes, skipped = _fit_sequential_models(sample, order, parent_map, treatment)
        if skipped or outcome not in models:
            continue
        lo = _simulate_sequential_mean(sample, order=order, parent_map=parent_map, treatment=treatment, outcome=outcome, value=low, models=models)
        hi = _simulate_sequential_mean(sample, order=order, parent_map=parent_map, treatment=treatment, outcome=outcome, value=high, models=models)
        if np.isfinite(hi - lo):
            vals.append(float(hi - lo))
    min_boot = min(5, max(2, int(draws)))
    if len(vals) < min_boot:
        return (float("nan"), float("nan"), len(vals))
    return (float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975)), len(vals))


def _gate_reasons(row: Mapping[str, object]) -> List[str]:
    reasons: List[str] = []
    if not has_canonical_id_authority(dict(row)):
        reasons.append("MISSING_CANONICAL_ID_AUTHORITY")
    if _s(row.get("authority_level")).lower() != "identified_estimable":
        reasons.append("CONTRACT_NOT_IDENTIFIED_ESTIMABLE")
    if not _truthy(row.get("estimation_enabled")):
        reasons.append("CONTRACT_ESTIMATION_NOT_ENABLED")
    if not _truthy(row.get("symbolic_formula_evaluable")):
        reasons.append("SYMBOLIC_FORMULA_NOT_EVALUABLE")
    if not _truthy(row.get("symbolic_numeric_estimator_ready")):
        reasons.append("SYMBOLIC_NUMERIC_ESTIMATOR_NOT_READY")
    route = _s(row.get("symbolic_estimator_route"))
    if route not in SUPPORTED_NUMERIC_ROUTES:
        reasons.append(f"UNSUPPORTED_SYMBOLIC_NUMERIC_ROUTE={route or 'missing'}")
    if _truthy(row.get("hedge_detected")):
        reasons.append("ID_HEDGE_DETECTED")
    for key in ["id_status", "recursive_id_status", "c_factor_status", "district_status"]:
        text = _s(row.get(key)).lower()
        if text.startswith("blocked") or "requires_symbolic_c_factor" in text or "possible_hedge" in text:
            reasons.append(f"{key.upper()}_BLOCKED")
    return reasons


def _base_diag(row: Mapping[str, object], treatment: str, outcome: str, route: str, formula_type: str, gate_reasons: Sequence[str]) -> Dict[str, object]:
    return {
        "effect_id": f"do:{treatment}->{outcome}",
        "treatment": treatment,
        "outcome": outcome,
        "contract_row_present": 1,
        "estimation_enabled": int(_truthy(row.get("estimation_enabled"))),
        "do_authorized": 0 if gate_reasons else 1,
        "do_mode": "symbolic_numeric" if not gate_reasons else "blocked",
        "overlap_pass": 0,
        "support_n_low": 0,
        "support_n_high": 0,
        "adjustment_set_status": _s(row.get("adjustment_set_status")),
        "adjustment_columns_missing": "",
        "data_columns_missing": "",
        "analysis_policy": "strict_contract_symbolic_numeric",
        "diagnostic_estimation_allowed": 0,
        "diagnostic_authority_level": "",
        "causal_authority_from_diagnostic": 0,
        "diagnostic_notes": "|".join(gate_reasons),
        "symbolic_numeric_version": SYMBOLIC_NUMERIC_VERSION,
        "symbolic_estimator_route": route,
        "symbolic_formula_type": formula_type,
        "topological_order": "",
        "symbolic_gate_pass": 0 if gate_reasons else 1,
        "symbolic_gate_reasons": "|".join(gate_reasons),
        "model_fit_nodes": "",
        "model_skipped_nodes": "",
    }


def estimate_graphical_zero(row: Mapping[str, object], data: pd.DataFrame, payload: Mapping[str, object]) -> Tuple[Dict[str, object], Dict[str, object]]:
    estimand = payload.get("estimand", {}) if isinstance(payload.get("estimand"), Mapping) else {}
    treatment = _s(estimand.get("intervention")) or _s(row.get("treatment_col") or row.get("source"))
    outcome = _s(estimand.get("outcome")) or _s(row.get("outcome_col") or row.get("target"))
    route = _s(row.get("symbolic_estimator_route"))
    gate = _gate_reasons(row)
    diag = _base_diag(row, treatment, outcome, route, "graphical_zero_effect", gate)
    if gate:
        est = _blocked_estimate(row, treatment, outcome, route, "graphical_zero_effect", "|".join(gate))
        return est, diag
    n = int(len(data)) if data is not None else 0
    outcome_mean = float(pd.to_numeric(data[outcome], errors="coerce").mean()) if data is not None and outcome in data.columns else float("nan")
    est = {
        "effect_id": f"do:{treatment}->{outcome}",
        "treatment": treatment,
        "outcome": outcome,
        "do_value_low": "",
        "do_value_high": "",
        "estimand": "graphical zero effect: no directed path from treatment to outcome",
        "identification_strategy": "graphical_zero_effect",
        "adjustment_set": "",
        "adjustment_set_status": "not_needed",
        "do_authorized": 1,
        "do_mode": "symbolic_numeric_graphical_zero",
        "effect_estimate": 0.0,
        "mean_do_low": outcome_mean,
        "mean_do_high": outcome_mean,
        "ci_low": 0.0,
        "ci_high": 0.0,
        "n": n,
        "support_n_low": 0,
        "support_n_high": 0,
        "authority_level": _s(row.get("authority_level")),
        "effect_semantics": "identified_graphical_zero_no_directed_path_contract_authorized",
        "analysis_policy": "strict_contract_symbolic_numeric",
        "diagnostic_estimation_allowed": 0,
        "diagnostic_authority_level": "",
        "causal_authority_from_diagnostic": 0,
        "reason_codes": "SYMBOLIC_GRAPHICAL_ZERO_CONTRACT_AUTHORIZED",
        "bootstrap_draws": 0,
        "bootstrap_success_n": 0,
        "bootstrap_status": "not_needed_graphical_zero",
        "ci_width": 0.0,
        "ci_width_to_effect_ratio": 0.0,
        "robustness_status": "ok",
        "overlap_score": 1.0,
        "support_min": n,
        "support_ratio": 1.0 if n else 0.0,
        "support_n_mid": n,
        "treatment_unique_n": 0,
        "extrapolation_risk": "none_graphical_zero",
        "sensitivity_warning": "",
        "symbolic_numeric_version": SYMBOLIC_NUMERIC_VERSION,
        "symbolic_formula_kind": _s(row.get("symbolic_formula_kind")),
        "symbolic_estimator_route": route,
        "symbolic_formula_type": "graphical_zero_effect",
        "topological_order": "",
        "symbolic_sum_over": _s(row.get("symbolic_sum_over")),
        "symbolic_product_terms": _s(row.get("symbolic_product_terms")),
    }
    diag.update(
        overlap_pass=1,
        diagnostic_notes=est["reason_codes"],
        bootstrap_draws=0,
        bootstrap_success_n=0,
        bootstrap_status="not_needed_graphical_zero",
        ci_width=0.0,
        ci_width_to_effect_ratio=0.0,
        robustness_status="ok",
        overlap_score=1.0,
        support_min=n,
        support_ratio=1.0 if n else 0.0,
        support_n_mid=n,
        treatment_unique_n=0,
        extrapolation_risk="none_graphical_zero",
        sensitivity_warning="",
    )
    return est, diag


def _blocked_estimate(row: Mapping[str, object], treatment: str, outcome: str, route: str, formula_type: str, reason: str) -> Dict[str, object]:
    return {
        "effect_id": f"do:{treatment}->{outcome}",
        "treatment": treatment,
        "outcome": outcome,
        "do_authorized": 0,
        "do_mode": "blocked",
        "identification_strategy": _s(row.get("identification_strategy")) or _s(row.get("id_status")),
        "authority_level": _s(row.get("authority_level")),
        "effect_semantics": "blocked_symbolic_numeric_contract_gate",
        "analysis_policy": "strict_contract_symbolic_numeric",
        "diagnostic_estimation_allowed": 0,
        "diagnostic_authority_level": "",
        "causal_authority_from_diagnostic": 0,
        "reason_codes": reason,
        "bootstrap_draws": "",
        "bootstrap_success_n": "",
        "bootstrap_status": "blocked",
        "ci_width": "",
        "ci_width_to_effect_ratio": "",
        "robustness_status": "blocked",
        "overlap_score": "",
        "support_min": "",
        "support_ratio": "",
        "support_n_mid": "",
        "treatment_unique_n": "",
        "extrapolation_risk": "",
        "sensitivity_warning": reason,
        "symbolic_numeric_version": SYMBOLIC_NUMERIC_VERSION,
        "symbolic_formula_kind": _s(row.get("symbolic_formula_kind")),
        "symbolic_estimator_route": route,
        "symbolic_formula_type": formula_type,
        "topological_order": "",
        "symbolic_sum_over": _s(row.get("symbolic_sum_over")),
        "symbolic_product_terms": _s(row.get("symbolic_product_terms")),
    }



def _fit_conditional_outcome_model(df: pd.DataFrame, outcome: str, predictors: Sequence[str]) -> Tuple[str, np.ndarray, List[str], int]:
    return _fit_linear_node(df, outcome, _dedupe(predictors))


def _simulate_conditional_mean(
    df: pd.DataFrame,
    *,
    treatment: str,
    outcome: str,
    conditions: Sequence[str],
    value: float,
    model: Tuple[str, np.ndarray, List[str], int],
) -> float:
    required = _dedupe([treatment, outcome, *conditions])
    work = _numeric_frame(df, [c for c in required if c in df.columns]).dropna()
    if work.empty or treatment not in work.columns or outcome not in work.columns:
        return float("nan")
    values = work.copy()
    values[treatment] = float(value)
    pred = _predict_node(values, model)
    pred = pd.to_numeric(pd.Series(pred), errors="coerce").dropna()
    return float(pred.mean()) if len(pred) else float("nan")


def _bootstrap_idc_fraction_ci(
    df: pd.DataFrame,
    *,
    treatment: str,
    outcome: str,
    conditions: Sequence[str],
    low: float,
    high: float,
    draws: int,
) -> Tuple[float, float, int]:
    if df is None or df.empty or len(df) < 8:
        return (float("nan"), float("nan"), 0)
    rng = np.random.default_rng(1805)
    vals: List[float] = []
    n = len(df)
    predictors = _dedupe([treatment, *conditions])
    for _ in range(int(draws)):
        idx = rng.integers(0, n, size=n)
        sample = df.iloc[idx].reset_index(drop=True)
        model = _fit_conditional_outcome_model(sample, outcome, predictors)
        if model[0] == "missing_data" or model[1].size == 0:
            continue
        lo = _simulate_conditional_mean(sample, treatment=treatment, outcome=outcome, conditions=conditions, value=low, model=model)
        hi = _simulate_conditional_mean(sample, treatment=treatment, outcome=outcome, conditions=conditions, value=high, model=model)
        if np.isfinite(hi - lo):
            vals.append(float(hi - lo))
    min_boot = min(5, max(2, int(draws)))
    if len(vals) < min_boot:
        return (float("nan"), float("nan"), len(vals))
    return (float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975)), len(vals))


def estimate_idc_fraction_ratio_ast(row: Mapping[str, object], data: pd.DataFrame, ast_payload: object, *, bootstrap_draws: int = 80) -> Tuple[Dict[str, object], Dict[str, object]]:
    """Evaluate resolved IDC ratio AST through a guarded conditional-mean route.

    This is deliberately conservative: it does not estimate arbitrary symbolic
    distributions.  It uses the IDC fraction AST only as authorization that the
    conditional interventional estimand is identified, then fits a linear
    conditional mean model for a scalar contrast over the observed condition
    distribution.  Weak support/overlap blocks authorization exactly like other
    symbolic numeric routes.
    """
    route = _s(row.get("symbolic_estimator_route")) or "symbolic_numeric_idc_fraction_ratio"
    treatment_hint = _as_list(row.get("treatments") or row.get("treatment_col") or row.get("treatment") or row.get("source"))
    outcome_hint = _as_list(row.get("outcomes") or row.get("outcome_col") or row.get("outcome") or row.get("target"))
    treatment = treatment_hint[0] if treatment_hint else _s(row.get("source"))
    outcome = outcome_hint[0] if outcome_hint else _s(row.get("target"))
    formula_type = "idc_fraction_ratio_ast"
    plan = analyze_idc_fraction_ast(ast_payload, outcome_hint=outcome_hint, treatment_hint=treatment_hint)
    if plan.intervention_variables and not treatment:
        treatment = plan.intervention_variables[0]
    if plan.outcome_variables and not outcome:
        outcome = plan.outcome_variables[0]
    conditions = [c for c in plan.condition_variables if c not in {treatment, outcome}]
    gate = _gate_reasons(row)
    if not plan.numeric_ready:
        gate.append(plan.blocker or plan.reason_codes or "IDC_FRACTION_NUMERIC_PLAN_NOT_READY")
    diag = _base_diag(row, treatment, outcome, route, formula_type, gate)
    diag["topological_order"] = "|".join(_dedupe([treatment, *conditions, outcome]))
    diag["model_fit_nodes"] = ""
    diag["model_skipped_nodes"] = ""
    if gate:
        est = _blocked_estimate(row, treatment, outcome, route, formula_type, "|".join(_dedupe(gate)))
        est["topological_order"] = diag["topological_order"]
        est["symbolic_sum_over"] = "|".join(plan.denominator_sum_over)
        est["symbolic_product_terms"] = "|".join(plan.node_types)
        return est, diag
    if data is None or data.empty:
        reason = "MISSING_DATA"
        diag.update(data_columns_missing=reason, diagnostic_notes=reason)
        est = _blocked_estimate(row, treatment, outcome, route, formula_type, reason)
        est["topological_order"] = diag["topological_order"]
        return est, diag
    required = _dedupe([treatment, outcome, *conditions])
    missing = [c for c in required if c not in data.columns]
    if missing:
        reason = "MISSING_REQUIRED_COLUMNS=" + ",".join(missing)
        diag.update(data_columns_missing=",".join(missing), diagnostic_notes=reason)
        est = _blocked_estimate(row, treatment, outcome, route, formula_type, reason)
        est["topological_order"] = diag["topological_order"]
        return est, diag
    numeric = _numeric_frame(data, required).dropna()
    if len(numeric) < 8:
        reason = "INSUFFICIENT_NUMERIC_ROWS"
        diag.update(diagnostic_notes=reason)
        est = _blocked_estimate(row, treatment, outcome, route, formula_type, reason)
        est["topological_order"] = diag["topological_order"]
        return est, diag
    low, high = _treatment_values(numeric[treatment])
    support_diag = _support_diagnostics(numeric[treatment], low, high)
    predictors = _dedupe([treatment, *conditions])
    model = _fit_conditional_outcome_model(numeric, outcome, predictors)
    fit_nodes = [outcome] if model[0] != "missing_data" and model[1].size else []
    skipped = [] if fit_nodes else [f"{outcome}:conditional_model_not_fit"]
    diag.update(
        model_fit_nodes="|".join(fit_nodes),
        model_skipped_nodes="|".join(skipped),
        support_n_low=support_diag["support_n_low"],
        support_n_high=support_diag["support_n_high"],
        support_n_mid=support_diag["support_n_mid"],
        support_min=support_diag["support_min"],
        support_ratio=support_diag["support_ratio"],
        overlap_score=support_diag["overlap_score"],
        treatment_unique_n=support_diag["treatment_unique_n"],
        extrapolation_risk=support_diag["extrapolation_risk"],
    )
    if skipped or not np.isfinite(low) or not np.isfinite(high):
        reason_bits = []
        if skipped:
            reason_bits.append("MODEL_SKIPPED_NODES=" + ";".join(skipped))
        if not np.isfinite(low) or not np.isfinite(high):
            reason_bits.append("INVALID_TREATMENT_VALUES")
        reason = "|".join(reason_bits) or "IDC_FRACTION_MODEL_NOT_READY"
        diag.update(diagnostic_notes=reason, robustness_status="blocked_model_not_ready", sensitivity_warning=reason)
        est = _blocked_estimate(row, treatment, outcome, route, formula_type, reason)
        est.update({k: support_diag.get(k, "") for k in ["support_n_mid", "support_min", "support_ratio", "overlap_score", "treatment_unique_n", "extrapolation_risk"]})
        est["topological_order"] = diag["topological_order"]
        return est, diag
    mean_low = _simulate_conditional_mean(numeric, treatment=treatment, outcome=outcome, conditions=conditions, value=low, model=model)
    mean_high = _simulate_conditional_mean(numeric, treatment=treatment, outcome=outcome, conditions=conditions, value=high, model=model)
    effect = float(mean_high - mean_low) if np.isfinite(mean_high - mean_low) else float("nan")
    ci_low, ci_high, bootstrap_success_n = _bootstrap_idc_fraction_ci(numeric, treatment=treatment, outcome=outcome, conditions=conditions, low=low, high=high, draws=bootstrap_draws)
    outcome_sd = float(pd.to_numeric(numeric[outcome], errors="coerce").std()) if outcome in numeric.columns else float("nan")
    robust = _robustness_diagnostics(effect=effect, ci_low=ci_low, ci_high=ci_high, bootstrap_draws=bootstrap_draws, bootstrap_success_n=bootstrap_success_n, support_diag=support_diag, outcome_sd=outcome_sd)
    robust_status = _s(robust.get("robustness_status"))
    do_authorized = 1 if robust_status == "ok" else 0
    do_mode = "symbolic_numeric_idc_fraction_ratio" if robust_status == "ok" else robust_status
    reason_codes = [
        "IDC_FRACTION_RATIO_AST_CONDITIONAL_MEAN_STANDARDIZATION_CONTRACT_AUTHORIZED",
        IDC_FRACTION_NUMERIC_VERSION,
    ]
    if robust_status != "ok":
        reason_codes.append("IDC_FRACTION_NUMERIC_ROBUSTNESS_" + robust_status.upper())
    if robust.get("sensitivity_warning"):
        reason_codes.extend(_as_list(robust.get("sensitivity_warning")))
    est = {
        "effect_id": f"do:{treatment}->{outcome}",
        "treatment": treatment,
        "outcome": outcome,
        "do_value_low": low,
        "do_value_high": high,
        "estimand": f"IDC fraction ratio conditional mean contrast: E[{outcome}|do({treatment}=high), conditions] - E[{outcome}|do({treatment}=low), conditions]",
        "identification_strategy": "idc_fraction_ratio_ast",
        "adjustment_set": "|".join(conditions),
        "adjustment_set_status": "idc_conditions_from_fraction_ast" if conditions else "idc_no_conditions_after_pruning",
        "do_authorized": do_authorized,
        "do_mode": do_mode,
        "effect_estimate": effect,
        "mean_do_low": mean_low,
        "mean_do_high": mean_high,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n": int(len(numeric)),
        "support_n_low": support_diag["support_n_low"],
        "support_n_high": support_diag["support_n_high"],
        "authority_level": _s(row.get("authority_level")),
        "effect_semantics": "idc_fraction_ratio_conditional_mean_standardization" if do_authorized else "diagnostic_or_blocked_idc_fraction_ratio_numeric_gate",
        "analysis_policy": "strict_contract_symbolic_numeric_idc_fraction_step65",
        "diagnostic_estimation_allowed": 0,
        "diagnostic_authority_level": "",
        "causal_authority_from_diagnostic": 0,
        "reason_codes": "|".join(_dedupe(reason_codes)),
        "symbolic_numeric_version": SYMBOLIC_NUMERIC_VERSION,
        "symbolic_formula_kind": _s(row.get("symbolic_formula_kind")),
        "symbolic_estimator_route": route,
        "symbolic_formula_type": formula_type,
        "topological_order": diag["topological_order"],
        "symbolic_sum_over": "|".join(plan.denominator_sum_over),
        "symbolic_product_terms": "|".join(plan.node_types),
        **robust,
    }
    diag.update(
        do_authorized=do_authorized,
        do_mode=do_mode,
        overlap_pass=int(float(robust.get("overlap_score") or 0.0) >= 1.0 and np.isfinite(effect)),
        diagnostic_notes=est["reason_codes"],
        **robust,
    )
    return est, diag


def estimate_truncated_factorization(row: Mapping[str, object], data: pd.DataFrame, payload: Mapping[str, object], *, bootstrap_draws: int = 80) -> Tuple[Dict[str, object], Dict[str, object]]:
    estimand = payload.get("estimand", {}) if isinstance(payload.get("estimand"), Mapping) else {}
    treatment = _s(estimand.get("intervention")) or _s(row.get("treatment_col") or row.get("source"))
    outcome = _s(estimand.get("outcome")) or _s(row.get("outcome_col") or row.get("target"))
    route = _s(row.get("symbolic_estimator_route"))
    formula_type = "truncated_factorization"
    gate = _gate_reasons(row)
    diag = _base_diag(row, treatment, outcome, route, formula_type, gate)
    product_terms = _as_list(payload.get("product_terms")) or _as_list(row.get("symbolic_product_terms"))
    order = _as_list(payload.get("topological_order"))
    if not order:
        order = _dedupe([treatment, *[n for n in _parse_factor_parent_map(product_terms).keys()]])
    diag["topological_order"] = "|".join(order)
    if gate:
        est = _blocked_estimate(row, treatment, outcome, route, formula_type, "|".join(gate))
        est["topological_order"] = "|".join(order)
        return est, diag
    if data is None or data.empty:
        reason = "MISSING_DATA"
        diag.update(data_columns_missing=reason, diagnostic_notes=reason)
        est = _blocked_estimate(row, treatment, outcome, route, formula_type, reason)
        est["topological_order"] = "|".join(order)
        return est, diag
    parent_map = _parse_factor_parent_map(product_terms)
    required = _dedupe([treatment, outcome, *order, *[p for ps in parent_map.values() for p in ps]])
    missing = [c for c in required if c not in data.columns]
    if missing:
        reason = "MISSING_REQUIRED_COLUMNS=" + ",".join(missing)
        diag.update(data_columns_missing=",".join(missing), diagnostic_notes=reason)
        est = _blocked_estimate(row, treatment, outcome, route, formula_type, reason)
        est["topological_order"] = "|".join(order)
        return est, diag
    numeric = _numeric_frame(data, required).dropna()
    if len(numeric) < 8:
        reason = "INSUFFICIENT_NUMERIC_ROWS"
        diag.update(data_columns_missing="", diagnostic_notes=reason)
        est = _blocked_estimate(row, treatment, outcome, route, formula_type, reason)
        est["topological_order"] = "|".join(order)
        return est, diag
    low, high = _treatment_values(numeric[treatment])
    support_diag = _support_diagnostics(numeric[treatment], low, high)
    models, fit_nodes, skipped = _fit_sequential_models(numeric, order, parent_map, treatment)
    diag.update(
        model_fit_nodes="|".join(fit_nodes),
        model_skipped_nodes="|".join(skipped),
        support_n_low=support_diag["support_n_low"],
        support_n_high=support_diag["support_n_high"],
        support_n_mid=support_diag["support_n_mid"],
        support_min=support_diag["support_min"],
        support_ratio=support_diag["support_ratio"],
        overlap_score=support_diag["overlap_score"],
        treatment_unique_n=support_diag["treatment_unique_n"],
        extrapolation_risk=support_diag["extrapolation_risk"],
    )
    if skipped or outcome not in models or not np.isfinite(low) or not np.isfinite(high):
        reason_bits = []
        if skipped:
            reason_bits.append("MODEL_SKIPPED_NODES=" + ";".join(skipped))
        if outcome not in models:
            reason_bits.append("OUTCOME_MODEL_MISSING")
        if not np.isfinite(low) or not np.isfinite(high):
            reason_bits.append("INVALID_TREATMENT_VALUES")
        reason = "|".join(reason_bits) or "SYMBOLIC_NUMERIC_MODEL_NOT_READY"
        diag.update(diagnostic_notes=reason, robustness_status="blocked_model_not_ready", sensitivity_warning=reason)
        est = _blocked_estimate(row, treatment, outcome, route, formula_type, reason)
        est.update({k: support_diag.get(k, "") for k in ["support_n_mid", "support_min", "support_ratio", "overlap_score", "treatment_unique_n", "extrapolation_risk"]})
        est["robustness_status"] = "blocked_model_not_ready"
        est["sensitivity_warning"] = reason
        est["topological_order"] = "|".join(order)
        return est, diag
    mean_low = _simulate_sequential_mean(numeric, order=order, parent_map=parent_map, treatment=treatment, outcome=outcome, value=low, models=models)
    mean_high = _simulate_sequential_mean(numeric, order=order, parent_map=parent_map, treatment=treatment, outcome=outcome, value=high, models=models)
    effect = float(mean_high - mean_low) if np.isfinite(mean_high - mean_low) else float("nan")
    ci_low, ci_high, bootstrap_success_n = _bootstrap_sequential_ci(
        numeric, order=order, parent_map=parent_map, treatment=treatment, outcome=outcome, low=low, high=high, draws=bootstrap_draws
    )
    outcome_sd = float(pd.to_numeric(numeric[outcome], errors="coerce").std()) if outcome in numeric.columns else float("nan")
    robust = _robustness_diagnostics(
        effect=effect,
        ci_low=ci_low,
        ci_high=ci_high,
        bootstrap_draws=bootstrap_draws,
        bootstrap_success_n=bootstrap_success_n,
        support_diag=support_diag,
        outcome_sd=outcome_sd,
    )
    overlap_pass = int(float(robust.get("overlap_score") or 0.0) >= 1.0 and np.isfinite(effect))
    robust_status = _s(robust.get("robustness_status"))
    do_authorized = 1 if robust_status == "ok" else 0
    do_mode = "symbolic_numeric_truncated_factorization" if robust_status == "ok" else robust_status
    reason_codes = ["SYMBOLIC_TRUNCATED_FACTORIZATION_SEQUENTIAL_STANDARDIZATION_CONTRACT_AUTHORIZED"]
    if robust_status != "ok":
        reason_codes.append("SYMBOLIC_NUMERIC_ROBUSTNESS_" + robust_status.upper())
    if robust.get("sensitivity_warning"):
        reason_codes.extend(_as_list(robust.get("sensitivity_warning")))
    est = {
        "effect_id": f"do:{treatment}->{outcome}",
        "treatment": treatment,
        "outcome": outcome,
        "do_value_low": low,
        "do_value_high": high,
        "estimand": "observed-DAG g-formula sequential standardization: E[Y|do(X=high)] - E[Y|do(X=low)]",
        "identification_strategy": "observed_dag_truncated_factorization",
        "adjustment_set": "",
        "adjustment_set_status": "not_needed_observed_dag_g_formula",
        "do_authorized": do_authorized,
        "do_mode": do_mode,
        "effect_estimate": effect,
        "mean_do_low": mean_low,
        "mean_do_high": mean_high,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n": int(len(numeric)),
        "support_n_low": support_diag["support_n_low"],
        "support_n_high": support_diag["support_n_high"],
        "authority_level": _s(row.get("authority_level")),
        "effect_semantics": (
            "observed_dag_g_formula_sequential_standardization_contract_authorized"
            if do_authorized
            else "diagnostic_or_blocked_symbolic_numeric_robustness_gate"
        ),
        "analysis_policy": "strict_contract_symbolic_numeric",
        "diagnostic_estimation_allowed": 0,
        "diagnostic_authority_level": "",
        "causal_authority_from_diagnostic": 0,
        "reason_codes": "|".join(reason_codes),
        "symbolic_numeric_version": SYMBOLIC_NUMERIC_VERSION,
        "symbolic_formula_kind": _s(row.get("symbolic_formula_kind")),
        "symbolic_estimator_route": route,
        "symbolic_formula_type": formula_type,
        "topological_order": "|".join(order),
        "symbolic_sum_over": "|".join(_as_list(payload.get("sum_over")) or _as_list(row.get("symbolic_sum_over"))),
        "symbolic_product_terms": "|".join(product_terms),
        **robust,
    }
    diag.update(
        do_authorized=do_authorized,
        do_mode=do_mode,
        overlap_pass=overlap_pass,
        diagnostic_notes=est["reason_codes"],
        **robust,
    )
    return est, diag


def estimate_symbolic_numeric_effects(out_dir: str = "out", data_path: Optional[str] = None, contract_path: Optional[str] = None, bootstrap_draws: int = 80) -> Tuple[pd.DataFrame, pd.DataFrame]:
    contract = load_causal_contract(out_dir, contract_path)
    data = _load_data(data_path=data_path, out_dir=out_dir)
    estimates: List[Dict[str, object]] = []
    diagnostics: List[Dict[str, object]] = []
    if contract is None or contract.empty:
        return pd.DataFrame(columns=SYMBOLIC_NUMERIC_ESTIMATE_COLUMNS), pd.DataFrame(columns=SYMBOLIC_NUMERIC_DIAGNOSTIC_COLUMNS)
    for _, series in contract.iterrows():
        row = series.to_dict()
        route = _s(row.get("symbolic_estimator_route"))
        if route not in SUPPORTED_NUMERIC_ROUTES:
            continue
        payload, err = parse_symbolic_formula_json(row.get("symbolic_formula_json"))
        if route == "symbolic_numeric_idc_fraction_ratio":
            ast, ast_err = parse_formula_ast_json(row.get("formula_ast_json"))
            if ast is None:
                treatment = _s(row.get("treatment_col") or row.get("source"))
                outcome = _s(row.get("outcome_col") or row.get("target"))
                est = _blocked_estimate(row, treatment, outcome, route, "invalid_formula_ast_json", ast_err or "INVALID_FORMULA_AST_JSON")
                diag = _base_diag(row, treatment, outcome, route, "invalid_formula_ast_json", [ast_err or "INVALID_FORMULA_AST_JSON"])
            else:
                est, diag = estimate_idc_fraction_ratio_ast(row, data, ast, bootstrap_draws=bootstrap_draws)
        elif payload is None:
            treatment = _s(row.get("treatment_col") or row.get("source"))
            outcome = _s(row.get("outcome_col") or row.get("target"))
            est = _blocked_estimate(row, treatment, outcome, route, "invalid_symbolic_json", err or "INVALID_SYMBOLIC_JSON")
            diag = _base_diag(row, treatment, outcome, route, "invalid_symbolic_json", [err or "INVALID_SYMBOLIC_JSON"])
        else:
            formula_type = _s(payload.get("type"))
            if formula_type == "truncated_factorization":
                est, diag = estimate_truncated_factorization(row, data, payload, bootstrap_draws=bootstrap_draws)
            elif formula_type == "graphical_zero_effect":
                est, diag = estimate_graphical_zero(row, data, payload)
            else:
                estimand = payload.get("estimand", {}) if isinstance(payload.get("estimand"), Mapping) else {}
                treatment = _s(estimand.get("intervention")) or _s(row.get("treatment_col") or row.get("source"))
                outcome = _s(estimand.get("outcome")) or _s(row.get("outcome_col") or row.get("target"))
                reason = f"UNSUPPORTED_SYMBOLIC_NUMERIC_FORMULA_TYPE={formula_type or 'missing'}"
                est = _blocked_estimate(row, treatment, outcome, route, formula_type, reason)
                diag = _base_diag(row, treatment, outcome, route, formula_type, [reason])
        estimates.append(est)
        diagnostics.append(diag)
    return (
        pd.DataFrame(estimates, columns=SYMBOLIC_NUMERIC_ESTIMATE_COLUMNS),
        pd.DataFrame(diagnostics, columns=SYMBOLIC_NUMERIC_DIAGNOSTIC_COLUMNS),
    )


def symbolic_numeric_summary(estimates: pd.DataFrame, diagnostics: pd.DataFrame) -> Dict[str, object]:
    est_rows = [] if estimates is None or estimates.empty else estimates.to_dict("records")
    diag_rows = [] if diagnostics is None or diagnostics.empty else diagnostics.to_dict("records")
    status_counts: Dict[str, int] = {}
    route_counts: Dict[str, int] = {}
    robustness_counts: Dict[str, int] = {}
    bootstrap_counts: Dict[str, int] = {}
    for row in est_rows:
        status = _s(row.get("do_mode")) or "missing"
        route = _s(row.get("symbolic_estimator_route")) or "none"
        robustness = _s(row.get("robustness_status")) or "missing"
        bootstrap = _s(row.get("bootstrap_status")) or "missing"
        status_counts[status] = status_counts.get(status, 0) + 1
        route_counts[route] = route_counts.get(route, 0) + 1
        robustness_counts[robustness] = robustness_counts.get(robustness, 0) + 1
        bootstrap_counts[bootstrap] = bootstrap_counts.get(bootstrap, 0) + 1
    return {
        "symbolic_numeric_version": SYMBOLIC_NUMERIC_VERSION,
        "n_estimate_rows": len(est_rows),
        "n_diagnostic_rows": len(diag_rows),
        "n_authorized_symbolic_numeric": sum(1 for r in est_rows if _truthy(r.get("do_authorized"))),
        "n_blocked_or_diagnostic_by_robustness": sum(1 for r in est_rows if _s(r.get("robustness_status")) not in {"", "ok"}),
        "do_mode_counts": status_counts,
        "symbolic_estimator_route_counts": route_counts,
        "robustness_status_counts": robustness_counts,
        "bootstrap_status_counts": bootstrap_counts,
        "policy": "evaluates only causal_contract-enabled symbolic formulas and resolved IDC fraction ASTs; unresolved Q/c-factor formulas remain blocked; weak numeric support degrades to diagnostic/blocked outputs",
    }


def write_symbolic_numeric_outputs(out_dir: str = "out", data_path: Optional[str] = None, contract_path: Optional[str] = None, bootstrap_draws: int = 80) -> Dict[str, str]:
    estimates, diagnostics = estimate_symbolic_numeric_effects(out_dir=out_dir, data_path=data_path, contract_path=contract_path, bootstrap_draws=bootstrap_draws)
    scm_dir = os.path.join(out_dir, "scm")
    estimates_path = os.path.join(scm_dir, "symbolic_numeric_estimates.csv")
    diagnostics_path = os.path.join(scm_dir, "symbolic_numeric_diagnostics.csv")
    summary_path = os.path.join(scm_dir, "symbolic_numeric_summary.json")
    _write_csv(estimates_path, estimates.to_dict("records"), SYMBOLIC_NUMERIC_ESTIMATE_COLUMNS)
    _write_csv(diagnostics_path, diagnostics.to_dict("records"), SYMBOLIC_NUMERIC_DIAGNOSTIC_COLUMNS)
    Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(symbolic_numeric_summary(estimates, diagnostics), f, ensure_ascii=False, indent=2)
    return {
        "symbolic_numeric_estimates": estimates_path,
        "symbolic_numeric_diagnostics": diagnostics_path,
        "symbolic_numeric_summary": summary_path,
    }


__all__ = [
    "SYMBOLIC_NUMERIC_VERSION",
    "SYMBOLIC_NUMERIC_ESTIMATE_COLUMNS",
    "SYMBOLIC_NUMERIC_DIAGNOSTIC_COLUMNS",
    "estimate_idc_fraction_ratio_ast",
    "estimate_symbolic_numeric_effects",
    "write_symbolic_numeric_outputs",
    "symbolic_numeric_summary",
]
