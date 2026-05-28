from __future__ import annotations

"""Deterministic estimation diagnostics for public/demo pipelines.

The diagnostics here are synthetic/demo diagnostics.  They make estimation,
negative controls, placebo checks, and basic sensitivity checks observable in
public demos without claiming real-world evidence.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Sequence
import math
import random


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _variance(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = _mean(values)
    return sum((x - mu) ** 2 for x in values) / (len(values) - 1)


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _clean_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _difference_in_means(rows: Sequence[Mapping[str, float]], treatment: str, outcome: str) -> Dict[str, Any]:
    treated = [float(row[outcome]) for row in rows if float(row.get(treatment, 0.0)) > 0.5 and outcome in row]
    control = [float(row[outcome]) for row in rows if float(row.get(treatment, 0.0)) <= 0.5 and outcome in row]
    if not treated or not control:
        return {"estimated": False, "effect_estimate": None, "ci_low": None, "ci_high": None, "treated_n": len(treated), "control_n": len(control), "support_n": len(treated) + len(control), "reason": "Need both treated and control rows."}
    effect = _mean(treated) - _mean(control)
    se = math.sqrt((_variance(treated) / len(treated)) + (_variance(control) / len(control)))
    return {"estimated": True, "effect_estimate": effect, "ci_low": effect - 1.96 * se, "ci_high": effect + 1.96 * se, "standard_error": se, "treated_n": len(treated), "control_n": len(control), "support_n": len(treated) + len(control), "treated_mean": _mean(treated), "control_mean": _mean(control)}


@dataclass(frozen=True)
class EstimationDiagnosticsResult:
    estimated: bool
    diagnostics_status: str
    treatment: str = "X"
    outcome: str = "Y"
    effect_estimate: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    support_n: int = 0
    treated_n: int = 0
    control_n: int = 0
    negative_control: Dict[str, Any] = field(default_factory=dict)
    placebo: Dict[str, Any] = field(default_factory=dict)
    sensitivity: Dict[str, Any] = field(default_factory=dict)
    robustness_status: str = ""
    negative_control_status: str = ""
    placebo_status: str = ""
    sensitivity_status: str = ""
    reason_codes: List[str] = field(default_factory=list)
    synthetic_data_summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def generate_synthetic_backdoor_rows(*, n: int = 240, seed: int = 1805) -> List[Dict[str, float]]:
    rng = random.Random(seed)
    rows: List[Dict[str, float]] = []
    for _ in range(n):
        z = rng.gauss(0.0, 1.0)
        propensity = 1.0 / (1.0 + math.exp(-0.85 * z))
        x = 1.0 if rng.random() < propensity else 0.0
        y = 0.12 * x + 0.35 * z + rng.gauss(0.0, 0.25)
        neg = 0.02 * z + rng.gauss(0.0, 0.24)
        future_x = 1.0 if rng.random() < 0.5 else 0.0
        rows.append({"Z": z, "X": x, "Y": y, "N": neg, "FutureX": future_x})
    return rows


def run_synthetic_demo_diagnostics(payload: Mapping[str, Any] | None = None) -> EstimationDiagnosticsResult:
    data = _as_dict(payload)
    treatment = _clean_str(data.get("treatment"), "X")
    outcome = _clean_str(data.get("outcome"), "Y")
    negative_control_outcome = _clean_str(data.get("negative_control_outcome"), "N")
    placebo_treatment = _clean_str(data.get("placebo_treatment"), "FutureX")
    n = int(data.get("support_n") or 240)
    seed = int(data.get("seed") or 1805)
    rows = generate_synthetic_backdoor_rows(n=n, seed=seed)

    # Synthetic rows use canonical X/Y columns.  The public hypothesis can use
    # domain-specific variable names; metadata records the original names.
    main = _difference_in_means(rows, "X", "Y")
    nc = _difference_in_means(rows, "X", negative_control_outcome if negative_control_outcome in {"N", "Y"} else "N")
    placebo = _difference_in_means(rows, placebo_treatment if placebo_treatment == "FutureX" else "FutureX", "Y")

    strata = [[row for row in rows if row["Z"] < 0.0], [row for row in rows if row["Z"] >= 0.0]]
    stratified = [_difference_in_means(stratum, "X", "Y") for stratum in strata]
    valid = [item for item in stratified if item.get("estimated")]
    stratified_mean = _mean([float(item["effect_estimate"]) for item in valid]) if valid else None
    main_effect = main.get("effect_estimate")
    sensitivity_delta = abs(float(main_effect) - float(stratified_mean)) if main_effect is not None and stratified_mean is not None else None

    nc_pass = bool(nc.get("estimated") and nc.get("ci_low") is not None and nc.get("ci_high") is not None and nc["ci_low"] <= 0.0 <= nc["ci_high"])
    placebo_pass = bool(placebo.get("estimated") and placebo.get("ci_low") is not None and placebo.get("ci_high") is not None and placebo["ci_low"] <= 0.0 <= placebo["ci_high"])
    sensitivity_pass = bool(sensitivity_delta is not None and sensitivity_delta < 0.12)

    reason_codes = ["SYNTHETIC_DEMO_DIAGNOSTICS_COMPUTED"]
    reason_codes.append("NEGATIVE_CONTROL_PASSED" if nc_pass else "NEGATIVE_CONTROL_ATTENTION_REQUIRED")
    reason_codes.append("PLACEBO_TEST_PASSED" if placebo_pass else "PLACEBO_TEST_ATTENTION_REQUIRED")
    reason_codes.append("SENSITIVITY_CHECK_PASSED" if sensitivity_pass else "SENSITIVITY_CHECK_ATTENTION_REQUIRED")
    diagnostics_status = "computed_passed" if nc_pass and placebo_pass and sensitivity_pass else "computed_attention_required"
    return EstimationDiagnosticsResult(
        estimated=bool(main.get("estimated")), diagnostics_status=diagnostics_status, treatment=treatment, outcome=outcome,
        effect_estimate=main.get("effect_estimate"), ci_low=main.get("ci_low"), ci_high=main.get("ci_high"),
        support_n=int(main.get("support_n") or len(rows)), treated_n=int(main.get("treated_n") or 0), control_n=int(main.get("control_n") or 0),
        negative_control={"tested": True, "passed": nc_pass, "outcome": negative_control_outcome, **nc},
        placebo={"tested": True, "passed": placebo_pass, "treatment": placebo_treatment, **placebo},
        sensitivity={"tested": True, "passed": sensitivity_pass, "method": "z_stratified_difference_in_means_delta", "stratified_effect_estimate": stratified_mean, "delta_from_main_effect": sensitivity_delta, "strata": stratified},
        robustness_status="computed_basic_synthetic_robustness_passed" if sensitivity_pass else "computed_basic_synthetic_robustness_attention_required",
        negative_control_status="computed_passed" if nc_pass else "computed_attention_required",
        placebo_status="computed_passed" if placebo_pass else "computed_attention_required",
        sensitivity_status="computed_passed" if sensitivity_pass else "computed_attention_required",
        reason_codes=reason_codes,
        synthetic_data_summary={"rows": len(rows), "seed": seed, "columns": ["X", "Y", "Z", "N", "FutureX"], "generator": "amantia.causal_core.estimation.diagnostics.generate_synthetic_backdoor_rows", "variable_mapping": {treatment: "X", outcome: "Y"}},
    )


def diagnostics_to_metadata(diagnostics: EstimationDiagnosticsResult) -> Dict[str, Any]:
    data = diagnostics.to_dict()
    return {
        "effect_estimate": data["effect_estimate"], "ci_low": data["ci_low"], "ci_high": data["ci_high"],
        "support_n": data["support_n"], "treated_n": data["treated_n"], "control_n": data["control_n"],
        "robustness_status": data["robustness_status"], "negative_control_status": data["negative_control_status"],
        "placebo_status": data["placebo_status"], "sensitivity_status": data["sensitivity_status"],
        "estimator": "synthetic_difference_in_means_with_diagnostics", "estimation_diagnostics": data,
    }


__all__ = ["EstimationDiagnosticsResult", "diagnostics_to_metadata", "generate_synthetic_backdoor_rows", "run_synthetic_demo_diagnostics"]
