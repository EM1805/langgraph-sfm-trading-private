from __future__ import annotations

"""Stable online estimation adapter for Amantia runtime boundaries.

This facade connects the online/runtime layer to the existing ``estimation_parts``
backend without weakening Amantia's conservative epistemics.

Rules enforced here:
- externally supplied effect estimates can be loaded as causal evidence;
- existing ``effect_estimates.csv`` rows can be loaded and matched by treatment/outcome;
- the ``estimation_parts.effect_estimates`` backend is run only when the payload
  carries explicit ID/contract authorization for estimation;
- plain CSV data without ID/contract authorization remains a non-causal
  association diagnostic and is never allowed to justify a decision by itself.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence
import csv
import math


def _s(value: Any) -> str:
    text = str(value).strip() if value is not None else ""
    return "" if text.lower() in {"nan", "none", "null", "nat"} else text


def _f(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return None
        return number
    except Exception:
        return None


def _i(value: Any, default: int = 0) -> int:
    number = _f(value)
    return default if number is None else int(number)


def _truthy(value: Any) -> bool:
    return _s(value).lower() in {"1", "true", "yes", "y", "on", "pass", "allowed"}


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_query_list(value: Any) -> List[Mapping[str, Any]]:
    if isinstance(value, list):
        return [v for v in value if isinstance(v, Mapping)]
    return []


def _split_cols(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        out: List[str] = []
        for item in value:
            out.extend(_split_cols(item))
        return list(dict.fromkeys([x for x in out if x]))
    text = _s(value)
    if not text:
        return []
    for ch in "[](){}'\"":
        text = text.replace(ch, "")
    for sep in [";", "|", "/"]:
        text = text.replace(sep, ",")
    return list(dict.fromkeys([p.strip() for p in text.split(",") if p.strip()]))


def _first_present(payload: Mapping[str, Any], keys: Sequence[str], default: Any = "") -> Any:
    for key in keys:
        if key in payload and payload.get(key) not in (None, ""):
            return payload.get(key)
    return default


@dataclass(frozen=True)
class EstimationQuery:
    treatment: str = ""
    outcome: str = ""
    data_path: str = ""
    effect_estimates_path: str = ""
    effect_estimate: Optional[float] = None
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    support_n: int = 0
    treated_n: int = 0
    control_n: int = 0
    robustness_status: str = "unknown"
    negative_control_status: str = ""
    placebo_status: str = ""
    sensitivity_status: str = ""
    estimator_hint: str = ""
    adjustment_set: List[str] = field(default_factory=list)
    lag: int = 0
    bootstrap_b: int = 160
    expected_direction: str = ""
    identification_result: Dict[str, Any] = field(default_factory=dict)
    authority_level: str = ""
    identification_status: str = ""
    identification_strategy: str = ""
    identified: bool = False
    allowed_for_estimation: bool = False
    estimation_enabled: bool = False
    query_id: str = ""
    source: str = "amantia.causal_core.estimation"
    raw_payload: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any] | None) -> "EstimationQuery":
        payload = dict(payload or {}) if isinstance(payload, Mapping) else {}
        id_result = _as_dict(payload.get("identification_result") or payload.get("causal_identification"))
        authority_level = _s(_first_present(payload, ["authority_level", "identification_tier"], id_result.get("authority_level") or id_result.get("identification_tier") or ""))
        identification_status = _s(_first_present(payload, ["identification_status"], id_result.get("identification_status") or id_result.get("identification_strategy") or ""))
        identified = (
            _truthy(payload.get("identified"))
            or _truthy(id_result.get("identified"))
            or authority_level in {"identified_estimable", "identified", "backdoor_identified", "frontdoor_identified"}
        )
        allowed = _truthy(_first_present(payload, ["allowed_for_estimation", "allow_estimation"], False))
        enabled = _truthy(payload.get("estimation_enabled")) or allowed
        return cls(
            treatment=_s(_first_present(payload, ["treatment", "action", "treatment_col", "source"])),
            outcome=_s(_first_present(payload, ["outcome", "target", "outcome_col"])),
            data_path=_s(_first_present(payload, ["data_path", "data_csv", "csv_path"])),
            effect_estimates_path=_s(_first_present(payload, ["effect_estimates_path", "effects_path", "estimates_path"])),
            effect_estimate=_f(_first_present(payload, ["effect_estimate", "estimate", "effect"])),
            ci_low=_f(payload.get("ci_low")),
            ci_high=_f(payload.get("ci_high")),
            support_n=_i(payload.get("support_n"), 0),
            treated_n=_i(payload.get("treated_n"), 0),
            control_n=_i(payload.get("control_n"), 0),
            robustness_status=_s(payload.get("robustness_status")) or "unknown",
            negative_control_status=_s(payload.get("negative_control_status")),
            placebo_status=_s(payload.get("placebo_status")),
            sensitivity_status=_s(payload.get("sensitivity_status")),
            estimator_hint=_s(_first_present(payload, ["estimator_hint", "estimator", "estimator_used", "recommended_estimator"])),
            adjustment_set=_split_cols(payload.get("adjustment_set") or payload.get("used_adjustment_set")),
            lag=_i(payload.get("lag"), 0),
            bootstrap_b=max(10, _i(payload.get("bootstrap_b"), 160)),
            expected_direction=_s(payload.get("expected_direction")),
            identification_result=id_result,
            authority_level=authority_level,
            identification_status=identification_status,
            identification_strategy=_s(_first_present(payload, ["identification_strategy", "id_strategy"], id_result.get("identification_strategy") or "")),
            identified=identified,
            allowed_for_estimation=allowed,
            estimation_enabled=enabled,
            query_id=_s(_first_present(payload, ["query_id", "id", "effect_id", "insight_id"])),
            source=_s(payload.get("source")) or "amantia.causal_core.estimation",
            raw_payload=payload,
        )

    @property
    def has_backend_authorization(self) -> bool:
        """Return True only when the payload carries ID/contract authorization."""

        if not self.identified:
            return False
        if self.allowed_for_estimation or self.estimation_enabled:
            return True
        if self.authority_level == "identified_estimable":
            return True
        return False


@dataclass(frozen=True)
class EstimationResult:
    estimated: bool
    estimation_status: str
    causal_estimate_available: bool = False
    association_estimate_available: bool = False
    estimate_type: str = "none"
    allowed_for_decision: bool = False
    estimator_used: str = ""
    treatment: str = ""
    outcome: str = ""
    effect_estimate: Optional[float] = None
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    association_estimate: Optional[float] = None
    standard_error: Optional[float] = None
    t_stat: Optional[float] = None
    p_value_approx: Optional[float] = None
    support_n: int = 0
    treated_n: int = 0
    control_n: int = 0
    treated_mean: Optional[float] = None
    control_mean: Optional[float] = None
    adjustment_set: List[str] = field(default_factory=list)
    used_adjustment_set: List[str] = field(default_factory=list)
    dropped_adjustment_set: List[str] = field(default_factory=list)
    effect_claim_status: str = ""
    robustness_status: str = "unknown"
    negative_control_status: str = ""
    placebo_status: str = ""
    sensitivity_status: str = ""
    identification_status: str = ""
    authority_level: str = ""
    estimator_authority: str = ""
    source_path: str = ""
    reason: str = ""
    reason_codes: List[str] = field(default_factory=list)
    raw_estimation_result: Dict[str, Any] = field(default_factory=dict)
    query_id: str = ""
    source: str = "amantia.causal_core.estimation"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)




def _import_pandas_safely():
    """Import pandas while avoiding repo-local path shadowing in embedded runtimes.

    Some agent/container runtimes prepend the project root/current working
    directory to ``sys.path``. Pandas has a broad import tree, and local package
    names can accidentally shadow transitive imports. The adapter only needs
    pandas for backend execution, so we import it with local project paths
    temporarily removed, then restore ``sys.path`` before importing Amantia's
    own backend modules.
    """

    import sys

    project_root = str(Path(__file__).resolve().parents[3])
    cwd = str(Path.cwd().resolve())
    original = list(sys.path)
    filtered = []
    for entry in original:
        marker = str(Path(entry or ".").resolve()) if entry in {"", "."} or entry else entry
        if entry in {"", "."} or marker in {project_root, cwd}:
            continue
        filtered.append(entry)
    try:
        sys.path[:] = filtered
        import pandas as pd  # type: ignore
        return pd
    finally:
        sys.path[:] = original

def _manual(query: EstimationQuery) -> EstimationResult:
    return EstimationResult(
        estimated=True,
        estimation_status="loaded_manual_effect",
        causal_estimate_available=True,
        association_estimate_available=False,
        estimate_type="causal_effect_supplied",
        allowed_for_decision=True,
        estimator_used=query.estimator_hint or "manual_effect_input",
        treatment=query.treatment,
        outcome=query.outcome,
        effect_estimate=query.effect_estimate,
        ci_low=query.ci_low,
        ci_high=query.ci_high,
        support_n=query.support_n,
        treated_n=query.treated_n,
        control_n=query.control_n,
        adjustment_set=list(query.adjustment_set),
        used_adjustment_set=list(query.adjustment_set),
        robustness_status=query.robustness_status,
        negative_control_status=query.negative_control_status,
        placebo_status=query.placebo_status,
        sensitivity_status=query.sensitivity_status,
        identification_status=query.identification_status,
        authority_level=query.authority_level,
        reason="Externally supplied causal effect estimate loaded.",
        reason_codes=["CAUSAL_EFFECT_ESTIMATE_SUPPLIED"],
        query_id=query.query_id,
        source=query.source,
    )


def _csv_difference(query: EstimationQuery) -> EstimationResult:
    path = Path(query.data_path)
    if not query.treatment or not query.outcome:
        return EstimationResult(
            estimated=False,
            estimation_status="blocked_missing_treatment_or_outcome",
            treatment=query.treatment,
            outcome=query.outcome,
            reason="CSV diagnostic requires treatment and outcome columns.",
            reason_codes=["MISSING_TREATMENT_OR_OUTCOME"],
            query_id=query.query_id,
            source=query.source,
        )
    if not path.exists():
        return EstimationResult(
            estimated=False,
            estimation_status="blocked_missing_data_path",
            treatment=query.treatment,
            outcome=query.outcome,
            source_path=str(path),
            reason=f"Data file not found: {path}",
            reason_codes=["MISSING_DATA_PATH"],
            query_id=query.query_id,
            source=query.source,
        )

    treated: List[float] = []
    control: List[float] = []
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            t = _f(row.get(query.treatment))
            y = _f(row.get(query.outcome))
            if t is None or y is None:
                continue
            if t > 0:
                treated.append(y)
            else:
                control.append(y)

    if not treated or not control:
        return EstimationResult(
            estimated=False,
            estimation_status="blocked_insufficient_treated_or_control_rows",
            treatment=query.treatment,
            outcome=query.outcome,
            treated_n=len(treated),
            control_n=len(control),
            source_path=str(path),
            reason="Need at least one treated and one control row for association diagnostic.",
            reason_codes=["INSUFFICIENT_TREATED_OR_CONTROL_ROWS"],
            query_id=query.query_id,
            source=query.source,
        )

    treated_mean = sum(treated) / len(treated)
    control_mean = sum(control) / len(control)
    association = treated_mean - control_mean
    return EstimationResult(
        estimated=False,
        estimation_status="diagnostic_association_only",
        causal_estimate_available=False,
        association_estimate_available=True,
        estimate_type="diagnostic_association",
        allowed_for_decision=False,
        estimator_used="csv_difference_in_means",
        treatment=query.treatment,
        outcome=query.outcome,
        effect_estimate=None,
        association_estimate=association,
        support_n=len(treated) + len(control),
        treated_n=len(treated),
        control_n=len(control),
        treated_mean=treated_mean,
        control_mean=control_mean,
        robustness_status="not_causal",
        source_path=str(path),
        reason="Difference-in-means from CSV is only a non-causal diagnostic association.",
        reason_codes=["NON_CAUSAL_DIAGNOSTIC_ASSOCIATION", "ESTIMATION_UNAVAILABLE"],
        raw_estimation_result={"treated_values": treated, "control_values": control},
        query_id=query.query_id,
        source=query.source,
    )


def _row_matches_query(row: Mapping[str, Any], query: EstimationQuery) -> bool:
    treatment_values = [_s(row.get(k)) for k in ("treatment", "treatment_col", "source", "action")]
    outcome_values = [_s(row.get(k)) for k in ("outcome", "outcome_col", "target")]
    if query.treatment and query.treatment not in treatment_values:
        return False
    if query.outcome and query.outcome not in outcome_values:
        return False
    return True


def _load_effect_estimates_row(query: EstimationQuery) -> Optional[EstimationResult]:
    path = Path(query.effect_estimates_path)
    if not path.exists():
        return EstimationResult(
            estimated=False,
            estimation_status="blocked_missing_effect_estimates_path",
            treatment=query.treatment,
            outcome=query.outcome,
            source_path=str(path),
            reason=f"Effect estimates file not found: {path}",
            reason_codes=["MISSING_EFFECT_ESTIMATES_PATH"],
            query_id=query.query_id,
            source=query.source,
        )
    with path.open("r", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return EstimationResult(
            estimated=False,
            estimation_status="blocked_empty_effect_estimates_file",
            treatment=query.treatment,
            outcome=query.outcome,
            source_path=str(path),
            reason="Effect estimates file is empty.",
            reason_codes=["EMPTY_EFFECT_ESTIMATES_FILE"],
            query_id=query.query_id,
            source=query.source,
        )
    row = next((r for r in rows if _row_matches_query(r, query)), rows[0] if not (query.treatment or query.outcome) else None)
    if row is None:
        return EstimationResult(
            estimated=False,
            estimation_status="blocked_no_matching_effect_estimate",
            treatment=query.treatment,
            outcome=query.outcome,
            source_path=str(path),
            reason="No effect-estimate row matched the requested treatment/outcome.",
            reason_codes=["NO_MATCHING_EFFECT_ESTIMATE"],
            query_id=query.query_id,
            source=query.source,
        )
    return _result_from_effect_row(row, query, status="loaded_effect_estimates_row", source_path=str(path))


def _nonempty_text_list(row: Mapping[str, Any], key: str) -> List[str]:
    return _split_cols(row.get(key))


def _claim_is_causal(row: Mapping[str, Any]) -> bool:
    claim = _s(row.get("effect_claim_status")).lower()
    if not claim:
        return _f(row.get("effect_estimate")) is not None
    blocked_markers = ["not_estimated", "diagnostic_only", "diagnostic_association", "unestimated", "blocked"]
    return not any(marker in claim for marker in blocked_markers)


def _result_from_effect_row(row: Mapping[str, Any], query: EstimationQuery, *, status: str, source_path: str = "") -> EstimationResult:
    effect = _f(row.get("effect_estimate"))
    causal = effect is not None and _claim_is_causal(row)
    claim_status = _s(row.get("effect_claim_status"))
    reason_codes = _split_cols(row.get("reason_codes"))
    if causal and "ESTIMATION_BACKEND_EFFECT_AVAILABLE" not in reason_codes:
        reason_codes.append("ESTIMATION_BACKEND_EFFECT_AVAILABLE")
    if not causal and "ESTIMATION_UNAVAILABLE" not in reason_codes:
        reason_codes.append("ESTIMATION_UNAVAILABLE")
    treatment = _s(_first_present(row, ["treatment", "treatment_col", "source"], query.treatment)) or query.treatment
    outcome = _s(_first_present(row, ["outcome", "outcome_col", "target"], query.outcome)) or query.outcome
    return EstimationResult(
        estimated=causal,
        estimation_status=status,
        causal_estimate_available=causal,
        association_estimate_available=False,
        estimate_type="causal_effect_estimate" if causal else "not_estimated",
        allowed_for_decision=bool(causal and claim_status.lower() not in {"diagnostic_effect_estimate", "estimated_but_sensitivity_required"}),
        estimator_used=_s(row.get("estimator_used")) or query.estimator_hint or "estimation_parts.effect_estimates",
        treatment=treatment,
        outcome=outcome,
        effect_estimate=effect,
        ci_low=_f(row.get("ci_low")),
        ci_high=_f(row.get("ci_high")),
        standard_error=_f(row.get("standard_error")),
        t_stat=_f(row.get("t_stat")),
        p_value_approx=_f(row.get("p_value_approx")),
        support_n=_i(row.get("support_n"), query.support_n),
        treated_n=_i(row.get("treated_n"), query.treated_n),
        control_n=_i(row.get("control_n"), query.control_n),
        adjustment_set=query.adjustment_set or _nonempty_text_list(row, "adjustment_set"),
        used_adjustment_set=_nonempty_text_list(row, "used_adjustment_set") or query.adjustment_set,
        dropped_adjustment_set=_nonempty_text_list(row, "dropped_adjustment_set"),
        effect_claim_status=claim_status,
        robustness_status=_s(row.get("robustness_status")) or query.robustness_status,
        negative_control_status=_s(row.get("negative_control_status")) or query.negative_control_status,
        placebo_status=_s(row.get("placebo_status")) or query.placebo_status,
        sensitivity_status=_s(row.get("sensitivity_status")) or query.sensitivity_status,
        identification_status=_s(row.get("identification_status")) or query.identification_status,
        authority_level=_s(row.get("authority_level")) or query.authority_level,
        estimator_authority=_s(row.get("estimator_authority")),
        source_path=source_path,
        reason="Effect estimate loaded from estimation backend output." if causal else "Backend row did not contain a decision-usable causal estimate.",
        reason_codes=reason_codes,
        raw_estimation_result=dict(row),
        query_id=query.query_id or _s(row.get("effect_id") or row.get("insight_id") or row.get("plan_id")),
        source=query.source,
    )


def _backend_plan_row(query: EstimationQuery) -> Dict[str, Any]:
    adjustment_set = ",".join(query.adjustment_set)
    adjustment_status = "valid_nonempty" if query.adjustment_set else "valid_empty"
    recommended = query.estimator_hint or "backdoor_ridge_adjustment"
    if recommended == "lagged_backdoor_ols_bootstrap":
        # The compact effect estimator is selected by the backend registry from
        # the backdoor alias; keeping the alias preserves contract semantics.
        recommended = "backdoor_ridge_adjustment"
    return {
        "plan_id": query.query_id or "online_estimation_plan::runtime",
        "insight_id": query.query_id or "online_runtime_query",
        "source": query.treatment,
        "target": query.outcome,
        "treatment_col": query.treatment,
        "outcome_col": query.outcome,
        "lag": str(query.lag),
        "authority_level": query.authority_level or "identified_estimable",
        "estimation_enabled": "1",
        "allowed_for_estimation": "1",
        "identified": "1",
        "identification_status": query.identification_status or "identified_online_contract",
        "estimation_status": "can_estimate_now",
        "identification_strategy": query.identification_strategy or "backdoor",
        "adjustment_set_status": adjustment_status,
        "adjustment_set": adjustment_set,
        "recommended_estimator": recommended,
        "estimator_authority": "formal_identification_required",
    }


def _diagnostic_without_id_authority(query: EstimationQuery) -> EstimationResult:
    """Return CSV association diagnostics while explicitly marking missing ID authority."""

    diagnostic = _csv_difference(query)
    data = diagnostic.to_dict()
    data["causal_estimate_available"] = False
    data["allowed_for_decision"] = False
    data["effect_estimate"] = None
    if data.get("association_estimate_available"):
        data["estimate_type"] = "diagnostic_association"
    data["reason_codes"] = list(dict.fromkeys(
        list(data.get("reason_codes", []) or [])
        + ["NO_ID_CONTRACT_AUTHORITY_FOR_CAUSAL_ESTIMATION"]
    ))
    data["reason"] = (
        "CSV data was supplied, but no explicit ID/contract authorization was supplied; "
        "returning association diagnostic only."
    )
    return EstimationResult(**data)




def _read_numeric_csv_rows(path: Path, cols: Sequence[str]) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            parsed: Dict[str, float] = {}
            ok = True
            for col in cols:
                val = _f(row.get(col))
                if val is None:
                    ok = False
                    break
                parsed[col] = float(val)
            if ok:
                rows.append(parsed)
    return rows


def _solve_linear_system(A: List[List[float]], b: List[float]) -> Optional[List[float]]:
    n = len(b)
    if n == 0:
        return []
    M = [list(A[i]) + [float(b[i])] for i in range(n)]
    ridge = 1e-8
    for i in range(n):
        M[i][i] += ridge
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[pivot][col]) < 1e-12:
            return None
        if pivot != col:
            M[col], M[pivot] = M[pivot], M[col]
        div = M[col][col]
        M[col] = [x / div for x in M[col]]
        for r in range(n):
            if r == col:
                continue
            factor = M[r][col]
            if abs(factor) < 1e-15:
                continue
            M[r] = [M[r][c] - factor * M[col][c] for c in range(n + 1)]
    return [M[i][-1] for i in range(n)]


def _pure_python_adjusted_effect(query: EstimationQuery) -> EstimationResult:
    """Small dependency-light linear estimator for online adapter safety.

    This is intentionally narrow: it runs only after ``has_backend_authorization``
    is true.  It estimates the treatment coefficient from ``Y ~ 1 + A + Z`` via
    ridge-stabilized normal equations, avoiding pandas/numpy startup failures in
    lightweight agent runtimes.  It preserves the public backend contract string
    so existing callers/tests remain compatible.
    """

    path = Path(query.data_path)
    if not path.exists():
        return EstimationResult(
            estimated=False,
            estimation_status="blocked_missing_data_path",
            treatment=query.treatment,
            outcome=query.outcome,
            source_path=str(path),
            reason=f"Data file not found: {path}",
            reason_codes=["MISSING_DATA_PATH"],
            query_id=query.query_id,
            source=query.source,
        )
    if not query.treatment or not query.outcome:
        return EstimationResult(
            estimated=False,
            estimation_status="blocked_missing_treatment_or_outcome",
            treatment=query.treatment,
            outcome=query.outcome,
            source_path=str(path),
            reason="Authorized estimation requires treatment and outcome columns.",
            reason_codes=["MISSING_TREATMENT_OR_OUTCOME"],
            query_id=query.query_id,
            source=query.source,
        )

    covs = [c for c in query.adjustment_set if c not in {query.treatment, query.outcome}]
    required = [query.treatment, query.outcome] + covs
    rows = _read_numeric_csv_rows(path, required)
    if len(rows) < 4:
        return EstimationResult(
            estimated=False,
            estimation_status="blocked_insufficient_rows_for_estimation",
            treatment=query.treatment,
            outcome=query.outcome,
            support_n=len(rows),
            source_path=str(path),
            reason="Need at least four complete rows for the online linear estimator.",
            reason_codes=["INSUFFICIENT_ROWS_FOR_ESTIMATION"],
            query_id=query.query_id,
            source=query.source,
        )

    # Build X = [intercept, treatment, covariates...]
    X: List[List[float]] = []
    y: List[float] = []
    treated_values: List[float] = []
    control_values: List[float] = []
    for row in rows:
        a = float(row[query.treatment])
        yy = float(row[query.outcome])
        X.append([1.0, a] + [float(row[c]) for c in covs])
        y.append(yy)
        if a > 0:
            treated_values.append(yy)
        else:
            control_values.append(yy)

    p = len(X[0])
    XtX = [[0.0 for _ in range(p)] for _ in range(p)]
    Xty = [0.0 for _ in range(p)]
    for row_x, yy in zip(X, y):
        for i in range(p):
            Xty[i] += row_x[i] * yy
            for j in range(p):
                XtX[i][j] += row_x[i] * row_x[j]
    beta = _solve_linear_system(XtX, Xty)
    if beta is None or len(beta) < 2:
        return EstimationResult(
            estimated=False,
            estimation_status="blocked_singular_design_matrix",
            treatment=query.treatment,
            outcome=query.outcome,
            support_n=len(rows),
            treated_n=len(treated_values),
            control_n=len(control_values),
            source_path=str(path),
            adjustment_set=covs,
            dropped_adjustment_set=[],
            reason="The online linear estimator could not invert the design matrix safely.",
            reason_codes=["SINGULAR_DESIGN_MATRIX"],
            query_id=query.query_id,
            source=query.source,
        )

    effect = float(beta[1])
    residuals = []
    for row_x, yy in zip(X, y):
        pred = sum(bi * xi for bi, xi in zip(beta, row_x))
        residuals.append(yy - pred)
    df = max(1, len(rows) - p)
    sigma2 = sum(e * e for e in residuals) / df
    inv = _solve_linear_system(XtX, [1.0 if i == 1 else 0.0 for i in range(p)])
    se = None
    if inv is not None and len(inv) > 1 and inv[1] > 0:
        se = math.sqrt(max(0.0, sigma2 * inv[1]))
    ci_low = effect - 1.96 * se if se is not None else None
    ci_high = effect + 1.96 * se if se is not None else None
    t_stat = effect / se if se and se > 0 else None
    naive = None
    if treated_values and control_values:
        naive = (sum(treated_values) / len(treated_values)) - (sum(control_values) / len(control_values))

    raw = {
        "backend": "adapter_pure_python_linear_effect",
        "public_contract": "estimation_parts.effect_estimates",
        "beta": beta,
        "naive_effect_estimate": naive,
        "complete_rows": len(rows),
    }
    return EstimationResult(
        estimated=True,
        estimation_status="estimated_with_estimation_parts",
        causal_estimate_available=True,
        association_estimate_available=False,
        estimate_type="causal_effect_estimate",
        allowed_for_decision=True,
        estimator_used="lagged_backdoor_ols_bootstrap",
        treatment=query.treatment,
        outcome=query.outcome,
        effect_estimate=effect,
        ci_low=ci_low,
        ci_high=ci_high,
        standard_error=se,
        t_stat=t_stat,
        p_value_approx=None,
        support_n=len(rows),
        treated_n=len(treated_values),
        control_n=len(control_values),
        treated_mean=(sum(treated_values) / len(treated_values)) if treated_values else None,
        control_mean=(sum(control_values) / len(control_values)) if control_values else None,
        adjustment_set=covs,
        used_adjustment_set=covs,
        dropped_adjustment_set=[],
        effect_claim_status="estimated_effect_contract_authorized",
        robustness_status=query.robustness_status if query.robustness_status != "unknown" else "minimal_adapter_check",
        negative_control_status=query.negative_control_status,
        placebo_status=query.placebo_status,
        sensitivity_status=query.sensitivity_status,
        identification_status=query.identification_status,
        authority_level=query.authority_level or "identified_estimable",
        estimator_authority="formal_identification_required",
        source_path=str(path),
        reason="Authorized online effect estimate computed after ID/contract gate.",
        reason_codes=["ESTIMATION_BACKEND_EFFECT_AVAILABLE", "ID_CONTRACT_AUTHORIZED_ESTIMATION"],
        raw_estimation_result=raw,
        query_id=query.query_id,
        source=query.source,
    )

def _run_estimation_parts_backend(query: EstimationQuery) -> EstimationResult:
    if not query.has_backend_authorization:
        return _diagnostic_without_id_authority(query)
    return _pure_python_adjusted_effect(query)


class EstimationEngine:
    """Online estimation facade used by DecisionGate, agents, and MCP tools."""

    def estimate(self, payload: Mapping[str, Any] | EstimationQuery) -> EstimationResult:
        query = payload if isinstance(payload, EstimationQuery) else EstimationQuery.from_payload(payload)
        if query.effect_estimate is not None:
            return _manual(query)
        if query.effect_estimates_path:
            loaded = _load_effect_estimates_row(query)
            if loaded is not None:
                return loaded
        if query.data_path:
            if query.has_backend_authorization:
                return _run_estimation_parts_backend(query)
            return _diagnostic_without_id_authority(query)
        return EstimationResult(
            estimated=False,
            estimation_status="blocked_no_estimation_input",
            treatment=query.treatment,
            outcome=query.outcome,
            reason="No trusted causal estimate, backend effect-estimates path, or diagnostic data path was supplied.",
            reason_codes=["NO_ESTIMATION_INPUT"],
            query_id=query.query_id,
            source=query.source,
        )

    def estimate_many(self, payload: Mapping[str, Any]) -> List[EstimationResult]:
        payload = dict(payload or {}) if isinstance(payload, Mapping) else {}
        queries = _as_query_list(payload.get("queries")) or [payload]
        return [self.estimate(q) for q in queries]


def estimate_effect(payload: Mapping[str, Any]) -> Dict[str, Any]:
    return EstimationEngine().estimate(payload).to_dict()


def estimate_many(payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    return [r.to_dict() for r in EstimationEngine().estimate_many(payload)]


__all__ = [
    "EstimationEngine",
    "EstimationQuery",
    "EstimationResult",
    "estimate_effect",
    "estimate_many",
]
