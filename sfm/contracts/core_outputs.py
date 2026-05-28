"""Helpers that make Amantia's user-facing output contract explicit.

The canonical command should always leave a small set of review artifacts on
 disk, even when upstream stages find no candidates.  This module writes header-
 only CSVs where needed and regenerates the compact causal report.
"""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List

from contracts.output_layout import write_output_layout_manifest
from contracts.gate_audit import UNIFIED_GATE_AUDIT_COLUMNS, write_unified_gate_audit


GATE_AUDIT_COLUMNS: List[str] = list(UNIFIED_GATE_AUDIT_COLUMNS)


ESTIMATION_PLAN_COLUMNS: List[str] = [
    "plan_id", "insight_id", "source", "target", "treatment_col", "outcome_col", "lag",
    "estimand_type", "identification_status", "identified", "authority_level", "estimation_enabled",
    "allowed_for_estimation", "estimation_status", "recommended_estimator", "estimator_authority",
    "unobserved_confounding_risk", "sensitivity_level", "sensitivity_status",
    "recommended_sensitivity_method", "minimum_report_before_effect_claim",
    "conditioning_set_used", "adjustment_set", "adjustment_set_status",
    "candidate_adjustment_set", "backdoor_status", "blocked_by", "assumption_notes",
    "adjustment_set_source", "eligible_for_estimation", "forbidden_adjustment_set",
    "negative_controls", "mci_status", "mci_q_value", "mci_n_eff",
    "pc1_parent_support", "scm_role_hint", "reason",
]

SENSITIVITY_ANALYSIS_COLUMNS: List[str] = [
    "sensitivity_id", "insight_id", "source", "target", "treatment_col", "outcome_col", "lag",
    "authority_level", "estimation_enabled", "identification_status", "estimand_type",
    "adjustment_set", "adjustment_set_status", "candidate_adjustment_set", "backdoor_status",
    "negative_controls", "forbidden_adjustment_set", "mci_status", "mci_q_value", "mci_n_eff",
    "pc1_parent_support", "scm_role_hint", "unobserved_confounding_risk", "sensitivity_level",
    "sensitivity_status", "recommended_sensitivity_method", "minimum_report_before_effect_claim",
    "reason",
]

EFFECT_ESTIMATE_COLUMNS: List[str] = [
    "effect_id", "plan_id", "insight_id", "source", "target", "treatment_col", "outcome_col", "lag",
    "estimand_type", "estimator_used", "effect_claim_status", "effect_estimate", "ci_low", "ci_high",
    "ci_level", "standard_error", "t_stat", "p_value_approx", "support_n", "treated_n", "control_n",
    "adjustment_set", "adjustment_set_size", "used_adjustment_set", "dropped_adjustment_set",
    "naive_effect_estimate", "adjusted_vs_naive_delta", "robustness_status", "drop_one_min_effect",
    "drop_one_max_effect", "drop_one_sign_stability", "partial_r2_treatment", "partial_r2_needed_to_explain_away",
    "sensitivity_quant_status", "negative_control_col", "negative_control_status", "negative_control_pass",
    "negative_control_effect_estimate", "negative_control_abs_ratio_to_main",
    "placebo_type", "placebo_status", "placebo_pass", "placebo_effect_estimate", "placebo_abs_ratio_to_main",
    "minimum_report_before_effect_claim", "authority_level", "identification_status",
    "estimation_status", "sensitivity_status", "reason_codes",
]

NEGATIVE_CONTROL_CHECK_COLUMNS: List[str] = [
    "check_id", "effect_id", "insight_id", "source", "target", "treatment_col", "outcome_col", "lag",
    "negative_control_col", "negative_control_status", "negative_control_pass",
    "negative_control_effect_estimate", "negative_control_standard_error", "negative_control_t_stat",
    "negative_control_p_value_approx", "negative_control_abs_ratio_to_main", "support_n", "reason",
]

PLACEBO_CHECK_COLUMNS: List[str] = [
    "placebo_id", "effect_id", "insight_id", "source", "target", "treatment_col", "outcome_col", "lag",
    "placebo_type", "placebo_status", "placebo_pass", "placebo_effect_estimate",
    "placebo_standard_error", "placebo_t_stat", "placebo_p_value_approx",
    "placebo_abs_ratio_to_main", "support_n", "reason",
]

ROBUSTNESS_DIAGNOSTICS_COLUMNS: List[str] = [
    "effect_id", "insight_id", "source", "target", "lag", "robustness_status", "base_effect",
    "naive_effect", "drop_one_min_effect", "drop_one_max_effect", "drop_one_sign_stability",
    "ci_crosses_zero", "used_adjustment_set", "reason",
]

SENSITIVITY_QUANT_COLUMNS: List[str] = [
    "effect_id", "insight_id", "source", "target", "lag", "partial_r2_treatment",
    "partial_r2_needed_to_explain_away", "unobserved_confounder_risk_band", "sensitivity_quant_status",
    "interpretation", "method",
]


def _write_header_csv(path: Path, columns: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns))
        writer.writeheader()


def _ensure_csv(path: Path, columns: Iterable[str], *, overwrite_empty: bool = True) -> str:
    if path.exists() and (path.stat().st_size > 0 or not overwrite_empty):
        return str(path)
    _write_header_csv(path, columns)
    return str(path)

def _mirror_csv(src: Path, dst: Path, columns: Iterable[str]) -> str:
    """Expose a stable root-level review artifact without deleting detailed copies."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.exists() and src.stat().st_size > 0:
        if src.resolve() != dst.resolve():
            shutil.copyfile(src, dst)
    else:
        _write_header_csv(dst, columns)
    return str(dst)


def _write_core_manifest(out: Path, paths: Dict[str, str]) -> str:
    manifest = {
        "contract_version": 2,
        "meaning": "Stable public review outputs for amantia run. Detailed debug artifacts remain in subdirectories.",
        "public_review_outputs": {
            "gate_audit_csv": paths.get("gate_audit_csv", ""),
            "causal_report_csv": paths.get("causal_report_csv", ""),
            "causal_report_md": paths.get("causal_report_md", ""),
            "causal_report_audit_csv": paths.get("causal_report_audit_csv", ""),
            "estimation_plan_csv": paths.get("estimation_plan_csv", ""),
            "effect_estimates_csv": paths.get("effect_estimates_csv", ""),
            "sensitivity_analysis_csv": paths.get("sensitivity_analysis_csv", ""),
            "gate_audit_manifest_json": paths.get("gate_audit_manifest_json", ""),
        },
        "detailed_outputs": {
            "discovery_gate_audit_csv": paths.get("discovery_gate_audit_csv", ""),
            "estimation_plan_detailed_csv": paths.get("estimation_plan_detailed_csv", ""),
            "effect_estimates_detailed_csv": paths.get("effect_estimates_detailed_csv", ""),
            "sensitivity_analysis_detailed_csv": paths.get("sensitivity_analysis_detailed_csv", ""),
            "robustness_diagnostics_csv": paths.get("robustness_diagnostics_csv", ""),
            "sensitivity_quantitative_csv": paths.get("sensitivity_quantitative_csv", ""),
            "negative_control_checks_csv": paths.get("negative_control_checks_csv", ""),
            "placebo_checks_csv": paths.get("placebo_checks_csv", ""),
        },
        "scm_identification_outputs": {
            "id_algorithm_audit_csv": paths.get("id_algorithm_audit_csv", ""),
            "symbolic_evaluation_csv": paths.get("symbolic_evaluation_csv", ""),
            "symbolic_numeric_estimates_csv": paths.get("symbolic_numeric_estimates_csv", ""),
            "symbolic_numeric_diagnostics_csv": paths.get("symbolic_numeric_diagnostics_csv", ""),
            "do_estimates_csv": paths.get("do_estimates_csv", ""),
            "do_diagnostics_csv": paths.get("do_diagnostics_csv", ""),
        },
        "runtime_bridge_outputs": {
            "causal_authority_cards_jsonl": paths.get("causal_authority_cards_jsonl", ""),
            "causal_authority_summary_json": paths.get("causal_authority_summary_json", ""),
        },
        "output_layout": {
            "output_layout_manifest_json": paths.get("output_layout_manifest_json", ""),
            "rule": "Root is for public review outputs; subdirectories/debug are for detailed artifacts.",
        },
    }
    path = out / "core_outputs_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return str(path)


def ensure_core_outputs(out_dir: str = "out", *, regenerate_report: bool = True) -> Dict[str, str]:
    """Ensure the compact user-facing artifacts exist.

    This function does not create causal authority. It only guarantees that the
    expected review files exist with stable schemas, even for empty/null runs.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    discovery = out / "discovery"
    estimation = out / "estimation"
    discovery.mkdir(parents=True, exist_ok=True)
    estimation.mkdir(parents=True, exist_ok=True)

    paths: Dict[str, str] = {}
    # Preserve Discovery-specific audit separately, then write/refresh the unified
    # cross-layer gate audit at the root. The root file is intentionally updated
    # after SCM/ID/do-estimation so it explains final authority, not only discovery.
    paths["discovery_gate_audit_csv"] = _ensure_csv(discovery / "gate_audit.csv", GATE_AUDIT_COLUMNS)
    try:
        paths.update(write_unified_gate_audit(str(out)))
    except (OSError, ValueError, TypeError, KeyError, ImportError, ModuleNotFoundError, RuntimeError) as exc:
        paths["gate_audit_csv"] = _ensure_csv(out / "gate_audit.csv", GATE_AUDIT_COLUMNS)
        manifest = out / "gate_audit_manifest.json"
        manifest.write_text(json.dumps({"status": "gate_audit_generation_failed", "error": str(exc)}, indent=2), encoding="utf-8")
        paths["gate_audit_manifest_json"] = str(manifest)

    estimation_plan_detailed = Path(_ensure_csv(estimation / "estimation_plan.csv", ESTIMATION_PLAN_COLUMNS))
    sensitivity_analysis_detailed = Path(_ensure_csv(estimation / "sensitivity_analysis.csv", SENSITIVITY_ANALYSIS_COLUMNS))
    effect_estimates_detailed = Path(_ensure_csv(estimation / "effect_estimates.csv", EFFECT_ESTIMATE_COLUMNS))

    paths["estimation_plan_detailed_csv"] = str(estimation_plan_detailed)
    paths["sensitivity_analysis_detailed_csv"] = str(sensitivity_analysis_detailed)
    paths["effect_estimates_detailed_csv"] = str(effect_estimates_detailed)
    paths["estimation_plan_csv"] = _mirror_csv(estimation_plan_detailed, out / "estimation_plan.csv", ESTIMATION_PLAN_COLUMNS)
    paths["sensitivity_analysis_csv"] = _mirror_csv(sensitivity_analysis_detailed, out / "sensitivity_analysis.csv", SENSITIVITY_ANALYSIS_COLUMNS)
    paths["effect_estimates_csv"] = _mirror_csv(effect_estimates_detailed, out / "effect_estimates.csv", EFFECT_ESTIMATE_COLUMNS)

    paths["robustness_diagnostics_csv"] = _ensure_csv(estimation / "robustness_diagnostics.csv", ROBUSTNESS_DIAGNOSTICS_COLUMNS)
    paths["sensitivity_quantitative_csv"] = _ensure_csv(estimation / "sensitivity_quantitative.csv", SENSITIVITY_QUANT_COLUMNS)
    paths["negative_control_checks_csv"] = _ensure_csv(estimation / "negative_control_checks.csv", NEGATIVE_CONTROL_CHECK_COLUMNS)
    paths["placebo_checks_csv"] = _ensure_csv(estimation / "placebo_checks.csv", PLACEBO_CHECK_COLUMNS)

    scm = out / "scm"
    scm_outputs = {
        "id_algorithm_audit_csv": scm / "id_algorithm_audit.csv",
        "symbolic_evaluation_csv": scm / "symbolic_evaluation.csv",
        "symbolic_numeric_estimates_csv": scm / "symbolic_numeric_estimates.csv",
        "symbolic_numeric_diagnostics_csv": scm / "symbolic_numeric_diagnostics.csv",
        "do_estimates_csv": scm / "do_estimates.csv",
        "do_diagnostics_csv": scm / "do_diagnostics.csv",
    }
    for key, path in scm_outputs.items():
        if path.exists():
            paths[key] = str(path)

    authority_cards = out / "veto" / "causal_authority_cards.jsonl"
    authority_summary = out / "veto" / "causal_authority_summary.json"
    if authority_cards.exists():
        paths["causal_authority_cards_jsonl"] = str(authority_cards)
    if authority_summary.exists():
        paths["causal_authority_summary_json"] = str(authority_summary)

    if regenerate_report:
        try:
            from contracts.causal_report import write_causal_report
            paths.update(write_causal_report(out_dir=str(out)))
        except (OSError, ValueError, TypeError, KeyError, ImportError, ModuleNotFoundError, RuntimeError) as exc:
            # Report failure should still leave stable schemas behind.
            manifest = out / "causal_report_manifest.json"
            manifest.write_text(json.dumps({"status": "report_generation_failed", "error": str(exc)}, indent=2), encoding="utf-8")
            paths["causal_report_manifest_json"] = str(manifest)
    else:
        paths["causal_report_csv"] = str(out / "causal_report.csv")
        paths["causal_report_md"] = str(out / "causal_report.md")

    paths["output_layout_manifest_json"] = write_output_layout_manifest(out, core_paths=paths)
    paths["core_outputs_manifest_json"] = _write_core_manifest(out, paths)
    return paths


__all__ = ["ensure_core_outputs", "GATE_AUDIT_COLUMNS", "ESTIMATION_PLAN_COLUMNS", "SENSITIVITY_ANALYSIS_COLUMNS"]
