from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, List

PUBLIC_REVIEW_FILENAMES: List[str] = [
    "causal_report.md",
    "causal_report.csv",
    "gate_audit.csv",
    "gate_audit_manifest.json",
    "estimation_plan.csv",
    "effect_estimates.csv",
    "sensitivity_analysis.csv",
    "core_outputs_manifest.json",
    "output_layout_manifest.json",
]

COMPAT_DEBUG_ROOT_FILENAMES: List[str] = [
    "edges.csv",
    "insights_level2.csv",
    "discovery_estimation_bridge.csv",
    "pcmci_scm_bridge.csv",
    "identified_effects.csv",
    "validation_plan_level2.csv",
    "path_candidates_level2.csv",
    "dangerous_paths_discovery_draft.yaml",
    "graph_authority_manifest.json",
    "causal_report_manifest.json",
]

DEBUG_SUBDIRS: List[str] = [
    "debug",
    "debug/legacy_root_outputs",
    "discovery",
    "scm",
    "identification",
    "estimation",
    "veto",
]


def _copy_if_present(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.exists() and src.is_file() and src.resolve() != dst.resolve():
        shutil.copyfile(src, dst)
    return str(dst)


def sync_debug_aliases(out_dir: str | Path = "out") -> Dict[str, str]:
    """Mirror legacy root artifacts into `out/debug/legacy_root_outputs/`.

    This does not delete root compatibility artifacts yet. Some optional
    commands and legacy users still expect them. The manifest makes the
    distinction explicit: root public files are the stable review contract;
    mirrored debug files are developer/compatibility aids.
    """
    out = Path(out_dir)
    debug_root = out / "debug" / "legacy_root_outputs"
    debug_root.mkdir(parents=True, exist_ok=True)
    mirrored: Dict[str, str] = {}
    for name in COMPAT_DEBUG_ROOT_FILENAMES:
        src = out / name
        if src.exists() and src.is_file():
            mirrored[name] = _copy_if_present(src, debug_root / name)

    readme = out / "debug" / "README.md"
    readme.write_text(
        "# Amantia debug outputs\n\n"
        "The root of `out/` is reserved for stable public review outputs. "
        "Detailed layer artifacts remain in subdirectories such as `discovery/`, "
        "`scm/`, `identification/`, `estimation/`, and `veto/`. Root-level "
        "compatibility artifacts, when produced, are mirrored under "
        "`debug/legacy_root_outputs/` for inspection without making them part of "
        "the public output contract.\n",
        encoding="utf-8",
    )
    return mirrored


def write_output_layout_manifest(out_dir: str | Path = "out", *, core_paths: Dict[str, str] | None = None) -> str:
    out = Path(out_dir)
    for subdir in DEBUG_SUBDIRS:
        (out / subdir).mkdir(parents=True, exist_ok=True)

    mirrored = sync_debug_aliases(out)
    core_paths = dict(core_paths or {})
    manifest = {
        "contract_version": 1,
        "meaning": "Output layout classifier: root public review files vs detailed/debug artifacts.",
        "public_root_contract": {name: str(out / name) for name in PUBLIC_REVIEW_FILENAMES},
        "public_root_outputs_present": {name: (out / name).exists() for name in PUBLIC_REVIEW_FILENAMES},
        "scm_identification_contract": {
            "id_algorithm_audit_csv": str(out / "scm" / "id_algorithm_audit.csv"),
            "symbolic_evaluation_csv": str(out / "scm" / "symbolic_evaluation.csv"),
            "symbolic_numeric_estimates_csv": str(out / "scm" / "symbolic_numeric_estimates.csv"),
            "symbolic_numeric_diagnostics_csv": str(out / "scm" / "symbolic_numeric_diagnostics.csv"),
            "do_estimates_csv": str(out / "scm" / "do_estimates.csv"),
            "do_diagnostics_csv": str(out / "scm" / "do_diagnostics.csv"),
        },
        "scm_identification_outputs_present": {
            "id_algorithm_audit_csv": (out / "scm" / "id_algorithm_audit.csv").exists(),
            "symbolic_evaluation_csv": (out / "scm" / "symbolic_evaluation.csv").exists(),
            "symbolic_numeric_estimates_csv": (out / "scm" / "symbolic_numeric_estimates.csv").exists(),
            "symbolic_numeric_diagnostics_csv": (out / "scm" / "symbolic_numeric_diagnostics.csv").exists(),
            "do_estimates_csv": (out / "scm" / "do_estimates.csv").exists(),
            "do_diagnostics_csv": (out / "scm" / "do_diagnostics.csv").exists(),
        },
        "debug_subdirectories": {
            "discovery": str(out / "discovery"),
            "scm": str(out / "scm"),
            "identification": str(out / "identification"),
            "estimation": str(out / "estimation"),
            "veto": str(out / "veto"),
            "legacy_root_mirror": str(out / "debug" / "legacy_root_outputs"),
        },
        "compatibility_root_outputs_mirrored": mirrored,
        "notes": [
            "Root-level public files are stable user-facing artifacts.",
            "Subdirectories contain detailed layer/debug artifacts.",
            "Legacy root outputs are mirrored, not deleted, to avoid breaking older commands.",
            "SCM ID/do-estimation outputs remain in out/scm but are summarized in root causal_report and gate_audit.",
        ],
    }
    if core_paths:
        manifest["core_outputs"] = core_paths

    path = out / "output_layout_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return str(path)


__all__ = [
    "PUBLIC_REVIEW_FILENAMES",
    "COMPAT_DEBUG_ROOT_FILENAMES",
    "DEBUG_SUBDIRS",
    "sync_debug_aliases",
    "write_output_layout_manifest",
]
