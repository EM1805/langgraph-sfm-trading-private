#!/usr/bin/env python3
"""Canonical graph authority registry for Amantia.

This module does not build a new graph.  It documents and validates which graph
artifacts are allowed to act as canonical inputs for each layer, so the package
has one offline SCM graph, one runtime operational graph, and one explicit
handoff contract.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

GRAPH_AUTHORITY_VERSION = 2

CANONICAL_GRAPHS: Dict[str, Dict[str, object]] = {
    "runtime_veto_graph": {
        "path": "operational_causal_graph.yaml",
        "role": "canonical_runtime_path_graph",
        "authority": "runtime_path_activation_and_veto",
        "used_by": [
            "runtime.causal_graph_runtime",
            "runtime.path_activation",
            "runtime.veto_gateway",
            "runtime.path_counterfactual",
        ],
        "must_exist_in_repo": True,
    },
    "offline_scm_graph": {
        "path": "out/scm/scm_graph.json",
        "role": "canonical_offline_scm_graph",
        "authority": "scm_identification_and_estimation",
        "produced_by": ["scm-build", "scm-pipeline", "causal-align"],
        "used_by": [
            "scm_parts.identifier",
            "scm_parts.fit",
            "estimation_parts.engine",
        ],
        "must_exist_in_repo": False,
    },
    "causal_contract": {
        "path": "out/causal_contract.csv",
        "role": "canonical_offline_handoff_contract",
        "authority": "pcmci_scm_identification_estimation_handoff",
        "produced_by": ["causal_contract.py", "causal-align", "scm-build", "scm-identify"],
        "used_by": ["estimation_parts.engine", "causal_authority_for_veto"],
        "must_exist_in_repo": False,
    },
    "causal_authority_cards": {
        "path": "out/veto/causal_authority_cards.jsonl",
        "role": "canonical_veto_causal_authority_bridge",
        "authority": "offline_causal_evidence_authorizes_runtime_causal_veto_claims",
        "produced_by": ["veto-authority", "causal_authority_for_veto.py"],
        "used_by": ["runtime.veto_gateway", "policy review / audit"],
        "must_exist_in_repo": False,
    },
}

NON_CANONICAL_ARTIFACTS: Dict[str, Dict[str, object]] = {
    "out/edges.csv": {
        "role": "discovery_raw_candidate_edges",
        "canonical_replacement": "out/scm/scm_graph.json and out/causal_contract.csv",
        "policy": "builder_input_only_not_downstream_authority",
    },
    "out/discovery/pcmci_links.csv": {
        "role": "pcmci_raw_candidate_links",
        "canonical_replacement": "out/scm/scm_graph.json and out/causal_contract.csv",
        "policy": "builder_input_only_not_downstream_authority",
    },
    "out/identified_effects.csv": {
        "role": "legacy_identification_mirror",
        "canonical_replacement": "out/identification/identified_effects.csv and out/causal_contract.csv",
        "policy": "legacy_compatibility_only",
    },
}


def _canonical_specs_for_out_dir(out_dir: str | Path = "out") -> Dict[str, Dict[str, object]]:
    """Return canonical graph specs with offline artifacts resolved under out_dir.

    The runtime graph remains repository-relative because it is a runtime policy
    asset. Offline SCM/contract/veto artifacts are run outputs and must follow
    the selected --out-dir to avoid manifests that silently point back to out/.
    """
    out_text = str(out_dir).strip() or "out"
    specs = {name: dict(spec) for name, spec in CANONICAL_GRAPHS.items()}
    specs["offline_scm_graph"]["path"] = str(Path(out_text) / "scm" / "scm_graph.json")
    specs["causal_contract"]["path"] = str(Path(out_text) / "causal_contract.csv")
    specs["causal_authority_cards"]["path"] = str(Path(out_text) / "veto" / "causal_authority_cards.jsonl")
    return specs


def _noncanonical_specs_for_out_dir(out_dir: str | Path = "out") -> Dict[str, Dict[str, object]]:
    out_text = str(out_dir).strip() or "out"
    specs = {path: dict(spec) for path, spec in NON_CANONICAL_ARTIFACTS.items()}
    specs[str(Path(out_text) / "edges.csv")] = specs.pop("out/edges.csv")
    specs[str(Path(out_text) / "discovery" / "pcmci_links.csv")] = specs.pop("out/discovery/pcmci_links.csv")
    specs[str(Path(out_text) / "identified_effects.csv")] = specs.pop("out/identified_effects.csv")
    for spec in specs.values():
        repl = str(spec.get("canonical_replacement", ""))
        repl = repl.replace("out/scm/scm_graph.json", str(Path(out_text) / "scm" / "scm_graph.json"))
        repl = repl.replace("out/causal_contract.csv", str(Path(out_text) / "causal_contract.csv"))
        repl = repl.replace("out/identification/identified_effects.csv", str(Path(out_text) / "identification" / "identified_effects.csv"))
        spec["canonical_replacement"] = repl
    return specs


def build_graph_authority_manifest(root: str | Path = ".", out_dir: str | Path = "out") -> Dict[str, object]:
    root_path = Path(root)
    specs = _canonical_specs_for_out_dir(out_dir)
    noncanonical_specs = _noncanonical_specs_for_out_dir(out_dir)
    canonical: List[Dict[str, object]] = []
    for name, spec in specs.items():
        path = root_path / str(spec["path"])
        canonical.append({
            "name": name,
            **spec,
            "exists": path.exists(),
        })
    noncanonical: List[Dict[str, object]] = []
    for path_text, spec in noncanonical_specs.items():
        path = root_path / path_text
        noncanonical.append({
            "path": path_text,
            **spec,
            "exists": path.exists(),
        })
    return {
        "graph_authority_version": GRAPH_AUTHORITY_VERSION,
        "policy": "one_runtime_graph_one_offline_scm_graph_one_handoff_contract_one_veto_authority_bridge_no_personal_graph_no_offline_prior_input",
        # Backwards-compatible summary fields for lightweight tests/audits.
        "out_dir": str(out_dir),
        "canonical_runtime_graph": specs["runtime_veto_graph"]["path"],
        "canonical_offline_scm_graph": specs["offline_scm_graph"]["path"],
        "canonical_handoff_contract": specs["causal_contract"]["path"],
        "canonical_graphs": canonical,
        "non_canonical_artifacts": noncanonical,
        "rules": [
            "Runtime veto reads operational_causal_graph.yaml for path activation and may read causal_authority_cards.jsonl for precomputed causal authority.",
            "Offline SCM/identification/estimation reads <out-dir>/scm/scm_graph.json and <out-dir>/causal_contract.csv as canonical artifacts.",
            "causal_authority_cards.jsonl is the only bridge allowed to translate offline causal support into runtime causal-veto authority.",
            "Discovery influence reports are optional human-facing exports and must not authorize Pearl-style causal claims.",
            "Raw edges.csv and discovery/pcmci_links.csv may seed SCM builder or low-authority causal_contract rows, but must not be consumed directly by estimation, identification, or veto.",
            "Legacy CSV mirrors are allowed for backwards compatibility but should not overwrite causal_contract.csv in estimation.",
        ],
    }


def write_graph_authority_manifest(root: str | Path = ".", out_path: str | Path = "out/graph_authority_manifest.json", out_dir: Optional[str | Path] = None) -> str:
    root_path = Path(root)
    resolved_out_dir = out_dir if out_dir is not None else "out"
    manifest = build_graph_authority_manifest(root_path, out_dir=resolved_out_dir)
    out = root_path / out_path
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return str(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write Amantia graph authority manifest")
    parser.add_argument("--root", default=".")
    parser.add_argument("--out", default="out/graph_authority_manifest.json")
    parser.add_argument("--out-dir", default=None, help="Output directory whose SCM/contract artifacts are canonical in the manifest")
    args = parser.parse_args(argv)
    path = write_graph_authority_manifest(args.root, args.out, out_dir=args.out_dir)
    print(json.dumps({"status": "ok", "graph_authority_manifest": path}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
