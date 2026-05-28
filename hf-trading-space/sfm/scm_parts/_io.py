from __future__ import annotations

from runtime_env import configure_scientific_runtime
configure_scientific_runtime()


"""I/O helpers for SCM builder.

Keeps disk access and optional DAG loading outside graph construction so
scm_parts.builder stays easier to test.
"""

import os
import warnings
from pathlib import Path
from typing import Optional, Tuple

from runtime_compat import assert_scientific_stack
assert_scientific_stack()

import pandas as pd

# Offline prior graph support was removed from the public input contract.
OfflinePriorGraph = None  # type: ignore


def _append_candidate(candidates: list[str], path: Optional[str | os.PathLike]) -> None:
    if not path:
        return
    text = str(path)
    if text and text not in candidates:
        candidates.append(text)


def _data_candidates(data_path: Optional[str], out_dir: str) -> list[str]:
    """Return robust data-path candidates for CLI and package-root runs."""

    candidates: list[str] = []
    root = Path(__file__).resolve().parents[1]
    out = Path(out_dir)
    out_parent = out.resolve().parent if not out.is_absolute() else out.parent

    def add_with_relatives(value: Optional[str | os.PathLike]) -> None:
        if not value:
            return
        p = Path(value)
        _append_candidate(candidates, p)
        if not p.is_absolute():
            _append_candidate(candidates, root / p)
            _append_candidate(candidates, out_parent / p)

    add_with_relatives(data_path)
    add_with_relatives(out / "data_clean.csv")
    add_with_relatives("data.csv")
    add_with_relatives(out / "demo_data.csv")
    return candidates


def load_data(data_path: Optional[str], out_dir: str) -> pd.DataFrame:
    candidates = _data_candidates(data_path, out_dir)
    attempted = []
    read_errors = []
    for path in candidates:
        attempted.append(path)
        try:
            if os.path.exists(path):
                return pd.read_csv(path)
        except (OSError, ValueError, TypeError, pd.errors.ParserError) as exc:
            msg = f"{path}: {type(exc).__name__}: {exc}"
            read_errors.append(msg)
            warnings.warn(f"[amantia][warning] SCM builder could not read data file {msg}", RuntimeWarning)
            continue
    warnings.warn(
        "[amantia][warning] SCM builder found no readable data file. "
        f"attempted={attempted}; read_errors={read_errors}. "
        "Continuing with metadata-only SCM graph construction.",
        RuntimeWarning,
    )
    return pd.DataFrame()


def load_dag(dag_path: Optional[str], out_dir: str):
    if not dag_path or OfflinePriorGraph is None:
        return None
    path = dag_path
    if not os.path.isabs(path) and not os.path.exists(path):
        path = os.path.join(out_dir, path)
    if not os.path.exists(path):
        return None
    try:
        return OfflinePriorGraph.load(path)
    except (OSError, ValueError, TypeError, RuntimeError, KeyError) as exc:
        warnings.warn(f"[amantia][warning] SCM builder could not load offline prior graph {path}: {type(exc).__name__}: {exc}", RuntimeWarning)
        return None


def read_discovery_frames(out_dir: str, normalize_fn) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    proposals = pd.DataFrame()
    insights = pd.DataFrame()
    bridge = pd.DataFrame()

    pcmci_path = os.path.join(out_dir, "discovery", "pcmci_links.csv")
    if os.path.exists(pcmci_path):
        proposals = pd.read_csv(pcmci_path)
        proposals["__source_artifact"] = pcmci_path
        proposals = normalize_fn(proposals)
    else:
        edges_path = os.path.join(out_dir, "edges.csv")
        if os.path.exists(edges_path):
            proposals = pd.read_csv(edges_path)
            proposals["__source_artifact"] = edges_path
            proposals = normalize_fn(proposals)

    insights_path = os.path.join(out_dir, "ranking", "insights_level2.csv")
    if os.path.exists(insights_path):
        insights = pd.read_csv(insights_path)
        insights["__source_artifact"] = insights_path
        insights = normalize_fn(insights)
    else:
        legacy = os.path.join(out_dir, "insights_level2.csv")
        if os.path.exists(legacy):
            insights = pd.read_csv(legacy)
            insights["__source_artifact"] = legacy
            insights = normalize_fn(insights)

    # Prefer the structural PCMCI→SCM bridge when present. It carries PC1 parent
    # sets, MCI conditioning sets, and SCM role hints. Fall back to the legacy
    # discovery_estimation_bridge for older runs.
    bridge_candidates = [
        os.path.join(out_dir, "pcmci_scm_bridge.csv"),
        os.path.join(out_dir, "discovery", "pcmci_scm_bridge.csv"),
        os.path.join(out_dir, "discovery_estimation_bridge.csv"),
    ]
    for bridge_path in bridge_candidates:
        if os.path.exists(bridge_path):
            bridge = pd.read_csv(bridge_path)
            bridge["__source_artifact"] = bridge_path
            bridge = normalize_fn(bridge)
            break

    return proposals, insights, bridge
