from __future__ import annotations

from runtime_env import configure_scientific_runtime
configure_scientific_runtime()

"""Strong do-estimation orchestrator.

Writes one canonical pair of outputs for all currently supported identified-do
estimators. Legacy diagnostic SCM simulation was removed in SCM Step 35.

Step 12 also adds contract-gated symbolic numeric formulas:
- observed-DAG truncated factorization via ``symbolic_numeric.py``;
- graphical-zero effects.
"""

import os
from typing import Dict, Optional, Tuple

import pandas as pd

from .do_backdoor import DO_DIAGNOSTIC_COLUMNS, DO_ESTIMATE_COLUMNS, _write_csv
from .do_backdoor import estimate_authorized_do_effects as estimate_authorized_backdoor_do_effects
from .do_frontdoor import estimate_authorized_frontdoor_do_effects
from .symbolic_numeric import estimate_symbolic_numeric_effects, write_symbolic_numeric_outputs
from .do_authority_audit import write_do_authority_audit
from .do_authority_validator import write_do_authority_validation


def estimate_authorized_do_effects(
    out_dir: str = "out",
    data_path: Optional[str] = None,
    contract_path: Optional[str] = None,
    bootstrap_draws: int = 80,
    policy: object = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    bd_est, bd_diag = estimate_authorized_backdoor_do_effects(
        out_dir=out_dir,
        data_path=data_path,
        contract_path=contract_path,
        bootstrap_draws=bootstrap_draws,
        policy=policy,
    )
    fd_est, fd_diag = estimate_authorized_frontdoor_do_effects(
        out_dir=out_dir,
        data_path=data_path,
        contract_path=contract_path,
        bootstrap_draws=bootstrap_draws,
        policy=policy,
    )
    sym_est, sym_diag = estimate_symbolic_numeric_effects(
        out_dir=out_dir,
        data_path=data_path,
        contract_path=contract_path,
        bootstrap_draws=bootstrap_draws,
    )
    estimate_frames = [df[DO_ESTIMATE_COLUMNS] for df in [bd_est, fd_est, sym_est] if df is not None and not df.empty]
    diagnostic_frames = [df[DO_DIAGNOSTIC_COLUMNS] for df in [bd_diag, fd_diag, sym_diag] if df is not None and not df.empty]
    estimates = pd.concat(estimate_frames, ignore_index=True) if estimate_frames else pd.DataFrame(columns=DO_ESTIMATE_COLUMNS)
    diagnostics = pd.concat(diagnostic_frames, ignore_index=True) if diagnostic_frames else pd.DataFrame(columns=DO_DIAGNOSTIC_COLUMNS)
    return estimates, diagnostics


def write_do_outputs(
    out_dir: str = "out",
    data_path: Optional[str] = None,
    contract_path: Optional[str] = None,
    bootstrap_draws: int = 80,
    policy: object = None,
) -> Dict[str, str]:
    estimates, diagnostics = estimate_authorized_do_effects(
        out_dir=out_dir,
        data_path=data_path,
        contract_path=contract_path,
        bootstrap_draws=bootstrap_draws,
        policy=policy,
    )
    scm_dir = os.path.join(out_dir, "scm")
    estimates_path = os.path.join(scm_dir, "do_estimates.csv")
    diagnostics_path = os.path.join(scm_dir, "do_diagnostics.csv")
    _write_csv(estimates_path, estimates.to_dict("records"), DO_ESTIMATE_COLUMNS)
    _write_csv(diagnostics_path, diagnostics.to_dict("records"), DO_DIAGNOSTIC_COLUMNS)
    symbolic_paths = write_symbolic_numeric_outputs(
        out_dir=out_dir,
        data_path=data_path,
        contract_path=contract_path,
        bootstrap_draws=bootstrap_draws,
    )
    audit_paths = write_do_authority_audit(
        out_dir=out_dir,
        contract_path=contract_path,
        estimates_path=estimates_path,
        diagnostics_path=diagnostics_path,
    )
    validation_paths = write_do_authority_validation(
        out_dir=out_dir,
        contract_path=contract_path,
        estimates_path=estimates_path,
        diagnostics_path=diagnostics_path,
        authority_audit_path=audit_paths.get("do_authority_audit"),
    )
    return {"do_estimates": estimates_path, "do_diagnostics": diagnostics_path, **symbolic_paths, **audit_paths, **validation_paths}


__all__ = ["estimate_authorized_do_effects", "write_do_outputs"]
