"""Runtime guards for scientific imports.

Amantia imports numpy/pandas in many modules. On constrained containers or
CI runners, BLAS/OpenMP libraries can try to spawn too many worker threads
at import time and appear to hang. This module must be imported before
numpy/pandas whenever possible.
"""
from __future__ import annotations

import os

_CONFIGURED = False


def configure_scientific_runtime() -> None:
    """Set conservative process-wide defaults before importing numpy/pandas.

    These defaults are intentionally safe and can still be overridden by the
    user by setting the environment variable before launching Python.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    defaults = {
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "BLIS_NUM_THREADS": "1",
        "GOTO_NUM_THREADS": "1",
        "OMP_DYNAMIC": "FALSE",
        "MKL_DYNAMIC": "FALSE",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "MPLBACKEND": "Agg",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)

    _CONFIGURED = True


configure_scientific_runtime()
