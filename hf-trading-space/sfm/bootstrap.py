from __future__ import annotations

"""Compatibility bootstrap for the relocated SFM core.

The original research core used absolute imports such as ``amantia.*``,
``runtime.*`` and ``scm_parts.*``.  In this public package all of that core
lives under the single top-level package ``sfm/``.  This module installs
safe runtime aliases so the core can run without exposing many top-level
packages in the published repository.
"""

import importlib
import sys
from types import ModuleType

_ALIASES = (
    "runtime_env",
    "runtime_compat",
    "config",
    "fs_utils",
    "common",
    "contracts",
    "runtime",
    "scm_parts",
    "estimation_parts",
    "path_parts",
    "amantia",
)

def _alias(name: str) -> ModuleType:
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    module = importlib.import_module(f"sfm.{name}")
    sys.modules[name] = module
    return module

def install_legacy_aliases() -> None:
    """Install aliases required by the bundled full SFM core."""
    for name in _ALIASES:
        _alias(name)

install_legacy_aliases()
