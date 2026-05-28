"""Amantia runtime package.

This package intentionally stays scientific-stack free: importing `runtime` must
not import pandas or numpy, and should not perform process-wide scientific
runtime configuration. Heavy dependencies belong to discovery/SCM/estimation
commands, not to veto/runtime safety checks.
"""

from __future__ import annotations

__all__: list[str] = []
