"""Deprecated compatibility wrapper.

The canonical path-level counterfactual implementation now lives in
``path_parts.path_counterfactual_core``. Runtime keeps this module only so older
imports do not break. This layer is diagnostic-only and does not grant SCM/ID
authority.
"""
from __future__ import annotations

from path_parts.path_counterfactual_core import *  # noqa: F401,F403
