"""Path-level counterfactual diagnostics for Amantia.

This package is intentionally outside ``scm_parts``.  It can score and enrich
candidate paths, but it does not grant SCM causal authority.  SCM authority must
flow through ``scm_parts.id_algorithm`` and ``scm_parts.scm_counterfactual``.
"""

from .path_counterfactual import *  # noqa: F401,F403
