"""Amantia agentic runtime facade.

Stable online layer above the existing causal/runtime backends.
The online path is intentionally stdlib-first so LLM/agent decisions do not
require pandas/numpy imports unless scientific backends are invoked.
"""

__version__ = "0.3.0.post106"
