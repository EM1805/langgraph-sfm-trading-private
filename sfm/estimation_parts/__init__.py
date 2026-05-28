from runtime_env import configure_scientific_runtime
configure_scientific_runtime()

"""Estimation package for Amantia.

The package initializer is intentionally lightweight: importing
``estimation_parts.pearl_backdoor`` or ``estimation_parts.contract_gate`` should
not also import the full Level 3.2 engine.

Use explicit module imports instead:
    from estimation_parts import engine
    from estimation_parts.effects import estimate_effect_bundle
"""

__all__ = ["contract_gate", "pearl_backdoor", "effects", "engine", "negative_controls", "placebo", "estimator_registry", "stat_core"]
