from runtime_env import configure_scientific_runtime
configure_scientific_runtime()

"""Canonical contract/authority bridge modules for Amantia.

This package owns handoff artifacts between Discovery, SCM, Estimation, and Veto:
- causal_contract.csv
- gate_audit.csv / gate_audit_manifest.json
- graph_authority_manifest.json
- causal_authority_cards.jsonl
- causal_confidence_report.csv

Step 13 makes the root gate audit a cross-layer authority explanation:
Discovery -> SCM ID -> symbolic evaluator -> causal_contract -> do-estimation.
"""
