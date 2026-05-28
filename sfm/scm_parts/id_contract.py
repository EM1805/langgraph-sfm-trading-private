from __future__ import annotations

"""Canonical ID success/failure contract.

This layer is intentionally small and audit-first.  It does not try to perform
identification itself.  It certifies the public contract of the ID engine:

- if an effect is identified, a formula and proof trace must be present;
- if an effect is blocked by a formal hedge, an impossibility certificate must be present;
- if a query is blocked for a weaker reason, the contract says so explicitly and
  does not call it a formal non-identifiability proof.
"""

from dataclasses import asdict, dataclass
import json
from typing import Dict, Mapping, Optional

from .hedge import FormalHedgeDiagnostic


def _safe_json_loads(text: str) -> object:
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        return {"raw": text}


def _safe_json_dumps(payload: object) -> str:
    try:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))
    except TypeError:
        return json.dumps({"repr": repr(payload)}, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class IDContractDiagnostic:
    id_contract_status: str
    id_contract_ok: bool = False
    identification_certificate_json: str = ""
    nonidentification_certificate_json: str = ""
    id_contract_reason_codes: str = ""

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def id_contract_diagnostic(
    *,
    treatment: str,
    outcome: str,
    identifiable: bool,
    id_strategy: str,
    id_algorithm_level: str,
    estimand_formula: str,
    id_proof_status: str,
    id_proof_steps_json: str,
    formula_tree_json: str,
    formal_hedge: Optional[FormalHedgeDiagnostic] = None,
    failure_reason: str = "",
    reason_codes: str = "",
    authority_status: str = "",
    identification_authority: str = "",
    authority_basis: str = "",
) -> IDContractDiagnostic:
    """Certify the result-shape contract for one ID query."""
    proof_trace = _safe_json_loads(id_proof_steps_json)
    formula_tree = _safe_json_loads(formula_tree_json)

    if identifiable:
        missing = []
        if not estimand_formula:
            missing.append("FORMULA_MISSING")
        if id_proof_status != "identified_proof_trace" or not id_proof_steps_json:
            missing.append("PROOF_TRACE_MISSING")
        if not formula_tree_json:
            missing.append("FORMULA_TREE_MISSING")
        if missing:
            return IDContractDiagnostic(
                "identified_contract_incomplete",
                False,
                id_contract_reason_codes="|".join(missing),
            )
        cert = {
            "type": "identification_certificate",
            "identified": True,
            "treatment": treatment,
            "outcome": outcome,
            "strategy": id_strategy,
            "id_algorithm_level": id_algorithm_level,
            "formula": estimand_formula,
            "proof_status": id_proof_status,
            "proof_trace": proof_trace,
            "formula_tree": formula_tree,
            "reason_codes": reason_codes,
            "authority_status": authority_status,
            "identification_authority": identification_authority,
            "authority_basis": authority_basis,
        }
        return IDContractDiagnostic(
            "identified_with_formula_and_proof_trace",
            True,
            identification_certificate_json=_safe_json_dumps(cert),
            id_contract_reason_codes="IDENTIFIED_FORMULA_AND_PROOF_TRACE_PRESENT",
        )

    if formal_hedge is not None and formal_hedge.formal_hedge_certified:
        cert = {
            "type": "nonidentification_certificate",
            "identified": False,
            "certificate_kind": "formal_hedge",
            "treatment": treatment,
            "outcome": outcome,
            "F": formal_hedge.hedge_F,
            "F_prime": formal_hedge.hedge_F_prime,
            "roots_F": formal_hedge.hedge_roots_F,
            "roots_F_prime": formal_hedge.hedge_roots_F_prime,
            "treatment_witness": formal_hedge.hedge_treatment_in_F_minus_F_prime,
            "outcome_witness": formal_hedge.hedge_outcome_witness,
            "checks": _safe_json_loads(formal_hedge.hedge_checks_json),
            "raw_certificate": _safe_json_loads(formal_hedge.hedge_certificate_json),
            "proof_status": id_proof_status,
            "proof_trace": proof_trace,
            "failure_reason": failure_reason or formal_hedge.hedge_reason_codes,
            "reason_codes": reason_codes or formal_hedge.hedge_reason_codes,
            "authority_status": authority_status,
            "identification_authority": identification_authority,
            "authority_basis": authority_basis,
        }
        return IDContractDiagnostic(
            "nonidentified_with_formal_hedge_certificate",
            True,
            nonidentification_certificate_json=_safe_json_dumps(cert),
            id_contract_reason_codes="FORMAL_HEDGE_NONIDENTIFICATION_CERTIFICATE_PRESENT",
        )

    cert = {
        "type": "blocked_without_formal_nonidentification_certificate",
        "identified": False,
        "treatment": treatment,
        "outcome": outcome,
        "strategy": id_strategy,
        "id_algorithm_level": id_algorithm_level,
        "proof_status": id_proof_status,
        "proof_trace": proof_trace,
        "formula_tree": formula_tree,
        "failure_reason": failure_reason,
        "reason_codes": reason_codes,
        "authority_status": authority_status,
        "identification_authority": identification_authority,
        "authority_basis": authority_basis,
        "note": "Blocked conservatively, but no formal hedge certificate was produced.",
    }
    return IDContractDiagnostic(
        "blocked_without_formal_impossibility_certificate",
        True,
        nonidentification_certificate_json=_safe_json_dumps(cert),
        id_contract_reason_codes="BLOCKED_CONSERVATIVE_NO_FORMAL_HEDGE_CERTIFICATE",
    )
