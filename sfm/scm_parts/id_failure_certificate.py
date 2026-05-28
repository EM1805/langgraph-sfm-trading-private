from __future__ import annotations

"""Step 68 failure-certificate authority layer for Amantia SCM-ID.

This module does not add new identification authority.  It standardises what
happens when ``full_id`` or IDC cannot identify a query:

* formal hedge certificates are surfaced as machine-readable certificates;
* invalid/cyclic queries get explicit rejection certificates;
* remaining blocked/pending branches are marked as *not certified* technical
  blockers, so they cannot be mistaken for mathematical non-identifiability.

Step 68 also makes the certificate layer an explicit non-formula authority for failures/rejections, so readiness metrics do not count certified failures as delegated formula authority.

The goal is to move toward the Full-ID invariant:
``identified OR formally certified blocked OR explicitly technical pending``.
"""

from dataclasses import asdict, dataclass
import json
from typing import Dict, Iterable, Mapping, Sequence

from .admg import ADMG
from .graph_criteria import directed_cycle_nodes
from .hedge import formal_hedge_diagnostic
from .id_algorithm_common import _dedupe, _json_formula, _s
from .id_ast import HedgeFail, Placeholder
from .id_ast_normalizer import ID_AST_NORMALIZER_VERSION, normalize_formula_ast

ID_FAILURE_CERTIFICATE_VERSION = "id_failure_certificate_v2_step68"
ID_FAILURE_CERTIFICATE_AUTHORITY = "id_failure_certificate_step68"
ID_FAILURE_CERTIFICATE_LEVEL = (
    "standardized_failure_certificate_layer_for_full_id_and_idc_blocks_"
    "formal_hedge_when_certified_else_explicit_technical_pending_no_full_id_claim"
)


def _json(payload: Mapping[str, object]) -> str:
    return json.dumps(dict(payload), sort_keys=True, separators=(",", ":"))


def _join(values: Iterable[object]) -> str:
    return "|".join(_dedupe(values or []))


@dataclass(frozen=True)
class IDFailureCertificate:
    certificate_status: str
    certified: bool = False
    failure_kind: str = ""
    blocker_class: str = ""
    pending_operator: str = ""
    blocker: str = ""
    reason_codes: str = ""
    formal_hedge_certified: int = 0
    formal_hedge_certificate_json: str = ""
    failure_ast_json: str = ""
    failure_trace_json: str = ""
    certificate_json: str = ""
    version: str = ID_FAILURE_CERTIFICATE_VERSION
    level: str = ID_FAILURE_CERTIFICATE_LEVEL

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _certificate_payload(
    *,
    status: str,
    certified: bool,
    failure_kind: str,
    treatments: Sequence[str],
    outcomes: Sequence[str],
    blocker_class: str,
    pending_operator: str,
    blocker: str = "",
    reason_codes: str = "",
    formal_hedge_json: str = "",
    ast_json: str = "",
    source_status: str = "",
    source_blocker_class: str = "",
    source_pending_operator: str = "",
) -> str:
    return _json_formula({
        "certificate_version": ID_FAILURE_CERTIFICATE_VERSION,
        "certificate_level": ID_FAILURE_CERTIFICATE_LEVEL,
        "certificate_status": status,
        "certified": int(bool(certified)),
        "failure_kind": failure_kind,
        "treatments": list(_dedupe(treatments)),
        "outcomes": list(_dedupe(outcomes)),
        "blocker": blocker,
        "blocker_class": blocker_class,
        "pending_operator": pending_operator,
        "formal_hedge_certified": int(bool(formal_hedge_json)),
        "formal_hedge_certificate": json.loads(formal_hedge_json) if formal_hedge_json else {},
        "failure_ast": json.loads(ast_json) if ast_json else {},
        "source_status": source_status,
        "source_blocker_class": source_blocker_class,
        "source_pending_operator": source_pending_operator,
        "reason_codes": reason_codes,
        "full_id_claim_allowed": 0,
    })


def rejection_certificate(
    *,
    treatments: Sequence[object],
    outcomes: Sequence[object],
    failure_kind: str,
    blocker_class: str,
    pending_operator: str,
    blocker: str = "",
    reason_codes: str = "",
    source_status: str = "",
) -> IDFailureCertificate:
    """Return a machine-readable certificate for non-ID rejections.

    Invalid queries and cyclic directed graphs are not hedge failures.  They are
    still important to represent in the same failure-certificate channel so
    downstream agents do not treat missing formulas as silent failures.
    """
    x = list(_dedupe(treatments))
    y = list(_dedupe(outcomes))
    ast = normalize_formula_ast(Placeholder(failure_kind, metadata={
        "blocker": blocker,
        "blocker_class": blocker_class,
        "pending_operator": pending_operator,
        "reason_codes": reason_codes,
        "failure_certificate_step68": 1,
    }))
    ast_json = _json(ast.to_dict())
    status = f"rejected_{failure_kind}_step68"
    cert_json = _certificate_payload(
        status=status,
        certified=False,
        failure_kind=failure_kind,
        treatments=x,
        outcomes=y,
        blocker_class=blocker_class,
        pending_operator=pending_operator,
        blocker=blocker,
        reason_codes=reason_codes,
        ast_json=ast_json,
        source_status=source_status,
    )
    trace_json = _json_formula({
        "trace_version": ID_FAILURE_CERTIFICATE_VERSION,
        "trace": [{"rule": "FAILURE-CERT", "status": status, "failure_kind": failure_kind}],
        "full_id_claim_allowed": 0,
    })
    return IDFailureCertificate(
        certificate_status=status,
        certified=False,
        failure_kind=failure_kind,
        blocker_class=blocker_class,
        pending_operator=pending_operator,
        blocker=blocker,
        reason_codes=reason_codes,
        failure_ast_json=ast_json,
        failure_trace_json=trace_json,
        certificate_json=cert_json,
    )


def failure_certificate_for_query(
    admg: ADMG,
    treatments: Sequence[object],
    outcomes: Sequence[object],
    *,
    source_status: str = "",
    source_blocker: str = "",
    source_blocker_class: str = "",
    source_pending_operator: str = "",
    source_reason_codes: str = "",
) -> IDFailureCertificate:
    """Certify a blocked Full-ID query when possible.

    If a formal hedge is certified, the returned object is a mathematical
    certificate of non-identifiability for the limited checked shape.  Otherwise
    the object is explicit technical pending/not-certified evidence.
    """
    x = list(_dedupe(treatments))
    y = list(_dedupe(outcomes))
    missing = sorted([n for n in x + y if n not in admg.node_set])
    if not x or not y or missing or set(x) & set(y):
        reasons = []
        if not x:
            reasons.append("MISSING_TREATMENT_SET")
        if not y:
            reasons.append("MISSING_OUTCOME_SET")
        if missing:
            reasons.append("QUERY_NODE_NOT_IN_GRAPH:" + "|".join(missing))
        if set(x) & set(y):
            reasons.append("TREATMENT_OUTCOME_OVERLAP:" + "|".join(sorted(set(x) & set(y))))
        reason = ";".join(reasons) or source_reason_codes or "INVALID_FULL_ID_QUERY"
        return rejection_certificate(
            treatments=x,
            outcomes=y,
            failure_kind="invalid_query",
            blocker_class="invalid_query",
            pending_operator="validate_full_id_query",
            blocker=reason,
            reason_codes=reason,
            source_status=source_status,
        )

    cycles = directed_cycle_nodes(admg)
    if cycles:
        return rejection_certificate(
            treatments=x,
            outcomes=y,
            failure_kind="directed_cycle",
            blocker_class="directed_cycle",
            pending_operator="repair_or_reject_cyclic_directed_graph",
            blocker="|".join(cycles),
            reason_codes="DIRECTED_CYCLE_NOT_ADMG_DAG",
            source_status=source_status,
        )

    hedge = formal_hedge_diagnostic(admg, x, y)
    if hedge.formal_hedge_certified:
        try:
            hedge_payload = json.loads(hedge.hedge_certificate_json) if hedge.hedge_certificate_json else {}
        except Exception:
            hedge_payload = {}
        F = hedge_payload.get("F", []) if isinstance(hedge_payload, Mapping) else []
        Fp = hedge_payload.get("F_prime", []) if isinstance(hedge_payload, Mapping) else []
        roots = hedge_payload.get("roots_F", []) if isinstance(hedge_payload, Mapping) else []
        ast = normalize_formula_ast(HedgeFail(F, Fp, roots=roots, label="formal_hedge_certificate_step68"))
        ast_json = _json(ast.to_dict())
        blocker = f"F={hedge.hedge_F};F_prime={hedge.hedge_F_prime}"
        reason = (source_reason_codes + ";" if source_reason_codes else "") + "FORMAL_HEDGE_CERTIFIED_STEP68"
        status = "formal_hedge_certified_step68"
        cert_json = _certificate_payload(
            status=status,
            certified=True,
            failure_kind="formal_hedge",
            treatments=x,
            outcomes=y,
            blocker_class="formal_hedge_certificate",
            pending_operator="fail_id_or_construct_full_hedge_certificate",
            blocker=blocker,
            reason_codes=reason,
            formal_hedge_json=hedge.hedge_certificate_json,
            ast_json=ast_json,
            source_status=source_status,
            source_blocker_class=source_blocker_class,
            source_pending_operator=source_pending_operator,
        )
        trace_json = _json_formula({
            "trace_version": ID_FAILURE_CERTIFICATE_VERSION,
            "trace": [
                {"rule": "ID-5", "status": "formal_hedge_certified", "F": hedge.hedge_F, "F_prime": hedge.hedge_F_prime, "roots": hedge.hedge_roots_F},
                {"rule": "FAILURE-CERT", "status": status, "source_status": source_status},
            ],
            "full_id_claim_allowed": 0,
        })
        return IDFailureCertificate(
            certificate_status=status,
            certified=True,
            failure_kind="formal_hedge",
            blocker_class="formal_hedge_certificate",
            pending_operator="fail_id_or_construct_full_hedge_certificate",
            blocker=blocker,
            reason_codes=reason,
            formal_hedge_certified=1,
            formal_hedge_certificate_json=hedge.hedge_certificate_json,
            failure_ast_json=ast_json,
            failure_trace_json=trace_json,
            certificate_json=cert_json,
        )

    # Not all non-identification is a certified hedge yet.  This branch is the
    # key safety guard: keep the block explicit, but do not pretend it is a
    # mathematical non-identifiability proof.
    blocker_class = source_blocker_class or "technical_pending_not_formal_hedge"
    pending = source_pending_operator or "complete_general_id_or_derive_formal_hedge_certificate"
    blocker = source_blocker or "NO_FORMAL_HEDGE_CERTIFICATE_FOUND"
    reason = (source_reason_codes + ";" if source_reason_codes else "") + hedge.hedge_reason_codes + ";FAILURE_NOT_CERTIFIED_AS_FORMAL_HEDGE_STEP68"
    ast = normalize_formula_ast(Placeholder("not_certified_failure_pending_step68", metadata={
        "source_status": source_status,
        "source_blocker_class": source_blocker_class,
        "source_pending_operator": source_pending_operator,
        "hedge_status": hedge.formal_hedge_status,
        "failure_certificate_step68": 1,
    }))
    ast_json = _json(ast.to_dict())
    status = "not_certified_pending_failure_step68"
    cert_json = _certificate_payload(
        status=status,
        certified=False,
        failure_kind="technical_pending",
        treatments=x,
        outcomes=y,
        blocker_class=blocker_class,
        pending_operator=pending,
        blocker=blocker,
        reason_codes=reason,
        ast_json=ast_json,
        source_status=source_status,
        source_blocker_class=source_blocker_class,
        source_pending_operator=source_pending_operator,
    )
    trace_json = _json_formula({
        "trace_version": ID_FAILURE_CERTIFICATE_VERSION,
        "trace": [
            {"rule": "ID-5", "status": "not_formal_hedge_certified", "hedge_status": hedge.formal_hedge_status},
            {"rule": "FAILURE-CERT", "status": status, "source_status": source_status, "pending_operator": pending},
        ],
        "full_id_claim_allowed": 0,
    })
    return IDFailureCertificate(
        certificate_status=status,
        certified=False,
        failure_kind="technical_pending",
        blocker_class=blocker_class,
        pending_operator=pending,
        blocker=blocker,
        reason_codes=reason,
        failure_ast_json=ast_json,
        failure_trace_json=trace_json,
        certificate_json=cert_json,
    )


__all__ = [
    "ID_FAILURE_CERTIFICATE_VERSION",
    "ID_FAILURE_CERTIFICATE_AUTHORITY",
    "ID_FAILURE_CERTIFICATE_LEVEL",
    "IDFailureCertificate",
    "failure_certificate_for_query",
    "rejection_certificate",
]
