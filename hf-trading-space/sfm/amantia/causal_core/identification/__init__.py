from . import engine as _engine
from .engine import (
    IdentificationEngine,
    IdentificationQuery,
    IdentificationResult,
    identify_effect,
    identify_many,
    normalize_scm_graph,
)
from .frontdoor_validator import install_frontdoor_mediator_validator


def _patch_direct_confounding_failure_certificate_route() -> None:
    """Preserve Step-68 failure certificates for direct hidden confounding.

    The online adapter has a deliberately conservative graphical fallback for
    LLM-supplied adjustment sets and simple missing-backend situations.  For the
    safety-critical graph pattern X -> Y plus X <-> Y, however, the packaged
    SCM-ID backend can produce a formal hedge/failure certificate.  This package
    initializer keeps that certificate channel authoritative without weakening
    the existing Step-90 online adjustment-set hardening.
    """

    if getattr(IdentificationEngine, "_amantia_step98_failure_route_installed", False):
        return

    original_identify = IdentificationEngine.identify

    def _has_direct_confounded_effect(query: IdentificationQuery) -> bool:
        if len(query.treatments) != 1 or len(query.outcomes) != 1:
            return False
        nodes, directed, bidirected = _engine._normalized_graph_components(query.scm_graph)
        del nodes
        x, y = query.treatment, query.outcome
        return (x, y) in directed and any({a, b} == {x, y} for a, b in bidirected)

    def _identify_with_failure_certificate_route(self, payload):
        query = payload if isinstance(payload, IdentificationQuery) else IdentificationQuery.from_payload(payload)

        # Keep Step-90 behavior: explicit adjustment/mediator claims go through
        # the online graphical validator first and remain fail-closed.
        if query.adjustment_set or query.mediators:
            return original_identify(self, query)

        if not _engine._has_graph(query.scm_graph) or not query.treatments or not query.outcomes:
            return original_identify(self, query)

        if not _has_direct_confounded_effect(query):
            return original_identify(self, query)

        try:
            from scm_parts.admg import admg_from_scm_graph
            from scm_parts.id_full import full_id, identify_conditional_effect

            admg = admg_from_scm_graph(query.scm_graph)
            if query.conditions:
                return _engine._map_conditional_id(
                    query,
                    identify_conditional_effect(
                        admg,
                        query.treatments,
                        query.outcomes,
                        query.conditions,
                        max_depth=query.max_depth,
                    ),
                )
            return _engine._map_full_id(
                query,
                full_id(admg, query.treatments, query.outcomes, max_depth=query.max_depth),
            )
        except Exception as exc:  # pragma: no cover - defensive safety boundary
            if isinstance(exc, (ImportError, ModuleNotFoundError)):
                return _engine._simple_graphical_identification(query)
            return _engine._blocked_result(
                strategy="adapter_runtime_error",
                reason=f"SCM-ID adapter failed safely: {type(exc).__name__}: {exc}",
                reason_codes=["SCM_ID_ADAPTER_ERROR"],
                query=query,
                raw={"error_type": type(exc).__name__, "error_message": str(exc)},
            )

    IdentificationEngine.identify = _identify_with_failure_certificate_route
    IdentificationEngine._amantia_step98_failure_route_installed = True
    IdentificationEngine._amantia_step98_original_identify = original_identify


_patch_direct_confounding_failure_certificate_route()
install_frontdoor_mediator_validator(_engine)

__all__ = [
    "IdentificationEngine",
    "IdentificationQuery",
    "IdentificationResult",
    "identify_effect",
    "identify_many",
    "normalize_scm_graph",
]
