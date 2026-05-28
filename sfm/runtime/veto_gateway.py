
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from fs_utils import write_text_safe

from .action_intent import normalize_action_intent
from .action_registry_v2 import get_action_spec, load_action_registry
from .action_semantics import derive_action_effects
from .path_activation import activate_paths
from .path_evidence import empirical_evidence_for_paths
from .path_counterfactual import enrich_paths_with_validation_guided_counterfactual
from .path_library import load_path_library
from .path_validation import enrich_paths_with_validation
from .causal_authority import attach_causal_authority
from .safety_invariants import check_pre_action_invariants, check_post_path_invariants, invariant_decision, pass_result
from .policy_engine import decide_veto
from .runtime_context import extract_runtime_context
from .mitigations import suggest_mitigations
from .veto_explanations import build_veto_explanations

def evaluate_action_request(payload: Dict[str, Any], registry_path: str = "action_registry.yaml", path_library_path: str = "dangerous_paths.yaml", graph_path: str = "operational_causal_graph.yaml", event_log_path: str = "historical_action_events.jsonl", validation_plan_path: str = "out/validation_plan_level2.csv", authority_cards_path: str = "out/veto/causal_authority_cards.jsonl") -> Dict[str, Any]:
    intent = normalize_action_intent(payload)
    registry = load_action_registry(registry_path)
    action_spec = get_action_spec(intent.action_name, registry)
    context_flags = extract_runtime_context(intent.to_dict(), action_spec=action_spec)
    direct_effects = derive_action_effects(intent.to_dict(), action_spec=action_spec)

    pre_invariants = check_pre_action_invariants(intent.to_dict(), context_flags)
    if not pre_invariants.get("ok", False):
        decision = invariant_decision(pre_invariants, activated_path_count=0)
        mitigation_bundle = suggest_mitigations(intent.to_dict(), context_flags, [], decision)
        explanation_bundle = build_veto_explanations(intent.to_dict(), context_flags, [], decision, mitigation_bundle)
        return {
            "action_intent": intent.to_dict(),
            "action_spec_found": bool(action_spec),
            "direct_effects": direct_effects,
            "context_flags": context_flags,
            "safety_invariants": {"pre_action": pre_invariants, "post_path": pass_result(stage="post_path")},
            "activated_paths": [],
            "decision": decision,
            "mitigations": mitigation_bundle,
            "explanations": explanation_bundle,
            "graph_path": graph_path,
            "event_log_path": event_log_path,
            "validation_plan_path": validation_plan_path,
            "authority_cards_path": authority_cards_path,
        }

    paths = activate_paths(direct_effects, context_flags, load_path_library(path_library_path), graph_path=graph_path)
    paths = empirical_evidence_for_paths(intent.to_dict(), paths, event_log_path=event_log_path)
    paths = enrich_paths_with_validation_guided_counterfactual(intent.to_dict(), paths, event_log_path=event_log_path, validation_plan_path=validation_plan_path)
    paths = enrich_paths_with_validation(intent.to_dict(), paths, event_log_path=event_log_path, registry_path=registry_path, path_library_path=path_library_path)
    paths = attach_causal_authority(paths, cards_path=authority_cards_path)
    post_invariants = check_post_path_invariants(paths, context_flags)
    if not post_invariants.get("ok", False):
        decision = invariant_decision(post_invariants, activated_path_count=len(paths))
    else:
        decision = decide_veto(paths, context_flags)
    mitigation_bundle = suggest_mitigations(intent.to_dict(), context_flags, paths, decision)
    explanation_bundle = build_veto_explanations(intent.to_dict(), context_flags, paths, decision, mitigation_bundle)
    return {
        "action_intent": intent.to_dict(),
        "action_spec_found": bool(action_spec),
        "direct_effects": direct_effects,
        "context_flags": context_flags,
        "safety_invariants": {
            "pre_action": pre_invariants,
            "post_path": post_invariants,
        },
        "activated_paths": paths,
        "decision": decision,
        "mitigations": mitigation_bundle,
        "explanations": explanation_bundle,
        "graph_path": graph_path,
        "event_log_path": event_log_path,
        "validation_plan_path": validation_plan_path,
        "authority_cards_path": authority_cards_path,
    }

def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate an agent action through the causal danger veto layer.")
    ap.add_argument("--input", required=True, help="Path to JSON payload for the action.")
    ap.add_argument("--registry-path", default="action_registry.yaml")
    ap.add_argument("--path-library-path", default="dangerous_paths.yaml")
    ap.add_argument("--graph-path", default="operational_causal_graph.yaml")
    ap.add_argument("--event-log-path", default="historical_action_events.jsonl")
    ap.add_argument("--validation-plan-path", default="out/validation_plan_level2.csv")
    ap.add_argument("--authority-cards-path", default="out/veto/causal_authority_cards.jsonl")
    ap.add_argument("--out", default="out/veto_gateway_output.json")
    args = ap.parse_args()
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    result = evaluate_action_request(payload, registry_path=args.registry_path, path_library_path=args.path_library_path, graph_path=args.graph_path, event_log_path=args.event_log_path, validation_plan_path=args.validation_plan_path, authority_cards_path=args.authority_cards_path)
    payload = json.dumps(result, indent=2)
    write_text_safe(args.out, payload)
    print(json.dumps({
        "status": "ok",
        "out": args.out,
        "decision": result.get("decision", {}).get("decision"),
        "activated_path_count": len(result.get("activated_paths", []) or []),
        "top_path_id": (result.get("activated_paths", []) or [{}])[0].get("path_id"),
    }, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
