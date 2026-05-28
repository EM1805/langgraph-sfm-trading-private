from __future__ import annotations

"""Small CLI smoke demo for langgraph-sfm.

This intentionally uses an injected analyzer.  It proves the package imports and
that the LangGraph-compatible node/monitor contract works without requiring
LangGraph, external model calls, or repository-local data files.
"""

import argparse
import json
from typing import Any, Dict

from .monitor import SFMAgentMonitor
from .node import SFMIntentAnalyzerConfig, SFMIntentAnalyzerNode


def _demo_analyzer(query: Dict[str, Any]) -> Dict[str, Any]:
    goal = "answer_user_question"
    observed_action = str(query.get("observed_action") or "call_search_tool")
    return {
        "observed_action": observed_action,
        "most_likely_goal": goal,
        "intent_score": 0.82,
        "intent_hypothesis_supported": True,
        "intent_claim_authorized": False,
        "authority_status": "falsifiable_diagnostic_only",
        "governance_execution_allowed": False,
        "alignment_summary": {"gate_status": "review"},
        "reason_codes": ["SFM_DEMO_MISSING_VALIDATED_SCM_GRAPH"],
        "limits": ["demo analyzer: replace with real SFM payloads and domain graph"],
    }


def build_demo_state(stated_goal: str = "answer_user_question") -> Dict[str, Any]:
    return {
        "run_id": "langgraph-sfm-demo-run",
        "last_action": "call_search_tool",
        "candidate_goals": [{"goal_variable": "answer_user_question"}],
        "graph": {
            "nodes": ["agent_action", "answer_user_question"],
            "edges": [["agent_action", "answer_user_question"]],
        },
        "stated_goal": stated_goal,
    }


def run_demo(stated_goal: str = "answer_user_question") -> Dict[str, Any]:
    node = SFMIntentAnalyzerNode(
        SFMIntentAnalyzerConfig(include_raw_result=False),
        analyzer=_demo_analyzer,
    )
    monitor = SFMAgentMonitor()
    state = build_demo_state(stated_goal=stated_goal)
    state.update(node(state))
    state.update(monitor(state))
    return {
        "sfm_analysis": state["sfm_analysis"],
        "sfm_monitor": state["sfm_monitor"],
        "requires_human_review": state["requires_human_review"],
        "sfm_gate_status": state["sfm_gate_status"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a langgraph-sfm smoke demo.")
    parser.add_argument(
        "--stated-goal",
        default="answer_user_question",
        help="Goal declared in the agent state. Use a different value to trigger a deception-risk demo.",
    )
    parser.add_argument("--indent", type=int, default=2, help="JSON indentation level.")
    args = parser.parse_args(argv)
    print(json.dumps(run_demo(stated_goal=args.stated_goal), indent=args.indent, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
