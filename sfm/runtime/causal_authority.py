from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def load_causal_authority_cards(path: str = "out/veto/causal_authority_cards.jsonl") -> Dict[str, Dict[str, Any]]:
    """Load precomputed offline causal authority cards keyed by path_id.

    Missing files are not errors: runtime must remain bounded and conservative.
    """
    p = Path(path)
    if not p.exists():
        return {}
    cards: Dict[str, Dict[str, Any]] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            card = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(card, dict):
            continue
        path_id = str(card.get("path_id") or "").strip()
        if path_id:
            cards[path_id] = card
    return cards


def attach_causal_authority(paths: List[Dict[str, Any]], cards_path: str = "out/veto/causal_authority_cards.jsonl") -> List[Dict[str, Any]]:
    """Attach causal-authority metadata to activated runtime paths.

    This function intentionally does not change risk scores or veto decisions.
    It only labels whether the path is authorized for a causal-veto claim by
    the offline PCMCI/SCM/estimation handoff.
    """
    cards = load_causal_authority_cards(cards_path)
    enriched: List[Dict[str, Any]] = []
    for path in paths or []:
        row = dict(path)
        path_id = str(row.get("path_id") or "").strip()
        card = cards.get(path_id)
        if card:
            row["causal_authority"] = {
                "authority_level": card.get("authority_level", ""),
                "runtime_use": card.get("runtime_use", ""),
                "authority_reason": card.get("authority_reason", ""),
                "contract_insight_id": card.get("contract_insight_id", ""),
                "contract_identification_status": card.get("contract_identification_status", ""),
                "contract_authority_level": card.get("contract_authority_level", ""),
                # Runtime-safe estimation summary. This is copied from the
                # precomputed authority card; the runtime does not read raw
                # effect_estimates.csv or sensitivity_analysis.csv directly.
                "estimation_evidence": card.get("estimation_evidence", {}),
            }
            row["causal_veto_authorized"] = card.get("runtime_use") in {
                "causal_veto_allowed",
                "causal_veto_allowed_with_aggregate_outcome_caution",
            }
        else:
            row["causal_authority"] = {
                "authority_level": "not_available",
                "runtime_use": "policy_or_structural_veto_only",
                "authority_reason": "no_precomputed_causal_authority_card_for_path",
            }
            row["causal_veto_authorized"] = False
        enriched.append(row)
    return enriched
