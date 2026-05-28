
from __future__ import annotations
from typing import Any, Dict, List


def _tier_text(path: Dict[str, Any]) -> str:
    return (
        f"strutturale={path.get('evidence_profile', {}).get('structural', 'unknown')}, "
        f"empirica={path.get('empirical_evidence_strength', 'none')}, "
        f"controfattuale={path.get('counterfactual_evidence_strength', 'none')}, "
        f"identificazione={path.get('identification_support', 'none')}"
    )


def build_veto_explanations(intent: Dict[str, Any], context_flags: Dict[str, Any], paths: List[Dict[str, Any]], decision: Dict[str, Any], mitigations: Dict[str, Any]) -> Dict[str, Any]:
    d = str(decision.get('decision', 'PASS'))
    if not paths:
        reason_codes = ','.join(decision.get('reason_codes', []) or [])
        notes = [str(n).strip() for n in decision.get('notes', []) or [] if str(n).strip()]
        if d == 'HARD_BLOCK':
            human = notes[0] if notes else 'Blocco: un invariant di sicurezza ha impedito l’esecuzione prima dell’analisi dei path.'
            operator = f'No dangerous path activated, but pre/post invariant blocked execution. Decision=HARD_BLOCK; reason_codes={reason_codes}'
        elif d == 'REVIEW':
            human = notes[0] if notes else 'Review richiesta: non ci sono path attivati, ma il gate richiede controllo umano o chiarimento.'
            operator = f'No dangerous path activated. Decision=REVIEW; reason_codes={reason_codes}'
        elif d == 'PASS_WITH_WARNING':
            human = notes[0] if notes else 'Pass con warning: nessun path pericoloso attivato, ma il gate conserva una cautela operativa.'
            operator = f'No dangerous path activated. Decision=PASS_WITH_WARNING; reason_codes={reason_codes}'
        else:
            human = 'Nessun path di rischio pericoloso è stato attivato; l’azione può procedere.'
            operator = f'No dangerous path activated. Decision={d}; reason_codes={reason_codes}'
        return {
            'human_summary': human,
            'operator_summary': operator,
            'path_explanations': [],
        }

    top = paths[0]
    path_lines: List[Dict[str, Any]] = []
    for p in paths[:3]:
        line = {
            'path_id': p.get('path_id'),
            'why_triggered': (
                f"Attivato da {', '.join(p.get('activated_by', []) or ['trigger sconosciuto'])}; "
                f"amplificatori: {', '.join(p.get('amplifiers', []) or ['nessuno'])}."
            ),
            'evidence_summary': _tier_text(p),
            'graph_summary': (
                f"harm={p.get('graph_harm')}, graph_supported={p.get('graph_supported')}, "
                f"graph_confidence={p.get('graph_path_confidence')}"
            ),
            'counterfactual_summary': (
                f"delta={p.get('counterfactual_risk_delta', 0)}, "
                f"treated={p.get('counterfactual_treated_support', 0)}, "
                f"control={p.get('counterfactual_control_support', 0)}"
            ),
            'caveats': [
                c for c in [
                    'negative control problematico' if str(p.get('negative_control_status')) == 'fail' else '',
                    'identificazione debole' if str(p.get('identification_support')) in {'low', 'none'} else '',
                    'matching dei controlli debole' if float(p.get('counterfactual_control_match_quality', 0.0) or 0.0) < 0.55 else '',
                ] if c
            ],
        }
        path_lines.append(line)

    action_name = str(intent.get('action_name', 'azione'))
    target = str(intent.get('target_resource', 'risorsa'))
    if d == 'HARD_BLOCK':
        human = (
            f"Blocco: l’azione {action_name} su {target} attiva il path {top.get('path_id')} con rischio alto "
            f"e supporto sufficiente per impedire l’esecuzione diretta."
        )
    elif d == 'REVIEW':
        human = (
            f"Review richiesta: l’azione {action_name} su {target} attiva il path {top.get('path_id')} con rischio non trascurabile; "
            f"serve conferma umana o mitigazione prima di procedere."
        )
    elif d == 'PASS_WITH_WARNING':
        human = (
            f"Pass con warning: l’azione {action_name} su {target} può procedere solo con attenzione, perché attiva il path {top.get('path_id')}."
        )
    else:
        human = f"Pass: l’azione {action_name} su {target} non attiva un rischio sufficiente a bloccare o deviare il flusso."

    if mitigations.get('suggestions'):
        first = mitigations['suggestions'][0]
        human += f" Mitigazione principale suggerita: {first.get('title')}."

    operator = (
        f"Decision={d}; top_path={top.get('path_id')}; risk={top.get('risk_score')}; "
        f"reason_codes={','.join(decision.get('reason_codes', []))}; {_tier_text(top)}"
    )
    return {
        'human_summary': human,
        'operator_summary': operator,
        'path_explanations': path_lines,
    }
