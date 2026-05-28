
from __future__ import annotations
from typing import Any, Dict, List


def _add(items: List[Dict[str, Any]], title: str, action: str, priority: str = "medium", reason: str = "") -> None:
    items.append({
        "title": title,
        "action": action,
        "priority": priority,
        "reason": reason,
    })


def suggest_mitigations(intent: Dict[str, Any], context_flags: Dict[str, Any], paths: List[Dict[str, Any]], decision: Dict[str, Any]) -> Dict[str, Any]:
    suggestions: List[Dict[str, Any]] = []
    alternatives: List[Dict[str, Any]] = []
    params = dict(intent.get("params", {}) or {})
    decision_name = str(decision.get("decision", "PASS"))

    for p in paths:
        pid = str(p.get("path_id", ""))
        if pid == 'external_data_leakage':
            if params.get('attachment_present'):
                _add(suggestions, 'Rimuovi allegati e usa un link interno protetto', 'replace_attachment_with_internal_link', 'high', 'Riduce la superficie di leakage verso l’esterno.')
                alternatives.append({"action_name": intent.get("action_name"), "params_patch": {"attachment_present": False, "share_scope": "internal"}, "why": "Invio senza allegato e con canale interno."})
            if str(params.get('recipient_scope', '')) == 'external':
                _add(suggestions, 'Richiedi approvazione umana prima dell’invio esterno', 'require_human_approval', 'high', 'L’invio esterno con risorsa sensibile richiede un controllo umano.')
                alternatives.append({"action_name": intent.get("action_name"), "params_patch": {"approval_present": True}, "why": "Aggiunge un gate umano prima dell’azione."})
            if str(params.get('resource_sensitivity', '')) in {'high', 'critical'}:
                _add(suggestions, 'Redigi o minimizza i dati prima della condivisione', 'redact_or_minimize_sensitive_content', 'high', 'Riduce l’impatto in caso di errore o forwarding.')
        elif pid == 'destructive_mutation':
            if not bool(params.get('rollback_available', False)):
                _add(suggestions, 'Crea snapshot o backup prima della cancellazione', 'create_backup_or_snapshot', 'high', 'Aumenta la reversibilità dell’azione.')
                alternatives.append({"action_name": intent.get("action_name"), "params_patch": {"rollback_available": True}, "why": "Esegui l’azione solo dopo un punto di ripristino."})
            _add(suggestions, 'Esegui prima in sandbox o su subset limitato', 'run_dry_run_or_subset_delete', 'high', 'Riduce il blast radius e rende visibili gli effetti.')
            if not bool(params.get('approval_present', False)):
                _add(suggestions, 'Richiedi doppia approvazione per delete in produzione', 'require_two_person_approval', 'high', 'Le mutazioni distruttive in produzione vanno controllate.')
        elif pid == 'privilege_escalation':
            _add(suggestions, 'Usa grant temporaneo con scadenza automatica', 'use_time_bounded_permission', 'high', 'Riduce persistenza ed esposizione della capability.')
            _add(suggestions, 'Limita il permesso alla risorsa minima necessaria', 'scope_permission_to_minimum', 'high', 'Applica il principio del least privilege.')
        elif pid == 'policy_bypass':
            _add(suggestions, 'Ripristina il flusso di approvazione standard', 'restore_standard_approval_flow', 'high', 'Il bypass elimina il controllo di governance.')
            _add(suggestions, 'Apri una review esplicita invece di sopprimere il controllo', 'open_review_instead_of_bypass', 'high', 'Sostituisce il bypass con un percorso verificabile.')
        elif pid == 'operational_failure':
            _add(suggestions, 'Esegui rollout graduale o canary', 'use_canary_rollout', 'high', 'Riduce il rischio operativo su tutta la superficie.')
            _add(suggestions, 'Predisponi piano di rollback e monitoraggio attivo', 'prepare_rollback_and_monitoring', 'high', 'Aumenta la capacità di contenere regressioni.')
            if bool(params.get('novel_action', False)):
                _add(suggestions, 'Prova prima in staging con osservabilità completa', 'stage_before_prod', 'medium', 'L’azione nuova ha maggiore incertezza operativa.')

    if decision_name in {'REVIEW', 'HARD_BLOCK'} and not any(s['action'] == 'require_human_approval' for s in suggestions):
        _add(suggestions, 'Richiedi approvazione umana esplicita', 'require_human_approval', 'medium', 'Serve un controllo umano prima dell’esecuzione.')

    seen = set()
    deduped = []
    for item in suggestions:
        key = item['action']
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    alt_seen = set()
    dedup_alt = []
    for item in alternatives:
        key = (item.get('action_name'), tuple(sorted((item.get('params_patch') or {}).items())))
        if key in alt_seen:
            continue
        alt_seen.add(key)
        dedup_alt.append(item)

    return {
        "suggestions": deduped,
        "safer_alternatives": dedup_alt,
    }
