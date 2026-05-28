from __future__ import annotations

LOW = "low"
MEDIUM = "medium"
HIGH = "high"
CRITICAL = "critical"
UNKNOWN = "unknown"

RISK_ORDER = {UNKNOWN: 0, LOW: 1, MEDIUM: 2, HIGH: 3, CRITICAL: 4}
VALID_RISK_LEVELS = set(RISK_ORDER)


def normalize_risk_level(value: object, default: str = UNKNOWN) -> str:
    text = str(value or "").strip().lower()
    return text if text in VALID_RISK_LEVELS else default


def max_risk(*levels: str) -> str:
    best = UNKNOWN
    for level in levels:
        norm = normalize_risk_level(level)
        if RISK_ORDER[norm] > RISK_ORDER[best]:
            best = norm
    return best
