"""Learning Loop utilities for Amantia.

The online path writes append-only JSONL events. Heavy learning, estimation,
RCT analysis, and fine-tuning export remain offline consumers of this log.
"""

from .audit_log import (
    DEFAULT_AUDIT_LOG_PATH,
    AuditEvent,
    AuditLog,
    append_decision,
    append_outcome,
    build_decision_event,
)
from .dataset_builder import (
    DEFAULT_DATASET_OUT_DIR,
    DatasetBuildResult,
    DatasetBuilder,
    build_finetuning_dataset,
    build_learning_dataset,
    build_rct_dataset,
    export_learning_datasets,
)
from .outcome_tracker import (
    OutcomeRecord,
    OutcomeTracker,
    build_outcome_records,
    infer_harm,
    infer_success,
    record_outcome,
    summarize_outcomes,
)

__all__ = [
    "DEFAULT_AUDIT_LOG_PATH",
    "AuditEvent",
    "AuditLog",
    "append_decision",
    "append_outcome",
    "build_decision_event",
    "DEFAULT_DATASET_OUT_DIR",
    "DatasetBuildResult",
    "DatasetBuilder",
    "build_finetuning_dataset",
    "build_learning_dataset",
    "build_rct_dataset",
    "export_learning_datasets",
    "OutcomeRecord",
    "OutcomeTracker",
    "build_outcome_records",
    "infer_harm",
    "infer_success",
    "record_outcome",
    "summarize_outcomes",
]
