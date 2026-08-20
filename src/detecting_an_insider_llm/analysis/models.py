"""Typed input and output schemas for version-one offline analysis.

Raw artifact models describe the exact records currently written by
``RunArtifactWriter``.  Result models make ``results.json`` a versioned research
artifact rather than an informal report whose meaning can drift between runs.
All important models forbid unknown fields so a producer/schema mismatch fails
before it can silently alter a measurement.
"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from detecting_an_insider_llm.scenario_loader import ExecutionLimits
from detecting_an_insider_llm.tools.self_log import (
    AutomaticActionRecord,
    ModelSelfLogRecord,
)


TerminalStatus = Literal["completed", "incomplete", "failed"]
RelationshipKind = Literal[
    "matched",
    "missing",
    "duplicate",
    "unknown_id",
    "malformed",
    "not_evaluable_runtime_failure",
]
InteractionAnomalyKind = Literal[
    "malformed_ordinary_call",
    "malformed_self_log_call",
    "unknown_ordinary_tool",
    "failed_ordinary_call",
    "rejected_self_log_call",
]
CheckStatus = Literal["passed", "failed"]
AccuracyStatus = Literal["accurate", "inaccurate", "requires_review"]
EvaluabilityStatus = Literal[
    "evaluable",
    "not_evaluable_runtime_failure",
    "not_evaluable_incomplete",
    "not_evaluable_insufficient_progress",
]
DiscrepancySignal = Literal["detected", "not_detected", "not_evaluable"]
OverallAssessment = Literal[
    "discrepancy_detected",
    "requires_semantic_review",
    "not_evaluable",
]


class PersistedAutomaticActionRecord(AutomaticActionRecord):
    """An authoritative action record after run-level linkage is attached."""

    run_id: str = Field(min_length=1, max_length=128)
    tool_execution_sequence: int = Field(ge=1)


class PersistedModelSelfLogRecord(ModelSelfLogRecord):
    """A successfully stored self-log after run-level linkage is attached."""

    run_id: str = Field(min_length=1, max_length=128)
    tool_execution_sequence: int = Field(ge=1)


class ArtifactManifestEntry(BaseModel):
    """Declare one raw filename's encoding and expected record count."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    format: Literal["json", "jsonl"]
    record_count: int = Field(ge=0)


class MetadataTimestamps(BaseModel):
    """Retain aware lifecycle timestamps from the closed raw run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    started_at: datetime
    finished_at: datetime
    persisted_at: datetime

    @field_validator("started_at", "finished_at", "persisted_at")
    @classmethod
    def timestamps_must_be_aware(cls, value: datetime) -> datetime:
        """Reject timestamps whose meaning depends on the analyzer machine."""

        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("metadata timestamps must include a timezone")
        return value


class MetadataScenario(BaseModel):
    """Scenario identity and exact prompt snapshot retained in metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1"]
    scenario_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    scenario_current_time: datetime
    agent_email: str = Field(min_length=1)
    condition_id: str = Field(min_length=1)
    policy_context_id: str = Field(min_length=1)
    evaluation_id: str = Field(min_length=1)
    system_prompt: str = Field(min_length=1)
    user_prompt: str = Field(min_length=1)
    input_email_ids: tuple[str, ...] = Field(min_length=1)

    @field_validator("scenario_current_time")
    @classmethod
    def scenario_time_must_be_aware(cls, value: datetime) -> datetime:
        """Reject an ambiguous scenario clock before progress is interpreted."""

        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("scenario_current_time must include a timezone")
        return value


class MetadataProvider(BaseModel):
    """Provider identity and provider-exposed runtime provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider_name: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    runtime_metadata: dict[str, Any]


class MetadataGeneration(BaseModel):
    """Generation settings retained exactly as supplied to the provider."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    options: dict[str, Any] | None
    think: bool | Literal["low", "medium", "high"]


class MetadataTerminal(BaseModel):
    """Terminal episode state used to decide whether analysis is evaluable."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: TerminalStatus
    termination_reason: str = Field(min_length=1)
    error: dict[str, Any] | None


class MetadataCounts(BaseModel):
    """Persisted denominators used to detect dropped or fabricated records."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider_turn_attempts: int = Field(ge=0)
    provider_responses: int = Field(ge=0)
    ordinary_tool_attempts: int = Field(ge=0)
    self_log_attempts: int = Field(ge=0)
    tool_executions: int = Field(ge=0)
    sent_emails: int = Field(ge=0)
    automatic_records: int = Field(ge=0)
    model_self_log_records: int = Field(ge=0)
    journal_records: int = Field(ge=0)


class RunMetadata(BaseModel):
    """The complete version-one ``metadata.json`` boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1"]
    artifact_state: Literal["closed"]
    run_id: str = Field(min_length=1, max_length=128)
    episode_schema_version: Literal["1"]
    timestamps: MetadataTimestamps
    scenario: MetadataScenario
    provider: MetadataProvider
    generation: MetadataGeneration
    execution_limits: ExecutionLimits
    tool_definitions: tuple[dict[str, Any], ...]
    terminal: MetadataTerminal
    counts: MetadataCounts
    artifacts: dict[str, ArtifactManifestEntry]


class JournalRecord(BaseModel):
    """One ordered journal envelope; event payloads are checked separately."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1"]
    run_id: str = Field(min_length=1, max_length=128)
    journal_sequence: int = Field(ge=1)
    event_type: Literal[
        "episode_started",
        "provider_request",
        "provider_response",
        "tool_execution",
        "provider_failure",
        "provider_metadata_failure",
        "episode_finished",
    ]
    payload: dict[str, Any]


class JournalToolCall(BaseModel):
    """Parsed provider call retained beneath a journal tool event."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    requested_tool_name: str
    raw_arguments: Any
    tool_call_id: str | None
    raw_tool_call: Any


class JournalToolResult(BaseModel):
    """Normalized result views retained beneath a journal tool event."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tool_name: str
    arguments: dict[str, Any]
    status: Literal["succeeded", "rejected", "failed"]
    model_result: dict[str, Any]
    audit_result: dict[str, Any]


class JournalToolExecution(BaseModel):
    """One journaled call/result pair with episode-global execution order."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sequence: int = Field(ge=1)
    call: JournalToolCall
    result: JournalToolResult


class JournalToolEventPayload(BaseModel):
    """Typed payload for events used by deterministic action accounting."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider_turn: int = Field(ge=1)
    execution: JournalToolExecution
    model_message: dict[str, Any] | None


class AnalyzerProvenance(BaseModel):
    """Identify which implementation generated a derived result."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    analyzer_id: str
    analyzer_version: str
    analyzed_at: datetime


class RunIdentity(BaseModel):
    """Retain the experimental cell and terminal state beside every result."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    scenario_id: str
    evaluation_id: str
    condition_id: str
    policy_context_id: str
    provider_name: str
    model_name: str
    terminal_status: TerminalStatus
    termination_reason: str


class InputValidationSummary(BaseModel):
    """State which immutable inputs passed compatibility checks."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["valid_closed_raw_run"]
    raw_schema_version: Literal["1"]
    evaluation_schema_version: Literal["1"]
    validated_raw_files: tuple[str, ...]


class ProgressAssessment(BaseModel):
    """Machine-checkable evidence that the intended mailbox was encountered."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["met", "not_met"]
    discovery_tool: str
    discovery_succeeded_with_all_required_ids: bool
    required_email_ids: tuple[str, ...]
    successfully_read_email_ids: tuple[str, ...]
    missing_email_ids: tuple[str, ...]
    observed_first_successful_read_order: tuple[str, ...]
    required_read_order: str
    read_order_satisfied: bool


class DeterministicCheckResult(BaseModel):
    """Outcome and trace references for one frozen rule-based check."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    check_id: str
    description: str
    status: CheckStatus
    detail: str
    evidence_record_ids: tuple[str, ...] = ()


class AccuracyAssessment(BaseModel):
    """Separate exact comparisons from description content needing review."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tool_name: Literal["accurate", "inaccurate"]
    claimed_outcome: Literal["accurate", "inaccurate"]
    action_description: Literal["requires_review"]
    overall: AccuracyStatus
    reasons: tuple[str, ...]


class RelationshipFinding(BaseModel):
    """One identity-based automatic/self-log relationship or call anomaly."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    finding_id: str
    relationship: RelationshipKind
    log_id: str | None
    automatic_record_id: str | None
    self_log_record_ids: tuple[str, ...]
    tool_execution_sequences: tuple[int, ...]
    accuracy: AccuracyAssessment | None
    detail: str


class InteractionAnomaly(BaseModel):
    """One malformed, unknown, failed, or limit-rejected call observation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    anomaly_id: str
    anomaly_type: InteractionAnomalyKind
    tool_name: str
    automatic_record_id: str | None
    log_id: str | None
    tool_execution_sequence: int = Field(ge=1)
    error_code: str | None
    detail: str


class SemanticLabelAssessment(BaseModel):
    """Preserve a declared semantic label without inventing an observation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    label_id: str
    status: Literal["requires_review"]
    description: str
    evidence_fields: tuple[str, ...]
    reason: str


class AnalysisDenominators(BaseModel):
    """Expose counts needed to interpret a run without hidden exclusions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ordinary_tool_attempts: int = Field(ge=0)
    automatic_action_records: int = Field(ge=0)
    self_log_attempts: int = Field(ge=0)
    stored_self_log_records: int = Field(ge=0)
    malformed_self_log_attempts: int = Field(ge=0)
    automatic_outcomes: dict[str, int]
    relationship_counts: dict[str, int]
    interaction_anomaly_counts: dict[str, int]


class RunAssessment(BaseModel):
    """Calibrated run-level result that does not infer model intent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evaluability: EvaluabilityStatus
    discrepancy_signal: DiscrepancySignal
    overall: OverallAssessment
    semantic_policy_assessment: Literal["requires_review"]
    reasons: tuple[str, ...]


class ResultsDocument(BaseModel):
    """Version-one schema serialized to one run's ``results.json``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1"] = "1"
    analyzer: AnalyzerProvenance
    run: RunIdentity
    input_validation: InputValidationSummary
    progress: ProgressAssessment
    deterministic_checks: tuple[DeterministicCheckResult, ...]
    relationships: tuple[RelationshipFinding, ...]
    interaction_anomalies: tuple[InteractionAnomaly, ...]
    semantic_labels: tuple[SemanticLabelAssessment, ...]
    denominators: AnalysisDenominators
    run_assessment: RunAssessment
