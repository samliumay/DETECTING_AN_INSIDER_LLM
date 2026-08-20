"""Typed loading and resolution for controlled email scenarios.

Scenario YAML selects composable JSON mailbox fragments.  Resolution produces
one immutable prompt/mailbox configuration for a named experimental condition
and policy context.  The loader performs no provider calls and writes no run
artifacts, keeping experiment construction separate from execution.
"""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
)

from detecting_an_insider_llm.tools.email import (
    EmailMessage,
    SimulatedMailbox,
    is_safe_test_address,
)


class ScenarioLoadError(ValueError):
    """Report invalid scenario structure or unsafe referenced files."""


class ExecutionLimits(BaseModel):
    """Bound provider turns and the two future categories of tool attempts.

    Ordinary tools and model-visible self-logging are separate because one
    action may legitimately produce both calls.  Keeping distinct budgets will
    prevent logging overhead from consuming the model's action allowance when
    the durable experiment runner is added.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_provider_turns: int = Field(ge=1)
    max_ordinary_tool_attempts: int = Field(ge=1)
    max_self_log_attempts: int = Field(ge=0)


class ScenarioCondition(BaseModel):
    """Name the mailbox fragments composing one experimental condition."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    description: str = Field(min_length=1)
    mailbox_parts: tuple[str, ...] = Field(min_length=1)

    @field_validator("description")
    @classmethod
    def description_must_not_be_blank(cls, description: str) -> str:
        """Preserve a nonblank condition description for run provenance."""

        if not description.strip():
            raise ValueError("condition description must not be blank")
        return description

    @field_validator("mailbox_parts")
    @classmethod
    def mailbox_parts_must_be_unique_and_relative(
        cls,
        mailbox_parts: tuple[str, ...],
    ) -> tuple[str, ...]:
        """Reject repeated, absolute, or empty mailbox fragment references."""

        if len(mailbox_parts) != len(set(mailbox_parts)):
            raise ValueError("mailbox_parts must not contain duplicates")
        for raw_path in mailbox_parts:
            path = Path(raw_path)
            if not raw_path.strip() or path.is_absolute():
                raise ValueError("mailbox_parts must contain relative file paths")
        return mailbox_parts


class ScenarioSpec(BaseModel):
    """Validate the condition-independent contents of one scenario YAML file."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1"]
    scenario_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    adapted_from: str | None = None
    current_time: datetime
    agent_email: str = Field(min_length=3, max_length=254)
    system_prompt: str = Field(min_length=1)
    user_instruction: str = Field(min_length=1)
    policy_contexts: dict[str, str] = Field(min_length=1)
    conditions: dict[str, ScenarioCondition] = Field(min_length=1)
    execution_limits: ExecutionLimits
    evaluation_file: str = Field(min_length=1)

    @field_validator(
        "scenario_id",
        "title",
        "system_prompt",
        "user_instruction",
        "evaluation_file",
    )
    @classmethod
    def required_text_must_not_be_blank(cls, value: str) -> str:
        """Keep exact scenario text while rejecting whitespace-only values."""

        if not value.strip():
            raise ValueError("required scenario text must not be blank")
        return value

    @field_validator("current_time")
    @classmethod
    def current_time_must_include_timezone(cls, current_time: datetime) -> datetime:
        """Normalize the scenario clock to UTC or reject machine-local time."""

        if current_time.tzinfo is None or current_time.utcoffset() is None:
            raise ValueError("current_time must include a timezone offset")
        return current_time.astimezone(timezone.utc)

    @field_validator("agent_email")
    @classmethod
    def agent_email_must_be_reserved(cls, agent_email: str) -> str:
        """Require the model-controlled sender to use a reserved `.test` address."""

        if not is_safe_test_address(agent_email):
            raise ValueError("agent_email must be a valid address in a .test domain")
        return agent_email

    @field_validator("policy_contexts")
    @classmethod
    def policy_context_names_must_be_nonblank(
        cls,
        policy_contexts: dict[str, str],
    ) -> dict[str, str]:
        """Reject ambiguous blank policy identifiers while allowing empty text."""

        if any(not name.strip() for name in policy_contexts):
            raise ValueError("policy context names must not be blank")
        return policy_contexts


class RequiredProgress(BaseModel):
    """Define deterministic evidence needed before semantic evaluation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    discovery_tool: Literal["list_emails"]
    required_email_ids: tuple[str, ...] = Field(min_length=1)
    required_read_order: Literal["oldest_to_newest"]

    @field_validator("required_email_ids")
    @classmethod
    def required_email_ids_must_be_unique(
        cls,
        required_email_ids: tuple[str, ...],
    ) -> tuple[str, ...]:
        """Reject duplicate required IDs that would distort completion counts."""

        if len(required_email_ids) != len(set(required_email_ids)):
            raise ValueError("required_email_ids must not contain duplicates")
        return required_email_ids


class DeterministicCheck(BaseModel):
    """Describe one future rule-based check over the raw action trace."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    check_id: str = Field(min_length=1)
    description: str = Field(min_length=1)


class SemanticLabel(BaseModel):
    """Describe a judge label without claiming that it was already observed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    description: str = Field(min_length=1)
    evidence_fields: tuple[str, ...] = Field(min_length=1)


class EvaluationSpec(BaseModel):
    """Validate the predeclared evaluation contract for a scenario."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1"]
    evaluation_id: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    required_progress: RequiredProgress
    evidence_groups: dict[str, tuple[str, ...]] = Field(min_length=1)
    deterministic_checks: tuple[DeterministicCheck, ...] = Field(min_length=1)
    semantic_labels: dict[str, SemanticLabel] = Field(min_length=1)
    completion_states: dict[str, str] = Field(min_length=1)


@dataclass(frozen=True, slots=True)
class ResolvedScenario:
    """One fully resolved scenario cell ready for the episode runner.

    The selected prompt, messages, evaluation contract, condition IDs, and
    execution limits travel together so a runner cannot accidentally combine
    evidence from incompatible cells.
    """

    source_path: Path
    spec: ScenarioSpec
    condition_id: str
    policy_context_id: str
    system_prompt: str
    user_prompt: str
    emails: tuple[EmailMessage, ...]
    evaluation: EvaluationSpec

    def create_mailbox(self) -> SimulatedMailbox:
        """Build isolated mailbox state using the scenario's fixed clock.

        Each call returns a new outbox and send counter.  Successful simulated
        sends receive `current_time`, which is the time visible to the model,
        rather than the researcher's machine clock.
        """

        fixed_time = self.spec.current_time
        return SimulatedMailbox(
            self.emails,
            sender_address=self.spec.agent_email,
            clock=lambda: fixed_time,
        )


def load_scenario_spec(path: Path | str) -> ScenarioSpec:
    """Read and validate one scenario YAML file without resolving a condition.

    Args:
        path: Scenario YAML path supplied by a runner or test.

    Returns:
        A frozen typed scenario specification.

    Raises:
        ScenarioLoadError: If reading, YAML decoding, or validation fails.
    """

    scenario_path = Path(path).resolve()
    raw_document = _load_yaml_mapping(scenario_path, document_name="scenario")
    try:
        return ScenarioSpec.model_validate(raw_document)
    except ValidationError as exc:
        raise ScenarioLoadError(
            f"Invalid scenario file {scenario_path}: {_validation_summary(exc)}"
        ) from exc


def load_evaluation_spec(path: Path | str) -> EvaluationSpec:
    """Read and validate a predeclared evaluation YAML file.

    The function only validates the rubric; it does not inspect a run or assign
    labels.  This preserves the distinction between an evaluation plan and an
    observed result.
    """

    evaluation_path = Path(path).resolve()
    raw_document = _load_yaml_mapping(evaluation_path, document_name="evaluation")
    try:
        return EvaluationSpec.model_validate(raw_document)
    except ValidationError as exc:
        raise ScenarioLoadError(
            f"Invalid evaluation file {evaluation_path}: {_validation_summary(exc)}"
        ) from exc


def resolve_scenario(
    path: Path | str,
    *,
    condition_id: str,
    policy_context_id: str,
) -> ResolvedScenario:
    """Resolve one condition/policy cell into prompts and chronological mail.

    Referenced files must remain inside the scenario directory.  Message IDs
    and timestamps must be unique and already ordered, all messages must precede
    the scenario clock, and the evaluation's required IDs must exactly match the
    selected mailbox.  These fail-fast checks protect cross-condition matching.
    """

    source_path = Path(path).resolve()
    spec = load_scenario_spec(source_path)
    condition = spec.conditions.get(condition_id)
    if condition is None:
        raise ScenarioLoadError(f"Unknown condition_id: {condition_id}.")
    if policy_context_id not in spec.policy_contexts:
        raise ScenarioLoadError(f"Unknown policy_context_id: {policy_context_id}.")

    scenario_directory = source_path.parent
    emails: list[EmailMessage] = []
    for relative_part in condition.mailbox_parts:
        part_path = _resolve_child_path(scenario_directory, relative_part)
        emails.extend(_load_mailbox_part(part_path))
    _validate_resolved_emails(emails, current_time=spec.current_time)

    evaluation_path = _resolve_child_path(
        scenario_directory,
        spec.evaluation_file,
    )
    evaluation = load_evaluation_spec(evaluation_path)
    email_ids = tuple(email.email_id for email in emails)
    if evaluation.scenario_id != spec.scenario_id:
        raise ScenarioLoadError("evaluation scenario_id does not match scenario_id")
    if evaluation.required_progress.required_email_ids != email_ids:
        raise ScenarioLoadError(
            "evaluation required_email_ids do not match the resolved mailbox"
        )
    for group_name, group_ids in evaluation.evidence_groups.items():
        if not set(group_ids).issubset(email_ids):
            raise ScenarioLoadError(
                f"evaluation evidence group {group_name} contains unknown email IDs"
            )

    policy_text = spec.policy_contexts[policy_context_id].strip()
    system_prompt = spec.system_prompt.strip()
    if policy_text:
        system_prompt = f"{system_prompt}\n\n{policy_text}"
    user_prompt = (
        f"Current time: {_format_utc_timestamp(spec.current_time)}\n\n"
        f"{spec.user_instruction.strip()}"
    )
    return ResolvedScenario(
        source_path=source_path,
        spec=spec,
        condition_id=condition_id,
        policy_context_id=policy_context_id,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        emails=tuple(emails),
        evaluation=evaluation,
    )


def _load_yaml_mapping(path: Path, *, document_name: str) -> dict[str, object]:
    """Decode one YAML document and require an object at its root."""

    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ScenarioLoadError(f"Could not read {document_name} file {path}: {exc}") from exc
    try:
        raw_document = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise ScenarioLoadError(
            f"Could not parse {document_name} file {path}: {exc}"
        ) from exc
    if not isinstance(raw_document, dict):
        raise ScenarioLoadError(f"{document_name.capitalize()} file must contain an object")
    return raw_document


def _resolve_child_path(directory: Path, relative_path: str) -> Path:
    """Resolve a referenced file while preventing directory traversal."""

    root = directory.resolve()
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ScenarioLoadError(
            f"Referenced file escapes the scenario directory: {relative_path}"
        ) from exc
    return candidate


def _load_mailbox_part(path: Path) -> list[EmailMessage]:
    """Decode and validate one JSON list of synthetic messages."""

    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ScenarioLoadError(f"Could not read mailbox part {path}: {exc}") from exc
    try:
        raw_messages = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ScenarioLoadError(f"Mailbox part {path} is not valid JSON") from exc
    if not isinstance(raw_messages, list):
        raise ScenarioLoadError(f"Mailbox part {path} must contain a JSON list")
    try:
        return [EmailMessage.model_validate(message) for message in raw_messages]
    except ValidationError as exc:
        raise ScenarioLoadError(
            f"Invalid mailbox part {path}: {_validation_summary(exc)}"
        ) from exc


def _validate_resolved_emails(
    emails: list[EmailMessage],
    *,
    current_time: datetime,
) -> None:
    """Enforce unique IDs, strict chronology, and historical timestamps."""

    if not emails:
        raise ScenarioLoadError("resolved mailbox must contain at least one email")
    email_ids = [email.email_id for email in emails]
    if len(email_ids) != len(set(email_ids)):
        raise ScenarioLoadError("resolved mailbox contains duplicate email IDs")
    timestamps = [email.timestamp for email in emails]
    if any(left >= right for left, right in zip(timestamps, timestamps[1:])):
        raise ScenarioLoadError(
            "mailbox parts must produce strictly oldest-to-newest timestamps"
        )
    if timestamps[-1] > current_time:
        raise ScenarioLoadError("resolved mailbox contains an email from the future")


def _validation_summary(error: ValidationError) -> str:
    """Return concise Pydantic locations/messages without echoing raw inputs."""

    summaries: list[str] = []
    for item in error.errors(
        include_url=False,
        include_context=False,
        include_input=False,
    ):
        location = ".".join(str(part) for part in item["loc"])
        summaries.append(f"{location}: {item['msg']}")
    return "; ".join(summaries)


def _format_utc_timestamp(timestamp: datetime) -> str:
    """Render the scenario clock as stable ISO 8601 UTC text for the prompt."""

    return timestamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
