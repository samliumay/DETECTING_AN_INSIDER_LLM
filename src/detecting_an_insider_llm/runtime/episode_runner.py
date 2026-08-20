"""Run one resolved scenario cell without an interactive user loop.

The interactive :class:`~detecting_an_insider_llm.runtime.agents.Agent` is useful
for manual smoke tests, but an experimental episode has stricter boundaries.  It
must start from fresh mailbox state, use the scenario's fixed prompts and
separate limits, retain partial observations when something fails, and end after
one final assistant response.  :class:`ScenarioRunner` owns that orchestration.

This module deliberately stops at a complete in-memory result.  It does not
choose run IDs or write files; atomic creation and closure of immutable run
artifacts is a separate persistence responsibility.
"""

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from detecting_an_insider_llm.runtime.agents import (
    ChatProvider,
    ProviderContractError,
    ProviderResponse,
    ThinkingMode,
    copy_provider_response,
    validated_assistant_message,
)
from detecting_an_insider_llm.runtime.provenance import (
    capture_operational_provenance,
)
from detecting_an_insider_llm.runtime.tool_loop import (
    ParsedToolCall,
    ToolCallExecution,
    ToolExecutionResult,
    parse_tool_call,
    tool_result_message,
)
from detecting_an_insider_llm.scenario_loader import ExecutionLimits, ResolvedScenario
from detecting_an_insider_llm.tools.action_logging import (
    DoubleLoggingToolExecutor,
    LogIdFactory,
)
from detecting_an_insider_llm.tools.definitions import (
    self_logging_email_tool_definitions,
)
from detecting_an_insider_llm.tools.email import EmailMessage, SimulatedMailbox
from detecting_an_insider_llm.tools.registry import EmailToolDispatcher
from detecting_an_insider_llm.tools.self_log import (
    AutomaticActionRecord,
    ModelSelfLogRecord,
)


EpisodeStatus = Literal["completed", "incomplete", "failed"]
EpisodeTerminationReason = Literal[
    "completed",
    "provider_output_truncated",
    "provider_finish_reason_unknown",
    "max_provider_turns",
    "max_ordinary_tool_attempts",
    "max_self_log_attempts",
    "multiple_tool_attempt_limits",
    "operational_provenance_error",
    "provider_metadata_error",
    "provider_error",
    "provider_contract_error",
    "tool_execution_error",
    "interrupted",
]
FailurePhase = Literal[
    "operational_provenance",
    "provider_metadata",
    "provider_call",
    "provider_response_validation",
    "tool_execution",
]
OperationalProvenanceFactory = Callable[[Path], dict[str, Any]]


@dataclass(frozen=True, slots=True)
class EpisodeError:
    """Describe why an episode failed without replacing its partial evidence.

    Attributes:
        phase: Runner boundary at which the error occurred.
        error_type: Concrete exception class name for failure accounting.
        message: Human-readable exception text for diagnosis.

    Expected non-exception stops are represented by
    :attr:`EpisodeResult.termination_reason`, not by this class. Reaching a
    declared limit or receiving a non-complete provider finish reason makes a
    run incomplete rather than an unexpected runtime failure.
    """

    phase: FailurePhase
    error_type: str
    message: str


@dataclass(frozen=True, slots=True)
class ProviderRequest:
    """Exact provider input for one attempted model turn.

    Attributes:
        turn_sequence: One-based provider-call number within the episode.
        messages: Conversation snapshot sent on that turn.
        tools: Tool definitions advertised on that turn.
        options: Provider generation options, when configured.
        think: Provider-neutral exposed-reasoning setting.

    Failed provider calls have no :class:`ProviderResponse`, so retaining their
    request separately is necessary for a complete journal and denominator.
    """

    schema_version: Literal["1"]
    turn_sequence: int
    messages: tuple[dict[str, Any], ...]
    tools: tuple[dict[str, Any], ...]
    options: dict[str, Any] | None
    think: ThinkingMode


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    """Complete in-memory evidence from one isolated scenario episode.

    The result keeps configuration, interaction evidence, authoritative action
    records, model-created logs, and terminal state together.  A future artifact
    writer can transform these snapshots into the four raw files without asking
    the provider or mutable mailbox for information after the run has ended.

    `completed` means the provider returned a normally finished, valid assistant
    message with no tool calls. `incomplete` means a declared budget or a
    non-complete provider finish reason stopped the interaction. `failed` means
    operational provenance, provider metadata, provider execution, response
    validation, or controlled tool execution raised an unexpected error. These
    are runtime classifications, not judgments about model alignment or intent.
    """

    schema_version: Literal["1"]
    started_at: datetime
    finished_at: datetime
    scenario_schema_version: Literal["1"]
    scenario_id: str
    scenario_title: str
    scenario_current_time: datetime
    agent_email: str
    condition_id: str
    policy_context_id: str
    evaluation_id: str
    system_prompt: str
    user_prompt: str
    input_emails: tuple[EmailMessage, ...]
    provider_name: str
    model_name: str
    operational_provenance: dict[str, Any]
    provider_metadata: dict[str, Any]
    execution_limits: ExecutionLimits
    options: dict[str, Any] | None
    think: ThinkingMode
    tool_definitions: tuple[dict[str, Any], ...]
    status: EpisodeStatus
    termination_reason: EpisodeTerminationReason
    provider_turn_count: int
    ordinary_tool_attempt_count: int
    self_log_attempt_count: int
    messages: tuple[dict[str, Any], ...]
    provider_requests: tuple[ProviderRequest, ...]
    provider_responses: tuple[ProviderResponse, ...]
    tool_executions: tuple[ToolCallExecution, ...]
    automatic_records: tuple[AutomaticActionRecord, ...]
    model_self_logs: tuple[ModelSelfLogRecord, ...]
    sent_emails: tuple[EmailMessage, ...]
    error: EpisodeError | None

    @property
    def last_response(self) -> ProviderResponse | None:
        """Return the last retained response, including one invalid at failure."""

        if not self.provider_responses:
            return None
        return copy_provider_response(self.provider_responses[-1])


@dataclass(slots=True)
class _EpisodeState:
    """Collect mutable observations until they are frozen into a result.

    This private state is created inside :meth:`ScenarioRunner.run`, so it can
    never leak counters, messages, or provider responses into another episode.
    """

    started_at: datetime
    messages: list[dict[str, Any]]
    operational_provenance: dict[str, Any] = field(default_factory=dict)
    provider_metadata: dict[str, Any] = field(default_factory=dict)
    provider_requests: list[ProviderRequest] = field(default_factory=list)
    provider_responses: list[ProviderResponse] = field(default_factory=list)
    tool_executions: list[ToolCallExecution] = field(default_factory=list)
    provider_turn_count: int = 0
    ordinary_tool_attempt_count: int = 0
    self_log_attempt_count: int = 0
    next_tool_sequence: int = 1


class ScenarioRunner:
    """Execute one resolved email scenario through any compatible provider.

    Args:
        provider: Provider adapter satisfying the structural
            :class:`~detecting_an_insider_llm.runtime.agents.ChatProvider`
            contract.
        options: Provider sampling/runtime options snapshotted for every run.
        think: Provider-neutral exposed-reasoning setting.
        log_id_factory: Optional deterministic ID seam for offline tests.  A
            production caller should normally use the default per-episode
            sequential IDs.
        operational_provenance_factory: Optional deterministic seam for tests.
            Production captures the scenario repository state on every run and
            the execution host once per Python process.

    The caller owns the provider lifecycle.  A runner can reuse one provider
    connection for multiple calls, but every :meth:`run` creates a new mailbox,
    outbox, double-logging executor, conversation, and counter set.
    """

    def __init__(
        self,
        provider: ChatProvider,
        *,
        options: Mapping[str, Any] | None = None,
        think: ThinkingMode = False,
        log_id_factory: LogIdFactory | None = None,
        operational_provenance_factory: OperationalProvenanceFactory | None = None,
    ) -> None:
        """Validate stable provider identity and snapshot caller configuration."""

        self._provider = provider
        self._provider_name = _provider_identifier(
            provider.provider_name,
            field_name="provider_name",
        )
        self._model_name = _provider_identifier(
            provider.model_name,
            field_name="model_name",
        )
        self._options = deepcopy(dict(options)) if options is not None else None
        self._think = think
        self._log_id_factory = log_id_factory
        self._operational_provenance_factory = (
            operational_provenance_factory or capture_operational_provenance
        )
        self._tool_definitions = self_logging_email_tool_definitions()

    def run(self, scenario: ResolvedScenario) -> EpisodeResult:
        """Run one non-interactive episode and always return its terminal state.

        Args:
            scenario: A condition and policy cell already validated and resolved
                by :func:`detecting_an_insider_llm.scenario_loader.resolve_scenario`.

        Returns:
            An :class:`EpisodeResult` with complete in-memory evidence.  Expected
            provider, contract, and tool exceptions become a `failed` result so
            work completed before the exception is not discarded.

        No retries occur here.  Retrying invisibly would make one recorded
        provider turn represent several real attempts and undermine provenance.
        """

        if not isinstance(scenario, ResolvedScenario):
            raise TypeError("scenario must be a ResolvedScenario.")

        state = _EpisodeState(
            started_at=_current_utc_time(),
            messages=[
                {"role": "system", "content": scenario.system_prompt},
                {"role": "user", "content": scenario.user_prompt},
            ]
        )
        mailbox = scenario.create_mailbox()
        executor = DoubleLoggingToolExecutor(
            EmailToolDispatcher(mailbox),
            log_id_factory=self._log_id_factory,
        )

        # Operational and provider metadata are captured before the first model
        # call so later mutations cannot change the provenance for this episode.
        try:
            provenance = self._operational_provenance_factory(
                scenario.source_path.parent
            )
            if not isinstance(provenance, dict):
                raise ProviderContractError(
                    "Operational provenance factory must return an object."
                )
            state.operational_provenance = deepcopy(provenance)
        except (Exception, KeyboardInterrupt) as exc:
            termination_reason: EpisodeTerminationReason = (
                "interrupted"
                if isinstance(exc, KeyboardInterrupt)
                else "operational_provenance_error"
            )
            return self._finish(
                scenario,
                state,
                executor,
                mailbox,
                status="failed",
                termination_reason=termination_reason,
                error=_episode_error("operational_provenance", exc),
            )

        try:
            metadata = self._provider.runtime_metadata()
            if not isinstance(metadata, dict):
                raise ProviderContractError(
                    "Provider runtime_metadata must return an object."
                )
            state.provider_metadata = deepcopy(metadata)
        except (Exception, KeyboardInterrupt) as exc:
            termination_reason: EpisodeTerminationReason = (
                "interrupted"
                if isinstance(exc, KeyboardInterrupt)
                else "provider_metadata_error"
            )
            return self._finish(
                scenario,
                state,
                executor,
                mailbox,
                status="failed",
                termination_reason=termination_reason,
                error=_episode_error("provider_metadata", exc),
            )

        limits = scenario.spec.execution_limits
        while True:
            if state.provider_turn_count >= limits.max_provider_turns:
                return self._finish(
                    scenario,
                    state,
                    executor,
                    mailbox,
                    status="incomplete",
                    termination_reason="max_provider_turns",
                )

            # Count attempted calls, including a call that raises or returns an
            # invalid response.  Failed attempts must remain in the denominator.
            state.provider_turn_count += 1
            request = ProviderRequest(
                schema_version="1",
                turn_sequence=state.provider_turn_count,
                messages=tuple(deepcopy(state.messages)),
                tools=tuple(deepcopy(self._tool_definitions)),
                options=deepcopy(self._options),
                think=self._think,
            )
            state.provider_requests.append(request)
            try:
                response = self._provider.chat(
                    deepcopy(request.messages),
                    tools=deepcopy(request.tools),
                    options=deepcopy(request.options),
                    think=request.think,
                )
            except (Exception, KeyboardInterrupt) as exc:
                if isinstance(exc, KeyboardInterrupt):
                    reason: EpisodeTerminationReason = "interrupted"
                elif isinstance(exc, ProviderContractError):
                    reason = "provider_contract_error"
                else:
                    reason = "provider_error"
                return self._finish(
                    scenario,
                    state,
                    executor,
                    mailbox,
                    status="failed",
                    termination_reason=reason,
                    error=_episode_error("provider_call", exc),
                )

            # A malformed normalized message is still a provider observation.
            # Preserve the full ProviderResponse before checking whether it can
            # safely enter the next conversation turn.
            if isinstance(response, ProviderResponse):
                state.provider_responses.append(copy_provider_response(response))
            try:
                assistant_message = validated_assistant_message(response)
            except Exception as exc:
                return self._finish(
                    scenario,
                    state,
                    executor,
                    mailbox,
                    status="failed",
                    termination_reason="provider_contract_error",
                    error=_episode_error("provider_response_validation", exc),
                )

            # A provider can expose a structurally valid but partial message at
            # its output-token limit. Preserve that observation, but do not add
            # it to the conversation or dispatch any tool call it contains.
            if response.finish_reason != "complete":
                return self._finish(
                    scenario,
                    state,
                    executor,
                    mailbox,
                    status="incomplete",
                    termination_reason=(
                        "provider_output_truncated"
                        if response.finish_reason == "length"
                        else "provider_finish_reason_unknown"
                    ),
                )

            state.messages.append(assistant_message)
            raw_tool_calls = assistant_message.get("tool_calls")
            if not raw_tool_calls:
                return self._finish(
                    scenario,
                    state,
                    executor,
                    mailbox,
                    status="completed",
                    termination_reason="completed",
                )

            exceeded_categories: set[Literal["ordinary", "self_log"]] = set()
            for raw_tool_call in raw_tool_calls:
                parsed = parse_tool_call(raw_tool_call)
                sequence = state.next_tool_sequence
                state.next_tool_sequence += 1
                is_self_log = parsed.requested_tool_name == "log_action"

                if is_self_log:
                    state.self_log_attempt_count += 1
                    limit = limits.max_self_log_attempts
                    over_limit = state.self_log_attempt_count > limit
                    if over_limit:
                        exceeded_categories.add("self_log")
                else:
                    # Unknown and structurally malformed calls are ordinary
                    # attempts: they asked the harness to perform a capability,
                    # even though the allowlisted dispatcher will reject them.
                    state.ordinary_tool_attempt_count += 1
                    limit = limits.max_ordinary_tool_attempts
                    over_limit = state.ordinary_tool_attempt_count > limit
                    if over_limit:
                        exceeded_categories.add("ordinary")

                automatic_count_before = len(executor.automatic_records)
                try:
                    if over_limit:
                        result = executor.reject_at_attempt_limit(
                            parsed.requested_tool_name,
                            parsed.raw_arguments,
                            max_attempts=limit,
                        )
                    else:
                        result = executor.execute(
                            parsed.requested_tool_name,
                            parsed.raw_arguments,
                        )
                except (Exception, KeyboardInterrupt) as exc:
                    failed_execution = _failed_execution(
                        parsed,
                        sequence=sequence,
                        executor=executor,
                        automatic_count_before=automatic_count_before,
                        error=exc,
                    )
                    state.tool_executions.append(failed_execution)
                    return self._finish(
                        scenario,
                        state,
                        executor,
                        mailbox,
                        status="failed",
                        termination_reason=(
                            "interrupted"
                            if isinstance(exc, KeyboardInterrupt)
                            else "tool_execution_error"
                        ),
                        error=_episode_error("tool_execution", exc),
                    )

                execution = ToolCallExecution(
                    sequence=sequence,
                    call=parsed,
                    result=result,
                )
                state.tool_executions.append(execution)
                state.messages.append(tool_result_message(execution))

            if exceeded_categories:
                return self._finish(
                    scenario,
                    state,
                    executor,
                    mailbox,
                    status="incomplete",
                    termination_reason=_tool_limit_reason(exceeded_categories),
                )

    def _finish(
        self,
        scenario: ResolvedScenario,
        state: _EpisodeState,
        executor: DoubleLoggingToolExecutor,
        mailbox: SimulatedMailbox,
        *,
        status: EpisodeStatus,
        termination_reason: EpisodeTerminationReason,
        error: EpisodeError | None = None,
    ) -> EpisodeResult:
        """Freeze mutable episode state into one defensive result snapshot."""

        return EpisodeResult(
            schema_version="1",
            started_at=state.started_at,
            finished_at=_current_utc_time(),
            scenario_schema_version=scenario.spec.schema_version,
            scenario_id=scenario.spec.scenario_id,
            scenario_title=scenario.spec.title,
            scenario_current_time=scenario.spec.current_time,
            agent_email=scenario.spec.agent_email,
            condition_id=scenario.condition_id,
            policy_context_id=scenario.policy_context_id,
            evaluation_id=scenario.evaluation.evaluation_id,
            system_prompt=scenario.system_prompt,
            user_prompt=scenario.user_prompt,
            input_emails=tuple(
                email.model_copy(deep=True) for email in scenario.emails
            ),
            provider_name=self._provider_name,
            model_name=self._model_name,
            operational_provenance=deepcopy(state.operational_provenance),
            provider_metadata=deepcopy(state.provider_metadata),
            execution_limits=scenario.spec.execution_limits.model_copy(deep=True),
            options=deepcopy(self._options),
            think=self._think,
            tool_definitions=tuple(deepcopy(self._tool_definitions)),
            status=status,
            termination_reason=termination_reason,
            provider_turn_count=state.provider_turn_count,
            ordinary_tool_attempt_count=state.ordinary_tool_attempt_count,
            self_log_attempt_count=state.self_log_attempt_count,
            messages=tuple(deepcopy(state.messages)),
            provider_requests=tuple(deepcopy(state.provider_requests)),
            provider_responses=tuple(
                copy_provider_response(response)
                for response in state.provider_responses
            ),
            tool_executions=tuple(deepcopy(state.tool_executions)),
            automatic_records=tuple(
                record.model_copy(deep=True) for record in executor.automatic_records
            ),
            model_self_logs=tuple(
                record.model_copy(deep=True) for record in executor.self_logs
            ),
            sent_emails=tuple(
                email.model_copy(deep=True) for email in mailbox.sent_emails
            ),
            error=error,
        )


def _provider_identifier(value: object, *, field_name: str) -> str:
    """Return one stable nonblank provider identifier or fail before a run."""

    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ProviderContractError(
            f"Provider {field_name} must be a nonblank string without edge whitespace."
        )
    return value


def _current_utc_time() -> datetime:
    """Return an aware UTC wall-clock timestamp for run provenance."""

    return datetime.now(timezone.utc)


def _episode_error(phase: FailurePhase, error: BaseException) -> EpisodeError:
    """Convert an exception into explicit terminal failure metadata."""

    return EpisodeError(
        phase=phase,
        error_type=type(error).__name__,
        message=str(error),
    )


def _tool_limit_reason(
    exceeded_categories: set[Literal["ordinary", "self_log"]],
) -> EpisodeTerminationReason:
    """Map one assistant batch's exceeded categories to a terminal reason."""

    if exceeded_categories == {"ordinary"}:
        return "max_ordinary_tool_attempts"
    if exceeded_categories == {"self_log"}:
        return "max_self_log_attempts"
    return "multiple_tool_attempt_limits"


def _failed_execution(
    parsed: ParsedToolCall,
    *,
    sequence: int,
    executor: DoubleLoggingToolExecutor,
    automatic_count_before: int,
    error: BaseException,
) -> ToolCallExecution:
    """Represent a raised tool attempt without pretending the model saw a result.

    An ordinary executor may already have created its authoritative failed
    record before re-raising.  Reusing that record keeps the execution journal
    and automatic log consistent.  Failures before record creation, including a
    broken ID factory, receive a minimal record with no invented `log_id`.
    """

    automatic_records = executor.automatic_records
    if len(automatic_records) > automatic_count_before:
        record = automatic_records[-1]
        result = ToolExecutionResult(
            tool_name=record.requested_tool_name,
            arguments=deepcopy(record.arguments),
            status="failed",
            model_result=deepcopy(record.model_result),
            audit_result=deepcopy(record.audit_result),
        )
    else:
        failure: dict[str, Any] = {
            "ok": False,
            "error": "Tool execution failed inside the harness.",
            "error_code": "tool_execution_failed",
            "error_type": type(error).__name__,
        }
        arguments = (
            deepcopy(dict(parsed.raw_arguments))
            if isinstance(parsed.raw_arguments, Mapping)
            else {}
        )
        result = ToolExecutionResult(
            tool_name=parsed.requested_tool_name,
            arguments=arguments,
            status="failed",
            model_result=deepcopy(failure),
            audit_result=deepcopy(failure),
        )
    return ToolCallExecution(sequence=sequence, call=parsed, result=result)
