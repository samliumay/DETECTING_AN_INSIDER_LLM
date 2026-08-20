"""Provider-neutral agent primitives.

Architecture
------------
The runtime owns the provider contract because it defines what an agent needs,
not how a particular API works. Provider adapters, such as `OllamaClient`,
implement that contract without being imported here. This dependency direction
allows the same `Agent` to use Ollama, Gemini, or a test double.

The basic :meth:`Agent.run` method still performs one provider turn.  The
separate :meth:`Agent.run_with_tools` path coordinates a bounded multi-turn loop
through an injected allowlisted executor.  Parsing and tool-result construction
live in :mod:`detecting_an_insider_llm.runtime.tool_loop`; concrete email dispatch
lives in the tools package.  Durable artifact persistence remains a later layer.
"""

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from detecting_an_insider_llm.runtime.tool_loop import (
    ToolCallExecution,
    ToolExecutor,
    execute_tool_call,
    reject_tool_call_at_round_limit,
    tool_result_message,
)

# Provider messages and tool definitions contain nested, provider-defined JSON.
# Keeping the boundary JSON-shaped avoids leaking an Ollama SDK class into the
# runtime while later schemas are still being designed.
Message = dict[str, Any]

# Ollama supports booleans and named effort levels. Other adapters can translate
# these shared values to their provider-specific representation.
ThinkingMode = bool | Literal["high", "medium", "low"]

# Provider adapters translate their native generation-stop values into this
# small shared vocabulary. ``complete`` means the provider finished one turn
# normally, whether that turn contains text or tool calls. ``unknown`` keeps a
# missing or unrecognized provider value visible instead of guessing success.
ProviderFinishReason = Literal["complete", "length", "unknown"]
_PROVIDER_FINISH_REASONS = frozenset({"complete", "length", "unknown"})


@dataclass(frozen=True, slots=True)
class ProviderResponse:
    """A provider response normalized for the runtime.

    Attributes:
        message:
            The assistant message to commit to the conversation. It may contain
            text, tool calls, exposed reasoning, or provider-supported fields.
        raw_response:
            The complete decoded provider response. Keeping it separate from the
            normalized message preserves provider metadata for the journal and
            later provenance analysis.
        finish_reason:
            Provider-neutral reason the generation turn ended. Experimental
            runtimes must not treat ``length`` or ``unknown`` as completion.

    The dataclass is frozen so its attributes cannot be reassigned accidentally.
    Nested JSON values remain mutable, so the runtime takes defensive copies
    before storing them.
    """

    message: Message
    raw_response: dict[str, Any]
    finish_reason: ProviderFinishReason


@dataclass(frozen=True, slots=True)
class ToolLoopResult:
    """Complete in-memory outcome of one bounded user/tool interaction.

    Attributes:
        last_response: Last valid provider response, whether it was a normal
            final answer or an additional tool request stopped by the limit.
        provider_responses: Every valid provider response produced during this
            user turn, in call order.  Raw responses remain available for later
            journaling.
        tool_executions: Every successful or rejected tool-call attempt in
            monotonic action order.
        tool_rounds: Number of rounds in which calls were actually dispatched.
        termination_reason: `completed` for a normal assistant answer or
            `max_tool_rounds` when further requested calls were rejected.

    This result is not a durable research artifact.  The non-interactive runner
    uses a stricter episode result, and a later artifact writer must preserve
    that result in the run journal.
    """

    last_response: ProviderResponse
    provider_responses: tuple[ProviderResponse, ...]
    tool_executions: tuple[ToolCallExecution, ...]
    tool_rounds: int
    termination_reason: Literal["completed", "max_tool_rounds"]


class ToolLoopProviderError(RuntimeError):
    """Report provider failure after at least one tool attempt was committed.

    Attributes:
        provider_responses: Valid responses observed before the failed call.
        tool_executions: Tool attempts already executed or rejected.

    The original provider exception is preserved as `__cause__`.  Carrying the
    partial structured trace prevents a caller from losing observed actions just
    because the provider failed before producing the final assistant message.
    """

    def __init__(
        self,
        provider_error: Exception,
        *,
        provider_responses: Sequence[ProviderResponse],
        tool_executions: Sequence[ToolCallExecution],
    ) -> None:
        """Copy partial progress and retain the provider's readable message."""

        super().__init__(str(provider_error))
        self.provider_responses = tuple(
            copy_provider_response(response) for response in provider_responses
        )
        self.tool_executions = tuple(deepcopy(tool_executions))


class ChatProvider(Protocol):
    """Structural interface required by :class:`Agent`.

    A provider does not need to inherit from this protocol. It is compatible
    when it exposes these properties and methods with matching behavior. This is
    what makes provider injection dynamic while keeping the expected contract
    visible to type checkers and reviewers.
    """

    @property
    def provider_name(self) -> str:
        """Return a stable provider identifier such as `"ollama"`."""
        ...

    @property
    def model_name(self) -> str:
        """Return the exact configured model identifier."""
        ...

    def runtime_metadata(self) -> dict[str, Any]:
        """Return provider/model provenance that can be recorded with a run."""
        ...

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
        options: Mapping[str, Any] | None = None,
        *,
        think: ThinkingMode = False,
    ) -> ProviderResponse:
        """Generate one normalized assistant response."""
        ...


class ProviderContractError(RuntimeError):
    """Raised when a provider returns data that the runtime cannot consume."""


class Agent:
    """Maintain conversation state while delegating generation to a provider.

    Args:
        provider:
            Any object satisfying :class:`ChatProvider`. The agent never checks
            for a concrete provider class.
        system_prompt:
            Optional instruction inserted once at the start of the conversation.
        tools:
            Optional provider-facing tool schemas. They are advertised to the
            model. :meth:`run` does not execute calls; :meth:`run_with_tools`
            executes them only through its explicitly supplied executor.
        options:
            Sampling/runtime options forwarded without provider-specific
            interpretation.
        think:
            Whether, or at what supported level, the provider should request
            exposed reasoning.

    :meth:`run` performs exactly one provider call and preserves returned tool
    calls without executing them. :meth:`run_with_tools` adds bounded execution
    while keeping the allowlist outside this class.  Neither path writes durable
    logs; the scenario runner handles isolated experiment orchestration, while
    the future artifact writer remains responsible for persistence.
    """

    def __init__(
        self,
        provider: ChatProvider,
        *,
        system_prompt: str | None = None,
        tools: Sequence[Mapping[str, Any]] | None = None,
        options: Mapping[str, Any] | None = None,
        think: ThinkingMode = False,
    ) -> None:
        if system_prompt is not None and not system_prompt.strip():
            raise ValueError("system_prompt must contain non-whitespace text.")

        self._provider = provider
        self._system_prompt = system_prompt

        # Snapshot caller-owned configuration. Otherwise a caller could mutate a
        # tool schema or sampling option midway through a run and make the actual
        # request differ from the recorded configuration.
        self._tools = deepcopy(list(tools)) if tools is not None else None
        self._options = deepcopy(dict(options)) if options is not None else None
        self._think = think
        self._messages = self._initial_messages()
        self._next_tool_sequence = 1

    @property
    def provider_name(self) -> str:
        """Return the injected provider's stable name."""
        return self._provider.provider_name

    @property
    def model_name(self) -> str:
        """Return the model identifier selected by the provider."""
        return self._provider.model_name

    @property
    def messages(self) -> tuple[Message, ...]:
        """Return a defensive copy of the committed conversation history.

        A tuple prevents callers from appending to the returned collection, and
        the deep copy prevents mutation of nested message/tool-call data.
        """
        return tuple(deepcopy(self._messages))

    def runtime_metadata(self) -> dict[str, Any]:
        """Return provider metadata without exposing its mutable cache."""
        return deepcopy(self._provider.runtime_metadata())

    def run(self, user_message: str) -> ProviderResponse:
        """Generate one assistant turn and commit it to the conversation.

        The method uses a prepare/call/commit sequence:

        1. Build a request from the last committed history and new user message.
        2. Ask the injected provider for one response.
        3. Validate the shared response contract.
        4. Commit both new messages only after validation succeeds.

        Therefore a timeout, provider exception, or malformed response cannot
        leave a half-finished user turn in the agent's in-memory history.
        """
        if not user_message.strip():
            raise ValueError("user_message must contain non-whitespace text.")

        # Work on copies until the provider response passes validation. This is
        # the temporary request state, not yet the committed conversation.
        request_messages = [
            *deepcopy(self._messages),
            {"role": "user", "content": user_message},
        ]
        response = self._provider.chat(
            request_messages,
            tools=deepcopy(self._tools),
            options=deepcopy(self._options),
            think=self._think,
        )
        assistant_message = validated_assistant_message(response)

        # This is the only state commit in the method.
        self._messages = [*request_messages, assistant_message]
        return response

    def run_with_tools(
        self,
        user_message: str,
        *,
        executor: ToolExecutor,
        max_tool_rounds: int,
    ) -> ToolLoopResult:
        """Run one user turn through a bounded provider/tool cycle.

        Args:
            user_message: Nonblank user input appended after committed history.
            executor: Explicit allowlisted dispatcher for parsed function calls.
            max_tool_rounds: Positive number of tool-call batches that may
                execute before later requests are rejected.

        Returns:
            A :class:`ToolLoopResult` containing provider responses, structured
            tool attempts, the number of executed rounds, and termination reason.

        Raises:
            ValueError: If the user message is blank or the round limit is not a
                positive integer.
            ToolLoopProviderError: If a provider fails after one or more tool
                attempts have already been committed.  Partial evidence is
                attached to the exception.
            Exception: The original provider exception is allowed to propagate
                when the first call fails before any tool action exists.

        One tool round may contain several calls.  All calls in that assistant
        message are handled in list order, then their model receipts are appended
        as `role="tool"` messages before the provider is called again.  Progress
        is committed after every tool attempt because an observed action must not
        disappear if a later provider call fails.
        """

        if not user_message.strip():
            raise ValueError("user_message must contain non-whitespace text.")
        if (
            isinstance(max_tool_rounds, bool)
            or not isinstance(max_tool_rounds, int)
            or max_tool_rounds < 1
        ):
            raise ValueError("max_tool_rounds must be a positive integer.")

        working_messages = [
            *deepcopy(self._messages),
            {"role": "user", "content": user_message},
        ]
        provider_responses: list[ProviderResponse] = []
        tool_executions: list[ToolCallExecution] = []
        tool_rounds = 0

        while True:
            try:
                response = self._provider.chat(
                    working_messages,
                    tools=deepcopy(self._tools),
                    options=deepcopy(self._options),
                    think=self._think,
                )
                assistant_message = validated_assistant_message(response)
            except Exception as exc:
                # Before any action, the normal one-turn transactional behavior
                # remains intact and callers receive their provider's exception.
                if not tool_executions:
                    raise
                raise ToolLoopProviderError(
                    exc,
                    provider_responses=provider_responses,
                    tool_executions=tool_executions,
                ) from exc

            copied_response = copy_provider_response(response)
            provider_responses.append(copied_response)
            working_messages.append(assistant_message)
            raw_tool_calls = assistant_message.get("tool_calls")

            if not raw_tool_calls:
                # A normal assistant answer completes the user turn.  This final
                # commit also covers turns that never requested a tool.
                self._messages = deepcopy(working_messages)
                return ToolLoopResult(
                    last_response=copied_response,
                    provider_responses=tuple(provider_responses),
                    tool_executions=tuple(tool_executions),
                    tool_rounds=tool_rounds,
                    termination_reason="completed",
                )

            if tool_rounds >= max_tool_rounds:
                # These are still observable requests.  Record each as rejected
                # rather than silently discarding the final assistant tool calls.
                for raw_tool_call in raw_tool_calls:
                    execution = reject_tool_call_at_round_limit(
                        raw_tool_call,
                        sequence=self._next_tool_sequence,
                        max_tool_rounds=max_tool_rounds,
                    )
                    self._next_tool_sequence += 1
                    tool_executions.append(execution)
                    working_messages.append(tool_result_message(execution))
                    self._messages = deepcopy(working_messages)
                return ToolLoopResult(
                    last_response=copied_response,
                    provider_responses=tuple(provider_responses),
                    tool_executions=tuple(tool_executions),
                    tool_rounds=tool_rounds,
                    termination_reason="max_tool_rounds",
                )

            tool_rounds += 1
            for raw_tool_call in raw_tool_calls:
                execution = execute_tool_call(
                    raw_tool_call,
                    sequence=self._next_tool_sequence,
                    executor=executor,
                )
                self._next_tool_sequence += 1
                tool_executions.append(execution)
                working_messages.append(tool_result_message(execution))

                # Commit after each attempt.  If another call in the same batch
                # unexpectedly fails, completed earlier actions remain visible.
                self._messages = deepcopy(working_messages)

    def reset(self) -> None:
        """Clear turns and action numbering while retaining configuration."""
        self._messages = self._initial_messages()
        self._next_tool_sequence = 1

    def _initial_messages(self) -> list[Message]:
        """Build fresh initial state so reset never reuses mutable messages."""
        if self._system_prompt is None:
            return []
        return [{"role": "system", "content": self._system_prompt}]


def validated_assistant_message(response: ProviderResponse) -> Message:
    """Validate the cross-provider fields required for conversation history.

    Provider-specific fields are preserved. Only the invariants needed by the
    runtime are checked: the normalized object type, assistant role, text type,
    and tool-call container type.
    """
    if not isinstance(response, ProviderResponse):
        raise ProviderContractError("Provider must return a ProviderResponse.")
    if response.finish_reason not in _PROVIDER_FINISH_REASONS:
        raise ProviderContractError(
            "Provider response finish_reason must be 'complete', 'length', or "
            "'unknown'."
        )

    message = response.message
    if not isinstance(message, dict):
        raise ProviderContractError("Provider response message must be an object.")
    if message.get("role") != "assistant":
        raise ProviderContractError("Provider response message must have role 'assistant'.")

    content = message.get("content")
    if content is not None and not isinstance(content, str):
        raise ProviderContractError("Assistant message content must be a string or null.")

    tool_calls = message.get("tool_calls")
    if tool_calls is not None and not isinstance(tool_calls, list):
        raise ProviderContractError("Assistant message tool_calls must be a list when present.")
    if content is None and not tool_calls:
        raise ProviderContractError(
            "Assistant message must contain text or at least one tool call."
        )

    return deepcopy(message)


def copy_provider_response(response: ProviderResponse) -> ProviderResponse:
    """Return a defensive copy of both normalized and raw provider evidence.

    The dataclass itself is frozen, but its dictionaries are not.  Copying at
    the runtime boundary prevents caller or provider mutations from rewriting a
    tool-loop trace after it has been returned or attached to an exception.
    """

    return ProviderResponse(
        message=deepcopy(response.message),
        raw_response=deepcopy(response.raw_response),
        finish_reason=response.finish_reason,
    )
