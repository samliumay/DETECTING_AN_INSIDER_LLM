"""Provider-neutral agent primitives.

Architecture
------------
The runtime owns the provider contract because it defines what an agent needs,
not how a particular API works. Provider adapters, such as `OllamaClient`,
implement that contract without being imported here. This dependency direction
allows the same `Agent` to use Ollama, Gemini, or a test double.

This module deliberately stops at one provider turn. A safe research agent also
needs an allowlisted tool dispatcher, bounded tool rounds, automatic action
logging, and journaling. Those concerns are kept out of this first component so
provider selection does not silently grant tool-execution authority.
"""

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal, Protocol

# Provider messages and tool definitions contain nested, provider-defined JSON.
# Keeping the boundary JSON-shaped avoids leaking an Ollama SDK class into the
# runtime while later schemas are still being designed.
Message = dict[str, Any]

# Ollama supports booleans and named effort levels. Other adapters can translate
# these shared values to their provider-specific representation.
ThinkingMode = bool | Literal["high", "medium", "low"]


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

    The dataclass is frozen so its attributes cannot be reassigned accidentally.
    Nested JSON values remain mutable, so the runtime takes defensive copies
    before storing them.
    """

    message: Message
    raw_response: dict[str, Any]


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
            model but are not executed by this class.
        options:
            Sampling/runtime options forwarded without provider-specific
            interpretation.
        think:
            Whether, or at what supported level, the provider should request
            exposed reasoning.

    This first version performs exactly one provider call per :meth:`run`. It
    preserves returned tool calls but intentionally does not execute them. Tool
    execution needs a bounded, allowlisted dispatcher with automatic logging,
    which belongs in a separate runtime component.
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
        assistant_message = _validated_assistant_message(response)

        # This is the only state commit in the method.
        self._messages = [*request_messages, assistant_message]
        return response

    def reset(self) -> None:
        """Clear prior turns while retaining the configured system prompt."""
        self._messages = self._initial_messages()

    def _initial_messages(self) -> list[Message]:
        """Build fresh initial state so reset never reuses mutable messages."""
        if self._system_prompt is None:
            return []
        return [{"role": "system", "content": self._system_prompt}]


def _validated_assistant_message(response: ProviderResponse) -> Message:
    """Validate the cross-provider fields required for conversation history.

    Provider-specific fields are preserved. Only the invariants needed by the
    runtime are checked: the normalized object type, assistant role, text type,
    and tool-call container type.
    """
    if not isinstance(response, ProviderResponse):
        raise ProviderContractError("Provider must return a ProviderResponse.")

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
