"""Unit tests for the provider-neutral agent runtime.

These tests use a structural fake provider rather than `OllamaClient`. The goal
is to prove that `Agent` depends only on the `ChatProvider` behavior and that
conversation-state rules can be verified without a network, model, or provider
SDK.
"""

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

import pytest

from detecting_an_insider_llm.runtime.agents import (
    Agent,
    ProviderResponse,
    ThinkingMode,
)


class StubProvider:
    """Minimal provider test double implementing the shared protocol.

    It deliberately does not inherit from `ChatProvider`. Successful injection
    therefore demonstrates structural compatibility: another provider only
    needs the required attributes and methods.
    """

    provider_name = "stub"
    model_name = "stub-model"

    def __init__(self) -> None:
        # Calls are retained so tests can inspect the exact boundary values the
        # agent sent to its provider.
        self.calls: list[dict[str, Any]] = []

    def runtime_metadata(self) -> dict[str, Any]:
        """Return deterministic metadata without external I/O."""
        return {"provider_version": "test"}

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
        options: Mapping[str, Any] | None = None,
        *,
        think: ThinkingMode = False,
    ) -> ProviderResponse:
        """Record one request and return a valid normalized assistant turn."""
        # Copy nested request data just as a real provider boundary may do. This
        # prevents later mutations from changing what the test observed.
        self.calls.append(
            {
                "messages": deepcopy(list(messages)),
                "tools": deepcopy(list(tools)) if tools is not None else None,
                "options": deepcopy(dict(options)) if options is not None else None,
                "think": think,
            }
        )
        message = {"role": "assistant", "content": "acknowledged"}
        return ProviderResponse(
            message=message,
            raw_response={"message": message, "provider_field": 1},
        )


def test_agent_accepts_structural_provider_and_commits_successful_turn() -> None:
    """A successful turn forwards frozen configuration and commits both messages."""
    # Arrange: provide mutable inputs so the test can verify that Agent snapshots
    # its configuration rather than holding caller-owned references.
    provider = StubProvider()
    tools = [{"type": "function", "function": {"name": "read_email"}}]
    options = {"temperature": 0.3}
    agent = Agent(
        provider,
        system_prompt="Follow the scenario.",
        tools=tools,
        options=options,
        think=False,
    )

    # Change the original objects after Agent construction. These changes must
    # not alter the request, or recorded experiment configuration could diverge
    # from the values actually sent to a model.
    tools[0]["function"]["name"] = "changed"
    options["temperature"] = 1.0

    # Act: one run should make exactly one provider call.
    response = agent.run("Read the first email.")

    # Assert: the normalized response is returned to the caller.
    assert response.message["content"] == "acknowledged"

    # The provider must receive the original snapshots and correctly ordered
    # system/user messages, not the post-construction mutations.
    assert provider.calls == [
        {
            "messages": [
                {"role": "system", "content": "Follow the scenario."},
                {"role": "user", "content": "Read the first email."},
            ],
            "tools": [
                {"type": "function", "function": {"name": "read_email"}}
            ],
            "options": {"temperature": 0.3},
            "think": False,
        }
    ]

    # Only a validated provider response is committed to conversation history.
    assert agent.messages == (
        {"role": "system", "content": "Follow the scenario."},
        {"role": "user", "content": "Read the first email."},
        {"role": "assistant", "content": "acknowledged"},
    )


def test_agent_does_not_commit_user_turn_when_provider_fails() -> None:
    """A provider failure leaves the last committed conversation unchanged."""

    class FailingProvider(StubProvider):
        """Test double that fails before producing an assistant message."""

        def chat(
            self,
            messages: Sequence[Mapping[str, Any]],
            tools: Sequence[Mapping[str, Any]] | None = None,
            options: Mapping[str, Any] | None = None,
            *,
            think: ThinkingMode = False,
        ) -> ProviderResponse:
            raise RuntimeError("provider unavailable")

    agent = Agent(FailingProvider(), system_prompt="Stay controlled.")

    # The original provider error remains visible to the future runner, which
    # will be responsible for journaling it and deciding whether to retry.
    with pytest.raises(RuntimeError, match="provider unavailable"):
        agent.run("Hello")

    # The user message was only temporary request state and must not be mistaken
    # for a completed conversation turn.
    assert agent.messages == ({"role": "system", "content": "Stay controlled."},)


def test_agent_reset_keeps_system_prompt_and_clears_turns() -> None:
    """Reset starts a new episode without losing the configured instruction."""
    agent = Agent(StubProvider(), system_prompt="Stay controlled.")
    agent.run("Hello")

    agent.reset()

    # The system prompt is configuration, while user/assistant turns are episode
    # state and must be removed.
    assert agent.messages == ({"role": "system", "content": "Stay controlled."},)
