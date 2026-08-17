"""Offline tests for the interactive CLI integration.

The tests inject a provider and scripted input reader directly into `run_chat`.
This exercises the real `Agent` conversation behavior without reading a terminal
or contacting Ollama.
"""

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from detecting_an_insider_llm.cli import build_parser, run_chat
from detecting_an_insider_llm.providers import OllamaClientError
from detecting_an_insider_llm.runtime import ProviderResponse
from detecting_an_insider_llm.runtime.agents import ThinkingMode


class StubChatClient:
    """Context-managed provider double used by the CLI loop."""

    provider_name = "ollama"
    model_name = "test-model"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.closed = False

    def __enter__(self) -> "StubChatClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.closed = True

    def runtime_metadata(self) -> dict[str, Any]:
        return {"provider_version": "test"}

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
        options: Mapping[str, Any] | None = None,
        *,
        think: ThinkingMode = False,
    ) -> ProviderResponse:
        """Record the full conversation and return a numbered response."""
        self.calls.append(
            {
                "messages": deepcopy(list(messages)),
                "options": deepcopy(dict(options)) if options is not None else None,
                "think": think,
            }
        )
        message = {
            "role": "assistant",
            "content": f"reply {len(self.calls)}",
        }
        return ProviderResponse(
            message=message,
            raw_response={"message": message},
        )


def _scripted_input(*messages: str):
    """Create an input function that returns predetermined terminal messages."""
    remaining = iter(messages)

    def read(_: str) -> str:
        return next(remaining)

    return read


def test_chat_parser_exposes_provider_and_sampling_configuration() -> None:
    """The public parser retains every setting needed to construct one session."""
    parser = build_parser()

    args = parser.parse_args(
        [
            "chat",
            "--provider",
            "ollama",
            "--model",
            "qwen3",
            "--base-url",
            "http://ollama.test",
            "--temperature",
            "0.4",
            "--top-k",
            "20",
            "--top-p",
            "0.8",
            "--seed",
            "7",
            "--max-output-tokens",
            "256",
            "--think",
            "medium",
        ]
    )

    # Parsing is tested separately from the loop so argument naming or type
    # conversion failures are easy to diagnose.
    assert args.command == "chat"
    assert args.provider == "ollama"
    assert args.model == "qwen3"
    assert args.base_url == "http://ollama.test"
    assert args.temperature == 0.4
    assert args.top_k == 20
    assert args.top_p == 0.8
    assert args.seed == 7
    assert args.max_output_tokens == 256
    assert args.think == "medium"


def test_chat_loop_preserves_history_until_quit() -> None:
    """Repeated prompts share one Agent and /quit is never sent to the model."""
    args = build_parser().parse_args(
        [
            "chat",
            "--model",
            "test-model",
            "--system-prompt",
            "Stay controlled.",
            "--temperature",
            "0.3",
        ]
    )
    provider = StubChatClient()
    output: list[str] = []

    result = run_chat(
        args,
        input_reader=_scripted_input("first message", "second message", "/quit"),
        write_line=output.append,
        client_factory=lambda _: provider,
    )

    assert result == 0
    assert provider.closed is True
    assert output == [
        "Interactive session with ollama/test-model. Type /quit to exit.",
        "assistant> reply 1",
        "assistant> reply 2",
    ]

    # The first request contains only the configured system prompt and first
    # user message.
    assert provider.calls[0] == {
        "messages": [
            {"role": "system", "content": "Stay controlled."},
            {"role": "user", "content": "first message"},
        ],
        "options": {"temperature": 0.3},
        "think": False,
    }

    # The second request includes the earlier assistant response, proving that
    # the CLI did not recreate Agent between prompts.
    assert provider.calls[1] == {
        "messages": [
            {"role": "system", "content": "Stay controlled."},
            {"role": "user", "content": "first message"},
            {"role": "assistant", "content": "reply 1"},
            {"role": "user", "content": "second message"},
        ],
        "options": {"temperature": 0.3},
        "think": False,
    }

    # Only two calls prove /quit was interpreted by the CLI, not sent as a third
    # user message.
    assert len(provider.calls) == 2


def test_chat_loop_allows_retry_without_committing_failed_turn() -> None:
    """A provider error is displayed and its user message stays out of history."""

    class FailOnceClient(StubChatClient):
        def __init__(self) -> None:
            super().__init__()
            self.attempts = 0

        def chat(
            self,
            messages: Sequence[Mapping[str, Any]],
            tools: Sequence[Mapping[str, Any]] | None = None,
            options: Mapping[str, Any] | None = None,
            *,
            think: ThinkingMode = False,
        ) -> ProviderResponse:
            self.attempts += 1
            if self.attempts == 1:
                raise OllamaClientError("temporary failure")
            return super().chat(messages, tools, options, think=think)

    args = build_parser().parse_args(["chat", "--model", "test-model"])
    provider = FailOnceClient()
    output: list[str] = []

    result = run_chat(
        args,
        input_reader=_scripted_input("failed turn", "retry turn", "/quit"),
        write_line=output.append,
        client_factory=lambda _: provider,
    )

    assert result == 0
    assert output == [
        "Interactive session with ollama/test-model. Type /quit to exit.",
        "error: temporary failure",
        "assistant> reply 1",
    ]

    # Agent prepares but does not commit failed turns. The successful retry is
    # therefore the first and only user message in provider-visible history.
    assert provider.calls == [
        {
            "messages": [{"role": "user", "content": "retry turn"}],
            "options": None,
            "think": False,
        }
    ]
