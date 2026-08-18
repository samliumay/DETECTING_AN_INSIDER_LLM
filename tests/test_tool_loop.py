"""Offline tests for bounded agent/tool-loop orchestration.

The provider is scripted and the mailbox is in memory, so these tests exercise
real parsing, dispatch, message construction, conversation commits, and limits
without contacting Ollama or sending email.
"""

import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

import pytest

from detecting_an_insider_llm.runtime import (
    Agent,
    ProviderResponse,
    ToolLoopProviderError,
)
from detecting_an_insider_llm.runtime.agents import ThinkingMode
from detecting_an_insider_llm.tools import (
    EmailToolDispatcher,
    SimulatedMailbox,
    email_tool_definitions,
)


class ScriptedProvider:
    """Return predetermined provider responses and retain exact requests.

    The double implements the structural provider protocol but does not inherit
    from it.  Responses may include exceptions, allowing failure behavior to be
    tested at the same boundary as a real Ollama error.
    """

    provider_name = "scripted"
    model_name = "scripted-model"

    def __init__(self, responses: Sequence[ProviderResponse | Exception]) -> None:
        """Copy the response script and initialize an empty request trace."""

        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def runtime_metadata(self) -> dict[str, Any]:
        """Return deterministic provenance without external I/O."""

        return {"provider_version": "test"}

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
        options: Mapping[str, Any] | None = None,
        *,
        think: ThinkingMode = False,
    ) -> ProviderResponse:
        """Record one call, then return or raise the next scripted item.

        Defensive copies make the recorded request a trustworthy snapshot even
        if the agent later appends messages during the same user turn.
        """

        self.calls.append(
            {
                "messages": deepcopy(list(messages)),
                "tools": deepcopy(list(tools)) if tools is not None else None,
                "options": deepcopy(dict(options)) if options is not None else None,
                "think": think,
            }
        )
        if not self._responses:
            raise AssertionError("ScriptedProvider received an unexpected call.")
        next_item = self._responses.pop(0)
        if isinstance(next_item, Exception):
            raise next_item
        return deepcopy(next_item)


def _tool_call(
    name: str,
    arguments: object,
    *,
    call_id: str | None = None,
) -> dict[str, Any]:
    """Build the provider tool-call shape consumed by the runtime parser.

    The optional ID exercises APIs that correlate tool results by an identifier;
    Ollama calls can omit it and rely on `tool_name`.
    """

    call: dict[str, Any] = {
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }
    if call_id is not None:
        call["id"] = call_id
    return call


def _response(
    *,
    content: str = "",
    tool_calls: Sequence[object] = (),
) -> ProviderResponse:
    """Build one normalized response with optional model-requested calls.

    A separate raw response copy is included because the loop result promises to
    preserve both normalized conversation data and provider evidence.
    """

    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = list(tool_calls)
    return ProviderResponse(
        message=message,
        raw_response={"message": deepcopy(message), "fixture": True},
    )


def _mailbox() -> SimulatedMailbox:
    """Create isolated older-mail state used by read-loop tests."""

    return SimulatedMailbox(
        [
            {
                "email_id": "older-001",
                "timestamp": "2026-08-15T09:00:00Z",
                "sender": "manager@company.test",
                "recipient": "agent@research.test",
                "subject": "Synthetic request",
                "body": "Review this controlled message.",
            }
        ]
    )


def test_tool_loop_reads_email_and_returns_to_provider_for_final_answer() -> None:
    """Verify the complete request → tool result → final answer lifecycle.

    The first response requests a known email.  The test confirms that the
    second provider request contains the assistant call and model-only receipt,
    while the returned execution retains the hidden simulation audit field.
    """

    provider = ScriptedProvider(
        [
            _response(
                tool_calls=[
                    _tool_call(
                        "read_email",
                        {"email_id": "older-001"},
                        call_id="call-1",
                    )
                ]
            ),
            _response(content="I read the requested email."),
        ]
    )
    mailbox = _mailbox()
    agent = Agent(
        provider,
        tools=email_tool_definitions(),
        options={"temperature": 0.2},
    )

    result = agent.run_with_tools(
        "Read older-001.",
        executor=EmailToolDispatcher(mailbox),
        max_tool_rounds=3,
    )

    assert result.termination_reason == "completed"
    assert result.tool_rounds == 1
    assert result.last_response.message["content"] == "I read the requested email."
    assert len(result.provider_responses) == 2
    assert len(result.tool_executions) == 1

    execution = result.tool_executions[0]
    assert execution.sequence == 1
    assert execution.call.requested_tool_name == "read_email"
    assert execution.call.raw_arguments == {"email_id": "older-001"}
    assert execution.result.status == "succeeded"
    assert execution.result.audit_result["simulated"] is True
    assert "simulated" not in execution.result.model_result

    second_request = provider.calls[1]["messages"]
    assert second_request[-1]["role"] == "tool"
    assert second_request[-1]["tool_name"] == "read_email"
    assert second_request[-1]["tool_call_id"] == "call-1"
    assert json.loads(second_request[-1]["content"]) == execution.result.model_result
    assert {
        tool["function"]["name"] for tool in provider.calls[0]["tools"]
    } == {"list_emails", "read_email", "send_email"}
    assert agent.messages[-1] == {
        "role": "assistant",
        "content": "I read the requested email.",
    }


def test_tool_loop_lists_headers_without_leaking_email_bodies() -> None:
    """Verify `list_emails` is connected through the allowlisted runtime path.

    The scripted model requests discovery with an empty argument object.  Its
    second request must receive the available ID and natural header, but not the
    full body that is reserved for a later `read_email` call.
    """

    provider = ScriptedProvider(
        [
            _response(tool_calls=[_tool_call("list_emails", {}, call_id="list-1")]),
            _response(content="I found one available email."),
        ]
    )
    agent = Agent(provider, tools=email_tool_definitions())

    result = agent.run_with_tools(
        "Discover available mail.",
        executor=EmailToolDispatcher(_mailbox()),
        max_tool_rounds=2,
    )

    execution = result.tool_executions[0]
    receipt = json.loads(provider.calls[1]["messages"][-1]["content"])
    assert execution.call.requested_tool_name == "list_emails"
    assert execution.result.status == "succeeded"
    assert receipt["emails"][0]["email_id"] == "older-001"
    assert receipt["emails"][0]["timestamp"] == "2026-08-15T09:00:00Z"
    assert "body" not in receipt["emails"][0]
    assert "Review this controlled message." not in provider.calls[1]["messages"][-1][
        "content"
    ]


def test_tool_loop_rejects_unknown_and_malformed_calls_then_continues() -> None:
    """Verify unsafe names and broken structures become observable rejections.

    Both invalid calls occur in one assistant batch.  They must each produce a
    provider-visible tool result, consume no email capability, and still allow
    the model to produce a final response on the following provider call.
    """

    provider = ScriptedProvider(
        [
            _response(
                tool_calls=[
                    _tool_call("delete_email", ["older-001"]),
                    {"type": "function", "function": "not-an-object"},
                ]
            ),
            _response(content="The requests were rejected."),
        ]
    )
    mailbox = _mailbox()
    agent = Agent(provider, tools=email_tool_definitions())

    result = agent.run_with_tools(
        "Try unsupported actions.",
        executor=EmailToolDispatcher(mailbox),
        max_tool_rounds=2,
    )

    assert result.termination_reason == "completed"
    assert [item.call.requested_tool_name for item in result.tool_executions] == [
        "delete_email",
        "<missing>",
    ]
    assert [item.result.status for item in result.tool_executions] == [
        "rejected",
        "rejected",
    ]
    assert [
        item.result.model_result["error_code"] for item in result.tool_executions
    ] == ["unknown_tool", "unknown_tool"]
    assert mailbox.sent_emails == ()
    assert [message["role"] for message in provider.calls[1]["messages"][-2:]] == [
        "tool",
        "tool",
    ]


def test_tool_loop_records_calls_rejected_by_round_limit() -> None:
    """Verify additional model requests are retained but never dispatched.

    With a one-round limit, the first send succeeds.  The model's second send is
    still an attempted action, but it receives a `max_tool_rounds` rejection and
    cannot add another outbox message or trigger a third provider call.
    """

    send_arguments = {
        "to": "archive@company.test",
        "subject": "Synthetic archive",
        "body": "Controlled content.",
    }
    provider = ScriptedProvider(
        [
            _response(tool_calls=[_tool_call("send_email", send_arguments)]),
            _response(tool_calls=[_tool_call("send_email", send_arguments)]),
        ]
    )
    mailbox = _mailbox()
    agent = Agent(provider, tools=email_tool_definitions())

    result = agent.run_with_tools(
        "Send repeatedly.",
        executor=EmailToolDispatcher(mailbox),
        max_tool_rounds=1,
    )

    assert result.termination_reason == "max_tool_rounds"
    assert result.tool_rounds == 1
    assert len(provider.calls) == 2
    assert [item.sequence for item in result.tool_executions] == [1, 2]
    assert [item.result.status for item in result.tool_executions] == [
        "succeeded",
        "rejected",
    ]
    assert result.tool_executions[1].result.model_result["error_code"] == (
        "max_tool_rounds"
    )
    assert [email.email_id for email in mailbox.sent_emails] == ["sent-0001"]
    assert agent.messages[-1]["role"] == "tool"


def test_tool_loop_provider_failure_preserves_partial_action_evidence() -> None:
    """Verify a later provider failure cannot erase an executed action.

    After one successful read, the scripted provider fails while receiving its
    result.  The raised wrapper must expose the action trace and original cause,
    while Agent history retains the user, assistant request, and tool receipt.
    """

    provider = ScriptedProvider(
        [
            _response(
                tool_calls=[
                    _tool_call("read_email", {"email_id": "older-001"})
                ]
            ),
            RuntimeError("provider unavailable"),
        ]
    )
    agent = Agent(provider, tools=email_tool_definitions())

    with pytest.raises(ToolLoopProviderError, match="provider unavailable") as caught:
        agent.run_with_tools(
            "Read older-001.",
            executor=EmailToolDispatcher(_mailbox()),
            max_tool_rounds=2,
        )

    assert isinstance(caught.value.__cause__, RuntimeError)
    assert len(caught.value.provider_responses) == 1
    assert len(caught.value.tool_executions) == 1
    assert caught.value.tool_executions[0].result.status == "succeeded"
    assert [message["role"] for message in agent.messages] == [
        "user",
        "assistant",
        "tool",
    ]


def test_tool_loop_first_provider_failure_keeps_transactional_history() -> None:
    """Verify failure before any tool attempt preserves the prior conversation.

    The original exception is propagated rather than wrapped because no partial
    action trace exists, and the uncompleted user message must not be committed.
    """

    provider = ScriptedProvider([RuntimeError("first call failed")])
    agent = Agent(
        provider,
        system_prompt="Stay controlled.",
        tools=email_tool_definitions(),
    )

    with pytest.raises(RuntimeError, match="first call failed"):
        agent.run_with_tools(
            "This turn fails.",
            executor=EmailToolDispatcher(_mailbox()),
            max_tool_rounds=2,
        )

    assert agent.messages == ({"role": "system", "content": "Stay controlled."},)


@pytest.mark.parametrize("invalid_limit", [0, -1, True])
def test_tool_loop_rejects_invalid_round_limits(invalid_limit: int) -> None:
    """Verify invalid bounds fail before calling the provider or a tool.

    Zero, negative integers, and booleans cannot represent a positive execution
    budget.  Early validation prevents an ambiguous or accidentally unbounded
    experiment configuration.
    """

    provider = ScriptedProvider([_response(content="unused")])
    agent = Agent(provider, tools=email_tool_definitions())

    with pytest.raises(ValueError, match="max_tool_rounds"):
        agent.run_with_tools(
            "Do something.",
            executor=EmailToolDispatcher(_mailbox()),
            max_tool_rounds=invalid_limit,
        )

    assert provider.calls == []
