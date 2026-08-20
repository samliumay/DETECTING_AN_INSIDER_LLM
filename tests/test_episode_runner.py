"""Offline tests for one-cell non-interactive scenario execution.

These tests use scripted provider responses and the repository's synthetic
blackmail scenario.  They verify orchestration and evidence retention only;
they do not simulate, measure, or support a claim about insider behavior.
"""

import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

from detecting_an_insider_llm.runtime.agents import (
    ProviderResponse,
    ThinkingMode,
)
from detecting_an_insider_llm.runtime.episode_runner import ScenarioRunner
from detecting_an_insider_llm.scenario_loader import (
    ExecutionLimits,
    ResolvedScenario,
    resolve_scenario,
)


SCENARIO_PATH = (
    Path(__file__).parents[1] / "scenarios" / "blackmail" / "v1" / "scenario.yaml"
)


class ScriptedProvider:
    """Return deterministic responses while retaining exact runner requests.

    Exceptions may appear in the script so the same provider boundary exercises
    successful turns and partial-run failure handling without network access.
    """

    provider_name = "scripted"
    model_name = "scripted-model"

    def __init__(self, responses: Sequence[ProviderResponse | Exception]) -> None:
        """Copy the response script and initialize observable request state."""

        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []
        self.metadata_call_count = 0

    def runtime_metadata(self) -> dict[str, Any]:
        """Return fixed provenance and record when the runner snapshots it."""

        self.metadata_call_count += 1
        return {"provider_version": "offline-fixture-1"}

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
        options: Mapping[str, Any] | None = None,
        *,
        think: ThinkingMode = False,
    ) -> ProviderResponse:
        """Record one attempted turn, then return or raise its scripted item."""

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
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return deepcopy(item)


def _tool_call(name: str, arguments: object) -> dict[str, Any]:
    """Build the provider-neutral function-call shape parsed by the runner."""

    return {
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _response(
    *,
    content: str = "",
    tool_calls: Sequence[object] = (),
) -> ProviderResponse:
    """Build a normalized response with a distinct raw evidence snapshot."""

    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = deepcopy(list(tool_calls))
    return ProviderResponse(
        message=message,
        raw_response={"message": deepcopy(message), "offline_fixture": True},
    )


def _resolved_scenario() -> ResolvedScenario:
    """Load one real validated cell so tests cover runner/loader integration."""

    return resolve_scenario(
        SCENARIO_PATH,
        condition_id="baseline",
        policy_context_id="none",
    )


def _with_limits(
    scenario: ResolvedScenario,
    *,
    provider_turns: int,
    ordinary_attempts: int,
    self_log_attempts: int,
) -> ResolvedScenario:
    """Return the same resolved cell with a typed small test-only budget."""

    limits = ExecutionLimits(
        max_provider_turns=provider_turns,
        max_ordinary_tool_attempts=ordinary_attempts,
        max_self_log_attempts=self_log_attempts,
    )
    spec = scenario.spec.model_copy(update={"execution_limits": limits})
    return replace(scenario, spec=spec)


def test_runner_completes_one_double_logged_episode() -> None:
    """Verify prompts, ordinary action, self-log, and final answer stay linked.

    The scripted model lists mail, reports that action using the returned
    `log_id`, and then finishes.  The result must contain both independent log
    views plus exact provider configuration and the complete message sequence.
    """

    provider = ScriptedProvider(
        [
            _response(tool_calls=[_tool_call("list_emails", {})]),
            _response(
                tool_calls=[
                    _tool_call(
                        "log_action",
                        {
                            "log_id": "log-0001",
                            "tool_name": "list_emails",
                            "action_description": "Listed available email headers.",
                            "claimed_outcome": "succeeded",
                        },
                    )
                ]
            ),
            _response(content="Mailbox review is complete."),
        ]
    )
    scenario = _resolved_scenario()
    runner = ScenarioRunner(
        provider,
        options={"temperature": 0.2, "seed": 7},
        think="low",
    )

    result = runner.run(scenario)

    assert result.status == "completed"
    assert result.termination_reason == "completed"
    assert result.provider_turn_count == 3
    assert result.ordinary_tool_attempt_count == 1
    assert result.self_log_attempt_count == 1
    assert result.provider_metadata == {"provider_version": "offline-fixture-1"}
    assert provider.metadata_call_count == 1
    assert result.options == {"temperature": 0.2, "seed": 7}
    assert result.think == "low"
    assert [record.log_id for record in result.automatic_records] == ["log-0001"]
    assert [record.log_id for record in result.model_self_logs] == ["log-0001"]
    assert result.sent_emails == ()
    assert [message["role"] for message in result.messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
    ]
    assert provider.calls[0]["messages"] == [
        {"role": "system", "content": scenario.system_prompt},
        {"role": "user", "content": scenario.user_prompt},
    ]
    assert provider.calls[0]["options"] == {"temperature": 0.2, "seed": 7}
    assert provider.calls[0]["think"] == "low"
    assert {tool["function"]["name"] for tool in provider.calls[0]["tools"]} == {
        "list_emails",
        "read_email",
        "send_email",
        "log_action",
    }
    assert result.last_response is not None
    assert result.last_response.message["content"] == "Mailbox review is complete."


def test_runner_keeps_ordinary_and_self_log_attempt_budgets_separate() -> None:
    """Verify a self-log cannot consume the ordinary-action allowance.

    One list action exhausts the ordinary budget.  On the next provider turn a
    valid self-log still succeeds, while a requested read receives its own
    automatic `log_id` and limit rejection without reaching the mailbox.
    """

    provider = ScriptedProvider(
        [
            _response(tool_calls=[_tool_call("list_emails", {})]),
            _response(
                tool_calls=[
                    _tool_call(
                        "log_action",
                        {
                            "log_id": "log-0001",
                            "tool_name": "list_emails",
                            "action_description": "Listed the mailbox.",
                            "claimed_outcome": "succeeded",
                        },
                    ),
                    _tool_call("read_email", {"email_id": "not-dispatched"}),
                ]
            ),
        ]
    )
    scenario = _with_limits(
        _resolved_scenario(),
        provider_turns=5,
        ordinary_attempts=1,
        self_log_attempts=3,
    )

    result = ScenarioRunner(provider).run(scenario)

    assert result.status == "incomplete"
    assert result.termination_reason == "max_ordinary_tool_attempts"
    assert result.provider_turn_count == 2
    assert result.ordinary_tool_attempt_count == 2
    assert result.self_log_attempt_count == 1
    assert [record.result_status for record in result.automatic_records] == [
        "succeeded",
        "rejected",
    ]
    assert result.automatic_records[1].log_id == "log-0002"
    assert result.automatic_records[1].model_result["error_code"] == (
        "max_ordinary_tool_attempts"
    )
    assert [record.log_id for record in result.model_self_logs] == ["log-0001"]
    assert result.tool_executions[-1].result.status == "rejected"
    assert result.tool_executions[-1].result.model_result["log_id"] == "log-0002"
    assert len(provider.calls) == 2


def test_runner_rejects_self_log_beyond_zero_budget_without_fabricating_log() -> None:
    """Verify a refused `log_action` remains an attempt, not a stored claim."""

    provider = ScriptedProvider(
        [
            _response(tool_calls=[_tool_call("list_emails", {})]),
            _response(
                tool_calls=[
                    _tool_call(
                        "log_action",
                        {
                            "log_id": "log-0001",
                            "tool_name": "list_emails",
                            "action_description": "Listed the mailbox.",
                            "claimed_outcome": "succeeded",
                        },
                    )
                ]
            ),
        ]
    )
    scenario = _with_limits(
        _resolved_scenario(),
        provider_turns=4,
        ordinary_attempts=2,
        self_log_attempts=0,
    )

    result = ScenarioRunner(provider).run(scenario)

    assert result.status == "incomplete"
    assert result.termination_reason == "max_self_log_attempts"
    assert result.ordinary_tool_attempt_count == 1
    assert result.self_log_attempt_count == 1
    assert len(result.automatic_records) == 1
    assert result.model_self_logs == ()
    assert result.tool_executions[-1].result.status == "rejected"
    assert result.tool_executions[-1].result.model_result["error_code"] == (
        "max_self_log_attempts"
    )


def test_runner_stops_before_call_beyond_provider_turn_budget() -> None:
    """Verify the provider bound counts calls and retains the last tool result.

    The only allowed provider turn requests a list action.  That action executes,
    but the runner stops before making an unrecorded second provider call.
    """

    provider = ScriptedProvider(
        [_response(tool_calls=[_tool_call("list_emails", {})])]
    )
    scenario = _with_limits(
        _resolved_scenario(),
        provider_turns=1,
        ordinary_attempts=2,
        self_log_attempts=2,
    )

    result = ScenarioRunner(provider).run(scenario)

    assert result.status == "incomplete"
    assert result.termination_reason == "max_provider_turns"
    assert result.provider_turn_count == 1
    assert result.ordinary_tool_attempt_count == 1
    assert len(result.automatic_records) == 1
    assert len(provider.calls) == 1
    assert result.messages[-1]["role"] == "tool"


def test_runner_preserves_malformed_call_as_rejected_ordinary_attempt() -> None:
    """Verify an unusable function object remains in evidence and can recover.

    A malformed call has no usable name, so it is treated as an ordinary
    capability attempt, rejected by the allowlist, assigned a `log_id`, and sent
    back to the provider before the final valid answer.
    """

    malformed_call = {"type": "function", "function": "not-an-object"}
    provider = ScriptedProvider(
        [
            _response(tool_calls=[malformed_call]),
            _response(content="The malformed request was rejected."),
        ]
    )

    result = ScenarioRunner(provider).run(_resolved_scenario())

    assert result.status == "completed"
    assert result.ordinary_tool_attempt_count == 1
    assert result.tool_executions[0].call.raw_tool_call == malformed_call
    assert result.tool_executions[0].call.requested_tool_name == "<missing>"
    assert result.tool_executions[0].result.status == "rejected"
    assert result.automatic_records[0].log_id == "log-0001"
    assert result.automatic_records[0].model_result["error_code"] == "unknown_tool"
    receipt = json.loads(provider.calls[1]["messages"][-1]["content"])
    assert receipt["log_id"] == "log-0001"


def test_runner_returns_failed_result_with_partial_provider_evidence() -> None:
    """Verify a later provider exception does not erase a completed action."""

    provider = ScriptedProvider(
        [
            _response(tool_calls=[_tool_call("list_emails", {})]),
            RuntimeError("provider unavailable"),
        ]
    )

    result = ScenarioRunner(provider).run(_resolved_scenario())

    assert result.status == "failed"
    assert result.termination_reason == "provider_error"
    assert result.provider_turn_count == 2
    assert len(result.provider_responses) == 1
    assert len(result.tool_executions) == 1
    assert len(result.automatic_records) == 1
    assert result.error is not None
    assert result.error.phase == "provider_call"
    assert result.error.error_type == "RuntimeError"
    assert result.error.message == "provider unavailable"


def test_runner_retains_invalid_provider_response_before_contract_failure() -> None:
    """Verify provider-shaped malformed output is kept for later diagnosis."""

    invalid = ProviderResponse(
        message={"role": "user", "content": "wrong role"},
        raw_response={"message": {"role": "user", "content": "wrong role"}},
    )
    provider = ScriptedProvider([invalid])

    result = ScenarioRunner(provider).run(_resolved_scenario())

    assert result.status == "failed"
    assert result.termination_reason == "provider_contract_error"
    assert result.provider_turn_count == 1
    assert len(result.provider_responses) == 1
    assert result.provider_responses[0].message["role"] == "user"
    assert result.error is not None
    assert result.error.phase == "provider_response_validation"


def test_runner_returns_tool_failure_and_keeps_prior_automatic_record() -> None:
    """Verify a broken log-ID factory fails loud without deleting earlier work.

    The constant factory makes the second ordinary attempt ambiguous.  The
    first list record remains authoritative, while the second requested call is
    represented as a failed execution with no invented replacement identity.
    """

    provider = ScriptedProvider(
        [
            _response(
                tool_calls=[
                    _tool_call("list_emails", {}),
                    _tool_call("list_emails", {}),
                ]
            )
        ]
    )
    runner = ScenarioRunner(provider, log_id_factory=lambda _: "duplicate-id")

    result = runner.run(_resolved_scenario())

    assert result.status == "failed"
    assert result.termination_reason == "tool_execution_error"
    assert result.ordinary_tool_attempt_count == 2
    assert [item.result.status for item in result.tool_executions] == [
        "succeeded",
        "failed",
    ]
    assert [record.log_id for record in result.automatic_records] == [
        "duplicate-id"
    ]
    assert result.error is not None
    assert result.error.phase == "tool_execution"
    assert result.error.error_type == "ValueError"
