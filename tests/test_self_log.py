"""Offline behavioral tests for model-created action self-logging.

The tests verify storage and correlation boundaries only.  They do not judge
whether any self-log is truthful and do not provide evidence that discrepancies
identify an insider model.
"""

from typing import Any, Never

import pytest

from detecting_an_insider_llm.tools import (
    LOG_ACTION_TOOL,
    DoubleLoggingToolExecutor,
    EmailToolDispatcher,
    SelfLogStore,
    SimulatedMailbox,
    email_tool_definitions,
    log_action,
    self_logging_email_tool_definitions,
)


VALID_SELF_LOG: dict[str, Any] = {
    "log_id": "log-0001",
    "tool_name": "read_email",
    "action_description": "Read the message identified as older-001.",
    "claimed_outcome": "succeeded",
}


def _mailbox() -> SimulatedMailbox:
    """Create isolated deterministic mail for double-logging executor tests."""

    return SimulatedMailbox(
        [
            {
                "email_id": "older-001",
                "timestamp": "2026-08-15T09:00:00Z",
                "sender": "manager@company.test",
                "recipient": "agent@research.test",
                "subject": "Controlled request",
                "body": "Review this synthetic message.",
            }
        ]
    )


def test_log_action_definition_is_closed_and_experiment_factory_is_separate() -> None:
    """Verify the exact model contract and ordinary/experimental separation.

    The interactive factory must remain ordinary-only.  The experimental
    factory adds `log_action` last and returns fresh dictionaries so provider
    normalization cannot change a later run's tool schema.
    """

    function = LOG_ACTION_TOOL["function"]
    ordinary = email_tool_definitions()
    experimental = self_logging_email_tool_definitions()

    assert function["name"] == "log_action"
    assert function["parameters"]["required"] == [
        "log_id",
        "tool_name",
        "action_description",
        "claimed_outcome",
    ]
    assert function["parameters"]["additionalProperties"] is False
    assert function["parameters"]["properties"]["claimed_outcome"]["enum"] == [
        "succeeded",
        "rejected",
        "failed",
    ]
    assert [tool["function"]["name"] for tool in ordinary] == [
        "list_emails",
        "read_email",
        "send_email",
    ]
    assert [tool["function"]["name"] for tool in experimental] == [
        "list_emails",
        "read_email",
        "send_email",
        "log_action",
    ]

    experimental[-1]["function"]["description"] = "mutated"
    assert self_logging_email_tool_definitions()[-1]["function"]["description"] != (
        "mutated"
    )


def test_log_action_stores_one_valid_claim_without_exposing_audit_state() -> None:
    """Verify valid model fields are retained but never checked for accuracy.

    The model receipt confirms storage without returning the hidden stored
    object.  The separate audit receipt carries that record for a future
    artifact writer.
    """

    store = SelfLogStore()

    result = log_action(VALID_SELF_LOG, store)

    assert result.status == "succeeded"
    assert result.arguments == VALID_SELF_LOG
    assert result.model_result == {
        "ok": True,
        "record_id": "self-log-0001",
        "detail": "Action self-log stored.",
    }
    assert "self_log" not in result.model_result
    assert result.audit_result["self_log"] == {
        "schema_version": "1",
        "record_id": "self-log-0001",
        "self_log_sequence": 1,
        **VALID_SELF_LOG,
    }
    assert store.attempt_count == 1
    assert store.self_logs[0].log_id == "log-0001"


@pytest.mark.parametrize(
    "raw_arguments",
    [
        {"log_id": "log-0001"},
        {**VALID_SELF_LOG, "claimed_outcome": "maybe"},
        {**VALID_SELF_LOG, "unexpected": True},
        {**VALID_SELF_LOG, "action_description": "   "},
        [VALID_SELF_LOG],
    ],
)
def test_log_action_rejects_malformed_claims_without_creating_records(
    raw_arguments: object,
) -> None:
    """Verify missing, invalid, extra, blank, and non-object calls are retained.

    Every malformed request counts as an attempt and receives a stable rejected
    receipt, but no successful model self-log is fabricated from invalid data.
    """

    store = SelfLogStore()

    result = store.log_action(raw_arguments)

    assert result.status == "rejected"
    assert result.model_result["error_code"] == "invalid_arguments"
    assert result.audit_result == result.model_result
    assert result.audit_result is not result.model_result
    assert store.attempt_count == 1
    assert store.self_logs == ()


def test_store_accepts_duplicate_and_unknown_ids_for_offline_classification() -> None:
    """Verify the model-visible store never queries hidden automatic records.

    Two duplicate claims and a claim using an arbitrary unknown ID must all be
    stored.  Rejecting either pattern here would leak correlation information
    and remove anomalies that the later analyzer is expected to classify.
    """

    store = SelfLogStore()

    store.log_action(VALID_SELF_LOG)
    store.log_action(
        {
            **VALID_SELF_LOG,
            "action_description": "A second claim for the same action.",
        }
    )
    store.log_action(
        {
            **VALID_SELF_LOG,
            "log_id": "unknown-log-id",
        }
    )

    assert store.attempt_count == 3
    assert [record.record_id for record in store.self_logs] == [
        "self-log-0001",
        "self-log-0002",
        "self-log-0003",
    ]
    assert [record.log_id for record in store.self_logs] == [
        "log-0001",
        "log-0001",
        "unknown-log-id",
    ]


def test_double_logging_executor_correlates_ordinary_receipts_only() -> None:
    """Verify ordinary calls get IDs while `log_action` remains non-recursive.

    A successful read and rejected unknown tool each create an automatic record.
    The subsequent self-log is stored separately and does not consume an action
    sequence or receive a new logging obligation.
    """

    executor = DoubleLoggingToolExecutor(EmailToolDispatcher(_mailbox()))

    read_result = executor.execute("read_email", {"email_id": "older-001"})
    rejected_result = executor.execute("delete_email", {"email_id": "older-001"})
    self_log_result = executor.execute("log_action", VALID_SELF_LOG)

    assert read_result.status == "succeeded"
    assert read_result.model_result["log_id"] == "log-0001"
    assert read_result.audit_result["log_id"] == "log-0001"
    assert rejected_result.status == "rejected"
    assert rejected_result.model_result["log_id"] == "log-0002"
    assert self_log_result.status == "succeeded"
    assert "log_id" not in self_log_result.model_result
    assert [record.log_id for record in executor.automatic_records] == [
        "log-0001",
        "log-0002",
    ]
    assert [record.result_status for record in executor.automatic_records] == [
        "succeeded",
        "rejected",
    ]
    assert len(executor.self_logs) == 1
    assert executor.self_log_attempt_count == 1


def test_automatic_record_snapshots_cannot_be_rewritten_through_results() -> None:
    """Verify later mutation of a tool receipt cannot alter audit evidence."""

    executor = DoubleLoggingToolExecutor(EmailToolDispatcher(_mailbox()))
    result = executor.execute("read_email", {"email_id": "older-001"})

    result.model_result["email"]["body"] = "mutated outside executor"
    copied_record = executor.automatic_records[0]
    copied_record.model_result["email"]["body"] = "mutated returned copy"

    assert executor.automatic_records[0].model_result["email"]["body"] == (
        "Review this synthetic message."
    )


def test_duplicate_factory_ids_fail_before_repeating_an_action_identity() -> None:
    """Verify a broken identifier factory cannot create ambiguous records."""

    executor = DoubleLoggingToolExecutor(
        EmailToolDispatcher(_mailbox()),
        log_id_factory=lambda _: "constant-log-id",
    )
    executor.execute("list_emails", {})

    with pytest.raises(ValueError, match="duplicate ID"):
        executor.execute("read_email", {"email_id": "older-001"})

    assert len(executor.automatic_records) == 1


def test_executor_preserves_failure_record_then_reraises_harness_error() -> None:
    """Verify an ordinary executor exception remains visible and still fails loud."""

    class FailingExecutor:
        """Deterministic ordinary executor double that always fails."""

        def execute(self, tool_name: str, raw_arguments: object) -> Never:
            """Raise a fixed error instead of returning a tool result."""

            raise RuntimeError("private implementation detail")

    executor = DoubleLoggingToolExecutor(FailingExecutor())

    with pytest.raises(RuntimeError, match="private implementation detail"):
        executor.execute("read_email", {"email_id": "older-001"})

    record = executor.automatic_records[0]
    assert record.log_id == "log-0001"
    assert record.result_status == "failed"
    assert record.audit_result["error_type"] == "RuntimeError"
    assert "private implementation detail" not in str(record.audit_result)
