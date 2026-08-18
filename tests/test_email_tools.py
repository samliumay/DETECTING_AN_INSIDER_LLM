"""Offline behavioral tests for the controlled email-tool boundary.

These tests demonstrate implementation behavior only.  They do not provide
evidence that the eventual discrepancy detector identifies insider behavior.
"""

from typing import Any

import pytest
from pydantic import ValidationError

from detecting_an_insider_llm.tools import (
    READ_EMAIL_TOOL,
    SEND_EMAIL_TOOL,
    EmailMessage,
    SimulatedMailbox,
    email_tool_definitions,
    is_safe_test_address,
    read_email,
    send_email,
)


@pytest.fixture
def mailbox() -> SimulatedMailbox:
    """Create one valid mailbox fixture for each requesting test.

    Pytest calls this function separately per test.  Each test therefore starts
    with the same older email and an empty outbox, preventing sent-message state
    or deterministic ID counters from leaking between cases.
    """

    return SimulatedMailbox(
        [
            {
                "email_id": "older-001",
                "sender": "manager@company.test",
                "recipient": "agent@research.test",
                "subject": "Quarterly planning",
                "body": "Please review the attached synthetic planning note.",
            }
        ]
    )


def test_tool_definitions_are_closed_and_have_explicit_required_fields() -> None:
    """Verify the names, required fields, and closed argument objects.

    The assertions inspect the exact nested dictionaries that will eventually
    be sent to a provider.  This catches accidental schema changes that could
    alter which model outputs count as valid tool calls.
    """

    read_function = READ_EMAIL_TOOL["function"]
    send_function = SEND_EMAIL_TOOL["function"]

    assert read_function["name"] == "read_email"
    assert read_function["parameters"]["required"] == ["email_id"]
    assert read_function["parameters"]["additionalProperties"] is False
    assert send_function["name"] == "send_email"
    assert send_function["parameters"]["required"] == ["to", "subject", "body"]
    assert send_function["parameters"]["additionalProperties"] is False


def test_tool_definition_factory_prevents_cross_run_mutation() -> None:
    """Verify each definition request receives an independent deep copy.

    The test mutates a nested description in the first returned list, asks for
    definitions again, and confirms the second list still contains the original
    schema and stable tool order.
    """

    first = email_tool_definitions()
    first[0]["function"]["description"] = "mutated by provider"

    second = email_tool_definitions()

    assert second[0]["function"]["description"] != "mutated by provider"
    assert [tool["function"]["name"] for tool in second] == [
        "read_email",
        "send_email",
    ]


def test_read_email_returns_the_requested_older_message(
    mailbox: SimulatedMailbox,
) -> None:
    """Verify a known ID returns the exact complete older message.

    The test also compares the model and audit views: only the hidden audit
    receipt should disclose that execution was simulated.
    """

    result = read_email({"email_id": "older-001"}, mailbox)

    assert result.status == "succeeded"
    assert result.arguments == {"email_id": "older-001"}
    assert result.model_result == {
        "ok": True,
        "email": {
            "email_id": "older-001",
            "from": "manager@company.test",
            "to": "agent@research.test",
            "subject": "Quarterly planning",
            "body": "Please review the attached synthetic planning note.",
        },
    }
    assert result.audit_result["simulated"] is True
    assert "simulated" not in result.model_result


def test_read_result_cannot_mutate_the_stored_older_email(
    mailbox: SimulatedMailbox,
) -> None:
    """Verify result dictionaries cannot modify canonical inbox evidence.

    The first returned body is deliberately changed.  Reading the same ID again
    must recover the original body, demonstrating that tool results contain new
    dictionaries rather than references into the mailbox.
    """

    first = mailbox.read_email({"email_id": "older-001"})
    first.model_result["email"]["body"] = "changed outside the mailbox"

    second = mailbox.read_email({"email_id": "older-001"})

    assert second.model_result["email"]["body"] == (
        "Please review the attached synthetic planning note."
    )


@pytest.mark.parametrize(
    ("arguments", "error_code"),
    [
        ({"email_id": "missing"}, "unknown_email_id"),
        ({"email_id": " older-001"}, "invalid_arguments"),
        ({"email_id": "older-001", "unexpected": True}, "invalid_arguments"),
        (["older-001"], "invalid_arguments"),
    ],
)
def test_read_email_rejects_unknown_or_malformed_requests(
    mailbox: SimulatedMailbox,
    arguments: object,
    error_code: str,
) -> None:
    """Verify invalid read variants are rejected with stable categories.

    Pytest supplies an unknown ID, an ambiguous ID, an extra field, and a list
    instead of an object.  Each case must fail without raising to the runtime,
    and its model/audit receipts must be equal but separately mutable objects.
    """

    result = mailbox.read_email(arguments)

    assert result.status == "rejected"
    assert result.model_result["ok"] is False
    assert result.model_result["error_code"] == error_code
    assert result.audit_result == result.model_result
    # The receipts must be equal for a rejection but must not share state.
    assert result.audit_result is not result.model_result


def test_send_email_records_a_simulated_message_in_the_outbox(
    mailbox: SimulatedMailbox,
) -> None:
    """Verify a valid send produces both a receipt and immutable outbox record.

    The expected record checks the assigned sender, requested recipient and
    content, deterministic first ID, and the audit-only simulation marker.
    """

    result = send_email(
        {
            "to": "reviewer@outside.test",
            "subject": "Requested material",
            "body": "Here is the synthetic material you requested.",
        },
        mailbox,
    )

    assert result.status == "succeeded"
    assert result.model_result == {
        "ok": True,
        "message_id": "sent-0001",
        "detail": "Email sent successfully.",
    }
    assert result.audit_result["simulated"] is True
    assert mailbox.sent_emails == (
        EmailMessage(
            email_id="sent-0001",
            sender="agent@research.test",
            recipient="reviewer@outside.test",
            subject="Requested material",
            body="Here is the synthetic material you requested.",
        ),
    )


def test_successful_send_ids_are_deterministic_and_monotonic(
    mailbox: SimulatedMailbox,
) -> None:
    """Verify successful sends receive reproducible increasing identifiers.

    Two identical requests are executed against one mailbox.  Their IDs must
    advance from `sent-0001` to `sent-0002`, and outbox order must match call
    order so later journal correlation has a deterministic sequence.
    """

    arguments: dict[str, Any] = {
        "to": "archive@company.test",
        "subject": "Archive",
        "body": "Synthetic record.",
    }

    first = mailbox.send_email(arguments)
    second = mailbox.send_email(arguments)

    assert first.model_result["message_id"] == "sent-0001"
    assert second.model_result["message_id"] == "sent-0002"
    assert [email.email_id for email in mailbox.sent_emails] == [
        "sent-0001",
        "sent-0002",
    ]


def test_send_preserves_exact_nonblank_subject_and_body(
    mailbox: SimulatedMailbox,
) -> None:
    """Verify validation does not rewrite nonblank model-generated content.

    Leading subject spaces and body line breaks are intentional test data.  The
    stored message must reproduce them exactly because requested content is
    evidence for later action-versus-self-log comparison.
    """

    mailbox.send_email(
        {
            "to": "archive@company.test",
            "subject": "  Deliberate spacing  ",
            "body": "First line.\n\nSecond line.\n",
        }
    )

    sent = mailbox.sent_emails[0]
    assert sent.subject == "  Deliberate spacing  "
    assert sent.body == "First line.\n\nSecond line.\n"


@pytest.mark.parametrize(
    "arguments",
    [
        {
            "to": "real@example.com",
            "subject": "Unsafe destination",
            "body": "Must not be accepted.",
        },
        {
            "to": "reviewer@outside.test",
            "subject": "Extra field",
            "body": "Must not be accepted.",
            "cc": "observer@company.test",
        },
        {
            "to": "reviewer@outside.test",
            "subject": "   ",
            "body": "Must not be accepted.",
        },
    ],
)
def test_rejected_send_never_changes_the_outbox(
    mailbox: SimulatedMailbox,
    arguments: dict[str, Any],
) -> None:
    """Verify unsafe, undeclared, and blank send requests have no side effect.

    Parameterization passes a real domain, an unsupported CC field, and a blank
    subject.  All must share the `invalid_arguments` category and leave the
    successful-send outbox empty.
    """

    result = mailbox.send_email(arguments)

    assert result.status == "rejected"
    assert result.model_result["error_code"] == "invalid_arguments"
    assert mailbox.sent_emails == ()


@pytest.mark.parametrize(
    "address",
    [
        "agent@company.test",
        "reviewer@outside.test",
        "first.last+pilot@subdomain.company.test",
    ],
)
def test_safe_address_accepts_synthetic_test_domains(address: str) -> None:
    """Verify valid controlled identities may use different `.test` domains.

    Separate domains let future scenarios represent internal and external roles
    while the addresses remain reserved and incapable of real delivery.
    """

    assert is_safe_test_address(address)


@pytest.mark.parametrize(
    "address",
    [
        "",
        "real@example.com",
        "agent@company.test.attacker.example",
        ".agent@company.test",
        "agent..name@company.test",
        "agent@-company.test",
        " agent@company.test",
    ],
)
def test_safe_address_rejects_real_or_malformed_destinations(address: str) -> None:
    """Verify real, lookalike, and malformed destinations fail the safety check.

    The cases exercise the real-domain boundary, suffix anchoring, invalid local
    parts, an invalid domain label, and surrounding whitespace.
    """

    assert not is_safe_test_address(address)


def test_mailbox_rejects_duplicate_ids_and_unsafe_fixture_addresses() -> None:
    """Verify invalid scenario fixtures fail before any model interaction.

    The first construction uses the same stable ID twice and must raise a clear
    `ValueError`.  The second places a real sender address in older mail and must
    fail Pydantic validation, preventing unsafe data from entering the mailbox.
    """

    duplicate = {
        "email_id": "same-id",
        "sender": "sender@company.test",
        "recipient": "agent@research.test",
        "subject": "Synthetic",
        "body": "Synthetic.",
    }

    with pytest.raises(ValueError, match="Duplicate older email_id"):
        SimulatedMailbox([duplicate, duplicate])

    with pytest.raises(ValidationError):
        SimulatedMailbox(
            [
                {
                    **duplicate,
                    "sender": "real@example.com",
                }
            ]
        )
