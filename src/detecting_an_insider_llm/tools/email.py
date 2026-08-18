"""Deterministic implementations of the controlled email tools.

No method in this module opens a network connection or writes to disk.  Older
messages are supplied by the caller and successful sends are appended only to
an in-memory outbox.  This boundary lets the experiment observe a model's tool
choices without delivering mail or creating effects outside the harness.
"""

import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


# `.test` is reserved for testing.  Requiring that suffix provides a second
# safety boundary in addition to the absence of SMTP or HTTP code.  Different
# subdomains (for example, `company.test` and `outside.test`) can still represent
# internal and external actors in a scenario.
_SAFE_EMAIL_PATTERN = re.compile(
    r"[A-Z0-9!#$%&'*+/=?^_`{|}~-]+"
    r"(?:\.[A-Z0-9!#$%&'*+/=?^_`{|}~-]+)*"
    r"@(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+TEST",
    re.IGNORECASE,
)


class ReadEmailArguments(BaseModel):
    """Describe and validate the single argument accepted by `read_email`.

    Pydantic creates this object from untrusted model-generated JSON.  The
    field length and validator below reject malformed identifiers before the
    mailbox performs a lookup.
    """

    # A model must not smuggle an undeclared option past the JSON schema when a
    # provider performs incomplete schema validation.
    model_config = ConfigDict(extra="forbid")

    email_id: str = Field(min_length=1, max_length=128)

    @field_validator("email_id")
    @classmethod
    def email_id_must_not_have_edge_whitespace(cls, email_id: str) -> str:
        """Return an exact identifier or reject surrounding whitespace.

        Calling `strip()` only for comparison detects ambiguous input without
        changing it.  This matters because the journal should retain the exact
        tool argument produced by the model instead of a cleaned replacement.
        """

        if email_id != email_id.strip():
            raise ValueError("email_id must not start or end with whitespace")
        return email_id


class SendEmailArguments(BaseModel):
    """Describe and validate the three arguments accepted by `send_email`.

    Extra fields are forbidden so a provider cannot introduce undeclared
    behavior such as CC, attachments, or delivery options.  Field limits bound
    the amount of model-generated data held by one simulated action.
    """

    model_config = ConfigDict(extra="forbid")

    to: str = Field(min_length=3, max_length=254)
    subject: str = Field(min_length=1, max_length=998)
    body: str = Field(min_length=1, max_length=100_000)

    @field_validator("to")
    @classmethod
    def recipient_must_be_reserved(cls, recipient: str) -> str:
        """Return a safe synthetic recipient or reject the arguments.

        The helper checks both basic address structure and the reserved `.test`
        suffix.  Therefore even a later integration mistake cannot place a real
        destination in this mailbox's successful outbox.
        """

        if not is_safe_test_address(recipient):
            raise ValueError("recipient must be a valid address in a .test domain")
        return recipient

    @field_validator("subject", "body")
    @classmethod
    def content_must_not_be_blank(cls, content: str) -> str:
        """Return nonblank content without normalizing its exact characters.

        `strip()` is used only to decide whether any visible content exists.
        Returning the original string preserves leading spaces, line breaks,
        and trailing whitespace as part of the model's observable action.
        """

        if not content.strip():
            raise ValueError("email content must contain non-whitespace text")
        return content


class EmailMessage(BaseModel):
    """One immutable message stored in the synthetic mailbox.

    `sender` and `recipient` use Python-friendly names internally.  The
    :meth:`as_tool_record` method translates them to the familiar `from` and
    `to` keys shown to the model.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    email_id: str = Field(min_length=1, max_length=128)
    sender: str = Field(min_length=3, max_length=254)
    recipient: str = Field(min_length=3, max_length=254)
    subject: str = Field(min_length=1, max_length=998)
    body: str = Field(min_length=1, max_length=100_000)

    @field_validator("sender", "recipient")
    @classmethod
    def addresses_must_be_reserved(cls, address: str) -> str:
        """Return a fixture address only when it belongs to a `.test` domain.

        Validating both sides of stored mail prevents scenario authors from
        accidentally adding a real identity to an older or newly sent message.
        """

        if not is_safe_test_address(address):
            raise ValueError("mailbox addresses must use a valid .test domain")
        return address

    @field_validator("email_id")
    @classmethod
    def stored_email_id_must_be_unambiguous(cls, email_id: str) -> str:
        """Return an exact fixture ID or reject surrounding whitespace.

        Later action records will match reads by this identifier.  Rejecting an
        ambiguous fixture at construction time is safer than allowing two
        textual forms to appear to identify the same email.
        """

        if email_id != email_id.strip():
            raise ValueError("email_id must not start or end with whitespace")
        return email_id

    @field_validator("subject", "body")
    @classmethod
    def stored_content_must_not_be_blank(cls, content: str) -> str:
        """Return exact nonblank fixture content.

        As with model-generated content, whitespace is examined to detect an
        empty value but is not removed.  Raw scenario wording therefore remains
        reproducible when the message is returned later.
        """

        if not content.strip():
            raise ValueError("email content must contain non-whitespace text")
        return content

    def as_tool_record(self) -> dict[str, str]:
        """Convert internal field names into the model-facing email object.

        Returns:
            A new dictionary containing `email_id`, `from`, `to`, `subject`,
            and `body`.  A new object is created each time so callers cannot
            mutate the immutable canonical message through the returned value.
        """

        return {
            "email_id": self.email_id,
            "from": self.sender,
            "to": self.recipient,
            "subject": self.subject,
            "body": self.body,
        }


@dataclass(frozen=True, slots=True)
class ToolExecutionResult:
    """Structured result of one attempted email-tool execution.

    Attributes:
        tool_name: The exact model-visible function name.
        arguments: Normalized arguments, or the object fields available before
            validation failed.
        status: Whether the simulated action succeeded or was rejected.
        model_result: Receipt that may later be returned to the model.
        audit_result: More complete receipt for the harness.  Successful audit
            receipts explicitly retain the fact that execution was simulated.

    The two receipts are independent dictionaries.  Code may safely add a
    future `log_id` to one without accidentally changing the other.
    """

    tool_name: Literal["read_email", "send_email"]
    arguments: dict[str, Any]
    status: Literal["succeeded", "rejected"]
    model_result: dict[str, Any]
    audit_result: dict[str, Any]


class SimulatedMailbox:
    """Own immutable older mail and an in-memory record of simulated sends.

    Args:
        older_emails: Scenario-provided messages that `read_email` may access.
        sender_address: Synthetic address assigned to the model-controlled agent.

    The class deliberately has no generic `execute(name, arguments)` dispatcher.
    Tool allowlisting, attempt logging, and malformed-call preservation belong
    to the later runtime integration requested as a separate step.
    """

    def __init__(
        self,
        older_emails: Sequence[EmailMessage | Mapping[str, Any]] = (),
        *,
        sender_address: str = "agent@research.test",
    ) -> None:
        """Create validated inbox state and an initially empty outbox.

        Each supplied older message is converted to an immutable
        :class:`EmailMessage` and indexed by `email_id` for deterministic reads.
        Duplicate IDs are rejected because they would make a model's read
        request ambiguous.  The send counter begins at one for every mailbox,
        making independent test and experiment runs reproducible.

        Raises:
            ValueError: If the sender is unsafe or an email ID is duplicated.
            pydantic.ValidationError: If an older-email fixture is invalid.
        """

        if not is_safe_test_address(sender_address):
            raise ValueError("sender_address must be a valid address in a .test domain.")

        indexed_emails: dict[str, EmailMessage] = {}
        for raw_email in older_emails:
            email = EmailMessage.model_validate(raw_email)
            if email.email_id in indexed_emails:
                raise ValueError(f"Duplicate older email_id: {email.email_id}.")
            indexed_emails[email.email_id] = email

        self._older_emails = indexed_emails
        self._sender_address = sender_address
        self._sent_emails: list[EmailMessage] = []
        self._next_sent_sequence = 1

    @property
    def older_email_ids(self) -> tuple[str, ...]:
        """Return readable older-email IDs in their supplied order.

        A tuple is returned instead of the internal dictionary so callers can
        discover available IDs without adding, removing, or replacing messages.
        """

        return tuple(self._older_emails)

    @property
    def sent_emails(self) -> tuple[EmailMessage, ...]:
        """Return successful simulated sends in execution order.

        The tuple is a read-only view of frozen :class:`EmailMessage` objects.
        Rejected attempts are intentionally absent; the future runtime journal,
        rather than this successful outbox, will preserve all attempts.
        """

        # EmailMessage is frozen, and a tuple prevents append/remove operations.
        # Callers can inspect evidence without mutating the mailbox's outbox.
        return tuple(self._sent_emails)

    def read_email(self, raw_arguments: object) -> ToolExecutionResult:
        """Execute one controlled `read_email` attempt.

        Args:
            raw_arguments: Untrusted JSON-like arguments produced by a model.

        Returns:
            A rejected result when validation fails or the ID is unknown.
            Otherwise, returns the complete stored message in the model receipt
            and a separate audit receipt marked as simulated.

        Validation runs before lookup.  Successful output is built from a fresh
        dictionary so provider-side mutation cannot change canonical inbox data.
        """

        try:
            request = ReadEmailArguments.model_validate(raw_arguments)
        except ValidationError as exc:
            return _rejected_result(
                tool_name="read_email",
                arguments=_object_fields(raw_arguments),
                message="Invalid read_email arguments.",
                error_code="invalid_arguments",
                validation_error=exc,
            )

        arguments = request.model_dump()
        email = self._older_emails.get(request.email_id)
        if email is None:
            return _rejected_result(
                tool_name="read_email",
                arguments=arguments,
                message=f"Unknown email_id: {request.email_id}.",
                error_code="unknown_email_id",
            )

        # Construct separate nested records so a future provider serializer
        # cannot alter the mailbox or the corresponding audit receipt.
        model_result: dict[str, Any] = {
            "ok": True,
            "email": email.as_tool_record(),
        }
        audit_result = deepcopy(model_result)
        audit_result["simulated"] = True
        return ToolExecutionResult(
            tool_name="read_email",
            arguments=arguments,
            status="succeeded",
            model_result=model_result,
            audit_result=audit_result,
        )

    def send_email(self, raw_arguments: object) -> ToolExecutionResult:
        """Execute one controlled `send_email` attempt entirely in memory.

        Args:
            raw_arguments: Untrusted JSON-like arguments produced by a model.

        Returns:
            A rejected result when arguments are invalid.  On success, returns
            a normal-looking model receipt and a richer audit receipt containing
            the stored message and `simulated=True`.

        A successful call creates the next deterministic message ID and appends
        an immutable message to the outbox.  No transport library is called, so
        the apparent send never leaves this Python object.
        """

        try:
            request = SendEmailArguments.model_validate(raw_arguments)
        except ValidationError as exc:
            return _rejected_result(
                tool_name="send_email",
                arguments=_object_fields(raw_arguments),
                message="Invalid send_email arguments.",
                error_code="invalid_arguments",
                validation_error=exc,
            )

        # IDs advance only for successful sends.  Rejected attempts remain the
        # runtime journal's responsibility and cannot create gaps in the outbox.
        message_id = f"sent-{self._next_sent_sequence:04d}"
        email = EmailMessage(
            email_id=message_id,
            sender=self._sender_address,
            recipient=request.to,
            subject=request.subject,
            body=request.body,
        )
        self._sent_emails.append(email)
        self._next_sent_sequence += 1

        model_result: dict[str, Any] = {
            "ok": True,
            "message_id": message_id,
            "detail": "Email sent successfully.",
        }
        audit_result = deepcopy(model_result)
        audit_result.update(
            {
                "simulated": True,
                "email": email.as_tool_record(),
            }
        )
        return ToolExecutionResult(
            tool_name="send_email",
            arguments=request.model_dump(),
            status="succeeded",
            model_result=model_result,
            audit_result=audit_result,
        )


def read_email(
    raw_arguments: object,
    mailbox: SimulatedMailbox,
) -> ToolExecutionResult:
    """Delegate a read request to explicitly supplied mailbox state.

    Args:
        raw_arguments: Untrusted model arguments to validate.
        mailbox: The scenario-specific in-memory mailbox to read.

    Returns:
        The same structured result produced by `mailbox.read_email`.

    Keeping state explicit avoids a global inbox that could leak messages or
    sent-mail counters between independent experiment runs.
    """

    return mailbox.read_email(raw_arguments)


def send_email(
    raw_arguments: object,
    mailbox: SimulatedMailbox,
) -> ToolExecutionResult:
    """Delegate a simulated send to explicitly supplied mailbox state.

    Args:
        raw_arguments: Untrusted model arguments to validate.
        mailbox: The scenario-specific in-memory mailbox that owns the outbox.

    Returns:
        The same structured result produced by `mailbox.send_email`.

    The wrapper gives a future allowlisted dispatcher a simple function entry
    point without creating mailbox state or integrating with an agent today.
    """

    return mailbox.send_email(raw_arguments)


def is_safe_test_address(address: str) -> bool:
    """Check whether an address is safe for the controlled simulation.

    Args:
        address: Complete address to inspect; callers pass an already validated
            string rather than an arbitrary model value.

    Returns:
        `True` only for a structurally valid address ending in a `.test` domain.

    The function first applies whole-address, local-part, and domain length
    limits.  It then uses a full regular-expression match, so trailing text and
    lookalike domains cannot pass.  This is intentionally narrower than a full
    RFC parser: research fixtures need predictable synthetic identities rather
    than support for every legal real-world address form.
    """

    if len(address) > 254:
        return False
    local_part, separator, domain = address.rpartition("@")
    if separator != "@" or len(local_part) > 64 or len(domain) > 253:
        return False
    return _SAFE_EMAIL_PATTERN.fullmatch(address) is not None


def _rejected_result(
    *,
    tool_name: Literal["read_email", "send_email"],
    arguments: dict[str, Any],
    message: str,
    error_code: str,
    validation_error: ValidationError | None = None,
) -> ToolExecutionResult:
    """Create the standard result for a rejected email-tool attempt.

    Args:
        tool_name: Which allowlisted email function was attempted.
        arguments: Object fields available from the attempted call.
        message: Human-readable rejection reason.
        error_code: Stable machine-readable rejection category.
        validation_error: Optional Pydantic details for malformed arguments.

    Returns:
        A rejected :class:`ToolExecutionResult` with equal but independently
        copied model and audit receipts.

    Pydantic URLs, raw inputs, and exception context are omitted to keep the
    receipt deterministic, JSON-safe, and less likely to echo sensitive text.
    """

    result: dict[str, Any] = {
        "ok": False,
        "error": message,
        "error_code": error_code,
    }
    if validation_error is not None:
        # Excluding raw inputs and Python exception context keeps this receipt
        # deterministic, JSON-safe, and less likely to echo sensitive content.
        result["details"] = validation_error.errors(
            include_url=False,
            include_context=False,
            include_input=False,
        )
    return ToolExecutionResult(
        tool_name=tool_name,
        arguments=arguments,
        status="rejected",
        model_result=deepcopy(result),
        audit_result=deepcopy(result),
    )


def _object_fields(raw_arguments: object) -> dict[str, Any]:
    """Preserve safe object fields from an invalid argument payload.

    Args:
        raw_arguments: Any JSON-like value received from a provider response.

    Returns:
        A deep copy when the value is mapping-like, otherwise an empty dict.

    Non-object shapes such as lists remain invalid and are not coerced into a
    misleading argument object.  The future journal can separately retain the
    untouched provider response as raw evidence.
    """

    if not isinstance(raw_arguments, Mapping):
        return {}
    return deepcopy(dict(raw_arguments))
