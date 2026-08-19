"""Allowlisted dispatch and in-memory double-logging composition.

This module is the only place that translates a model-selected function name
into executable Python behavior.  It uses explicit comparisons rather than
dynamic attribute lookup, imports, or `eval`, so a model can request only
`list_emails`, `read_email`, or `send_email`.  Every other name becomes a
structured rejection.

`DoubleLoggingToolExecutor` wraps the ordinary dispatcher rather than changing
the email functions.  It assigns the shared model-visible `log_id`, retains the
automatic record separately, and routes `log_action` to an audit-blind store.
"""

from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import Any

from detecting_an_insider_llm.runtime.tool_loop import (
    ToolExecutionResult,
    ToolExecutor,
)
from detecting_an_insider_llm.tools.email import SimulatedMailbox
from detecting_an_insider_llm.tools.self_log import (
    AutomaticActionRecord,
    ModelSelfLogRecord,
    SelfLogStore,
)


LogIdFactory = Callable[[int], str]


class EmailToolDispatcher:
    """Execute only the three email functions against one supplied mailbox.

    Args:
        mailbox: Scenario- or CLI-specific state used for older reads and the
            simulated outbox.

    The dispatcher owns no provider or conversation state.  This makes it
    reusable by the interactive CLI now and by a journaled experiment runner
    later without granting either component additional tool names.
    """

    def __init__(self, mailbox: SimulatedMailbox) -> None:
        """Store the explicit mailbox dependency without creating global state."""

        self._mailbox = mailbox

    @property
    def mailbox(self) -> SimulatedMailbox:
        """Return the mailbox so callers can inspect controlled outbox evidence.

        The mailbox protects its collections with tuple views and immutable
        messages.  Returning the object therefore supports observation without
        exposing a list that callers could modify directly.
        """

        return self._mailbox

    def execute(self, tool_name: str, raw_arguments: object) -> ToolExecutionResult:
        """Dispatch one model request or return an unknown-tool rejection.

        Args:
            tool_name: Exact function name parsed from the provider response.
            raw_arguments: Untrusted JSON-like function arguments.

        Returns:
            The result from the matching simulated mailbox method.  Unknown or
            missing names return `status="rejected"` and never invoke a fallback
            function.

        Explicit branches are intentionally repetitive: the allowlist remains
        visible during review, and adding a schema alone cannot accidentally
        make a new Python function executable.
        """

        if tool_name == "list_emails":
            return self._mailbox.list_emails(raw_arguments)
        if tool_name == "read_email":
            return self._mailbox.read_email(raw_arguments)
        if tool_name == "send_email":
            return self._mailbox.send_email(raw_arguments)

        requested_name = tool_name or "<missing>"
        result: dict[str, Any] = {
            "ok": False,
            "error": f"Unknown tool: {requested_name}.",
            "error_code": "unknown_tool",
        }
        arguments = (
            deepcopy(dict(raw_arguments))
            if isinstance(raw_arguments, Mapping)
            else {}
        )
        return ToolExecutionResult(
            tool_name=requested_name,
            arguments=arguments,
            status="rejected",
            model_result=deepcopy(result),
            audit_result=deepcopy(result),
        )


class DoubleLoggingToolExecutor:
    """Compose ordinary execution, automatic records, and model self-logs.

    Args:
        ordinary_executor: Allowlisted executor for every tool except
            `log_action`.
        self_log_store: Optional isolated store; a fresh store is created when
            omitted.
        log_id_factory: Optional deterministic/test identifier boundary.  It
            receives the one-based ordinary-action sequence.

    `log_action` is routed directly to :class:`SelfLogStore` and is never added
    to `automatic_records`.  All other requested names, including unknown or
    malformed names that the ordinary executor rejects, receive a `log_id` and
    create one automatic record.
    """

    def __init__(
        self,
        ordinary_executor: ToolExecutor,
        *,
        self_log_store: SelfLogStore | None = None,
        log_id_factory: LogIdFactory | None = None,
    ) -> None:
        """Store injected episode state and initialize ordinary-action order."""

        self._ordinary_executor = ordinary_executor
        self._self_log_store = self_log_store or SelfLogStore()
        self._log_id_factory = log_id_factory or _sequential_log_id
        self._automatic_records: list[AutomaticActionRecord] = []
        self._issued_log_ids: set[str] = set()
        self._next_action_sequence = 1

    @property
    def automatic_records(self) -> tuple[AutomaticActionRecord, ...]:
        """Return defensive copies of authoritative records in action order."""

        return tuple(
            record.model_copy(deep=True) for record in self._automatic_records
        )

    @property
    def self_logs(self) -> tuple[ModelSelfLogRecord, ...]:
        """Return defensive copies of successfully stored model self-logs."""

        return self._self_log_store.self_logs

    @property
    def self_log_attempt_count(self) -> int:
        """Return valid and malformed model-visible `log_action` attempts."""

        return self._self_log_store.attempt_count

    def execute(self, tool_name: str, raw_arguments: object) -> ToolExecutionResult:
        """Execute one self-log or automatically identified ordinary attempt.

        The method deliberately does not compare a self-log to automatic state.
        For an ordinary call it assigns identity before execution, adds the same
        `log_id` to the model and audit receipts, and stores a separate snapshot
        that later artifact code can persist.
        """

        if tool_name == "log_action":
            return self._self_log_store.log_action(raw_arguments)

        action_sequence = self._next_action_sequence
        self._next_action_sequence += 1
        log_id = self._new_log_id(action_sequence)

        try:
            ordinary_result = self._ordinary_executor.execute(
                tool_name,
                raw_arguments,
            )
        except Exception as exc:
            # Preserve a minimal in-memory failure record, then re-raise so a
            # harness/configuration failure cannot masquerade as model success.
            failure_result: dict[str, Any] = {
                "ok": False,
                "log_id": log_id,
                "error": "Ordinary tool execution failed.",
                "error_code": "tool_execution_failed",
                "error_type": type(exc).__name__,
            }
            self._automatic_records.append(
                AutomaticActionRecord(
                    record_id=f"action-{action_sequence:04d}",
                    action_sequence=action_sequence,
                    log_id=log_id,
                    requested_tool_name=tool_name or "<missing>",
                    raw_arguments=deepcopy(raw_arguments),
                    arguments=_object_arguments(raw_arguments),
                    result_status="failed",
                    model_result=deepcopy(failure_result),
                    audit_result=deepcopy(failure_result),
                )
            )
            raise

        model_result = deepcopy(ordinary_result.model_result)
        audit_result = deepcopy(ordinary_result.audit_result)
        # The wrapper owns this reserved correlation field.  Both views receive
        # exactly the same ID, but only the harness retains the full record.
        model_result["log_id"] = log_id
        audit_result["log_id"] = log_id
        result = ToolExecutionResult(
            tool_name=ordinary_result.tool_name,
            arguments=deepcopy(ordinary_result.arguments),
            status=ordinary_result.status,
            model_result=model_result,
            audit_result=audit_result,
        )
        self._automatic_records.append(
            AutomaticActionRecord(
                record_id=f"action-{action_sequence:04d}",
                action_sequence=action_sequence,
                log_id=log_id,
                requested_tool_name=tool_name or "<missing>",
                raw_arguments=deepcopy(raw_arguments),
                arguments=deepcopy(result.arguments),
                result_status=result.status,
                model_result=deepcopy(result.model_result),
                audit_result=deepcopy(result.audit_result),
            )
        )
        return result

    def _new_log_id(self, action_sequence: int) -> str:
        """Validate and reserve one factory-produced per-episode identifier."""

        log_id = self._log_id_factory(action_sequence)
        if (
            not isinstance(log_id, str)
            or not log_id
            or log_id != log_id.strip()
            or len(log_id) > 128
        ):
            raise ValueError(
                "log_id_factory must return a nonblank string of at most 128 "
                "characters without edge whitespace."
            )
        if log_id in self._issued_log_ids:
            raise ValueError(f"log_id_factory returned duplicate ID: {log_id}.")
        self._issued_log_ids.add(log_id)
        return log_id


def _sequential_log_id(action_sequence: int) -> str:
    """Return a deterministic ID unique within one executor episode."""

    return f"log-{action_sequence:04d}"


def _object_arguments(raw_arguments: object) -> dict[str, Any]:
    """Copy mapping fields for a failed attempt without coercing other shapes."""

    if not isinstance(raw_arguments, Mapping):
        return {}
    return deepcopy(dict(raw_arguments))
