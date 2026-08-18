"""Allowlisted dispatch for the controlled email-tool implementations.

This module is the only place that translates a model-selected function name
into executable Python behavior.  It uses explicit comparisons rather than
dynamic attribute lookup, imports, or `eval`, so a model can request only
`read_email` or `send_email`.  Every other name becomes a structured rejection.
"""

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from detecting_an_insider_llm.runtime.tool_loop import ToolExecutionResult
from detecting_an_insider_llm.tools.email import SimulatedMailbox


class EmailToolDispatcher:
    """Execute only the two email functions against one supplied mailbox.

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
