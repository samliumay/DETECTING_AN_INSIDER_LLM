"""Provider-neutral parsing and records for a bounded model/tool loop.

The functions here do not call a model and do not select a concrete tool set.
They translate untrusted provider tool-call objects into structured attempts,
ask an injected allowlisted executor to handle them, and construct deterministic
tool-result messages for the next provider turn.
"""

import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal, Protocol


@dataclass(frozen=True, slots=True)
class ToolExecutionResult:
    """Normalized outcome returned by an allowlisted tool implementation.

    Attributes:
        tool_name: Exact requested name.  Unknown or missing names remain visible
            instead of being forced into an allowlisted value.
        arguments: Validated arguments, or object fields available when
            validation failed.
        status: Whether the controlled implementation succeeded or rejected the
            request.  Runtime failures will later require a separate `failed`
            status when durable automatic logging is added.
        model_result: Receipt that may be returned to the model.
        audit_result: Separate harness receipt that may retain hidden fields such
            as `simulated=True`.

    The result belongs to the runtime contract rather than one email function,
    allowing future controlled tools to implement the same narrow interface.
    """

    tool_name: str
    arguments: dict[str, Any]
    status: Literal["succeeded", "rejected"]
    model_result: dict[str, Any]
    audit_result: dict[str, Any]


class ToolExecutor(Protocol):
    """Structural interface required by the agent's tool loop.

    A concrete executor does not need to inherit from this protocol.  Providing
    a compatible `execute` method is enough, which keeps the agent independent
    of email-specific classes and makes offline test doubles straightforward.
    """

    def execute(self, tool_name: str, raw_arguments: object) -> ToolExecutionResult:
        """Execute or reject one parsed model-requested function call."""
        ...


@dataclass(frozen=True, slots=True)
class ParsedToolCall:
    """Provider-neutral fields extracted from one raw tool-call object.

    Attributes:
        requested_tool_name: Exact requested function name, or `<missing>` when
            the provider object has no usable name.
        raw_arguments: Arguments exactly as exposed by the provider.  A string,
            list, or missing value remains malformed rather than being coerced.
        tool_call_id: Optional provider correlation ID for APIs that supply one.
        raw_tool_call: Defensive copy of the complete untrusted call object.
    """

    requested_tool_name: str
    raw_arguments: object
    tool_call_id: str | None
    raw_tool_call: object


@dataclass(frozen=True, slots=True)
class ToolCallExecution:
    """One observable tool-call attempt and its controlled execution result.

    Attributes:
        sequence: Monotonic attempt number within the current agent episode.
        call: Parsed provider request, including the untouched raw structure.
        result: Normalized arguments, status, model receipt, and audit receipt.

    This object is the in-memory seam that a later journal writer can persist.
    It represents successful, malformed, unknown, and round-limit calls alike.
    """

    sequence: int
    call: ParsedToolCall
    result: ToolExecutionResult


def parse_tool_call(raw_tool_call: object) -> ParsedToolCall:
    """Extract a function name, arguments, and optional ID without coercion.

    Args:
        raw_tool_call: One item from an assistant message's `tool_calls` list.

    Returns:
        A :class:`ParsedToolCall`.  Missing outer/function objects yield the
        `<missing>` name and empty arguments, while a present `arguments` value
        is preserved in its original JSON shape for validation and auditing.

    Only dictionary access occurs here; no model-provided text is imported,
    evaluated, or used for dynamic Python lookup.
    """

    copied_call = deepcopy(raw_tool_call)
    if not isinstance(raw_tool_call, dict):
        return ParsedToolCall(
            requested_tool_name="<missing>",
            raw_arguments={},
            tool_call_id=None,
            raw_tool_call=copied_call,
        )

    raw_call_id = raw_tool_call.get("id")
    tool_call_id = raw_call_id if isinstance(raw_call_id, str) and raw_call_id else None
    function = raw_tool_call.get("function")
    if not isinstance(function, dict):
        return ParsedToolCall(
            requested_tool_name="<missing>",
            raw_arguments={},
            tool_call_id=tool_call_id,
            raw_tool_call=copied_call,
        )

    raw_name = function.get("name")
    requested_name = raw_name if isinstance(raw_name, str) and raw_name else "<missing>"
    return ParsedToolCall(
        requested_tool_name=requested_name,
        raw_arguments=deepcopy(function.get("arguments")),
        tool_call_id=tool_call_id,
        raw_tool_call=copied_call,
    )


def execute_tool_call(
    raw_tool_call: object,
    *,
    sequence: int,
    executor: ToolExecutor,
) -> ToolCallExecution:
    """Parse and execute one tool request through the injected allowlist.

    Args:
        raw_tool_call: Complete provider-supplied call object.
        sequence: Positive attempt number assigned by the agent.
        executor: Controlled dispatcher responsible for name allowlisting and
            argument validation.

    Returns:
        A structured attempt containing both the raw call and execution result.

    Raises:
        ValueError: If `sequence` is not positive.

    Parsing is intentionally separate from dispatch so malformed structures
    remain visible even when the executor must reject them.
    """

    if sequence < 1:
        raise ValueError("sequence must be greater than zero.")
    parsed = parse_tool_call(raw_tool_call)
    result = executor.execute(parsed.requested_tool_name, parsed.raw_arguments)
    return ToolCallExecution(sequence=sequence, call=parsed, result=result)


def reject_tool_call_at_round_limit(
    raw_tool_call: object,
    *,
    sequence: int,
    max_tool_rounds: int,
) -> ToolCallExecution:
    """Record a requested call that cannot execute because the bound was met.

    Args:
        raw_tool_call: Complete provider-supplied call object.
        sequence: Positive attempt number assigned by the agent.
        max_tool_rounds: Configured number of executable tool-call rounds.

    Returns:
        A rejected attempt whose receipts explain the round-limit boundary.

    The raw name and arguments are retained because reaching the limit does not
    erase the fact that the model requested an additional action.
    """

    if sequence < 1:
        raise ValueError("sequence must be greater than zero.")
    parsed = parse_tool_call(raw_tool_call)
    result_body: dict[str, Any] = {
        "ok": False,
        "error": f"Maximum tool-call rounds reached ({max_tool_rounds}).",
        "error_code": "max_tool_rounds",
    }
    normalized_arguments = (
        deepcopy(dict(parsed.raw_arguments))
        if isinstance(parsed.raw_arguments, dict)
        else {}
    )
    result = ToolExecutionResult(
        tool_name=parsed.requested_tool_name,
        arguments=normalized_arguments,
        status="rejected",
        model_result=deepcopy(result_body),
        audit_result=deepcopy(result_body),
    )
    return ToolCallExecution(sequence=sequence, call=parsed, result=result)


def tool_result_message(execution: ToolCallExecution) -> dict[str, Any]:
    """Convert an execution receipt into the next provider-visible message.

    Args:
        execution: Completed or rejected tool-call attempt.

    Returns:
        A new JSON-shaped `role="tool"` message.  Its content is deterministically
        serialized from only the model receipt; audit-only fields are excluded.
        A provider call ID is copied when one was present in the request.

    `tool_name` matches Ollama's chat message shape.  A future provider adapter
    may translate this shared message into its provider-specific representation.
    """

    message: dict[str, Any] = {
        "role": "tool",
        "tool_name": execution.call.requested_tool_name,
        "content": json.dumps(
            execution.result.model_result,
            sort_keys=True,
            separators=(",", ":"),
        ),
    }
    if execution.call.tool_call_id is not None:
        message["tool_call_id"] = execution.call.tool_call_id
    return message
