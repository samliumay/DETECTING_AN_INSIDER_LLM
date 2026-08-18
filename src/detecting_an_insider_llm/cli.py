"""Command-line interface for the insider-LLM detection harness.

The `chat` command is intentionally a small interactive integration test.  It
creates one provider, one `Agent`, and one simulated mailbox, then reuses them
until the user enters `/quit`.  Reusing the same objects preserves conversation
history and simulated sent mail across turns.

The command now executes only the allowlisted email tools through a bounded
loop.  It still does not write research artifacts; the future experiment runner
must journal every attempt, failure, prompt, provider response, and run status.
"""

import argparse
import json
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from detecting_an_insider_llm.providers import OllamaClient, OllamaClientError
from detecting_an_insider_llm.runtime import (
    Agent,
    ToolCallExecution,
    ToolLoopProviderError,
)
from detecting_an_insider_llm.runtime.agents import ThinkingMode
from detecting_an_insider_llm.tools import (
    EmailToolDispatcher,
    SimulatedMailbox,
    email_tool_definitions,
)

PROGRAM_NAME = "detecting-an-insider-llm"
PROGRAM_VERSION = "0.1.0"
DEFAULT_MAX_TOOL_ROUNDS = 8

InputReader = Callable[[str], str]
LineWriter = Callable[[str], None]
ClientFactory = Callable[[argparse.Namespace], OllamaClient]


def build_parser() -> argparse.ArgumentParser:
    """Create the top-level parser and its subcommands."""
    parser = argparse.ArgumentParser(
        prog=PROGRAM_NAME,
        description="Run and analyze double-logging experiments for LLM agents.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {PROGRAM_VERSION}",
    )

    subparsers = parser.add_subparsers(dest="command")
    chat_parser = subparsers.add_parser(
        "chat",
        help="start a provider-backed interactive conversation",
        description=(
            "Talk to one model until /quit. Conversation history is preserved, "
            "and allowlisted email tools execute only inside a simulated mailbox."
        ),
    )
    chat_parser.add_argument(
        "--provider",
        choices=("ollama",),
        default="ollama",
        help="provider adapter to use (default: ollama)",
    )
    chat_parser.add_argument(
        "--model",
        help="Ollama model identifier; defaults to OLLAMA_MODEL",
    )
    chat_parser.add_argument(
        "--base-url",
        help="Ollama server root; defaults to OLLAMA_BASE_URL or localhost",
    )
    chat_parser.add_argument(
        "--timeout-seconds",
        type=_positive_float,
        help="positive HTTP timeout; defaults to OLLAMA_TIMEOUT_SECONDS",
    )
    chat_parser.add_argument(
        "--keep-alive",
        help="Ollama model residency duration, for example 5m or 0",
    )
    chat_parser.add_argument(
        "--system-prompt",
        help="optional instruction retained for the entire conversation",
    )
    chat_parser.add_argument(
        "--temperature",
        type=_non_negative_float,
        help="non-negative Ollama temperature",
    )
    chat_parser.add_argument(
        "--top-k",
        type=_non_negative_int,
        help="non-negative Ollama top_k value",
    )
    chat_parser.add_argument(
        "--top-p",
        type=_probability,
        help="Ollama top_p value between 0 and 1",
    )
    chat_parser.add_argument(
        "--seed",
        type=int,
        help="optional deterministic sampling seed",
    )
    chat_parser.add_argument(
        "--max-output-tokens",
        type=_positive_int,
        help="positive Ollama num_predict limit",
    )
    chat_parser.add_argument(
        "--max-tool-rounds",
        type=_positive_int,
        help=(
            "positive tool-call round limit; defaults to "
            "OLLAMA_MAX_TOOL_ROUNDS or 8"
        ),
    )
    chat_parser.add_argument(
        "--mailbox-file",
        type=Path,
        help="optional JSON list of synthetic older emails available to read_email",
    )
    chat_parser.add_argument(
        "--think",
        choices=("false", "true", "low", "medium", "high"),
        default="false",
        help="request exposed reasoning when supported (default: false)",
    )
    chat_parser.set_defaults(handler=run_chat)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, dispatch the selected command, and return an exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        # Showing help is friendlier than silently doing nothing when the command
        # is launched without a subcommand.
        parser.print_help()
        return 0

    try:
        return int(handler(args))
    except (OllamaClientError, ValueError) as exc:
        # Configuration and startup errors are concise CLI failures. Per-turn
        # provider failures are handled inside the interactive loop instead.
        print(f"{PROGRAM_NAME}: error: {exc}", file=sys.stderr)
        return 1


def run_chat(
    args: argparse.Namespace,
    *,
    input_reader: InputReader = input,
    write_line: LineWriter = print,
    client_factory: ClientFactory | None = None,
) -> int:
    """Run a bounded tool-enabled conversation until the user requests exit.

    Args:
        args: Parsed CLI configuration produced by :func:`build_parser`.
        input_reader: Injectable terminal input function used by offline tests.
        write_line: Injectable one-line output function used by offline tests.
        client_factory: Optional provider constructor; production uses Ollama.

    Returns:
        Process-style exit code: zero for `/quit` or EOF and 130 for Ctrl-C.

    One agent, dispatcher, and mailbox are reused for the whole session.  Each
    user message may therefore require several provider calls, but the number of
    executable tool-call batches is bounded.  The injectable boundaries keep
    tests deterministic and offline.
    """
    factory = client_factory or _create_ollama_client
    options = _ollama_options(args)
    think = _thinking_mode(args.think)
    max_tool_rounds = _configured_max_tool_rounds(args.max_tool_rounds)
    mailbox = _load_mailbox(args.mailbox_file)
    dispatcher = EmailToolDispatcher(mailbox)

    # Construct the provider once. Recreating it on every prompt would discard
    # connection reuse and make runtime metadata less stable within a session.
    with factory(args) as provider:
        agent = Agent(
            provider,
            system_prompt=args.system_prompt,
            tools=email_tool_definitions(),
            options=options,
            think=think,
        )
        write_line(
            f"Interactive session with {agent.provider_name}/{agent.model_name}. "
            "Type /quit to exit."
        )
        if mailbox.older_email_ids:
            # IDs are shown to the human operator, not silently injected into
            # the model prompt.  A future scenario determines what the model is
            # legitimately told about available messages.
            write_line(
                "Synthetic older email IDs: " + ", ".join(mailbox.older_email_ids)
            )

        while True:
            try:
                user_message = input_reader("you> ")
            except EOFError:
                # Ctrl-D or a closed input pipe is a normal end of an interactive
                # session and should not be reported as a model/provider failure.
                write_line("")
                return 0
            except KeyboardInterrupt:
                write_line("\nSession interrupted.")
                return 130

            command = user_message.strip()
            if command.casefold() in {"/quit", "/exit"}:
                return 0
            if not command:
                continue

            try:
                turn = agent.run_with_tools(
                    user_message,
                    executor=dispatcher,
                    max_tool_rounds=max_tool_rounds,
                )
            except ToolLoopProviderError as exc:
                # Completed attempts cannot be rolled back after a later
                # provider failure.  Display the partial trace before allowing
                # the human operator to continue the same session.
                _write_tool_executions(exc.tool_executions, write_line)
                write_line(f"error: {exc}")
                continue
            except OllamaClientError as exc:
                # A first-call failure has no tool side effect, so Agent leaves
                # that user message outside committed conversation history.
                write_line(f"error: {exc}")
                continue

            _write_tool_executions(turn.tool_executions, write_line)

            content = turn.last_response.message.get("content")
            if isinstance(content, str) and content:
                write_line(f"assistant> {content}")

            if turn.termination_reason == "max_tool_rounds":
                write_line(
                    f"error: maximum tool-call rounds reached ({max_tool_rounds})"
                )


def _create_ollama_client(args: argparse.Namespace) -> OllamaClient:
    """Build the selected provider from parsed CLI/environment configuration."""
    if args.provider != "ollama":
        # argparse currently prevents this path. Keeping the boundary explicit
        # makes a future provider addition local to parser choices and this
        # construction function.
        raise ValueError(f"Unsupported provider: {args.provider}")

    return OllamaClient(
        model=args.model,
        base_url=args.base_url,
        timeout_seconds=args.timeout_seconds,
        keep_alive=args.keep_alive,
    )


def _configured_max_tool_rounds(explicit_value: int | None) -> int:
    """Resolve and validate the bounded tool-loop setting.

    Args:
        explicit_value: Value parsed from `--max-tool-rounds`, if supplied.

    Returns:
        A positive integer using CLI-first, environment-second, default-last
        precedence.

    Raises:
        ValueError: If `OLLAMA_MAX_TOOL_ROUNDS` is not a positive integer.

    Argparse validates explicit input.  Environment input needs separate
    validation because it enters the process as an arbitrary string.
    """

    if explicit_value is not None:
        return explicit_value
    environment_value = os.getenv("OLLAMA_MAX_TOOL_ROUNDS", "").strip()
    if not environment_value:
        return DEFAULT_MAX_TOOL_ROUNDS
    try:
        parsed_value = int(environment_value)
    except ValueError as exc:
        raise ValueError("OLLAMA_MAX_TOOL_ROUNDS must be a positive integer.") from exc
    if parsed_value < 1:
        raise ValueError("OLLAMA_MAX_TOOL_ROUNDS must be greater than zero.")
    return parsed_value


def _load_mailbox(mailbox_file: Path | None) -> SimulatedMailbox:
    """Load optional synthetic older mail into a validated in-memory mailbox.

    Args:
        mailbox_file: JSON path supplied with `--mailbox-file`, or `None` for an
            empty inbox.

    Returns:
        A new :class:`SimulatedMailbox`.  The JSON root must be a list whose
        items match the documented `EmailMessage` fields.

    Raises:
        ValueError: If the file cannot be read, decoded, or validated.

    Loading occurs once before provider construction.  An invalid experiment
    fixture therefore fails before any model call or simulated action occurs.
    """

    if mailbox_file is None:
        return SimulatedMailbox()
    try:
        raw_text = mailbox_file.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Could not read mailbox file {mailbox_file}: {exc}") from exc
    try:
        raw_emails = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Mailbox file {mailbox_file} is not valid JSON.") from exc
    if not isinstance(raw_emails, list):
        raise ValueError(f"Mailbox file {mailbox_file} must contain a JSON list.")
    try:
        return SimulatedMailbox(raw_emails)
    except ValueError as exc:
        raise ValueError(f"Mailbox file {mailbox_file} is invalid: {exc}") from exc


def _write_tool_executions(
    executions: Sequence[ToolCallExecution],
    write_line: LineWriter,
) -> None:
    """Display concise statuses without dumping email bodies or audit fields.

    Args:
        executions: Structured attempts returned by the bounded loop.
        write_line: Output boundary selected by the interactive caller.

    Successful sends include their deterministic message ID.  Rejections show a
    stable error code.  Full arguments and receipts stay in memory for later
    journaling rather than being printed to a terminal where sensitive synthetic
    scenario text could be copied accidentally.
    """

    for execution in executions:
        result = execution.result
        suffix = ""
        message_id = result.model_result.get("message_id")
        error_code = result.model_result.get("error_code")
        if isinstance(message_id, str):
            suffix = f" ({message_id})"
        elif isinstance(error_code, str):
            suffix = f" ({error_code})"
        write_line(
            f"tool> {execution.call.requested_tool_name}: {result.status}{suffix}"
        )


def _ollama_options(args: argparse.Namespace) -> dict[str, Any] | None:
    """Translate provider-neutral CLI flags into Ollama option names."""
    values = {
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "seed": args.seed,
        "num_predict": args.max_output_tokens,
    }
    options = {name: value for name, value in values.items() if value is not None}
    return options or None


def _thinking_mode(value: str) -> ThinkingMode:
    """Convert argparse's string choice into the shared runtime type."""
    if value == "true":
        return True
    if value == "false":
        return False
    if value in {"low", "medium", "high"}:
        return value
    raise ValueError(f"Unsupported thinking mode: {value}")


def _positive_float(value: str) -> float:
    """Parse a strictly positive floating-point CLI argument."""
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def _non_negative_float(value: str) -> float:
    """Parse a non-negative floating-point CLI argument."""
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be zero or greater")
    return parsed


def _probability(value: str) -> float:
    """Parse a probability-like value in the closed interval [0, 1]."""
    parsed = float(value)
    if not 0 <= parsed <= 1:
        raise argparse.ArgumentTypeError("value must be between 0 and 1")
    return parsed


def _positive_int(value: str) -> int:
    """Parse a strictly positive integer CLI argument."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def _non_negative_int(value: str) -> int:
    """Parse a non-negative integer CLI argument."""
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be zero or greater")
    return parsed
