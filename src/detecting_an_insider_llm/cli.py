"""Command-line interface for the insider-LLM detection harness.

The `chat` command is intentionally a small interactive integration test. It
creates one provider and one `Agent`, then reuses them until the user enters
`/quit`. Reusing the same agent preserves conversation history across turns.

This command does not execute model-requested tools or write research artifacts.
Those behaviors require the future controlled tool dispatcher and run journal.
"""

import argparse
import sys
from collections.abc import Callable, Sequence
from typing import Any

from detecting_an_insider_llm.providers import OllamaClient, OllamaClientError
from detecting_an_insider_llm.runtime import Agent
from detecting_an_insider_llm.runtime.agents import ThinkingMode

PROGRAM_NAME = "detecting-an-insider-llm"
PROGRAM_VERSION = "0.1.0"

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
            "but model-requested tools are not executed."
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
    """Run a stateful interactive conversation until the user requests exit.

    The injectable input, output, and client factory keep tests deterministic and
    offline. Production callers use the built-in defaults.
    """
    factory = client_factory or _create_ollama_client
    options = _ollama_options(args)
    think = _thinking_mode(args.think)

    # Construct the provider once. Recreating it on every prompt would discard
    # connection reuse and make runtime metadata less stable within a session.
    with factory(args) as provider:
        agent = Agent(
            provider,
            system_prompt=args.system_prompt,
            options=options,
            think=think,
        )
        write_line(
            f"Interactive session with {agent.provider_name}/{agent.model_name}. "
            "Type /quit to exit."
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
                response = agent.run(user_message)
            except OllamaClientError as exc:
                # Agent commits state only after successful validation, so the
                # user can retry without a failed turn entering model history.
                write_line(f"error: {exc}")
                continue

            content = response.message.get("content")
            if isinstance(content, str) and content:
                write_line(f"assistant> {content}")

            tool_calls = response.message.get("tool_calls")
            if isinstance(tool_calls, list) and tool_calls:
                # Tool schemas will be added with the controlled dispatcher.
                # Until then, making non-execution visible is safer than silently
                # ignoring a request or invoking arbitrary model-selected code.
                write_line(
                    "assistant requested tool execution; "
                    "this CLI does not execute tools yet."
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
