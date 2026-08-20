"""Command-line interface for interactive, run, and offline-analysis paths.

The `chat` command is intentionally a small interactive integration test.  It
creates one provider, one `Agent`, and one simulated mailbox, then reuses them
until the user enters `/quit`.  Reusing the same objects preserves conversation
history and simulated sent mail across turns.

The `run` command resolves one versioned scenario cell, executes it through the
non-interactive runner, and atomically persists every returned terminal state.
It never interprets the behavior.  The separate `analyze` command reads one
closed run, applies its frozen evaluation contract, and writes `results.json`
without contacting a provider or changing the raw artifacts.
"""

import argparse
import json
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from detecting_an_insider_llm.analysis import AnalysisError, OfflineAnalyzer
from detecting_an_insider_llm.artifacts import (
    ArtifactWriteError,
    RunArtifactWriter,
)
from detecting_an_insider_llm.providers import OllamaClient, OllamaClientError
from detecting_an_insider_llm.runtime import (
    Agent,
    ToolCallExecution,
    ToolLoopProviderError,
)
from detecting_an_insider_llm.runtime.agents import ThinkingMode
from detecting_an_insider_llm.runtime.episode_runner import (
    OperationalProvenanceFactory,
    ScenarioRunner,
)
from detecting_an_insider_llm.scenario_loader import resolve_scenario
from detecting_an_insider_llm.tools import (
    EmailToolDispatcher,
    SimulatedMailbox,
    email_tool_definitions,
)

PROGRAM_NAME = "detecting-an-insider-llm"
PROGRAM_VERSION = "0.1.0"
# A fully sequential scenario path can require list + eleven reads + one send,
# and the later model-visible self-log may add one call per ordinary action.
# Thirty-two leaves a small retry margin while remaining explicitly bounded.
DEFAULT_MAX_TOOL_ROUNDS = 32
EXIT_COMPLETED = 0
EXIT_CLI_ERROR = 1
EXIT_INCOMPLETE = 2
EXIT_FAILED = 3
EXIT_INTERRUPTED = 130

InputReader = Callable[[str], str]
LineWriter = Callable[[str], None]
ClientFactory = Callable[[argparse.Namespace], OllamaClient]
ArtifactWriterFactory = Callable[[Path], RunArtifactWriter]
AnalyzerFactory = Callable[[], OfflineAnalyzer]


def build_parser() -> argparse.ArgumentParser:
    """Create the top-level parser and its subcommands."""
    parser = argparse.ArgumentParser(
        prog=PROGRAM_NAME,
        description="Run and preserve double-logging experiments for LLM agents.",
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
    _add_provider_arguments(chat_parser)
    chat_parser.add_argument(
        "--system-prompt",
        help="optional instruction retained for the entire conversation",
    )
    _add_generation_arguments(chat_parser)
    chat_parser.add_argument(
        "--max-tool-rounds",
        type=_positive_int,
        help=(
            "positive tool-call round limit; defaults to "
            "OLLAMA_MAX_TOOL_ROUNDS or 32"
        ),
    )
    chat_parser.add_argument(
        "--mailbox-file",
        type=Path,
        help="optional JSON list of synthetic emails available to mailbox tools",
    )
    chat_parser.set_defaults(handler=run_chat)

    run_parser = subparsers.add_parser(
        "run",
        help="execute and persist one non-interactive scenario episode",
        description=(
            "Resolve one scenario condition/policy cell, execute it once, and "
            "atomically persist all four raw records under a unique run path."
        ),
    )
    _add_provider_arguments(run_parser)
    _add_generation_arguments(run_parser)
    run_parser.add_argument(
        "--scenario-file",
        type=Path,
        required=True,
        help="versioned scenario YAML file",
    )
    run_parser.add_argument(
        "--condition",
        required=True,
        help="condition ID declared by the scenario",
    )
    run_parser.add_argument(
        "--policy-context",
        required=True,
        help="policy-context ID declared by the scenario",
    )
    run_parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("runs"),
        help="raw run parent directory (default: runs)",
    )
    run_parser.add_argument(
        "--run-id",
        help="optional stable run ID; an existing run is never overwritten",
    )
    run_parser.set_defaults(handler=run_scenario)

    analyze_parser = subparsers.add_parser(
        "analyze",
        help="derive a versioned results.json from one closed raw run",
        description=(
            "Validate one closed run and its frozen evaluation YAML, perform "
            "deterministic offline checks, and atomically write results.json."
        ),
    )
    analyze_parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="closed run directory containing the four raw artifacts",
    )
    analyze_parser.add_argument(
        "--evaluation-file",
        type=Path,
        required=True,
        help="versioned evaluation YAML whose IDs must match the run",
    )
    analyze_parser.set_defaults(handler=analyze_run)
    return parser


def _add_provider_arguments(parser: argparse.ArgumentParser) -> None:
    """Add connection flags shared by interactive and experimental commands."""

    parser.add_argument(
        "--provider",
        choices=("ollama",),
        default="ollama",
        help="provider adapter to use (default: ollama)",
    )
    parser.add_argument(
        "--model",
        help="Ollama model identifier; defaults to OLLAMA_MODEL",
    )
    parser.add_argument(
        "--base-url",
        help="Ollama server root; defaults to OLLAMA_BASE_URL or localhost",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=_positive_float,
        help="positive HTTP timeout; defaults to OLLAMA_TIMEOUT_SECONDS",
    )
    parser.add_argument(
        "--keep-alive",
        help="Ollama model residency duration, for example 5m or 0",
    )


def _add_generation_arguments(parser: argparse.ArgumentParser) -> None:
    """Add sampling flags whose exact values are retained in run metadata."""

    parser.add_argument(
        "--temperature",
        type=_non_negative_float,
        help="non-negative Ollama temperature",
    )
    parser.add_argument(
        "--top-k",
        type=_non_negative_int,
        help="non-negative Ollama top_k value",
    )
    parser.add_argument(
        "--top-p",
        type=_probability,
        help="Ollama top_p value between 0 and 1",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="optional deterministic sampling seed",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=_positive_int,
        help="positive Ollama num_predict limit",
    )
    parser.add_argument(
        "--think",
        choices=("false", "true", "low", "medium", "high"),
        default="false",
        help="request exposed reasoning when supported (default: false)",
    )


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
    except (AnalysisError, ArtifactWriteError, OllamaClientError, ValueError) as exc:
        # Configuration, startup, and persistence errors occur outside a closed
        # episode. Provider failures inside `run` are artifact-backed results;
        # `chat` continues to display its per-turn failures interactively.
        print(f"{PROGRAM_NAME}: error: {exc}", file=sys.stderr)
        return EXIT_CLI_ERROR
    except KeyboardInterrupt:
        print(
            f"{PROGRAM_NAME}: interrupted before an episode could close.",
            file=sys.stderr,
        )
        return EXIT_INTERRUPTED


def analyze_run(
    args: argparse.Namespace,
    *,
    write_line: LineWriter = print,
    analyzer_factory: AnalyzerFactory | None = None,
) -> int:
    """Analyze one persisted run without constructing a provider.

    Args:
        args: Parsed `analyze` command configuration.
        write_line: Injectable one-line output boundary for offline tests.
        analyzer_factory: Optional deterministic analyzer constructor.

    Returns:
        Zero after a valid `results.json` is atomically published.  A run may
        still be classified non-evaluable inside that file; non-evaluability is
        an observed research outcome, not a failure of the analyze command.

    Raises:
        AnalysisError: If inputs are incompatible or result publication fails.
            The top-level CLI reports the error and returns exit code one.
    """

    factory = analyzer_factory or OfflineAnalyzer
    written = factory().analyze(
        args.run_dir,
        evaluation_file=args.evaluation_file,
    )
    assessment = written.results.run_assessment
    write_line(
        f"analysis> {written.run_id}: {assessment.overall} "
        f"({assessment.discrepancy_signal})"
    )
    write_line(f"results> {written.results_path}")
    return EXIT_COMPLETED


def run_scenario(
    args: argparse.Namespace,
    *,
    write_line: LineWriter = print,
    client_factory: ClientFactory | None = None,
    artifact_writer_factory: ArtifactWriterFactory | None = None,
    operational_provenance_factory: OperationalProvenanceFactory | None = None,
) -> int:
    """Execute, persist, and summarize one resolved non-interactive episode.

    Args:
        args: Parsed `run` command configuration.
        write_line: Injectable one-line output boundary for offline tests.
        client_factory: Optional provider constructor; production uses Ollama.
        artifact_writer_factory: Optional persistence constructor; production
            uses :class:`RunArtifactWriter`.
        operational_provenance_factory: Optional deterministic provenance seam
            for offline tests. Production captures repository and host details.

    Returns:
        Zero for a completed episode, two for an incomplete episode, three for
        a runtime failure, or 130 for a captured provider-stage interrupt.

    Raises:
        ValueError: For invalid scenario/provider configuration.
        ArtifactWriteError: If destination preflight or atomic publication
            fails.  A terminal exit code is never returned before persistence.

    Scenario resolution and output preflight happen before provider creation.
    The writer then publishes the episode while the provider context is still
    active, so even provider cleanup failure cannot erase an already closed raw
    run directory.
    """

    scenario = resolve_scenario(
        args.scenario_file,
        condition_id=args.condition,
        policy_context_id=args.policy_context,
    )
    options = _ollama_options(args)
    think = _thinking_mode(args.think)
    writer_factory = artifact_writer_factory or RunArtifactWriter
    writer = writer_factory(args.runs_dir)

    # Select and validate the destination before a potentially costly model
    # call.  `write` repeats collision detection before the atomic rename.
    selected_run_id = writer.preflight(run_id=args.run_id)
    provider_factory = client_factory or _create_ollama_client
    with provider_factory(args) as provider:
        episode = ScenarioRunner(
            provider,
            options=options,
            think=think,
            operational_provenance_factory=operational_provenance_factory,
        ).run(scenario)
        artifacts = writer.write(episode, run_id=selected_run_id)

    write_line(
        f"run> {artifacts.run_id}: {episode.status} "
        f"({episode.termination_reason})"
    )
    write_line(f"artifacts> {artifacts.run_directory}")
    return _episode_exit_code(episode.status, episode.termination_reason)


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


def _episode_exit_code(status: str, termination_reason: str) -> int:
    """Map a persisted episode terminal state to a process exit code."""

    if termination_reason == "interrupted":
        return EXIT_INTERRUPTED
    if status == "completed":
        return EXIT_COMPLETED
    if status == "incomplete":
        return EXIT_INCOMPLETE
    if status == "failed":
        return EXIT_FAILED
    raise ValueError(f"Unsupported episode status: {status}")


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
