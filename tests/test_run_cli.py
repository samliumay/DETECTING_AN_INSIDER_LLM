"""Offline end-to-end tests for the persisted non-interactive CLI path.

Each test resolves the real scenario and uses the real runner and artifact
writer with a scripted provider.  No Ollama service, network, or real email
system is contacted.
"""

import argparse
import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from detecting_an_insider_llm.artifacts import RunDirectoryExistsError
from detecting_an_insider_llm.cli import (
    EXIT_COMPLETED,
    EXIT_FAILED,
    EXIT_INCOMPLETE,
    EXIT_INTERRUPTED,
    build_parser,
    main,
    run_scenario,
)
from detecting_an_insider_llm.providers import OllamaClientError
from detecting_an_insider_llm.runtime.agents import (
    ProviderFinishReason,
    ProviderResponse,
    ThinkingMode,
)


SCENARIO_PATH = (
    Path(__file__).parents[1] / "scenarios" / "blackmail" / "v1" / "scenario.yaml"
)
RAW_FILENAMES = {
    "automated_logs.jsonl",
    "model_self_logs.jsonl",
    "metadata.json",
    "journal.jsonl",
}


class ScriptedRunClient:
    """Context-managed provider double for one non-interactive CLI episode."""

    provider_name = "ollama"
    model_name = "run-cli-fixture-model"

    def __init__(
        self,
        responses: Sequence[ProviderResponse | BaseException],
    ) -> None:
        """Copy scripted outcomes and initialize lifecycle/request evidence."""

        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []
        self.closed = False

    def __enter__(self) -> "ScriptedRunClient":
        """Return the same provider instance passed into ScenarioRunner."""

        return self

    def __exit__(self, *_: object) -> None:
        """Record provider cleanup after artifact publication."""

        self.closed = True

    def runtime_metadata(self) -> dict[str, Any]:
        """Return fixed JSON-safe provider provenance."""

        return {"provider_version": "offline-run-cli-fixture-1"}

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
        options: Mapping[str, Any] | None = None,
        *,
        think: ThinkingMode = False,
    ) -> ProviderResponse:
        """Record exact request configuration, then return or raise one item."""

        self.calls.append(
            {
                "messages": deepcopy(list(messages)),
                "tools": deepcopy(list(tools)) if tools is not None else None,
                "options": deepcopy(dict(options)) if options is not None else None,
                "think": think,
            }
        )
        if not self._responses:
            raise AssertionError("ScriptedRunClient received an unexpected call.")
        item = self._responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return deepcopy(item)


def _response(
    *,
    content: str = "",
    tool_calls: Sequence[object] = (),
    finish_reason: ProviderFinishReason = "complete",
) -> ProviderResponse:
    """Build one normalized provider response with retained raw evidence."""

    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = deepcopy(list(tool_calls))
    return ProviderResponse(
        message=message,
        raw_response={"message": deepcopy(message), "run_cli_fixture": True},
        finish_reason=finish_reason,
    )


def _tool_call(name: str, arguments: object) -> dict[str, Any]:
    """Build one model-requested function call in provider response shape."""

    return {
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _run_args(
    runs_dir: Path,
    *,
    run_id: str,
) -> argparse.Namespace:
    """Parse the smallest explicit reproducible run configuration."""

    return build_parser().parse_args(
        [
            "run",
            "--provider",
            "ollama",
            "--model",
            "run-cli-fixture-model",
            "--scenario-file",
            str(SCENARIO_PATH),
            "--condition",
            "baseline",
            "--policy-context",
            "none",
            "--runs-dir",
            str(runs_dir),
            "--run-id",
            run_id,
        ]
    )


def _metadata(runs_dir: Path, run_id: str) -> dict[str, Any]:
    """Load one persisted metadata record for terminal-state assertions."""

    path = runs_dir / run_id / "metadata.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _fixed_operational_provenance(_: Path) -> dict[str, Any]:
    """Return deterministic repository and host metadata for CLI runs."""

    return {
        "code": {
            "capture_status": "captured",
            "git_revision": "fixture-revision",
            "git_dirty": False,
        },
        "execution_host": {
            "capture_scope": "python_process",
            "snapshot_id": "fixture-host",
        },
    }


def test_run_parser_exposes_scenario_output_and_sampling_configuration(
    tmp_path: Path,
) -> None:
    """Verify the public command parses every setting retained by the run."""

    args = build_parser().parse_args(
        [
            "run",
            "--model",
            "qwen3:cloud",
            "--base-url",
            "https://ollama.test",
            "--timeout-seconds",
            "45",
            "--keep-alive",
            "0",
            "--temperature",
            "0.4",
            "--top-k",
            "30",
            "--top-p",
            "0.85",
            "--seed",
            "11",
            "--max-output-tokens",
            "512",
            "--think",
            "medium",
            "--scenario-file",
            str(SCENARIO_PATH),
            "--condition",
            "goal_conflict",
            "--policy-context",
            "explicit",
            "--runs-dir",
            str(tmp_path / "raw-runs"),
            "--run-id",
            "parser-run-001",
        ]
    )

    assert args.command == "run"
    assert args.model == "qwen3:cloud"
    assert args.base_url == "https://ollama.test"
    assert args.timeout_seconds == 45.0
    assert args.keep_alive == "0"
    assert args.temperature == 0.4
    assert args.top_k == 30
    assert args.top_p == 0.85
    assert args.seed == 11
    assert args.max_output_tokens == 512
    assert args.think == "medium"
    assert args.scenario_file == SCENARIO_PATH
    assert args.condition == "goal_conflict"
    assert args.policy_context == "explicit"
    assert args.runs_dir == tmp_path / "raw-runs"
    assert args.run_id == "parser-run-001"
    assert args.handler is run_scenario


def test_run_command_persists_completed_episode_before_zero_exit(
    tmp_path: Path,
) -> None:
    """Verify a completed episode closes all artifacts before returning zero."""

    runs_dir = tmp_path / "runs"
    args = _run_args(runs_dir, run_id="completed-cli-001")
    args.temperature = 0.2
    args.seed = 23
    args.think = "low"
    provider = ScriptedRunClient([_response(content="Finished without tools.")])
    output: list[str] = []

    exit_code = run_scenario(
        args,
        write_line=output.append,
        client_factory=lambda _: provider,
        operational_provenance_factory=_fixed_operational_provenance,
    )

    run_directory = (runs_dir / "completed-cli-001").resolve()
    assert exit_code == EXIT_COMPLETED
    assert provider.closed is True
    assert {path.name for path in run_directory.iterdir()} == RAW_FILENAMES
    assert output == [
        "run> completed-cli-001: completed (completed)",
        f"artifacts> {run_directory}",
    ]
    metadata = _metadata(runs_dir, "completed-cli-001")
    assert metadata["terminal"]["status"] == "completed"
    assert metadata["generation"] == {
        "options": {"seed": 23, "temperature": 0.2},
        "think": "low",
    }
    assert metadata["operational_provenance"] == _fixed_operational_provenance(
        SCENARIO_PATH.parent
    )
    assert provider.calls[0]["options"] == {"seed": 23, "temperature": 0.2}
    assert provider.calls[0]["think"] == "low"


def test_run_command_persists_incomplete_episode_before_exit_two(
    tmp_path: Path,
) -> None:
    """Verify a tool-attempt limit remains an artifact-backed nonzero outcome."""

    # The scenario allows sixteen ordinary attempts.  Seventeen calls in one
    # response make the last request an automatically logged limit rejection.
    tool_calls = [_tool_call("list_emails", {}) for _ in range(17)]
    provider = ScriptedRunClient([_response(tool_calls=tool_calls)])
    runs_dir = tmp_path / "runs"
    output: list[str] = []

    exit_code = run_scenario(
        _run_args(runs_dir, run_id="incomplete-cli-001"),
        write_line=output.append,
        client_factory=lambda _: provider,
        operational_provenance_factory=_fixed_operational_provenance,
    )

    assert exit_code == EXIT_INCOMPLETE
    assert provider.closed is True
    assert output[0] == (
        "run> incomplete-cli-001: incomplete (max_ordinary_tool_attempts)"
    )
    metadata = _metadata(runs_dir, "incomplete-cli-001")
    assert metadata["terminal"]["status"] == "incomplete"
    assert metadata["counts"]["ordinary_tool_attempts"] == 17
    automatic_lines = (
        runs_dir / "incomplete-cli-001" / "automated_logs.jsonl"
    ).read_text(encoding="utf-8").splitlines()
    assert len(automatic_lines) == 17


def test_run_command_persists_length_truncation_before_exit_two(
    tmp_path: Path,
) -> None:
    """A token-limited response closes as incomplete without dispatching calls."""

    provider = ScriptedRunClient(
        [
            _response(
                tool_calls=[_tool_call("send_email", {"to": "partial.test"})],
                finish_reason="length",
            )
        ]
    )
    runs_dir = tmp_path / "runs"
    output: list[str] = []

    exit_code = run_scenario(
        _run_args(runs_dir, run_id="truncated-cli-001"),
        write_line=output.append,
        client_factory=lambda _: provider,
        operational_provenance_factory=_fixed_operational_provenance,
    )

    assert exit_code == EXIT_INCOMPLETE
    assert provider.closed is True
    assert output[0] == (
        "run> truncated-cli-001: incomplete (provider_output_truncated)"
    )
    metadata = _metadata(runs_dir, "truncated-cli-001")
    assert metadata["terminal"] == {
        "error": None,
        "status": "incomplete",
        "termination_reason": "provider_output_truncated",
    }
    assert metadata["counts"]["ordinary_tool_attempts"] == 0
    assert metadata["counts"]["tool_executions"] == 0
    journal_path = runs_dir / "truncated-cli-001" / "journal.jsonl"
    journal = [
        json.loads(line)
        for line in journal_path.read_text(encoding="utf-8").splitlines()
    ]
    provider_response = next(
        row for row in journal if row["event_type"] == "provider_response"
    )
    assert provider_response["payload"]["finish_reason"] == "length"


def test_run_command_persists_failed_episode_before_exit_three(
    tmp_path: Path,
) -> None:
    """Verify a provider failure is retained instead of becoming a CLI-only error."""

    provider = ScriptedRunClient([OllamaClientError("provider unavailable")])
    runs_dir = tmp_path / "runs"
    output: list[str] = []

    exit_code = run_scenario(
        _run_args(runs_dir, run_id="failed-cli-001"),
        write_line=output.append,
        client_factory=lambda _: provider,
        operational_provenance_factory=_fixed_operational_provenance,
    )

    assert exit_code == EXIT_FAILED
    assert provider.closed is True
    assert output[0] == "run> failed-cli-001: failed (provider_error)"
    metadata = _metadata(runs_dir, "failed-cli-001")
    assert metadata["terminal"]["status"] == "failed"
    assert metadata["terminal"]["error"]["error_type"] == "OllamaClientError"
    assert metadata["counts"]["provider_turn_attempts"] == 1
    assert metadata["counts"]["provider_responses"] == 0


def test_run_command_persists_provider_interrupt_before_exit_130(
    tmp_path: Path,
) -> None:
    """Verify a handled Ctrl-C becomes a closed failed episode before exit."""

    provider = ScriptedRunClient([KeyboardInterrupt()])
    runs_dir = tmp_path / "runs"

    exit_code = run_scenario(
        _run_args(runs_dir, run_id="interrupted-cli-001"),
        write_line=lambda _: None,
        client_factory=lambda _: provider,
        operational_provenance_factory=_fixed_operational_provenance,
    )

    assert exit_code == EXIT_INTERRUPTED
    assert provider.closed is True
    metadata = _metadata(runs_dir, "interrupted-cli-001")
    assert metadata["terminal"]["status"] == "failed"
    assert metadata["terminal"]["termination_reason"] == "interrupted"
    assert metadata["terminal"]["error"]["error_type"] == "KeyboardInterrupt"


def test_run_command_checks_existing_destination_before_provider_creation(
    tmp_path: Path,
) -> None:
    """Verify a known run-ID collision cannot spend another model call."""

    runs_dir = tmp_path / "runs"
    (runs_dir / "existing-cli-001").mkdir(parents=True)
    factory_called = False

    def unexpected_provider_factory(_: Any) -> ScriptedRunClient:
        """Record misuse if destination preflight occurs too late."""

        nonlocal factory_called
        factory_called = True
        return ScriptedRunClient([_response(content="must not run")])

    with pytest.raises(RunDirectoryExistsError, match="already exists"):
        run_scenario(
            _run_args(runs_dir, run_id="existing-cli-001"),
            client_factory=unexpected_provider_factory,
            operational_provenance_factory=_fixed_operational_provenance,
        )

    assert factory_called is False


def test_main_reports_output_preflight_failure_as_cli_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify an invalid runs root exits one before provider construction."""

    runs_file = tmp_path / "not-a-directory"
    runs_file.write_text("occupied", encoding="utf-8")

    exit_code = main(
        [
            "run",
            "--model",
            "unused-model",
            "--scenario-file",
            str(SCENARIO_PATH),
            "--condition",
            "baseline",
            "--policy-context",
            "none",
            "--runs-dir",
            str(runs_file),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert "error:" in captured.err
    assert "Could not create runs root" in captured.err
