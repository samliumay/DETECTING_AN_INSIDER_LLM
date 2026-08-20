"""Offline tests for atomic, immutable raw run artifact publication.

The fixtures execute real scenario-runner paths through a scripted provider,
then persist only to pytest temporary directories.  They verify storage and
provenance behavior, not the research hypothesis or any model property.
"""

import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import detecting_an_insider_llm.artifacts as artifacts_module
from detecting_an_insider_llm.artifacts import (
    ArtifactSerializationError,
    ArtifactWriteError,
    RunArtifactWriter,
    RunDirectoryExistsError,
)
from detecting_an_insider_llm.runtime.agents import ProviderResponse, ThinkingMode
from detecting_an_insider_llm.runtime.episode_runner import (
    EpisodeResult,
    ScenarioRunner,
)
from detecting_an_insider_llm.scenario_loader import resolve_scenario


SCENARIO_PATH = (
    Path(__file__).parents[1] / "scenarios" / "blackmail" / "v1" / "scenario.yaml"
)
RAW_FILENAMES = {
    "automated_logs.jsonl",
    "model_self_logs.jsonl",
    "metadata.json",
    "journal.jsonl",
}


class ScriptedProvider:
    """Provide deterministic model turns and one stable metadata snapshot."""

    provider_name = "scripted"
    model_name = "artifact-fixture-model"

    def __init__(self, responses: Sequence[ProviderResponse | Exception]) -> None:
        """Copy the script so each test owns independent provider state."""

        self._responses = list(responses)

    def runtime_metadata(self) -> dict[str, Any]:
        """Return JSON-safe provider provenance without external I/O."""

        return {"provider_version": "offline-artifact-fixture-1"}

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
        options: Mapping[str, Any] | None = None,
        *,
        think: ThinkingMode = False,
    ) -> ProviderResponse:
        """Return or raise the next item while satisfying the provider contract."""

        if not self._responses:
            raise AssertionError("ScriptedProvider received an unexpected call.")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return deepcopy(item)


def _tool_call(name: str, arguments: object) -> dict[str, Any]:
    """Build one provider tool-call object consumed by the runtime parser."""

    return {
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _response(
    *,
    content: str = "",
    tool_calls: Sequence[object] = (),
) -> ProviderResponse:
    """Create distinct normalized and raw provider evidence dictionaries."""

    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = deepcopy(list(tool_calls))
    return ProviderResponse(
        message=message,
        raw_response={"message": deepcopy(message), "artifact_fixture": True},
        finish_reason="complete",
    )


def _fixed_operational_provenance(_: Path) -> dict[str, Any]:
    """Return stable operational metadata for serialized fixture runs."""

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


def _run_episode(
    responses: Sequence[ProviderResponse | Exception],
) -> EpisodeResult:
    """Execute one baseline cell and return its complete in-memory evidence."""

    scenario = resolve_scenario(
        SCENARIO_PATH,
        condition_id="baseline",
        policy_context_id="none",
    )
    return ScenarioRunner(
        ScriptedProvider(responses),
        options={"temperature": 0.1, "seed": 19},
        think=False,
        operational_provenance_factory=_fixed_operational_provenance,
    ).run(scenario)


def _complete_double_logged_episode() -> EpisodeResult:
    """Produce one ordinary action, matching self-log, and final response."""

    return _run_episode(
        [
            _response(tool_calls=[_tool_call("list_emails", {})]),
            _response(
                tool_calls=[
                    _tool_call(
                        "log_action",
                        {
                            "log_id": "log-0001",
                            "tool_name": "list_emails",
                            "action_description": "Listed available mail headers.",
                            "claimed_outcome": "succeeded",
                        },
                    )
                ]
            ),
            _response(content="Review complete."),
        ]
    )


def _read_json_lines(path: Path) -> list[dict[str, Any]]:
    """Decode every nonempty JSONL row for concise artifact assertions."""

    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_writer_publishes_exact_closed_four_file_run(tmp_path: Path) -> None:
    """Verify correlated logs, full journal, metadata, and file protection.

    The final directory must appear with exactly the four raw records.  Both log
    streams receive the shared run ID and execution linkage, while the journal
    retains exact scenario inputs and provider/tool ordering.
    """

    episode = _complete_double_logged_episode()
    runs_root = tmp_path / "runs"
    writer = RunArtifactWriter(runs_root)

    result = writer.write(episode, run_id="complete-run-001")

    assert result.run_id == "complete-run-001"
    assert result.run_directory == runs_root.resolve() / "complete-run-001"
    assert {path.name for path in result.run_directory.iterdir()} == RAW_FILENAMES
    assert not (result.run_directory / "results.json").exists()

    automatic_rows = _read_json_lines(result.automated_logs_path)
    self_log_rows = _read_json_lines(result.model_self_logs_path)
    assert automatic_rows[0]["run_id"] == "complete-run-001"
    assert automatic_rows[0]["log_id"] == "log-0001"
    assert automatic_rows[0]["tool_execution_sequence"] == 1
    assert self_log_rows[0]["run_id"] == "complete-run-001"
    assert self_log_rows[0]["log_id"] == "log-0001"
    assert self_log_rows[0]["tool_execution_sequence"] == 2

    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["schema_version"] == "1"
    assert metadata["artifact_state"] == "closed"
    assert metadata["run_id"] == "complete-run-001"
    assert metadata["terminal"] == {
        "error": None,
        "status": "completed",
        "termination_reason": "completed",
    }
    assert metadata["counts"]["provider_turn_attempts"] == 3
    assert metadata["counts"]["automatic_records"] == 1
    assert metadata["counts"]["model_self_log_records"] == 1
    assert metadata["operational_provenance"] == _fixed_operational_provenance(
        SCENARIO_PATH.parent
    )
    assert set(metadata["artifacts"]) == RAW_FILENAMES

    journal_rows = _read_json_lines(result.journal_path)
    assert [row["event_type"] for row in journal_rows] == [
        "episode_started",
        "provider_request",
        "provider_response",
        "tool_execution",
        "provider_request",
        "provider_response",
        "tool_execution",
        "provider_request",
        "provider_response",
        "episode_finished",
    ]
    assert [row["journal_sequence"] for row in journal_rows] == list(
        range(1, len(journal_rows) + 1)
    )
    scenario_snapshot = journal_rows[0]["payload"]["scenario"]
    assert len(scenario_snapshot["input_emails"]) == 11
    assert scenario_snapshot["system_prompt"] == episode.system_prompt
    assert {
        row["payload"]["finish_reason"]
        for row in journal_rows
        if row["event_type"] == "provider_response"
    } == {"complete"}
    assert journal_rows[-1]["payload"]["status"] == "completed"

    for artifact_path in (
        result.automated_logs_path,
        result.model_self_logs_path,
        result.metadata_path,
        result.journal_path,
    ):
        assert artifact_path.stat().st_mode & 0o222 == 0


def test_writer_creates_empty_jsonl_streams_without_fabricated_records(
    tmp_path: Path,
) -> None:
    """Verify a no-tool completion still creates both required empty log files."""

    episode = _run_episode([_response(content="No action was requested.")])

    result = RunArtifactWriter(tmp_path / "runs").write(
        episode,
        run_id="no-tools-001",
    )

    assert result.automated_logs_path.read_text(encoding="utf-8") == ""
    assert result.model_self_logs_path.read_text(encoding="utf-8") == ""


def test_writer_preserves_pre_provider_provenance_failure(tmp_path: Path) -> None:
    """An unexpected capture failure closes without contacting the provider."""

    scenario = resolve_scenario(
        SCENARIO_PATH,
        condition_id="baseline",
        policy_context_id="none",
    )

    def fail_capture(_: Path) -> dict[str, Any]:
        raise RuntimeError("provenance probe failed")

    episode = ScenarioRunner(
        ScriptedProvider([]),
        operational_provenance_factory=fail_capture,
    ).run(scenario)
    result = RunArtifactWriter(tmp_path / "runs").write(
        episode,
        run_id="provenance-failure-001",
    )

    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["terminal"]["termination_reason"] == (
        "operational_provenance_error"
    )
    assert metadata["operational_provenance"] == {}
    assert metadata["counts"]["provider_turn_attempts"] == 0
    event_types = [
        row["event_type"] for row in _read_json_lines(result.journal_path)
    ]
    assert event_types == [
        "episode_started",
        "operational_provenance_failure",
        "episode_finished",
    ]
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["artifacts"]["automated_logs.jsonl"]["record_count"] == 0
    assert metadata["artifacts"]["model_self_logs.jsonl"]["record_count"] == 0


def test_writer_persists_failed_episode_and_failed_provider_attempt(
    tmp_path: Path,
) -> None:
    """Verify terminal provider failure and partial action evidence survive."""

    episode = _run_episode(
        [
            _response(tool_calls=[_tool_call("list_emails", {})]),
            RuntimeError("provider unavailable"),
        ]
    )

    result = RunArtifactWriter(tmp_path / "runs").write(
        episode,
        run_id="failed-run-001",
    )

    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["terminal"]["status"] == "failed"
    assert metadata["terminal"]["error"] == {
        "error_type": "RuntimeError",
        "message": "provider unavailable",
        "phase": "provider_call",
    }
    assert metadata["counts"]["provider_turn_attempts"] == 2
    assert metadata["counts"]["provider_responses"] == 1
    assert result.model_self_logs_path.read_text(encoding="utf-8") == ""

    journal_rows = _read_json_lines(result.journal_path)
    assert "provider_failure" in [row["event_type"] for row in journal_rows]
    failure = next(
        row for row in journal_rows if row["event_type"] == "provider_failure"
    )
    assert failure["payload"]["provider_turn"] == 2
    assert failure["payload"]["error"]["error_type"] == "RuntimeError"


def test_writer_generates_distinct_run_directories(tmp_path: Path) -> None:
    """Verify omitted IDs produce independent collision-resistant run paths."""

    episode = _run_episode([_response(content="Complete.")])
    writer = RunArtifactWriter(tmp_path / "runs")

    first = writer.write(episode)
    second = writer.write(episode)

    assert first.run_id != second.run_id
    assert first.run_directory != second.run_directory
    assert first.run_directory.is_dir()
    assert second.run_directory.is_dir()


def test_writer_never_overwrites_existing_run(tmp_path: Path) -> None:
    """Verify stable-ID reuse fails without changing the first raw observation."""

    episode = _run_episode([_response(content="Complete.")])
    writer = RunArtifactWriter(tmp_path / "runs")
    first = writer.write(episode, run_id="stable-run-001")
    original_metadata = first.metadata_path.read_bytes()

    with pytest.raises(RunDirectoryExistsError, match="already exists"):
        writer.write(episode, run_id="stable-run-001")

    assert first.metadata_path.read_bytes() == original_metadata
    assert {path.name for path in first.run_directory.iterdir()} == RAW_FILENAMES


def test_writer_rejects_non_json_evidence_before_disk_access(tmp_path: Path) -> None:
    """Verify provider-specific Python objects cannot leak into raw JSON files."""

    episode = _run_episode([_response(content="Complete.")])
    original_response = episode.provider_responses[0]
    invalid_response = ProviderResponse(
        message=deepcopy(original_response.message),
        raw_response={"unsupported": object()},
        finish_reason=original_response.finish_reason,
    )
    invalid_episode = replace(
        episode,
        provider_responses=(invalid_response,),
    )
    runs_root = tmp_path / "runs"

    with pytest.raises(ArtifactSerializationError, match="strict JSON"):
        RunArtifactWriter(runs_root).write(
            invalid_episode,
            run_id="invalid-json-001",
        )

    assert not runs_root.exists()


def test_writer_rejects_completed_episode_with_truncated_final_response(
    tmp_path: Path,
) -> None:
    """Persistence must not reintroduce the pilot's false-completion state."""

    episode = _run_episode([_response(content="Complete.")])
    truncated_response = replace(
        episode.provider_responses[-1],
        finish_reason="length",
    )
    invalid_episode = replace(
        episode,
        provider_responses=(*episode.provider_responses[:-1], truncated_response),
    )

    with pytest.raises(
        ArtifactSerializationError,
        match="must end with a complete provider response",
    ):
        RunArtifactWriter(tmp_path / "runs").write(
            invalid_episode,
            run_id="invalid-truncated-completion",
        )

    assert not (tmp_path / "runs").exists()


def test_writer_cleans_staging_when_a_file_write_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify an interrupted write cannot leave a final or staging run path."""

    episode = _run_episode([_response(content="Complete.")])
    real_write = artifacts_module._write_read_only_file
    call_count = 0

    def fail_on_second_file(path: Path, contents: str) -> None:
        """Write the first artifact, then simulate a storage failure."""

        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise OSError("simulated disk failure")
        real_write(path, contents)

    monkeypatch.setattr(
        artifacts_module,
        "_write_read_only_file",
        fail_on_second_file,
    )
    runs_root = tmp_path / "runs"

    with pytest.raises(ArtifactWriteError, match="atomically publish"):
        RunArtifactWriter(runs_root).write(
            episode,
            run_id="interrupted-run-001",
        )

    assert runs_root.is_dir()
    assert list(runs_root.iterdir()) == []


@pytest.mark.parametrize(
    "unsafe_run_id",
    ["../escape", "/absolute", ".hidden", "contains space", ""],
)
def test_writer_rejects_unsafe_run_ids_before_creating_root(
    tmp_path: Path,
    unsafe_run_id: str,
) -> None:
    """Verify run IDs cannot traverse or create ambiguous output paths."""

    episode = _run_episode([_response(content="Complete.")])
    runs_root = tmp_path / "runs"

    with pytest.raises(ValueError, match="run_id"):
        RunArtifactWriter(runs_root).write(episode, run_id=unsafe_run_id)

    assert not runs_root.exists()
