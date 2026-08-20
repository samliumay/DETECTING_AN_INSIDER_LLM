"""Atomically persist the four immutable raw records for one episode.

The episode runner owns provider and tool orchestration.  This module accepts
only its finished in-memory :class:`EpisodeResult`, converts that evidence to
versioned JSON documents, and publishes a closed run directory.  It performs no
model calls and no behavioral analysis.

All documents are written to a hidden sibling staging directory first.  The
staging directory is renamed to the final unique run path only after every file
has been flushed and synced.  Consequently a normal reader never observes a
partially populated `runs/<run_id>` directory.
"""

import json
import os
import re
import shutil
import stat
import tempfile
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from detecting_an_insider_llm.runtime.episode_runner import (
    EpisodeResult,
    ProviderRequest,
)
from detecting_an_insider_llm.runtime.tool_loop import (
    ToolCallExecution,
    tool_result_message,
)


AUTOMATED_LOGS_FILENAME = "automated_logs.jsonl"
MODEL_SELF_LOGS_FILENAME = "model_self_logs.jsonl"
METADATA_FILENAME = "metadata.json"
JOURNAL_FILENAME = "journal.jsonl"
RAW_ARTIFACT_FILENAMES = (
    AUTOMATED_LOGS_FILENAME,
    MODEL_SELF_LOGS_FILENAME,
    METADATA_FILENAME,
    JOURNAL_FILENAME,
)

# Restrict caller-supplied IDs to one path component.  Generated IDs use the
# same alphabet, so a run directory can never escape the configured root.
_RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_READ_ONLY_FILE_MODE = stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH


class ArtifactWriteError(RuntimeError):
    """Report an operational failure while publishing a raw run directory."""


class ArtifactSerializationError(ArtifactWriteError):
    """Report episode evidence that cannot be represented as strict JSON."""


class RunDirectoryExistsError(ArtifactWriteError):
    """Prevent an existing run from being overwritten or silently resumed."""


@dataclass(frozen=True, slots=True)
class ArtifactWriteResult:
    """Paths of one successfully closed raw run.

    The files already exist when this object is returned.  They are made
    read-only as a practical guard against accidental mutation; the future
    analyzer may add `results.json` without rewriting these four paths.
    """

    run_id: str
    run_directory: Path
    automated_logs_path: Path
    model_self_logs_path: Path
    metadata_path: Path
    journal_path: Path


class RunArtifactWriter:
    """Publish complete episode evidence beneath one configured runs root.

    Args:
        runs_root: Parent directory under which unique run directories are
            created.  It is created on the first write when necessary.

    The writer never reopens or updates an existing run.  Passing a repeated
    `run_id` raises :class:`RunDirectoryExistsError`, which keeps retry behavior
    explicit instead of merging incompatible observations.
    """

    def __init__(self, runs_root: Path | str) -> None:
        """Snapshot the caller's output root without creating files yet."""

        self._runs_root = Path(runs_root)

    @property
    def runs_root(self) -> Path:
        """Return the configured root as an independent path object."""

        return Path(self._runs_root)

    def preflight(self, *, run_id: str | None = None) -> str:
        """Validate and select a writable destination before provider work.

        Args:
            run_id: Optional stable identifier.  When omitted, a fresh ID is
                generated and returned for later use with :meth:`write`.

        Returns:
            The validated ID whose final path is currently available.

        Raises:
            ValueError: If `run_id` is unsafe.
            RunDirectoryExistsError: If the final run path already exists.
            ArtifactWriteError: If the runs root cannot be created or is not a
                directory.

        This check does not reserve the name.  :meth:`write` repeats collision
        detection immediately before publication, which remains authoritative
        if another process selects the same explicit ID concurrently.
        """

        selected_run_id = _validated_run_id(
            _new_run_id() if run_id is None else run_id
        )
        runs_root = self._ensure_runs_root()
        final_directory = runs_root / selected_run_id
        if os.path.lexists(final_directory):
            raise RunDirectoryExistsError(
                f"Run directory already exists: {final_directory}"
            )
        return selected_run_id

    def write(
        self,
        episode: EpisodeResult,
        *,
        run_id: str | None = None,
    ) -> ArtifactWriteResult:
        """Atomically publish all four raw artifacts for one finished episode.

        Args:
            episode: Complete in-memory result returned by `ScenarioRunner`.
            run_id: Optional stable identifier.  When omitted, a UTC timestamp
                plus random UUID is generated.  Supplied IDs must be one safe
                path component and may not identify an existing run.

        Returns:
            An :class:`ArtifactWriteResult` containing the closed paths.

        Raises:
            TypeError: If `episode` is not an :class:`EpisodeResult`.
            ValueError: If a caller-supplied run ID is unsafe.
            ArtifactSerializationError: If evidence violates runner invariants
                or contains a value that strict JSON cannot represent.
            RunDirectoryExistsError: If the selected run path already exists.
            ArtifactWriteError: If directory creation, writing, syncing,
                permission changes, cleanup, or publication fails.

        Serialization and invariant checks happen before staging begins.  Once
        staging exists, every handled failure removes it before the exception is
        returned, unless cleanup itself fails and the raised message identifies
        the orphan path explicitly.
        """

        if not isinstance(episode, EpisodeResult):
            raise TypeError("episode must be an EpisodeResult.")

        selected_run_id = _validated_run_id(
            _new_run_id() if run_id is None else run_id
        )
        _validate_episode_for_persistence(episode)
        persisted_at = datetime.now(timezone.utc)
        documents = _serialized_artifacts(
            episode,
            run_id=selected_run_id,
            persisted_at=persisted_at,
        )

        runs_root = self._ensure_runs_root()

        final_directory = runs_root / selected_run_id
        if os.path.lexists(final_directory):
            raise RunDirectoryExistsError(
                f"Run directory already exists: {final_directory}"
            )

        try:
            staging_directory = Path(
                tempfile.mkdtemp(
                    prefix=f".{selected_run_id}.tmp-",
                    dir=runs_root,
                )
            )
        except OSError as exc:
            raise ArtifactWriteError(
                f"Could not create a staging directory under {runs_root}: {exc}"
            ) from exc

        try:
            for filename in RAW_ARTIFACT_FILENAMES:
                _write_read_only_file(
                    staging_directory / filename,
                    documents[filename],
                )
            _sync_directory(staging_directory)

            # Check again immediately before publication.  Generated UUID-based
            # IDs make collision extremely unlikely; this second check also
            # catches ordinary concurrent reuse of an explicit stable ID.
            if os.path.lexists(final_directory):
                raise RunDirectoryExistsError(
                    f"Run directory already exists: {final_directory}"
                )
            staging_directory.rename(final_directory)
        except (Exception, KeyboardInterrupt) as exc:
            _cleanup_staging_directory(staging_directory, original_error=exc)
            if isinstance(exc, ArtifactWriteError):
                raise
            raise ArtifactWriteError(
                f"Could not atomically publish run {selected_run_id}: {exc}"
            ) from exc

        return ArtifactWriteResult(
            run_id=selected_run_id,
            run_directory=final_directory,
            automated_logs_path=final_directory / AUTOMATED_LOGS_FILENAME,
            model_self_logs_path=final_directory / MODEL_SELF_LOGS_FILENAME,
            metadata_path=final_directory / METADATA_FILENAME,
            journal_path=final_directory / JOURNAL_FILENAME,
        )

    def _ensure_runs_root(self) -> Path:
        """Create and return the resolved output root or raise a clear error."""

        runs_root = self._runs_root.resolve()
        try:
            runs_root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ArtifactWriteError(
                f"Could not create runs root {runs_root}: {exc}"
            ) from exc
        if not runs_root.is_dir():
            raise ArtifactWriteError(f"Runs root is not a directory: {runs_root}")
        return runs_root


def _validated_run_id(run_id: str) -> str:
    """Return a safe single-component run ID or reject it before disk access."""

    if not isinstance(run_id, str):
        raise TypeError("run_id must be a string when provided.")
    if not _RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(
            "run_id must be 1-128 characters, begin with an alphanumeric "
            "character, and contain only letters, numbers, '.', '_', or '-'."
        )
    return run_id


def _new_run_id() -> str:
    """Generate a sortable UTC prefix plus collision-resistant random suffix."""

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{timestamp}-{uuid.uuid4().hex}"


def _validate_episode_for_persistence(episode: EpisodeResult) -> None:
    """Fail before writing if related episode collections are inconsistent."""

    _require_aware_timestamp(episode.started_at, field_name="started_at")
    _require_aware_timestamp(episode.finished_at, field_name="finished_at")
    if episode.finished_at < episode.started_at:
        raise ArtifactSerializationError(
            "Episode finished_at must not precede started_at."
        )
    if episode.provider_turn_count != len(episode.provider_requests):
        raise ArtifactSerializationError(
            "provider_turn_count does not match provider_requests."
        )
    if len(episode.provider_responses) > len(episode.provider_requests):
        raise ArtifactSerializationError(
            "provider_responses cannot exceed attempted provider requests."
        )
    expected_turns = list(range(1, len(episode.provider_requests) + 1))
    observed_turns = [
        request.turn_sequence for request in episode.provider_requests
    ]
    if observed_turns != expected_turns:
        raise ArtifactSerializationError(
            "Provider request sequences must be contiguous and one-based."
        )

    expected_tool_sequences = list(range(1, len(episode.tool_executions) + 1))
    if [item.sequence for item in episode.tool_executions] != expected_tool_sequences:
        raise ArtifactSerializationError(
            "Tool execution sequences must be contiguous and one-based."
        )
    if len(episode.tool_executions) != (
        episode.ordinary_tool_attempt_count + episode.self_log_attempt_count
    ):
        raise ArtifactSerializationError(
            "Tool executions do not match ordinary and self-log attempt counts."
        )
    if len(episode.automatic_records) > episode.ordinary_tool_attempt_count:
        raise ArtifactSerializationError(
            "Automatic records cannot exceed ordinary tool attempts."
        )
    if len(episode.model_self_logs) > episode.self_log_attempt_count:
        raise ArtifactSerializationError(
            "Stored self-logs cannot exceed model self-log attempts."
        )

    if episode.status == "completed":
        if episode.termination_reason != "completed" or episode.error is not None:
            raise ArtifactSerializationError(
                "A completed episode must have completed termination and no error."
            )
    elif episode.status == "failed":
        if episode.error is None:
            raise ArtifactSerializationError(
                "A failed episode must retain structured error evidence."
            )
    elif episode.error is not None:
        raise ArtifactSerializationError(
            "An incomplete episode must not contain an unexpected runtime error."
        )


def _serialized_artifacts(
    episode: EpisodeResult,
    *,
    run_id: str,
    persisted_at: datetime,
) -> dict[str, str]:
    """Build and strictly serialize all documents before touching staging."""

    automatic_rows = _automatic_log_rows(episode, run_id=run_id)
    self_log_rows = _model_self_log_rows(episode, run_id=run_id)
    journal_rows = _journal_rows(episode, run_id=run_id)
    metadata = _metadata_document(
        episode,
        run_id=run_id,
        persisted_at=persisted_at,
        automatic_record_count=len(automatic_rows),
        self_log_record_count=len(self_log_rows),
        journal_record_count=len(journal_rows),
    )
    return {
        AUTOMATED_LOGS_FILENAME: _json_lines(automatic_rows),
        MODEL_SELF_LOGS_FILENAME: _json_lines(self_log_rows),
        METADATA_FILENAME: _json_document(metadata),
        JOURNAL_FILENAME: _json_lines(journal_rows),
    }


def _automatic_log_rows(
    episode: EpisodeResult,
    *,
    run_id: str,
) -> list[dict[str, Any]]:
    """Attach run and journal linkage to authoritative automatic records."""

    execution_by_log_id: dict[str, int] = {}
    for execution in episode.tool_executions:
        log_id = execution.result.audit_result.get("log_id")
        if isinstance(log_id, str):
            execution_by_log_id[log_id] = execution.sequence

    rows: list[dict[str, Any]] = []
    for record in episode.automatic_records:
        tool_sequence = execution_by_log_id.get(record.log_id)
        if tool_sequence is None:
            raise ArtifactSerializationError(
                f"Automatic record {record.record_id} has no tool execution linkage."
            )
        row = record.model_dump(mode="json")
        row["run_id"] = run_id
        row["tool_execution_sequence"] = tool_sequence
        rows.append(row)
    return rows


def _model_self_log_rows(
    episode: EpisodeResult,
    *,
    run_id: str,
) -> list[dict[str, Any]]:
    """Attach run and execution linkage only to self-logs actually stored."""

    execution_by_record_id: dict[str, int] = {}
    for execution in episode.tool_executions:
        if execution.call.requested_tool_name != "log_action":
            continue
        record_id = execution.result.model_result.get("record_id")
        if isinstance(record_id, str):
            execution_by_record_id[record_id] = execution.sequence

    rows: list[dict[str, Any]] = []
    for record in episode.model_self_logs:
        tool_sequence = execution_by_record_id.get(record.record_id)
        if tool_sequence is None:
            raise ArtifactSerializationError(
                f"Self-log record {record.record_id} has no tool execution linkage."
            )
        row = record.model_dump(mode="json")
        row["run_id"] = run_id
        row["tool_execution_sequence"] = tool_sequence
        rows.append(row)
    return rows


def _journal_rows(
    episode: EpisodeResult,
    *,
    run_id: str,
) -> list[dict[str, Any]]:
    """Create an ordered replayable journal from requests and observations."""

    rows: list[dict[str, Any]] = []

    def append_event(event_type: str, payload: dict[str, Any]) -> None:
        """Append one event with a monotonic per-run journal sequence."""

        rows.append(
            {
                "schema_version": "1",
                "run_id": run_id,
                "journal_sequence": len(rows) + 1,
                "event_type": event_type,
                "payload": payload,
            }
        )

    append_event(
        "episode_started",
        {
            "observed_at": _utc_text(episode.started_at),
            "scenario": _scenario_snapshot(episode),
        },
    )

    execution_cursor = 0
    for response_index, request in enumerate(episode.provider_requests):
        append_event(
            "provider_request",
            {"provider_turn": request.turn_sequence, "request": _request_dict(request)},
        )
        if response_index >= len(episode.provider_responses):
            if episode.error is not None:
                append_event(
                    "provider_failure",
                    {
                        "provider_turn": request.turn_sequence,
                        "error": asdict(episode.error),
                    },
                )
            continue

        response = episode.provider_responses[response_index]
        append_event(
            "provider_response",
            {
                "provider_turn": request.turn_sequence,
                "message": deepcopy(response.message),
                "raw_response": deepcopy(response.raw_response),
            },
        )

        raw_tool_calls = response.message.get("tool_calls")
        if not isinstance(raw_tool_calls, list):
            continue
        for raw_tool_call in raw_tool_calls:
            if execution_cursor >= len(episode.tool_executions):
                break
            execution = episode.tool_executions[execution_cursor]
            if execution.call.raw_tool_call != raw_tool_call:
                break
            append_event(
                "tool_execution",
                {
                    "provider_turn": request.turn_sequence,
                    "execution": _tool_execution_dict(execution),
                    "model_message": (
                        None
                        if execution.result.status == "failed"
                        else tool_result_message(execution)
                    ),
                },
            )
            execution_cursor += 1
            if execution.result.status == "failed":
                break

    if execution_cursor != len(episode.tool_executions):
        raise ArtifactSerializationError(
            "Tool executions cannot be ordered beneath provider responses."
        )

    if (
        episode.error is not None
        and episode.error.phase == "provider_metadata"
        and not episode.provider_requests
    ):
        append_event("provider_metadata_failure", {"error": asdict(episode.error)})

    append_event(
        "episode_finished",
        {
            "observed_at": _utc_text(episode.finished_at),
            "status": episode.status,
            "termination_reason": episode.termination_reason,
            "counts": _episode_counts(episode),
            "error": asdict(episode.error) if episode.error is not None else None,
        },
    )
    return rows


def _metadata_document(
    episode: EpisodeResult,
    *,
    run_id: str,
    persisted_at: datetime,
    automatic_record_count: int,
    self_log_record_count: int,
    journal_record_count: int,
) -> dict[str, Any]:
    """Build the closed-run configuration, provenance, and manifest record."""

    counts = _episode_counts(episode)
    counts.update(
        {
            "automatic_records": automatic_record_count,
            "model_self_log_records": self_log_record_count,
            "journal_records": journal_record_count,
        }
    )
    return {
        "schema_version": "1",
        "artifact_state": "closed",
        "run_id": run_id,
        "episode_schema_version": episode.schema_version,
        "timestamps": {
            "started_at": _utc_text(episode.started_at),
            "finished_at": _utc_text(episode.finished_at),
            "persisted_at": _utc_text(persisted_at),
        },
        "scenario": {
            "schema_version": episode.scenario_schema_version,
            "scenario_id": episode.scenario_id,
            "title": episode.scenario_title,
            "scenario_current_time": _utc_text(episode.scenario_current_time),
            "agent_email": episode.agent_email,
            "condition_id": episode.condition_id,
            "policy_context_id": episode.policy_context_id,
            "evaluation_id": episode.evaluation_id,
            "system_prompt": episode.system_prompt,
            "user_prompt": episode.user_prompt,
            "input_email_ids": [email.email_id for email in episode.input_emails],
        },
        "provider": {
            "provider_name": episode.provider_name,
            "model_name": episode.model_name,
            "runtime_metadata": deepcopy(episode.provider_metadata),
        },
        "generation": {
            "options": deepcopy(episode.options),
            "think": episode.think,
        },
        "execution_limits": episode.execution_limits.model_dump(mode="json"),
        "tool_definitions": deepcopy(list(episode.tool_definitions)),
        "terminal": {
            "status": episode.status,
            "termination_reason": episode.termination_reason,
            "error": asdict(episode.error) if episode.error is not None else None,
        },
        "counts": counts,
        "artifacts": {
            AUTOMATED_LOGS_FILENAME: {
                "format": "jsonl",
                "record_count": automatic_record_count,
            },
            MODEL_SELF_LOGS_FILENAME: {
                "format": "jsonl",
                "record_count": self_log_record_count,
            },
            METADATA_FILENAME: {"format": "json", "record_count": 1},
            JOURNAL_FILENAME: {
                "format": "jsonl",
                "record_count": journal_record_count,
            },
        },
    }


def _scenario_snapshot(episode: EpisodeResult) -> dict[str, Any]:
    """Retain exact synthetic inputs needed to reproduce the episode."""

    return {
        "schema_version": episode.scenario_schema_version,
        "scenario_id": episode.scenario_id,
        "title": episode.scenario_title,
        "scenario_current_time": _utc_text(episode.scenario_current_time),
        "agent_email": episode.agent_email,
        "condition_id": episode.condition_id,
        "policy_context_id": episode.policy_context_id,
        "evaluation_id": episode.evaluation_id,
        "system_prompt": episode.system_prompt,
        "user_prompt": episode.user_prompt,
        "input_emails": [
            email.model_dump(mode="json") for email in episode.input_emails
        ],
    }


def _request_dict(request: ProviderRequest) -> dict[str, Any]:
    """Convert a typed provider request into strict JSON-shaped evidence."""

    return {
        "schema_version": request.schema_version,
        "turn_sequence": request.turn_sequence,
        "messages": deepcopy(list(request.messages)),
        "tools": deepcopy(list(request.tools)),
        "options": deepcopy(request.options),
        "think": request.think,
    }


def _tool_execution_dict(execution: ToolCallExecution) -> dict[str, Any]:
    """Convert parsed call and separate result views without losing raw data."""

    return {
        "sequence": execution.sequence,
        "call": {
            "requested_tool_name": execution.call.requested_tool_name,
            "raw_arguments": deepcopy(execution.call.raw_arguments),
            "tool_call_id": execution.call.tool_call_id,
            "raw_tool_call": deepcopy(execution.call.raw_tool_call),
        },
        "result": {
            "tool_name": execution.result.tool_name,
            "arguments": deepcopy(execution.result.arguments),
            "status": execution.result.status,
            "model_result": deepcopy(execution.result.model_result),
            "audit_result": deepcopy(execution.result.audit_result),
        },
    }


def _episode_counts(episode: EpisodeResult) -> dict[str, int]:
    """Return denominators shared by metadata and terminal journal events."""

    return {
        "provider_turn_attempts": episode.provider_turn_count,
        "provider_responses": len(episode.provider_responses),
        "ordinary_tool_attempts": episode.ordinary_tool_attempt_count,
        "self_log_attempts": episode.self_log_attempt_count,
        "tool_executions": len(episode.tool_executions),
        "sent_emails": len(episode.sent_emails),
    }


def _json_lines(rows: list[dict[str, Any]]) -> str:
    """Serialize independent JSON objects with one trailing newline per row."""

    return "".join(f"{_strict_json(row)}\n" for row in rows)


def _json_document(document: dict[str, Any]) -> str:
    """Serialize readable deterministic JSON and end the file with a newline."""

    try:
        return (
            json.dumps(
                document,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    except (TypeError, ValueError) as exc:
        raise ArtifactSerializationError(
            f"Artifact metadata is not strict JSON: {exc}"
        ) from exc


def _strict_json(value: dict[str, Any]) -> str:
    """Serialize one compact object while rejecting NaN and custom Python data."""

    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ArtifactSerializationError(
            f"Artifact record is not strict JSON: {exc}"
        ) from exc


def _write_read_only_file(path: Path, contents: str) -> None:
    """Create, flush, sync, and protect one staging artifact from mutation."""

    with path.open("x", encoding="utf-8", newline="\n") as artifact_file:
        artifact_file.write(contents)
        artifact_file.flush()
        os.fsync(artifact_file.fileno())
    path.chmod(_READ_ONLY_FILE_MODE)


def _sync_directory(path: Path) -> None:
    """Sync staging directory entries before the atomic publication rename."""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_descriptor = os.open(path, flags)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


def _cleanup_staging_directory(
    staging_directory: Path,
    *,
    original_error: BaseException,
) -> None:
    """Remove only the exact writer-created staging path after a failed write."""

    try:
        shutil.rmtree(staging_directory)
    except OSError as cleanup_error:
        raise ArtifactWriteError(
            "Artifact publication failed and staging cleanup also failed; "
            f"inspect {staging_directory}. Original error: {original_error}"
        ) from cleanup_error


def _require_aware_timestamp(timestamp: datetime, *, field_name: str) -> None:
    """Reject machine-local timestamps at the artifact boundary."""

    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ArtifactSerializationError(
            f"Episode {field_name} must include a timezone offset."
        )


def _utc_text(timestamp: datetime) -> str:
    """Format an aware timestamp in canonical UTC ISO 8601 form."""

    _require_aware_timestamp(timestamp, field_name="timestamp")
    return timestamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
