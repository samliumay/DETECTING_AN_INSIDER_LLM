"""Capture privacy-conscious operational provenance for experimental runs.

Code state is checked for every episode because the worktree can change between
runs.  Execution-host hardware is captured once per Python process and copied
into every episode, making each artifact self-contained without repeatedly
probing an unchanged study machine.  The snapshot deliberately excludes host
names, user names, environment variables, device serials, and GPU UUIDs.
"""

import hashlib
import json
import os
import platform
import subprocess
from collections.abc import Sequence
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any


COMMAND_TIMEOUT_SECONDS = 5.0
_EXPECTED_COMMAND_ERRORS = (
    OSError,
    subprocess.CalledProcessError,
    subprocess.TimeoutExpired,
)


def capture_operational_provenance(repository_path: Path) -> dict[str, Any]:
    """Return code and execution-host metadata before an episode starts.

    Expected discovery failures are represented inside the returned object so
    missing provenance remains visible. Unexpected programming errors are not
    swallowed; the episode runner records them as a provenance-stage failure.
    """

    if not isinstance(repository_path, Path):
        raise TypeError("repository_path must be a pathlib.Path.")

    return {
        "code": _capture_git_metadata(repository_path),
        "execution_host": deepcopy(_cached_execution_host_metadata()),
    }


def _capture_git_metadata(repository_path: Path) -> dict[str, Any]:
    """Capture the target repository's commit and current dirty state."""

    metadata: dict[str, Any] = {
        "capture_status": "unavailable",
        "git_revision": None,
        "git_dirty": None,
    }
    try:
        revision_result = _run_command(
            ["git", "-C", str(repository_path), "rev-parse", "HEAD"]
        )
    except _EXPECTED_COMMAND_ERRORS as exc:
        metadata["error_type"] = _command_error_type(exc)
        return metadata

    revision = revision_result.stdout.strip()
    if not revision:
        metadata["error_type"] = "empty_revision"
        return metadata
    metadata["git_revision"] = revision

    try:
        status_result = _run_command(
            [
                "git",
                "-C",
                str(repository_path),
                "status",
                "--porcelain=v1",
                "--untracked-files=normal",
            ]
        )
    except _EXPECTED_COMMAND_ERRORS as exc:
        metadata["capture_status"] = "partial"
        metadata["error_type"] = _command_error_type(exc)
        return metadata

    metadata["capture_status"] = "captured"
    metadata["git_dirty"] = bool(status_result.stdout.strip())
    return metadata


@lru_cache(maxsize=1)
def _cached_execution_host_metadata() -> dict[str, Any]:
    """Probe stable execution-host details once per Python process."""

    metadata: dict[str, Any] = {
        "capture_scope": "python_process",
        "operating_system": platform.system() or None,
        "operating_system_release": platform.release() or None,
        "machine_architecture": platform.machine() or None,
        "python_version": platform.python_version(),
        "logical_cpu_count": os.cpu_count(),
        "cpu_model": _cpu_model(),
        "total_memory_mib": _total_memory_mib(),
    }
    metadata.update(_capture_nvidia_gpu_metadata())
    canonical_snapshot = json.dumps(
        metadata,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    metadata["snapshot_id"] = hashlib.sha256(canonical_snapshot).hexdigest()
    return metadata


def _cpu_model() -> str | None:
    """Return a descriptive CPU model without collecting machine identity."""

    cpuinfo_path = Path("/proc/cpuinfo")
    try:
        lines = cpuinfo_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        lines = []
    else:
        for line in lines:
            key, separator, value = line.partition(":")
            if separator and key.strip().casefold() in {"model name", "hardware"}:
                model = value.strip()
                if model:
                    return model

    processor = platform.processor().strip()
    return processor or None


def _capture_nvidia_gpu_metadata() -> dict[str, Any]:
    """Capture non-identifying NVIDIA GPU configuration when available."""

    try:
        result = _run_command(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ]
        )
    except _EXPECTED_COMMAND_ERRORS as exc:
        return {
            "nvidia_smi_status": "unavailable",
            "nvidia_smi_error_type": _command_error_type(exc),
            "nvidia_gpus": [],
        }

    gpus: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",", maxsplit=3)]
        if len(fields) != 4:
            return {
                "nvidia_smi_status": "invalid_output",
                "nvidia_gpus": [],
            }
        index, name, memory_total_mib, driver_version = fields
        gpus.append(
            {
                "index": _optional_non_negative_int(index),
                "name": name or None,
                "memory_total_mib": _optional_non_negative_int(memory_total_mib),
                "driver_version": driver_version or None,
            }
        )
    return {"nvidia_smi_status": "captured", "nvidia_gpus": gpus}


def _total_memory_mib() -> int | None:
    """Return total system memory when the platform exposes page counts."""

    try:
        page_count = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        return None
    if not isinstance(page_count, int) or not isinstance(page_size, int):
        return None
    if page_count <= 0 or page_size <= 0:
        return None
    return page_count * page_size // (1024 * 1024)


def _optional_non_negative_int(value: str) -> int | None:
    """Parse a non-negative integer field without rejecting all provenance."""

    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed >= 0 else None


def _run_command(
    arguments: Sequence[str],
) -> subprocess.CompletedProcess[str]:
    """Run one bounded read-only discovery command."""

    return subprocess.run(
        list(arguments),
        check=True,
        capture_output=True,
        text=True,
        timeout=COMMAND_TIMEOUT_SECONDS,
    )


def _command_error_type(error: BaseException) -> str:
    """Normalize discovery failures without persisting machine-specific text."""

    if isinstance(error, FileNotFoundError):
        return "executable_not_found"
    if isinstance(error, subprocess.TimeoutExpired):
        return "timeout"
    if isinstance(error, subprocess.CalledProcessError):
        return "command_failed"
    return "os_error"
