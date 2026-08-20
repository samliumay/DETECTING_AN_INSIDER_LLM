"""Offline tests for code and execution-host provenance capture."""

import subprocess
from pathlib import Path

import pytest

import detecting_an_insider_llm.runtime.provenance as provenance_module
from detecting_an_insider_llm.runtime.provenance import (
    capture_operational_provenance,
)


def _completed(
    arguments: list[str],
    *,
    stdout: str,
) -> subprocess.CompletedProcess[str]:
    """Build a successful read-only command result."""

    return subprocess.CompletedProcess(arguments, 0, stdout=stdout, stderr="")


def test_capture_records_code_each_run_and_hardware_once_per_process(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Code can change per run, while unchanged host probing remains cached."""

    calls: list[tuple[str, ...]] = []

    def fake_run(arguments: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(tuple(arguments))
        if arguments[0] == "git" and arguments[-2:] == ["rev-parse", "HEAD"]:
            return _completed(arguments, stdout="abc123\n")
        if arguments[0] == "git":
            return _completed(arguments, stdout=" M README.md\n")
        return _completed(
            arguments,
            stdout="0, NVIDIA Test GPU, 24564, 999.1\n",
        )

    monkeypatch.setattr(provenance_module, "_run_command", fake_run)
    monkeypatch.setattr(provenance_module.platform, "system", lambda: "TestOS")
    monkeypatch.setattr(provenance_module.platform, "release", lambda: "1.0")
    monkeypatch.setattr(provenance_module.platform, "machine", lambda: "test-arch")
    monkeypatch.setattr(
        provenance_module.platform,
        "python_version",
        lambda: "3.13.0",
    )
    monkeypatch.setattr(provenance_module.os, "cpu_count", lambda: 8)
    monkeypatch.setattr(provenance_module, "_cpu_model", lambda: "Test CPU")
    monkeypatch.setattr(provenance_module, "_total_memory_mib", lambda: 32768)
    provenance_module._cached_execution_host_metadata.cache_clear()

    first = capture_operational_provenance(tmp_path)
    second = capture_operational_provenance(tmp_path)

    assert first["code"] == {
        "capture_status": "captured",
        "git_revision": "abc123",
        "git_dirty": True,
    }
    assert second["code"] == first["code"]
    assert first["execution_host"] == second["execution_host"]
    assert first["execution_host"]["capture_scope"] == "python_process"
    assert first["execution_host"]["cpu_model"] == "Test CPU"
    assert first["execution_host"]["total_memory_mib"] == 32768
    assert first["execution_host"]["nvidia_gpus"] == [
        {
            "index": 0,
            "name": "NVIDIA Test GPU",
            "memory_total_mib": 24564,
            "driver_version": "999.1",
        }
    ]
    assert len(first["execution_host"]["snapshot_id"]) == 64
    assert sum(call[0] == "git" for call in calls) == 4
    assert sum(call[0] == "nvidia-smi" for call in calls) == 1

    provenance_module._cached_execution_host_metadata.cache_clear()


def test_capture_makes_unavailable_discovery_explicit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Missing discovery executables produce null evidence and status fields."""

    def missing_command(arguments: list[str]) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError(arguments[0])

    monkeypatch.setattr(provenance_module, "_run_command", missing_command)
    monkeypatch.setattr(provenance_module, "_cpu_model", lambda: None)
    monkeypatch.setattr(provenance_module, "_total_memory_mib", lambda: None)
    provenance_module._cached_execution_host_metadata.cache_clear()

    captured = capture_operational_provenance(tmp_path)

    assert captured["code"] == {
        "capture_status": "unavailable",
        "git_revision": None,
        "git_dirty": None,
        "error_type": "executable_not_found",
    }
    assert captured["execution_host"]["nvidia_smi_status"] == "unavailable"
    assert (
        captured["execution_host"]["nvidia_smi_error_type"]
        == "executable_not_found"
    )
    assert captured["execution_host"]["nvidia_gpus"] == []

    provenance_module._cached_execution_host_metadata.cache_clear()
