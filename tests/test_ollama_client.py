"""Offline unit tests for the Ollama provider adapter.

The HTTP boundary is replaced at `requests.Session.request`. This is the
narrowest boundary that still exercises configuration resolution, payload
construction, response normalization, metadata extraction, and public error
translation. No test requires an Ollama daemon, model download, credentials, or
network access.
"""

from typing import Any

import pytest
import requests

from detecting_an_insider_llm.providers.ollama_client import (
    OllamaClient,
    OllamaClientError,
)


class StubResponse:
    """Small response double implementing the methods used by OllamaClient."""

    def __init__(self, body: object, *, status_code: int = 200) -> None:
        self._body = body
        self.status_code = status_code

    def raise_for_status(self) -> None:
        """Model the requests behavior needed for HTTP-status validation."""
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self) -> object:
        """Return the predetermined decoded response body."""
        return self._body


def test_chat_sends_non_streaming_payload_and_normalizes_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Chat sends the reproducible payload and retains normalized plus raw data."""
    # Arrange: use a real Session object but replace only its request method.
    # This keeps the test compatible with the production constructor while
    # guaranteeing that no HTTP connection can be opened.
    session = requests.Session()
    captured: dict[str, Any] = {}
    response_body = {
        "model": "test-model",
        "message": {"role": "assistant", "content": "done"},
        "done": True,
        "done_reason": "stop",
    }

    def fake_request(method: str, url: str, **kwargs: Any) -> StubResponse:
        # Capture every transport argument so the assertion detects accidental
        # changes to endpoints, timeouts, authentication, or generation payload.
        captured.update({"method": method, "url": url, **kwargs})
        return StubResponse(response_body)

    monkeypatch.setattr(session, "request", fake_request)
    client = OllamaClient(
        model="test-model",
        base_url="http://ollama.test/",
        timeout_seconds=30,
        keep_alive="10m",
        session=session,
    )

    # Act: include tools, sampling options, and a named thinking level because
    # these are important experimental settings that must reach Ollama unchanged.
    result = client.chat(
        [{"role": "user", "content": "hello"}],
        tools=[{"type": "function", "function": {"name": "read_email"}}],
        options={"temperature": 0.4, "seed": 7},
        think="medium",
    )

    # Assert the shared runtime view and the retained provider evidence.
    assert result.message == {"role": "assistant", "content": "done"}
    assert result.finish_reason == "complete"
    assert result.raw_response == response_body

    # Exact payload comparison is intentional: silently dropping `stream=False`,
    # a seed, or a tool schema would change the executed protocol.
    assert captured == {
        "method": "POST",
        "url": "http://ollama.test/api/chat",
        "json": {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
            "keep_alive": "10m",
            "think": "medium",
            "tools": [
                {"type": "function", "function": {"name": "read_email"}}
            ],
            "options": {"temperature": 0.4, "seed": 7},
        },
        "headers": None,
        "timeout": 30.0,
    }

    # The client does not own an injected session, so the test/caller closes it.
    session.close()


@pytest.mark.parametrize(
    ("done", "done_reason", "expected"),
    [
        (True, "length", "length"),
        (True, "future_reason", "unknown"),
        (True, None, "unknown"),
        (False, "stop", "unknown"),
    ],
)
def test_chat_normalizes_nonstandard_finish_reasons_without_guessing_completion(
    monkeypatch: pytest.MonkeyPatch,
    done: bool,
    done_reason: object,
    expected: str,
) -> None:
    """Unknown or unfinished provider states must not become normal completion."""

    session = requests.Session()
    response_body = {
        "model": "test-model",
        "message": {"role": "assistant", "content": "partial"},
        "done": done,
    }
    if done_reason is not None:
        response_body["done_reason"] = done_reason

    monkeypatch.setattr(
        session,
        "request",
        lambda *_args, **_kwargs: StubResponse(response_body),
    )
    client = OllamaClient(model="test-model", session=session)

    result = client.chat([{"role": "user", "content": "hello"}])

    assert result.finish_reason == expected
    assert result.raw_response == response_body
    session.close()


def test_client_requires_a_model_when_environment_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Construction fails early when no explicit or environment model exists."""
    # monkeypatch restores the process environment after the test, keeping tests
    # isolated from each other and from the developer's shell configuration.
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)

    with pytest.raises(ValueError, match="OLLAMA_MODEL"):
        OllamaClient()


def test_timeout_is_translated_to_provider_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A transport timeout becomes the stable provider-facing exception type."""
    session = requests.Session()

    def timeout(*args: Any, **kwargs: Any) -> StubResponse:
        # Raising here proves the request never leaves the offline test boundary.
        raise requests.Timeout

    monkeypatch.setattr(session, "request", timeout)
    client = OllamaClient(
        model="test-model",
        base_url="http://ollama.test",
        timeout_seconds=2,
        session=session,
    )

    # Runtime code should catch OllamaClientError rather than depend on the
    # requests library, while the message preserves the configured timeout.
    with pytest.raises(OllamaClientError, match="timed out after 2 seconds"):
        client.chat([{"role": "user", "content": "hello"}])

    session.close()


def test_runtime_metadata_is_cached_and_includes_matching_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Metadata records provenance once and does not repeat hidden HTTP calls."""
    session = requests.Session()
    requested_urls: list[str] = []

    def fake_request(method: str, url: str, **kwargs: Any) -> StubResponse:
        requested_urls.append(url)
        if url.endswith("/version"):
            return StubResponse({"version": "1.2.3"})
        return StubResponse(
            {
                "models": [
                    {
                        "name": "test-model",
                        "model": "test-model",
                        "digest": "abc123",
                        "details": {"family": "test"},
                    }
                ]
            }
        )

    monkeypatch.setattr(session, "request", fake_request)
    client = OllamaClient(
        model="test-model",
        base_url="http://ollama.test",
        timeout_seconds=45,
        keep_alive=0,
        session=session,
    )

    # Act twice: the second call should return a defensive copy of the cache.
    first_result = client.runtime_metadata()
    second_result = client.runtime_metadata()

    # These fields establish which runtime and model artifact produced a run.
    assert first_result == {
        "base_url": "http://ollama.test",
        "model": "test-model",
        "model_kind": "local",
        "request_timeout_seconds": 45.0,
        "keep_alive": 0,
        "model_metadata_status": "captured",
        "ollama_version": "1.2.3",
        "resolved_model": "test-model",
        "model_digest": "abc123",
        "model_details": {"family": "test"},
    }
    assert second_result == first_result

    # Exactly two URLs proves caching prevents extra, unrecorded metadata calls.
    assert requested_urls == [
        "http://ollama.test/api/version",
        "http://ollama.test/api/tags",
    ]
    session.close()


def test_runtime_metadata_retains_missing_model_digest_explicitly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tags miss keeps identifier and null digest instead of omitting keys."""

    session = requests.Session()

    def fake_request(method: str, url: str, **kwargs: Any) -> StubResponse:
        if url.endswith("/version"):
            return StubResponse({"version": "1.2.3"})
        return StubResponse({"models": []})

    monkeypatch.setattr(session, "request", fake_request)
    client = OllamaClient(model="remote-model:cloud", session=session)

    metadata = client.runtime_metadata()

    assert metadata["model"] == "remote-model:cloud"
    assert metadata["model_kind"] == "cloud"
    assert metadata["resolved_model"] is None
    assert metadata["model_digest"] is None
    assert metadata["model_details"] is None
    assert metadata["model_metadata_status"] == "model_not_found"
    assert metadata["request_timeout_seconds"] == 120.0
    assert metadata["keep_alive"] == "5m"
    session.close()


def test_runtime_metadata_failure_preserves_config_and_null_model_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failed enrichment remains visible without dropping configured settings."""

    session = requests.Session()
    monkeypatch.setattr(
        session,
        "request",
        lambda *_args, **_kwargs: StubResponse({}, status_code=503),
    )
    client = OllamaClient(
        model="test-model",
        timeout_seconds=9,
        keep_alive="1m",
        session=session,
    )

    metadata = client.runtime_metadata()

    assert metadata["model"] == "test-model"
    assert metadata["request_timeout_seconds"] == 9.0
    assert metadata["keep_alive"] == "1m"
    assert metadata["resolved_model"] is None
    assert metadata["model_digest"] is None
    assert metadata["model_details"] is None
    assert metadata["model_metadata_status"] == "unavailable"
    assert metadata["metadata_error_type"] == "OllamaClientError"
    assert "HTTP 503" in metadata["metadata_error"]
    session.close()
