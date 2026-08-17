"""Synchronous Ollama chat provider.

Why use the HTTP API directly?
------------------------------
The adapter retains Ollama's complete decoded response without translating it
through SDK-specific response classes. That raw provider object can later be
written to the run journal while the runtime consumes only the normalized
assistant message.

Configuration values may be passed explicitly or read from environment
variables. Explicit constructor values take precedence. Loading a particular
`.env` path is intentionally left to the application/CLI layer; the provider
must not assume a repository layout or mutate process-wide configuration.

The client does not retry requests. Hidden retries would make one apparent
provider call represent several unrecorded attempts. A future runner can add a
bounded retry policy at the level where every attempt can be journaled.
"""

import os
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

import requests

from detecting_an_insider_llm.runtime.agents import (
    ProviderResponse,
    ThinkingMode,
)

# Defaults match a normal local Ollama installation while remaining overridable
# for remote or test servers.
DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_KEEP_ALIVE = "5m"


class OllamaClientError(RuntimeError):
    """Raised when Ollama cannot return a usable response.

    Transport, HTTP, and response-decoding failures share one public exception so
    the provider-neutral runtime does not need to import `requests` exceptions.
    The original exception is preserved as `__cause__` for diagnosis.
    """


class OllamaClient:
    """Provider adapter for Ollama's non-streaming `/api/chat` endpoint.

    Args:
        model:
            Exact Ollama model identifier. When omitted, `OLLAMA_MODEL` is used.
            Unlike the other settings, a model has no safe implicit default.
        base_url:
            Ollama server root. Precedence is constructor value,
            `OLLAMA_BASE_URL`, then :data:`DEFAULT_BASE_URL`.
        timeout_seconds:
            Positive request timeout. Precedence is constructor value,
            `OLLAMA_TIMEOUT_SECONDS`, then the default.
        keep_alive:
            Ollama duration string (for example `"5m"`) or a non-negative
            integer. Zero asks Ollama to unload the model immediately.
        api_key:
            Optional bearer token for the hosted Ollama API. The
            `OLLAMA_API_KEY` environment variable is used when omitted.
        session:
            Optional `requests.Session` transport. Injection keeps tests offline
            and allows callers to provide custom adapters.

    The client owns and closes only sessions that it creates itself.
    """

    def __init__(
        self,
        model: str | None = None,
        *,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        keep_alive: str | int | None = None,
        api_key: str | None = None,
        session: requests.Session | None = None,
    ) -> None:
        # Resolve and validate configuration once. Every later request therefore
        # uses the same model and transport settings, which supports reproducible
        # run metadata.
        self._model = _required_text(model, "OLLAMA_MODEL")
        self._base_url = _base_url(base_url)
        self._timeout_seconds = _timeout(timeout_seconds)
        self._keep_alive = _keep_alive(keep_alive)

        # Local Ollama needs no authentication. A header is added only when a
        # direct hosted-API key is explicitly provided or exported.
        configured_api_key = api_key if api_key is not None else os.getenv("OLLAMA_API_KEY")
        resolved_api_key = configured_api_key.strip() if configured_api_key else ""
        self._headers = (
            {"Authorization": f"Bearer {resolved_api_key}"}
            if resolved_api_key
            else None
        )

        # Injecting a session gives tests and callers control over the transport.
        # Only sessions created here are closed by this client.
        self._session = session or requests.Session()
        self._owns_session = session is None

        # Metadata requires two extra HTTP calls. Cache the result so repeatedly
        # reading run metadata cannot change it or add untracked network traffic.
        self._runtime_metadata_cache: dict[str, Any] | None = None

    @property
    def provider_name(self) -> str:
        """Return the stable name used in run metadata."""
        return "ollama"

    @property
    def model_name(self) -> str:
        """Return the exact model identifier sent to Ollama."""
        return self._model

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
        options: Mapping[str, Any] | None = None,
        *,
        think: ThinkingMode = False,
    ) -> ProviderResponse:
        """Generate one complete assistant message.

        Args:
            messages:
                Ordered conversation messages in Ollama's chat format.
            tools:
                Optional JSON tool schemas advertised to the model. This method
                never executes a returned tool call.
            options:
                Ollama generation options such as `temperature`, `top_k`,
                `top_p`, `seed`, or `num_predict`.
            think:
                Boolean or supported named reasoning effort.

        Returns:
            A :class:`ProviderResponse` containing both the normalized assistant
            message and the complete decoded Ollama response.

        Raises:
            ValueError:
                If messages are empty or the thinking mode is unsupported.
            OllamaClientError:
                If transport, HTTP, JSON decoding, or response validation fails.

        Streaming is disabled so one provider call maps to one durable response
        record. A higher-level tool loop can append returned tool calls and tool
        results before making the next call.
        """
        if not messages:
            raise ValueError("messages must contain at least one message.")

        # Deep copies prevent a transport mock or serializer from mutating the
        # agent's committed conversation and configured tool definitions.
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": deepcopy(list(messages)),
            "stream": False,
            "keep_alive": self._keep_alive,
            "think": _thinking_mode(think),
        }
        if tools is not None:
            payload["tools"] = deepcopy(list(tools))
        if options is not None:
            payload["options"] = deepcopy(dict(options))

        raw_response = self._request_json("POST", "chat", payload)

        # Normalize only the field required by Agent. All other provider fields
        # remain available in raw_response for provenance and later analysis.
        message = raw_response.get("message")
        if not isinstance(message, dict):
            raise OllamaClientError("Ollama response is missing a message object.")

        return ProviderResponse(
            message=deepcopy(message),
            raw_response=deepcopy(raw_response),
        )

    def runtime_metadata(self) -> dict[str, Any]:
        """Return best-effort provider and model provenance.

        Ollama's version endpoint identifies the runtime. The tags endpoint may
        add the resolved model name, digest, and family details. Metadata failure
        is captured inside the returned object instead of blocking the primary
        experiment, but the failure remains explicit rather than silently
        disappearing.
        """
        if self._runtime_metadata_cache is not None:
            return deepcopy(self._runtime_metadata_cache)

        metadata: dict[str, Any] = {
            "base_url": self._base_url,
            "model": self._model,
            "model_kind": (
                "cloud" if self._model.casefold().endswith(":cloud") else "local"
            ),
        }
        try:
            version_response = self._request_json("GET", "version")
            tags_response = self._request_json("GET", "tags")
            metadata["ollama_version"] = version_response.get("version")
            metadata.update(_matching_model_metadata(self._model, tags_response))
        except OllamaClientError as exc:
            # Metadata enrichment must not hide the primary model interaction.
            # The error is retained so the missing provenance remains visible.
            metadata["metadata_error_type"] = type(exc).__name__
            metadata["metadata_error"] = str(exc)

        self._runtime_metadata_cache = metadata
        return deepcopy(metadata)

    def close(self) -> None:
        """Close the HTTP session only when this client created it."""
        if self._owns_session:
            self._session.close()

    def __enter__(self) -> "OllamaClient":
        """Allow deterministic cleanup with `with OllamaClient(...) as client`."""
        return self

    def __exit__(self, *_: object) -> None:
        """Release the internally owned session when leaving a context block."""
        self.close()

    def _request_json(
        self,
        method: str,
        endpoint: str,
        payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Send one HTTP request and return a validated JSON object.

        This is the transport boundary: provider-independent code sees
        `OllamaClientError`, while the original `requests` exception remains
        chained for debugging.
        """
        url = f"{self._base_url}/api/{endpoint.lstrip('/')}"

        # No retry occurs here. The research runner must own retry accounting so
        # every attempted provider call can be represented in the journal.
        try:
            response = self._session.request(
                method,
                url,
                json=deepcopy(dict(payload)) if payload is not None else None,
                headers=self._headers,
                timeout=self._timeout_seconds,
            )
        except requests.Timeout as exc:
            raise OllamaClientError(
                f"Ollama request timed out after {self._timeout_seconds:g} seconds."
            ) from exc
        except requests.ConnectionError as exc:
            raise OllamaClientError(
                f"Could not connect to Ollama at {self._base_url}."
            ) from exc
        except requests.RequestException as exc:
            raise OllamaClientError("Ollama request failed.") from exc

        # Separate status validation from transport failures so error messages
        # preserve whether Ollama was reachable but rejected the request.
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise OllamaClientError(
                f"Ollama returned HTTP {response.status_code}."
            ) from exc

        try:
            result = response.json()
        except ValueError as exc:
            raise OllamaClientError("Ollama returned invalid JSON.") from exc
        if not isinstance(result, dict):
            raise OllamaClientError("Ollama returned a non-object JSON response.")
        return result


def _required_text(explicit_value: str | None, environment_name: str) -> str:
    """Resolve a required string using explicit-value-first precedence."""
    configured_value = (
        explicit_value if explicit_value is not None else os.getenv(environment_name, "")
    )
    resolved_value = configured_value.strip()
    if not resolved_value:
        raise ValueError(
            f"Provide a value or configure the {environment_name} environment variable."
        )
    return resolved_value


def _base_url(explicit_value: str | None) -> str:
    """Resolve the server root and remove one source of malformed API URLs."""
    configured_value = (
        explicit_value
        if explicit_value is not None
        else os.getenv("OLLAMA_BASE_URL", DEFAULT_BASE_URL)
    )
    resolved_value = configured_value.strip().rstrip("/")
    if not resolved_value:
        raise ValueError("OLLAMA_BASE_URL must contain a URL.")
    return resolved_value


def _timeout(explicit_value: float | None) -> float:
    """Resolve and validate the timeout before any network activity occurs."""
    configured_value: object = (
        explicit_value
        if explicit_value is not None
        else os.getenv("OLLAMA_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS))
    )
    if isinstance(configured_value, bool):
        raise ValueError("OLLAMA_TIMEOUT_SECONDS must be a positive number.")
    try:
        resolved_value = float(configured_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("OLLAMA_TIMEOUT_SECONDS must be a positive number.") from exc
    if resolved_value <= 0:
        raise ValueError("OLLAMA_TIMEOUT_SECONDS must be greater than zero.")
    return resolved_value


def _keep_alive(explicit_value: str | int | None) -> str | int:
    """Resolve Ollama's model residency setting and reject ambiguous booleans."""
    configured_value: object = (
        explicit_value
        if explicit_value is not None
        else os.getenv("OLLAMA_KEEP_ALIVE", DEFAULT_KEEP_ALIVE)
    )
    if isinstance(configured_value, bool) or not isinstance(configured_value, (str, int)):
        raise ValueError("OLLAMA_KEEP_ALIVE must be a duration string or integer.")
    if isinstance(configured_value, str):
        resolved_value = configured_value.strip()
        if not resolved_value:
            raise ValueError("OLLAMA_KEEP_ALIVE must not be empty.")
        return resolved_value
    if configured_value < 0:
        raise ValueError("OLLAMA_KEEP_ALIVE must be zero or greater.")
    return configured_value


def _thinking_mode(value: ThinkingMode) -> ThinkingMode:
    """Validate the provider-neutral reasoning setting before serialization."""
    if value not in {True, False, "high", "medium", "low"}:
        raise ValueError("think must be true, false, 'high', 'medium', or 'low'.")
    return value


def _matching_model_metadata(
    model: str,
    tags_response: Mapping[str, Any],
) -> dict[str, Any]:
    """Extract provenance only for the exact configured model identifier."""
    models = tags_response.get("models")
    if not isinstance(models, list):
        return {}

    for item in models:
        if not isinstance(item, dict):
            continue
        known_names = {
            str(item.get("name") or ""),
            str(item.get("model") or ""),
        }
        if model in known_names:
            return {
                "resolved_model": item.get("model") or item.get("name"),
                "model_digest": item.get("digest"),
                "model_details": deepcopy(item.get("details")),
            }
    return {}
