"""Provider adapters."""

from detecting_an_insider_llm.providers.ollama_client import (
    OllamaClient,
    OllamaClientError,
)

__all__ = ["OllamaClient", "OllamaClientError"]
