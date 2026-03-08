"""
Shared utilities used across pipeline and experiment modules.

Centralised here to eliminate duplication of ``detect_language``,
``extract_code_from_response``, ``LLMClient``, and ``LLMError``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

import requests

from pipeline.configs import config


# ============================================================================
# LANGUAGE DETECTION
# ============================================================================

#: Canonical mapping from file extension to language name.
EXTENSION_MAP: dict[str, str] = {
    ".py": "python",
    ".java": "java",
    ".go": "go",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".scala": "scala",
    ".rs": "rust",
    ".rb": "ruby",
    ".php": "php",
    ".c": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".swift": "swift",
    ".m": "objective-c",
    ".mm": "objective-cpp",
    ".vue": "vue",
}


def detect_language(file_path: Path | str) -> str:
    """Detect programming language from file extension.

    Parameters
    ----------
    file_path:
        A ``Path`` object or string representing the file path.

    Returns
    -------
    str
        Lower-case language name, or ``'unknown'`` if the extension is not
        recognised.
    """
    suffix = Path(file_path).suffix.lower()
    return EXTENSION_MAP.get(suffix, "unknown")


# ============================================================================
# CODE EXTRACTION FROM LLM RESPONSES
# ============================================================================

def extract_code_from_response(response: str, language: str) -> Optional[str]:
    """Extract a fenced code block from an LLM response.

    Tries language-specific fences first (e.g. ````python`), then a
    partial-language match, and finally a bare fence.  Returns ``None`` if no
    fenced block can be found — the caller should treat this as a parse
    failure rather than guessing.

    Parameters
    ----------
    response:
        Raw text returned by the LLM.
    language:
        Expected language tag (e.g. ``"python"``, ``"java"``).
    """
    patterns = [
        rf"```{re.escape(language)}\s*\n(.*?)```",
        rf"```{re.escape(language[:4])}\s*\n(.*?)```",  # partial match
        r"```\s*\n(.*?)```",  # generic fence
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

    return None


# ============================================================================
# LLM CLIENT
# ============================================================================

class LLMError(Exception):
    """Exception for LLM API errors."""
    pass


class LLMClient:
    """Client for OpenAI-compatible or Ollama chat completions API."""

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        provider: str | None = None,
    ):
        """
        Initialise the LLM client.

        Args:
            base_url: Base URL for API.
            api_key: Optional API key for authentication.
            provider: Optional provider hint (e.g., ``"ollama"``).
        """
        base_url = base_url.strip().rstrip("/")
        if not base_url:
            raise ValueError("LLM_BASE_URL must be configured")

        provider = (provider or "").strip().lower()
        use_ollama = provider == "ollama" or "11434" in base_url
        self.provider = "ollama" if use_ollama else "openai"

        if self.provider == "ollama":
            if base_url.endswith("/v1"):
                base_url = base_url[:-3]
            self.endpoint = f"{base_url}/api/chat"
        else:
            if not base_url.endswith("/v1"):
                base_url = f"{base_url}/v1"
            self.endpoint = f"{base_url}/chat/completions"

        self.headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

        self.timeout = config.LLM_TIMEOUT_S

    def complete(
        self,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int = 1000,
        temperature: float = 0.1,
    ) -> dict[str, Any]:
        """
        Send a chat completion request.

        Args:
            model: Model identifier.
            messages: List of message dicts with ``'role'`` and ``'content'``.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (low = deterministic).

        Returns:
            API response dict.

        Raises:
            LLMError: On API failures.
        """
        if self.provider == "ollama":
            payload: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "format": "json",
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            }
        else:
            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }

        try:
            response = requests.post(
                self.endpoint,
                headers=self.headers,
                json=payload,
                timeout=self.timeout,
            )
        except requests.exceptions.Timeout:
            raise LLMError("Request timed out")
        except requests.exceptions.ConnectionError as e:
            raise LLMError(f"Connection failed: {e}")

        if response.status_code >= 400:
            raise LLMError(f"API error ({response.status_code}): {response.text[:500]}")

        try:
            return response.json()
        except ValueError:
            raise LLMError(f"Invalid JSON response: {response.text[:500]}")

    def extract_content(self, response: dict) -> str:
        """Extract text content from API response."""
        if "choices" in response:
            choices = response.get("choices", [])
            if not choices:
                return ""
            message = choices[0].get("message", {})
            return message.get("content", "") or choices[0].get("text", "")
        message = response.get("message", {})
        return message.get("content", "")
