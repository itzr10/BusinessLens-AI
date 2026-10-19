"""Provider-neutral types for talking to an AI model.

A *provider adapter* turns one vendor SDK into this small interface. Adapters
only move messages: they never run tools and never see the dataframe. The
agent loop in ``businesslens/agent.py`` owns tool dispatch and limits.

Credentials live in a ``ProviderConfig`` that the caller creates per user
session. Nothing in this package keeps a key in a module-level variable.
"""

from __future__ import annotations

import enum
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

# Bounds applied to every request, whatever the provider.
REQUEST_TIMEOUT_SECONDS = 30.0
MAX_OUTPUT_TOKENS = 1024
MAX_KEY_CHARS = 512
MAX_MODEL_CHARS = 128


class ErrorKind(str, enum.Enum):
    MISSING_CREDENTIALS = "missing_credentials"
    INVALID_CREDENTIALS = "invalid_credentials"
    PERMISSION = "permission_denied"
    MODEL_NOT_FOUND = "model_not_found"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    CONNECTION = "connection"
    TOOLS_UNSUPPORTED = "tools_unsupported"
    BAD_REQUEST = "bad_request"
    INVALID_CONFIG = "invalid_config"
    PROVIDER = "provider_error"
    EMPTY_RESPONSE = "empty_response"


_USER_MESSAGES = {
    ErrorKind.MISSING_CREDENTIALS: "No API key is set. Open Settings and enter your key.",
    ErrorKind.INVALID_CREDENTIALS: "The provider rejected the API key. Check the key and the selected provider.",
    ErrorKind.PERMISSION: "The key is not allowed to use this model or API (permission denied).",
    ErrorKind.MODEL_NOT_FOUND: "The provider does not know this model ID, or your key cannot use it.",
    ErrorKind.RATE_LIMIT: "The provider's rate limit or quota was reached. Wait a minute, or check your plan and billing.",
    ErrorKind.TIMEOUT: "The provider did not answer in time. Try again.",
    ErrorKind.CONNECTION: "Could not reach the provider. Check your network and base URL.",
    ErrorKind.TOOLS_UNSUPPORTED: "This model or endpoint does not support tool (function) calling, which BusinessLens needs. Choose another model.",
    ErrorKind.BAD_REQUEST: "The provider rejected the request. Check the model ID.",
    ErrorKind.INVALID_CONFIG: "The provider settings are invalid.",
    ErrorKind.PROVIDER: "The provider returned an error. Try again later.",
    ErrorKind.EMPTY_RESPONSE: "The model returned an empty answer. Try again.",
}


class ProviderError(RuntimeError):
    """A user-presentable provider failure. The message never contains secrets."""

    def __init__(self, kind: ErrorKind, detail: str = ""):
        self.kind = kind
        message = _USER_MESSAGES[kind]
        super().__init__(f"{message} {detail}".strip())


@dataclass(frozen=True)
class ProviderConfig:
    """One user's provider settings. ``api_key`` is hidden from repr/str."""

    provider: str
    model: str
    api_key: str = field(default="", repr=False)
    base_url: str | None = None

    def __repr__(self) -> str:  # explicit: never show the key, even by accident
        return (f"ProviderConfig(provider={self.provider!r}, model={self.model!r}, "
                f"api_key={'<set>' if self.api_key else '<empty>'}, base_url={self.base_url!r})")

    __str__ = __repr__


@dataclass(frozen=True)
class ToolSpec:
    """Provider-neutral tool declaration (JSON-schema parameters)."""

    name: str
    description: str
    properties: dict[str, Any]
    required: tuple[str, ...] = ()


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    args: dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    call: ToolCall
    result: dict[str, Any]  # JSON-safe; {"error": "..."} when the call failed


@dataclass
class Turn:
    """One model reply: final text, or approved-tool requests (or both)."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


class Conversation(ABC):
    """Opaque, provider-specific message history. Created by ``Provider.start``."""


class Provider(ABC):
    """The interface every adapter implements.

    ``start`` opens a conversation, ``generate`` sends it to the model and
    returns a ``Turn``, and ``add_tool_results`` appends the model's request and
    our tool results so the next ``generate`` can produce the final answer.
    """

    name: str = ""

    @abstractmethod
    def start(self, system: str, user_text: str, tools: list[ToolSpec]) -> Conversation: ...

    @abstractmethod
    def generate(self, conversation: Conversation) -> Turn: ...

    @abstractmethod
    def add_tool_results(self, conversation: Conversation, turn: Turn,
                         results: list[ToolResult]) -> None: ...


# --------------------------------------------------------------------------
# Secret handling
# --------------------------------------------------------------------------

_KEY_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{8,}"),       # OpenAI-style (incl. sk-proj-...)
    re.compile(r"AIza[A-Za-z0-9_\-]{20,}"),     # Google API keys
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{8,}"),
]


def redact(text: str, *secrets: str) -> str:
    """Remove the given secrets and anything that looks like an API key from text."""
    for secret in secrets:
        if secret and len(secret) >= 4:
            text = text.replace(secret, "[REDACTED]")
    for pattern in _KEY_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    return text


def clean_config(config: ProviderConfig) -> ProviderConfig:
    """Trim and sanity-check the fields (no network). Raises ProviderError."""
    key = (config.api_key or "").strip()
    model = (config.model or "").strip()
    if not key:
        raise ProviderError(ErrorKind.MISSING_CREDENTIALS)
    if len(key) > MAX_KEY_CHARS or any(c.isspace() for c in key) or not key.isascii():
        raise ProviderError(ErrorKind.INVALID_CONFIG, "The API key has an invalid format.")
    if not model or len(model) > MAX_MODEL_CHARS or not re.fullmatch(r"[A-Za-z0-9._:/\-]+", model):
        raise ProviderError(ErrorKind.INVALID_CONFIG, "Enter a valid model ID.")
    base = (config.base_url or "").strip() or None
    return ProviderConfig(config.provider, model, key, base)
