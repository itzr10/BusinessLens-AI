"""Provider adapters and the registry that builds them from a per-session config."""

from __future__ import annotations

from dataclasses import dataclass

from businesslens.providers.base import (
    ErrorKind, Provider, ProviderConfig, ProviderError, ToolCall, ToolResult, ToolSpec, Turn,
    clean_config, redact,
)
from businesslens.providers.urlcheck import validate_base_url

__all__ = [
    "ErrorKind", "Provider", "ProviderConfig", "ProviderError", "ProviderInfo", "PROVIDERS",
    "ToolCall", "ToolResult", "ToolSpec", "Turn", "create_provider", "redact",
    "validate_base_url",
]


@dataclass(frozen=True)
class ProviderInfo:
    id: str
    label: str
    default_model: str
    supports_base_url: bool
    key_help: str


PROVIDERS: dict[str, ProviderInfo] = {
    "gemini": ProviderInfo("gemini", "Google Gemini", "gemini-3.8-flash", False,
                           "Create a key in Google AI Studio."),
    "openai": ProviderInfo("openai", "OpenAI and compatible endpoints (experimental, not live-verified)", "gpt-5-mini", True,
                           "Create a key in the OpenAI dashboard. For other vendors, "
                           "use their key and their https base URL."),
}


def create_provider(config: ProviderConfig, client=None) -> Provider:
    """Validate ``config`` and build the adapter. ``client`` injects a fake SDK client in tests."""
    info = PROVIDERS.get(config.provider)
    if info is None:
        raise ProviderError(ErrorKind.INVALID_CONFIG, "Unknown provider.")
    config = clean_config(config)
    if config.base_url:
        if not info.supports_base_url:
            raise ProviderError(ErrorKind.INVALID_CONFIG,
                                f"{info.label} does not support a custom base URL.")
        config = ProviderConfig(config.provider, config.model, config.api_key,
                                validate_base_url(config.base_url))
    if config.provider == "gemini":
        from businesslens.providers.gemini import GeminiProvider
        return GeminiProvider(config, client)
    from businesslens.providers.openai_provider import OpenAIProvider
    return OpenAIProvider(config, client)
