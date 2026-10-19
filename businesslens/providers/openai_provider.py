"""OpenAI adapter (Chat Completions API with function tools).

Chat Completions is used, rather than the newer Responses API, because most
"OpenAI-compatible" endpoints implement it. An optional ``base_url`` is
accepted only after ``urlcheck.validate_base_url`` and redirects are disabled.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from businesslens.providers.base import (
    MAX_OUTPUT_TOKENS, REQUEST_TIMEOUT_SECONDS, Conversation, ErrorKind, Provider,
    ProviderConfig, ProviderError, ToolCall, ToolResult, ToolSpec, Turn, redact,
)

log = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-5-mini"


@dataclass
class _OpenAIConversation(Conversation):
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    last_message: dict[str, Any] | None = None


def classify_error(exc: BaseException) -> ErrorKind:
    import openai

    text = str(exc).lower()
    if isinstance(exc, openai.APITimeoutError):
        return ErrorKind.TIMEOUT
    if isinstance(exc, openai.APIConnectionError):
        return ErrorKind.CONNECTION
    if isinstance(exc, openai.AuthenticationError):
        return ErrorKind.INVALID_CREDENTIALS
    if isinstance(exc, openai.PermissionDeniedError):
        return ErrorKind.PERMISSION
    if isinstance(exc, openai.RateLimitError):
        return ErrorKind.RATE_LIMIT
    if isinstance(exc, openai.NotFoundError) or getattr(exc, "code", None) == "model_not_found":
        return ErrorKind.MODEL_NOT_FOUND
    if isinstance(exc, openai.BadRequestError):
        if "tool" in text or "function" in text:
            return ErrorKind.TOOLS_UNSUPPORTED
        return ErrorKind.BAD_REQUEST
    if isinstance(exc, openai.APIStatusError) and exc.status_code >= 500:
        return ErrorKind.PROVIDER
    return ErrorKind.PROVIDER


class OpenAIProvider(Provider):
    name = "openai"

    def __init__(self, config: ProviderConfig, client: Any = None):
        self._model = config.model
        self._key = config.api_key
        self._custom_base = bool(config.base_url)
        if client is None:
            import httpx2
            import openai

            client = openai.OpenAI(
                api_key=config.api_key,
                base_url=config.base_url,  # already validated by the registry
                timeout=REQUEST_TIMEOUT_SECONDS,
                max_retries=0,  # no hidden retries: every API call is counted by the agent
                http_client=httpx2.Client(timeout=REQUEST_TIMEOUT_SECONDS, follow_redirects=False),
            )
        self._client = client

    def start(self, system: str, user_text: str, tools: list[ToolSpec]) -> Conversation:
        declared = []
        for t in tools:
            params: dict[str, Any] = {"type": "object", "properties": t.properties}
            if t.required:
                params["required"] = list(t.required)
            declared.append({"type": "function", "function": {
                "name": t.name, "description": t.description, "parameters": params}})
        return _OpenAIConversation(
            [{"role": "system", "content": system}, {"role": "user", "content": user_text}],
            declared)

    def generate(self, conversation: Conversation) -> Turn:
        conv: _OpenAIConversation = conversation  # type: ignore[assignment]
        limit = {"max_tokens": MAX_OUTPUT_TOKENS} if self._custom_base else {
            "max_completion_tokens": MAX_OUTPUT_TOKENS}
        try:
            response = self._client.chat.completions.create(
                model=self._model, messages=conv.messages, tools=conv.tools,
                tool_choice="auto", **limit)
        except Exception as exc:
            kind = classify_error(exc)
            log.error("OpenAI request failed (%s, model=%s): %s", kind.value, self._model,
                      redact(f"{type(exc).__name__}: {exc}", self._key))
            raise ProviderError(kind) from None
        if not getattr(response, "choices", None):
            raise ProviderError(ErrorKind.EMPTY_RESPONSE)
        message = response.choices[0].message
        raw_calls = [c for c in (message.tool_calls or []) if getattr(c, "type", "function") == "function"]
        if raw_calls:
            conv.last_message = {
                "role": "assistant", "content": message.content or None,
                "tool_calls": [{"id": c.id, "type": "function", "function": {
                    "name": c.function.name, "arguments": c.function.arguments or "{}"}}
                    for c in raw_calls]}
            calls = []
            for c in raw_calls:
                try:
                    args = json.loads(c.function.arguments or "{}")
                    if not isinstance(args, dict):
                        raise ValueError
                except ValueError:
                    args = {"__unparseable_arguments__": True}  # rejected later by tool validation
                calls.append(ToolCall(id=c.id, name=c.function.name, args=args))
            return Turn(text=(message.content or "").strip(), tool_calls=calls)
        text = (message.content or "").strip()
        if not text:
            raise ProviderError(ErrorKind.EMPTY_RESPONSE)
        return Turn(text=text)

    def add_tool_results(self, conversation: Conversation, turn: Turn,
                         results: list[ToolResult]) -> None:
        conv: _OpenAIConversation = conversation  # type: ignore[assignment]
        conv.messages.append(conv.last_message)
        for r in results:
            conv.messages.append({"role": "tool", "tool_call_id": r.call.id,
                                  "content": json.dumps(r.result, allow_nan=False)})
