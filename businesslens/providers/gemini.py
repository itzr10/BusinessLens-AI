"""Google Gemini adapter (``google-genai`` SDK). No custom base URL is supported."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from businesslens.providers.base import (
    MAX_OUTPUT_TOKENS, REQUEST_TIMEOUT_SECONDS, Conversation, ErrorKind, Provider,
    ProviderConfig, ProviderError, ToolCall, ToolResult, ToolSpec, Turn, redact,
)

log = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-3.8-flash"


@dataclass
class _GeminiConversation(Conversation):
    contents: list[Any]
    config: Any
    last_raw: Any = None
    tool_ids: dict[str, str] = field(default_factory=dict)


def classify_error(exc: BaseException) -> ErrorKind:
    """Map an SDK/network exception to an ``ErrorKind`` (uses type, status code, wording)."""
    code = getattr(exc, "code", None)
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    if "timeout" in name or "timed out" in text:
        return ErrorKind.TIMEOUT
    if code == 429 or "resource_exhausted" in text:
        return ErrorKind.RATE_LIMIT
    if code in (401,) or "api key not valid" in text or "api_key_invalid" in text:
        return ErrorKind.INVALID_CREDENTIALS
    if code == 403:
        return ErrorKind.PERMISSION
    if code == 404:
        return ErrorKind.MODEL_NOT_FOUND
    if code == 400 and ("function calling" in text or "tool use" in text or "tools" in text and "not supported" in text):
        return ErrorKind.TOOLS_UNSUPPORTED
    if code == 400:
        return ErrorKind.BAD_REQUEST
    if isinstance(code, int) and code >= 500:
        return ErrorKind.PROVIDER
    if "connect" in name or "transport" in name or "network" in name:
        return ErrorKind.CONNECTION
    return ErrorKind.PROVIDER


class GeminiProvider(Provider):
    name = "gemini"

    def __init__(self, config: ProviderConfig, client: Any = None):
        self._model = config.model
        self._key = config.api_key  # only used to redact log lines
        if client is None:
            from google import genai
            from google.genai import types

            client = genai.Client(
                api_key=config.api_key,
                http_options=types.HttpOptions(timeout=int(REQUEST_TIMEOUT_SECONDS * 1000)),
            )
        self._client = client

    @staticmethod
    def _declarations(tools: list[ToolSpec]):
        from google.genai import types

        out = []
        for tool in tools:
            schema = None
            if tool.properties:
                schema = {"type": "OBJECT", "properties": {
                    k: {**v, "type": v["type"].upper()} for k, v in tool.properties.items()
                }}
                if tool.required:
                    schema["required"] = list(tool.required)
            out.append(types.FunctionDeclaration(
                name=tool.name, description=tool.description, parameters=schema))
        return out

    def start(self, system: str, user_text: str, tools: list[ToolSpec]) -> Conversation:
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system,
            tools=[types.Tool(function_declarations=self._declarations(tools))],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            temperature=0,
            max_output_tokens=MAX_OUTPUT_TOKENS,
        )
        contents = [types.Content(role="user", parts=[types.Part(text=user_text)])]
        return _GeminiConversation(contents, config)

    def generate(self, conversation: Conversation) -> Turn:
        conv: _GeminiConversation = conversation  # type: ignore[assignment]
        try:
            response = self._client.models.generate_content(
                model=self._model, contents=conv.contents, config=conv.config)
        except Exception as exc:  # classified; details are logged redacted, never shown
            kind = classify_error(exc)
            log.error("Gemini request failed (%s, model=%s): %s", kind.value, self._model,
                      redact(f"{type(exc).__name__}: {exc}", self._key))
            raise ProviderError(kind) from None
        calls = response.function_calls or []
        if calls:
            conv.last_raw = response.candidates[0].content
            return Turn(tool_calls=[
                ToolCall(id=f"call_{i}", name=c.name, args=dict(c.args or {}))
                for i, c in enumerate(calls)])
        text = (response.text or "").strip()
        if not text:
            raise ProviderError(ErrorKind.EMPTY_RESPONSE)
        return Turn(text=text)

    def add_tool_results(self, conversation: Conversation, turn: Turn,
                         results: list[ToolResult]) -> None:
        from google.genai import types

        conv: _GeminiConversation = conversation  # type: ignore[assignment]
        conv.contents.append(conv.last_raw)
        conv.contents.append(types.Content(role="user", parts=[
            types.Part.from_function_response(
                name=r.call.name, response={"result": json.loads(json.dumps(r.result))})
            for r in results]))
