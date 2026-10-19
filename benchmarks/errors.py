"""Error-handling cases. They run the REAL provider adapters against fake SDK clients,
so no network is used. Each ``run`` must raise; the harness checks the category
and that no secret appears in the message.
"""

from __future__ import annotations

import httpx2
import openai
import pandas as pd

from businesslens import agent
from businesslens.providers import ProviderConfig, create_provider
from businesslens.providers.scripted import fake_gemini_client, fake_openai_client
from cases import ErrCase

SECRET = "sk-benchmark-SECRET-1234567890"
DF = pd.DataFrame({"order_date": ["2025-01-01"], "product": ["A"], "region": ["N"],
                   "revenue": [1.0], "cost": [1.0]})
REQ = httpx2.Request("POST", "https://api.openai.com/v1/chat/completions")


def _oai_err(cls, status, message=None):
    exc = cls(f"{SECRET} rejected", response=httpx2.Response(status, request=REQ), body=None)
    if message:
        exc.args = (message,)
    return exc


def _gem_err(code, message):
    from google.genai import errors
    return errors.APIError(code, {"error": {"code": code, "message": message + " " + SECRET}})


def _ask_openai(*steps, key=SECRET):
    provider = create_provider(ProviderConfig("openai", "gpt-x", key), client=fake_openai_client(*steps))
    return lambda: agent.ask("What is total revenue?", DF, provider)


def _ask_gemini(*steps, key=SECRET):
    provider = create_provider(ProviderConfig("gemini", "gemini-x", key), client=fake_gemini_client(*steps))
    return lambda: agent.ask("What is total revenue?", DF, provider)


def _empty_openai():
    from types import SimpleNamespace
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="", tool_calls=None))])


def _limiter_questions():
    from businesslens.providers.scripted import ScriptedProvider
    from businesslens.providers import Turn
    lim = agent.UsageLimiter(max_questions=1, min_seconds_between_questions=0)
    agent.ask("q", DF, ScriptedProvider(Turn(text="a")), lim)
    agent.ask("q", DF, ScriptedProvider(Turn(text="a")), lim)


def _limiter_calls():
    from businesslens.providers.scripted import ScriptedProvider
    from businesslens.providers import Turn, ToolCall
    lim = agent.UsageLimiter(max_api_calls=1, min_seconds_between_questions=0)
    loop = Turn(tool_calls=[ToolCall("1", "get_summary_metrics", {})])
    agent.ask("q", DF, ScriptedProvider(loop, loop), lim)


def _limiter_pace():
    from businesslens.providers.scripted import ScriptedProvider
    from businesslens.providers import Turn
    lim = agent.UsageLimiter(min_seconds_between_questions=60)
    agent.ask("q", DF, ScriptedProvider(Turn(text="a")), lim)
    agent.ask("q", DF, ScriptedProvider(Turn(text="a")), lim)


def _bad_url(url):
    return lambda: create_provider(ProviderConfig("openai", "gpt-x", SECRET, url), client=object())


ERROR_CASES: tuple[ErrCase, ...] = (
    # missing / invalid credentials
    ErrCase("CRED-001", "missing_credentials", "OpenAI: empty key",
            lambda: create_provider(ProviderConfig("openai", "gpt-x", "")), "missing_credentials"),
    ErrCase("CRED-002", "missing_credentials", "Gemini: whitespace-only key",
            lambda: create_provider(ProviderConfig("gemini", "gemini-x", "   ")), "missing_credentials"),
    ErrCase("CRED-003", "missing_credentials", "OpenAI: key rejected (401)",
            _ask_openai(_oai_err(openai.AuthenticationError, 401)), "invalid_credentials"),
    ErrCase("CRED-004", "missing_credentials", "Gemini: key rejected (400 API key not valid)",
            _ask_gemini(_gem_err(400, "API key not valid. Please pass a valid API key.")),
            "invalid_credentials"),
    ErrCase("CRED-005", "missing_credentials", "OpenAI: key with whitespace inside",
            lambda: create_provider(ProviderConfig("openai", "gpt-x", "sk-a b")), "invalid_config"),
    # provider failures
    ErrCase("PROV-001", "provider_failure", "OpenAI: model not found (404)",
            _ask_openai(_oai_err(openai.NotFoundError, 404)), "model_not_found"),
    ErrCase("PROV-002", "provider_failure", "Gemini: model not found (404)",
            _ask_gemini(_gem_err(404, "models/x is not found")), "model_not_found"),
    ErrCase("PROV-003", "provider_failure", "OpenAI: timeout",
            _ask_openai(openai.APITimeoutError(request=REQ)), "timeout"),
    ErrCase("PROV-004", "provider_failure", "Gemini: timeout",
            _ask_gemini(httpx2.ReadTimeout("timed out")), "timeout"),
    ErrCase("PROV-005", "provider_failure", "OpenAI: tools unsupported by model",
            _ask_openai(_oai_err(openai.BadRequestError, 400, "this model does not support tools")),
            "tools_unsupported"),
    ErrCase("PROV-006", "provider_failure", "Gemini: tools unsupported by model",
            _ask_gemini(_gem_err(400, "Function calling is not enabled for this model")),
            "tools_unsupported"),
    ErrCase("PROV-007", "provider_failure", "OpenAI: server error (500)",
            _ask_openai(_oai_err(openai.InternalServerError, 500)), "provider_error"),
    ErrCase("PROV-008", "provider_failure", "OpenAI: connection error",
            _ask_openai(openai.APIConnectionError(request=REQ)), "connection"),
    ErrCase("PROV-009", "provider_failure", "OpenAI: empty model answer",
            _ask_openai(_empty_openai()), "empty_response"),
    ErrCase("PROV-010", "provider_failure", "OpenAI: permission denied (403)",
            _ask_openai(_oai_err(openai.PermissionDeniedError, 403)), "permission_denied"),
    # rate limits and cost guards
    ErrCase("RATE-001", "rate_limit", "OpenAI: 429 quota",
            _ask_openai(_oai_err(openai.RateLimitError, 429)), "rate_limit"),
    ErrCase("RATE-002", "rate_limit", "Gemini: 429 RESOURCE_EXHAUSTED",
            _ask_gemini(_gem_err(429, "RESOURCE_EXHAUSTED")), "rate_limit"),
    ErrCase("RATE-003", "rate_limit", "Session question cap", _limiter_questions,
            expect_agent_error="Session limit"),
    ErrCase("RATE-004", "rate_limit", "Session AI-request cap stops a tool loop", _limiter_calls,
            expect_agent_error="AI requests"),
    ErrCase("RATE-005", "rate_limit", "Question pacing", _limiter_pace, expect_agent_error="wait"),
    # unsafe configuration
    ErrCase("CFG-001", "unsafe_config", "Base URL pointing at cloud metadata IP",
            _bad_url("https://169.254.169.254/v1"), "invalid_config"),
    ErrCase("CFG-002", "unsafe_config", "Base URL with plain http",
            _bad_url("http://api.example.com/v1"), "invalid_config"),
    ErrCase("CFG-003", "unsafe_config", "Base URL localhost",
            _bad_url("https://localhost/v1"), "invalid_config"),
    ErrCase("CFG-004", "unsafe_config", "Base URL with embedded credentials",
            _bad_url("https://user:pw@api.example.com/v1"), "invalid_config"),
)
