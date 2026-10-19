"""Offline tests for the provider abstraction and both adapters, using fake SDK clients."""

import json
import logging
from types import SimpleNamespace

import httpx2
import openai
import pytest

from businesslens import agent
from businesslens.providers import (
    PROVIDERS, ErrorKind, ProviderConfig, ProviderError, ToolCall, ToolResult, create_provider,
    redact, validate_base_url,
)
from businesslens.providers import gemini as gem
from businesslens.providers import openai_provider as oai
from businesslens.providers.base import clean_config

KEY = "sk-test-SECRETKEY1234567890"


# ---- config and secrets ---------------------------------------------------

def test_config_repr_and_str_hide_the_key():
    cfg = ProviderConfig("openai", "gpt-x", KEY)
    assert KEY not in repr(cfg) and KEY not in str(cfg) and "<set>" in repr(cfg)


@pytest.mark.parametrize("key", ["", "   ", None])
def test_missing_key_is_reported(key):
    with pytest.raises(ProviderError) as exc:
        create_provider(ProviderConfig("openai", "gpt-x", key))
    assert exc.value.kind is ErrorKind.MISSING_CREDENTIALS


@pytest.mark.parametrize("key,model", [
    ("has space", "m"), ("k" * 600, "m"), ("kéy", "m"), ("key", ""), ("key", "bad model!"), ("key", "m" * 200),
])
def test_invalid_key_or_model_format(key, model):
    with pytest.raises(ProviderError) as exc:
        clean_config(ProviderConfig("openai", model, key))
    assert exc.value.kind is ErrorKind.INVALID_CONFIG
    assert key not in str(exc.value) or key == ""


def test_unknown_provider_and_gemini_rejects_base_url():
    with pytest.raises(ProviderError):
        create_provider(ProviderConfig("nope", "m", "k"))
    with pytest.raises(ProviderError, match="does not support"):
        create_provider(ProviderConfig("gemini", "m", "k", "https://example.com"))


def test_redact_removes_known_shapes_and_exact_secret():
    text = f"bad key {KEY} and AIza{'x' * 30} and Bearer abcdefgh12345 and custom-secret-value"
    out = redact(text, "custom-secret-value")
    for secret in (KEY, "AIza", "abcdefgh12345", "custom-secret-value"):
        assert secret not in out


def test_two_sessions_do_not_share_credentials():
    a = create_provider(ProviderConfig("openai", "m", "key-A-1234"), client=object())
    b = create_provider(ProviderConfig("openai", "m", "key-B-5678"), client=object())
    assert a._key == "key-A-1234" and b._key == "key-B-5678"
    la, lb = agent.UsageLimiter(), agent.UsageLimiter()
    la.start_question()
    assert lb.questions == 0


# ---- base URL validation --------------------------------------------------

def resolver_for(*addresses):
    def resolve(host, port, type=None):
        return [(2, 1, 6, "", (a, port)) for a in addresses]
    return resolve


PUBLIC = resolver_for("93.184.216.34")


@pytest.mark.parametrize("url", [
    "http://api.example.com/v1",                 # not https
    "https://user:pw@api.example.com/v1",        # credentials
    "https://api.example.com/v1?x=1",            # query
    "https://api.example.com/v1#frag",           # fragment
    "https://api.example.com:8443/v1",           # port
    "https://127.0.0.1/v1", "https://[::1]/v1", "https://169.254.169.254/latest",
    "https://10.0.0.5/v1", "https://localhost/v1", "https://db.internal/v1",
    "https://printer.local/v1", "ftp://api.example.com", "", "https://", "https://exa mple.com",
])
def test_unsafe_urls_refused(url):
    with pytest.raises(ProviderError) as exc:
        validate_base_url(url, resolver=PUBLIC)
    assert exc.value.kind is ErrorKind.INVALID_CONFIG


@pytest.mark.parametrize("addr", ["10.1.2.3", "192.168.0.9", "127.0.0.1", "169.254.169.254",
                                  "::1", "fd00::1", "::ffff:10.0.0.1", "0.0.0.0"])
def test_hostname_resolving_to_internal_address_refused(addr):
    with pytest.raises(ProviderError):
        validate_base_url("https://api.example.com/v1", resolver=resolver_for(addr))


def test_mixed_public_and_private_resolution_refused():
    with pytest.raises(ProviderError):
        validate_base_url("https://api.example.com/v1", resolver=resolver_for("93.184.216.34", "10.0.0.1"))


def test_unresolvable_host_refused_and_good_url_normalised():
    def fail(*a, **k):
        raise OSError
    with pytest.raises(ProviderError):
        validate_base_url("https://nope.example.com", resolver=fail)
    assert validate_base_url("https://api.example.com/v1/", resolver=PUBLIC) == "https://api.example.com/v1"


# ---- fake SDK helpers -------------------------------------------------------

def oai_client(*steps):
    """Fake OpenAI client: each step is a response or an exception to raise."""
    steps = list(steps)
    seen = []

    def create(**kwargs):
        seen.append(json.loads(json.dumps(kwargs, default=str)))
        step = steps.pop(0)
        if isinstance(step, BaseException):
            raise step
        return step
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)), seen=seen)


def oai_text(text):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text, tool_calls=None))])


def oai_calls(*calls):
    tcs = [SimpleNamespace(id=f"c{i}", type="function",
                           function=SimpleNamespace(name=n, arguments=a))
           for i, (n, a) in enumerate(calls)]
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=tcs))])


def sdk_error(cls, status, body=None):
    resp = httpx2.Response(status, request=httpx2.Request("POST", "https://api.openai.com/v1/x"))
    return cls(f"boom {KEY}", response=resp, body=body)


def make_oai(*steps):
    client = oai_client(*steps)
    return oai.OpenAIProvider(ProviderConfig("openai", "gpt-x", KEY), client=client), client


# ---- OpenAI adapter -----------------------------------------------------------

def test_openai_full_tool_round_trip():
    provider, client = make_oai(oai_calls(("get_summary_metrics", "{}")), oai_text("Profit is 140."))
    conv = provider.start("sys", "q", agent.tool_specs())
    turn = provider.generate(conv)
    assert turn.tool_calls == [ToolCall("c0", "get_summary_metrics", {})]
    provider.add_tool_results(conv, turn, [ToolResult(turn.tool_calls[0], {"total_profit": 140.0})])
    assert provider.generate(conv).text == "Profit is 140."
    second = client.seen[1]["messages"]
    assert [m["role"] for m in second] == ["system", "user", "assistant", "tool"]
    assert second[3]["tool_call_id"] == "c0" and "140.0" in second[3]["content"]
    assert client.seen[0]["tools"][0]["type"] == "function"
    assert client.seen[0]["max_completion_tokens"] > 0 and "temperature" not in client.seen[0]


def test_openai_bad_json_arguments_become_rejectable_call():
    provider, _ = make_oai(oai_calls(("get_summary_metrics", "not json")))
    turn = provider.generate(provider.start("s", "q", []))
    assert "__unparseable_arguments__" in turn.tool_calls[0].args


def test_openai_empty_answer_and_no_choices():
    for resp in (oai_text(""), SimpleNamespace(choices=[])):
        provider, _ = make_oai(resp)
        with pytest.raises(ProviderError) as exc:
            provider.generate(provider.start("s", "q", []))
        assert exc.value.kind is ErrorKind.EMPTY_RESPONSE


@pytest.mark.parametrize("exc,kind", [
    (sdk_error(openai.AuthenticationError, 401), ErrorKind.INVALID_CREDENTIALS),
    (sdk_error(openai.PermissionDeniedError, 403), ErrorKind.PERMISSION),
    (sdk_error(openai.NotFoundError, 404), ErrorKind.MODEL_NOT_FOUND),
    (sdk_error(openai.RateLimitError, 429), ErrorKind.RATE_LIMIT),
    (sdk_error(openai.InternalServerError, 500), ErrorKind.PROVIDER),
    (sdk_error(openai.BadRequestError, 400), ErrorKind.BAD_REQUEST),
    (openai.APITimeoutError(request=httpx2.Request("POST", "https://x")), ErrorKind.TIMEOUT),
    (openai.APIConnectionError(request=httpx2.Request("POST", "https://x")), ErrorKind.CONNECTION),
    (RuntimeError(f"weird {KEY}"), ErrorKind.PROVIDER),
])
def test_openai_errors_are_classified_and_never_leak_key(exc, kind, caplog):
    provider, _ = make_oai(exc)
    with caplog.at_level(logging.ERROR), pytest.raises(ProviderError) as raised:
        provider.generate(provider.start("s", "q", []))
    assert raised.value.kind is kind
    assert KEY not in str(raised.value) and KEY not in caplog.text and "SECRETKEY" not in caplog.text


def test_openai_tools_unsupported_is_detected():
    exc = sdk_error(openai.BadRequestError, 400)
    exc.args = ("This model does not support tools",)
    provider, _ = make_oai(exc)
    with pytest.raises(ProviderError) as raised:
        provider.generate(provider.start("s", "q", agent.tool_specs()))
    assert raised.value.kind is ErrorKind.TOOLS_UNSUPPORTED


def test_openai_model_not_found_by_code():
    exc = sdk_error(openai.BadRequestError, 400)
    exc.code = "model_not_found"
    provider, _ = make_oai(exc)
    with pytest.raises(ProviderError) as raised:
        provider.generate(provider.start("s", "q", []))
    assert raised.value.kind is ErrorKind.MODEL_NOT_FOUND


def test_custom_base_url_adapter_uses_max_tokens_and_validated_url(monkeypatch):
    monkeypatch.setattr("businesslens.providers.validate_base_url", lambda u: u)
    client = oai_client(oai_text("ok"))
    provider = create_provider(ProviderConfig("openai", "m", KEY, "https://api.example.com/v1"), client=client)
    provider.generate(provider.start("s", "q", []))
    assert "max_tokens" in client.seen[0] and "max_completion_tokens" not in client.seen[0]


def test_real_openai_client_is_built_without_redirects_or_retries():
    provider = create_provider(ProviderConfig("openai", "gpt-x", KEY))
    sdk = provider._client
    assert sdk.max_retries == 0
    assert sdk._client.follow_redirects is False


# ---- Gemini adapter -----------------------------------------------------------

def gem_call(name, args=None):
    from google.genai import types
    call = types.FunctionCall(name=name, args=args or {})
    content = types.Content(role="model", parts=[types.Part(function_call=call)])
    return SimpleNamespace(function_calls=[call], text=None, candidates=[SimpleNamespace(content=content)])


def gem_text(text):
    return SimpleNamespace(function_calls=None, text=text, candidates=[])


def make_gem(*steps):
    steps = list(steps)
    seen = []

    def generate_content(model, contents, config):
        seen.append(list(contents))
        step = steps.pop(0)
        if isinstance(step, BaseException):
            raise step
        return step
    client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content), seen=seen)
    return gem.GeminiProvider(ProviderConfig("gemini", "gemini-x", KEY), client=client), client


def api_error(code, message):
    from google.genai import errors
    return errors.APIError(code, {"error": {"code": code, "message": message, "status": "X"}})


def test_gemini_full_tool_round_trip_and_declarations():
    provider, client = make_gem(gem_call("get_summary_metrics"), gem_text("Done."))
    conv = provider.start("sys", "q", agent.tool_specs())
    turn = provider.generate(conv)
    assert turn.tool_calls[0].name == "get_summary_metrics"
    provider.add_tool_results(conv, turn, [ToolResult(turn.tool_calls[0], {"total_profit": 1.0})])
    assert provider.generate(conv).text == "Done."
    assert [c.role for c in client.seen[1]] == ["user", "model", "user"]
    assert client.seen[1][-1].parts[0].function_response.response == {"result": {"total_profit": 1.0}}
    names = {d.name for d in provider._declarations(agent.tool_specs())}
    assert names == set(agent.TOOLS)


@pytest.mark.parametrize("exc,kind", [
    (api_error(400, "API key not valid. Please pass a valid API key."), ErrorKind.INVALID_CREDENTIALS),
    (api_error(401, "unauthenticated"), ErrorKind.INVALID_CREDENTIALS),
    (api_error(403, "denied"), ErrorKind.PERMISSION),
    (api_error(404, "models/x is not found"), ErrorKind.MODEL_NOT_FOUND),
    (api_error(429, "RESOURCE_EXHAUSTED"), ErrorKind.RATE_LIMIT),
    (api_error(500, "internal"), ErrorKind.PROVIDER),
    (api_error(400, "Function calling is not enabled for this model"), ErrorKind.TOOLS_UNSUPPORTED),
    (api_error(400, "bad field"), ErrorKind.BAD_REQUEST),
    (httpx2.ReadTimeout("timed out"), ErrorKind.TIMEOUT),
    (RuntimeError(f"odd {KEY}"), ErrorKind.PROVIDER),
])
def test_gemini_errors_are_classified_and_never_leak_key(exc, kind, caplog):
    provider, _ = make_gem(exc)
    with caplog.at_level(logging.ERROR), pytest.raises(ProviderError) as raised:
        provider.generate(provider.start("s", "q", agent.tool_specs()))
    assert raised.value.kind is kind
    assert KEY not in str(raised.value) and KEY not in caplog.text


def test_gemini_empty_answer():
    provider, _ = make_gem(gem_text("  "))
    with pytest.raises(ProviderError) as raised:
        provider.generate(provider.start("s", "q", []))
    assert raised.value.kind is ErrorKind.EMPTY_RESPONSE


def test_registry_lists_both_providers():
    assert {"gemini", "openai"} <= set(PROVIDERS)
    assert PROVIDERS["openai"].supports_base_url and not PROVIDERS["gemini"].supports_base_url
