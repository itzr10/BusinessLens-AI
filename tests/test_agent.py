"""Offline tests for the agent loop with a scripted provider. No network, no API key."""

import pandas as pd
import pytest

from businesslens import agent, analytics as an
from businesslens.providers import ErrorKind, ProviderError, ToolCall, Turn
from businesslens.providers.scripted import ScriptedProvider

DF = an.validate_sales_data(pd.DataFrame({
    "order_date": ["2025-01-05", "2025-02-10", "2025-04-01"],
    "product": ["A", "B", "A"],
    "region": ["North", "South", "North"],
    "revenue": [100, 200, 0],
    "cost": [60, 100, 0],
}))


def call(name, args=None, id="c"):
    return Turn(tool_calls=[ToolCall(id, name, args or {})])


def text(t):
    return Turn(text=t)


def test_agent_calls_tool_then_answers_and_returns_result_to_model():
    provider = ScriptedProvider(call("get_summary_metrics"), text("Profit is 140."))
    answer = agent.ask("What is total profit?", DF, provider)
    assert answer.text == "Profit is 140."
    assert answer.tool_calls == [{"name": "get_summary_metrics", "args": {}, "ok": True}]
    assert answer.api_calls == 2
    sent = provider.conversations[0].tool_results[0][0].result
    assert sent["total_profit"] == 140.0


def test_tool_declarations_are_provider_neutral_and_match_allowlist():
    specs = agent.tool_specs()
    assert {s.name for s in specs} == set(agent.TOOLS)


@pytest.mark.parametrize("name,args", [
    ("delete_everything", {}), ("os_system", {"cmd": "rm -rf /"}),
    ("get_summary_metrics", {"sql": "DROP TABLE sales"}),
    ("get_revenue_growth", {"period": "x; import os"}),
])
def test_agent_survives_unapproved_requests_and_tells_model(name, args):
    provider = ScriptedProvider(call(name, args), text("Cannot do that."))
    answer = agent.ask("hack", DF, provider)
    assert answer.tool_calls[0]["ok"] is False
    assert "error" in provider.conversations[0].tool_results[0][0].result


def test_agent_stops_runaway_loops():
    provider = ScriptedProvider(*[call("get_summary_metrics")] * agent.MAX_STEPS)
    with pytest.raises(agent.AgentError, match="too many steps"):
        agent.ask("loop", DF, provider)
    assert provider.generate_calls == agent.MAX_STEPS


def test_per_step_and_total_tool_call_caps():
    many = Turn(tool_calls=[ToolCall(str(i), "get_summary_metrics", {}) for i in range(10)])
    provider = ScriptedProvider(many, many, many, text("ok"))
    answer = agent.ask("q", DF, provider)
    assert sum(c["ok"] for c in answer.tool_calls) == agent.MAX_TOTAL_TOOL_CALLS
    first = provider.conversations[0].tool_results[0]
    assert sum("error" in r.result for r in first) == 10 - agent.MAX_CALLS_PER_STEP


def test_multi_round_flow():
    provider = ScriptedProvider(call("get_summary_metrics"),
                                call("get_revenue_growth", {"period": "quarter"}), text("Done."))
    answer = agent.ask("q", DF, provider)
    assert [c["name"] for c in answer.tool_calls] == ["get_summary_metrics", "get_revenue_growth"]


@pytest.mark.parametrize("q", ["", "   ", "x" * 501])
def test_agent_rejects_bad_questions_without_calling_provider(q):
    provider = ScriptedProvider()
    with pytest.raises(agent.AgentError):
        agent.ask(q, DF, provider)
    assert provider.generate_calls == 0


def test_provider_errors_propagate_with_kind():
    provider = ScriptedProvider(ProviderError(ErrorKind.RATE_LIMIT))
    with pytest.raises(ProviderError) as exc:
        agent.ask("q", DF, provider)
    assert exc.value.kind is ErrorKind.RATE_LIMIT


def test_empty_model_answer_is_an_error():
    with pytest.raises(ProviderError):
        agent.ask("q", DF, ScriptedProvider(text("  ")))


def test_answer_text_has_no_links_or_images():
    out = agent.sanitize_answer("See ![x](http://evil.test/?d=1) and [here](https://evil.test) "
                                "or http://evil.test/a?b=c and www.evil.test now.")
    assert "evil" not in out and "(" not in out
    assert "See x and here or" in out


def test_system_prompt_treats_data_as_untrusted():
    provider = ScriptedProvider(text("ok"))
    agent.ask("q", DF, provider)
    system = provider.conversations[0].system
    assert "untrusted" in system and "never reveal" in system


# ---- limiter ------------------------------------------------------------------

class Clock:
    t = 100.0

    def __call__(self):
        return self.t


def test_limiter_caps_questions_pace_and_api_calls():
    clock = Clock()
    lim = agent.UsageLimiter(max_questions=2, max_api_calls=3, clock=clock)
    agent.ask("q1", DF, ScriptedProvider(text("a")), lim)
    with pytest.raises(agent.AgentError, match="wait"):
        agent.ask("q2", DF, ScriptedProvider(text("a")), lim)
    clock.t += 5
    agent.ask("q2", DF, ScriptedProvider(text("a")), lim)
    clock.t += 5
    with pytest.raises(agent.AgentError, match="Session limit"):
        agent.ask("q3", DF, ScriptedProvider(text("a")), lim)


def test_limiter_stops_api_calls_mid_question():
    lim = agent.UsageLimiter(max_api_calls=2, min_seconds_between_questions=0)
    provider = ScriptedProvider(*[call("get_summary_metrics")] * 5)
    with pytest.raises(agent.AgentError, match="AI requests"):
        agent.ask("q", DF, provider, lim)
    assert provider.generate_calls == 2


def test_rejected_questions_do_not_use_budget():
    lim = agent.UsageLimiter()
    with pytest.raises(agent.AgentError):
        agent.ask("", DF, ScriptedProvider(), lim)
    assert lim.questions == 0


def test_test_connection_declares_tools_and_counts_a_call():
    lim = agent.UsageLimiter()
    provider = ScriptedProvider(text("OK"))
    agent.test_connection(provider, lim)
    assert lim.api_calls == 1 and provider.conversations[0].tools
    with pytest.raises(ProviderError):
        agent.test_connection(ScriptedProvider(ProviderError(ErrorKind.INVALID_CREDENTIALS)))
