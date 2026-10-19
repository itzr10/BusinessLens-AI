"""Live provider smoke tests. Skipped unless explicitly enabled; normal runs make no API calls.

    BUSINESSLENS_LIVE=1 BUSINESSLENS_LIVE_OPENAI_KEY=... BUSINESSLENS_LIVE_OPENAI_MODEL=... pytest tests/test_live_providers.py
"""

import os

import pytest

from businesslens import agent
from businesslens.providers import ProviderConfig, create_provider
from businesslens import analytics as an

pytestmark = pytest.mark.skipif(os.environ.get("BUSINESSLENS_LIVE") != "1",
                                reason="live tests are opt-in: set BUSINESSLENS_LIVE=1")


@pytest.mark.parametrize("provider", ["openai", "gemini"])
def test_live_tool_round_trip(provider):
    key = os.environ.get(f"BUSINESSLENS_LIVE_{provider.upper()}_KEY", "").strip()
    model = os.environ.get(f"BUSINESSLENS_LIVE_{provider.upper()}_MODEL", "").strip()
    if not key or not model:
        pytest.skip(f"no live key/model configured for {provider}")
    df = an.load_sales_data("data/sample_sales.csv")
    answer = agent.ask("What is the total revenue?", df, create_provider(ProviderConfig(provider, model, key)))
    assert "86,670" in answer.text.replace(" ", "") or "86670" in answer.text
    assert any(c["name"] == "get_summary_metrics" and c["ok"] for c in answer.tool_calls)
