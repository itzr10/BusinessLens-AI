"""Offline tests for the approved tools. No network, no API key."""

import json

import pandas as pd
import pytest

from businesslens import analytics as an
from businesslens.tools import TOOLS, ToolError, run_tool

DF = an.validate_sales_data(pd.DataFrame({
    "order_date": ["2025-01-05", "2025-02-10", "2025-04-01"],
    "product": ["A", "B", "A"],
    "region": ["North", "South", "North"],
    "revenue": [100, 200, 0],
    "cost": [60, 100, 0],
}))


def test_tools_match_analytics_and_are_json_safe():
    m = run_tool("get_summary_metrics", {}, DF)
    assert m == {"total_revenue": 300.0, "total_cost": 160.0,
                 "total_profit": 140.0, "profit_margin": pytest.approx(140 / 300)}
    for name in TOOLS:
        json.dumps(run_tool(name, {}, DF), allow_nan=False)  # NaN would raise
    rows = run_tool("get_revenue_growth", {"period": "month"}, DF)["rows"]
    assert rows[0]["growth"] is None and rows[1]["growth"] == 1.0


@pytest.mark.parametrize("name,args", [
    ("os_system", {}),
    ("get_revenue_growth", {"period": "decade"}),
    ("get_summary_metrics", {"sql": "drop"}),
    ("get_revenue_growth", {"period": 5}),
    ("get_revenue_growth", {"period": "month" * 30}),
    ("get_filtered_summary", {"product": ["A"]}),
    ("get_filtered_summary", {"product": "Nope"}),
    ("get_filtered_summary", {"start_date": "01/02/2025"}),
    ("get_filtered_summary", {"start_date": "2025-03-01", "end_date": "2025-01-01"}),
    ("get_filtered_summary", {"product": "A' OR 1=1 --"}),
])
def test_unapproved_tool_and_bad_args_rejected(name, args):
    with pytest.raises(ToolError):
        run_tool(name, args, DF)


def test_filtered_summary_filters_and_canonicalises_names():
    r = run_tool("get_filtered_summary", {"product": "a", "region": "NORTH"}, DF)
    assert r["rows_matched"] == 2 and r["total_revenue"] == 100.0
    assert r["filters"] == {"product": "A", "region": "North"}
    r = run_tool("get_filtered_summary", {"start_date": "2025-02-01", "end_date": "2025-02-28"}, DF)
    assert r["total_revenue"] == 200.0 and r["profit_margin"] == 0.5


def test_filtered_summary_with_no_matching_rows_is_zero_not_error():
    r = run_tool("get_filtered_summary", {"start_date": "2030-01-01"}, DF)
    assert r["rows_matched"] == 0 and r["total_revenue"] == 0.0 and r["profit_margin"] is None


def test_every_tool_schema_is_plain_strings():
    for tool in TOOLS.values():
        for spec in tool.parameters.values():
            assert spec["type"] == "string"
