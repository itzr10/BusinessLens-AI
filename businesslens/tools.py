"""Approved analytics tools that the AI agent may call.

Each tool wraps a tested function from ``analytics.py`` and returns plain,
JSON-safe data (no NaN, no numpy types). The agent can only run tools listed in
``TOOLS``; it can never run arbitrary code or touch the data directly.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from businesslens import analytics as an


class ToolError(ValueError):
    """Raised for an unknown tool or invalid arguments."""


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    run: Callable[..., dict[str, Any]]
    parameters: dict[str, Any] = field(default_factory=dict)  # JSON-schema properties
    required: tuple[str, ...] = ()


def _clean(value: Any) -> Any:
    """Make a value JSON-safe: NaN/None -> None, numpy scalars -> Python."""
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        value = value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _records(table: pd.DataFrame) -> list[dict[str, Any]]:
    return [{k: _clean(v) for k, v in row.items()} for row in table.to_dict("records")]


def data_overview(df: pd.DataFrame) -> dict[str, Any]:
    clean = an.validate_sales_data(df)
    return {
        "rows": len(clean),
        "first_date": clean["order_date"].min().strftime("%Y-%m-%d"),
        "last_date": clean["order_date"].max().strftime("%Y-%m-%d"),
        "products": sorted(clean["product"].unique()),
        "regions": sorted(clean["region"].unique()),
    }


def summary_metrics(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "total_revenue": an.total_revenue(df),
        "total_cost": an.total_cost(df),
        "total_profit": an.total_profit(df),
        "profit_margin": an.profit_margin(df),  # fraction; None if revenue is zero
    }


def revenue_growth(df: pd.DataFrame, period: str = "month") -> dict[str, Any]:
    if period not in ("month", "quarter", "year"):
        raise ToolError("period must be 'month', 'quarter' or 'year'.")
    return {"period": period, "rows": _records(an.period_growth(df, period))}


MAX_ARG_CHARS = 64


def _canonical(value: str, allowed: list[str], label: str) -> str:
    """Match ``value`` to a name that exists in the data (case-insensitive)."""
    for name in allowed:
        if name.casefold() == value.strip().casefold():
            return name
    raise ToolError(f"Unknown {label} {value!r}. Known {label}s: {allowed}.")


def _parse_day(value: str, label: str) -> pd.Timestamp:
    try:
        return pd.Timestamp(pd.to_datetime(value, format="%Y-%m-%d"))
    except (ValueError, TypeError):
        raise ToolError(f"{label} must be a date written YYYY-MM-DD.") from None


def filtered_summary(df: pd.DataFrame, product: str | None = None, region: str | None = None,
                     start_date: str | None = None, end_date: str | None = None) -> dict[str, Any]:
    """Summary metrics for the rows matching optional product/region/date filters."""
    clean = an.validate_sales_data(df)
    applied: dict[str, str] = {}
    keep = pd.Series(True, index=clean.index)
    if product is not None:
        applied["product"] = _canonical(product, sorted(clean["product"].unique()), "product")
        keep &= clean["product"] == applied["product"]
    if region is not None:
        applied["region"] = _canonical(region, sorted(clean["region"].unique()), "region")
        keep &= clean["region"] == applied["region"]
    start = _parse_day(start_date, "start_date") if start_date is not None else None
    end = _parse_day(end_date, "end_date") if end_date is not None else None
    if start is not None and end is not None and start > end:
        raise ToolError("start_date must not be after end_date.")
    days = clean["order_date"].dt.normalize()
    if start is not None:
        applied["start_date"] = start_date
        keep &= days >= start
    if end is not None:
        applied["end_date"] = end_date
        keep &= days <= end
    subset = clean[keep]
    if subset.empty:
        return {"filters": applied, "rows_matched": 0, "total_revenue": 0.0, "total_cost": 0.0,
                "total_profit": 0.0, "profit_margin": None}
    return {"filters": applied, "rows_matched": len(subset), **summary_metrics(subset)}


def product_performance(df: pd.DataFrame) -> dict[str, Any]:
    return {"rows": _records(an.product_performance(df))}


def region_performance(df: pd.DataFrame) -> dict[str, Any]:
    return {"rows": _records(an.region_performance(df))}


TOOLS: dict[str, Tool] = {
    t.name: t
    for t in [
        Tool(
            "get_data_overview",
            "Row count, first and last date, and the product and region names in the data.",
            data_overview,
        ),
        Tool(
            "get_summary_metrics",
            "Total revenue, total cost, total profit and profit margin "
            "(margin is a fraction, 0.25 = 25%; null if revenue is zero).",
            summary_metrics,
        ),
        Tool(
            "get_revenue_growth",
            "Revenue per period with previous-period revenue and growth "
            "(a fraction; null when undefined).",
            revenue_growth,
            {"period": {"type": "string", "enum": ["month", "quarter", "year"],
                        "description": "Grouping period. Defaults to month."}},
        ),
        Tool(
            "get_filtered_summary",
            "Total revenue, cost, profit and margin for only the sales matching optional "
            "filters: one product, one region, and/or an inclusive date range. "
            "Use for questions like 'Laptop revenue in North' or 'revenue in March'.",
            filtered_summary,
            {"product": {"type": "string", "description": "Exact product name from the data."},
             "region": {"type": "string", "description": "Exact region name from the data."},
             "start_date": {"type": "string", "description": "First day, YYYY-MM-DD."},
             "end_date": {"type": "string", "description": "Last day, YYYY-MM-DD (inclusive)."}},
        ),
        Tool(
            "get_product_performance",
            "Revenue, cost, profit and profit margin for every product, best revenue first.",
            product_performance,
        ),
        Tool(
            "get_region_performance",
            "Revenue, cost, profit and profit margin for every region, best revenue first.",
            region_performance,
        ),
    ]
}


def run_tool(name: str, args: dict[str, Any] | None, df: pd.DataFrame) -> dict[str, Any]:
    """Run an approved tool on ``df``. Unknown tools and bad arguments raise ``ToolError``."""
    tool = TOOLS.get(name)
    if tool is None:
        raise ToolError(f"Tool {name!r} is not approved.")
    args = dict(args or {})
    unknown = set(args) - set(tool.parameters)
    missing = set(tool.required) - set(args)
    if unknown or missing:
        raise ToolError(
            f"Bad arguments for {name}: unexpected {sorted(unknown)}, missing {sorted(missing)}."
        )
    for key, value in args.items():
        _check_arg(name, key, value, tool.parameters[key])
    return tool.run(df, **args)


def _check_arg(tool: str, key: str, value: Any, schema: dict[str, Any]) -> None:
    """Enforce the declared type, enum and length so the model cannot pass anything else."""
    if schema.get("type") == "string":
        if not isinstance(value, str) or len(value) > MAX_ARG_CHARS:
            raise ToolError(f"Argument {key!r} of {tool} must be a string of at most "
                            f"{MAX_ARG_CHARS} characters.")
        if "enum" in schema and value not in schema["enum"]:
            raise ToolError(f"Argument {key!r} of {tool} must be one of {schema['enum']}.")
