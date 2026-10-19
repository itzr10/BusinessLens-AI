"""Deterministic sales analytics.

Every function here is plain Python/pandas: the same input always gives the
same output. There is no UI code and no AI code in this module, so it can be
tested on its own and reused later by Streamlit or an AI agent.

Conventions
-----------
* Ratios are returned as fractions (0.25 means 25%), not percentages.
* A ratio whose denominator is zero is "undefined". Scalar functions return
  ``None`` and table columns hold ``NaN`` (never a made-up number such as 0).
* Revenue and cost must be finite and non-negative. Refunds/returns are not
  modelled yet, so negative values are rejected rather than guessed at.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["order_date", "product", "region", "revenue", "cost"]
NUMERIC_COLUMNS = ["revenue", "cost"]
TEXT_COLUMNS = ["product", "region"]


class DataValidationError(ValueError):
    """Raised when sales data is missing, malformed, or unusable."""


# --------------------------------------------------------------------------
# Loading and validation
# --------------------------------------------------------------------------

def load_sales_data(path: str | Path) -> pd.DataFrame:
    """Read a sales CSV and return validated data."""
    try:
        raw = pd.read_csv(path)
    except FileNotFoundError as exc:
        raise DataValidationError(f"File not found: {path}") from exc
    except pd.errors.EmptyDataError as exc:
        raise DataValidationError(f"File is empty: {path}") from exc
    return validate_sales_data(raw)


def validate_sales_data(df: pd.DataFrame) -> pd.DataFrame:
    """Check the data and return a clean copy.

    The returned frame has ``order_date`` as datetime, ``revenue``/``cost`` as
    floats, and trimmed text in ``product``/``region``. The input is never
    modified. Any problem raises ``DataValidationError`` with a clear message.
    Validating already-clean data is safe: it just returns an equal copy.
    """
    if not isinstance(df, pd.DataFrame):
        raise DataValidationError("Sales data must be a pandas DataFrame.")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(f"Missing required column(s): {', '.join(missing)}")

    if df.empty:
        raise DataValidationError("Sales data has no rows.")

    clean = df[REQUIRED_COLUMNS].copy()

    # Dates: anything that cannot be read as a date becomes NaT, which we reject.
    # format="mixed" parses each value on its own (also avoids a pandas warning).
    dates = pd.to_datetime(clean["order_date"], errors="coerce", format="mixed")
    bad = dates.isna()
    if bad.any():
        raise DataValidationError(
            f"Invalid or missing order_date in row(s): {_row_list(bad)}"
        )
    clean["order_date"] = dates

    # Numbers: non-numeric text becomes NaN, which we reject.
    for col in NUMERIC_COLUMNS:
        numbers = pd.to_numeric(clean[col], errors="coerce").astype("float64")
        bad = numbers.isna() | np.isinf(numbers)
        if bad.any():
            raise DataValidationError(
                f"Non-numeric or missing {col} in row(s): {_row_list(bad)}"
            )
        negative = numbers < 0
        if negative.any():
            raise DataValidationError(
                f"Negative {col} is not supported, row(s): {_row_list(negative)}"
            )
        clean[col] = numbers

    # Text: must be present and not blank.
    for col in TEXT_COLUMNS:
        text = clean[col].astype("object").where(clean[col].notna(), "")
        text = text.map(lambda v: str(v).strip())
        bad = text == ""
        if bad.any():
            raise DataValidationError(f"Missing {col} in row(s): {_row_list(bad)}")
        clean[col] = text

    return clean.reset_index(drop=True)


def _row_list(mask: pd.Series, limit: int = 5) -> str:
    """Describe flagged rows as 1-based data row numbers (header excluded)."""
    rows = [str(i + 1) for i in np.flatnonzero(mask.to_numpy())]
    shown = ", ".join(rows[:limit])
    return shown + (f" (and {len(rows) - limit} more)" if len(rows) > limit else "")


# --------------------------------------------------------------------------
# Headline metrics
# --------------------------------------------------------------------------

def total_revenue(df: pd.DataFrame) -> float:
    """Sum of all revenue."""
    return float(validate_sales_data(df)["revenue"].sum())


def total_cost(df: pd.DataFrame) -> float:
    """Sum of all cost."""
    return float(validate_sales_data(df)["cost"].sum())


def total_profit(df: pd.DataFrame) -> float:
    """Profit = total revenue - total cost (can be negative)."""
    clean = validate_sales_data(df)
    return float(clean["revenue"].sum() - clean["cost"].sum())


def profit_margin(df: pd.DataFrame) -> float | None:
    """Profit margin = profit / revenue, or ``None`` if revenue is zero."""
    clean = validate_sales_data(df)
    revenue = clean["revenue"].sum()
    if revenue == 0:
        return None
    return float((revenue - clean["cost"].sum()) / revenue)


# --------------------------------------------------------------------------
# Growth over time
# --------------------------------------------------------------------------

_PERIOD_CODES = {"month": "M", "quarter": "Q", "year": "Y"}


def period_growth(df: pd.DataFrame, period: str = "month") -> pd.DataFrame:
    """Revenue per calendar period with period-over-period growth.

    ``period`` is ``"month"``, ``"quarter"`` or ``"year"``. Columns returned:
    ``period``, ``revenue``, ``previous_revenue``, ``growth``.

    * Periods with no sales inside the date range appear with revenue 0, so a
      gap is never silently skipped (growth compares neighbouring periods).
    * ``growth = (revenue - previous) / previous``. It is ``NaN`` for the first
      period (nothing to compare) and when the previous revenue is zero.
    """
    if period not in _PERIOD_CODES:
        raise ValueError(f"period must be one of {sorted(_PERIOD_CODES)}, got {period!r}")

    clean = validate_sales_data(df)
    code = _PERIOD_CODES[period]
    periods = clean["order_date"].dt.to_period(code)

    revenue = clean["revenue"].groupby(periods).sum()
    full_range = pd.period_range(periods.min(), periods.max(), freq=code)
    revenue = revenue.reindex(full_range, fill_value=0.0)

    result = pd.DataFrame(
        {
            "period": [str(p) for p in full_range],
            "revenue": revenue.to_numpy(dtype="float64"),
        }
    )
    result["previous_revenue"] = result["revenue"].shift(1)
    previous = result["previous_revenue"].replace(0, np.nan)  # zero -> undefined
    result["growth"] = (result["revenue"] - previous) / previous
    return result


# --------------------------------------------------------------------------
# Product and regional performance
# --------------------------------------------------------------------------

def _performance_by(df: pd.DataFrame, column: str) -> pd.DataFrame:
    clean = validate_sales_data(df)
    grouped = clean.groupby(column, as_index=False)[["revenue", "cost"]].sum()
    grouped["profit"] = grouped["revenue"] - grouped["cost"]
    revenue = grouped["revenue"].replace(0, np.nan)  # zero -> undefined margin
    grouped["profit_margin"] = grouped["profit"] / revenue
    # Highest revenue first; ties broken by name so the order is always the same.
    return grouped.sort_values(
        ["revenue", column], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)


def product_performance(df: pd.DataFrame) -> pd.DataFrame:
    """Revenue, cost, profit and margin per product (best revenue first)."""
    return _performance_by(df, "product")


def region_performance(df: pd.DataFrame) -> pd.DataFrame:
    """Revenue, cost, profit and margin per region (best revenue first)."""
    return _performance_by(df, "region")
