"""Small helpers for the Streamlit dashboard.

This module does NOT import Streamlit, so it can be tested with plain pytest.
It contains no business formulas: all metrics come from ``analytics.py``.
"""

from __future__ import annotations

import io
import math
from collections.abc import Iterable
from datetime import date

import pandas as pd

from businesslens.analytics import DataValidationError, validate_sales_data

NOT_AVAILABLE = "N/A"
MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # keep in step with .streamlit/config.toml maxUploadSize
MAX_UPLOAD_ROWS = 100_000
MAX_DISTINCT_LABELS = 500  # products / regions: keeps tool results and prompts small


def read_uploaded_csv(content: bytes) -> pd.DataFrame:
    """Parse CSV bytes (kept in memory, never written to disk) and validate them.

    Raises ``DataValidationError`` with a user-friendly message on any problem.
    """
    if not content or not content.strip():
        raise DataValidationError("The uploaded file is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise DataValidationError(
            f"The file is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB. Upload a smaller file.")
    try:
        raw = pd.read_csv(io.BytesIO(content), nrows=MAX_UPLOAD_ROWS + 1)
    except UnicodeDecodeError as exc:
        raise DataValidationError(
            "The file could not be read as text. Please upload a UTF-8 CSV file."
        ) from exc
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise DataValidationError(f"The file is not a valid CSV: {exc}") from exc
    if len(raw) > MAX_UPLOAD_ROWS:
        raise DataValidationError(f"The file has more than {MAX_UPLOAD_ROWS:,} rows.")
    clean = validate_sales_data(raw)
    for col in ("product", "region"):
        if clean[col].nunique() > MAX_DISTINCT_LABELS or clean[col].str.len().max() > 100:
            raise DataValidationError(
                f"Too many distinct {col} values (max {MAX_DISTINCT_LABELS}) or a {col} name "
                "longer than 100 characters.")
    return clean


def date_bounds(df: pd.DataFrame) -> tuple[date, date]:
    """First and last order date in the data, as plain dates."""
    return df["order_date"].min().date(), df["order_date"].max().date()


def filter_sales(
    df: pd.DataFrame,
    start: date,
    end: date,
    products: Iterable[str] | None = None,
    regions: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Keep rows inside the date range (both ends included) and chosen products/regions.

    ``products`` / ``regions`` of ``None`` means "do not filter on this".
    An empty list means "nothing selected", which returns no rows.
    Rows are only selected, never changed. The result may be empty.
    """
    days = df["order_date"].dt.normalize()  # drop time-of-day so the end date is inclusive
    keep = (days >= pd.Timestamp(start)) & (days <= pd.Timestamp(end))
    if products is not None:
        keep &= df["product"].isin(list(products))
    if regions is not None:
        keep &= df["region"].isin(list(regions))
    return df[keep].reset_index(drop=True)


def format_number(value: float | None) -> str:
    """Thousands separators and two decimals, e.g. ``86,670.00``.

    No currency symbol: the data does not say which currency it uses.
    """
    if _is_missing(value):
        return NOT_AVAILABLE
    return f"{value:,.2f}"


def format_percent(value: float | None) -> str:
    """Show a fraction as a percentage (0.3121 -> ``31.2%``); undefined -> ``N/A``."""
    if _is_missing(value):
        return NOT_AVAILABLE
    return f"{value * 100:.1f}%"


def _is_missing(value: float | None) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))
