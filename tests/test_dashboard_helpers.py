"""Tests for businesslens.dashboard_helpers (no Streamlit needed)."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from businesslens import dashboard_helpers as h
from businesslens.analytics import DataValidationError, validate_sales_data

GOOD_CSV = (
    b"order_date,product,region,revenue,cost\n"
    b"2025-01-05,A,North,100,60\n"
    b"2025-01-20,B,South,200,100\n"
    b"2025-02-10 15:30:00,A,North,300,200\n"
    b"2025-04-01,B,North,400,240\n"
)


@pytest.fixture
def df():
    return h.read_uploaded_csv(GOOD_CSV)


# ---------------------------- read_uploaded_csv ---------------------------

def test_read_good_csv(df):
    assert len(df) == 4
    assert df["revenue"].sum() == 1000


@pytest.mark.parametrize("content", [b"", b"   \n  "])
def test_read_empty_upload(content):
    with pytest.raises(DataValidationError, match="empty"):
        h.read_uploaded_csv(content)


def test_read_missing_columns():
    with pytest.raises(DataValidationError, match="Missing required column"):
        h.read_uploaded_csv(b"a,b\n1,2\n")


def test_read_header_only():
    with pytest.raises(DataValidationError, match="no rows"):
        h.read_uploaded_csv(b"order_date,product,region,revenue,cost\n")


def test_read_non_utf8_bytes():
    with pytest.raises(DataValidationError, match="UTF-8"):
        h.read_uploaded_csv(b"\xff\xfe\x00\x80\x81")


def test_read_bad_value_reports_row():
    bad = GOOD_CSV.replace(b"200,100", b"abc,100")
    with pytest.raises(DataValidationError, match=r"revenue.*2"):
        h.read_uploaded_csv(bad)


def test_read_keeps_negative_rule():
    bad = GOOD_CSV.replace(b"100,60", b"-100,60")
    with pytest.raises(DataValidationError, match="Negative"):
        h.read_uploaded_csv(bad)


# ---------------------------- date_bounds / filter ------------------------

def test_date_bounds(df):
    assert h.date_bounds(df) == (date(2025, 1, 5), date(2025, 4, 1))


def test_filter_full_range_keeps_everything(df):
    out = h.filter_sales(df, date(2025, 1, 5), date(2025, 4, 1))
    pd.testing.assert_frame_equal(out, df)


def test_filter_date_range_is_inclusive_including_time_of_day(df):
    # The Feb 10 row is at 15:30 and must still be included on its own day.
    out = h.filter_sales(df, date(2025, 2, 10), date(2025, 2, 10))
    assert out["revenue"].tolist() == [300]


def test_filter_by_product_and_region(df):
    out = h.filter_sales(df, date(2025, 1, 1), date(2025, 12, 31), ["A"], ["North"])
    assert out["revenue"].tolist() == [100, 300]


def test_filter_none_means_no_filter_and_empty_list_means_nothing(df):
    wide = (date(2025, 1, 1), date(2025, 12, 31))
    assert len(h.filter_sales(df, *wide, products=None, regions=None)) == 4
    assert h.filter_sales(df, *wide, products=[]).empty
    assert h.filter_sales(df, *wide, regions=[]).empty


def test_filter_can_return_empty(df):
    assert h.filter_sales(df, date(2026, 1, 1), date(2026, 1, 31)).empty


def test_filtered_data_is_still_valid_for_analytics(df):
    out = h.filter_sales(df, date(2025, 1, 1), date(2025, 1, 31))
    assert validate_sales_data(out)["revenue"].sum() == 300


def test_filter_does_not_mutate_input(df):
    before = df.copy()
    h.filter_sales(df, date(2025, 1, 1), date(2025, 1, 31), ["A"], ["North"])
    pd.testing.assert_frame_equal(df, before)


# ---------------------------- formatting ----------------------------------

def test_format_number():
    assert h.format_number(86670) == "86,670.00"
    assert h.format_number(0) == "0.00"
    assert h.format_number(-1234.5) == "-1,234.50"


def test_format_percent():
    assert h.format_percent(0.3121495) == "31.2%"
    assert h.format_percent(0) == "0.0%"
    assert h.format_percent(-0.4) == "-40.0%"


@pytest.mark.parametrize("fn", [h.format_number, h.format_percent])
@pytest.mark.parametrize("missing", [None, float("nan"), np.nan])
def test_missing_values_show_na(fn, missing):
    assert fn(missing) == "N/A"


def test_upload_size_row_and_label_limits():
    big = b"order_date,product,region,revenue,cost\n" + b"2025-01-01,A,N,1,1\n" * 10
    assert len(h.read_uploaded_csv(big)) == 10
    with pytest.raises(DataValidationError, match="larger than"):
        h.read_uploaded_csv(b"x" * (h.MAX_UPLOAD_BYTES + 1))
    many = "order_date,product,region,revenue,cost\n" + "".join(
        f"2025-01-01,P{i},N,1,1\n" for i in range(h.MAX_DISTINCT_LABELS + 1))
    with pytest.raises(DataValidationError, match="Too many distinct"):
        h.read_uploaded_csv(many.encode())
    long_name = "order_date,product,region,revenue,cost\n2025-01-01," + "p" * 101 + ",N,1,1\n"
    with pytest.raises(DataValidationError, match="longer than"):
        h.read_uploaded_csv(long_name.encode())
