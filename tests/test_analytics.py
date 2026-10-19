"""Tests for businesslens.analytics.

Datasets are tiny so every expected number can be checked by hand.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from businesslens import analytics as a
from businesslens.analytics import DataValidationError

SAMPLE_CSV = Path(__file__).resolve().parent.parent / "data" / "sample_sales.csv"


@pytest.fixture
def small():
    """4 rows. Revenue 1000, cost 600, profit 400, margin 0.4."""
    return pd.DataFrame(
        {
            "order_date": ["2025-01-05", "2025-01-20", "2025-02-10", "2025-04-01"],
            "product": ["A", "B", "A", "B"],
            "region": ["North", "South", "North", "North"],
            "revenue": [100, 200, 300, 400],
            "cost": [60, 100, 200, 240],
        }
    )


# ---------------------------- validation ----------------------------------

def test_validate_returns_typed_copy_and_does_not_mutate(small):
    original = small.copy()
    clean = a.validate_sales_data(small)
    assert pd.api.types.is_datetime64_any_dtype(clean["order_date"])
    assert clean["revenue"].dtype == "float64"
    pd.testing.assert_frame_equal(small, original)


def test_validate_is_idempotent(small):
    once = a.validate_sales_data(small)
    pd.testing.assert_frame_equal(once, a.validate_sales_data(once))


def test_validate_trims_text(small):
    small.loc[0, "product"] = "  A "
    assert a.validate_sales_data(small)["product"].iloc[0] == "A"


def test_validate_accepts_numeric_strings(small):
    small["revenue"] = small["revenue"].astype(str)
    assert a.total_revenue(small) == 1000


def test_not_a_dataframe():
    with pytest.raises(DataValidationError, match="DataFrame"):
        a.validate_sales_data([1, 2, 3])


def test_empty_dataframe(small):
    with pytest.raises(DataValidationError, match="no rows"):
        a.validate_sales_data(small.iloc[0:0])


def test_missing_columns(small):
    with pytest.raises(DataValidationError, match="region, cost"):
        a.validate_sales_data(small.drop(columns=["cost", "region"]))


def test_invalid_date_reports_row(small):
    small.loc[2, "order_date"] = "not-a-date"
    with pytest.raises(DataValidationError, match=r"order_date.*3"):
        a.validate_sales_data(small)


def test_missing_date(small):
    small.loc[0, "order_date"] = None
    with pytest.raises(DataValidationError, match="order_date"):
        a.validate_sales_data(small)


def test_non_numeric_revenue(small):
    small["revenue"] = small["revenue"].astype(object)
    small.loc[1, "revenue"] = "abc"
    with pytest.raises(DataValidationError, match=r"revenue.*2"):
        a.validate_sales_data(small)


def test_missing_cost(small):
    small.loc[3, "cost"] = np.nan
    with pytest.raises(DataValidationError, match="cost"):
        a.validate_sales_data(small)


def test_infinite_value(small):
    small["revenue"] = small["revenue"].astype(float)
    small.loc[0, "revenue"] = np.inf
    with pytest.raises(DataValidationError, match="revenue"):
        a.validate_sales_data(small)


def test_negative_value(small):
    small.loc[0, "cost"] = -5
    with pytest.raises(DataValidationError, match="Negative cost"):
        a.validate_sales_data(small)


@pytest.mark.parametrize("bad", [None, "", "   "])
def test_blank_product(small, bad):
    small.loc[0, "product"] = bad
    with pytest.raises(DataValidationError, match="Missing product"):
        a.validate_sales_data(small)


def test_error_lists_at_most_five_rows():
    df = pd.DataFrame(
        {
            "order_date": ["x"] * 8,
            "product": ["A"] * 8,
            "region": ["N"] * 8,
            "revenue": [1] * 8,
            "cost": [1] * 8,
        }
    )
    with pytest.raises(DataValidationError, match=r"1, 2, 3, 4, 5 \(and 3 more\)"):
        a.validate_sales_data(df)


def test_load_missing_file(tmp_path):
    with pytest.raises(DataValidationError, match="not found"):
        a.load_sales_data(tmp_path / "nope.csv")


def test_load_empty_file(tmp_path):
    p = tmp_path / "empty.csv"
    p.write_text("")
    with pytest.raises(DataValidationError, match="empty"):
        a.load_sales_data(p)


def test_load_header_only_file(tmp_path):
    p = tmp_path / "header.csv"
    p.write_text("order_date,product,region,revenue,cost\n")
    with pytest.raises(DataValidationError, match="no rows"):
        a.load_sales_data(p)


# ---------------------------- headline metrics ----------------------------

def test_totals(small):
    assert a.total_revenue(small) == 1000
    assert a.total_cost(small) == 600
    assert a.total_profit(small) == 400


def test_profit_margin(small):
    assert a.profit_margin(small) == pytest.approx(0.4)


def test_negative_profit(small):
    small["cost"] = [200, 300, 400, 500]  # cost 1400 > revenue 1000
    assert a.total_profit(small) == -400
    assert a.profit_margin(small) == pytest.approx(-0.4)


def test_margin_is_none_when_revenue_zero(small):
    small["revenue"] = 0
    assert a.profit_margin(small) is None
    assert a.total_profit(small) == -600


# ---------------------------- period growth -------------------------------

def test_monthly_growth_with_gap_month(small):
    # Jan 300, Feb 300, Mar 0 (gap), Apr 400
    g = a.period_growth(small, "month")
    assert g["period"].tolist() == ["2025-01", "2025-02", "2025-03", "2025-04"]
    assert g["revenue"].tolist() == [300, 300, 0, 400]
    assert np.isnan(g["growth"].iloc[0])  # first period: nothing to compare
    assert g["growth"].iloc[1] == pytest.approx(0.0)
    assert g["growth"].iloc[2] == pytest.approx(-1.0)  # 300 -> 0 is -100%
    assert np.isnan(g["growth"].iloc[3])  # previous revenue is 0: undefined


def test_default_period_is_month(small):
    pd.testing.assert_frame_equal(a.period_growth(small), a.period_growth(small, "month"))


def test_quarterly_growth(small):
    # Q1 = 600, Q2 = 400 -> growth = (400 - 600) / 600
    g = a.period_growth(small, "quarter")
    assert g["period"].tolist() == ["2025Q1", "2025Q2"]
    assert g["revenue"].tolist() == [600, 400]
    assert g["growth"].iloc[1] == pytest.approx(-1 / 3)


def test_yearly_growth():
    df = pd.DataFrame(
        {
            "order_date": ["2024-06-01", "2025-06-01"],
            "product": ["A", "A"],
            "region": ["N", "N"],
            "revenue": [100, 150],
            "cost": [50, 50],
        }
    )
    g = a.period_growth(df, "year")
    assert g["period"].tolist() == ["2024", "2025"]
    assert g["growth"].iloc[1] == pytest.approx(0.5)


def test_single_period_has_no_growth():
    df = pd.DataFrame(
        {
            "order_date": ["2025-01-01"],
            "product": ["A"],
            "region": ["N"],
            "revenue": [10],
            "cost": [5],
        }
    )
    g = a.period_growth(df)
    assert len(g) == 1 and np.isnan(g["growth"].iloc[0])


def test_invalid_period_name(small):
    with pytest.raises(ValueError, match="period must be one of"):
        a.period_growth(small, "week")


def test_growth_validates_input(small):
    with pytest.raises(DataValidationError):
        a.period_growth(small.iloc[0:0])


# ---------------------------- product / region ----------------------------

def test_product_performance(small):
    p = a.product_performance(small)
    # B: revenue 600, cost 340. A: revenue 400, cost 260.
    assert p["product"].tolist() == ["B", "A"]
    assert p["revenue"].tolist() == [600, 400]
    assert p["cost"].tolist() == [340, 260]
    assert p["profit"].tolist() == [260, 140]
    assert p["profit_margin"].tolist() == pytest.approx([260 / 600, 140 / 400])


def test_region_performance(small):
    r = a.region_performance(small)
    # North: revenue 800, cost 500. South: revenue 200, cost 100.
    assert r["region"].tolist() == ["North", "South"]
    assert r["revenue"].tolist() == [800, 200]
    assert r["profit"].tolist() == [300, 100]
    assert r["profit_margin"].tolist() == pytest.approx([300 / 800, 0.5])


def test_performance_ties_sorted_by_name():
    df = pd.DataFrame(
        {
            "order_date": ["2025-01-01"] * 3,
            "product": ["Zed", "Alpha", "Mid"],
            "region": ["N"] * 3,
            "revenue": [100, 100, 100],
            "cost": [10, 10, 10],
        }
    )
    assert a.product_performance(df)["product"].tolist() == ["Alpha", "Mid", "Zed"]


def test_performance_zero_revenue_margin_is_nan(small):
    small.loc[small["product"] == "A", "revenue"] = 0
    p = a.product_performance(small).set_index("product")
    assert np.isnan(p.loc["A", "profit_margin"])
    assert p.loc["A", "profit"] == -260
    assert p.loc["B", "profit_margin"] == pytest.approx(260 / 600)


def test_performance_does_not_mutate_input(small):
    before = small.copy()
    a.product_performance(small)
    a.region_performance(small)
    a.period_growth(small)
    pd.testing.assert_frame_equal(small, before)


# ---------------------------- sample dataset ------------------------------

def test_sample_dataset_loads_with_known_totals():
    df = a.load_sales_data(SAMPLE_CSV)
    assert len(df) == 36
    # By hand: units sold per product over 6 months are Laptop 27, Headphones 39,
    # Monitor 51; North+South price factors add up to 1.8.
    #   revenue = 1.8 * (1000*27 + 150*39 + 300*51) = 86670
    #   cost    = 1.8 * (700*27 + 90*39 + 210*51)   = 59616
    assert a.total_revenue(df) == pytest.approx(86670)
    assert a.total_cost(df) == pytest.approx(59616)
    assert a.total_profit(df) == pytest.approx(27054)
    assert a.product_performance(df)["product"].tolist() == ["Laptop", "Monitor", "Headphones"]
    assert set(a.product_performance(df)["product"]) == {"Laptop", "Headphones", "Monitor"}
    assert set(a.region_performance(df)["region"]) == {"North", "South"}
    assert len(a.period_growth(df)) == 6  # Jan..Jun 2025
