"""Independent reference values for the benchmark.

Computed straight from ``data/sample_sales.csv`` with only the standard
library (``csv``). It deliberately does NOT import ``businesslens.analytics``,
pandas or any tool, so it can catch bugs in them.
"""

import csv
from collections import defaultdict
from pathlib import Path

SAMPLE_CSV = Path(__file__).resolve().parent.parent / "data" / "sample_sales.csv"


def load(path=SAMPLE_CSV):
    with open(path, newline="", encoding="utf-8") as fh:
        return [{"date": r["order_date"], "product": r["product"], "region": r["region"],
                 "revenue": float(r["revenue"]), "cost": float(r["cost"])}
                for r in csv.DictReader(fh)]


ROWS = load()


def select(product=None, region=None, start=None, end=None):
    return [r for r in ROWS
            if (product is None or r["product"] == product)
            and (region is None or r["region"] == region)
            and (start is None or r["date"] >= start)
            and (end is None or r["date"] <= end)]


def revenue(**f):
    return sum(r["revenue"] for r in select(**f))


def cost(**f):
    return sum(r["cost"] for r in select(**f))


def profit(**f):
    return revenue(**f) - cost(**f)


def margin_pct(**f):
    return 100 * profit(**f) / revenue(**f)


def revenue_by(period):
    """Revenue per 'YYYY-MM' (month) or 'YYYY-Qn' (quarter) or 'YYYY' (year)."""
    out = defaultdict(float)
    for r in ROWS:
        y, m = r["date"][:4], int(r["date"][5:7])
        key = {"month": r["date"][:7], "quarter": f"{y}Q{(m - 1) // 3 + 1}", "year": y}[period]
        out[key] += r["revenue"]
    return dict(sorted(out.items()))


def growth_pct(period, key):
    series = revenue_by(period)
    keys = list(series)
    prev = series[keys[keys.index(key) - 1]]
    return 100 * (series[key] - prev) / prev


PRODUCTS = sorted({r["product"] for r in ROWS})
REGIONS = sorted({r["region"] for r in ROWS})


def ranked(field):
    """Entities sorted by revenue, best first: [(name, revenue)]."""
    totals = defaultdict(float)
    for r in ROWS:
        totals[r[field]] += r["revenue"]
    return sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))


# Values the task statement says must hold; asserted when the benchmark starts.
KNOWN = {"revenue": 86670.0, "cost": 59616.0, "profit": 27054.0, "laptop_north_revenue": 27000.0}


def check_known_values():
    got = {"revenue": revenue(), "cost": cost(), "profit": profit(),
           "laptop_north_revenue": revenue(product="Laptop", region="North")}
    bad = {k: (got[k], v) for k, v in KNOWN.items() if abs(got[k] - v) > 1e-9}
    assert not bad, f"reference values differ from the expected known values: {bad}"
    assert len(ROWS) == 36
