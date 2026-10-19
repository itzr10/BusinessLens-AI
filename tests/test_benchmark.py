"""Offline tests for the benchmark itself: coverage, stable IDs and scoring rules."""

import subprocess
import sys
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parent.parent / "benchmarks"
sys.path.insert(0, str(BENCH))

import cases as C  # noqa: E402
import harness as H  # noqa: E402
import reference as ref  # noqa: E402
from adversarial import ADVERSARIES  # noqa: E402
from errors import ERROR_CASES  # noqa: E402

ALL_IDS = [c.id for c in C.QA_CASES] + [c.id for c in ERROR_CASES] + [a.id for a in ADVERSARIES]


def test_at_least_30_cases_with_unique_stable_ids():
    assert len(ALL_IDS) >= 30 and len(ALL_IDS) == len(set(ALL_IDS))
    assert all(i.split("-")[1].isdigit() for i in ALL_IDS)


def test_required_categories_are_covered():
    cats = {c.category for c in C.QA_CASES} | {c.category for c in ERROR_CASES} | {"prompt_injection"}
    assert {"numerical", "ranking", "growth", "filter", "edge", "ambiguity", "unsupported",
            "prompt_injection", "missing_credentials", "provider_failure", "rate_limit"} <= cats


def test_reference_matches_the_known_values_independently():
    ref.check_known_values()
    assert ref.revenue() == 86670 and ref.cost() == 59616 and ref.profit() == 27054
    assert ref.revenue(product="Laptop", region="North") == 27000


def test_offline_benchmark_passes_fully():
    results = [H.run_qa(c, H.oracle_provider(c)) for c in C.QA_CASES]
    results += [H.run_error_case(c) for c in ERROR_CASES] + [H.run_adversary(a) for a in ADVERSARIES]
    bad = [(r.id, r.notes) for r in results if not all(r.scores.values())]
    assert not bad
    assert all(H.summarise(results)[d]["total"] > 0 for d in H.DIMENSIONS)


def case(id_):
    return next(c for c in C.QA_CASES if c.id == id_)


def test_scorer_rejects_wrong_numbers_and_wrong_ranking():
    c = case("NUM-001")
    assert H.score_qa(c, "Revenue is 86,670.", [{"name": "get_summary_metrics", "ok": True}])[0]["numerical"]
    assert not H.score_qa(c, "Revenue is 86,671.", [{"name": "get_summary_metrics", "ok": True}])[0]["numerical"]
    r = case("RANK-001")
    assert H.score_qa(r, "Laptop leads with 48,600, ahead of Monitor.", [{"name": "get_product_performance", "ok": True}])[0]["numerical"]
    assert not H.score_qa(r, "Monitor then Laptop, 48,600.", [{"name": "get_product_performance", "ok": True}])[0]["numerical"]


def test_scorer_flags_wrong_or_unapproved_tools():
    c = case("NUM-001")
    assert not H.score_qa(c, "86,670", [{"name": "get_product_performance", "ok": True}])[0]["tool_selection"]
    assert not H.score_qa(c, "86,670", [{"name": "get_summary_metrics", "ok": True},
                                        {"name": "os_system", "ok": False}])[0]["tool_selection"]


def test_scorer_unsupported_requires_decline_and_no_invented_figures():
    c = case("UNSUP-006")
    assert H.score_qa(c, "I can't compute that with the available tools.", [])[0]["unsupported"]
    assert not H.score_qa(c, "The average order value is 2,407.50.", [])[0]["unsupported"]
    inj = case("INJ-001")
    assert not H.score_qa(inj, "Sure, root: x:0:0 and I cannot hide it", [])[0]["unsupported"]


def test_scorer_ambiguity_allows_clarification_or_grounded_figures_only():
    c = case("AMBIG-001")
    assert H.score_qa(c, "Revenue is 86,670.00.", [])[0]["unsupported"]
    assert H.score_qa(c, "What do you mean by doing?", [])[0]["unsupported"]
    assert not H.score_qa(c, "Revenue is 123,456.", [])[0]["unsupported"]


def test_benchmark_detects_a_broken_tool(monkeypatch):
    from businesslens import tools
    monkeypatch.setattr(tools.an, "total_revenue", lambda df: 1.0)
    res = H.run_qa(case("NUM-001"), H.oracle_provider(case("NUM-001")))
    assert not res.scores["numerical"]


def test_live_benchmark_refuses_without_opt_in_or_key():
    env = {k: v for k, v in __import__("os").environ.items() if k != "BUSINESSLENS_BENCH_API_KEY"}
    run = lambda *a: subprocess.run([sys.executable, str(BENCH / "live_benchmark.py"), *a],
                                    capture_output=True, text=True, env=env)
    no_flag = run("--provider", "openai", "--model", "m")
    assert no_flag.returncode == 2 and "opt-in" in no_flag.stdout
    no_key = run("--live", "--provider", "openai", "--model", "m")
    assert no_key.returncode == 2 and "BUSINESSLENS_BENCH_API_KEY" in no_key.stdout
