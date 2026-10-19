"""Offline benchmark: no network, no API key, no cost.

Run: python benchmarks/offline_benchmark.py

Verifies the tools against independent reference values (stdlib csv), the agent
loop and its limits, provider error mapping (real adapters, fake SDK clients),
and the scoring code. It does NOT measure any language model: see harness.py and
live_benchmark.py for that.
"""

import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import reference as ref  # noqa: E402
from adversarial import ADVERSARIES  # noqa: E402
from cases import QA_CASES  # noqa: E402
from errors import ERROR_CASES  # noqa: E402
from harness import (oracle_provider, print_report, run_adversary, run_error_case, run_qa,  # noqa: E402
                     save, build_df)
from businesslens.tools import TOOLS, run_tool  # noqa: E402


def main() -> int:
    logging.getLogger("businesslens").setLevel(logging.CRITICAL)  # expected-failure cases log errors
    ref.check_known_values()
    results = [run_qa(c, oracle_provider(c)) for c in QA_CASES]
    results += [run_error_case(c) for c in ERROR_CASES]
    results += [run_adversary(a) for a in ADVERSARIES]

    df = build_df("sample")
    start = time.perf_counter()
    for _ in range(200):
        for name in TOOLS:
            run_tool(name, {}, df)
    per_call = (time.perf_counter() - start) / (200 * len(TOOLS)) * 1000
    print_report(results, "OFFLINE (oracle script, no model)",
                 f"{per_call:.2f} ms per tool call. Offline scores do not measure any AI model.")
    path = save(results, "offline", {"model": None})
    print(f"saved {path.relative_to(ROOT)}")
    failed = any(not all(r.scores.values()) for r in results)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
