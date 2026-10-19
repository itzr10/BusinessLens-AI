"""LIVE benchmark: sends real requests to a paid provider. OPT-IN ONLY.

    export BUSINESSLENS_BENCH_API_KEY="..."        # never pass keys on the command line
    python benchmarks/live_benchmark.py --live --provider openai --model <model-id>

Without --live nothing is sent. Cost is bounded: 41 questions x at most
MAX_STEPS requests each, paced by the session limiter. The run aborts on
credential, quota or model errors instead of scoring them as model failures.
Results (answers + scores, never the key) are written to benchmarks/results/.
"""

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import reference as ref  # noqa: E402
from cases import QA_CASES  # noqa: E402
from errors import ERROR_CASES  # noqa: E402
from harness import print_report, run_error_case, save  # noqa: E402
from businesslens import agent  # noqa: E402
from businesslens.providers import ErrorKind, PROVIDERS, ProviderConfig, ProviderError, create_provider  # noqa: E402

KEY_ENV = "BUSINESSLENS_BENCH_API_KEY"
FATAL = {ErrorKind.INVALID_CREDENTIALS, ErrorKind.MISSING_CREDENTIALS, ErrorKind.PERMISSION,
         ErrorKind.MODEL_NOT_FOUND, ErrorKind.TOOLS_UNSUPPORTED, ErrorKind.RATE_LIMIT,
         ErrorKind.INVALID_CONFIG}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live", action="store_true", help="required: confirms real, billable API calls")
    ap.add_argument("--provider", choices=sorted(PROVIDERS), required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--limit", type=int, default=None, help="run only the first N question cases")
    args = ap.parse_args()
    if not args.live:
        print("Refusing to run: live benchmarking is opt-in. Add --live to send real requests.")
        return 2
    key = os.environ.get(KEY_ENV, "").strip()
    if not key:
        print(f"Refusing to run: set {KEY_ENV} in your environment first.")
        return 2

    ref.check_known_values()
    try:
        provider = create_provider(ProviderConfig(args.provider, args.model, key, args.base_url))
    except ProviderError as exc:
        print(f"Cannot start: {exc}")
        return 2
    cases = QA_CASES[: args.limit] if args.limit else QA_CASES
    limiter = agent.UsageLimiter(max_questions=len(cases), max_api_calls=len(cases) * agent.MAX_STEPS,
                                 min_seconds_between_questions=0)
    results, aborted = [], None
    for case in cases:
        res = run_qa_with_limiter(case, provider, limiter)
        results.append(res)
        print(f"  {case.id}: {'skip' if res.infra_error else ('ok' if all(res.scores.values()) else 'FAIL')}")
        if res.infra_error and res.infra_error.startswith("FATAL"):
            aborted = res.infra_error
            break
        time.sleep(1.0)
    results += [run_error_case(c) for c in ERROR_CASES]  # offline by nature
    print_report(results, f"LIVE {args.provider}/{args.model}",
                 f"ABORTED EARLY: {aborted}" if aborted else "")
    path = save(results, "live", {"provider": args.provider, "model": args.model,
                                  "date": time.strftime("%Y-%m-%d"), "aborted": aborted})
    print(f"saved {path.relative_to(ROOT)}")
    return 1 if aborted else 0


def run_qa_with_limiter(case, provider, limiter):
    import harness
    res = harness.Result(case.id, case.category)
    try:
        out = agent.ask(case.question, harness.build_df(case.dataset), provider, limiter)
    except ProviderError as exc:
        res.infra_error = ("FATAL " if exc.kind in FATAL else "") + f"{exc.kind.value}: {exc}"
        return res
    except agent.AgentError as exc:
        res.infra_error = f"agent: {exc}"
        return res
    res.answer = out.text
    res.tools = [c["name"] for c in out.tool_calls]
    res.scores, res.notes = harness.score_qa(case, out.text, out.tool_calls)
    return res


if __name__ == "__main__":
    sys.exit(main())
