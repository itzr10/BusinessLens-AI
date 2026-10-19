# BusinessLens AI

A beginner-friendly business analytics app. The plan is a Streamlit dashboard
plus an AI assistant, built on top of **deterministic, tested** calculations.
The AI will never do the math itself; it will call the tested functions.

## Current status

| Stage | State |
|---|---|
| Analytics core (`businesslens/analytics.py`) | Done, tested |
| Sample dataset (`data/sample_sales.csv`) | Done |
| Streamlit dashboard (`app.py`) | Done, runs locally |
| Provider-independent AI layer (`agent.py`, `providers/`, `tools.py`) | Done, tested offline with mocks; Gemini verified live on 2026-10-19 (see below); OpenAI mocked only |
| Benchmark (72 cases) | Offline harness runs; **no live model score exists yet** |
| Git / GitHub / deployment | Not started (local only) |
| License | Proprietary, All Rights Reserved (see `LICENSE`); **not open source** |

## Setup (Python 3.12, macOS)

```bash
cd BusinessLens-AI
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Check it worked: `python --version` should print `Python 3.12.x`.

Leave the environment with `deactivate`. Re-activate it any time with
`source .venv/bin/activate`.

## Run the dashboard

```bash
source .venv/bin/activate
python -m streamlit run app.py
```

Streamlit prints a local address, normally **http://localhost:8501**. It usually
opens your browser by itself; if not, paste that address into any browser.
Press `Ctrl+C` in the terminal to stop it.

### Dashboard features

- Loads `data/sample_sales.csv` by default; upload your own CSV from the sidebar.
- Uploads are read in memory only: nothing is saved to disk (see the privacy notes for what AI questions send).
- Clear error messages for bad files (wrong columns, bad dates, non-numeric,
  negative or blank values) instead of crashes.
- Data preview, plus sidebar filters for date range, product and region. The
  date limits come from your data.
- KPI cards: total revenue, total cost, total profit, profit margin
  (shown as `N/A` when revenue is zero).
- Revenue trend by month, quarter or year, with period-over-period growth.
- Revenue and profit charts and tables by product and by region.
- Amounts show no currency symbol because the data does not specify one.

All numbers come from `businesslens/analytics.py`; the UI only displays them.

## Ask questions with your own AI provider

Open **Settings: AI provider** in the app, choose a provider, type a model ID and
your API key, press **Save** (or **Test connection**). No Python edits and no
environment variables are needed, and the old `GEMINI_API_KEY` variable is no
longer read (a process-wide variable would be shared by every user of a hosted
copy).

| Provider | Adapter | Custom base URL |
|---|---|---|
| Google Gemini | `google-genai` | No |
| OpenAI and OpenAI-compatible services (**experimental**) | `openai` (Chat Completions + function tools) | Yes, https only |

**Gemini is the only live-verified provider**, and only by a small smoke test with
`gemini-3.8-flash` (see the table below). The **OpenAI adapter is experimental: it is
implemented and tested against mocked SDK responses only, and has never been run
against the real API.** Expect to hit incompatibilities, especially with third-party
"OpenAI-compatible" endpoints.

The model must support **tool (function) calling**. The model ID is free text:
there is no built-in list of "supported" models.

### Privacy and data handling

- When you ask a question, **your question text and the aggregated tool results are sent
  to the provider you selected.** Tool results include totals, rankings, growth
  figures and the **product and region names** in your data. Raw CSV rows are not sent.
- Providers differ in logging, retention, training and regional processing. Read the
  policy of the provider and plan tied to your key before use. BusinessLens does not
  control or guarantee any of that.
- The key lives only in the Streamlit session's memory: it is never written to disk,
  logged, or included in error messages, and each browser session has its own copy.
  **Clear key** removes it; closing the session discards it. It may still be visible
  to anyone who can see your screen while you type it (the field is masked).
- Do not paste real keys into screenshots, issues or the README.

### Optional base URL (OpenAI-compatible services)

Only the OpenAI adapter accepts one. It must be `https://`, use the default port, have
no credentials/query string, be a hostname (not an IP) and resolve only to public
addresses. Localhost, private ranges, link-local/metadata addresses and `.local` /
`.internal` names are refused, and HTTP redirects are not followed. DNS rebinding is
a residual risk: on a hosted deployment also restrict outbound traffic at the network
level. Compatibility of a third-party endpoint is **not** guaranteed.

### Error handling

Failures are mapped to the same short messages for every provider: missing or
invalid key, permission denied, model not found, rate limit/quota, timeout,
connection problem, model without tool calling, empty answer, other provider error.

### Cost and abuse limits (per browser session)

30 questions, 150 provider requests, 2 s between questions, 6 model round-trips and
10 tool calls per question, 500-character questions, 1,024 output tokens, 30 s
timeout, no automatic retries. Uploads: 5 MB, 100,000 rows, 500 distinct products
or regions. These are safeguards, not billing controls: set spend limits at your
provider as well.

### Tested provider/model combinations

| Provider | Model | Status |
|---|---|---|
| Gemini adapter | `gemini-3.8-flash` | **Live-verified 2026-10-19**: connection test, a tool-calling question (Laptop/North revenue = 27,000 via `get_filtered_summary`) and an unsupported question (churn; declined, no tools). Also live: invalid key -> `invalid_credentials`; unknown model -> `model_not_found`. Only about 6 requests; **no live benchmark score** |
| Gemini adapter | `gemini-2.5-flash` | Not usable for new keys: the API returns 404 "no longer available to new users" (reported as model-not-found) |
| OpenAI adapter | any | **Mocked SDK responses only. Not tested live.** |

The default Gemini model changes over time; check Google's model list if you get a model-not-found error. Update this table only after a live run (`BUSINESSLENS_LIVE=1`, see below) and record
the model ID, date and result. Known limitations: compatible endpoints may reject
`tool_choice`, `max_tokens` or tool schemas; reasoning models may be slow or need a
larger output limit; free tiers hit rate limits quickly (each question uses 2+ requests).

### Why not LiteLLM?

LiteLLM's Python SDK (MIT, Python 3.10-3.14) was evaluated and **not adopted**:
versions 1.82.7 and 1.82.8 on PyPI were malicious in March 2026 (credential
stealer; reported as a maintainer-account compromise), its latest release pulls
about 20 direct dependencies (boto3, aiohttp, huggingface-hub, tokenizers ...) for
two providers, and its security-advisory page keeps listing SSRF, file-read and
credential-exfiltration issues around request routing parameters. Two thin
adapters over the vendors' own SDKs are a smaller attack surface. New providers can
be added by implementing `Provider` in `businesslens/providers/base.py`.

## Benchmark

```bash
python benchmarks/offline_benchmark.py        # free, offline, no key
```

72 cases with stable IDs: 41 questions (`NUM-`, `RANK-`, `GROW-`, `FILT-`, `EDGE-`,
`AMBIG-`, `UNSUP-`, `INJ-`), 24 error/limit cases (`CRED-`, `PROV-`, `RATE-`,
`CFG-`, run through the real adapters with fake SDK clients) and 7 adversarial
model scripts (`ADV-`). Expected values come from `benchmarks/reference.py`, which
reads the CSV with the standard library only (revenue 86,670; cost 59,616;
profit 27,054; Laptop/North revenue 27,000). Four dimensions are scored separately:
tool selection, numerical correctness, unsupported-question handling, error handling
(rules in `benchmarks/harness.py`).

**The offline run does not measure any AI model.** It replays a scripted "oracle" that
calls the expected tools and writes its answer from the tool output, so it checks the
tools, agent loop, limits, error mapping and scoring code. Model quality is measured
only by the opt-in live run:

```bash
export BUSINESSLENS_BENCH_API_KEY="..."   # never put keys on the command line
python benchmarks/live_benchmark.py --live --provider openai --model <model-id>
```

This sends up to 41 x 6 billable requests. Without `--live` and the key it refuses to
run. Results go to `benchmarks/results/` (git-ignored). Live pytest smoke tests are
also opt-in: `BUSINESSLENS_LIVE=1 BUSINESSLENS_LIVE_<PROVIDER>_KEY=... BUSINESSLENS_LIVE_<PROVIDER>_MODEL=...`.

Live scoring is strict and string-based (plain numbers, percent for margins/growth);
a correct answer phrased unusually can fail, and rules for "decline" and "clarify" are
keyword-based. Treat live scores as indicative, and read the saved answers.

## License

BusinessLens AI is **proprietary software, All Rights Reserved**. The source is visible
for evaluation, but you may not use it commercially, modify it, redistribute it or host it
as a service without written permission. This is not an open-source license. Third-party
dependencies keep their own licenses: see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Security review

See [SECURITY_REVIEW.md](SECURITY_REVIEW.md) for scope, checks, findings and remaining
risks. It is a self-review, not a formal audit.

## Run the tests

```bash
python -m pytest -q    # offline; live tests are skipped unless BUSINESSLENS_LIVE=1
```


## Using the analytics

```python
from businesslens.analytics import (
    load_sales_data, total_revenue, total_profit, profit_margin,
    period_growth, product_performance, region_performance,
)

df = load_sales_data("data/sample_sales.csv")
total_revenue(df)            # 86670.0
profit_margin(df)            # 0.3121... (a fraction, i.e. 31.2%)
period_growth(df, "month")   # revenue + growth per month
product_performance(df)      # revenue/cost/profit/margin per product
```

## Data rules

Input needs five columns: `order_date`, `product`, `region`, `revenue`, `cost`.

- Bad data raises `DataValidationError` with the offending row numbers
  (empty data, missing columns, unreadable dates, non-numeric, infinite or
  negative revenue/cost, blank product/region).
- Ratios are fractions (`0.25` = 25%).
- A ratio with a zero denominator is *undefined*: `None` for single values,
  `NaN` in tables. It is never silently turned into 0.
- Growth compares each period to the one right before it. Months with no sales
  inside the date range show as revenue 0, so gaps are visible.
- Negative values (refunds/returns) are not supported yet.

## Project layout

```
app.py                      Streamlit dashboard (layout only)
businesslens/analytics.py   pure calculation functions (no UI, no AI)
businesslens/dashboard_helpers.py  upload parsing/limits, filtering, number formatting
businesslens/tools.py       approved tools the model may request (provider-independent)
businesslens/agent.py       tool loop, limits, answer sanitising (provider-independent)
businesslens/providers/     Gemini and OpenAI adapters, URL checks, scripted test doubles
benchmarks/                 reference values, cases, harness, offline and live runners
.streamlit/config.toml      upload cap, no telemetry, minimal error details
tests/test_analytics.py     pytest tests with hand-checkable numbers
tests/test_dashboard_helpers.py  tests for the dashboard helpers
tests/test_tools.py, test_agent.py, test_providers.py, test_benchmark.py, test_app_settings.py
data/sample_sales.csv       36 synthetic rows, Jan-Jun 2025
requirements.txt            pinned dependencies
LICENSE, THIRD_PARTY_NOTICES.md   proprietary notice; dependency licenses
pytest.ini                  test configuration
```

`.gitignore` already excludes `.venv/`, `.env`, caches, and everything in
`data/` except the sample file, so private data stays out of Git later.

## Known limitations

- Refunds/returns (negative revenue or cost) are rejected, not interpreted.
- Filters use the whole row set; very large files are not optimised.
- The dashboard layout has been checked with automated headless runs, not yet
  reviewed by eye in a browser.
