# BusinessLens AI: security self-review

**This is a self-review by the developer's AI assistant, not a formal security audit,
and it does not guarantee the product is secure.** Date: 2026-10-19. Code state: local
only: a local Git repository has been initialized and the release files are staged, but nothing has been committed or pushed, and nothing is deployed. Findings below are about this code; the app has not been
penetration-tested or deployed.

## Scope and checks performed

| Area | Checks |
|---|---|
| Secrets | Read every code path that touches a key; tests assert keys are absent from `repr`/`str` of configs, user-facing errors, and captured logs for both adapters (401/403/404/429/5xx/timeouts/unknown exceptions); regex scan of the tree for key shapes and for the real `GEMINI_API_KEY` value present in this shell (not found in any file); fake keys exist only in tests and are obviously labelled |
| Session isolation | No module-level mutable credential or limiter; provider, label and limiter live in `st.session_state`; headless test shows one session's saved provider is invisible to another |
| Uploads | 5 MB cap (Streamlit config **and** code), 100,000 rows, 500 distinct products/regions, 100-char names, in-memory only |
| Prompt injection | Strict allowlist (`TOOLS`), per-argument type/enum/length checks, product/region matched against values in the data, ISO dates only; 7 scripted hostile-model cases; poisoned-CSV question cases (live only); markdown links/images/URLs stripped from answers |
| Bounded loops | 6 round-trips, 4 calls per reply, 10 calls per question, 20 kB tool result cap |
| Cost / DoS | Per-session limiter (30 questions, 150 provider requests, 2 s pacing), 1,024 output tokens, 30 s timeout, `max_retries=0` |
| Endpoint safety | https-only, public-address-only base URL validation, redirects disabled, only the OpenAI adapter accepts one |
| Dependencies | `pip-audit` on the installed environment: no known vulnerabilities (OSV/PyPI data, run 2026-10-19). LiteLLM evaluated and rejected (see README) |
| Repo hygiene | `.gitignore` covers `.venv`, `.env*`, `*.key`, secrets, private `data/*`, benchmark results; the staged file list and staged content were checked for secrets before the first commit (test fixtures only: fake keys and rejected-URL cases); nothing has been committed or pushed yet, and Git history has not been reviewed |

## Findings

| # | Severity | Finding | Status |
|---|---|---|---|
| 1 | High | A model reply could carry a markdown image/link whose URL encodes aggregated data; the browser would fetch it (exfiltration via poisoned CSV product names) | Fixed: `sanitize_answer` strips images, links and bare URLs (tested). The system prompt also forbids them; do not rely on the prompt alone |
| 2 | High | A process-wide `GEMINI_API_KEY` would be shared by every user of a hosted copy | Fixed: env fallback removed; keys are per session |
| 3 | Medium | Old tool layer only checked argument *names*, not types/length; the model could send huge or non-string values | Fixed: type, enum and length checks; unknown product/region rejected |
| 4 | Medium | Uploads had no size, row or label limits (memory/DoS, oversized prompts) | Fixed (limits above) |
| 5 | Medium | No spend/pace guard: one user could issue unlimited paid requests, and tool loops were bounded only per question | Fixed per session. **Not** a billing control; a user can reload the page to reset the counters |
| 6 | Medium | A user-supplied base URL enables SSRF and redirect-based key leakage | Mitigated (validation, no redirects). Residual: DNS rebinding between validation and connect; hosted deployment needs egress filtering |
| 7 | Low | Streamlit tracebacks can show exception text | Mitigated: `client.showErrorDetails = "type"` |
| 8 | Low | `duckdb` was pinned in `requirements.txt` but unused | Fixed: grep found no import in app, tests, benchmarks or docs; removed from `requirements.txt` (the local `.venv` still has it installed) |
| 9 | Low | The masked key field still keeps the key in widget state for the session; browser autofill/extensions or a shoulder-surfer could see it | Open, inherent; documented. "Clear key" removes it |
| 10 | Info | `httpx2` (new HTTP stack used by `openai` 3.x) is imported directly by the adapter | Pinned explicitly |

## Remaining risks and release blockers

- **Limited live verification.** Gemini (`gemini-3.8-flash`) was exercised live on 2026-10-19 (connection, tool call, invalid key, unknown model); rate-limit, timeout and unsupported-tools errors were not triggered live. OpenAI has not been run live; its real-API behaviour (tool-schema acceptance, error shapes, `max_completion_tokens`, rate-limit exceptions) is only mocked. Error classification relies on SDK exception types and message wording that vendors may change.
- **No live model benchmark score exists.** Prompt-injection resistance of real models is untested; only the deterministic defences are.
- **Disclosure of data to providers is by design**: product/region names and aggregates leave the machine. Business users with confidential data need their own legal/privacy review; retention terms are the provider's.
- **No authentication, rate limiting by identity, or server-side budget.** Fine for local use; a public deployment would need all three plus HTTPS termination, egress filtering and monitoring.
- The Streamlit upload cap and `showErrorDetails` live in `.streamlit/config.toml`; a deployment that ignores that file loses them (code-level 5 MB check remains).
- Not assessed: Streamlit/Plotly front-end vulnerabilities beyond `pip-audit`, hosting, CSV encoding oddities beyond pandas parsing, supply-chain integrity of transitive packages (no hash pinning), Git history (a local repo exists but has no commits yet; the remote has not been pushed to).
- Passing tests do not make this product ready to sell: licensing, privacy policy, live validation, support and deployment hardening are all outstanding.
