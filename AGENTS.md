# AGENTS.md — JobSpy Coach Project Memory & Operational Handbook

> **Project Memory & Continuity Guide**: This file serves as the definitive memory bank for AI agents, developers, and autonomous systems maintaining and extending `jobspy-coach`. It encapsulates architectural decisions, hard-earned operational gotchas, candidate persona rules, and strict agent development protocols.

---

## 1. Candidate Privacy Protocol & Persona Architecture

> [!CAUTION]
> **STRICT PRIVACY MANDATE FOR ALL AI AGENTS & DEVELOPERS**:
> Under **NO circumstances** should private user data (candidate name, specific personal background, city/country of residence, native languages, personal credentials, or customized search targets) be committed, documented, or pushed to any public or remote repository.
> 
> - **Local Storage Only**: All actual candidate profiles and user-specific personas are stored exclusively in local, git-ignored configuration files (such as `.candidate_profile.md` and `.env`).
> - **Public Neutrality**: Tracked repository files, documentation, commit messages, and PRs must strictly use sanitized, generic templates and configurable placeholders (e.g., `<TARGET_ROLE>`, `<TARGET_REGION>`).
> - **Runtime Persona Injection**: When running locally, agents and the evaluator engine read `.candidate_profile.md` to evaluate vacancies against the user's private persona. If missing, the engine falls back to a generic professional template.


---

## 2. Core Architecture & Subsystems

```
src/
├── config.py              # Pydantic Settings with comma-separated list validators
├── cli.py                 # Unified Typer CLI (scrape, evaluate, digest, stats, run)
├── filters/
│   └── pre_filter.py      # Zero-cost deterministic filters: sports/pets noise, title traps, seniority (classify_title), stub descriptions
├── scraper/
│   ├── query_catalog.py   # Curated search queries mapped by country & board
│   └── jobspy_scraper.py  # python-jobspy wrapper with pacing, jitter, proxies
├── storage/
│   ├── models.py          # Dataclasses (JobPosting, JobEvaluationResult, ScrapeRun)
│   └── db.py              # SQLite WAL repository with SHA-256 deduplication
├── evaluator/
│   ├── chain.py           # ProviderChainEvaluator: NIM -> OpenRouter fallback, validation, score post-processing
│   ├── providers.py       # NvidiaNimProvider / OpenRouterProvider (httpx, retries, error classification, daily budget)
│   ├── rate_limiter.py    # Thread-safe sliding-window limiter (stop-event aware)
│   ├── circuit_breaker.py # Per-provider breaker with a single half-open probe
│   ├── errors.py          # Typed errors (transient, permanent, quota, invalid response, providers unavailable, ...)
│   ├── parsing.py         # 3-tier JSON extraction (direct -> code fence -> outermost {...})
│   ├── prompts.py         # Generic scoring rubric; persona only from the local profile
│   └── schemas.py         # Pydantic schemas for structured LLM responses
├── notifier/
│   ├── formatter.py       # Telegram HTML formatting, length guards & chunking
│   └── telegram_notifier.py # Telegram Bot API client (WOW alerts & daily digest)
└── scheduler/
    └── daemon.py          # APScheduler 24/7 background service orchestration
scripts/
└── bakeoff.py             # Model bake-off for NIM / OpenRouter free models (writes to git-ignored local/)
docs/
└── model-selection.md     # Recorded bake-off results and chosen defaults
```

### Key Workflows
1. **Scraping**: `JobSpyScraper` queries configured job boards (LinkedIn, Indeed, Glassdoor, Google) with randomized jitter (`4.0–9.0s`).
2. **Deterministic Pre-Filtering**: Zero-cost multilingual regex eliminates athletic, sports, fitness, vocal, animal, and gaming roles, and drops stub descriptions (<150 chars or <30 words) before invoking LLMs.
3. **Cross-Board Deduplication**: Normalizes corporate legal suffixes (`LLC`, `GmbH`, `Inc.`), casing, and punctuation to compute SHA-256 `dedup_hash` over `company:title:location`.
4. **LLM Evaluation**: `DaemonService.evaluate_unprocessed` evaluates non-prefiltered vacancies concurrently (worker threads, `EVAL_MAX_WORKERS`) through `ProviderChainEvaluator`: NVIDIA NIM first, then OpenRouter pinned free models. The chain extracts a 1–10 score, executive summary (`summary_uk`), English rationale, and the model that actually answered (`eval_model`), and applies hard-reject / bonus adjustments in code. The chain is built lazily on first use (a fresh clone without keys must still construct `DaemonService`); with no provider configured it raises `ConfigurationError`.
5. **Two-Tier Telegram Alerts**:
   - **Instant WOW Alerts**: Vacancies with a final score $\ge 8.5$ (`WOW_SCORE_THRESHOLD`) immediately trigger Telegram push notifications with an inline application button. `is_wow` is derived by the daemon from the final score; the LLM does not decide it.
   - **Daily Morning Digest**: Every morning at the scheduled digest time, qualifying vacancies ($6.5 \le \text{score} < 8.5$) are formatted in clean HTML chunks ($\le 3,800$ chars) and dispatched.
   - **Low-Score Dismissal**: Vacancies scoring $< 6.5$ (`DIGEST_SCORE_THRESHOLD`) are immediately marked `dismissed` to keep queues lean.
   - **Stale Expiration**: Pending items older than 7 days transition to `expired_dismissed`.

---

## 3. Critical Learnings & Operational Gotchas

### A. LinkedIn Scraping (`python-jobspy`)
1. **`fetch_description=True` is Mandatory**:
   - In `python-jobspy`, Indeed returns descriptions inline in search cards, but **LinkedIn requires a secondary detail request** (`/jobs/view/{job_id}`).
   - JobSpy defaults `fetch_description=False`. If omitted, JobSpy returns `description=None` for all LinkedIn jobs, causing our pre-filter to drop 100% of LinkedIn jobs as `DESCRIPTION_TOO_SHORT`.
   - **Always pass `fetch_description=True`** in `JobSpyScraper.scrape_board`.
2. **Rate Limiting & Guest Endpoints (HTTP 429)**:
   - LinkedIn guest endpoints aggressively throttle when blasted with large batch sizes.
   - **Batch size floor**: Keep `SCRAPER_RESULTS_PER_QUERY=10` (never $\ge 25$).
   - **Mandatory Pacing**: A minimum 12.0s cooldown (`linkedin_cooldown_seconds`) must elapse between consecutive LinkedIn requests.
3. **Datacenter vs. Residential IPs**:
   - On residential IPs (local dev machine), LinkedIn scraping works smoothly.
   - On cloud VPS/datacenter IPs (AWS, DigitalOcean, Hetzner), LinkedIn blocks datacenter ASNs. For server deployment, configure residential or rotating proxies via `SCRAPER_PROXIES="http://user:pass@host:port"`.

### B. Job Board Quirks by Country
1. **Glassdoor**:
   - In `python-jobspy`, Glassdoor throws `Glassdoor: Glassdoor is not available for POLAND`.
   - **Rule**: Never query Glassdoor with a country it does not support (e.g. `country_code="poland"`). Only include Glassdoor in US/Global remote queries (`Q-GL-REM`).
2. **ZipRecruiter**:
   - ZipRecruiter blocks European IP traffic with HTTP 307 redirects.
   - **Rule**: Only include ZipRecruiter on US/UK/Canada queries; never on European queries (`Q-EU-REM`).

### C. LLM Evaluation (NVIDIA NIM + OpenRouter)
1. **NVIDIA NIM facts (primary provider)**:
   - Model `nvidia/nemotron-3-ultra-550b-a55b` is a reasoning model controlled with `reasoning_effort` (`low` chosen; a weak lever for this model).
   - Do **not** send `response_format` (unsupported) or `thinking_token_budget` (the call then runs to `max_tokens` with empty content). Keep temperature at 0.6 (never 0). Reasoning arrives in `reasoning_content`, ignored; `content` is clean JSON.
   - HTTP 429 has **no `Retry-After` and no `x-ratelimit-*` headers**, so backoff is computed client-side (5 s x 2^n with jitter), never header-driven. Treat 429 as transient.
   - Treat the 30 rpm cap (`NVIDIA_RPM_LIMIT`) as an assumed conservative client-side limit (provider limit reported ~40 rpm), enforced **per process**. Never run `evaluate` while the daemon is evaluating.
2. **OpenRouter fallback**:
   - Always use **pinned** models via `OPENROUTER_MODELS` (the `models` array). **Never use `openrouter/free`**: it is a random router whose pool includes content-safety, code and audio models (this produced garbage scores and `User Safety: safe` payloads).
   - Free models are frequently rate-limited upstream; treat the fallback as best-effort. The local daily budget (`OPENROUTER_DAILY_BUDGET`, default 45 of ~50/day) counts every HTTP attempt, rolls over at 00:00 UTC and resets on restart.
   - Only the first default model was verified in the bake-off (see `docs/model-selection.md`).
3. **Model logging (`eval_model`)**: store the model that actually answered (`data.get("model", self.model)`) on `JobEvaluationResult.eval_model` and log it at `INFO`.
4. **Concurrency model (threads)**:
   - Worker threads run **only** `chain.evaluate_job` (HTTP + parsing). The thread that called `evaluate_unprocessed` is the **only** one that touches SQLite or Telegram (caller-thread invariant); do not move DB/Telegram work into workers.
   - Shutdown never calls `executor.shutdown(wait=True)`; signal handlers only set stop events and cancel queued futures (`request_shutdown`), because `with ThreadPoolExecutor` or executor shutdown inside a handler can deadlock.
   - Provider outage => jobs stay `unprocessed` with attempts unchanged; only job-specific failures increment `eval_attempts` (marked `failed` at `EVAL_MAX_ATTEMPTS`).
5. **Thresholds**: WOW $\ge 8.5$, digest $6.5$–$8.4$, below $6.5$ dismissed. Defaults live in `src/config.py`; keep README/`.env.example` in sync.
6. **Response Parsing Robustness**: free models may emit markdown fences, `<think>` blocks, trailing text or safety banners. Parsing strips `<think>` and uses 3-tier extraction: direct JSON parse $\to$ code fence extraction $\to$ outermost `{...}` regex fallback.
7. **Tests**: tests must never hit the network or depend on the real `.env` (use `Settings(_env_file=None)` / monkeypatch and `httpx.MockTransport`). The only exception is `tests/test_live_smoke.py` (`@pytest.mark.live`), skipped by default and run with `pytest -m live`.

### D. Telegram API Safeguards
1. **Message Length**: Telegram hard-limits messages to 4,096 characters. All digests must chunk at $\le 3,800$ characters.
2. **HTML Entity Escaping**: Job titles, company names, and URLs must pass through `html.escape()` (`&`, `<`, `>`) before insertion into HTML message strings to prevent 400 Bad Request parsing errors.
3. **Summary & Reasoning Guards**: Summaries are clamped to 600 chars and reasoning to 800 chars before formatting.

### E. SQLite Concurrency & Database Operations
1. **WAL Mode**: SQLite must always run in Write-Ahead Logging mode (`PRAGMA journal_mode = WAL;`) with `PRAGMA busy_timeout = 5000;` and `PRAGMA synchronous = NORMAL;`.
2. **Stale Records Cleanup**: If bad scrape runs populate the database with stub records (e.g. empty descriptions), use `Database.delete_empty_description_jobs("linkedin")` to remove them so deduplication hashes don't permanently suppress valid future scrapes.

---

## 4. Agent Operating Guidelines & Rules of Engagement

Any AI agent working on this codebase must adhere strictly to these principles:

1. **Role: Senior Project Manager of Subagents**:
   - Coordinate, plan, and verify. Do NOT write application code inline in the PM session.
   - Dispatch `code_worker` subagents for implementation and `verifier_subagent` for independent code review and diff inspection.
2. **Strict Test-Driven Development (TDD)**:
   - Step 1: Write a focused failing test capturing the exact requirement.
   - Step 2: Run test to verify RED state.
   - Step 3: Implement minimal code to satisfy the test.
   - Step 4: Run test to verify GREEN state, then run full suite.
   - Step 5: Atomic Git commit following Conventional Commits (`feat:`, `fix:`, `docs:`, `test:`).
3. **Git Hygiene & Development Branches**:
   - Never implement directly on `main`. Always create a descriptive feature branch (`feature/<name>`).
   - Verify all tests pass, conduct task-level reviews, perform a whole-branch review, merge into `main`, and push.
4. **Security, Privacy & Secrets**:
   - **NEVER hardcode or commit secrets** (OpenRouter API keys, Telegram bot tokens, chat IDs, proxy credentials).
   - **NEVER commit private user/candidate data** (name, background, residence, native languages, target locations, or custom prompts).
   - All secrets belong in `.env` (which is in `.gitignore`). All private candidate personas belong in `.candidate_profile.md` (which is in `.gitignore`).
   - Use `.env.example` with sanitized placeholders for documentation.
5. **BypassSandbox Execution**:
   - Shell commands that modify files, run tests, or execute git commands (`pytest`, `git`, `python`) require `BypassSandbox: true` to avoid sandbox read-only filesystem errors.

---

## 5. Quick CLI & Testing Reference

```bash
# Run full automated test suite (~600 tests, no network; the `live` smoke test is skipped)
.venv/bin/pytest -v

# Optional: 2 real calls through the provider chain (needs keys in .env)
.venv/bin/pytest -m live -v

# Model bake-off (writes to git-ignored local/bakeoff/)
.venv/bin/python scripts/bakeoff.py list-free

# Test Telegram connectivity
.venv/bin/jobspy-coach test-telegram

# View database metrics & lifecycle counts
.venv/bin/jobspy-coach stats

# Run dry-run scraping cycle (no DB writes, no Telegram alerts)
.venv/bin/jobspy-coach scrape --dry-run

# Run dry-run LLM evaluation on queued unprocessed jobs (never while the daemon is evaluating)
.venv/bin/jobspy-coach evaluate --limit 5 --dry-run

# Preview daily digest formatting
.venv/bin/jobspy-coach digest --dry-run

# Live single-cycle scrape & alert
.venv/bin/jobspy-coach scrape

# Start 24/7 background daemon scheduler
.venv/bin/jobspy-coach run
```
