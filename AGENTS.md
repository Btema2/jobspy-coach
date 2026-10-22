# AGENTS.md — JobSpy Coach Project Memory & Operational Handbook

> **Project Memory & Continuity Guide**: This file serves as the definitive memory bank for AI agents, developers, and autonomous systems maintaining and extending `jobspy-coach`. It encapsulates architectural decisions, hard-earned operational gotchas, candidate persona rules, and strict agent development protocols.

---

## 1. Project Overview & Candidate Persona

### The Candidate
- **Profile**: Certified professional coach specializing in corporate coaching and leadership development.
- **Languages (Configurable)**:
  - Configurable via local profile (multilingual evaluation support).
  
  
- **Target Domains**:
  - Executive Coaching & Leadership Development
  - Agile Coaching, Scrum Coaching & Enterprise Transformation
  - Career Coaching & Mentorship
  - Corporate Training & Learning & Development (L&D Specialist / Manager / Lead)
  - Team Coaching & Business Coaching
- **Target Markets**:
  - Configurable target markets (local, hybrid, and remote).
  - European Union & Global Remote: International companies hiring EU-based talent.

---

## 2. Core Architecture & Subsystems

```
src/
├── config.py              # Pydantic Settings with comma-separated list validators
├── cli.py                 # Unified Typer CLI (scrape, evaluate, digest, stats, run)
├── filters/
│   └── pre_filter.py      # Zero-cost deterministic regex & length filters
├── scraper/
│   ├── query_catalog.py   # Curated search queries mapped by country & board
│   └── jobspy_scraper.py  # python-jobspy wrapper with pacing, jitter, proxies
├── storage/
│   ├── models.py          # Dataclasses (JobPosting, JobEvaluationResult, ScrapeRun)
│   └── db.py              # SQLite WAL repository with SHA-256 deduplication
├── evaluator/
│   ├── prompts.py         # System and user prompts for candidate evaluation
│   ├── schemas.py         # Pydantic schemas for structured LLM responses
│   └── llm_evaluator.py   # OpenRouter client with rate limiting & model extraction
├── notifier/
│   ├── formatter.py       # Telegram HTML formatting, length guards & chunking
│   └── telegram_notifier.py # Telegram Bot API client (WOW alerts & daily digest)
└── scheduler/
    └── daemon.py          # APScheduler 24/7 background service orchestration
```

### Key Workflows
1. **Scraping**: `JobSpyScraper` queries configured job boards (LinkedIn, Indeed, Glassdoor, Google) with randomized jitter (`4.0–9.0s`).
2. **Deterministic Pre-Filtering**: Zero-cost multilingual regex eliminates athletic, sports, fitness, vocal, animal, and gaming roles, and drops stub descriptions (<150 chars or <30 words) before invoking LLMs.
3. **Cross-Board Deduplication**: Normalizes corporate legal suffixes (`Sp. z o.o.`, `LLC`, `GmbH`, `Inc.`), casing, and punctuation to compute SHA-256 `dedup_hash` over `company:title:location`.
4. **LLM Evaluation**: Evaluates non-prefiltered vacancies via OpenRouter, extracting 1–10 match score, boolean `is_wow`, Ukrainian summary (`summary_uk`), English rationale, and dynamically routed model name (`eval_model`).
5. **Two-Tier Telegram Alerts**:
   - **Instant WOW Alerts**: Vacancies scoring $\ge 9.0$ or flagged `is_wow = True` immediately trigger Telegram push notifications with an inline application button.
   - **Daily Morning Digest**: Every morning at 09:00 Europe/Warsaw, qualifying vacancies ($6.0 \le \text{score} < 9.0$) are formatted in clean HTML chunks ($\le 3,800$ chars) and dispatched.
   - **Low-Score Dismissal**: Vacancies scoring $< 6.0$ are immediately marked `dismissed` to keep queues lean.
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
   - **Rule**: Never query Glassdoor with `country_code="poland"`. Only include Glassdoor in US/Global remote queries (`Q-GL-REM`).
2. **ZipRecruiter**:
   - ZipRecruiter blocks European IP traffic with HTTP 307 redirects.
   - **Rule**: Only include ZipRecruiter on US/UK/Canada queries; never on European queries (`Q-EU-REM`).

### C. OpenRouter LLM Evaluation
1. **Model Logging (`eval_model`)**:
   - When using meta-routers like `openrouter/free`, OpenRouter dynamically routes requests to underlying zero-cost models.
   - Always extract `actual_model = data.get("model", self.model) or self.model` from the response payload, store it on `JobEvaluationResult.eval_model`, and log it at `INFO` level.
2. **Daily Free Tier Quotas (50 RPD)**:
   - Free OpenRouter accounts have a daily quota (~50 Requests Per Day). If a large batch triggers HTTP 429, the daemon catches it, leaves jobs in `unprocessed` status, and gracefully skips to the next cycle without crashing.
3. **Response Parsing Robustness**:
   - Free models frequently emit markdown fences (` ```json `), trailing text, or safety banners (`User Safety: safe.`).
   - The evaluator uses 3-tier parsing: direct JSON parse $\to$ code fence extraction $\to$ outermost `{...}` regex fallback.

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
4. **Security & Secrets**:
   - **NEVER hardcode or commit secrets** (OpenRouter API keys, Telegram bot tokens, chat IDs, proxy credentials).
   - All secrets belong in `.env` (which is in `.gitignore`). Use `.env.example` with sanitized placeholders for documentation.
5. **BypassSandbox Execution**:
   - Shell commands that modify files, run tests, or execute git commands (`pytest`, `git`, `python`) require `BypassSandbox: true` to avoid sandbox read-only filesystem errors.

---

## 5. Quick CLI & Testing Reference

```bash
# Run full automated test suite (143+ tests, 97%+ coverage)
.venv/bin/pytest -v

# Test Telegram connectivity
.venv/bin/jobspy-coach test-telegram

# View database metrics & lifecycle counts
.venv/bin/jobspy-coach stats

# Run dry-run scraping cycle (no DB writes, no Telegram alerts)
.venv/bin/jobspy-coach scrape --dry-run

# Run dry-run LLM evaluation on queued unprocessed jobs
.venv/bin/jobspy-coach evaluate --limit 5 --dry-run

# Preview daily digest formatting
.venv/bin/jobspy-coach digest --dry-run

# Live single-cycle scrape & alert
.venv/bin/jobspy-coach scrape

# Start 24/7 background daemon scheduler
.venv/bin/jobspy-coach run
```
