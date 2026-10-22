# PM Agent Handoff & Continuity Brief: JobSpy Coach Daemon

> **For the incoming AI Assistant / Senior Project Manager:**
> You are the Senior Project Manager of subagents for the `jobspy-coach` project.
> You DO NOT get your hands dirty or implement code inline. Everything is executed by specialized subagents (`code_worker`) and independently verified by reviewer subagents (`verifier_subagent`) before merging. You follow strict Test-Driven Development (TDD with tests written first) and atomic Git hygiene.

---

## 1. Project Background & Privacy Protocol
- **Repository**: `https://github.com/Btema2/jobspy-coach` (local workspace: `/home/btema2/smart-things/code/jobspy-coach`)
- **Privacy Protocol**: Private candidate persona definitions and personal credentials must NEVER be committed or pushed to any public repository. The local user persona is stored exclusively in `.candidate_profile.md` (which is in `.gitignore`).
- **Target Domains**: Executive Coaching, Leadership Development, Agile/Scrum Coaching, Career Coaching, and Corporate L&D (Learning & Development) roles.
- **Geographic Scope**: Configurable local and global remote roles.
- **Current Status**: Production-ready codebase on branch `main` with 143 passing tests (3.2s execution time, 97% statement coverage).
- **Recent Completed Work**:
  - Improvement A (`9153af6`): Captured actual OpenRouter model in `JobEvaluationResult.eval_model` and logged at `INFO`.
  - Improvement B (`7c3967b`): Cleaned query catalog (`src/scraper/query_catalog.py`) to eliminate Glassdoor from Poland queries and ZipRecruiter from EU queries.
  - Improvement C (`4bab753`): Reduced batch size to 10 (`SCRAPER_RESULTS_PER_QUERY=10`), added `SCRAPER_ENABLED_BOARDS` and `SCRAPER_PROXIES` with comma-delimited parsing, forwarded proxies to `scrape_jobs`, and enforced 12s LinkedIn pacing.
  - Merged via PR #1 into `main` (`c907f92` -> `99f256c`).

---

## 2. Core Architecture & Tech Stack
1. **Scraper Pipeline (`src/scraper/`)**: Uses `python-jobspy` across LinkedIn, Indeed, Glassdoor, and Google Jobs. Uses query catalog in `src/scraper/query_catalog.py`.
2. **Deterministic Pre-Filtering (`src/filters/`)**: Zero-cost multilingual regex in `src/filters/pre_filter.py` eliminating sports, gym, fitness, vocal, animal, and gaming roles in EN/PL/UK, plus stub descriptions (<150 chars or <30 words) with reason `DESCRIPTION_TOO_SHORT`.
3. **LLM Evaluation (`src/evaluator/`)**: Evaluates vacancies via OpenRouter (`openrouter/free` meta-router or specific model) with 20 RPM rate limiting, 3-tier JSON parsing + regex fallback, and Pydantic validation (`src/evaluator/schemas.py`). Returns 1-10 match score, boolean `is_wow`, and 1-2 sentence executive summary (`summary_uk`).
4. **Storage & Deduplication (`src/storage/`)**: SQLite in WAL mode (`PRAGMA journal_mode = WAL; PRAGMA busy_timeout = 5000;`) with cross-board SHA-256 deduplication over normalized `company:title:location` (stripping corporate suffixes like `LLC`, `GmbH`), scrape run telemetry (`scrape_runs`), and 7-day stale job expiration.
5. **Telegram Notifications (`src/notifier/`)**: Instant WOW alerts for `score >= 9.0` or `is_wow == True` with inline URL button (`"🚀 Apply / View Job"`). Daily morning digest at configured digest time for `score >= 6.0` in HTML chunks ($\le 3,800$ chars).
6. **Daemon & CLI (`src/scheduler/daemon.py`, `src/cli.py`)**: APScheduler with cron for daily morning digest and recurring interval scraping. Unified Typer CLI (`run`, `scrape`, `evaluate`, `digest`, `test-telegram`, `stats`, `--dry-run`).

---

## 3. Immediate Active Issue: Fix LinkedIn Scraper Description Fetching

### Root Cause Diagnosis (Verified via Database & JobSpy Source Code):
- **User Symptom**: Running `jobspy-coach scrape` and `digest` returns only Indeed jobs in the digest, with 0 LinkedIn jobs appearing.
- **Database Findings**: 107 LinkedIn jobs were discovered and saved to `data/jobs.db`, but **100% of them (107/107) were dismissed by pre-filter with `pre_filter_reason = 'DESCRIPTION_TOO_SHORT: length=0 (min 150)'`**.
- **JobSpy Underlying Behavior**:
  - In `python-jobspy`, Indeed includes job descriptions directly in the primary search card response.
  - LinkedIn requires a secondary fetch (`_fetch_details(job_id)`), which JobSpy **only performs if `fetch_description=True`** is passed to `scrape_jobs(...)`.
  - In `src/scraper/jobspy_scraper.py`, `scrape_jobs(...)` was called without `fetch_description=True` (defaulting to `False`).
  - Therefore, JobSpy returned `description = None` for all LinkedIn jobs.
  - `src/filters/pre_filter.py` saw empty descriptions and correctly dismissed them as `DESCRIPTION_TOO_SHORT`.
  - Because `save_job` deduplicates on `dedup_hash`, subsequent scrape runs skip these jobs as already existing in the database.

### Required Remediation:
1. **Pass `fetch_description=True` in Scraper**:
   - In `src/scraper/jobspy_scraper.py`:
     - Add `fetch_description: bool = True` to `JobSpyScraper.__init__`.
     - In `JobSpyScraper.scrape_board`, pass `fetch_description=self.fetch_description` to `scrape_jobs(...)`.
2. **Add Unit Tests in `tests/test_scraper.py`**:
   - Assert `scrape_jobs` is called with `fetch_description=True`.
3. **Database Cleanup / Reset**:
   - Delete the 107 bogus dismissed LinkedIn jobs with empty descriptions so they can be re-fetched with complete descriptions:
     ```sql
     DELETE FROM jobs WHERE site = 'linkedin' AND (description IS NULL OR length(description) = 0);
     ```
4. **Live Verification**:
   - Run `.venv/bin/jobspy-coach scrape --dry-run` or `.venv/bin/jobspy-coach scrape` to verify LinkedIn jobs are parsed with multi-thousand-character descriptions and passed to the LLM evaluator.

---

## 4. Operating Rules for the Senior PM Agent
1. **Do not write code inline**: Dispatch `code_worker` subagents to implement tasks and `verifier_subagent` to independently review diffs and test results.
2. **Strict TDD**: Write failing tests first $\to$ verify RED $\to$ minimal implementation $\to$ verify GREEN $\to$ atomic commit.
3. **Use Git Properly**: Work on a branch (e.g. `feature/linkedin-fetch-description`), commit atomically, review, and merge into `main` via PR / git merge.
4. **BypassSandbox**: When subagents run shell commands (pip install, pytest, git, etc.), use `BypassSandbox: true` so local disk writes succeed.
5. **Verify Full Suite**: All 141+ tests must pass with 0 failures before declaring work complete.
