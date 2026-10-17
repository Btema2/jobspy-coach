# PM Agent Handoff & Continuity Brief: JobSpy Coach Daemon

> **For the incoming AI Assistant / Senior Project Manager:**
> You are the Senior Project Manager of subagents for the `jobspy-coach` project.
> You DO NOT get your hands dirty or implement code inline. Everything is executed by specialized subagents and independently verified by reviewer subagents before merging. You follow strict Test-Driven Development (TDD with tests written first) and atomic Git hygiene.

---

## 1. Project Background & Candidate Profile
- **Repository**: `https://github.com/Btema2/jobspy-coach` (local workspace: `/home/btema2/smart-things/code/jobspy-coach`)
- **Candidate Profile**: Certified professional coach specializing in corporate coaching and leadership development.
- **Target Domains**: Executive Coaching, Leadership Coaching, Business Coaching, Career Coaching, Life Coaching, and Corporate L&D (Learning & Development) Specialist / Manager roles.
- **Geographic Scope**: Poland local/hybrid + European Union / Global Remote roles.
- **Status**: Production-ready codebase on branch `main` with 135 passing tests (0.94s execution time, 98% statement coverage).

---

## 2. Core Architecture & Tech Stack
1. **Scraper Pipeline (`src/scraper/`)**: Uses `python-jobspy` across LinkedIn, Indeed, Glassdoor, and Google Jobs. Uses query catalog in `src/scraper/query_catalog.py`.
2. **Deterministic Pre-Filtering (`src/filters/`)**: Zero-cost multilingual regex in `src/filters/pre_filter.py` eliminating sports, gym, fitness, vocal, animal, and gaming roles in EN/PL/UK, plus stub descriptions (<150 chars or <30 words) with reason `DESCRIPTION_TOO_SHORT`.
3. **LLM Evaluation (`src/evaluator/`)**: Evaluates vacancies via OpenRouter (`openrouter/free` meta-router or specific free model) with 20 RPM rate limiting, 3-tier JSON parsing + regex fallback, and Pydantic validation (`src/evaluator/schemas.py`). Returns 1-10 match score, boolean `is_wow`, and 1-2 sentence Ukrainian summary (`summary_uk`).
4. **Storage & Deduplication (`src/storage/`)**: SQLite in WAL mode (`PRAGMA journal_mode = WAL; PRAGMA busy_timeout = 5000;`) with cross-board SHA-256 deduplication over normalized `company:title:location` (stripping corporate suffixes like `Sp. z o.o.`, `LLC`, `GmbH`), scrape run telemetry (`scrape_runs`), and 7-day stale job expiration.
5. **Telegram Notifications (`src/notifier/`)**: Instant WOW alerts for `score >= 9.0` or `is_wow == True` with inline URL button (`"🚀 Apply / View Job"`). Daily morning digest at 09:00 Europe/Warsaw for `score >= 6.0` in HTML chunks ($\le 3,800$ chars).
6. **Daemon & CLI (`src/scheduler/daemon.py`, `src/cli.py`)**: APScheduler with cron at 09:00 Europe/Warsaw and recurring interval scraping. Unified Typer CLI (`run`, `scrape`, `evaluate`, `digest`, `test-telegram`, `stats`, `--dry-run`).

---

## 3. Immediate Backlog: Improvements A, B, and C
The user verified live scraping and received 2 live WOW Telegram alerts, but identified 3 key improvements to implement next:

### Improvement A: Log the Actual Routed OpenRouter Model
- **Problem**: When using `openrouter/free`, OpenRouter dynamically routes to different underlying zero-cost models. Currently, the user cannot see which model actually scored each job.
- **Solution**: In `src/evaluator/llm_evaluator.py`, extract `actual_model = data.get("model", self.model)` from the OpenRouter API JSON response (`resp.json()`).
- **Action**:
  1. Add `actual_model` extraction to `OpenRouterEvaluator.evaluate_job`.
  2. Log it at `INFO` level: `logger.info("OpenRouter evaluated '%s' via model '%s' -> Score: %.1f, is_wow: %s", job.title, actual_model, result.score, result.is_wow)`.
  3. Store `actual_model` in `JobEvaluationResult` (or database column `eval_model`).
  4. Pin with unit test asserting `data["model"]` is captured and logged.

### Improvement B: Clean Up Query Catalog (Remove Incompatible Boards for Poland)
- **Problem**: In `src/scraper/query_catalog.py`, Polish queries (`country_code="poland"`) include `glassdoor`, and European queries include `zip_recruiter`. JobSpy's Glassdoor backend throws `Glassdoor: Glassdoor is not available for POLAND`, and ZipRecruiter throws `307` redirect for European traffic.
- **Solution**:
  1. In `src/scraper/query_catalog.py`, remove `glassdoor` from Polish queries (`Q-PL-ENG`, `Q-PL-BIZ`, `Q-PL-LD`). Use `["linkedin", "indeed", "google"]`.
  2. In `Q-EU-REM`, remove `zip_recruiter`. Use `["linkedin", "indeed", "google"]`.
  3. Keep `glassdoor` only on US/Global queries where `country_code="usa"` or international Glassdoor is supported.
  4. Update unit tests in `tests/test_scraper.py` to match the cleaned catalog.

### Improvement C: LinkedIn & Job Board Rate Limit Mitigations
- **Problem**: LinkedIn returned HTTP 429 when queried back-to-back with 25 results across 5 queries without proxies.
- **Solution**:
  1. **Batch Size Tuning**: Reduce default `SCRAPER_RESULTS_PER_QUERY` in `src/config.py` from 25 to 10 (or 12) to reduce guest burstiness.
  2. **Configurable Enabled Boards (`SCRAPER_ENABLED_BOARDS`)**: Add `scraper_enabled_boards: Optional[List[str]] = None` in `src/config.py` (parsed from comma-separated env var `SCRAPER_ENABLED_BOARDS="linkedin,indeed"`). In `JobSpyScraper`, filter query `site_names` to only scrape enabled boards.
  3. **Rotating Proxy Support (`SCRAPER_PROXIES`)**: Add `scraper_proxies: Optional[List[str]] = None` in `src/config.py` (parsed from comma-separated env var `SCRAPER_PROXIES="http://proxy1,http://proxy2"`). Pass `proxies=self.proxies` to `scrape_jobs(...)`.
  4. **Dedicated LinkedIn Pacing**: In `JobSpyScraper.scrape_board`, if `site_name == "linkedin"`, enforce a minimum cooldown (e.g. 10–15 seconds) so LinkedIn guest endpoints are never hit in rapid succession.
  5. Add unit tests in `tests/test_scraper.py` and `tests/test_config.py` verifying board filtering, proxy passing, and pacing.

---

## 4. Operating Rules for the Senior PM Agent
1. **Do not write code inline**: Dispatch `code_worker` subagents to implement tasks and `verifier_subagent` to independently review diffs and test results.
2. **Strict TDD**: Write failing tests first $\to$ verify RED $\to$ minimal implementation $\to$ verify GREEN $\to$ atomic commit.
3. **Use Git Properly**: Work on a branch (e.g. `feature/scraper-and-eval-optimizations`), commit atomically, review, and merge into `main`.
4. **BypassSandbox**: When subagents run shell commands (pip install, pytest, git, etc.), use `BypassSandbox: true` so local disk writes succeed.
5. **Verify Full Suite**: All 135+ tests must pass with 0 failures before declaring work complete.
