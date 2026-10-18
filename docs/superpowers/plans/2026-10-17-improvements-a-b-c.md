# Improvements A, B, and C Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement production optimizations for JobSpy Coach: actual OpenRouter model routing logging (Improvement A), query catalog cleanup for Poland and EU queries (Improvement B), and rate limit mitigations for LinkedIn and job boards (Improvement C).

**Architecture:** 
1. In the LLM evaluation subsystem, extract `actual_model` from OpenRouter API responses (`resp.json().get("model")`), log it at INFO level, and store it on `JobEvaluationResult`.
2. In the scraper query catalog, remove `glassdoor` from Poland queries (where JobSpy does not support Poland) and `zip_recruiter` from European queries (which redirects with 307), reserving them only for valid regions.
3. In configuration and scraping pipelines, lower default batch size from 25 to 10, add support for `SCRAPER_ENABLED_BOARDS` and `SCRAPER_PROXIES` with comma-separated string parsing, pass proxies to JobSpy, and enforce a dedicated 12s cooldown on LinkedIn calls.

**Tech Stack:** Python 3.10+, Pydantic v2, python-jobspy, httpx, Typer, pytest

**Spec:** `docs/PM_AGENT_HANDOFF.md`

## Global Constraints

- Python 3.10+ compatible (running in Python 3.14.7 virtual environment `.venv`)
- Strict backward compatibility: default configurations must work out-of-the-box without requiring user intervention
- No regression: all 135 existing unit and integration tests must continue to pass
- Type hints, Pydantic field validators, and robust error handling throughout
- Proper Git hygiene: atomic commits per task

## Review Focus

1. OpenRouter responses lacking a `"model"` key: must fall back gracefully to configured `self.model` without throwing `KeyError`.
2. Whitespace in comma-separated strings for `SCRAPER_ENABLED_BOARDS` (e.g. `" linkedin , indeed "`) and `SCRAPER_PROXIES`: must be stripped cleanly into valid lists.
3. Empty string or None for `SCRAPER_ENABLED_BOARDS` or `SCRAPER_PROXIES`: must evaluate to None (disabled) rather than `[""]`.
4. Board case-sensitivity: `SCRAPER_ENABLED_BOARDS` check must be case-insensitive (e.g. `"LinkedIn"` matches `"linkedin"`).
5. LinkedIn cooldown must not delay other boards if LinkedIn is not being queried.

---

### Task 1: Improvement A - Actual OpenRouter Model Extraction & Logging

**Files:**
- Modify: `src/evaluator/schemas.py`
- Modify: `src/storage/models.py`
- Modify: `src/evaluator/llm_evaluator.py`
- Test: `tests/test_evaluator.py`

**Interfaces:**
- Consumes: OpenRouter API response JSON structure (`resp.json()["model"]`)
- Produces: `JobEvaluationResult.eval_model: Optional[str] = None`, logged via `logger.info("OpenRouter evaluated '%s' via model '%s' -> Score: %.1f, is_wow: %s", job.title, actual_model, result.score, result.is_wow)`

- [ ] **Step 1: Write the failing test**

In `tests/test_evaluator.py`, add `test_openrouter_evaluator_captures_actual_routed_model` verifying that:
1. When OpenRouter returns `{"model": "deepseek/deepseek-chat", "choices": [{"message": {"content": "..."}}]}`, `eval_res.eval_model == "deepseek/deepseek-chat"`.
2. If OpenRouter response omits `"model"`, `eval_res.eval_model` falls back to `evaluator.model`.
3. The INFO log message contains the actual routed model name.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_evaluator.py -k test_openrouter_evaluator_captures_actual_routed_model -v`
Expected: FAIL with AttributeError or AssertionError (`eval_model` not present).

- [ ] **Step 3: Implement model extraction and logging**

1. In `src/evaluator/schemas.py`, add `eval_model: Optional[str] = Field(default=None, description="Actual model routed by OpenRouter")` to `JobEvaluationResult`.
2. In `src/storage/models.py`, add `eval_model: Optional[str] = None` to `JobEvaluationResult` dataclass.
3. In `src/evaluator/llm_evaluator.py` (`OpenRouterEvaluator.evaluate_job`):
   - Extract `actual_model = data.get("model", self.model)`
   - Set `result.eval_model = actual_model`
   - Add INFO log:
     `logger.info("OpenRouter evaluated '%s' via model '%s' -> Score: %.1f, is_wow: %s", job.title, actual_model, result.score, result.is_wow)`

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_evaluator.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/evaluator/schemas.py src/storage/models.py src/evaluator/llm_evaluator.py tests/test_evaluator.py
git commit -m "feat(evaluator): log and capture actual routed OpenRouter model"
```

---

### Task 2: Improvement B - Query Catalog Cleanup

**Files:**
- Modify: `src/scraper/query_catalog.py`
- Modify: `tests/test_scraper.py`

**Interfaces:**
- Consumes: `SearchQuery` dataclass
- Produces: Cleaned `DEFAULT_QUERY_CATALOG` with zero unsupported board-country combinations

- [ ] **Step 1: Write the failing test**

In `tests/test_scraper.py`, add `test_query_catalog_incompatible_boards_excluded` verifying:
1. For every query in `DEFAULT_QUERY_CATALOG` where `country_code == "poland"`, `"glassdoor"` is NOT in `site_names`.
2. For query `Q-EU-REM`, `"zip_recruiter"` is NOT in `site_names`.
3. Polish queries have at least `["linkedin", "indeed", "google"]`.
4. `Q-GL-REM` retains `["linkedin", "indeed", "glassdoor", "google"]`.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_scraper.py -k test_query_catalog_incompatible_boards_excluded -v`
Expected: FAIL (current catalog has glassdoor in Polish queries and zip_recruiter in Q-EU-REM).

- [ ] **Step 3: Clean query catalog in `src/scraper/query_catalog.py`**

1. `Q-PL-ENG`: change `site_names` to `["linkedin", "indeed", "google"]`.
2. `Q-PL-BIZ`: change `site_names` to `["linkedin", "indeed", "google"]`.
3. `Q-PL-LD`: change `site_names` to `["linkedin", "indeed", "google"]`.
4. `Q-EU-REM`: change `site_names` to `["linkedin", "indeed", "google"]`.
5. `Q-GL-REM`: change `site_names` to `["linkedin", "indeed", "glassdoor", "google"]`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_scraper.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/scraper/query_catalog.py tests/test_scraper.py
git commit -m "fix(scraper): remove incompatible job boards from Poland and EU search queries"
```

---

### Task 3: Improvement C - LinkedIn & Job Board Rate Limit Mitigations

**Files:**
- Modify: `src/config.py`
- Modify: `.env.example`
- Modify: `src/scraper/jobspy_scraper.py`
- Modify: `src/scheduler/daemon.py`
- Modify: `tests/test_config.py`
- Modify: `tests/test_scraper.py`

**Interfaces:**
- Consumes: `Settings.scraper_enabled_boards`, `Settings.scraper_proxies`, `Settings.scraper_results_per_query`
- Produces:
  - `JobSpyScraper(enabled_boards=..., proxies=..., results_per_query=10, linkedin_cooldown_seconds=12.0)`
  - Filtering out non-enabled boards in `scrape_board`
  - Passing `proxies=self.proxies` to `scrape_jobs(...)`
  - Enforcing `linkedin_cooldown_seconds` pacing on `site_name == "linkedin"`

- [ ] **Step 1: Write the failing tests**

1. In `tests/test_config.py`, add `test_scraper_enabled_boards_and_proxies_parsing` testing:
   - Default `scraper_results_per_query == 10`.
   - String `"linkedin, indeed, google"` parses into `["linkedin", "indeed", "google"]`.
   - String `" http://p1:8080 , http://p2:8080 "` parses into `["http://p1:8080", "http://p2:8080"]`.
   - Empty or None values parse into `None`.
2. In `tests/test_scraper.py`, add:
   - `test_jobspy_scraper_enabled_boards_filter`: verifies that when `enabled_boards=["indeed"]`, queries requesting `linkedin` return `[]` without calling `scrape_jobs`.
   - `test_jobspy_scraper_passes_proxies`: verifies that `proxies` passed to `JobSpyScraper` are forwarded to `scrape_jobs`.
   - `test_jobspy_scraper_linkedin_pacing`: verifies that back-to-back calls to `scrape_board("linkedin", ...)` enforce the cooldown sleep.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_config.py tests/test_scraper.py -k "test_scraper_enabled_boards or test_jobspy_scraper_enabled_boards or test_jobspy_scraper_passes_proxies or test_jobspy_scraper_linkedin_pacing" -v`
Expected: FAIL

- [ ] **Step 3: Implement Configuration & Scraper Enhancements**

1. In `src/config.py`:
   - Change `scraper_results_per_query` default from 25 to 10.
   - Add `scraper_enabled_boards: Optional[List[str]] = Field(default=None, alias="SCRAPER_ENABLED_BOARDS")`.
   - Add `scraper_proxies: Optional[List[str]] = Field(default=None, alias="SCRAPER_PROXIES")`.
   - Add `@field_validator("scraper_enabled_boards", "scraper_proxies", mode="before")` to handle comma-delimited string parsing, stripping whitespace, and returning `None` for empty inputs.
2. In `src/scraper/jobspy_scraper.py`:
   - Update `JobSpyScraper.__init__` with parameters:
     - `results_per_query: int = 10`
     - `enabled_boards: Optional[List[str]] = None`
     - `proxies: Optional[List[str]] = None`
     - `linkedin_cooldown_seconds: float = 12.0`
     - Store `self.last_linkedin_call_time = 0.0`.
   - In `scrape_board(self, site_name: str, query: SearchQuery)`:
     - If `self.enabled_boards` is not None and `site_name.lower() not in [b.lower() for b in self.enabled_boards]`:
       `logger.debug("Skipping board '%s' as it is not in enabled_boards", site_name)`
       `return []`
     - If `site_name.lower() == "linkedin"`:
       Check `elapsed = time.time() - self.last_linkedin_call_time`. If `elapsed < self.linkedin_cooldown_seconds`, sleep `self.linkedin_cooldown_seconds - elapsed`. Update `self.last_linkedin_call_time = time.time()`.
     - Pass `proxies=self.proxies` to `scrape_jobs(...)`.
3. In `src/scheduler/daemon.py`:
   - In `DaemonService.__init__`, pass `enabled_boards=settings.scraper_enabled_boards` and `proxies=settings.scraper_proxies` to `JobSpyScraper`.
4. In `.env.example`:
   - Add documentation for `SCRAPER_RESULTS_PER_QUERY=10`, `SCRAPER_ENABLED_BOARDS=linkedin,indeed,google`, and `SCRAPER_PROXIES=http://user:pass@host:port`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_config.py tests/test_scraper.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/config.py src/scraper/jobspy_scraper.py src/scheduler/daemon.py .env.example tests/test_config.py tests/test_scraper.py
git commit -m "feat(scraper): add board filtering, proxy support, and LinkedIn pacing cooldown"
```
