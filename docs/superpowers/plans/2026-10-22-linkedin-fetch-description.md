# LinkedIn Description Fetching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable full job description fetching for LinkedIn in `JobSpyScraper` (resolving the issue where 100% of LinkedIn jobs were dismissed due to empty descriptions) and purge existing empty LinkedIn stub records from `data/jobs.db`.

**Architecture:** 
1. In `src/scraper/jobspy_scraper.py`, add `fetch_description: bool = True` to `JobSpyScraper` and forward `fetch_description=self.fetch_description` to `jobspy.scrape_jobs`.
2. In `src/storage/db.py`, add a clean method `delete_empty_description_jobs(site: str = "linkedin") -> int` to remove stub records, and execute it against `data/jobs.db` to remove the 107 empty records.

**Tech Stack:** Python 3.10+, python-jobspy, SQLite, pytest

**Spec:** `docs/PM_AGENT_HANDOFF.md`

## Global Constraints

- Python 3.10+ compatible (running in Python 3.14.7 virtual environment `.venv`)
- Strict backward compatibility: `fetch_description` defaults to `True`
- No regressions: all 141 existing unit and integration tests must continue to pass
- Type hints throughout
- Proper Git hygiene: atomic commits per task

## Review Focus

1. `fetch_description` must default to `True` so LinkedIn descriptions are fetched automatically without requiring manual configuration.
2. Passing `fetch_description=True` to `scrape_jobs` must not break other boards (Indeed, Glassdoor, Google).
3. `delete_empty_description_jobs` must only delete jobs where `description` is NULL or empty string (`""`) for the designated site, never deleting jobs with valid descriptions.
4. All unit tests must pass without regressions.

---

### Task 1: Scraper LinkedIn Description Fetching

**Files:**
- Modify: `src/scraper/jobspy_scraper.py`
- Test: `tests/test_scraper.py`

**Interfaces:**
- Consumes: `JobSpyScraper(fetch_description: bool = True)`
- Produces: `scrape_jobs(..., fetch_description=self.fetch_description)`

- [x] **Step 1: Write the failing test**

In `tests/test_scraper.py`, add `test_jobspy_scraper_passes_fetch_description` verifying that:
1. `JobSpyScraper` has `fetch_description=True` by default.
2. When `scrape_board` is invoked, `scrape_jobs` receives `fetch_description=True` in its keyword arguments.
3. When `JobSpyScraper(fetch_description=False)` is configured, `scrape_jobs` receives `fetch_description=False`.

- [x] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_scraper.py -k test_jobspy_scraper_passes_fetch_description -v`
Expected: FAIL (`fetch_description` not passed in kwargs).

- [x] **Step 3: Implement `fetch_description` parameter in `JobSpyScraper`**

1. In `src/scraper/jobspy_scraper.py`:
   - Add `fetch_description: bool = True` to `JobSpyScraper.__init__` and set `self.fetch_description = fetch_description`.
   - In `JobSpyScraper.scrape_board`, pass `fetch_description=self.fetch_description` to `scrape_jobs(...)`.

- [x] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_scraper.py -v`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add src/scraper/jobspy_scraper.py tests/test_scraper.py
git commit -m "feat(scraper): pass fetch_description to JobSpy for full LinkedIn job details"
```

---

### Task 2: Database Empty Description Remediation

**Files:**
- Modify: `src/storage/db.py`
- Test: `tests/test_storage.py`

**Interfaces:**
- Produces: `Database.delete_empty_description_jobs(site: str = "linkedin") -> int`

- [x] **Step 1: Write the failing test**

In `tests/test_storage.py`, add `test_database_delete_empty_description_jobs` verifying that:
1. A job with empty description (`""` or `None`) on `linkedin` is deleted.
2. A job with valid description on `linkedin` is NOT deleted.
3. A job with empty description on `indeed` is NOT deleted when `site="linkedin"`.
4. Returns the count of deleted rows.

- [x] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_storage.py -k test_database_delete_empty_description_jobs -v`
Expected: FAIL (AttributeError: 'Database' object has no attribute 'delete_empty_description_jobs').

- [x] **Step 3: Implement `delete_empty_description_jobs` in `src/storage/db.py` and clean `data/jobs.db`**

1. In `src/storage/db.py`, implement:
   ```python
   def delete_empty_description_jobs(self, site: str = "linkedin") -> int:
       with self._get_connection() as conn:
           cursor = conn.cursor()
           cursor.execute(
               "DELETE FROM jobs WHERE site = ? AND (description IS NULL OR trim(description) = '')",
               (site,),
           )
           conn.commit()
           return cursor.rowcount
   ```
2. Execute one-off cleanup against `data/jobs.db` using python to purge the 107 empty LinkedIn records.

- [x] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_storage.py -v`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add src/storage/db.py tests/test_storage.py
git commit -m "fix(storage): add delete_empty_description_jobs and purge empty LinkedIn stub records"
```
