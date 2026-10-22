# JobSpy Version Checker (Approach A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a lightweight, non-blocking version checker for `python-jobspy` that checks PyPI for updates, caches results to avoid spamming PyPI, logs update notices on daemon start and `stats` CLI, and includes an informational note in the daily Telegram digest when a new version is available.

**Architecture:**
1. Create `src/scraper/version_checker.py` providing `VersionCheckResult` and `check_jobspy_update()` with in-memory TTL caching and graceful non-blocking failure modes using `httpx`.
2. Update `src/notifier/templates.py` with `format_version_update_notice(result: VersionCheckResult) -> str` (Ukrainian localization with safe HTML tags) and update `format_digest_chunks` and `TelegramNotifier.send_daily_digest` to append `update_notice` to the final chunk's footer.
3. Integrate update check into `DaemonService.run_daily_digest_cycle`, daemon startup logging in `DaemonService.start`, and `jobspy-coach stats` display in `src/cli.py`.

**Tech Stack:** Python 3.10+, httpx, packaging, pytest, rich, Typer

**Spec:** `docs/PM_AGENT_HANDOFF.md` & Approach A specifications.

## Global Constraints

- Python 3.10+ compatible (tested on Python 3.14.x environment `.venv`).
- Non-blocking: version check must time out quickly (default $\le 3.0$ seconds) and never crash, block, or delay core scraping or evaluating flows.
- Zero-leak privacy: no private candidate data or secrets involved.
- Network failure tolerance: PyPI errors, HTTP 4xx/5xx, timeouts, or lack of internet connection must return `None` without raising exceptions.
- Backward compatibility: existing 152 tests must remain 100% green.
- No global Typer callback: only `stats` and `daemon.start()` execute the check; one-shot CLI commands remain fast and isolated.
- Zero-job digest behavior: when `pending` is empty, no empty digest is sent to Telegram; version notice is sent when qualifying jobs are dispatched.

## Review Focus

1. Version comparison must handle semantic versioning cleanly (e.g. `1.4.0` > `1.3.0`, `1.10.0` > `1.9.0`) using `packaging.version.parse`.
2. Cache TTL must avoid repeated PyPI network calls when called frequently within a short window.
3. Digest formatting must not exceed Telegram's 4,096 char limit when the update notice is appended.
4. If installed version cannot be determined (e.g. development editable install without metadata), it falls back gracefully without exception.
5. All mocks in tests must isolate PyPI calls to ensure offline test suite reproducibility.

---

### Task 1: Version Checker Core Module

**Files:**
- Create: `src/scraper/version_checker.py`
- Test: `tests/test_version_checker.py`

**Interfaces:**
- Produces:
  ```python
  @dataclass
  class VersionCheckResult:
      has_update: bool
      installed_version: str
      latest_version: str

  def check_jobspy_update(
      package_name: str = "python-jobspy",
      timeout: float = 3.0,
      cache_ttl_seconds: float = 43200.0,
      force_refresh: bool = False,
  ) -> Optional[VersionCheckResult]: ...
  ```

- [ ] **Step 1: Write the failing tests**

In `tests/test_version_checker.py`, write tests covering:
1. `test_version_check_update_available`: Mock PyPI returning version `1.4.0` when installed is `1.3.0` -> `has_update=True`, `installed_version='1.3.0'`, `latest_version='1.4.0'`.
2. `test_version_check_already_latest`: Mock PyPI returning `1.3.0` when installed is `1.3.0` -> `has_update=False`.
3. `test_version_check_installed_newer`: Mock PyPI returning `1.2.0` when installed is `1.3.0` -> `has_update=False`.
4. `test_version_check_pypi_error_returns_none`: Mock PyPI 500 error or timeout -> returns `None` without raising.
5. `test_version_check_caching`: Second call within TTL does not query PyPI again unless `force_refresh=True`.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_version_checker.py -v`
Expected: FAIL (ModuleNotFoundError: No module named 'src.scraper.version_checker').

- [ ] **Step 3: Implement `src/scraper/version_checker.py`**

Implement `VersionCheckResult` dataclass and `check_jobspy_update` with module-level TTL caching, `importlib.metadata.version`, `packaging.version.parse`, and httpx GET `https://pypi.org/pypi/{package_name}/json`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_version_checker.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/scraper/version_checker.py tests/test_version_checker.py
git commit -m "feat(scraper): add lightweight cached PyPI version checker for python-jobspy"
```

---

### Task 2: Digest Template & Notifier Integration

**Files:**
- Modify: `src/notifier/templates.py:96-133`
- Modify: `src/notifier/telegram_notifier.py:112-123`
- Test: `tests/test_notifier.py`

**Interfaces:**
- Consumes: `VersionCheckResult` from `src.scraper.version_checker`
- Produces:
  ```python
  def format_version_update_notice(result: VersionCheckResult) -> str: ...

  def format_digest_chunks(
      jobs: List[JobPosting],
      date_str: str,
      update_notice: Optional[str] = None,
  ) -> List[str]: ...

  def send_daily_digest(
      self,
      jobs: List[JobPosting],
      date_str: str,
      update_notice: Optional[str] = None,
  ) -> int: ...
  ```

- [ ] **Step 1: Write the failing tests**

In `tests/test_notifier.py`:
1. `test_format_version_update_notice`: Verify formatting of `VersionCheckResult` into Ukrainian notice containing escaped versions and `<code>pip install -U python-jobspy</code>`.
2. `test_format_digest_chunks_with_update_notice`: Verify `update_notice` is appended to the final chunk's footer and chunk length $\le 3800$.
3. `test_format_digest_chunks_without_update_notice`: Ensure backward compatibility when `update_notice=None`.
4. `test_send_daily_digest_forwards_update_notice`: Verify `send_daily_digest` forwards `update_notice` to `format_digest_chunks`.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_notifier.py -k update_notice -v`
Expected: FAIL.

- [ ] **Step 3: Implement `format_version_update_notice` and update `templates.py` and `telegram_notifier.py`**

Update `src/notifier/templates.py` and `src/notifier/telegram_notifier.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_notifier.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/notifier/templates.py src/notifier/telegram_notifier.py tests/test_notifier.py
git commit -m "feat(notifier): add format_version_update_notice and digest chunk integration"
```

---

### Task 3: Daemon & CLI Integration

**Files:**
- Modify: `src/scheduler/daemon.py`
- Modify: `src/cli.py`
- Test: `tests/test_daemon.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `check_jobspy_update` from `src.scraper.version_checker`, `format_version_update_notice` from `src.notifier.templates`
- Produces:
  - `DaemonService.run_daily_digest_cycle`: checks for updates, formats notice, and passes to `send_daily_digest`.
  - `DaemonService.start`: logs version check notice at startup.
  - `cli.stats`: retrieves and displays `JobSpy Version` in rich table.

- [ ] **Step 1: Write the failing tests**

In `tests/test_daemon.py`:
1. `test_daemon_daily_digest_includes_update_notice`: When `check_jobspy_update` returns `has_update=True`, `send_daily_digest` is called with the formatted notice.
2. `test_daemon_daily_digest_no_update_notice_when_none`: When `check_jobspy_update` returns `has_update=False` or `None`, `update_notice` is None.

In `tests/test_cli.py`:
1. `test_cli_stats_includes_jobspy_version`: `jobspy-coach stats` displays JobSpy version row in table.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_daemon.py -k update_notice tests/test_cli.py -k stats -v`
Expected: FAIL.

- [ ] **Step 3: Implement integration in `daemon.py` and `cli.py`**

Wire `check_jobspy_update()` into `DaemonService.run_daily_digest_cycle`, `DaemonService.start`, and `src/cli.py:stats()`.

- [ ] **Step 4: Run full test suite**

Run: `.venv/bin/pytest -v`
Expected: 155+ tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/scheduler/daemon.py src/cli.py tests/test_daemon.py tests/test_cli.py
git commit -m "feat(daemon,cli): integrate JobSpy version check into daily digest, daemon startup, and stats"
```
