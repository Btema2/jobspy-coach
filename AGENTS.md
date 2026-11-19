# AGENTS.md — JobSpy Coach Project Memory & Operational Handbook

> **Project Memory & Continuity Guide**: This file serves as the definitive memory bank for AI agents, developers, and autonomous systems maintaining and extending `jobspy-coach`. It encapsulates architectural decisions, hard-earned operational gotchas, candidate persona rules, and strict agent development protocols.

---

## 1. Candidate Privacy Protocol & Persona Architecture

> [!CAUTION]
> **STRICT PRIVACY MANDATE FOR ALL AI AGENTS & DEVELOPERS**:
> Under **NO circumstances** should private user data (candidate name, specific personal background, city/country of residence, native languages, personal credentials, or customized search targets) be committed, documented, or pushed to any public or remote repository.
> 
> - **Local Storage Only**: All actual candidate profiles and user-specific personas are stored exclusively in local, git-ignored places: `.env`, the legacy `.candidate_profile.md` / `queries.json` (imported once) and the SQLite database `data/jobs.db`, which holds the stored profile (persona, queries, rules and the last 10 versions) after the import. Never commit the database or a profile export.
> - **Public Neutrality**: Tracked repository files, documentation, commit messages, and PRs must strictly use sanitized, generic templates and configurable placeholders (e.g., `<TARGET_ROLE>`, `<TARGET_REGION>`).
> - **Runtime Persona Injection**: When running locally, the evaluator uses the persona stored in the profile (SQLite) to evaluate vacancies against the user's private persona; a legacy `.candidate_profile.md` is only imported once, after which editing it changes nothing (use `profile export` / `profile import`). If the profile has no persona, the engine falls back to a generic professional template.


---

## 2. Core Architecture & Subsystems

```
src/
├── config.py              # Pydantic Settings with comma-separated list validators
├── cli.py                 # Unified Typer CLI (scrape, evaluate, digest, stats, run, init, rehash, cleanup-empty, test-telegram, requeue [--thresholds] [--dry-run], profile preview|export|import, bot unpair); `init` is the interactive bootstrap (hidden credential prompts, pairing, paused starter profile; options `--skip-credentials`, `--start/--no-start`, `--language`, never a secret); a callback auto-imports legacy files before each command (not for --help)
├── logging_config.py      # configure_logging(): httpx/httpcore at WARNING + secret-redacting log filter
├── i18n.py                # Static message tables for alerts, digests and `test-telegram`: `en` and `uk` (= today's strings); `table_language` / `t`
├── envfile.py             # Safe `.env` edits: `merge_env_file` (line-preserving, advisory lock, mode 600, atomic), `read_env_value`, `validate_env_value`
├── bot/                   # Telegram management bot (runs as a daemon thread inside `run`)
│   ├── client.py          # TelegramBotClient: getUpdates / sendMessage / editMessageText / answerCallbackQuery / setMyCommands, typed errors, stop-aware 429 wait
│   ├── updates.py         # `parse_update` (never raises) -> `Update`; `OutgoingMessage`
│   ├── state.py           # BotStateStore: the only code touching `bot_state` (offset, pairing, pairing code hash, attempt counter, wizard state)
│   ├── auth.py            # Pairing codes, `Authorizer.decide` on every update (user id + chat id + private chat), single-use 15-minute code, lock after 5 wrong codes
│   ├── poller.py          # BotPoller: long poll loop; offset persisted before anything executes; 15-minute staleness; back-off; stops on the shutdown event
│   ├── commands.py        # CommandRouter + `COMMANDS` menu: all commands go through `profile/ops.py`; `BotContext`
│   ├── countries.py       # Built-in country table (names, aliases, languages, regions) for `/add_market`
│   ├── render.py          # `esc`, `utf16_len`, `chunk_html` (3,800 UTF-16 units, never inside a tag or entity), `paginate`, `render_apply`
│   ├── wizard.py          # `/setup`: deterministic stepwise wizard with nonce-bound inline buttons (state in `bot_state.wizard_state`)
│   ├── stats.py           # `/stats`: 7-day per-role calls, new jobs, relevant jobs; failures separate from empty results
│   ├── probe.py           # `/test <role>`: small trial search on one worker thread, never writes to the DB
│   └── bootstrap.py       # `run_init`: credential validation, `.env` merge, starter profile, pairing step, optional daemon start
├── filters/
│   ├── pre_filter.py      # `pre_filter_job` / `classify_title`: apply compiled rules (rules=None -> legacy seed), stub-description checks
│   ├── rules.py           # CompiledRule + `match_title`: block -> rescue -> soft -> seniority, stable rule keys
│   └── seniority.py       # Built-in English seniority term table + extra terms per market language
├── scraper/
│   ├── query_catalog.py   # Only `SearchQuery` and `query_key`; there is no curated query catalog anymore (queries come from the stored profile)
│   ├── jobspy_scraper.py  # python-jobspy wrapper with jitter, proxies, ScrapeOutcome
│   ├── outcome.py         # ScrapeOutcome: per-(query, board) result (ok / empty / failed / skipped)
│   └── coordinator.py     # ScrapeCoordinator: per-call lock + shared LinkedIn pacing
├── profile/               # Stored, versioned search profile (single source of queries and the evaluator prompt)
│   ├── errors.py          # Typed profile errors (validation, version conflict, compile, legacy import, export refused)
│   ├── snapshot.py        # Plain dataclasses + strict dict/YAML shape of a whole profile
│   ├── store.py           # The only code touching the profile tables (consistent read, optimistic-version write)
│   ├── ops.py             # Closed operation set, validation, diff, versions, undo/restore, starter profile
│   ├── compiler.py        # Pure snapshot -> CompiledProfile (queries, rules, rubric prompt, ScoringPolicy, estimate, warnings)
│   ├── importer.py        # Legacy queries.json / .candidate_profile.md auto-import, YAML export / import
│   ├── legacy_seed.py     # Today's pre-filter regexes and rubric slot text as data (seed rules, V1 text for upgrades)
│   ├── backtest.py        # Dry run of a rule change against stored titles (count + samples; not meaningful under 200 jobs)
│   └── requeue.py         # `requeue_prefiltered` (revive rows that now pass) and `rebucket_evaluated` (re-score from stored fields)
├── storage/
│   ├── models.py          # Dataclasses (JobPosting, JobEvaluationResult, ScrapeRun)
│   ├── db.py              # SQLite WAL repository with SHA-256 deduplication
│   └── migrations.py      # Add-only ordered migrations + schema_version table (4: `bot_state`)
├── evaluator/
│   ├── chain.py           # ProviderChainEvaluator: NIM -> OpenRouter fallback, validation, score post-processing
│   ├── providers.py       # NvidiaNimProvider / OpenRouterProvider (httpx, retries, error classification, daily budget)
│   ├── rate_limiter.py    # Thread-safe sliding-window limiter (stop-event aware)
│   ├── circuit_breaker.py # Per-provider breaker with a single half-open probe
│   ├── errors.py          # Typed errors (transient, permanent, quota, invalid response, providers unavailable, ...)
│   ├── parsing.py         # 3-tier JSON extraction (direct -> code fence -> outermost {...})
│   ├── prompts.py         # No-profile fallback prompt (legacy rubric inputs + persona from the local file) and the job prompt
│   ├── rubric.py          # Rubric template: bands from the profile thresholds, slots inside a delimited `<profile_data>` block
│   ├── scoring.py         # ScoringPolicy + pure `final_score` / `decide_wow` (also used to re-bucket stored rows)
│   └── schemas.py         # Pydantic schemas for structured LLM responses
├── notifier/
│   ├── templates.py       # Telegram HTML formatting (strings from `i18n.py`), length guards & chunking
│   └── telegram_notifier.py # Telegram Bot API client for alerts (WOW alerts & daily digest)
└── scheduler/
    └── daemon.py          # APScheduler 24/7 background service orchestration; starts the bot poller, `trigger_cycle`, cycle and digest locks
scripts/
└── bakeoff.py             # Model bake-off for NIM / OpenRouter free models (writes to git-ignored local/)
docs/
└── model-selection.md     # Recorded bake-off results and chosen defaults
```

### Key Workflows
1. **Scraping**: `JobSpyScraper` queries configured job boards (LinkedIn, Indeed, Glassdoor, Google) with randomized jitter (`4.0–9.0s`).
2. **Deterministic Pre-Filtering**: Zero-cost title rules from the stored profile run before any LLM call, then stub descriptions (<150 chars or <30 words) are dropped. Four rule kinds: `block` (always dropped), `rescue`, `soft` (dropped unless a rescue term is present) and `seniority` (dropped unless rescued, built from `target_level` plus a built-in English term table and extra terms per market language); order: block rules, then any rescue match passes, then soft, then seniority. User terms are literals (`exact` = `(?<!\w)TERM(?!\w)`, `prefix` = `TERM\w*`, length 3-60, at most 500 rules); `pattern` rules exist only in the tracked seed. An imported legacy profile carries the old regex groups as seed `pattern` rules, so behaviour is unchanged; a new profile ships no domain terms. `pre_filter_job(title, description, rules=None)` keeps `rules=None` = legacy seed. Each pre-filtered row stores the dropping rule's stable key in `jobs.pre_filter_rule` (migration 3; None for legacy rows and stubs); `Database.get_recent_prefiltered` feeds the planned `/dropped` view.
3. **Cross-Board Deduplication**: Normalizes each field to compute SHA-256 `dedup_hash` over `company:title:location` by exact equality of the normalized strings (no fuzzy matching). Company: casing, diacritics, parentheticals, trailing legal forms (`LLC`, `GmbH`, `S.A.`, `Sp. z o.o.`) and trailing regional/group words (`Polska`, `International`, `Group`, ...). Title: connector words (`and`, `&`, `und`, a middle `i`) and gender markers like `(m/w/d)` are dropped, but other parentheticals, seniority and language variants are kept. Location: reduced to the city (first comma segment, `Metropolitan Area` removed); remote locations and country-only locations keep their full text. Existing rows keep their old hashes until `jobspy-coach rehash` is run once (see section 5).
4. **LLM Evaluation**: `DaemonService.evaluate_unprocessed` evaluates non-prefiltered vacancies concurrently (worker threads, `EVAL_MAX_WORKERS`) through `ProviderChainEvaluator`: NVIDIA NIM first, then OpenRouter pinned free models. The chain extracts a 1–10 score, executive summary (`summary`), English rationale, and the model that actually answered (`eval_model`), and applies hard-reject / bonus adjustments in code (`src/evaluator/scoring.py`, `final_score`). The model's `summary` and `is_core_role` JSON keys replace `summary_uk` / `role_is_direct_coaching` (the old keys are still accepted; the DB columns `eval_summary_uk` / `eval_is_core_role` are unchanged). Thresholds, bonus (applies only when the profile's `bonus_criteria` is non-empty; points = `bonus_points`), the non-core cap and the output language come from the stored profile and are applied **per cycle** through a `ScoringPolicy` passed to `evaluate_job(job, policy=...)`: after the bonus, a role the model does not flag `is_core_role` is capped at `wow - 0.1` (applied after rounding; a missing flag counts as false; **off** when the profile has no core-role definition), so only core roles can reach WOW; the cap is logged at INFO. The job prompt carries `Remote: yes` (remote_status `verified`) or `Remote: unverified`; for a job found by a `remote_only` market query it reads `Remote: unverified (found with the remote filter)` and the rubric adds a remote-only hard-reject condition. Rows from before migration 1 read as `unverified`. The chain is built lazily on first use (a fresh clone without keys must still construct `DaemonService`); with no provider configured it raises `ConfigurationError`.
5. **Two-Tier Telegram Alerts**:
   - **Instant WOW Alerts**: Vacancies with a final score at or above the profile's `wow_threshold` (default 8.5; only possible for core roles while the cap is on) immediately trigger Telegram push notifications with an inline application button. `is_wow` is derived by the daemon (`decide_wow`) from the final score AND `is_core_role` (and not hard-reject), so a lowered threshold cannot promote capped roles; the LLM does not decide it.
   - **Daily Morning Digest**: Every morning at the scheduled digest time, qualifying vacancies (`digest_threshold` <= score, defaults 6.5 and up; WOW-level rows that never alerted, e.g. after `requeue --thresholds`, join it) are formatted in clean HTML chunks ($\le 3,800$ chars) and dispatched.
   - **Low-Score Dismissal**: Vacancies scoring below the profile's `digest_threshold` (default 6.5) are immediately marked `dismissed` to keep queues lean.
   - **Stale Expiration**: Pending items older than 7 days transition to `expired_dismissed`.
   - **Language tables**: alert, digest, update-notice and `test-telegram` strings come from `src/i18n.py` (`en` and `uk`). The profile's `output_language` selects a table: unset keeps today's strings byte-identical (the `uk` table), a code with a table uses it, any other code falls back to English. `init` creates its starter profile with `output_language="en"` (`--language` overrides). Bot replies are English only. A golden fixture (`tests/fixtures/legacy_alert_strings.json`) pins today's output.

---

## 3. Critical Learnings & Operational Gotchas

### A. LinkedIn Scraping (`python-jobspy`)
1. **`fetch_description=True` is Mandatory**:
   - In `python-jobspy`, Indeed returns descriptions inline in search cards, but **LinkedIn requires a secondary detail request** (`/jobs/view/{job_id}`).
   - JobSpy defaults `fetch_description=False`. If omitted, JobSpy returns `description=None` for all LinkedIn jobs, causing our pre-filter to drop 100% of LinkedIn jobs as `DESCRIPTION_TOO_SHORT`.
   - **Always pass `fetch_description=True`** in `JobSpyScraper.scrape_board_outcome` (`scrape_board` is a thin wrapper).
   - A failed detail request (HTTP 429, redirect) is only logged by python-jobspy and yields `description=None`. `scrape_board_outcome` skips such LinkedIn rows (WARNING with a count; not saved, so a later cycle retries them) instead of storing them as dismissed, because the dedup hash would suppress them forever. Short but non-empty descriptions, other boards, and titles the pre-filter drops on their own are unaffected.
   - `scrape_board_outcome(query, board, track_cycle=True)` returns a `ScrapeOutcome` (postings, rows returned, pre-filtered count, `status`, error/skip reason). A board error reported in `df.attrs["errors"]` with no rows is `failed` (a blocked board, not an empty result); boards skipped for a missing country are `skipped`. `scrape_board` is a thin wrapper returning only the postings. The daemon records `scrape_runs.outcome/rows_returned/prefiltered/query_key` and `job_hits` from it; a skipped board is stored with status `completed`, outcome `skipped`.
   - All board calls go through `ScrapeCoordinator` (one lock per call, shared LinkedIn cooldown); the scraper exposes `last_linkedin_call_time` and `linkedin_cooldown_seconds` as delegating properties.
   - Scrape progress (`[i/N] query board ...` before/after each call plus a per-board summary) is logged by the scraper itself (`begin_cycle`/`scrape_board_outcome`/`end_cycle`), because the CLI `scrape` and the daemon call `DaemonService.run_pipeline_cycle`, which calls `DaemonService._scrape_call` -> `scrape_board_outcome` (falling back to `scrape_board` only for test doubles) and never `scrape_all_queries`.
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
   - **Rule**: Never query Glassdoor with a country it does not support (e.g. `country_code="poland"`). The profile compiler enforces this with a country-support code table (a market with no country skips Indeed and Glassdoor and says why).
2. **ZipRecruiter**:
   - ZipRecruiter blocks European IP traffic with HTTP 307 redirects.
   - **Rule**: ZipRecruiter only for US/UK/Canada countries; the compiler's code table enforces it for expanded queries and warns for pinned ones.
3. **Indeed** (verified live against `python-jobspy` 1.4.0):
   - Indeed searches **one country at a time**. A pan-regional location such as `"Europe"` returns 0 rows (also under `country_indeed="usa"`), and defaulting `country_indeed` to `"usa"` for queries without a `country_code` silently searches the wrong market. **Rule**: Indeed and Glassdoor require `query.country_code`; the scraper skips them (one warning per process, then DEBUG) when it is missing. Never default to `"usa"`.
   - **Quotes are always stripped** from the Indeed search term (`"Career Coach"` becomes `Career Coach`), with or without `is_remote`; LinkedIn and Google keep them. Live probe (one country, 10 results): a quoted term at 48h returned 1 row vs 8 unquoted, so quotes cut yield several-fold even without the remote filter.
   - **Minimum 168 h window for Indeed**: the scraper passes `hours_old = max(HOURS_OLD_WINDOW, INDEED_MIN_HOURS_OLD=168)` to Indeed only (unquoted: 8 rows at 48h vs 10 at 168h and wider); other boards keep `HOURS_OLD_WINDOW`. Dedup hashes make re-seen postings harmless.
   - In 1.4.0 the Indeed filters (`hours_old`, `job_type`, `is_remote`, `easy_apply`) are AND-combined, not mutually exclusive (older versions documented exclusive groups), so `hours_old` and `is_remote` are both passed.
4. **Google Jobs**:
   - The scraper passes an explicit `google_search_term` (`<term> jobs near <location> <time window> [remote]`, same shape python-jobspy derives itself; per-query override via `google_search_term` on a pinned query, imported from `queries.json`). Time windows: <=24h "since yesterday", <=72h "in the last 3 days", <=168h "in the last week", else "in the last month".
   - From some networks Google answers the request with a JavaScript/bot-challenge page, so python-jobspy reports `no job data returned` regardless of the query string. The scraper logs this once per process (WARNING) instead of returning nothing silently; remove `google` from `site_names` if it persists.

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
   - `ProviderChainEvaluator.evaluate_job(job, system_prompt=None, policy=None)`: the daemon passes the cycle's `ScoringPolicy` (prompt, thresholds, bonus, cap, rules, remote-filter query ids); an explicit `system_prompt` argument wins over the policy's prompt; without a policy a lazily built no-profile default is used. The policy is a frozen dataclass handed to workers by argument, never stored on the chain. `JobEvaluationResult.raw_score` keeps the pre-adjustment score; `update_evaluation` stores `eval_raw_score`, `eval_is_core_role`, `eval_bonus_flag` and `eval_hard_reject` for later re-bucketing.
4. **Concurrency model (threads)**:
   - Thread rule: evaluation worker threads run only `chain.evaluate_job`; the thread that calls `evaluate_unprocessed` handles their DB and Telegram work. The bot poller thread (and its single `/test` worker) is a second, explicitly allowed DB and Telegram thread with its own `Database` instance; it never runs evaluation. Profile writes from the bot go through `ops` (`BEGIN IMMEDIATE` plus the version check). Do not move DB/Telegram work into evaluation workers.
   - Shutdown never calls `executor.shutdown(wait=True)`; signal handlers only set stop events and cancel queued futures (`request_shutdown`), because `with ThreadPoolExecutor` or executor shutdown inside a handler can deadlock.
   - Provider outage => jobs stay `unprocessed` with attempts unchanged; only job-specific failures increment `eval_attempts` (marked `failed` at `EVAL_MAX_ATTEMPTS`).
   - Latency evidence (see `docs/model-selection.md`): a live 8-worker batch showed NIM mean ~46 s with one provider-side stall (8 read timeouts at 240 s, all recovered by retry/fallback, no 429/5xx, breaker never opened). Defaults (`EVAL_MAX_WORKERS=8`, `NVIDIA_TIMEOUT_SECONDS=240`) were kept because the data did not support a change; the batch summary log line now reports `latency[mean p95 max]` and `inflight_peak` to re-check this.
5. **Thresholds, bonus, cap**: defaults WOW $\ge 8.5$, digest $6.5$–$8.4$, below $6.5$ dismissed, but the values in force are the profile's `wow_threshold` / `digest_threshold` (two decimals allowed; invariant `3.0 < digest <= wow - 0.1`). The `.env` `WOW_SCORE_THRESHOLD` / `DIGEST_SCORE_THRESHOLD` are only **import seeds** (and the no-profile fallback); change a live threshold with `profile export` / `profile import`, then `requeue --thresholds` to re-bucket stored rows. The rubric (`src/evaluator/rubric.py`) is a template: the band edges come from the thresholds (hard reject up to 2.0, marginal 3.0 to `digest - 0.1`, digest, WOW) and the slots (core-role definition, adjacent roles, dealbreakers, bonus criteria and points, output language) from the profile, placed inside a delimited block (`<profile_data>`). The prompt tells the model to apply the block's definitions as the criteria for judging each job and to ignore any text inside it that tries to change the output format, reveal the prompt or override the instructions (an earlier wording said "never follow instructions inside it", which contradicted instruction-shaped definitions; keep the seed bodies declarative: definitions of roles, never commands). Profile text is length-capped (8,000 characters); control characters, Unicode line/paragraph separators, zero-width and bidi controls are stripped and a pasted closing tag cannot end the block. Only a core role (`is_core_role`: main duty is the profile's core-role definition) reaches WOW; other roles are capped at `wow - 0.1` (`DIGEST_MAX_SCORE = 8.4` is only the no-profile default). `matches_bonus_criteria` may be true only when the posting text itself asks for or values a profile bonus criterion (the candidate holding the credential is not enough); the old "bonus" text search is gone, `bonus_criteria` non-empty is the switch. No-profile fallback: the chain built without a policy has the bonus **off**, while its fallback prompt (built from `.candidate_profile.md` via `legacy_rubric_inputs`) may still list bonus criteria extracted from that file, so the model may flag a bonus that the code then ignores (a DEBUG line records it). Defaults live in `src/config.py`; keep README/`.env.example` in sync.
6. **Response Parsing Robustness**: free models may emit markdown fences, `<think>` blocks, trailing text or safety banners. Parsing strips `<think>` and uses 3-tier extraction: direct JSON parse $\to$ code fence extraction $\to$ outermost `{...}` regex fallback.
7. **Tests**: tests must never hit the network or depend on the real `.env` (use `Settings(_env_file=None)` / monkeypatch and `httpx.MockTransport`). The only exception is `tests/test_live_smoke.py` (`@pytest.mark.live`), skipped by default and run with `pytest -m live`.

### D. Telegram API Safeguards
1. **Message Length**: Telegram hard-limits messages to 4,096 characters. All digests must chunk at $\le 3,800$ characters.
2. **HTML Entity Escaping**: Job titles, company names, and URLs must pass through `html.escape()` (`&`, `<`, `>`) before insertion into HTML message strings to prevent 400 Bad Request parsing errors.
3. **Summary & Reasoning Guards**: Summaries are clamped to 600 chars and reasoning to 800 chars before formatting.
4. **Token Hygiene**: The bot token is part of the API URL, so `httpx` INFO logging would print it. `configure_logging()` (`src/logging_config.py`) keeps `httpx`/`httpcore` at WARNING and attaches a redacting filter to the root handlers. Never log `str(exception)` from an httpx call or the request URL. If a token was ever logged or pasted anywhere, rotate it (BotFather `/revoke`) and update `.env`. Error text stored in `scrape_runs` goes through `redact_secrets` (which also masks `scheme://user:pass@` proxy credentials) and is truncated to 300 chars by the daemon's `sanitize_error_message`.
5. **Bot client and `.env`**: the bot client (`src/bot/client.py`) follows the same token rules: it never logs the request URL, a response description or `str()` of an httpx error (network errors surface as the exception type name only), and the token is also string-replaced in any error text before `redact_secrets`. `.env` is written only through `src/envfile.py` (atomic, line-preserving, mode 600 from creation, advisory lock `.env.lock`, which is git-ignored); `init` never takes a secret from an argument (hidden prompts only) and refuses values with control characters, `$` or backticks (dotenv interpolation).

### E. SQLite Concurrency & Database Operations
1. **WAL Mode**: SQLite must always run in Write-Ahead Logging mode (`PRAGMA journal_mode = WAL;`) with `PRAGMA busy_timeout = 5000;` and `PRAGMA synchronous = NORMAL;`. `_get_connection` sets `busy_timeout` first and retries the WAL pragma on "database is locked", so two processes initialising a fresh file at once do not fail.
2. **Dedup rehash**: `Database.rehash_all()` (CLI `rehash`) is idempotent. Per group of colliding rows it keeps the alert-sent row, else the evaluated row, else the oldest; other rows are deleted, but a second alert-sent row is never deleted (it keeps a unique filler hash). `job_hits` of deleted duplicates are re-pointed to the surviving row. Stop the daemon before running it.
3. **Stale Records Cleanup**: If bad scrape runs populate the database with stub records (e.g. empty descriptions), use `Database.delete_empty_description_jobs("linkedin")` to remove them so deduplication hashes don't permanently suppress valid future scrapes.
4. **Schema changes** go through `src/storage/migrations.py` (ordered, add-only: no renames, no drops; tracked in `schema_version`), never ad-hoc `ALTER TABLE`. New columns include `jobs.source_query`, `jobs.remote_status`, `jobs.eval_raw_score/eval_is_core_role/eval_bonus_flag/eval_hard_reject` (filled for new evaluations only) and the `scrape_runs`/`job_hits` additions. Migration 4 (`bot_core`) adds only the single-row `bot_state` table (update offset, paired user and chat id, pairing-code hash and expiry, wrong-attempt counter, wizard state); the Branch 5 tables (`pending_change`, `conversation`, `llm_consent`) do not exist yet. Migration 3 (`rules_and_rubric`, the only one Branch 3 adds) adds `jobs.pre_filter_rule` (+ index) and rewrites the stored `profile` row's definition fields that still equal the first-version seed definitions (fixed band numbers, `role_is_direct_coaching`) to the new seed text; text the user edited is left alone, and running it twice changes nothing. Snapshot schema is now 2: the stored `profile_versions` snapshots are not rewritten; loading a schema-1 snapshot or YAML export upgrades exactly those three definition fields the same way, so `restore` of an old version yields the new seed text. The seed core and adjacent bodies were reworded once more (declarative, see 3.C.5); migration 3 and the snapshot upgrade recognise both the first-version text and the short-lived first Branch 3 wording (exact match only, idempotent, edited text untouched).

### F. Search Profile (`src/profile/`)
1. **Source of truth**: the profile tables (`profile`, `roles`, `role_terms`, `markets`, `pinned_queries`, `rules`, `profile_versions`) are the only source of search queries and of the evaluator system prompt. There is no default query catalog and no fallback to one; `queries or get_active_queries()` no longer exists.
2. **Auto-import (CLI only)**: the `@app.callback()` in `src/cli.py` runs `auto_import` before every CLI command: when no profile exists and `queries.json` / `.candidate_profile.md` exist in the working directory, they are imported idempotently (files untouched; each `queries.json` entry becomes one pinned query; a persona with bonus criteria keeps bonus enabled; python-jobspy country aliases such as `us` are normalised to their canonical value; `query_id`s are kept exactly, case-only twins get a `~N` suffix and a note; a persona without `queries.json` is imported paused, with a note). After the import the persona lives in the profile database: editing `.candidate_profile.md` no longer changes scoring. The daemon never auto-imports; it only reads the profile. A broken legacy file or an unknown board/country in `queries.json` stops every command with exit code 1 and a message naming the file and entry; nothing is imported. Tests that can reach the CLI must `monkeypatch.chdir(tmp_path)`.
3. **"No profile" behaviour**: with neither a profile nor legacy files, `run` prints `no profile, run \`jobspy-coach init\`` and exits 1 before building the daemon. A daemon that is already running and finds no profile skips scraping and alerts once.
4. **Changes go through `ops`**: every change (CLI `profile import`, restore, later the bot) is operations -> validation (including a compile) -> one transaction with an optimistic version check -> a new `profile_versions` row (the last 10 snapshots are kept). Threshold invariants: `3.0 < digest <= wow - 0.1`, `wow <= 10`; free text at most 8,000 characters.
5. **Daemon cycle**: the profile is compiled once at the start of each cycle (`current_profile`) and that result drives everything in the cycle: scraping (queries and `rules` passed to the scraper), the per-call `ScoringPolicy` (system prompt, thresholds, bonus on/off and points, non-core cap, rules for the title safety net) and the digest threshold. Nothing is fixed at chain construction any more: the bonus switch is per call, so a profile change takes effect on the next cycle without a restart (a batch in flight keeps the policy it started with). A SQLite lock or busy timeout while reading the profile (`db_busy`) skips scraping without an alert and reuses the last good profile; **with no last good profile (e.g. at process start) evaluation is skipped as well** and the jobs stay `unprocessed` (a score computed against the no-profile prompt would dismiss rows irreversibly). Any other database error (no such table, disk I/O error) and an unreadable stored profile are compile errors: alert once per distinct text, keep the last good profile. The no-profile policy (legacy rules, cap at `wow - 0.1`) is built by one helper, `no_profile_policy`, for both the daemon (`.env` thresholds) and the chain (which receives the same `.env` thresholds from `build_chain_from_settings`). When a scraper double cannot take `rules`, the daemon logs a WARNING once and scrapes without them. An empty query list skips scraping and alerts once. A compile failure skips scraping, alerts once per distinct (redacted, truncated) error and keeps evaluating with the last good profile (in memory, else the newest compilable stored version). An alert is recorded as sent only after Telegram accepted it, so a failed send is retried next cycle. `paused` stops scraping only (evaluation and digests continue).
6. **History, backtest and requeue (Branch 3)**: pre-filtered and score-dismissed rows keep their dedup hash, so a rule or threshold change never revives them by itself. `profile import` prints a backtest of any rule change against the newest 5,000 stored titles (count of titles newly dropped / newly passing plus up to 5 sample titles each; with fewer than 200 stored jobs it prints `backtest not meaningful` instead of numbers; no old profile compares against the legacy seed ruleset). `requeue [--dry-run]` re-runs the current rules over pre-filtered rows and moves those that now pass back to `unprocessed` (dedup hash kept, attempts reset); stub descriptions stay dropped, and rows created more than 7 days ago (`STALE_PENDING_DAYS`) are only counted (`too old`) because they would expire before any digest. `requeue --thresholds [--dry-run]` recomputes the final score of `pending`/`dismissed` evaluated rows created within the last 7 days (older ones are only counted as `too old`: a row that became `pending` would expire at the next digest) from `eval_raw_score`, `eval_is_core_role`, `eval_bonus_flag`, `eval_hard_reject` under the current policy (also re-applying the title rules), with no LLM call: rows missing any of those fields (evaluated before they were stored) are skipped and counted; a row that now reaches the WOW level becomes `pending` with `eval_is_wow = 0` and joins the next digest (no instant alert is ever sent from here); `instant_sent`, `digest_sent` and `expired_dismissed` rows are never touched. Both write in one transaction; never run them while the daemon is running. `requeue --dry-run` is read-only: the CLI callback skips initialisation, migration and legacy auto-import for it, and it reports a database without `schema_version` (or an older schema) in one line instead of migrating. `ApplyResult` carries `old` and `new` snapshots, so a caller (CLI import, later the bot) backtests against exactly the base the change was applied to.
7. **Export**: `profile export` writes YAML (mode 0600) only to a git-ignored path (default `local/profile-export.yaml`; `profile-export*.yaml` is in `.gitignore`) unless `--force`. It contains private data; never commit or paste it.

### G. Telegram bot (`src/bot/`)
1. **Pairing**: an existing `TELEGRAM_CHAT_ID` (a positive integer) is adopted as the paired chat, by `init` and also when the poller starts on an unpaired `bot_state` (the local `.env` is the trust anchor; existing installs get the bot without re-running `init`). Otherwise `init` prints a one-time 8-character code (40 bits, stored only as a SHA-256 hash). `/start <code>` is accepted only from a private chat; it stores user id and chat id and writes `TELEGRAM_CHAT_ID` into `.env`. The code is single-use and expires after 15 minutes; 5 wrong attempts lock pairing in memory until the process restarts (the counter is reset when the `Authorizer` is constructed) with one WARNING, so a new code only works after a daemon restart; with `TELEGRAM_BOT_POLLING=false` `init` issues no code. `jobspy-coach bot unpair` clears the pairing **and blanks `TELEGRAM_CHAT_ID` in `.env`** (otherwise the next start would adopt it again); restart the daemon afterwards.
2. **Auth on every update**: `Authorizer.decide` checks user id, chat id and chat type (private) for messages and callback queries; forwarded messages are never commands. Anything else is dropped without a reply and logged as `rejected update <id> (<kind>, <chat type>)`, never with content. A group, a stranger, or the paired user inside a group cannot pair, run a command or press a forwarded button.
3. **Offset before execute (at-most-once)**: `BotPoller.process` persists `update_id + 1` before anything of the update runs, so a crash can lose a command but never replay `/undo`. The offset is stored exactly as `update_id + 1` and may go DOWN (Telegram can restart its update ids after a long idle; never skip an id just because it is below the saved offset, or the bot goes deaf); only an exact repeat of an update id inside one batch is dropped. `tests/fake_telegram.py` honours `offset` like the real API (confirm-drops lower ids, `ignore_offset=True` hands out server-chosen ids once). Mutating commands older than 15 minutes (`STALE_SECONDS = 900`) get a one-line "ignored, send again" reply; read commands still run. The staleness check applies to commands only; wizard callbacks are protected by the wizard nonce, the 30-minute wizard expiry and the profile version bound at wizard start. Updates left in a batch after the stop event are not acknowledged.
4. **Polling and shutdown**: `TELEGRAM_BOT_POLLING` (default `true`) switches the poller off for an install whose bot token is polled by another program; the daemon then only sends alerts. The poller is a daemon thread started before the startup cycle, with a 25 s long poll, its own `Database` and `TelegramBotClient`; both take the daemon's `_shutdown_requested` event (never the evaluation `_stop_event`, which is re-armed per batch). Never `join()` the poller without a timeout. 401 stops polling with an ERROR; 409 (a second consumer, or a webhook is set) logs one WARNING and backs off 30 s; 429 honours `retry_after`; other failures back off 1-60 s with one WARNING per outage. **Only one `getUpdates` consumer per bot token may run**: two pollers (for example two daemons, or a daemon plus another bot program) make Telegram answer 409 to each; use a dedicated bot token or `TELEGRAM_BOT_POLLING=false`.
5. **`/run_now` and the cycle lock**: `DaemonService` has a non-blocking `_cycle_lock` around `run_pipeline_cycle` (a scheduled cycle that finds it held is skipped with a WARNING) and a `_digest_lock` around the daily digest; `is_cycle_running()` covers both. `/run_now` and `/resume` call `trigger_cycle()` (schedules a one-off APScheduler job; answers started / busy / not ready); `/run_now` refuses while a cycle runs, `/requeue` runs under `cycle_guard()`, which takes both locks atomically for the duration of its writes (refused when one is held; a scheduled cycle that wants to start meanwhile is skipped). The CLI `requeue`/`evaluate` are other processes: never run them while the daemon works.
6. **`/test` limits**: one worker thread; at most one probe per 5 minutes (`TEST_COOLDOWN_SECONDS`); at most 3 calls of 5 results each (`TEST_MAX_CALLS`, `TEST_MAX_RESULTS`), only boards enabled in `SCRAPER_ENABLED_BOARDS` (boards the scraper would refuse are dropped before the 3-call budget and named in the reply), and LinkedIn only when the word `linkedin` is part of the command; through the same `ScrapeCoordinator` as the cycle (LinkedIn pacing is shared); never writes jobs, hits or scrape runs and is not counted against `max_calls_per_cycle`; it reports rows, new jobs, pre-filtered share and example titles and says it does not measure relevance.
7. **Output rules**: every reply is chunked at 3,800 UTF-16 units (`chunk_html`, never inside a tag or entity), all user-supplied terms go through `esc`, long lists are paginated, and handler errors are logged by exception type name only. The command menu is registered once with `setMyCommands` when the poller starts. `/setup` wizard buttons carry a random nonce bound to the profile version at wizard start and expire after 30 minutes.
8. **Command set** (`COMMANDS` in `src/bot/commands.py`): `/help`, `/profile`, `/roles`, `/markets`, `/exclusions`, `/preview`, `/dropped`, `/add_role`, `/remove_role`, `/add_market`, `/exclude`, `/threshold`, `/pause`, `/resume`, `/run_now`, `/setup`, `/undo`, `/history`, `/restore`, `/requeue`, `/stats`, `/test`. Free-text LLM changes and `pending_change` confirmation are Branch 5 and not implemented.

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
   - All secrets belong in `.env` (which is in `.gitignore`). Private candidate personas live in the stored profile (git-ignored `data/jobs.db`) and in the legacy `.candidate_profile.md` (also in `.gitignore`).
   - Use `.env.example` with sanitized placeholders for documentation.
5. **BypassSandbox Execution**:
   - Shell commands that modify files, run tests, or execute git commands (`pytest`, `git`, `python`) require `BypassSandbox: true` to avoid sandbox read-only filesystem errors.

---

## 5. Quick CLI & Testing Reference

```bash
# Run full automated test suite (~1,670 tests, no network; the `live` smoke test is skipped)
.venv/bin/pytest -v

# Optional: 2 real calls through the provider chain (needs keys in .env)
.venv/bin/pytest -m live -v

# Model bake-off (writes to git-ignored local/bakeoff/)
.venv/bin/python scripts/bakeoff.py list-free

# Recompute dedup hashes after a normalization change and merge duplicate postings (never while the daemon runs)
.venv/bin/jobspy-coach rehash --dry-run
.venv/bin/jobspy-coach rehash

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

# Revive pre-filtered rows the current rules now pass / re-bucket evaluated rows under the profile thresholds (no LLM call; never while the daemon runs)
.venv/bin/jobspy-coach requeue --dry-run
.venv/bin/jobspy-coach requeue
.venv/bin/jobspy-coach requeue --thresholds --dry-run

# First-time bootstrap (interactive): hidden prompts for the bot token and LLM key(s), each validated with one call and merged into .env (mode 600);
# pairing (adopts an existing TELEGRAM_CHAT_ID, else prints a one-time code to send as `/start <code>`); paused starter profile; offers to start the daemon
.venv/bin/jobspy-coach init
.venv/bin/jobspy-coach init --skip-credentials --no-start --language en   # profile and pairing step only, never starts the daemon

# Forget the paired Telegram account and blank TELEGRAM_CHAT_ID in .env (restart the daemon afterwards, then run `init` to pair again)
.venv/bin/jobspy-coach bot unpair

# Profile: inspect, export (to git-ignored local/), import
.venv/bin/jobspy-coach profile preview
.venv/bin/jobspy-coach profile export
.venv/bin/jobspy-coach profile import local/profile-export.yaml

# Start 24/7 background daemon scheduler
.venv/bin/jobspy-coach run
```
