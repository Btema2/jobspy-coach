# Generic Search Profile and Telegram Management Bot: Design

Status: revision 3, after two independent reviews against the code. Date: 2026-11-18.
Scope: architectural. One design, five sequenced implementation branches (section 14).

All examples use placeholders (`<ROLE>`, `<MARKET>`). This file must never contain candidate data (AGENTS.md section 1).

## 1. Problem

What the user searches for and how vacancies are judged is spread over five places, four of them code:

| Concern | Where it lives today | Editable without code? |
|---|---|---|
| Search queries | `queries.json`: a hand-built list of queries, each with its own term, boards, country and optional Google string | yes, but unvalidated |
| Persona, bonus criteria, output language | `.candidate_profile.md`, read once when the evaluator chain is built | yes, but needs a restart |
| Title exclusions, traps, seniority, rescue words | five regexes in `src/filters/pre_filter.py`; `classify_title` is imported by the scraper and by the evaluator's title safety net (`chain.py`) | no |
| Scoring rubric and the WOW rule | prompt text in `src/evaluator/prompts.py`, `role_is_direct_coaching`, `DIGEST_MAX_SCORE`, `BONUS_POINTS` | no |
| Alert and digest language | hard-coded non-English strings in `src/notifier/templates.py` and `src/cli.py` | no |

Further defects found in the current code:

- A JSON error in `queries.json` is logged as a warning and the app falls back to the generic default queries, so it searches a different market. `run_pipeline_cycle` also does `queries or get_active_queries()`, so an empty list falls back too.
- Of the board rules in AGENTS.md, only "Indeed and Glassdoor need a country" is enforced in code (`COUNTRY_BOUND_BOARDS`, skipped with a one-time warning). The Glassdoor country list and the ZipRecruiter US/UK/Canada rule are prose only. Region-wide queries therefore never reach Indeed.
- The rubric hard-codes the WOW and digest bands (8.5 / 6.5) while the thresholds are configurable, so changing them makes the prompt wrong.
- Whether the bonus applies is inferred by searching the persona text for the word "bonus" (`chain.py`).
- Setup means hand-editing about 35 `.env` variables, a JSON file and a markdown file, with no validation. `scrape --dry-run` makes real network calls.
- `scrape_runs` has per-call rows, but they are not truthful. `scrape_board` swallows exceptions and returns an empty list, and python-jobspy reports blocks without raising: it fills `df.attrs["errors"]` and `df.attrs["warnings"]` (the scraper only logs these once per process). The daemon then records `completed` with zero rows, so a blocked board looks like a dead query. Nothing attributes evaluated jobs back to a query.
- Remoteness is a keyword guess. python-jobspy sets a concrete `is_remote` per row from keywords in title, location or text (for example LinkedIn looks for a remote word in title and location), so `False` is not evidence that a job is on-site, and a remote job found through the server-side remote filter can be flagged `False`.
- The only schema-evolution tool is `_ensure_column` (add-only); there is no migration framework. `jobs` has no query attribution column.
- The evaluation result keeps only the final score. `role_is_direct_coaching`, the raw model score and the bonus flag are not stored, so a score cannot be re-bucketed later.

## 2. Goals and non-goals

Goals:

1. Anyone, in any profession, can set up and change what they search for without editing code.
2. Changes are made from Telegram, mostly in plain language, and take effect on the next cycle without a restart.
3. Bad configuration fails loudly and the last good state keeps working.
4. The user can see which queries yield relevant jobs, and can tell a dead query from a blocked one.
5. An existing setup keeps working unchanged after an automatic import.

Non-goals for this design: pagination, per-query schedules, skipping known jobs before the LinkedIn description fetch, a web UI, an autonomous agent loop, answering "why was this dismissed" questions, LLM-translated alert templates, a weekly health message, PDF CV upload, unpair from Telegram itself.

## 3. Architecture

```
src/
  profile/
    store.py        only code touching profile tables
    ops.py          closed operation set; validate, apply, revert; render diffs
    compiler.py     pure: profile snapshot -> CompiledProfile
    importer.py     legacy files + legacy seed -> profile rows; export to YAML
    legacy_seed.py  tracked: today's regex groups and rubric definitions as seed data
  scraper/
    coordinator.py  per-call lock and shared LinkedIn pacing for every scrape
  bot/
    client.py       own HTTP client: getUpdates long-poll, sendMessage, callbacks, 429 handling
    poller.py       daemon thread, stop-event aware, offset persisted
    auth.py         pairing and sender check
    commands.py     deterministic commands and the /setup wizard
    conversation.py per-chat state, debounce, pending proposal
    translator.py   free text -> typed diff (branch 5)
    chat_llm.py     dedicated JSON-completion path (branch 5)
  i18n.py           static string tables
```

The daemon recompiles the profile at the start of each cycle and uses that immutable `CompiledProfile` for the whole cycle: scraper, pre-filter, evaluator prompt and title safety net all receive it by argument. This replaces today's direct imports (`get_active_queries`, `classify_title` over module-level regexes, `load_candidate_persona`, the prompt frozen in the chain constructor). The chain takes the system prompt per call from the compiled profile instead of caching it.

The compiler reads all profile tables inside one read transaction (WAL gives a consistent snapshot), so a multi-table edit committed by the bot thread is seen entirely or not at all.

## 4. Data model

Introduced with a real migration framework: a `schema_version` table and an ordered list of migrations, used for everything new instead of ad-hoc `_ensure_column`. Migrations are add-only (no renames, no drops), so an older process still running against the file keeps working.

Profile tables:

- `profile` (single row): `version` (integer, bumped on every applied change), persona text (up to 8,000 characters), core-role definition, adjacent-role definition, dealbreakers, bonus criteria, bonus points, output language, target level, WOW and digest thresholds, `max_calls_per_cycle`, `paused`.
- `roles`: name, enabled. `role_terms`: role, term, optional language tag (untagged = all markets).
- `markets`: name, location text, `countries` (list), remote flag, `remote_only`, languages, enabled.
- `pinned_queries`: term, location, country, boards, is_remote, optional Google string (imported 1:1, section 5).
- `rules`: kind, term, match mode (`exact`, `prefix`, or `pattern`), reason label, source (`user`, `llm`, `import`, `seed`). `pattern` rules (a regex) can only come from the tracked legacy seed; users and the LLM can create only `exact` and `prefix`.
- `profile_versions`: snapshot of the profile tables per applied change, stamped with a snapshot schema version, last 10 kept.
- `pending_change`: single slot per chat: proposed diff, `base_version`, created at. Raw pasted text is not stored in it.
- `conversation`: per-chat state (`idle`, `awaiting_clarification`, `awaiting_confirm`) and short context.
- `bot_state`: update offset, paired user and chat id, pairing code hash and attempt counter.
- `llm_consent`: acknowledgement timestamp (section 10).

Scrape and job additions (add-only):

- `scrape_runs` gets `rows_returned`, `prefiltered`, `outcome` and `query_key`.
- `job_hits(job_id, query_key, first_seen, last_seen)` records every query that returned a job, including already-known jobs. It is keyed by `jobs.id`, and `rehash_all` re-points hits of deleted duplicates to the surviving row.
- `jobs.source_query` (new column) records the first finder only. `jobs.remote_status` (new column, `verified` or `unverified`).
- `jobs.eval_raw_score`, `jobs.eval_is_core_role`, `jobs.eval_bonus_flag` and `jobs.eval_hard_reject` are filled for new evaluations. Re-bucketing needs the hard-reject flag because a hard reject caps the stored score regardless of the raw score. Older rows have NULL and are excluded from re-bucketing, with a count reported to the user.

Existing columns, including `eval_summary_uk`, are not renamed.

Retention: raw pasted text lives only in memory and in `conversation.context` for at most 24 hours. It is purged when a proposal is applied, cancelled, superseded or expires.

## 5. Compiler and import

`compile(snapshot) -> CompiledProfile(queries, rules, system_prompt, estimate, warnings)`. Pure and network-free.

Queries are `pinned_queries` plus the expansion of roles x markets x boards, deduplicated by key.

Expansion rules:

- A term is sent only to markets whose languages include its tag (untagged terms go to all markets).
- A market has a list of `countries`. Indeed and Glassdoor get one call per country; a market with no country skips them, and `/preview` states why. Glassdoor and ZipRecruiter country support is a code table in the compiler, not prose.
- Board specifics stay in the scraper: Indeed quote stripping and the 168 h window, the Google search-term derivation, LinkedIn pacing.

Cost control: the compiler counts calls and estimates cycle duration (LinkedIn cooldown x LinkedIn calls plus a per-call allowance). A profile whose call count exceeds `max_calls_per_cycle` is rejected by `ops` and never applied. The estimate versus `SCRAPE_INTERVAL_HOURS` is a runtime warning shown in `/preview` and diffs, not a validation rule, so an `.env` change can never invalidate a stored profile.

Remote handling. The row flag is positive evidence only: `True` means `remote_status = verified`; `False` or missing means `unverified`. Rows are never dropped because of a `False` flag. In branch 1 the scraper stops substituting the query flag for a missing value; this affects only boards that return no value. For a `remote_only` market the query always runs with the board's remote filter, and unverified rows are kept; the scorer receives `Remote: unverified (found with the remote filter)` and a remote-only requirement among the dealbreakers, so the judgement comes from the description. `remote_status` is stored in branch 1 and used by the scorer in branch 3.

Automatic import and legacy seed, so an existing setup keeps working:

- When the profile tables are empty and legacy files exist (`queries.json` and/or `.candidate_profile.md`), the first start of any command imports them automatically and idempotently. The original files are left untouched.
- Every `queries.json` entry becomes a pinned query 1:1.
- The importer seeds rules from `legacy_seed.py`, which encodes today's five regex groups with their reason labels (`EXCLUDED_TITLE_REGEX`, `EXCLUDED_TITLE_TRAP`, `EXCLUDED_SENIORITY`) as `pattern` rules, so behaviour and reason strings are preserved, including terms like `cross-?fit` and `l&d`.
- It seeds the core-role definition, the dealbreakers and the non-core cap from today's hard-coded rubric text, so corporate-trainer and L&D-style roles stay capped below WOW.
- If the persona contains bonus criteria it copies them into `bonus_criteria`, preserving today's bonus behaviour.
- A persona over 8,000 characters makes the import stop with a clear message; nothing is truncated silently.
- With neither a profile nor legacy files, `run` prints "no profile, run `jobspy-coach init`" and exits non-zero. The generic default queries are removed. In the daemon, `queries or get_active_queries()` is replaced by the compiled profile; an empty compiled list skips scraping and alerts once, never falling back.

Failure handling: `ops` validates every change, so an invalid profile cannot normally be stored. If compilation still fails (import or migration bug), the daemon skips scraping for that cycle, alerts once per distinct error, and keeps evaluating the backlog and sending digests with the last successfully compiled profile (rebuilt from the last good `profile_versions` snapshot).

Pause and first run: a profile created by `init` or `/setup` starts `paused`. Pause stops scraping only; evaluating the backlog and the digest continue. `ops` refuses `/resume` unless the compiled profile has at least one enabled query. `/resume` triggers an immediate cycle; `/run_now` triggers one on demand (it refuses while a cycle is running).

## 6. Scraper outcome, attribution and coordination

`scrape_board` returns a structured `ScrapeOutcome(postings, rows, status, error)` with status `ok`, `empty`, `failed` or `skipped(reason)`. `failed` is set both for an exception after retries and for a board entry in `df.attrs["errors"]` (a block or challenge page), `empty` only when the board answered without error and returned nothing. The daemon records it truthfully in `scrape_runs`. For every posting returned, known or new, a `job_hits` row is upserted for the call's query key.

A scrape coordinator wraps each individual scrape call in a lock and owns the shared LinkedIn pacing timestamp. A cycle takes the lock per call, so an on-demand search can interleave between calls instead of waiting for a whole cycle, and the 12 s LinkedIn cooldown still holds across all callers.

`/test` and onboarding searches run on a worker thread, never on the poller thread. They count against their own limits: at most one `/test` per 5 minutes, at most 5 results, non-LinkedIn boards unless LinkedIn is named explicitly. They do not count against `max_calls_per_cycle` but go through the same coordinator and pacing.

## 7. Rules (pre-filter)

The five existing regexes become four rule kinds with unchanged semantics:

- `block`: always dropped, never rescued. Covers `EXCLUSION_TITLE_REGEX` and `HARD_TRAP_REGEX`, which keep their separate reason labels.
- `soft`: dropped unless a `rescue` term is present.
- `seniority`: dropped unless rescued. The user picks a target level; a built-in English table supplies terms, and extra terms are added per language of the markets.
- `rescue`: saves a `soft` or `seniority` hit.

`pre_filter_job(title, description, rules=None)` and `classify_title(title, rules=None)` keep their signatures; with `rules=None` they use the legacy seed ruleset, so existing callers and tests keep working unchanged, including the reason prefixes asserted today. The compiled profile passes its own rules.

User-created terms: stored as literals and `re.escape`d. Match mode `exact` (whole term) or `prefix` (term plus `\w*`). Minimum length 3, maximum 60, at most 500 rules. The compiler uses lookaround boundaries `(?<!\w)...(?!\w)` for every term because terms such as `l&d` do not work with `\b`; ported tests must show current behaviour is unchanged, including the singular-only rule that depends on exact matching. Stub-description checks (150 characters / 30 words) are unchanged. No domain terms ship for new profiles; onboarding drafts starter rules.

Backtest: any diff that adds or changes a rule shows a dry run against stored jobs (count and sample titles). With fewer than 200 stored jobs it says the backtest is not meaningful instead of showing a number. `/dropped` lists the last 20 pre-filtered titles with the rule that hit.

History: pre-filtered and score-dismissed rows keep their dedup hash, so a rule or threshold change never revives them by itself.

- `/requeue` re-runs the current rules over pre-filtered rows and moves those that now pass back to `unprocessed`.
- `/requeue thresholds` recomputes the final score of already evaluated rows from `eval_raw_score`, `eval_is_core_role` and `eval_bonus_flag` under the current thresholds, bonus and cap, with no LLM call. Rows without those fields are skipped and counted. Re-bucketed rows never send instant alerts; any that now qualify join the next digest.

## 8. Rubric template

Score bands keep their meaning (hard reject up to 2.0, marginal, digest, WOW); band edges are interpolated from the profile thresholds. Slots: core-role definition, adjacent roles, dealbreakers, bonus criteria and points, output language.

Invariants enforced by `ops`: `3.0 < digest <= wow - 0.1` and `wow <= 10`. The non-core cap is `wow - 0.1`, so it can never fall below the digest threshold; the marginal band is 3.0 to `digest - 0.1`.

`role_is_direct_coaching` becomes `is_core_role` (Python field and JSON key; the old key is accepted when parsing). Only a core role can reach WOW; other roles are capped at `wow - 0.1`. A new profile with no core-role definition has the cap off, and onboarding asks for the definition explicitly. The model's summary key becomes `summary` (accepting `summary_uk`); the `eval_summary_uk` column keeps its name.

The profile is the source of truth once it exists. `.env` threshold values are only seeds at import time and are documented as such. The bonus applies when `bonus_criteria` is non-empty; the text search for "bonus" is removed.

Persona and definitions are length-capped, stripped of control characters and inserted into the prompt inside a delimited data block that the prompt states is data. The job prompt's remote line becomes `Remote: yes / unverified` as described in section 5.

## 9. Telegram bot

Bootstrap, `jobspy-coach init`:

1. Reads the bot token and the LLM key with hidden input (never from arguments), merges them into the existing `.env` without overwriting other keys, and sets mode 600.
2. Validates each credential with one call.
3. Pairing. If `TELEGRAM_CHAT_ID` is already set in `.env` (an existing setup), it is adopted as the paired chat, because the local `.env` is the trust anchor. Otherwise `init` prints a one-time 8-character pairing code (about 40 bits). The bot accepts `/start <code>` only from a private chat, stores that user id and chat id, writes `TELEGRAM_CHAT_ID` into `.env` so the notifier and the bot share one value, and from then on ignores every other sender, group or chat type. The code is single-use, expires after 15 minutes, and after 5 wrong attempts pairing locks until the process restarts and a warning is logged. `jobspy-coach bot unpair` clears pairing for a lost account.
4. Creates a starter profile (paused) unless one was imported, and starts the daemon.

Auth is checked on every update, including callback queries (`from.id` and chat type). Rejected updates are dropped and logged without content.

Process model: the poller is a daemon thread started before the startup scrape cycle, so the bot works during it. It has its own HTTP client with a long-poll timeout, honours 429 `retry_after` without blocking other work, and stops through the same stop-event wiring as the daemon so shutdown does not wait for a 30 s poll. It uses its own SQLite connection (WAL). AGENTS.md section 3.E is amended: the caller-thread invariant applies to evaluation workers; the bot thread is a second, explicitly allowed DB and Telegram thread that never runs evaluation.

Delivery semantics: the update offset is persisted before a mutating command executes (at-most-once). A crash can lose a command, and the user resends; it can never replay a non-idempotent `/undo`. Messages older than 15 minutes are ignored for mutating commands. Callback buttons are bound to a `pending_change` id and its `base_version` and are validated by those, not by message date.

Commands (all through `ops`; each applies immediately, bumps `profile.version`, snapshots, and shows a code-rendered one-line diff; the confirmation step exists only for LLM-authored changes because the author is a model, not because of size):

| Group | Commands |
|---|---|
| Read | `/help`, `/profile`, `/roles`, `/markets`, `/exclusions`, `/preview`, `/dropped` |
| Change | `/add_role <name> [= synonym, synonym]`, `/remove_role`, `/add_market <country> [remote] [remote_only]`, `/exclude <term> [prefix]`, `/threshold wow\|digest <n>`, `/pause`, `/resume`, `/run_now` |
| Setup | `/setup` (stepwise wizard: role, synonyms, market, remote preference, confirm; uses conversation state with inline buttons) |
| Safety | `/undo`, `/history`, `/restore <n>`, `/requeue` |
| Insight | `/stats`, `/test <role>` |

Country names are resolved from a built-in table; an ambiguous name gets a reply asking which one. `/undo` restores the snapshot taken before the last applied change as a new version, so repeating it toggles between two states; `/history` lists versions and `/restore <n>` returns to any of the last 10. Snapshots carry a schema-version stamp and are upgraded on restore. Menus are registered with `setMyCommands`. All outputs go through chunking at 3,800 characters and HTML-escape user-supplied terms; long lists are paginated.

`/stats` shows a 7-day per-role table from `scrape_runs` and `job_hits`: calls, new jobs and relevant jobs, with failures shown separately from empty results so a blocked board is never blamed on the query. "Relevant" means evaluated with a final score at or above the current digest threshold. `/test` reports rows, new jobs, the pre-filtered share and example titles; it does not measure relevance and says so.

Before branch 5, `/setup` is the plain-language-free onboarding path: a deterministic wizard that produces the same operations the commands do. Branch 5 replaces its questions with a free-text description.

## 10. LLM translator and conversation (branch 5)

Free text is the main interface. Any non-command message goes to the translator, which makes one call and must return a diff validating against a Pydantic schema over the closed operation set: add, remove or modify a role, term, market or rule; change a threshold; change persona or definitions; enable or disable an item; pause or resume. It cannot touch the bot token, the paired chat, providers, pacing settings or anything outside the profile. Job text never reaches the translator; only the user's own messages and pasted material do.

The diff the user sees is rendered by code from the structured diff, never LLM prose. A persona or definition change is shown in full (chunked, not truncated), with its own button label ("Apply persona change"). Confirmation is by inline Confirm and Cancel buttons; typed `/ok` and `/cancel` also work.

Conversation state: one pending proposal per chat; a new proposal supersedes the old one and the user is told. A `pending_change` records `base_version`; applying is rejected if `profile.version` has moved ("the profile changed, send your request again"). A pending proposal expires after 24 hours with a notice. Bare replies such as "ok", "yes", "no" map to confirm or cancel when a proposal is pending and otherwise get a help hint without an LLM call. A clarifying question from the translator puts the chat in `awaiting_clarification`, and the next message is sent together with the earlier context. Messages arriving within 3 seconds are merged, up to 3 parts and 12,000 characters; anything longer gets "too long, send it as a .txt file". `.txt` and `.md` uploads up to 20 KB are accepted.

Dedicated chat LLM path. It cannot reuse the evaluator chain, which is hard-wired to the scoring prompt and the evaluation result parser. Required refactors (all in branch 5):

- `SlidingWindowRateLimiter.acquire` gains a `priority` flag and a per-call stop event. Priority acquirers register in a waiter count; batch acquirers yield while one is waiting. The limiter and its `pause_for` backoff stay shared.
- Providers gain a generic `complete_json(messages, options)` where options carry `timeout`, `max_attempts` and the stop event, instead of the hard-wired three attempts and the shared batch event.
- The chat path passes a stop event tied to process shutdown only, so it is unaffected by the evaluator batch stop event that stays set between a batch abort and the next batch start.
- Chat calls neither consult nor update the batch circuit breakers; they try NIM, then OpenRouter, and report failure to the user.
- Timeout: documented NIM latency is a mean of about 46 s with provider stalls (`docs/model-selection.md`), so chat uses 120 s per provider with one attempt. The bot posts "working" immediately, a progress note at 20 s, and runs the call off the poller thread.
- Abuse and cost: at most 30 chat LLM calls per hour.

Privacy and injection: before the first LLM use that involves personal text, the bot explains that the text is sent to the configured providers (NIM and, on fallback, OpenRouter free models, whose logging or training policies may differ), recommends leaving out names, addresses and phone numbers, and requires an explicit acknowledgement recorded in `llm_consent`. Obvious e-mail addresses and phone numbers are masked locally before sending (best effort). Prompts, responses and user messages are never logged; only lengths and ids. A pasted CV can carry injected instructions; the schema, the closed operation set, the code-rendered diff, the full-text persona diff and the confirmation step are the defence.

Export and import: `profile export` writes to `local/profile-export.yaml` by default and refuses paths outside git-ignored locations unless `--force`; `profile-export*.yaml` is added to `.gitignore`. Import validates through `ops` like any other change.

## 11. Localisation

Alert, digest and `test-telegram` chrome strings move to `i18n.py` as static tables: English (default) and a second table holding today's hard-coded strings, so existing output is preserved. The profile's output language selects a table, falling back to English. The model-written summary already follows the profile's output language through the prompt. Generated templates are out of scope; a broken tag in a generated template would make Telegram return 400 and silently stop alerts, so tests assert that every shipped table has the required placeholders and valid tags.

## 12. Error handling summary

- Invalid profile at compile time: skip scraping, alert once per distinct error, keep evaluating and digesting with the last good profile.
- LLM timeout, bad JSON or schema failure: reply "couldn't understand that, try a command"; nothing is applied.
- No LLM available: the bot says so; commands and `/setup` keep working.
- Edits are transactional with the `version` check; a cycle uses the snapshot taken at its start.
- Failed or blocked scrape calls are recorded as such and never counted as empty results.

## 13. Testing

All tests run without network or the real `.env` (`Settings(_env_file=None)`, `httpx.MockTransport`).

- Compiler: compiled queries for an imported `queries.json` equal today's queries; compiled-prompt snapshots; call-count and duration estimates; country expansion; language tags; read-transaction consistency.
- Import: auto-import is idempotent; seed reproduces today's pre-filter reasons and the non-core cap; bonus preserved; over-long persona stops with a message; no profile and no legacy files exits non-zero; empty compiled list never falls back.
- Rules: existing `test_pre_filter.py` cases pass unchanged through the `rules=None` path; term validation (length, escaping, match modes, `pattern` only from seed); backtest and `/requeue`; `/requeue thresholds` with and without stored raw fields.
- Ops: apply, revert, undo of undo, restore, version conflict, threshold invariants, call-cap rejection, resume refused without queries, snapshot upgrade.
- Scraper: `ScrapeOutcome` for exception, `attrs["errors"]` block, empty and skipped calls; `remote_status` from a positive-only flag; `job_hits` for known jobs and after `rehash_all`; per-call lock interleaving between a cycle and `/test`; `/test` limits.
- Bot: pairing (wrong code, lockout, group chat, expired code, adopted `TELEGRAM_CHAT_ID`), auth on callbacks, offset persisted before execution, stale-update drop, callbacks validated by pending id and version, multi-part merge and the length limit, bare "ok" with and without a pending proposal, supersede, expiry notice, chunking and escaping, `/setup` wizard.
- Chat LLM: priority acquire against saturated workers, independence from the batch stop event, per-call timeout and attempts, hourly cap, redaction of e-mail and phone, no prompt text in logs, raw text purged after apply or expiry.
- i18n: all shipped tables have identical keys and valid tags.
- The live smoke test and `eval_cases.json` remain a live, optional check of scoring behaviour (`pytest -m live`, bake-off); they are not an offline regression gate.

## 14. Build order

1. Foundations: migration framework, `ScrapeOutcome` including `attrs["errors"]`, `scrape_runs` extension, `job_hits`, `jobs.source_query` and `remote_status`, positive-only remote flag, new eval columns, scrape coordinator, per-call (not frozen) system prompt in the chain. No change to what the scorer sees; scrape records become accurate.
2. Profile core: store, compiler, importer with legacy seed and automatic import, last-good fallback, removal of the default query catalog and the `queries or get_active_queries()` fallback, `profile preview` and `export`, `init` creating a starter profile without the bot. Existing installs keep running through the automatic import.
3. Rules and rubric: rule kinds with the `rules=None` legacy path, rubric template, `is_core_role`, thresholds and bonus as profile data, remote line for the scorer, backtest, `/requeue` logic.
4. Bot core: `init` pairing and credential handling, auth, poller and client, deterministic commands and `/setup`, versions and undo, `/stats`, `/test`, `/dropped`, i18n tables, AGENTS.md amendment.
5. LLM layer: limiter priority and provider `complete_json` refactors, `chat_llm.py`, translator, conversation state, LLM onboarding, consent, redaction and abuse limits.

Each branch is mergeable and leaves the app working for existing and new users.

## 15. Risks and open items

- The rubric was tuned on a coaching fixture set that only runs live; the generic template is guarded offline by prompt snapshots of an imported profile, and by the optional live run.
- Reasoning-model latency is accepted only because the chat path is asynchronous and bounded to 120 s per provider.
- The exact seniority term table and the number of reserved rate-limit slots for chat are decided in the implementation plans of branches 3 and 5.
