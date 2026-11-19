# Interview onboarding, CV upload, reliable chat LLM and run report — design

Status: revision 2 (2026-11-19), after an independent review of revision 1 (5 blockers, all resolved below).
Branch: `feature/onboarding-interview`, stacked on `feature/llm-layer`.
Builds on `2026-11-18-generic-profile-and-bot-design.md` (Branches 1-5, all implemented). `AGENTS.md` sections 2,
3.C.4, 3.C.8, 3.D, 3.E and 3.G stay binding; this change updates them (section 10).

## 1. Why

Feedback from a first hands-on test of the bot:

1. The `/setup` intro is weak: one open "describe yourself" prompt, one proposal, no questions, no follow-ups.
2. A CV should be droppable to fill the profile.
3. NIM seemed to refuse requests **during setup**. Evidence: chat gets one attempt of up to 120 s per provider
   (`CHAT_TIMEOUT_SECONDS`, `CHAT_MAX_ATTEMPTS = 1`); the single onboarding call (a full profile draft from a 550B
   reasoning model) is the slowest call the bot makes; a timeout falls through to free OpenRouter models, and the user
   then sees only `Couldn't understand that, try a command (/help).` with no reason and no retry.
4. After `/run_now` only WOW alerts arrive. This is by design (the digest runs only from the 09:00 cron) but gives
   no feedback: the user cannot tell what the run did, what failed or what is waiting.

Nemotron has **no image input**, so a CV can only be used through its text layer.

## 2. Scope

In: the interview, CV (PDF) upload, chat LLM reliability and failure reasons, the `/run_now` / `/resume` report, the
`/digest` command, docs and tests.

Out: fixing the Google bot-challenge and LinkedIn 429 scrape failures (only reported), OCR, image input, `.docx`,
streaming, any change to evaluation scoring, a second dependency such as `cryptography`.

## 3. Interview (hybrid)

`/setup` with a configured LLM starts an interview. `/setup guided` and "no LLM configured" keep the deterministic
wizard unchanged. Starting `/setup` discards a pending proposal with `PROPOSAL_DISCARDED_REPLY`, as today.

**Intro message** (replaces `ONBOARDING_PROMPT`): what will happen (6 to 8 questions, a few minutes), that a CV PDF or
text can be sent at any point and answers several questions at once, that nothing changes until the user confirms, and
the buttons `Draft now`, `Step-by-step questions instead` and `Cancel`.

**Checklist (owned by code, in this order)**: `roles` (what jobs, with synonyms), `markets` (countries and/or remote),
`languages` (the candidate's working languages: they go into the persona text and decide the languages of the synonyms; they
do **not** map to `set_output_language`, which is the alert language), `level` (seniority), `avoid` (dealbreakers),
`bonus` (credentials, languages or background a posting could value). Each slot has fixed question text and a status
`empty | partial | filled | skipped`.

**Turn loop** (one request at a time on the `bot-conversation` worker):

1. The user answers. The answer is masked (`mask_personal_data` plus the URL/handle masking of section 4) **before it is
   written anywhere** and stored **verbatim as the slot's facts** (code, no model needed). `no`, `none`, `skip`, `-` (and
   other bare words) are answers to the current slot, never routed to proposal confirmation (see "Bare words" below): code
   marks the slot `skipped` without any LLM call. Answers of fewer than 4 words that are not a skip are stored and the slot
   is `filled` without an LLM call.
2. For longer answers one **short turn call** (`kind: interview_turn`) runs. Its input is only the slot question, the
   answer and the list of slots (the stored profile is **not** sent). Its output, validated by a Pydantic schema, is
   tiny: `{covers: [slot names this answer also answers], followup: string|null}` (follow-up <= 300 characters, used
   only when the answer is vague or contradictory). It does not extract structured data; the draft call does that from the
   raw facts. A failed or invalid turn call is treated as "no follow-up, covers nothing", with a note that the step could
   not be checked, never as a lost interview.
3. Code picks the next step: the follow-up (at most one per slot), else the next slot that is `empty`, else done. At most
   8 questions are asked per interview (follow-ups included), then it drafts. Slots covered by earlier answers or by a CV are
   skipped.
4. When no slot is `empty` (or after `Draft now`) one **draft call** turns all facts plus the stored profile into the existing
   onboarding operation set and the existing confirm-to-apply proposal (`mode="onboarding"`, same buttons, same `Apply persona
   change` handling). The draft schema allows `proposal` and `none` only (no `clarify`). Nothing is applied before
   confirmation.
5. A draft problem (provider failure of any `ChatFailure` class, `TranslationProblem`, `ProfileValidationError` such as an
   unknown country) **never ends the interview**: the bot shows the reason and `Retry` / `Draft now` /
   `Step-by-step questions instead`, and for a validation problem names the slot it concerns so the user can answer it again.

The model never applies anything and never decides order or stop conditions; those are code. Facts are untrusted data: they
enter prompts only inside a delimited block like `<user_message>` (`_clean` with an explicit limit of 12,000 characters for
the facts block), tags neutralised, and a schema failure drops the update.

**Bare words.** In `awaiting_interview` every bare word (`ok yes no nope sure ...`) is an answer to the current slot, handled
by the rule above; the `on_text` bare-word router must not run for this state (only `awaiting_clarification` is exempt today,
`awaiting_interview` joins it).

**Buttons.** `iv:<nonce>:draft|cancel|retry|guided` (nonce = 8 hex stored in `context`, so a stale button answers `This
choice is no longer active; send your answer again.`; at most 20 bytes). `Retry` re-runs the failed step from the in-memory
copy of the answer; after a daemon restart that copy is gone and the button says to send the answer again (the interview
state itself persists).

**State.** New conversation state `awaiting_interview` added to `STATES` in `conversation_store.py` (no migration: the column
has no CHECK). `conversation.context` JSON: `{nonce, created_at, slots, asked, facts}`; **`created_at` is set once at the
first stored answer and the 24 h purge counts from it** (the existing first-storage rule, today enforced only for
`awaiting_clarification`, extends to `awaiting_interview`; `purge_contexts` and `_purge_old_contexts` cover both states and
every turn rewrite keeps `created_at`). `facts` is dropped when the draft proposal is stored (`_store_proposal` sets context
None), on apply, cancel, supersede and expiry, with `secure_delete` and the WAL checkpoint as today. Failures keep the
interview; only `/cancel`, expiry and apply end it. A new `/setup` restarts it.

## 4. CV upload (PDF)

Accepted wherever `.txt` / `.md` uploads are accepted today (`_add_upload` gets a PDF branch: suffix `.pdf` or mime
`application/pdf`, size checked against `MAX_CV_BYTES` from Telegram's `file_size` before download and again by the
size-limited `download_file`). During the interview a CV answers several slots at once. In an idle chat it starts the
**update interview** (section 8): it first discards any pending proposal with `PROPOSAL_DISCARDED_REPLY`.

- **Limits**: <= 5 MB (`MAX_CV_BYTES`), <= 10 pages, <= 20,000 extracted characters, `%PDF-` within the first 1,024 bytes.
  Encrypted PDFs: try `decrypt("")` (owner-password-only files open without a user password); if that fails or needs a
  crypto library that is not installed, refuse with `This PDF is password-protected. Send an unprotected copy or paste the text.`
  **No text layer** means fewer than 200 non-whitespace characters overall: refused with `This PDF has no readable text (it
  looks scanned). Paste the text or send a .txt file.`
- **Extraction runs in a child process**, never in the worker thread (a thread cannot be stopped, and a hostile file could
  keep the single worker busy and starve the GIL): `python -I -m src.bot.pdf_text` receives the bytes on stdin and prints
  `{pages, text}` JSON on stdout; the parent kills it after 15 s wall clock; the child sets `RLIMIT_AS` (512 MB) and
  `RLIMIT_CPU` (20 s) with `resource.setrlimit` (Linux), checks the page count before reading any page, and stops reading
  once 20,000 characters exist. Any failure (timeout, kill, non-zero exit, bad JSON) is logged by type name only and answered
  with `This PDF could not be read. Paste the text or send a .txt file.` The child never receives the file name and the parent
  never logs it (file names often contain the candidate's name).
- **Dependency**: `pypdf` is not installed or declared today. The plan adds it to `pyproject.toml` with a floor at a release
  that contains the decompression-limit fixes (`pypdf>=6.1`, verified at install time) and installs it into `.venv`.
- **Privacy pipeline** (before anything leaves the machine): `mask_personal_data` (e-mail, phone) plus masking of URLs and
  social handles; the contact header block (everything before the first section-like heading, or the first 6 non-empty
  lines when none is found) is removed. Names and addresses elsewhere cannot be recognised; the consent notice says so.
- **Order: extract and preview first, consent gates `Send`.** Extraction and the preview send nothing, so they need no consent.
  The bot shows `Read <N> pages, <M> characters. First lines: <short masked preview>` with buttons `cv:<nonce>:send|cancel`
  (<= 19 bytes). On `Send`, if there is no consent for `CONSENT_VERSION` the consent notice is shown first and `Send`
  completes after `yes`. The CV text is held in a **dedicated in-memory slot keyed by the `cv:` nonce** (separate from the
  single `_held` paste slot), expires after 24 h and is dropped on `Cancel`, `/cancel`, `reset_memory`, a new pairing and
  expiry. A new CV replaces the old one.
- **Consent v2**: `CONSENT_VERSION` 1 -> 2. The notice names CVs, the interview facts and that these facts (masked answers,
  not the CV text) are stored in the local database for up to 24 hours; recommends leaving out names and addresses; the rest
  as today.
- **Use**: after `Send`, the masked CV text goes through one `extract` call (`kind: extract`, input = the CV text only, no
  stored profile) that returns `{covers: [slots], facts: {slot: short text}}`; the facts are stored like answers (verbatim,
  masked), the covered slots become `filled`, and the interview continues with the gaps. The CV text itself is dropped after
  the call; only the extracted facts (short texts, bounded) are stored. PDF bytes are never stored. The upload caption is
  treated as an extra answer.

## 5. Reliable chat LLM

- **Retry in `ChatLLM`, not in the provider.** `CompletionOptions.max_attempts` stays 1 and `ChatLLM` calls `complete_json`
  again itself, so **every** HTTP attempt takes a slot from the hourly cap (today one slot per provider precedes the
  provider's internal retries, which would make retries invisible to the cap). Rules: NIM only, at most one extra attempt,
  only after `ProviderTimeout`, a 5xx, or a 429; never after `InvalidResponse`, `ResponseTruncated`, an HTTP-200 error
  envelope or a validation failure (those go straight to the next provider, as today). OpenRouter gets **no** retry (its
  free budget is shared with evaluation).
- **Bounded waits.** A 429 retry waits `min(retry-after or limiter pause, remaining deadline)` only when that is <=
  `CHAT_LIMITER_MAX_WAIT_SECONDS` (30 s); otherwise the attempt is abandoned and the next provider is tried. The provider's own
  sleep/backoff is not used (attempts stay at 1).
- **Overall deadline per request**: 150 s for an interview turn or `extract`, 300 s for the draft call (`CHAT_DRAFT_TIMEOUT_SECONDS`
  per attempt 180 s). Every attempt's timeout is clipped to the remaining deadline; when it is spent the request ends with
  `timeout`. httpx `read` timeouts are per-read inactivity timeouts, so the deadline is the real wall-clock bound. The
  progress note reads `Still working; this can take a few minutes.` (replacing the "up to 2 minutes" text).
- **Caps**: `CHAT_HOURLY_CAP` 30 -> 60 attempts per sliding hour (an interview with a CV is about 8 to 12 calls plus
  retries). New `CHAT_OPENROUTER_DAILY_SHARE = 15`: chat may spend at most 15 of the shared `OPENROUTER_DAILY_BUDGET` (45) per
  UTC day, counted in `ChatLLM`, so a chatty setup cannot exhaust the evaluation fallback. All `CHAT_*` values stay module
  constants (they were never settings).
- **Failure classes** (`ChatFailure` enum carried by the typed exceptions; replies are code-written text, never LLM prose).
  Mapping and precedence when providers fail differently (highest wins): `capped` (hourly cap or OpenRouter share spent) >
  `bad_answer` (`InvalidResponse`, `ResponseTruncated`, schema failure) > `timeout` (`ProviderTimeout`, deadline spent) >
  `rate_limited` (429, `RateLimitWaitExceeded`) > `unavailable` (no key, auth, quota, 5xx, nothing reachable). Texts: `The model
  timed out`, `The model is rate-limited`, `No model is reachable`, `The model's answer could not be used`, `Too many model
  requests this hour`; each with the retry time when known (a 429 from NIM has none: say `try again in a minute`) and what
  happens next.
- **Free-text change requests** (not the interview) get the same reason text and a `Retry` button; the original text is a
  second in-memory holder (24 h), dropped by `/cancel`, `reset_memory` and a new pairing like the CV slot.
- **Logging**: provider name, failure class, HTTP status and `request_id` only; never prompts, answers, URLs, file names or
  `str(exc)`.

## 6. Run report and `/digest`

**Report.** `evaluate_unprocessed` returns a small `EvalSummary` (evaluated, wow_sent, still unprocessed, `stopped_reason`
from the existing `batch_stopped` state) in addition to today's tuple, and a new `DaemonService.run_pipeline_cycle_report()`
builds a frozen `CycleReport`. Sources per field:

| Field | Source |
|---|---|
| `new_to_evaluate`, `prefiltered` | `_scrape_queries` split of `total_new_saved` (it counts pre-filtered rows today) |
| `calls` (ok, empty, failed) and `problems` | per-(query, board) `ScrapeOutcome` loop in the daemon; `problems` = distinct `(board, reason)` with a count; reasons come from `outcome.error`, with a classifier for LinkedIn 429 (`blocked (429)`, also from `skipped_empty`) and Google (`no data returned`) |
| `skipped_boards` | `begin_cycle` / `end_cycle` of the scraper gains a returned count of boards skipped for a missing country (today a private `_cycle.skipped`) |
| `evaluated`, `unscored`, `stopped_reason`, `wow_sent` | `EvalSummary`; `unscored` splits "beyond `eval_batch_limit`" from "provider unavailable" |
| `digest_waiting` | `get_pending_digest_jobs` count at or above the digest threshold (all of them, not capped by `digest_max_items`) |

A `ConfigurationError` or any exception in the cycle is caught by the report wrapper: the user still gets a report with
`The cycle stopped: <class>` (type name or the no-LLM-key hint), so a failed run is never silent.

**Trigger and delivery.** `trigger_cycle(notify: bool = False)`; `BotContext.trigger_cycle` becomes `Callable[..., str]`
and `/run_now`, `/resume` and the chat-proposal resume (`ConversationService._trigger`) all pass `notify=True`. The daemon
keeps a `_notify_requested` flag set by `trigger_cycle(notify=True)` and read-and-cleared by whichever cycle runs next
(including a scheduled one that took the lock first, so a skipped `run_now` job never loses its report). The report is sent
**after** `_cycle_lock` is released, from the scheduler thread, through `notifier.send_message` as one message of at most
~12 lines (plain English bot text, not `src/i18n.py`, escaped with `esc`; no chunking needed). A failed send is logged and
not retried. Scheduled and startup cycles without the flag stay silent.

**`/digest`.** New mutating command (15-minute staleness; added to `COMMANDS`, `MUTATING_COMMANDS`, README and the command
menu). It must not run on the poller thread. `DaemonService.trigger_digest() -> started | busy | not_ready` uses a
non-blocking `_digest_lock.acquire` check (the pattern of `trigger_cycle`) and schedules a one-off APScheduler job
(`id="digest_now"`, `misfire_grace_time=None`); busy means a digest is already running (a running scrape cycle does **not**
refuse it, like the cron digest). The job calls the shared digest body with a flag that skips the PyPI update notice, then
sends one result line through the notifier: `Sent <N> jobs` when `send_daily_digest` delivered, `Nothing is waiting` or `Sending
failed; the jobs stay waiting` otherwise, plus `<K> more waiting, send /digest again` when `digest_max_items` truncated.
Rows are marked sent exactly as the scheduled digest marks them (unchanged rule), so they are not repeated at 09:00. Same
threshold fallback as the cron: no stored profile means `.env` thresholds. It does not use `cycle_guard()` (which takes the
digest lock and would deadlock a nested acquire).

## 7. Error handling and edge cases

- Consent not given: the interview waits (state kept), `no` starts the wizard as today; the CV slot waits for `yes`.
- LLM unavailable mid-interview: reason text, state kept, `Retry` / `Draft now` / `Step-by-step questions instead` (the wizard
  starts and the facts are discarded).
- A CV and a typed answer in the same debounce window: debounce rules of 3.G.9 apply; an upload is its own request.
- The profile version moved while an interview drafts: the proposal binds to the version at draft time (`pc:` buttons,
  base-version check).
- Prompt injection inside a CV or answers: same schema barrier as 3.G.9; operations outside the closed set are rejected; CV
  text and facts never act as instructions.
- `/digest` or the report while the profile is missing or unreadable: behave like the cron (`.env` thresholds), nothing special.
- Daemon restart mid-interview: state and masked facts persist until 24 h after the first answer; in-memory copies (Retry
  text, CV text) are gone and the bot says to send them again.

## 8. Update interview and mode

A CV (or `/setup`) on a chat whose profile is no longer nearly empty runs the same interview in **update** mode: new
`MODES` value `update`, a prompt addendum (`UPDATE_ADDENDUM`: extend the profile, keep existing text, change a persona or
definition only by merging, never drop existing roles or markets unless the user says so), `pending_change.mode = "update"`,
label `llm update`. `ONBOARDING_ADDENDUM` keeps its assumption of an empty profile. The diff shown to the user is the
safeguard: persona and definition changes are shown in full as today.

## 9. Testing (no network)

- `tests/test_llm_e2e.py` (`Harness`, `FakeTelegram`, `ScriptedLLM`; `make_providers` injects `wait_fn` so retries never
  really sleep): interview happy path to a confirmed profile; a vague answer gives one follow-up; an answer covering two slots
  skips the second; a bare `no` is an answer, not a cancel, also with a pending proposal in update mode; the 8-question cap; a
  failed turn call keeps the interview; a draft problem keeps the interview; `Draft now`; `Step-by-step` switch; `/cancel`
  purges the context; **24 h expiry counted from the first answer even with turns every hour**; restart of `/setup`; stale `iv:`
  button; consent v2 re-asked; CV `Send` before consent shows the notice first and completes after `yes`.
- CV: fixtures built in the test (a minimal PDF, an owner-password-only PDF that opens with `decrypt("")`, a user-password
  one, a text-less one, an 11-page one, an oversize one, a non-PDF named `.pdf`, a header with junk before `%PDF-`); the
  preview contains no e-mail, phone or header block; **nothing reaches the provider before `Send`**; Cancel discards; no PDF
  bytes and no CV text in the database; the extractor subprocess has its own tests, including a hang (kill at the timeout) and
  a decompression bomb (limits).
- Reliability: one test per failure class and for the precedence table (NIM timeout + OpenRouter quota, NIM 429 + OpenRouter
  invalid, ...); retry after timeout succeeds and spends two cap slots; 429 wait longer than 30 s moves to the next provider;
  the overall deadline ends a request; OpenRouter gets no retry and the chat daily share stops at 15; `Retry` resumes without
  the user re-sending text; a caplog scan finds no prompt, answer or file name.
- Daemon-level fixture (new, not in `test_llm_e2e.py`: a real `DaemonService` with a fake scraper and notifier and a
  direct call of the scheduled job): `CycleReport` fields for ok, empty, failed (LinkedIn 429, Google no data), skipped
  boards and unscored-after-outage; a `ConfigurationError` cycle still reports; `/run_now` sends exactly one report, also when
  a scheduled cycle held the lock; scheduled cycles send none; `/digest` sends waiting rows once, is `busy` while a digest
  runs, never deadlocks the poller (the command returns at once), skips the update notice, reports truncation, and its rows are
  absent from the next digest.
- Injection: instruction-shaped CV text and answers produce no operation outside the closed set.

## 10. Documentation and plan hand-offs

AGENTS.md: section 2 tree (`src/bot/interview.py`, `src/bot/pdf_text.py`, new `/digest`), 3.C.4 (who writes `jobs` for
`/digest`: the scheduler thread, like the cron digest), 3.C.8 (retries in `ChatLLM`, deadlines, caps, OpenRouter share, failure
classes), 3.G.5 (`trigger_cycle(notify)`, `trigger_digest`), 3.G.7 (new callbacks `iv:`, `cv:`), 3.G.8 (command set), 3.G.9 (the
interview replaces the one-shot onboarding), 3.G.10 (CV-derived facts are stored in the DB for up to 24 h; only the CV text
is memory-only; consent v2). README (setup flow, CV, `/digest`, report), `.env.example` (nothing new unless a setting is added),
test count.

Left to the plan: exact question and report wording; schema field names of `interview_turn`, `extract` and the draft;
measuring real NIM latency of a turn call on a live probe in the first task (the spec's hybrid design assumes the tiny turn
output is fast; if it is not, the < 4 words rule and the follow-up cap are the tuning levers).
