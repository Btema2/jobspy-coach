# Interview onboarding, CV upload, reliable chat LLM and run report — design

Status: revision 3 (2026-11-19), after two independent reviews (revision 1: 5 blockers; revision 2: 1 blocker; all
resolved below). Branch: `feature/onboarding-interview`, stacked on `feature/llm-layer`.
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

**Entry and consent (nothing is stored before consent).** `/setup` with a configured LLM checks consent for
`CONSENT_VERSION` **before the first question**. Without consent the bot shows the notice (`cs:` buttons) and asks nothing;
`yes` posts the intro and the first question, `no` starts the deterministic wizard (as `_on_consent_button` does for
onboarding today). No `conversation.context` row for an interview exists before consent, so no answer can be persisted
before the user has seen the notice that discloses the storage. `/setup guided` and "no LLM configured" keep the wizard
unchanged. Starting an interview discards a pending proposal and an open clarification, each with a notice
(`PROPOSAL_DISCARDED_REPLY`, as `/setup` does today). A leftover `awaiting_description` row from before the upgrade is
retired (treated as idle). `_in_description` is replaced by an `_in_interview` check for the new state.

**Intro message** (replaces `ONBOARDING_PROMPT`): what will happen (6 to 8 questions, a few minutes), that a CV PDF, a
`.txt`/`.md` file or pasted text can be sent at any point and answers several questions at once, that nothing changes until the
user confirms, and the buttons `Draft now`, `Step-by-step questions instead` and `Cancel`.

**Checklist (owned by code, in this order)**: `roles` (what jobs, with synonyms), `markets` (countries and/or remote),
`languages` (the candidate's working languages: they go into the persona text and decide the languages of the synonyms; they
do **not** map to `set_output_language`, which is the alert language), `level` (seniority), `avoid` (dealbreakers),
`bonus` (credentials, languages or background a posting could value). Each slot has fixed question text and a status
`empty | partial | filled | skipped`.

**Turn loop.** Every interview input (a text answer of any length, a button, an upload) goes through the `bot-conversation`
worker queue in arrival order, and **all reads and writes of the interview context happen on the worker under
`_state_lock`**, never on the poller (so a short answer cannot race a running turn call and lose an update). Each queued
input records the slot (`asked` index) that was current when it arrived.

1. **Store.** The answer is masked (`mask_personal_data` plus the URL/handle masking of section 4) **before it is written
   anywhere** and stored **verbatim as the slot's facts** (code). Per-slot facts are capped at 2,000 characters
   (`MAX_SLOT_FACTS_CHARS`): an answer that would exceed it is refused with `That is too long for one answer; shorten it or
   send it as a file.` and nothing is stored. Bare words: `no none skip - n nope` mean **skip** (slot `skipped`, no LLM call);
   a bare confirm word (`yes y ok okay sure yep confirm apply`) means "yes" and is stored as `partial` with a code-written
   follow-up (`Which ones?`, no LLM call); `cancel` is not stored and answers `Send /cancel to stop the interview.` The
   `on_text` bare-word router never runs in `awaiting_interview` (only `awaiting_clarification` is exempt today; the
   interview joins it).
2. **Attribution.** An answer whose recorded slot is no longer the current one (it was typed while a turn call ran) is
   appended to **that** slot's facts and does not advance the loop. A follow-up answer likewise appends to its slot's facts
   and gets no further turn call.
3. **Turn call.** An answer is "short" when `len(text.split()) < 4 and len(text) < 40` (so scripts written without spaces
   are judged by length); a short non-skip answer marks the slot `filled` with no LLM call. A longer answer runs one **short
   turn call** (`kind: interview_turn`): input = only the slot question, the answer and the slot list (the stored profile is
   **not** sent); output, validated by a Pydantic schema, is tiny: `{covers: [slot names], followup: string|null}`
   (follow-up <= 300 characters, only when the answer is vague or contradictory). It does not extract structured data (the
   draft call does, from the raw facts). A failed or invalid turn call is treated as "no follow-up, covers nothing" (a
   one-line note that the step could not be checked), never as a lost interview, and is not retried by the user.
4. **Next step**, decided by code: the follow-up (at most one per slot), else the next slot that is `empty`, else done. At most
   8 questions are asked per interview (follow-ups included), then it drafts. Slots covered by earlier answers or by a CV
   are skipped.
5. **Draft call.** When no slot is `empty` (or after `Draft now`) one call turns all facts (block limit 20,000 characters, i.e.
   8 slots x 2,000 plus overhead, so nothing is silently cut) plus the stored profile into the existing onboarding operation
   set and the existing confirm-to-apply proposal (`mode="onboarding"` or `"update"`, section 8; same buttons, same `Apply
   persona change` handling). The draft schema allows `proposal` and `none` only (no `clarify`); `none` is a draft problem
   with the reason `nothing to propose`. Nothing is applied before confirmation.
6. **Draft problems keep the interview.** A `ChatFailure` of any class, `TranslationProblem` or `ProfileValidationError` shows
   the reason with `Retry` / `Draft now` / `Step-by-step questions instead`. For a validation error the bot names the slot by
   the failing operation's kind (market -> `markets`; role or term -> `roles`; rule -> `avoid`; target level -> `level`;
   anything else: no slot named) and asks that question again.

Facts are untrusted data: they enter prompts only inside a delimited block like `<user_message>` (`_clean` with the explicit
limit above), tags neutralised; a schema failure drops the update. The model never applies anything and never decides order
or stop conditions.

**Buttons.** `iv:<nonce>:draft|cancel|retry|guided` (nonce = 8 hex stored in `context`, so a stale button answers `This
choice is no longer active; send your answer again.`; at most 20 bytes). `Retry` exists for the **draft** (its facts are in
the database) and for the **extract** step of a CV (the CV text is kept until extract succeeds, section 4); a failed turn call
has nothing to retry. After a daemon restart in-memory data (CV text, free-text retry text) is gone and the button says to
send it again; the interview state itself persists.

**State.** New conversation state `awaiting_interview` added to `STATES` in `conversation_store.py` (no migration: the column
has no CHECK). `conversation.context` JSON: `{nonce, created_at, slots, asked, facts}`; **`created_at` is set once at the
first stored answer and the 24 h purge counts from it** (the first-storage rule, enforced today only for
`awaiting_clarification`, extends to `awaiting_interview`; `purge_contexts` and `_purge_old_contexts` cover both and every
turn rewrite keeps `created_at`). `facts` is dropped when the draft proposal is stored (`_store_proposal` sets context None),
on apply, cancel, supersede and expiry, with `secure_delete` and the WAL checkpoint as today. The interview ends on: storing
the draft proposal, apply, `/cancel`, expiry, the step-by-step switch, and a new `/setup`; any failure keeps it. Cancelling a
draft proposal does **not** resume the interview (the facts are gone); the user starts `/setup` again.

## 4. CV upload (PDF) and text files

Accepted wherever `.txt` / `.md` uploads are accepted today; `_add_upload` gets a PDF branch (suffix `.pdf` or mime
`application/pdf`; size checked against `MAX_CV_BYTES` from Telegram's `file_size` before download and again by the
size-limited `download_file`). **A `.txt` / `.md` upload during an interview is treated as a CV** (the same pipeline below
minus the PDF child; its 20 KB limit is unchanged). Inside `awaiting_interview` a CV answers several slots at once. In any
other state a CV (and `/setup`) starts or restarts the interview (section 8); the bot never changes the existing
state at upload time: extraction and the preview happen first, and the interview (with its notices) starts only at `Send`
after consent.

- **Limits**: <= 5 MB (`MAX_CV_BYTES`), <= 10 pages, <= 20,000 extracted characters, `%PDF-` within the first 1,024 bytes.
  Encrypted PDFs: try `decrypt("")` (owner-password-only files open without a user password); if that fails or needs a
  crypto library that is not installed, refuse with `This PDF is password-protected. Send an unprotected copy or paste the text.`
  **No text layer** means fewer than 200 non-whitespace characters overall: refused with `This PDF has no readable text (it
  looks scanned). Paste the text or send a .txt file.`
- **Extraction runs in a child process**, never in the worker thread (a thread cannot be stopped; a hostile file could keep the
  single worker busy and starve the GIL): `[sys.executable, "-I", "-m", "src.bot.pdf_text"]` (resolved through the installed
  editable package, not the working directory) receives the bytes on stdin and prints `{pages, text}` JSON on stdout;
  `stderr=subprocess.DEVNULL` (pypdf warnings must not reach the log), a minimal `env`, **no `preexec_fn`** (unsafe with
  threads); the parent kills it after 15 s wall clock; the child sets its **own** `RLIMIT_AS` (512 MB) and `RLIMIT_CPU` (10 s,
  a backstop) with `resource.setrlimit` (Linux), checks the page count before reading any page and stops once 20,000 characters
  exist. Any failure (timeout, kill, non-zero exit, bad JSON) is logged by type name only and answered with `This PDF could not
  be read. Paste the text or send a .txt file.` The child never receives the file name and nothing logs it.
- **Dependency**: `pypdf` is not installed or declared today. The plan adds it to `pyproject.toml` with a floor at a release
  that contains the decompression-limit fixes (`pypdf>=6.1`, verified at install time) and installs it into `.venv`.
- **Privacy pipeline** (before anything leaves the machine): `mask_personal_data` (e-mail, phone) plus masking of URLs and
  social handles; the contact header block (everything before the first section-like heading, or the first 6 non-empty lines
  when none is found) is removed. Names and addresses elsewhere cannot be recognised; the consent notice says so.
- **Order: extract and preview first, consent gates `Send`.** Extraction and the preview send and store nothing, so they need
  no consent. The bot shows `Read <N> pages, <M> characters. First lines: <short masked preview>` with buttons
  `cv:<nonce>:send|cancel` (<= 19 bytes). On `Send`, without consent for `CONSENT_VERSION` the consent notice is shown and
  nothing proceeds until `yes` (`no` drops the CV slot; in `/setup` the wizard starts). Only after consent does the interview
  start (if needed, with its discard notices) and the `extract` call run. The CV text is held in a **dedicated in-memory slot
  keyed by the `cv:` nonce** (separate from the single `_held` paste slot), expires after 24 h and is dropped on `Cancel`,
  `/cancel`, `reset_memory`, a new pairing, expiry, a declined consent and **after `extract` succeeds** (a failed extract keeps
  it so `Retry` works). A new CV replaces the old one.
- **Consent v2**: `CONSENT_VERSION` 1 -> 2. The notice names CVs and the interview facts, says that these facts (masked
  answers, not the CV text) are stored in the local database for up to 24 hours, recommends leaving out names and addresses;
  the rest as today.
- **Use**: after `Send`, the masked CV text goes through one `extract` call (`kind: extract`, input = the CV text only, no
  stored profile) returning `{covers: [slots], facts: {slot: short text}}` (each fact <= 2,000 characters by schema); the facts
  are stored like answers (masked), the covered slots become `filled`, and the interview continues with the gaps. PDF bytes
  are never stored; the CV text is dropped once `extract` succeeds. The upload caption is treated as an extra answer.

## 5. Reliable chat LLM

- **Retry in `ChatLLM`, not in the provider.** `CompletionOptions.max_attempts` stays 1 and `ChatLLM` calls `complete_json`
  again itself, so **every** HTTP attempt takes a slot from the hourly cap (today one slot per provider precedes the
  provider's internal retries, which would make retries invisible to the cap). Providers gain two subclasses of
  `TransientProviderError` so the rule is implementable by type: `ServerError` (5xx and 408) and `UpstreamErrorEnvelope` (the
  HTTP-200 error envelope). **NIM only, at most one extra attempt, and only after `ProviderTimeout`, `ServerError` or a
  429 (`RateLimited`)**; never after `InvalidResponse`, `ResponseTruncated`, `UpstreamErrorEnvelope`, `BadRequest`, a
  transport error or a validation failure (those go straight to the next provider). OpenRouter gets **no** retry (its free
  budget is shared with evaluation).
- **Reserve the fallback.** A NIM retry happens only when `remaining_deadline - OPENROUTER_RESERVE_SECONDS (60) >=` the
  attempt timeout; otherwise the next provider is tried at once. Consequence, intended: after a full-length NIM timeout the
  request goes to OpenRouter (as today) instead of burning the deadline on a second NIM timeout; the retry covers fast
  transient failures (429, 5xx, quick timeouts). A 429 retry waits `min(retry-after or limiter pause, remaining deadline -
  reserve)` and only when that is <= `CHAT_LIMITER_MAX_WAIT_SECONDS` (30 s); `limiter_max_wait` is also clipped to the remaining
  deadline. The provider's own sleep/backoff is not used (attempts stay at 1).
- **Per-attempt timeouts and overall deadline**: interview turn and `extract`: 90 s per attempt, 150 s overall; draft: 180 s
  per attempt (`CHAT_DRAFT_TIMEOUT_SECONDS`), 300 s overall. Each attempt's timeout is clipped to the remaining deadline.
  httpx `read` timeouts are inactivity timeouts and connect (10 s), write (30 s) and pool timeouts are not clipped, so the
  deadline is an approximate wall-clock bound (a few seconds over at worst). The progress note reads `Still working; this can
  take a few minutes.` (replacing the "up to 2 minutes" text). `ChatLLM` takes an injectable clock for tests.
- **Caps**: `CHAT_HOURLY_CAP` 30 -> 60 attempts per sliding hour (an interview with a CV is about 8 to 12 calls plus retries).
  New `CHAT_OPENROUTER_DAILY_SHARE = 15`: chat may spend at most 15 of the shared `OPENROUTER_DAILY_BUDGET` (45) per UTC day,
  counted in `ChatLLM`, so a chatty setup cannot exhaust the evaluation fallback. All `CHAT_*` values stay module constants.
- **Failure classes** (`ChatFailure` enum carried by the typed exceptions; replies are code-written text, never LLM prose).
  Mapping, precedence when providers fail differently (highest wins): `capped` (hourly cap or OpenRouter share spent) >
  `bad_answer` (`InvalidResponse`, `ResponseTruncated`, `UpstreamErrorEnvelope`, schema failure) > `timeout`
  (`ProviderTimeout`, deadline spent) > `rate_limited` (`RateLimited`, `RateLimitWaitExceeded`) > `unavailable` (no key, auth,
  quota, `ServerError`, `BadRequest`, `ModelNotFound`, `UnexpectedStatus`, transport errors, nothing reachable). Texts: `The
  model timed out`, `The model is rate-limited`, `No model is reachable`, `The model's answer could not be used`, `Too many
  model requests this hour`; each with the retry time when known (a NIM 429 has none: say `try again in a minute`) and what
  happens next.
- **Free-text change requests** (not the interview) get the same reason text and a `Retry` button; the original text is a
  second in-memory holder (24 h), dropped by `/cancel`, `reset_memory` and a new pairing like the CV slot.
- **Logging**: provider name, failure class, HTTP status and `request_id` only; never prompts, answers, URLs, file names or
  `str(exc)`.

## 6. Run report and `/digest`

**Report.** `evaluate_unprocessed` returns one `EvalSummary` dataclass (evaluated, wow_sent, still unprocessed,
`stopped_reason` from the existing `batch_stopped` state) replacing today's tuple; its callers (the CLI `evaluate` and
`run_pipeline_cycle`) are updated, and `run_pipeline_cycle` keeps its `(new, wow)` return for existing callers. A new
`DaemonService.run_pipeline_cycle_report()` builds a frozen `CycleReport`. Sources per field:

| Field | Source |
|---|---|
| `new_to_evaluate`, `prefiltered` | `_scrape_queries` split of `total_new_saved` (it counts pre-filtered rows today) |
| `calls` (ok, empty, failed) and `problems` | per-(query, board) `ScrapeOutcome` loop in the daemon; `problems` = distinct `(board, reason)` with a count; reasons come from `outcome.error`, with a classifier for LinkedIn 429 (`blocked (429)`; detail-page failures seen through `skipped_empty` read `detail pages blocked`) and Google (`no data returned`) |
| `skipped_boards` | `begin_cycle` / `end_cycle` of the scraper gains a returned count of boards skipped for a missing country (today a private `_cycle.skipped`) |
| `evaluated`, `unscored`, `stopped_reason`, `wow_sent` | `EvalSummary`; `unscored` splits "beyond `eval_batch_limit`" from "provider unavailable" |
| `digest_waiting` | `get_pending_digest_jobs` count at or above the digest threshold (all of them, not capped by `digest_max_items`) |

**Trigger and delivery (flag semantics).** `trigger_cycle(notify: bool = False)`; `BotContext.trigger_cycle` becomes
`Callable[..., str]`; `/run_now`, `/resume` and the chat-proposal resume (`ConversationService._trigger`) pass `notify=True`.
The daemon sets `_notify_requested` **only when it answers `started`, and before `add_job`**. **Every** cycle (interval,
startup, `run_now`) runs through the report wrapper; the flag is **read and cleared at the end of `_run_pipeline_cycle`,
still inside `_cycle_lock`**, so a scheduled cycle that was already running when the flag was set still reports, and a
`run_now` job skipped because the lock was held loses nothing. The report is **sent after `_cycle_lock` is released** (so the
`/digest` hint works at once), from an APScheduler executor thread, through `notifier.send_message` as one message of at
most ~12 lines (plain English bot text, not `src/i18n.py`, escaped with `esc`; no chunking). A failed send is logged and not
retried. The wrapper catches any exception (including `ConfigurationError`) and, when flagged, reports `The cycle stopped:
<class or no-LLM-key hint>`; unflagged cycles keep today's behaviour (exception logged at ERROR; `start()` keeps its
`ConfigurationError` log line).

**`/digest`.** New mutating command (15-minute staleness; added to `COMMANDS`, `MUTATING_COMMANDS`, README and the command
menu). It never runs on the poller thread. `DaemonService.trigger_digest() -> started | busy | not_ready`: `not_ready` unless
`self.scheduler` exists **and `.running`** (during the long startup cycle the scheduler is not started yet, so an
immediate `started` would be a lie); `busy` when `_digest_lock.locked()` (a running scrape cycle does **not** refuse it, like
the cron digest); otherwise it schedules a one-off job (`id="digest_now"`, `misfire_grace_time=None`). The job, on an
APScheduler executor thread, takes `_digest_lock` **non-blocking** (if it loses a race with the cron digest it sends `A digest is
already running`), runs the shared digest body with a flag that skips the PyPI update notice, then sends one result line:
`Sent <N> jobs` when every message was delivered, `Sent <k> of <m> messages; some jobs may be missing` on a partial send,
`Sending failed; the jobs stay waiting` when none, `Nothing is waiting` when empty, plus `<K> more waiting, send /digest again`
when `digest_max_items` truncated. The row-marking rule is deliberately unchanged (rows are marked sent when at least one
message was delivered, exactly as the cron digest does), which is why the partial line warns honestly. Threshold fallback as the
cron: no stored profile means `.env` thresholds. It does not use `cycle_guard()` (which takes the digest lock and would
deadlock a nested acquire).

## 7. Error handling and edge cases

- Consent not given: no interview state exists (section 3); `no` starts the wizard.
- LLM unavailable mid-interview: reason text, state kept, `Retry` (draft/extract) / `Draft now` / `Step-by-step questions
  instead` (the wizard starts and the facts are discarded).
- A CV and a typed answer in the same debounce window: debounce rules of 3.G.9 apply; an upload is its own request.
- The profile version moved while an interview drafts: the proposal binds to the version at draft time (`pc:` buttons,
  base-version check).
- Prompt injection inside a CV or answers: same schema barrier as 3.G.9; operations outside the closed set are rejected; CV
  text and facts never act as instructions.
- Daemon restart mid-interview: state and masked facts persist until 24 h after the first answer; in-memory copies (CV text,
  free-text retry text) are gone and the bot says to send them again.

## 8. Update interview and mode

The profile counts as **starter** when it has no enabled role and no market; then the interview drafts with `mode="onboarding"`
(`ONBOARDING_ADDENDUM`, assumes an empty profile). Otherwise it runs in **update** mode: new `MODES` value `update`, a prompt
addendum (`UPDATE_ADDENDUM`: extend the profile, keep existing text, change a persona or definition only by merging, never drop
existing roles or markets unless the user says so), `pending_change.mode = "update"`, label `llm update`. The diff shown to the
user is the safeguard: persona and definition changes are shown in full as today. Any state except `awaiting_interview` (an
idle chat, `awaiting_confirm`, `awaiting_clarification`) starts or restarts the interview on `/setup` or a CV `Send`, with the
discard notices of section 3.

## 9. Testing (no network)

- `tests/test_llm_e2e.py` (`Harness`, `FakeTelegram`, `ScriptedLLM`): the limiter is built with injected `wait_fn` / `time_fn`
  and `ChatLLM` with a fake clock, so retries and deadlines never really sleep (`make_providers` has no such parameter today;
  the plan adds one or builds the limiter in the harness). Interview happy path to a confirmed profile; **no `conversation` row
  exists before consent** and a short answer is never stored before it; a vague answer gives one follow-up; an answer covering
  two slots skips the second; bare `no` is a skip (also with a pending proposal present: update mode discards it with a
  notice), `yes` becomes `partial` + `Which ones?`, `cancel` is not stored; a long answer is refused over 2,000 characters; an
  answer typed during a turn call is attributed to its slot and does not advance the loop, with no lost context update; the
  8-question cap; a failed turn call keeps the interview; a draft problem (provider failure, validation error naming the slot,
  `none`) keeps it; `Draft now`; `Step-by-step` switch; `/cancel` purges the context; **24 h expiry counted from the first
  answer even with turns every hour**; restart of `/setup`; stale `iv:` button; consent v2 re-asked; a `.txt` upload during
  the interview takes the CV path.
- CV: fixtures built in the test (a minimal PDF, an owner-password-only PDF that opens with `decrypt("")`, a user-password
  one, a text-less one, an 11-page one, an oversize one, a non-PDF named `.pdf`, a header with junk before `%PDF-`); the
  preview contains no e-mail, phone or header block; **nothing reaches the provider and nothing is stored before `Send` and
  consent**; `Send` without consent shows the notice and completes after `yes`; `no` drops the CV slot; Cancel discards; a
  failed extract keeps the CV for `Retry`, success drops it; no PDF bytes and no CV text in the database; the extractor
  subprocess has its own tests, including a hang (killed at the timeout), a decompression bomb (limits) and stderr silence.
- Reliability: one test per failure class and for the precedence table (NIM timeout + OpenRouter quota, NIM 429 + OpenRouter
  invalid, ...); a fast 429/5xx on NIM is retried once and spends two cap slots; a full-length NIM timeout goes straight to
  OpenRouter (reserve rule), not to a second NIM attempt; `UpstreamErrorEnvelope` and `InvalidResponse` are not retried; a
  429 wait over 30 s moves to the next provider; the overall deadline ends a request; OpenRouter gets no retry and the chat
  daily share stops at 15; `Retry` resumes without the user re-sending text; a caplog scan finds no prompt, answer or file
  name.
- Daemon-level fixture (new, not in `test_llm_e2e.py`: a real `DaemonService` with a fake scraper and notifier and a direct
  call of the scheduled job): `CycleReport` fields for ok, empty, failed (LinkedIn 429, Google no data), skipped boards,
  detail-pages-blocked and unscored-after-outage; a `ConfigurationError` cycle still reports when flagged and logs when not;
  `/run_now` sends exactly one report, also when a scheduled cycle already held the lock and when the flag is set mid-cycle;
  a `busy` answer sets no flag; scheduled cycles send none; `/digest` returns `not_ready` before the scheduler runs, `busy`
  while a digest runs, never blocks the poller, skips the update notice, reports a partial send and truncation honestly, loses
  a race with the cron cleanly, and its rows are absent from the next digest.
- Injection: instruction-shaped CV text and answers produce no operation outside the closed set.

## 10. Documentation and plan hand-offs

AGENTS.md: section 2 tree (`src/bot/interview.py`, `src/bot/pdf_text.py`, new `/digest`), 3.C.4 (who writes `jobs` for
`/digest`: an APScheduler executor thread, like the cron digest; the worker now also owns interview context writes), 3.C.8
(retries in `ChatLLM`, per-attempt timeouts, deadlines, reserve, caps, OpenRouter share, failure classes, the two new error
subclasses), 3.G.5 (`trigger_cycle(notify)`, `trigger_digest`), 3.G.7 (new callbacks `iv:`, `cv:`), 3.G.8 (command set), 3.G.9
(the interview replaces the one-shot onboarding), 3.G.10 (consent is checked before the first question; CV-derived facts and
interview answers are stored in the DB for up to 24 h from the first answer; only the CV text is memory-only; consent v2).
README (setup flow, CV, `/digest`, report), `.env.example` (nothing new unless a setting is added), test count.

Left to the plan: exact question and report wording; schema field names of `interview_turn`, `extract` and the draft;
measuring the real NIM latency of a turn call with a live probe in the first task (the hybrid design assumes the tiny turn
output is fast; if it is not, the short-answer rule and the follow-up cap are the tuning levers).
