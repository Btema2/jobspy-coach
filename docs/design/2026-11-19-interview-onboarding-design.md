# Interview onboarding, CV upload, reliable chat LLM and run report — design

Status: draft for review (2026-11-19). Branch: `feature/onboarding-interview`, stacked on `feature/llm-layer`.
Builds on `2026-11-18-generic-profile-and-bot-design.md` (Branches 1-5, all implemented).

## 1. Why

Feedback from a first hands-on test of the bot:

1. The `/setup` intro is weak: one open "describe yourself" prompt, one proposal, no questions, no follow-ups.
2. A CV should be droppable to fill the profile.
3. NIM seemed to refuse requests **during setup**. Evidence: chat gets one attempt of up to 120 s per provider
   (`CHAT_TIMEOUT_SECONDS`, `CHAT_MAX_ATTEMPTS = 1`); the single onboarding call (full profile draft, a 550B
   reasoning model) is the slowest call the bot makes; a timeout falls through to free OpenRouter models, and the user
   then sees only `Couldn't understand that, try a command (/help).` with no reason and no retry.
4. After `/run_now` only WOW alerts arrive. This is by design (the digest runs only from the 09:00 cron) but gives
   no feedback; the user cannot tell what the run did, what failed or what is waiting.

Nemotron has **no image input** (checked by the user), so a CV can only be used through its text layer.

## 2. Scope

In: the interview, CV (PDF) upload, chat LLM reliability and failure reasons, the `/run_now` / `/resume` report,
the `/digest` command, docs and tests.

Out: fixing the Google bot-challenge and LinkedIn 429 scrape failures (they are only reported), OCR, image input,
`.docx`, streaming, any change to evaluation scoring.

## 3. Interview (hybrid)

`/setup` with a configured LLM (and consent) starts an interview. `/setup guided` and "no LLM configured" keep the
deterministic wizard unchanged.

**Intro message** (replaces `ONBOARDING_PROMPT`): what will happen (about 6 questions, a few minutes), that a CV PDF or
text can be sent at any point to answer several questions at once, that nothing changes until the user confirms,
and the buttons `Draft now` (stop asking, draft from what is known), `Step-by-step questions instead` (wizard) and
`Cancel`.

**Checklist (owned by code, in this order)**: `roles` (what jobs, with synonyms), `markets` (countries and/or
remote), `languages` (the candidate's working languages), `level` (seniority), `avoid` (dealbreakers), `bonus`
(credentials, languages or background a posting could value). Each slot has a fixed question text and a status
`empty | partial | filled | skipped`.

**Turn loop** (one request at a time on the existing `bot-conversation` worker):

1. The user answers (text, or a CV, see 4). One short LLM call (`kind: interview_turn`) gets the facts so far, the slot
   asked, the answer, and returns structured JSON validated by a Pydantic schema: updates for any slots the answer
   covers (not only the asked one), and optionally a single follow-up question (<= 300 characters) when the answer
   is vague or contradictory.
2. Code decides what is asked next: the follow-up (at most one per slot), else the next slot that is `empty`, else
   done. At most 8 questions are asked per interview; then it drafts.
3. When no slot is `empty` (or after `Draft now`) one **draft call** turns the facts into the existing onboarding
   operation set and the existing confirm-to-apply proposal (`mode="onboarding"`, same buttons, same
   `Apply persona change` handling). Nothing is applied before confirmation.

The model only extracts facts and writes follow-up wording; it never applies anything, and the question order and
the stop conditions are code. Facts and answers are untrusted data: they go into a delimited block exactly like
`<user_message>` today (`_clean`, tag neutralisation), and a schema failure on any slot update drops that update.

**State.** New conversation state `awaiting_interview` (the column has no CHECK, so no migration). `conversation.context`
JSON holds `{slots, asked, followups, facts}` where `facts` is the user's own words and extracted values. The existing
24 h retention (counted from first storage), purge on apply / cancel / supersede / expiry, `secure_delete` and
WAL checkpoint rules apply unchanged. Starting `/setup` again restarts the interview; `/cancel` is the full reset.

## 4. CV upload (PDF)

Accepted wherever `.txt` / `.md` uploads are accepted today (during the interview, or in an idle chat where it starts
the interview in update mode: it fills facts and drafts operations against the existing profile).

- **Limits**: file size <= 5 MB (`MAX_CV_BYTES`), <= 10 pages, <= 20,000 extracted characters, must start with `%PDF`,
  not encrypted. Violations get a one-line reason. A PDF with no text layer (scan) is refused with
  `This PDF has no readable text (it looks scanned). Paste the text or send a .txt file.`
- **Extraction**: local, with `pypdf` (new pure-Python dependency, added to `pyproject.toml`), in the worker thread with a
  wall-clock guard; any parser exception is logged by type name and answered with the same generic reason.
- **Privacy pipeline** (before anything leaves the machine): `mask_personal_data` (e-mail, phone) plus masking of URLs
  and handles; the contact header block (everything before the first section-like heading or the first 6 non-empty
  lines when none is found) is removed. Names and addresses elsewhere cannot be recognised; the consent notice says so.
- **Confirm step**: the bot shows `Read <N> pages, <M> characters. First lines: <short preview, masked>` with
  `Send to the model` / `Cancel` buttons (`cv:<nonce>:send|cancel`). Nothing is sent to a provider before `Send`.
  The consent gate (`CONSENT_VERSION`) still applies first; the notice gains one sentence naming CVs
  (**bump `CONSENT_VERSION` to 2** so existing users see it).
- **Storage**: PDF bytes are never stored (downloaded with the existing size-limited `download_file`, parsed, dropped).
  The masked text is held in memory only until sent, 24 h at most, like a held paste; after `Send` it is processed
  as an `extract` call that fills slots (it never becomes a proposal directly). Cancel discards it.
- The caption of the upload is prepended as an extra user answer.

## 5. Reliable chat LLM

- **Retry**: `ChatLLM.complete` makes up to 2 attempts on the same provider for `ProviderTimeout`, 5xx and 429
  (`CHAT_MAX_ATTEMPTS = 2`, second attempt only when the first failed *before* a response arrived or with a transient
  status; never for invalid JSON or schema failures, which go to the next provider as today). 429 waits for the
  limiter's `pause_for`, bounded by `CHAT_LIMITER_MAX_WAIT_SECONDS`.
- **Timeouts**: interview turns and `extract` use `CHAT_TIMEOUT_SECONDS = 120` unchanged; the **draft** call uses
  `CHAT_DRAFT_TIMEOUT_SECONDS = 180`. Both stay overridable from settings.
- **Cap**: `CHAT_HOURLY_CAP` 30 -> 60 provider HTTP attempts per sliding hour (an interview with a CV is about 8 to 12
  calls, and retries count). The OpenRouter daily budget stays shared with evaluation, unchanged.
- **Failure reasons** (new `ChatFailure` enum carried by the typed exceptions, so the translator and the interview
  render one text per class, never LLM prose):
  `timeout` ("The model timed out"), `rate_limited` (with the retry time), `unavailable` (providers down or no key),
  `bad_answer` (answer could not be used), `capped` (hourly cap, with the retry time). Each reply says what happens
  next: the interview keeps its state and shows a `Retry` button that re-runs the same step (no text is re-sent by the
  user); for a free-text change request the original text is re-offered the same way (held in memory, 24 h).
- **Logging**: provider name, failure class and HTTP status only. Never prompts, answers, URLs or `str(exc)` (3.C.8, 3.D
  rules unchanged).

## 6. Run report and `/digest`

- `DaemonService._run_pipeline_cycle` returns a `CycleReport` (frozen dataclass): `new_jobs`, `scrape_calls` (ok, empty,
  failed, skipped counts), `scrape_problems` (distinct `(board, reason)` with a count, reasons already redacted,
  e.g. `linkedin: blocked (429)`, `google: no data returned`), `evaluated`, `unscored` (rows still
  `unprocessed` after the batch) with the reason when the batch stopped early (`provider unavailable`), `wow_sent`,
  `digest_waiting` (rows `pending` at or above the digest threshold). `run_pipeline_cycle` keeps its `(new, wow)` tuple
  for existing callers; a new `run_pipeline_cycle_report()` returns the report.
- `trigger_cycle(notify: bool = False)`: `/run_now` and `/resume` pass `notify=True`; when that cycle ends the daemon sends
  the report to the paired chat through the notifier (HTML, `esc` on all text, chunked). Scheduled and startup cycles
  never send a report. A failed send is logged and not retried.
- **Report text** (English, from `i18n`-independent bot strings): one block of at most ~12 lines, e.g. new jobs, scraped
  boards with problems, scored / left unscored and why, WOW alerts sent, N jobs waiting for the digest and the hint
  `/digest` to receive them now.
- **`/digest`**: mutating command (15-minute staleness applies). Runs `run_daily_digest_cycle` on the poller's behalf
  through `cycle_guard()`-style locking (refused while a cycle or digest runs) and answers with the count sent; the
  sent rows are marked like a normal digest, so they are not repeated at 09:00. With nothing waiting it says so.

## 7. Error handling and edge cases

- Consent not given: the interview waits (state kept 24 h); `no` starts the wizard as today.
- LLM unavailable mid-interview: answer the failure reason, keep state, `Retry` / `Draft now` / `Step-by-step questions
  instead` (wizard starts with the facts discarded).
- A CV and a typed answer in the same debounce window: the debounce rules of 3.G.9 apply; an upload is its own request.
- The profile version moved while an interview drafts: the proposal binds to the version at draft time, like any
  proposal (`pc:` buttons, base-version check).
- Prompt injection inside a CV or answers: same schema barrier as 3.G.9; slot updates and operations outside the
  closed set are rejected; CV text never reaches the translator as instructions.
- Digest while the profile is missing or unreadable: reply with the compile problem; nothing is sent.

## 8. Testing (no network)

Extend `tests/test_llm_e2e.py` (`Harness`, `FakeTelegram`, `ScriptedLLM`):

- Interview: full happy path to a confirmed profile; a vague answer triggers exactly one follow-up; an answer covering
  two slots skips the second question; 8-question cap; `Draft now`; `Step-by-step` switch; `/cancel` purges
  `conversation.context`; 24 h expiry; restart of `/setup`.
- CV: fixture PDFs built in the test (a minimal hand-written PDF, an encrypted one, a text-less one, a 11-page one, an
  oversize one, a non-PDF named `.pdf`); preview shows no e-mail / phone / header block; nothing reaches the provider
  before `Send`; Cancel discards; bytes never in the DB; consent v2 re-asked.
- Reliability: per failure class one test (timeout then success on retry; 429 with `pause_for`; both providers down;
  schema failure falls to OpenRouter without retry; hourly cap message); `Retry` button resumes without re-sending text;
  logs contain no prompt text (caplog scan).
- Report: `CycleReport` fields for ok, empty, failed (linkedin 429, google no data) and unscored-after-outage cases;
  `/run_now` sends exactly one report, scheduled cycles none; `/digest` sends waiting rows once, is refused while a
  cycle runs, and those rows are absent from the next digest.
- Injection: instruction-shaped CV text and answers produce no operation outside the closed set.
- Docs: AGENTS.md (3.C.8, 3.G.7-10, new 3.G.11), README (setup flow, CV, `/digest`, report), `.env.example`
  (`CHAT_*` settings if exposed). Test count updated.

## 9. Open points deliberately left to the plan

Exact wording of questions and the report; the schema field names of `interview_turn`; whether the `pypdf` text
extraction needs a page-level timeout beyond the thread guard (measure on a real CV first).
