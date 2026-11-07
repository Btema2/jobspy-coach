# Model selection (Phase 0 bake-off)

This records how the evaluator's default models and request settings were chosen, using
`scripts/bakeoff.py` on 2026-11-05. It contains no candidate details: the fixtures
(`tests/fixtures/eval_cases.json`) are persona-neutral, and the neutral built-in persona was used.
Raw per-call results are kept locally under the git-ignored `local/bakeoff/` (no API keys, no job text).

## Methodology

- Fixtures: 14 generic vacancies. 7 "trap" cases that must be rejected (score <= 2.0: IT/engineering/sales
  management roles with a coaching side-duty, scrum master, fitness coach, executive coaching that needs
  native C2 language skills) and 7 relevant cases that must land in 6.0-10.0.
- Prompt: the real `build_system_prompt()` and `build_job_user_prompt()`, with the neutral default persona.
  The `--persona local` option exists but was not used for the decision.
- Per call we record: model id returned, latency, `finish_reason`, token usage (including reasoning
  tokens), JSON validity (`extract_json_payload` + schema validation) and whether the score is in band.
- Decision rule: pick the cheapest NIM `reasoning_effort` where every trap scores <= 2.0 and >= 80% of the
  positives are in band with valid JSON; otherwise step up one level. A backup qualifies only if it returns
  valid JSON on every fixture it was given and keeps traps <= 3.0.
- Budget: 39 NIM calls (3 of them immediate 429s) and 15 OpenRouter calls.

## NVIDIA NIM (`nvidia/nemotron-3-ultra-550b-a55b`)

Endpoint `https://integrate.api.nvidia.com/v1/chat/completions`. The model id is accepted exactly as written
and echoed back in the response `model` field.

| Run (generic persona, temperature 0.6, top_p 0.95, max_tokens 8192) | Calls | Valid JSON | Max trap score | Positives in band | Latency mean / p50 / p95 / max (s) | Completion tokens (of which reasoning) |
|---|---|---|---|---|---|---|
| `reasoning_effort=low`, 8 parallel, all fixtures | 14 | 14 | 1.5 | 7/7 | 14.5 / 11.7 / 25.3 / 25.3 | 13035 (8262) |
| `reasoning_effort=medium`, 6 parallel, 6-case balanced subset | 6 | 6 | 1.5 | 3/3 | 11.6 / 9.7 / 19.6 / 19.6 | 5312 (3386) |
| `enable_thinking=false`, 2-case subset | 2 | 2 | 1.5 | 1/1 | 4.1 / 4.0 / 4.2 / 4.2 | 563 (0) |
| `reasoning_effort=low`, `stream=true`, 2-case subset | 2 | 2 | 1.5 | 1/1 | 14.4 / 11.0 / 17.9 / 17.9 | 1566 (919) |

Findings:

- `reasoning_effort` is honoured but is a weak lever for this model: `low` and `medium` gave identical
  scores on the shared cases and similar reasoning-token counts (the medium subset was not slower; the
  difference is within run-to-run noise). `low` meets the decision rule, so `low` is chosen.
- `enable_thinking=false` cuts latency to about 4 s with zero reasoning tokens and was correct on the two
  cases tried. It is a valid speed option but was not validated beyond two calls, so it is not the default.
- Temperature 0.6 never looped or truncated: every successful call ended with `finish_reason=stop`.
  Highest completion usage seen was about 2.1k tokens (reasoning up to about 1.8k). `max_tokens=4096` leaves
  roughly 2x headroom, and a `length` retry at x1.5 (6144) covers outliers.
- The output was clean JSON every time: no markdown fences and no `<think>` tags in `content`. Reasoning
  arrives separately in `message.reasoning_content` (ignored). The parser should still keep its fence and
  `<think>` fallbacks because other providers differ.
- Non-streaming calls of up to 35 s survived; `stream=true` also works (usage is returned with
  `stream_options.include_usage`) but is not needed at these latencies. Streaming is off by default;
  re-evaluate if latencies reach minutes.
- `response_format` and `thinking_token_budget` were deliberately not sent.

### Concurrency and rate limits

- Observations: with 12 worker threads and 14 queued jobs, 3 requests were rejected with an immediate HTTP 429
  (about 0.1 s each) and the other 11 succeeded, so roughly 11 requests were accepted in flight at once
  (the 2 queued jobs started after 3 rejected threads freed, with about 9 still running). The same 14
  jobs on 8 parallel threads produced no 429s. The provider's real limit is unknown: the 3 rejections are
  as consistent with a burst-rate limit (12 requests in the same instant) as with a concurrency cap, and
  these data do not show a concurrency cap below 11.
- 429 body: `{"status": 429, "title": "Too Many Requests"}`. Response header names on the 429:
  `connection, content-length, content-type, date, nvcf-error-source, vary, x-request-id`.
  There is no `Retry-After` and no `x-ratelimit-*` header, so client backoff cannot be header-driven.
- Header names on 200 responses: `access-control-expose-headers, connection, content-type, date, nvcf-reqid,
  nvcf-status, transfer-encoding, vary, x-request-id`. No rate-limit headers were exposed.
- Sizing (plan section 3.2): by Little's law, in-flight requests = rate x latency. At the mean latency of
  about 14.5 s, a 30 rpm rate needs about 7.3 requests in flight, so 8 workers (about 33 rpm of capacity)
  are enough to saturate the client-side limiter. 8 was also the largest parallelism that was actually run
  without any 429. A p95-based figure (30/60 x 25.3 s is about 13, or 15 with the script's default margin
  of 2) was deliberately not used: the client-side rate limiter spaces requests, which removes the burst
  that likely caused the 12-thread rejections, and 429 backoff plus retry covers the latency tail.
- `NVIDIA_RPM_LIMIT=30` is an assumed, conservative client-side cap. It was not measured: the user-reported
  provider limit is about 40 rpm, and no rate-limit headers were observed. The longest sustained rate in the
  bake-off was only about 24 requests per minute (14 calls in 35 s; observed, unverified).
- Eight workers is chosen, and 429s must be handled as transient with computed backoff.

## OpenRouter free fallbacks

Free models were listed from the public `GET /api/v1/models` (no key), keeping `pricing.prompt == 0` and
`pricing.completion == 0`, then excluding content-safety, code, audio/non-text, tiny (< 12B), stealth,
router and any model with `expiration_date`. That left 8 candidates. Every call used a pinned `model`,
`provider.require_parameters=true` and `response_format={"type": "json_object"}` where the model lists it.

| Candidate | Calls | Outcome |
|---|---|---|
| `nvidia/nemotron-3.5-super-vl:free` | 3 (+1 via `models` array) | 4/4 valid JSON, traps 1.5, relevant case 9.0; 11-15 s. **Qualifies** (small sample). |
| `xingchen-agi/xing4.0-29b-a4b:free` | 4 | 2 upstream 429s; a trap scored 3.5 (> 3.0). **Does not qualify.** |
| `google/gemma-4-31b-it:free` | 4 | All 4 upstream 429 ("temporarily rate-limited upstream"). Unverified. |
| `google/gemma-4-26b-a4b-it:free` | 3 | All 3 upstream 429. Unverified. |

- A `models` array whose first entry was upstream rate-limited fell through server-side to the second
  entry and returned a valid result (response `model` named the model that answered). Server-side
  fallback therefore works for rate-limit errors, but it does not protect against HTTP 200 garbage.
- OpenRouter 429 body shape: `{"error": {"code": 429, "message": "Provider returned error", "metadata":
  {"provider_name", "raw", "is_byok", "limit_source", "remedy_hint", ...}}, "user_id": ...}`.
  These are upstream-provider limits on the shared free pool, not the account's 20 rpm / 50 per day quota.
  No `Retry-After` or `x-ratelimit-*` header was seen. They may still count against the daily quota.
- `nvidia/nemotron-3-ultra-550b-a55b:free` (the primary model, free on OpenRouter) also exists in the pool
  but does not advertise `response_format`; it was not tested. It is worth trying as a same-model backup.
- OpenRouter free models are frequently rate-limited upstream: treat the fallback as best-effort.

## Latency and timeouts under concurrency (live-run analysis and drill)

Evidence from one live evaluation batch (148 postings, 8 workers, `reasoning_effort=low`, read timeout 240 s,
the candidate's local persona) plus a small follow-up drill. Job text and names are deliberately omitted.

Live batch, per-job latency as logged by the daemon (includes retries and backoff):

| Slice | Jobs | Mean / p50 / p95 / max (s) |
|---|---|---|
| Whole batch | 148 | 59.8 / 43.1 / 270.9 / 338.8 |
| Healthy jobs (< 200 s) | 140 | 46.5 / 41.5 / 97.6 / 115.0 |
| Stalled jobs (>= 200 s, 270-339 s) | 8 | one read timeout (240 s) plus a successful retry of 30-100 s |

- Wall time 1140 s, effective 7.8 jobs/min, against a 30 rpm assumption. By Little's law 8 workers at a 46.5 s
  mean give a ceiling of about 10 jobs/min; the first 132 jobs ran at about 9.8 jobs/min. The gap to 30 rpm is
  latency, not the client-side limiter (which never delayed anything) and not provider throttling.
- Error classes: 8 `ReadTimeout` and 2 `ConnectTimeout` transport errors, zero HTTP 429, zero 5xx. All 8 read
  timeouts fell in one 2-minute window (calls that started around the same time, ending about 240 s later):
  one provider-side stall, not steady degradation. 7 of those jobs recovered on the first retry; one job
  burned a read timeout and two connect timeouts (about 100 s of backoff and attempts) and was answered by the
  OpenRouter fallback (338.8 s in total). No job was lost, no failures.
- The circuit breaker never opened (it needs 3 consecutive failed jobs; only that one job failed on NIM).
- About 7% of healthy jobs (10 of 140) took longer than 90 s, so a 90 s read timeout would have cut about as
  many healthy calls as stalled ones.

Drill (16 postings built from the generic fixtures, each description repeated to about 2.5k characters, the
real provider chain with NIM only, same settings as the live run, run one configuration at a time;
40 NIM calls in total and no retries occurred, 2026-11-07):

| Configuration | Jobs | OK | Latency mean / p50 / p95 / max (s) | Wall (s) | Jobs/min | In-flight peak |
|---|---|---|---|---|---|---|
| `EVAL_MAX_WORKERS=4`, read timeout 240 | 16 | 16 | 22.2 / 19.0 / 38.6 / 57.5 | 115.6 | 8.3 | 4 |
| `EVAL_MAX_WORKERS=8`, read timeout 240 | 16 | 16 | 21.8 / 20.5 / 37.0 / 39.3 | 56.0 | 17.1 | 8 |
| `EVAL_MAX_WORKERS=8`, read timeout 30 | 8 | 8 | 12.6 / 10.8 / 24.7 / 24.7 | 24.7 | 19.5 | 8 |

Reading the numbers (small samples, one run each, and NIM latency varied by roughly 2x between runs minutes
apart):

- Per-call latency did not depend on 4 versus 8 concurrent calls (22.2 versus 21.8 s mean), so throughput
  scaled with workers (8.3 to 17.1 jobs/min). That supports keeping 8 workers; it does not show a
  concurrency penalty up to 8, so the live run's 46 s mean is better explained by the provider being slower at
  that time (and longer real descriptions) than by the worker count. It does not show that more than 8 is safe:
  the bake-off saw HTTP 429 at 12 parallel requests.
- The 30 s read-timeout variant never timed out (max 24.7 s that run), so it demonstrates nothing about tail
  behaviour. It cannot be used to justify a shorter timeout.
- Decision: no default changed (`EVAL_MAX_WORKERS=8`, `NVIDIA_TIMEOUT_SECONDS=240`, connect timeout 10 s, 3
  attempts). A shorter read timeout (for example 90 s) would recover a stalled call about 150 s sooner, but
  it would also cancel and repeat roughly 7% of healthy calls, wasting a rate-limited request each, and the
  evidence is a single stall event. Revisit if stalls recur: look at `inflight_peak` and the latency stats
  below, and consider a read timeout of about 150 s (above the healthy maximum seen, 115 s).

Each evaluation batch now logs `latency[mean=.. p95=.. max=..]` (per-job seconds, retries included, failed
jobs included) and `inflight_peak=N` on the existing "Evaluation batch finished" line, so this comparison
can be repeated from the normal log without a drill.

## Chosen defaults

| Setting | Value |
|---|---|
| NIM model | `nvidia/nemotron-3-ultra-550b-a55b` |
| `NVIDIA_REASONING_EFFORT` | `low` |
| `NVIDIA_TEMPERATURE` | `0.6` (never 0) |
| `NVIDIA_MAX_TOKENS` | `4096` |
| `NVIDIA_TIMEOUT_SECONDS` (read) | `240` |
| Streaming | off |
| `EVAL_MAX_WORKERS` | `8` (mean-latency sizing; 8 parallel ran clean) |
| `NVIDIA_RPM_LIMIT` | `30` (assumed conservative client-side cap; not measured. The user-reported provider limit is about 40 rpm and no rate-limit headers were observed) |
| `OPENROUTER_MODELS` (ordered) | `nvidia/nemotron-3.5-super-vl:free`, `google/gemma-4-31b-it:free` (unverified), `google/gemma-4-26b-a4b-it:free` (unverified) |

Only the first OpenRouter entry met the qualification rule, and it was tested on only 4 of the 14 fixtures
(3 single-model calls plus 1 via the `models` array). The Gemma entries are untested because of upstream 429s
and are included only as lower-confidence alternates. Client-side validation must still reject unparseable
output and move on.

Run settings that differ from the defaults above: OpenRouter bake-off calls used temperature 0.3 (the script
default) while NIM used 0.6. `NVIDIA_MAX_TOKENS=4096` was not itself exercised: the NIM runs used
`max_tokens=8192`, and the observed maximum completion was about 2.1k tokens (OpenRouter runs used 4096).

## Caveats

- Fixture bands apply to the neutral persona only. Results with a stricter local persona will differ;
  tune with local cases.
- Small samples: NIM `medium`, `enable_thinking=false` and streaming were each tried on 2-6 calls, and
  each OpenRouter candidate on 3-4. They show feasibility, not statistically solid quality gaps.
- Concurrency tolerance and the free pool both change without notice; re-run `scripts/bakeoff.py`
  (`nim`, `list-free`, `openrouter`) when behaviour changes. `list-free` costs nothing.
- The NIM service is a trial offering with undocumented limits; keep the fallback chain and backoff.
