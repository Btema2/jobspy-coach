#!/usr/bin/env python3
"""Model bake-off for the evaluator: NVIDIA NIM and OpenRouter free models.

Runs the generic evaluation fixtures (tests/fixtures/eval_cases.json) through a
chat-completions endpoint using the real system prompt (build_system_prompt) and
user prompt builder, then records latency, finish_reason, token usage, JSON
validity and whether the score falls inside the expected band.

API keys are read from the environment or a local ``.env`` file and are never
printed or stored. Raw per-call results (no keys, no job text) are written under
the git-ignored ``local/bakeoff/`` directory.

Examples:
    python scripts/bakeoff.py nim --effort low --parallel 12
    python scripts/bakeoff.py nim --effort medium --parallel 6 --cases 3
    python scripts/bakeoff.py nim --effort none --cases 3          # enable_thinking=false probe
    python scripts/bakeoff.py nim --effort low --stream --cases 2
    python scripts/bakeoff.py list-free
    python scripts/bakeoff.py openrouter --candidates vendor/model:free --cases 4
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FIXTURE_PATH = ROOT / "tests" / "fixtures" / "eval_cases.json"
OUT_DIR = ROOT / "local" / "bakeoff"

NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
NIM_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

TRAP_MAX_SCORE = 2.0
BACKUP_TRAP_MAX_SCORE = 3.0
POSITIVE_BAND_SHARE = 0.8

_THINK_BLOCK = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)
_THINK_TAIL = re.compile(r"^[\s\S]*?</think>", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# Pure helpers (unit tested)
# --------------------------------------------------------------------------- #
def strip_think(text: str) -> str:
    """Remove <think>...</think> blocks (and an unopened leading reasoning tail)."""
    if not text:
        return ""
    cleaned = _THINK_BLOCK.sub("", text)
    if "</think>" in cleaned.lower():
        cleaned = _THINK_TAIL.sub("", cleaned, count=1)
    return cleaned.strip()


def score_in_band(score: Optional[float], expected: Dict[str, float]) -> bool:
    if score is None:
        return False
    return expected["min"] <= score <= expected["max"]


def percentile(values: List[float], pct: float) -> Optional[float]:
    """Nearest-rank percentile; None for empty input."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100.0 * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def parse_model_output(content: Optional[str]) -> Dict[str, Any]:
    """Validate raw model text against the evaluation schema."""
    from src.evaluator.parsing import extract_json_payload
    from src.evaluator.schemas import JobEvaluationResult

    raw = content or ""
    info: Dict[str, Any] = {
        "valid_json": False,
        "score": None,
        "hard_reject": None,
        "role_is_direct_coaching": None,
        "matches_bonus_criteria": None,
        "had_fence": "```" in raw,
        "had_think": "</think>" in raw.lower() or "<think>" in raw.lower(),
        "content_chars": len(raw),
        "parse_error": None,
    }
    if not raw.strip():
        info["parse_error"] = "empty content"
        return info
    try:
        payload = extract_json_payload(strip_think(raw))
        result = JobEvaluationResult.model_validate(payload)
    except Exception as exc:  # noqa: BLE001 - record any parse/validation failure
        info["parse_error"] = type(exc).__name__
        return info
    info.update(
        valid_json=True,
        score=float(result.score),
        hard_reject=result.hard_reject,
        role_is_direct_coaching=result.role_is_direct_coaching,
        matches_bonus_criteria=result.matches_bonus_criteria,
    )
    return info


def summarize(records: Iterable[Dict[str, Any]], trap_limit: float = TRAP_MAX_SCORE) -> Dict[str, Any]:
    """Aggregate per-call records into decision-rule metrics."""
    recs = list(records)
    ok = [r for r in recs if r.get("status") == 200]
    traps = [r for r in recs if r.get("category") == "hard_reject"]
    positives = [r for r in recs if r.get("category") != "hard_reject"]
    trap_scores = [r["score"] for r in traps if r.get("valid_json") and r.get("score") is not None]
    trap_failures = [
        r["case_id"]
        for r in traps
        if not (r.get("valid_json") and r.get("score") is not None and r["score"] <= trap_limit)
    ]
    positives_in_band = [r for r in positives if r.get("valid_json") and r.get("in_band")]
    latencies = [r["latency_s"] for r in ok if r.get("latency_s") is not None]
    finish: Dict[str, int] = {}
    for r in recs:
        key = str(r.get("finish_reason"))
        finish[key] = finish.get(key, 0) + 1
    return {
        "calls": len(recs),
        "http_ok": len(ok),
        "valid_json": sum(1 for r in recs if r.get("valid_json")),
        "traps": len(traps),
        "trap_max_score": max(trap_scores) if trap_scores else None,
        "trap_failures": trap_failures,
        "positives": len(positives),
        "positives_in_band": len(positives_in_band),
        "positive_share": (len(positives_in_band) / len(positives)) if positives else None,
        "latency_mean_s": statistics.fmean(latencies) if latencies else None,
        "latency_p50_s": percentile(latencies, 50),
        "latency_p95_s": percentile(latencies, 95),
        "latency_max_s": max(latencies) if latencies else None,
        "finish_reasons": finish,
        "completion_tokens": sum(r.get("completion_tokens") or 0 for r in recs),
        "reasoning_tokens": sum(r.get("reasoning_tokens") or 0 for r in recs),
        "prompt_tokens": sum(r.get("prompt_tokens") or 0 for r in recs),
        "fenced": sum(1 for r in recs if r.get("had_fence")),
        "with_think_tags": sum(1 for r in recs if r.get("had_think")),
    }


def passes_nim_rule(summary: Dict[str, Any]) -> bool:
    """All traps <= 2.0 (valid JSON) and >= 80% of positives in band with valid JSON."""
    if summary["traps"] and summary["trap_failures"]:
        return False
    share = summary["positive_share"]
    return share is not None and share >= POSITIVE_BAND_SHARE


def backup_qualifies(summary: Dict[str, Any]) -> bool:
    """Valid JSON on every call and every trap <= 3.0."""
    if summary["calls"] == 0 or summary["valid_json"] != summary["calls"]:
        return False
    return not summary_trap_failures_over(summary, BACKUP_TRAP_MAX_SCORE)


def summary_trap_failures_over(summary: Dict[str, Any], limit: float) -> bool:
    max_score = summary.get("trap_max_score")
    return max_score is not None and max_score > limit


def workers_for(rpm: float, p95_latency_s: float, margin: int = 2) -> int:
    """In-flight requests = rate x latency; add a small margin."""
    return max(1, math.ceil(rpm / 60.0 * p95_latency_s) + margin)


def is_excluded_free_model(model: Dict[str, Any]) -> Optional[str]:
    """Return a reason string if a free model must not be used for scoring, else None."""
    mid = (model.get("id") or "").lower()
    name = (model.get("name") or "").lower()
    blob = f"{mid} {name}"
    if model.get("expiration_date"):
        return "has expiration_date"
    if mid == "openrouter/free" or mid.startswith("openrouter/"):
        return "router"
    for token in ("guard", "safety", "shield", "classif", "moderat"):
        if token in blob:
            return "content-safety"
    if re.search(r"(^|[^a-z])(code|coder|codex|devstral)([^a-z]|$)", blob):
        return "code"
    for token in ("audio", "tts", "whisper", "speech", "voice", "embed", "rerank"):
        if token in blob:
            return "audio/embedding"
    if "stealth" in blob:
        return "stealth"
    arch = model.get("architecture") or {}
    out_mods = arch.get("output_modalities") or ["text"]
    in_mods = arch.get("input_modalities") or ["text"]
    if out_mods != ["text"] or "text" not in in_mods:
        return "non-text output"
    size = re.search(r"(?<![a-z0-9.])(\d+(?:\.\d+)?)b(?![a-z])", blob)
    if size and float(size.group(1)) < 12:
        return "tiny"
    if (model.get("context_length") or 0) < 16000:
        return "small context"
    return None


# --------------------------------------------------------------------------- #
# Environment / fixtures
# --------------------------------------------------------------------------- #
def load_env_value(name: str, env_file: Path = ROOT / ".env") -> Optional[str]:
    """Read a variable from the process environment, then from a local .env file."""
    value = os.environ.get(name)
    if value:
        return value
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, raw = line.partition("=")
            if key.strip() == name:
                return raw.strip().strip('"').strip("'") or None
    return None


def load_cases(path: Path = FIXTURE_PATH, limit: Optional[int] = None, ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    cases = json.loads(path.read_text(encoding="utf-8"))
    if ids:
        cases = [c for c in cases if c["id"] in ids]
    elif limit:
        # Balanced subset: alternate traps and positives, traps first.
        traps = [c for c in cases if c["category"] == "hard_reject"]
        pos = [c for c in cases if c["category"] != "hard_reject"]
        mixed: List[Dict[str, Any]] = []
        for i in range(max(len(traps), len(pos))):
            if i < len(traps):
                mixed.append(traps[i])
            if i < len(pos):
                mixed.append(pos[i])
        cases = mixed[:limit]
    return cases


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
def _interesting_headers(headers: Any) -> Dict[str, str]:
    out = {}
    for key, value in headers.items():
        lk = key.lower()
        if lk.startswith("x-ratelimit") or lk in {"retry-after", "x-request-id", "nvcf-reqid"} or "limit" in lk or "remaining" in lk or "reset" in lk:
            out[lk] = value
    return out


def build_messages(case: Dict[str, Any], persona: str = "generic") -> List[Dict[str, str]]:
    """Build the system+user messages.

    persona="generic" uses the neutral built-in persona (matches the persona-neutral fixture bands);
    persona="local" injects the git-ignored local candidate profile, as the daemon does at runtime.
    """
    from src.evaluator.prompts import DEFAULT_CANDIDATE_PERSONA, build_job_user_prompt, build_system_prompt

    system = build_system_prompt(DEFAULT_CANDIDATE_PERSONA if persona == "generic" else None)
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": build_job_user_prompt(
                case["title"], case["company"], case["location"], bool(case.get("is_remote")), case["description"]
            ),
        },
    ]


def call_chat(
    client: Any,
    url: str,
    headers: Dict[str, str],
    body: Dict[str, Any],
    stream: bool = False,
) -> Dict[str, Any]:
    """POST one chat completion and normalise the outcome (never raises on HTTP errors)."""
    import httpx

    start = time.monotonic()
    out: Dict[str, Any] = {
        "status": None,
        "content": None,
        "model": None,
        "finish_reason": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "reasoning_tokens": None,
        "reasoning_chars": 0,
        "error_shape": None,
        "rate_headers": {},
        "header_names": [],
        "first_token_s": None,
    }
    try:
        if stream:
            payload = dict(body, stream=True, stream_options={"include_usage": True})
            parts: List[str] = []
            reasoning_chars = 0
            with client.stream("POST", url, headers=headers, json=payload) as resp:
                out["status"] = resp.status_code
                out["rate_headers"] = _interesting_headers(resp.headers)
                out["header_names"] = sorted(resp.headers.keys())
                if resp.status_code != 200:
                    raw = resp.read().decode("utf-8", "replace")
                    out["error_shape"] = _error_shape(raw)
                else:
                    for line in resp.iter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        out["model"] = chunk.get("model") or out["model"]
                        usage = chunk.get("usage")
                        if usage:
                            _apply_usage(out, usage)
                        for choice in chunk.get("choices") or []:
                            delta = choice.get("delta") or {}
                            piece = delta.get("content")
                            if piece:
                                if out["first_token_s"] is None:
                                    out["first_token_s"] = round(time.monotonic() - start, 2)
                                parts.append(piece)
                            reasoning_chars += len(delta.get("reasoning_content") or delta.get("reasoning") or "")
                            if choice.get("finish_reason"):
                                out["finish_reason"] = choice["finish_reason"]
                    out["content"] = "".join(parts)
                    out["reasoning_chars"] = reasoning_chars
        else:
            resp = client.post(url, headers=headers, json=body)
            out["status"] = resp.status_code
            out["rate_headers"] = _interesting_headers(resp.headers)
            out["header_names"] = sorted(resp.headers.keys())
            if resp.status_code != 200:
                out["error_shape"] = _error_shape(resp.text)
            else:
                data = resp.json()
                out["model"] = data.get("model")
                choice = (data.get("choices") or [{}])[0]
                message = choice.get("message") or {}
                out["content"] = message.get("content")
                out["finish_reason"] = choice.get("finish_reason")
                out["reasoning_chars"] = len(message.get("reasoning_content") or message.get("reasoning") or "")
                if data.get("usage"):
                    _apply_usage(out, data["usage"])
    except httpx.HTTPError as exc:
        out["status"] = out["status"] or 0
        out["error_shape"] = {"exception": type(exc).__name__}
    out["latency_s"] = round(time.monotonic() - start, 2)
    return out


def _apply_usage(out: Dict[str, Any], usage: Dict[str, Any]) -> None:
    out["prompt_tokens"] = usage.get("prompt_tokens")
    out["completion_tokens"] = usage.get("completion_tokens")
    details = usage.get("completion_tokens_details") or {}
    out["reasoning_tokens"] = details.get("reasoning_tokens") or usage.get("reasoning_tokens")


def _error_shape(raw: str) -> Dict[str, Any]:
    """Describe an error body (keys + truncated message) without echoing request content."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {"non_json_body_chars": len(raw), "head": raw[:160]}
    shape: Dict[str, Any] = {"keys": sorted(data.keys()) if isinstance(data, dict) else type(data).__name__}
    err = data.get("error") if isinstance(data, dict) else None
    if isinstance(err, dict):
        shape["error_keys"] = sorted(err.keys())
        shape["message"] = str(err.get("message", ""))[:200]
        shape["code"] = err.get("code")
        meta = err.get("metadata")
        if isinstance(meta, dict):
            shape["metadata_keys"] = sorted(meta.keys())
            shape["provider_name"] = meta.get("provider_name")
            shape["raw"] = str(meta.get("raw", ""))[:160]
    elif isinstance(data, dict):
        shape["message"] = str(data.get("detail") or data.get("message") or "")[:200]
    return shape


def make_record(case: Dict[str, Any], outcome: Dict[str, Any], **extra: Any) -> Dict[str, Any]:
    parsed = parse_model_output(outcome.get("content"))
    record = {
        "case_id": case["id"],
        "category": case["category"],
        "expected": case["expected"],
        "in_band": score_in_band(parsed["score"], case["expected"]),
        **{k: v for k, v in outcome.items() if k != "content"},
        **parsed,
        **extra,
    }
    return record


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def print_table(records: List[Dict[str, Any]]) -> None:
    header = f"{'case':<42}{'cat':<12}{'http':>5}{'lat_s':>8}{'finish':>9}{'comp':>7}{'reas':>7}{'json':>6}{'score':>7}{'band':>6}"
    print(header)
    print("-" * len(header))
    for r in records:
        score = "-" if r.get("score") is None else f"{r['score']:.1f}"
        print(
            f"{r['case_id'][:41]:<42}{r['category'][:11]:<12}{str(r.get('status')):>5}"
            f"{(r.get('latency_s') or 0):>8.1f}{str(r.get('finish_reason'))[:8]:>9}"
            f"{str(r.get('completion_tokens') or '-'):>7}{str(r.get('reasoning_tokens') or '-'):>7}"
            f"{'yes' if r.get('valid_json') else 'NO':>6}{score:>7}{'ok' if r.get('in_band') else 'MISS':>6}"
        )


def print_summary(label: str, summary: Dict[str, Any]) -> None:
    def fmt(v: Any) -> str:
        return "-" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))

    print(f"\n== {label} ==")
    for key in (
        "calls", "http_ok", "valid_json", "trap_max_score", "trap_failures", "positives_in_band", "positives",
        "positive_share", "latency_mean_s", "latency_p50_s", "latency_p95_s", "latency_max_s",
        "finish_reasons", "prompt_tokens", "completion_tokens", "reasoning_tokens", "fenced", "with_think_tags",
    ):
        print(f"  {key:<20}{fmt(summary[key])}")
    print(f"  passes_nim_rule     {passes_nim_rule(summary)}")
    print(f"  backup_qualifies    {backup_qualifies(summary)}")


def save_results(tag: str, config: Dict[str, Any], records: List[Dict[str, Any]], summary: Dict[str, Any]) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = OUT_DIR / f"{tag}-{stamp}.json"
    path.write_text(json.dumps({"config": config, "summary": summary, "records": records}, indent=2), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Phases
# --------------------------------------------------------------------------- #
def run_nim(args: argparse.Namespace) -> int:
    import httpx

    key = load_env_value("NVIDIA_API_KEY")
    if not key:
        print("NVIDIA_API_KEY is not set (environment or .env)", file=sys.stderr)
        return 2
    cases = load_cases(limit=args.cases, ids=args.case_ids)
    body_base: Dict[str, Any] = {
        "model": args.model,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
    }
    if args.effort in {"low", "medium", "high"}:
        body_base["reasoning_effort"] = args.effort
    elif args.effort == "none":
        body_base["chat_template_kwargs"] = {"enable_thinking": False}
    # Deliberately no response_format / thinking_token_budget for NIM.
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    url = f"{args.base_url.rstrip('/')}/chat/completions"
    timeout = httpx.Timeout(connect=10, read=args.timeout, write=30, pool=30)
    limits = httpx.Limits(max_connections=args.parallel + 4)
    label = f"nim persona={args.persona} effort={args.effort} temp={args.temperature} stream={args.stream} parallel={args.parallel}"
    print(f"Running {len(cases)} NIM calls: {label}")

    with httpx.Client(timeout=timeout, limits=limits) as client:

        def one(case: Dict[str, Any]) -> Dict[str, Any]:
            body = dict(body_base, messages=build_messages(case, args.persona))
            outcome = call_chat(client, url, headers, body, stream=args.stream)
            return make_record(case, outcome, effort=args.effort, temperature=args.temperature, stream=args.stream)

        wall = time.monotonic()
        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            records = list(pool.map(one, cases))
        wall = round(time.monotonic() - wall, 1)

    summary = summarize(records)
    summary["wall_s"] = wall
    print_table(records)
    print_summary(label, summary)
    print(f"  wall_s              {wall}")
    seen_status = sorted({r["status"] for r in records})
    rate = {k: v for r in records for k, v in r.get("rate_headers", {}).items()}
    print(f"  http statuses       {seen_status}")
    print(f"  rate/limit headers  {json.dumps(rate) if rate else 'none seen'}")
    errors = [r["error_shape"] for r in records if r.get("error_shape")]
    if errors:
        print(f"  error shapes        {json.dumps(errors[:3])}")
    if records:
        print(f"  response header names: {records[0].get('header_names')}")
    config = {k: v for k, v in vars(args).items() if k not in {"func"}}
    path = save_results(f"nim-{args.persona}-{args.effort}{'-stream' if args.stream else ''}", config, records, summary)
    print(f"Saved {path.relative_to(ROOT)}")
    return 0


def fetch_free_models(client: Any) -> List[Dict[str, Any]]:
    resp = client.get(f"{OPENROUTER_BASE_URL}/models")
    resp.raise_for_status()
    free = []
    for m in resp.json().get("data", []):
        pricing = m.get("pricing") or {}
        try:
            is_free = float(pricing.get("prompt", 1)) == 0 and float(pricing.get("completion", 1)) == 0
        except (TypeError, ValueError):
            is_free = False
        if is_free:
            free.append(m)
    return free


def run_list_free(args: argparse.Namespace) -> int:
    import httpx

    with httpx.Client(timeout=30) as client:
        free = fetch_free_models(client)
    kept, dropped = [], []
    for m in free:
        reason = is_excluded_free_model(m)
        (dropped if reason else kept).append((m, reason))
    print(f"{len(free)} free models; {len(kept)} pass exclusions (no API key used)\n")
    print("KEPT (id | context | response_format | structured_outputs | reasoning)")
    for m, _ in sorted(kept, key=lambda x: -(x[0].get("context_length") or 0)):
        sp = set(m.get("supported_parameters") or [])
        print(
            f"  {m['id']:<58}{m.get('context_length'):>9}  "
            f"{'RF' if 'response_format' in sp else '--'}  {'SO' if 'structured_outputs' in sp else '--'}  "
            f"{'R' if ('reasoning' in sp or 'include_reasoning' in sp) else '-'}"
        )
    print("\nEXCLUDED")
    for m, reason in dropped:
        print(f"  {m['id']:<58}{reason}")
    return 0


def run_openrouter(args: argparse.Namespace) -> int:
    import httpx

    key = load_env_value("OPENROUTER_API_KEY")
    if not key:
        print("OPENROUTER_API_KEY is not set (environment or .env)", file=sys.stderr)
        return 2
    cases = load_cases(limit=args.cases, ids=args.case_ids)
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    url = f"{OPENROUTER_BASE_URL}/chat/completions"
    candidates = [c for c in args.candidates.split(",") if c]
    total = len(cases) * len(candidates) if not args.models_array else len(cases)
    print(f"Running {total} OpenRouter calls sequentially (20 rpm cap, 50/day free)")
    all_records: List[Dict[str, Any]] = []
    timeout = httpx.Timeout(connect=10, read=args.timeout, write=30, pool=30)
    with httpx.Client(timeout=timeout) as client:
        if args.models_array:
            runs = [("models-array", candidates)]
        else:
            runs = [(c, [c]) for c in candidates]
        for label, chain in runs:
            records = []
            for case in cases:
                body: Dict[str, Any] = {
                    "temperature": args.temperature,
                    "max_tokens": args.max_tokens,
                    "messages": build_messages(case, args.persona),
                    "provider": {"require_parameters": True},
                }
                if args.models_array:
                    body["model"] = chain[0]
                    body["models"] = chain
                else:
                    body["model"] = chain[0]
                if not args.no_response_format:
                    body["response_format"] = {"type": "json_object"}
                outcome = call_chat(client, url, headers, body)
                records.append(make_record(case, outcome, candidate=label, requested_models=chain))
                time.sleep(3.2)  # stay under 20 rpm
            summary = summarize(records, trap_limit=BACKUP_TRAP_MAX_SCORE)
            print_table(records)
            print_summary(f"openrouter {label}", summary)
            returned = sorted({str(r.get("model")) for r in records})
            print(f"  returned models     {returned}")
            errors = [r["error_shape"] for r in records if r.get("error_shape")]
            if errors:
                print(f"  error shapes        {json.dumps(errors[:2])}")
            rate = {k: v for r in records for k, v in r.get("rate_headers", {}).items()}
            print(f"  rate/limit headers  {json.dumps(rate) if rate else 'none seen'}")
            all_records.extend(records)
            config = {k: v for k, v in vars(args).items() if k != "func"}
            path = save_results(f"openrouter-{re.sub(r'[^a-z0-9]+', '_', label.lower())}", config, records, summary)
            print(f"Saved {path.relative_to(ROOT)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="phase", required=True)

    nim = sub.add_parser("nim", help="bake-off against NVIDIA NIM")
    nim.add_argument("--model", default=NIM_MODEL)
    nim.add_argument("--base-url", default=NIM_BASE_URL)
    nim.add_argument("--effort", choices=["low", "medium", "high", "none"], default="low",
                     help="reasoning_effort; 'none' sends chat_template_kwargs.enable_thinking=false instead")
    nim.add_argument("--temperature", type=float, default=0.6)
    nim.add_argument("--top-p", type=float, default=0.95)
    nim.add_argument("--max-tokens", type=int, default=8192)
    nim.add_argument("--parallel", type=int, default=4)
    nim.add_argument("--stream", action="store_true")
    nim.add_argument("--persona", choices=["generic", "local"], default="generic")
    nim.add_argument("--timeout", type=float, default=240.0, help="read timeout in seconds")
    nim.add_argument("--cases", type=int, default=None, help="balanced subset size (traps first)")
    nim.add_argument("--case-ids", nargs="*", default=None)
    nim.set_defaults(func=run_nim)

    lst = sub.add_parser("list-free", help="list OpenRouter free models after exclusions (no key, no cost)")
    lst.set_defaults(func=run_list_free)

    orr = sub.add_parser("openrouter", help="bake-off against OpenRouter free models")
    orr.add_argument("--candidates", required=True, help="comma-separated model ids")
    orr.add_argument("--models-array", action="store_true",
                     help="send all candidates as one ordered `models` fallback array instead of separately")
    orr.add_argument("--no-response-format", action="store_true")
    orr.add_argument("--persona", choices=["generic", "local"], default="generic")
    orr.add_argument("--temperature", type=float, default=0.3)
    orr.add_argument("--max-tokens", type=int, default=4096)
    orr.add_argument("--timeout", type=float, default=120.0)
    orr.add_argument("--cases", type=int, default=4)
    orr.add_argument("--case-ids", nargs="*", default=None)
    orr.set_defaults(func=run_openrouter)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
