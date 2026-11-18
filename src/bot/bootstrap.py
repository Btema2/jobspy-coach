"""``jobspy-coach init``: credentials, starter profile, pairing code and daemon start.

Secrets are only ever read through ``Prompter.secret`` (hidden input), written through ``merge_env_file`` and
sent to their own provider for one validation call. They are never printed, logged, put in a label or passed
as an argument; validator details are fixed strings plus the exception type name.
"""
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Protocol, Tuple

import httpx
import typer

from src.bot.auth import PAIRING_TTL_SECONDS, adopt_chat_id, format_code, issue_pairing_code
from src.bot.client import (
    TelegramApiError,
    TelegramBotClient,
    TelegramNetworkError,
    TelegramUnauthorized,
)
from src.bot.state import BotStateStore
from src.config import Settings
from src.envfile import merge_env_file, read_env_value, validate_env_value
from src.profile.errors import ProfileError, ProfileValidationError, VersionConflict
from src.profile.ops import create_starter_profile
from src.profile.store import ProfileStore

# Keep identical to OPENROUTER_MODELS in .env.example (a test compares them).
DEFAULT_OPENROUTER_MODELS = (
    "nvidia/nemotron-3.5-super-vl:free,google/gemma-4-31b-it:free,google/gemma-4-26b-a4b-it:free"
)
MAX_ATTEMPTS = 3
_USERNAME_RE = re.compile(r"[A-Za-z0-9_]{1,64}")


class Prompter(Protocol):
    def secret(self, label: str) -> str: ...

    def confirm(self, label: str, default: bool) -> bool: ...


class TyperPrompter:
    def secret(self, label: str) -> str:
        return typer.prompt(label, hide_input=True, default="", show_default=False)

    def confirm(self, label: str, default: bool) -> bool:
        return typer.confirm(label, default=default)


@dataclass(frozen=True)
class CheckResult:
    ok: Optional[bool]  # True valid, False rejected, None not verifiable
    detail: str  # never contains the secret


@dataclass(frozen=True)
class Validators:
    telegram: Callable[[str], CheckResult]
    nim: Callable[[str], CheckResult]
    openrouter: Callable[[str], CheckResult]


def _status_result(name: str, status: int) -> CheckResult:
    if 200 <= status < 300:
        return CheckResult(True, "valid")
    if status in (401, 403):
        return CheckResult(False, f"{name} rejected the key (HTTP {status})")
    return CheckResult(None, f"{name} answered HTTP {status}; the key could not be checked")


def http_validators(settings: Settings, transport: Optional[httpx.BaseTransport] = None) -> Validators:
    def telegram(token: str) -> CheckResult:
        client = TelegramBotClient(token, transport=transport, timeout=15.0)
        try:
            me = client.get_me()
        except TelegramUnauthorized:
            return CheckResult(False, "Telegram rejected the token (HTTP 401)")
        except TelegramNetworkError as exc:
            return CheckResult(None, f"could not reach Telegram ({exc.description})")
        except TelegramApiError as exc:
            if exc.status == 404:  # Telegram answers a malformed token with 404
                return CheckResult(False, "Telegram rejected the token (HTTP 404)")
            return CheckResult(None, f"Telegram answered HTTP {exc.status}; the token could not be checked")
        finally:
            client.close()
        username = me.get("username") if isinstance(me, dict) else None
        if isinstance(username, str) and _USERNAME_RE.fullmatch(username):
            return CheckResult(True, f"@{username}")
        return CheckResult(True, "valid")

    def _request(name: str, method: str, url: str, key: str, timeout: float, body: Optional[dict]) -> CheckResult:
        try:
            with httpx.Client(transport=transport, timeout=timeout) as client:
                response = client.request(method, url, headers={"Authorization": f"Bearer {key}"}, json=body)
        except httpx.HTTPError as exc:  # no chaining or str(): httpx text embeds the URL and headers
            return CheckResult(None, f"could not reach {name} ({type(exc).__name__})")
        return _status_result(name, response.status_code)

    def nim(key: str) -> CheckResult:
        body = {
            "model": settings.nvidia_model,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 16,
            "temperature": settings.nvidia_temperature,
        }
        url = settings.nvidia_base_url.rstrip("/") + "/chat/completions"
        return _request("NVIDIA", "POST", url, key, 60.0, body)

    def openrouter(key: str) -> CheckResult:
        url = settings.openrouter_base_url.rstrip("/") + "/key"
        return _request("OpenRouter", "GET", url, key, 30.0, None)

    return Validators(telegram=telegram, nim=nim, openrouter=openrouter)


@dataclass(frozen=True)
class InitOptions:
    skip_credentials: bool = False
    start: Optional[bool] = None
    language: str = "en"


_CREDENTIALS = (
    ("TELEGRAM_BOT_TOKEN", "telegram", "telegram_bot_token"),
    ("NVIDIA_API_KEY", "nim", "nvidia_api_key"),
    ("OPENROUTER_API_KEY", "openrouter", "openrouter_api_key"),
)


class _CredentialsFailed(Exception):
    pass


def _check(validate: Callable[[str], CheckResult], value: str, name: str, out: Callable[[str], None]) -> bool:
    """True when ``value`` may be saved (valid or not verifiable); False when it was rejected."""
    try:
        validate_env_value(value)
    except ValueError:
        out(f"{name}: the value contains a space, quote, '#', '$', backslash or control character; not accepted.")
        return False
    result = validate(value)
    if result.ok is False:
        out(f"{name}: {result.detail}.")
        return False
    if result.ok is None:
        out(f"Warning: {name}: {result.detail}; saved without verification.")
    else:
        out(f"{name}: ok ({result.detail}).")
    return True


def _collect_credentials(
    env_path: Path,
    settings: Settings,
    prompter: Prompter,
    validators: Validators,
    out: Callable[[str], None],
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Returns (values to write, effective values by name). Raises ``_CredentialsFailed`` after 3 rejections."""
    updates: Dict[str, str] = {}
    effective: Dict[str, str] = {}
    for name, validator_name, settings_field in _CREDENTIALS:
        validate = getattr(validators, validator_name)
        existing = read_env_value(env_path, name) or getattr(settings, settings_field) or ""
        if existing and prompter.confirm(f"Keep the existing {name}?", True):
            if _check(validate, existing, name, out):
                effective[name] = existing
                continue
            out(f"{name}: the existing value does not work; enter a new one.")
        for _ in range(MAX_ATTEMPTS):
            value = prompter.secret(f"{name} (input hidden, press Enter to skip)").strip()
            if not value:
                break
            if _check(validate, value, name, out):
                updates[name] = value
                effective[name] = value
                break
        else:
            raise _CredentialsFailed(name)
    return updates, effective


def _store_credentials(
    env_path: Path, updates: Dict[str, str], effective: Dict[str, str], out: Callable[[str], None]
) -> None:
    updates = dict(updates)
    if effective.get("OPENROUTER_API_KEY") and not read_env_value(env_path, "OPENROUTER_MODELS"):
        updates["OPENROUTER_MODELS"] = DEFAULT_OPENROUTER_MODELS
    if updates:
        merge_env_file(env_path, updates)
        out("Saved credentials to .env (mode 600).")
    elif env_path.exists():
        os.chmod(env_path, 0o600)


def _profile_step(
    store: ProfileStore, settings: Settings, options: InitOptions, out: Callable[[str], None]
) -> bool:
    existing = store.read_snapshot()
    if existing is not None:
        out(f"Profile exists (version {existing.version}).")
        return True
    try:
        result = create_starter_profile(
            store,
            wow_threshold=settings.wow_score_threshold,
            digest_threshold=settings.digest_score_threshold,
            output_language=options.language,
        )
    except VersionConflict:  # a concurrent `init` (or import) won the race
        current = store.read_snapshot()
        out(f"Profile exists (version {current.version if current else '?'}).")
        return True
    except ProfileError as exc:
        for message in exc.messages if isinstance(exc, ProfileValidationError) else [str(exc)]:
            out(f"- {message}")
        return False
    out(f"Created an empty starter profile (version {result.version}); it is paused.")
    for note in result.warnings:
        out(f"- {note}")
    out(
        "Fill it in: send /setup to the bot, or run `jobspy-coach profile export`, edit the YAML file, "
        "then run `jobspy-coach profile import <file>`."
    )
    return True


def _pairing_step(
    env_path: Path,
    state: BotStateStore,
    settings: Settings,
    clock: Callable[[], float],
    out: Callable[[str], None],
) -> bool:
    """Returns True when a pairing code was issued."""
    if state.get_pairing() is not None:
        out("Already paired.")
        return False
    chat_id = read_env_value(env_path, "TELEGRAM_CHAT_ID") or settings.telegram_chat_id
    if adopt_chat_id(state, chat_id):
        out("Adopted TELEGRAM_CHAT_ID as the paired chat.")
        return False
    if chat_id and chat_id.strip():
        out("TELEGRAM_CHAT_ID is set but is not a user id the bot can adopt; pairing will replace it.")
    code = issue_pairing_code(state, clock())
    minutes = PAIRING_TTL_SECONDS // 60
    out(
        f"Send /start {format_code(code)} to the bot in a private chat within {minutes} minutes "
        "(the daemon must be running)."
    )
    return True


def run_init(
    *,
    env_path: Path,
    store: ProfileStore,
    state: BotStateStore,
    settings: Settings,
    prompter: Prompter,
    validators: Validators,
    options: InitOptions,
    out: Callable[[str], None],
    start_daemon: Callable[[], None],
    clock: Callable[[], float] = time.time,
) -> int:
    effective: Dict[str, str] = {}
    if not options.skip_credentials:
        try:
            updates, effective = _collect_credentials(env_path, settings, prompter, validators, out)
        except _CredentialsFailed as failed:
            out(f"{failed.args[0]} was rejected three times; nothing was written.")
            return 1
        try:
            _store_credentials(env_path, updates, effective, out)
        except (OSError, ValueError) as exc:
            out(f"Could not write .env ({type(exc).__name__}); nothing was changed.")
            return 1
        if not (effective.get("NVIDIA_API_KEY") or effective.get("OPENROUTER_API_KEY")):
            out(
                "Warning: no LLM key configured (NVIDIA_API_KEY or OPENROUTER_API_KEY); "
                "evaluation cannot run until one is set."
            )

    if not _profile_step(store, settings, options, out):
        return 1

    token = (
        effective.get("TELEGRAM_BOT_TOKEN")
        or read_env_value(env_path, "TELEGRAM_BOT_TOKEN")
        or settings.telegram_bot_token
    )
    code_issued = _pairing_step(env_path, state, settings, clock, out) if token else False

    start = options.start
    if start is None:
        start = prompter.confirm("Start the daemon now?", code_issued)
    if start:
        start_daemon()
    return 0
