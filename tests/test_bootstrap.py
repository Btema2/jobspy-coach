"""``run_init``: hidden credential prompts, .env merge, starter profile, pairing code, daemon start."""
import json
import os
import re
import stat
from pathlib import Path

import httpx
import pytest

from src.bot.auth import PAIRING_TTL_SECONDS, hash_code
from src.bot.bootstrap import (
    DEFAULT_OPENROUTER_MODELS,
    CheckResult,
    InitOptions,
    Validators,
    http_validators,
    run_init,
)
from src.bot.state import BotStateStore, Pairing
from src.config import Settings
from src.envfile import read_env_value
from src.profile.ops import ReplaceProfile, apply_changes
from src.profile.snapshot import ProfileFields, ProfileSnapshot
from src.profile.store import ProfileStore
from src.storage.db import Database

NOW = 1_700_000_000.0
BOT = "111111:<BOT-TOKEN>"
NIM = "nvapi-<NIM-KEY>"
OR = "sk-or-<OR-KEY>"


class ScriptedPrompter:
    def __init__(self, secrets=(), confirms=()):
        self.secrets = list(secrets)
        self.confirms = list(confirms)
        self.secret_labels = []
        self.confirm_labels = []

    def secret(self, label):
        self.secret_labels.append(label)
        return self.secrets.pop(0) if self.secrets else ""

    def confirm(self, label, default):
        self.confirm_labels.append((label, default))
        return self.confirms.pop(0) if self.confirms else default


class FakeValidators:
    def __init__(self, results=None):
        self.calls = []
        self.results = results or {}

    def _check(self, name):
        def check(value):
            self.calls.append((name, value))
            queue = self.results.get(name)
            if queue:
                return queue.pop(0)
            return CheckResult(True, f"@fake_bot" if name == "telegram" else "ok")

        return check

    def validators(self):
        return Validators(self._check("telegram"), self._check("nim"), self._check("openrouter"))


class Env:
    def __init__(self, tmp_path):
        self.path = tmp_path / ".env"
        self.db = Database(tmp_path / "init.db")
        self.db.init_db()
        self.store = ProfileStore(self.db)
        self.state = BotStateStore(self.db)
        self.lines = []
        self.started = []

    def run(self, prompter, validators=None, options=None, settings=None):
        validators = validators or FakeValidators()
        return run_init(
            env_path=self.path,
            store=self.store,
            state=self.state,
            settings=settings or Settings(_env_file=None),
            prompter=prompter,
            validators=validators.validators(),
            options=options or InitOptions(start=False),
            out=self.lines.append,
            start_daemon=lambda: self.started.append(1),
            clock=lambda: NOW,
        )

    @property
    def output(self):
        return "\n".join(self.lines)


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "NVIDIA_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    return Env(tmp_path)


def test_init_writes_secrets_hidden_and_merges(env):
    env.path.write_text("# my config\nTIMEZONE=UTC\n")
    os.chmod(env.path, 0o644)
    prompter = ScriptedPrompter(secrets=[BOT, NIM, ""])

    assert env.run(prompter) == 0

    assert env.path.read_text().startswith("# my config\nTIMEZONE=UTC\n")
    assert read_env_value(env.path, "TELEGRAM_BOT_TOKEN") == BOT
    assert read_env_value(env.path, "NVIDIA_API_KEY") == NIM
    assert read_env_value(env.path, "OPENROUTER_API_KEY") is None
    assert stat.S_IMODE(os.stat(env.path).st_mode) == 0o600
    assert len(prompter.secret_labels) == 3
    for label in prompter.secret_labels:
        assert BOT not in label and NIM not in label
    for secret in (BOT, NIM, OR):
        assert secret not in env.output


def test_init_keeps_existing_values_when_confirmed(env):
    env.path.write_text(f"TELEGRAM_BOT_TOKEN={BOT}\nNVIDIA_API_KEY={NIM}\n")
    prompter = ScriptedPrompter(secrets=[""], confirms=[True, True])  # keep, keep; OpenRouter skipped

    assert env.run(prompter) == 0

    assert [c[0] for c in prompter.confirm_labels[:2]] == [
        "Keep the existing TELEGRAM_BOT_TOKEN?",
        "Keep the existing NVIDIA_API_KEY?",
    ]
    assert prompter.confirm_labels[0][1] is True
    assert len(prompter.secret_labels) == 1  # only the OpenRouter key was asked for
    assert read_env_value(env.path, "TELEGRAM_BOT_TOKEN") == BOT
    assert read_env_value(env.path, "NVIDIA_API_KEY") == NIM


def test_init_replaces_existing_value_when_not_kept(env):
    env.path.write_text(f"NVIDIA_API_KEY={NIM}\n")
    prompter = ScriptedPrompter(secrets=["", "nvapi-<NEW>", ""], confirms=[False])

    assert env.run(prompter) == 0

    assert read_env_value(env.path, "NVIDIA_API_KEY") == "nvapi-<NEW>"


def test_init_rejected_existing_value_is_reprompted(env):
    env.path.write_text(f"TELEGRAM_BOT_TOKEN={BOT}\n")
    validators = FakeValidators({"telegram": [CheckResult(False, "rejected (401)"), CheckResult(True, "@fake_bot")]})
    prompter = ScriptedPrompter(secrets=["111111:<GOOD>", "", ""], confirms=[True])

    assert env.run(prompter, validators) == 0

    assert read_env_value(env.path, "TELEGRAM_BOT_TOKEN") == "111111:<GOOD>"
    assert BOT not in env.output


def test_init_invalid_token_reprompts_then_fails_without_writing(env):
    env.path.write_text("KEEP=1\n")
    before = env.path.read_bytes()
    validators = FakeValidators({"telegram": [CheckResult(False, "rejected (401)")] * 3})
    prompter = ScriptedPrompter(secrets=["bad1", "bad2", "bad3"])

    assert env.run(prompter, validators) == 1

    assert env.path.read_bytes() == before
    assert len(prompter.secret_labels) == 3
    assert [c[1] for c in validators.calls] == ["bad1", "bad2", "bad3"]
    assert env.store.read_snapshot() is None  # nothing happened after the failure
    for secret in ("bad1", "bad2", "bad3"):
        assert secret not in env.output


def test_init_value_with_forbidden_characters_is_refused_without_echo(env):
    prompter = ScriptedPrompter(secrets=["has space <S1>", "has\"quote <S2>", "ok:<GOOD>", "", ""])

    assert env.run(prompter) == 0

    assert read_env_value(env.path, "TELEGRAM_BOT_TOKEN") == "ok:<GOOD>"
    assert "<S1>" not in env.output and "<S2>" not in env.output


def test_init_unverifiable_credential_saved_with_warning(env):
    validators = FakeValidators({"nim": [CheckResult(None, "could not reach NVIDIA (ConnectError)")]})
    prompter = ScriptedPrompter(secrets=["", NIM, ""])

    assert env.run(prompter, validators) == 0

    assert read_env_value(env.path, "NVIDIA_API_KEY") == NIM
    assert "could not reach NVIDIA" in env.output
    assert "saved" in env.output.lower()


def test_init_openrouter_adds_default_models(env):
    prompter = ScriptedPrompter(secrets=["", "", OR])

    assert env.run(prompter) == 0

    assert read_env_value(env.path, "OPENROUTER_API_KEY") == OR
    assert read_env_value(env.path, "OPENROUTER_MODELS") == DEFAULT_OPENROUTER_MODELS
    example = Path(__file__).resolve().parent.parent / ".env.example"
    assert DEFAULT_OPENROUTER_MODELS == read_env_value(example, "OPENROUTER_MODELS")
    assert "openrouter/free" not in DEFAULT_OPENROUTER_MODELS


def test_init_openrouter_keeps_existing_models_line(env):
    env.path.write_text("OPENROUTER_MODELS=vendor/model:free\n")
    prompter = ScriptedPrompter(secrets=["", "", OR])

    assert env.run(prompter) == 0

    assert read_env_value(env.path, "OPENROUTER_MODELS") == "vendor/model:free"


def test_init_warns_without_any_llm_key(env):
    assert env.run(ScriptedPrompter()) == 0

    assert "no LLM key" in env.output
    assert not env.path.exists()  # nothing entered: nothing written


def test_init_skip_credentials_never_prompts_or_touches_env(env):
    env.path.write_text("KEEP=1\n")
    os.chmod(env.path, 0o644)
    prompter = ScriptedPrompter()

    assert env.run(prompter, options=InitOptions(skip_credentials=True, start=False)) == 0

    assert prompter.secret_labels == [] and prompter.confirm_labels == []
    assert stat.S_IMODE(os.stat(env.path).st_mode) == 0o644
    assert env.store.read_snapshot() is not None


def test_init_chmods_an_existing_env_even_without_new_values(env):
    env.path.write_text("KEEP=1\n")
    os.chmod(env.path, 0o644)

    assert env.run(ScriptedPrompter()) == 0

    assert stat.S_IMODE(os.stat(env.path).st_mode) == 0o600


def test_init_adopts_existing_chat_id(env):
    env.path.write_text(f"TELEGRAM_BOT_TOKEN={BOT}\nTELEGRAM_CHAT_ID=4242\n")
    prompter = ScriptedPrompter(secrets=["", ""], confirms=[True])

    assert env.run(prompter) == 0

    assert env.state.get_pairing() == Pairing(4242, 4242)
    assert env.state.get_pairing_code() is None
    assert "Adopted TELEGRAM_CHAT_ID as the paired chat." in env.output
    assert not re.search(r"/start [0-9A-Z]{4}-[0-9A-Z]{4}", env.output)


def test_init_reports_already_paired(env):
    env.state.set_pairing(5, 5)
    env.path.write_text(f"TELEGRAM_BOT_TOKEN={BOT}\n")

    assert env.run(ScriptedPrompter(secrets=["", ""], confirms=[True])) == 0

    assert "Already paired." in env.output
    assert env.state.get_pairing() == Pairing(5, 5)
    assert env.state.get_pairing_code() is None


def test_init_issues_code_when_unpaired(env):
    env.path.write_text("TELEGRAM_CHAT_ID=your_telegram_user_or_chat_id_here\n")
    prompter = ScriptedPrompter(secrets=[BOT, "", ""])

    assert env.run(prompter) == 0

    match = re.search(r"/start ([0-9A-Z]{4}-[0-9A-Z]{4}) to the bot in a private chat within 15 minutes "
                      r"\(the daemon must be running\)\.", env.output)
    assert match, env.output
    code = env.state.get_pairing_code()
    assert code.code_hash == hash_code(match.group(1))
    assert code.expires_at == NOW + PAIRING_TTL_SECONDS
    assert "pairing will replace" in env.output
    assert env.state.get_pairing() is None


def test_init_without_a_bot_token_skips_pairing(env):
    assert env.run(ScriptedPrompter()) == 0

    assert env.state.get_pairing_code() is None
    assert "/start" not in env.output


def test_init_creates_paused_english_starter_profile(env):
    assert env.run(ScriptedPrompter(), options=InitOptions(skip_credentials=True, start=False)) == 0

    snapshot = env.store.read_snapshot()
    assert snapshot.fields.paused is True
    assert snapshot.fields.output_language == "en"
    assert "paused" in env.output


def test_init_language_option(env):
    env.run(ScriptedPrompter(), options=InitOptions(skip_credentials=True, start=False, language="pl"))

    assert env.store.read_snapshot().fields.output_language == "pl"


def test_init_leaves_an_existing_profile_untouched(env):
    apply_changes(env.store, [ReplaceProfile(ProfileSnapshot(ProfileFields(paused=True, wow_threshold=9.0)))],
                  label="imported", expected_version=0)

    assert env.run(ScriptedPrompter(), options=InitOptions(skip_credentials=True, start=False)) == 0

    snapshot = env.store.read_snapshot()
    assert snapshot.version == 1 and snapshot.fields.wow_threshold == 9.0
    assert snapshot.fields.output_language is None
    assert "Profile exists (version 1)." in env.output


def test_init_start_prompt_defaults(env):
    prompter = ScriptedPrompter(secrets=[BOT, "", ""], confirms=[True])
    options = InitOptions(start=None)

    assert env.run(prompter, options=options) == 0

    assert prompter.confirm_labels[-1] == ("Start the daemon now?", True)  # a code was issued
    assert env.started == [1]


def test_init_start_prompt_default_false_without_code_and_no(env):
    prompter = ScriptedPrompter(confirms=[False])

    assert env.run(prompter, options=InitOptions(skip_credentials=True, start=None)) == 0

    assert prompter.confirm_labels[-1] == ("Start the daemon now?", False)
    assert env.started == []


def test_init_no_start_never_asks_or_starts(env):
    prompter = ScriptedPrompter(secrets=[BOT, "", ""])

    assert env.run(prompter, options=InitOptions(start=False)) == 0

    assert all(label != "Start the daemon now?" for label, _ in prompter.confirm_labels)
    assert env.started == []


def test_init_start_flag_starts_without_asking(env):
    prompter = ScriptedPrompter()

    assert env.run(prompter, options=InitOptions(skip_credentials=True, start=True)) == 0

    assert prompter.confirm_labels == []
    assert env.started == [1]


# --- http validators ----------------------------------------------------------------------------


def _validators(handler):
    return http_validators(Settings(_env_file=None), transport=httpx.MockTransport(handler))


def test_telegram_validator_maps_statuses_and_hides_secrets():
    seen = []

    def ok(request):
        seen.append(request.url.path)
        return httpx.Response(200, json={"ok": True, "result": {"username": "fake_bot"}})

    assert _validators(ok).telegram(BOT) == CheckResult(True, "@fake_bot")
    assert seen == [f"/bot{BOT}/getMe"]

    def unauthorized(request):
        return httpx.Response(401, json={"ok": False, "error_code": 401, "description": f"Unauthorized {BOT}"})

    result = _validators(unauthorized).telegram(BOT)
    assert result.ok is False and BOT not in result.detail

    def down(request):
        raise httpx.ConnectError(f"cannot reach {request.url}")

    result = _validators(down).telegram(BOT)
    assert result.ok is None and BOT not in result.detail and "ConnectError" in result.detail


def test_nim_validator_is_one_minimal_chat_completion():
    seen = []

    def handler(request):
        seen.append((request.method, str(request.url), request.headers["authorization"], json.loads(request.content)))
        return httpx.Response(200, json={"choices": []})

    assert _validators(handler).nim(NIM).ok is True

    method, url, auth, body = seen[0]
    assert (method, url) == ("POST", "https://integrate.api.nvidia.com/v1/chat/completions")
    assert auth == f"Bearer {NIM}"
    assert body["max_tokens"] == 16 and body["model"] == "nvidia/nemotron-3-ultra-550b-a55b"
    assert "response_format" not in body and "thinking_token_budget" not in body and body["temperature"] != 0


@pytest.mark.parametrize("status,expected", [(200, True), (401, False), (403, False), (429, None), (500, None)])
def test_nim_and_openrouter_status_mapping(status, expected):
    def handler(request):
        return httpx.Response(status, json={"error": f"echo {NIM} {OR}"})

    validators = _validators(handler)
    for result in (validators.nim(NIM), validators.openrouter(OR)):
        assert result.ok is expected
        assert NIM not in result.detail and OR not in result.detail


def test_openrouter_validator_gets_key_endpoint():
    seen = []

    def handler(request):
        seen.append((request.method, str(request.url), request.headers["authorization"]))
        return httpx.Response(200, json={"data": {}})

    assert _validators(handler).openrouter(OR).ok is True
    assert seen == [("GET", "https://openrouter.ai/api/v1/key", f"Bearer {OR}")]


def test_validators_report_network_errors_without_the_url():
    def handler(request):
        raise httpx.ReadTimeout(f"timeout for {request.url} {request.headers['authorization']}")

    validators = _validators(handler)
    for result in (validators.nim(NIM), validators.openrouter(OR)):
        assert result.ok is None
        assert "ReadTimeout" in result.detail and NIM not in result.detail and OR not in result.detail


def test_rejected_existing_token_skipped_issues_no_pairing_code(env):
    env.path.write_text(f"TELEGRAM_BOT_TOKEN={BOT}\nNVIDIA_API_KEY={NIM}\n")
    validators = FakeValidators({"telegram": [CheckResult(False, "Telegram rejected the token (HTTP 401)")]})
    prompter = ScriptedPrompter(secrets=["", ""], confirms=[True, True])  # keep token (rejected), skip re-entry

    assert env.run(prompter, validators, options=InitOptions(start=None)) == 0

    assert env.state.get_pairing_code() is None
    assert "no pairing code" in env.output.lower() and "TELEGRAM_BOT_TOKEN" in env.output
    assert "no LLM key" not in env.output  # the NVIDIA key is still in the file
    assert prompter.confirm_labels[-1] == ("Start the daemon now?", False)


def test_rejected_existing_token_replaced_still_pairs(env):
    env.path.write_text(f"TELEGRAM_BOT_TOKEN={BOT}\n")
    validators = FakeValidators({"telegram": [CheckResult(False, "x"), CheckResult(True, "@fake_bot")]})
    prompter = ScriptedPrompter(secrets=["111:<NEW>", "", ""], confirms=[True])

    assert env.run(prompter, validators) == 0

    assert env.state.get_pairing_code() is not None
    assert "no pairing code" not in env.output.lower()


def test_no_llm_warning_when_a_key_is_still_in_the_file(env):
    env.path.write_text(f"OPENROUTER_API_KEY={OR}\n")
    prompter = ScriptedPrompter(secrets=["", ""], confirms=[False])  # decline keeping it, skip re-entry

    env.run(prompter)

    assert "no LLM key" not in env.output


@pytest.mark.parametrize("language", ["", "english", "EN", "e1", "en\n", "x" * 40])
def test_invalid_language_fails_before_anything_is_written(env, language):
    env.path.write_text("KEEP=1\n")
    before = env.path.read_bytes()
    prompter = ScriptedPrompter(secrets=[BOT, NIM, ""])

    assert env.run(prompter, options=InitOptions(start=False, language=language)) == 1

    assert env.path.read_bytes() == before
    assert prompter.secret_labels == []
    assert env.store.read_snapshot() is None
    assert "--language" in env.output or "language" in env.output.lower()
