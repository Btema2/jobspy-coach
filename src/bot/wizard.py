"""``/setup``: a deterministic, stepwise wizard (role, synonyms, market, remote preference, confirm).

The wizard produces the same operations as ``/add_role`` and ``/add_market`` and applies them as one versioned
change after an explicit Apply. Its state is one JSON document in ``bot_state.wizard_state``; every button
carries the wizard's random nonce, so a button of an earlier or replaced wizard can never act. State is
cleared on apply, cancel, replacement and expiry (30 minutes without an answer).
"""
import secrets
from dataclasses import replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from src.bot.countries import resolve_country
from src.bot.render import esc, render_apply
from src.bot.state import BotStateStore
from src.bot.updates import OutgoingMessage, Update
from src.profile.compiler import compile_profile
from src.profile.errors import ProfileValidationError, VersionConflict
from src.profile.ops import (
    AddMarket,
    AddRole,
    Operation,
    RemoveMarket,
    apply_changes,
    apply_operation,
    diff_snapshots,
    normalise_snapshot,
    validate_snapshot,
)
from src.profile.snapshot import Market, ProfileFields, ProfileSnapshot, RoleTerm
from src.profile.store import ProfileStore

WIZARD_TTL_SECONDS = 1800
STALE = "This setup is no longer active. Send /setup to start again."
_PAUSED_NOTE = "The profile is paused; send /resume to start searching."
_CONFLICT = "The profile changed meanwhile; send /setup again."
_CANCELLED = "Setup cancelled."
_ECHO_CLIP = 60
MAX_PROBLEMS_SHOWN = 4

_REMOTE_FLAGS = {"onsite": (False, False), "remote": (True, False), "remote_only": (True, True)}
_REMOTE_LABELS = {"onsite": "Onsite", "remote": "Remote", "remote_only": "Remote only"}


def _clip(text: str, limit: int = _ECHO_CLIP) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _words(text: str) -> str:
    return " ".join(text.split())


def _button(nonce: str, label: str, choice: str) -> Dict[str, str]:
    return {"text": label, "callback_data": f"wz:{nonce}:{choice}"}


def _markup(nonce: str, *rows: Sequence[Tuple[str, str]]) -> dict:
    keyboard = [[_button(nonce, label, choice) for label, choice in row] for row in rows]
    keyboard.append([_button(nonce, "Cancel", "cancel")])
    return {"inline_keyboard": keyboard}


class SetupWizard:
    def __init__(self, state: BotStateStore, store: ProfileStore, *, clock: Callable[[], float]):
        self._state = state
        self._store = store
        self._clock = clock

    # --- entry points -------------------------------------------------------------------

    def start(self, update: Update) -> List[OutgoingMessage]:
        snapshot = self._store.read_snapshot()
        wizard: Dict[str, Any] = {
            "nonce": secrets.token_hex(4),
            "step": "role",
            "base_version": snapshot.version if snapshot is not None else 0,
            "role": "",
            "synonyms": [],
            "market": {"country": None, "name": "", "languages": []},
            "remote": "onsite",
            "updated_at": self._clock(),
        }
        self._state.set_wizard(wizard)
        return [self._prompt(wizard)]

    def on_text(self, update: Update) -> Optional[List[OutgoingMessage]]:
        wizard = self._state.get_wizard()
        if wizard is None:
            return None
        if self._expired(wizard):
            self._state.set_wizard(None)
            return [OutgoingMessage(STALE)]
        step = wizard.get("step")
        text = _words(update.text)
        if step == "role":
            return self._role_text(wizard, text)
        if step == "synonyms":
            return self._synonyms_text(wizard, text)
        if step == "market":
            return self._market_text(wizard, text)
        return [OutgoingMessage("Please use the buttons below, or send /setup to start again."), self._prompt(wizard)]

    def on_callback(self, update: Update) -> List[OutgoingMessage]:
        parts = update.text.split(":")
        wizard = self._state.get_wizard()
        if (
            len(parts) != 3
            or parts[0] != "wz"
            or wizard is None
            or parts[1] != wizard.get("nonce")
            or self._expired(wizard)
        ):
            if wizard is not None and self._expired(wizard):
                self._state.set_wizard(None)
            return [OutgoingMessage(STALE)]
        choice, step = parts[2], wizard.get("step")
        if choice == "cancel":
            self._state.set_wizard(None)
            return [OutgoingMessage(_CANCELLED)]
        if step == "synonyms" and choice == "skip":
            return self._advance(wizard, "market")
        if step == "market" and choice[:1] == "c" and choice[1:].isdigit():
            options = wizard.get("options") or []
            index = int(choice[1:])
            if index < len(options):
                wizard["market"] = options[index]
                wizard.pop("options", None)
                return self._advance(wizard, "remote")
        if step == "remote" and choice in _REMOTE_FLAGS:
            wizard["remote"] = choice
            return self._advance_to_confirm(wizard)
        if step == "confirm" and choice == "apply":
            return self._apply(wizard)
        return [OutgoingMessage(STALE)]

    # --- helpers ------------------------------------------------------------------------

    def _expired(self, wizard: Dict[str, Any]) -> bool:
        return self._clock() - float(wizard.get("updated_at", 0)) > WIZARD_TTL_SECONDS

    def _save(self, wizard: Dict[str, Any]) -> None:
        wizard["updated_at"] = self._clock()
        self._state.set_wizard(wizard)

    def _advance(self, wizard: Dict[str, Any], step: str) -> List[OutgoingMessage]:
        wizard["step"] = step
        self._save(wizard)
        return [self._prompt(wizard)]

    def _base(self) -> ProfileSnapshot:
        return self._store.read_snapshot() or ProfileSnapshot(ProfileFields())

    def _terms(self, wizard: Dict[str, Any]) -> Tuple[RoleTerm, ...]:
        return tuple(RoleTerm(term, None) for term in wizard.get("synonyms", []))

    def _market(self, wizard: Dict[str, Any]) -> Market:
        data = wizard["market"]
        remote, remote_only = _REMOTE_FLAGS[wizard.get("remote", "onsite")]
        name = data["name"] + (" (remote only)" if remote_only else "")
        country = data.get("country")
        return Market(
            name=name,
            location=data["name"],
            countries=(country,) if country else (),
            remote=remote,
            remote_only=remote_only,
            languages=tuple(data.get("languages", [])),
        )

    def _operations(self, wizard: Dict[str, Any], base: ProfileSnapshot) -> List[Operation]:
        market = self._market(wizard)
        ops: List[Operation] = []
        if any(m.name.strip().casefold() == market.name.casefold() for m in base.markets):
            ops.append(RemoveMarket(market.name))  # same as /add_market: re-adding replaces the market
        ops.append(AddRole(wizard["role"], self._terms(wizard)))
        ops.append(AddMarket(market))
        return ops

    def _problems(self, snapshot: ProfileSnapshot, new_ops: Sequence[Operation]) -> List[str]:
        """Validation problems of ``snapshot`` plus ``new_ops``; paused is forced so that resuming is not judged."""
        paused = replace(snapshot, fields=replace(snapshot.fields, paused=True))
        for op in new_ops:
            paused = apply_operation(paused, op)
        return validate_snapshot(normalise_snapshot(paused), None)

    @staticmethod
    def _problem_text(heading: str, problems: Sequence[str], tail: str) -> str:
        shown = [f"- {esc(_clip(p, 160))}" for p in problems[:MAX_PROBLEMS_SHOWN]]
        if len(problems) > MAX_PROBLEMS_SHOWN:
            shown.append(f"- ... and {len(problems) - MAX_PROBLEMS_SHOWN} more")
        return "\n".join([heading] + shown + [tail])

    # --- steps --------------------------------------------------------------------------

    def _prompt(self, wizard: Dict[str, Any]) -> OutgoingMessage:
        nonce, step = wizard["nonce"], wizard["step"]
        if step == "role":
            return OutgoingMessage(
                "Setup, step 1 of 4: send the name of the role you are looking for.", _markup(nonce)
            )
        if step == "synonyms":
            return OutgoingMessage(
                f"Step 2 of 4: send synonyms (other names) for <b>{esc(_clip(wizard['role']))}</b>, "
                "separated by commas, or press Skip.",
                _markup(nonce, [("Skip", "skip")]),
            )
        if step == "market":
            options = wizard.get("options")
            if options:
                return OutgoingMessage(
                    "Which one did you mean?",
                    _markup(nonce, *([(o["name"], f"c{i}")] for i, o in enumerate(options))),
                )
            return OutgoingMessage(
                "Step 3 of 4: send the country to search, by its English name (or Europe / Worldwide).",
                _markup(nonce),
            )
        if step == "remote":
            name = esc(_clip(wizard["market"]["name"]))
            return OutgoingMessage(
                f"Step 4 of 4: how should I treat remote work in <b>{name}</b>?",
                _markup(nonce, [(_REMOTE_LABELS[c], c) for c in _REMOTE_LABELS]),
            )
        return self._confirm_message(wizard)

    def _confirm_message(self, wizard: Dict[str, Any]) -> OutgoingMessage:
        current = self._store.read_snapshot()
        base = self._base()
        proposed = normalise_snapshot(self._apply_all(base, wizard))
        lines = ["<b>Apply these changes?</b>"]
        lines.extend(esc(line) for line in diff_snapshots(current, proposed))
        lines.extend("Warning: " + esc(w) for w in compile_profile(proposed).warnings)
        return OutgoingMessage(
            "\n".join(lines), _markup(wizard["nonce"], [("Apply", "apply")])
        )

    def _apply_all(self, base: ProfileSnapshot, wizard: Dict[str, Any]) -> ProfileSnapshot:
        snapshot = base
        for op in self._operations(wizard, base):
            snapshot = apply_operation(snapshot, op)
        return snapshot

    def _role_text(self, wizard: Dict[str, Any], text: str) -> List[OutgoingMessage]:
        if not text:
            return [self._prompt(wizard)]
        problems = self._problems(self._base(), [AddRole(text)])
        if problems:
            return [
                OutgoingMessage(
                    self._problem_text("That role name is not valid:", problems, "Send another name."),
                    _markup(wizard["nonce"]),
                )
            ]
        wizard["role"] = text
        return self._advance(wizard, "synonyms")

    def _synonyms_text(self, wizard: Dict[str, Any], text: str) -> List[OutgoingMessage]:
        synonyms: List[str] = []
        for raw in text.split(","):
            synonym = _words(raw)
            if (
                synonym
                and synonym.casefold() != wizard["role"].casefold()
                and synonym.casefold() not in {s.casefold() for s in synonyms}
            ):
                synonyms.append(synonym)
        if not synonyms:
            return [self._prompt(wizard)]
        terms = tuple(RoleTerm(s, None) for s in synonyms)
        problems = self._problems(self._base(), [AddRole(wizard["role"], terms)])
        if problems:
            return [
                OutgoingMessage(
                    self._problem_text("Those synonyms are not valid:", problems, "Send them again, or press Skip."),
                    _markup(wizard["nonce"], [("Skip", "skip")]),
                )
            ]
        wizard["synonyms"] = synonyms
        return self._advance(wizard, "market")

    def _market_text(self, wizard: Dict[str, Any], text: str) -> List[OutgoingMessage]:
        resolution = resolve_country(text)
        if resolution.match is not None:
            match = resolution.match
            wizard["market"] = {"country": match.country, "name": match.name, "languages": list(match.languages)}
            wizard.pop("options", None)
            return self._advance(wizard, "remote")
        if resolution.options:
            wizard["options"] = [
                {"country": o.country, "name": o.name, "languages": list(o.languages)} for o in resolution.options
            ]
            self._save(wizard)
            return [self._prompt(wizard)]
        wizard.pop("options", None)
        return [
            OutgoingMessage(
                f"Unknown country '{esc(_clip(text))}'. Try the English name, e.g. Germany.",
                _markup(wizard["nonce"]),
            )
        ]

    def _advance_to_confirm(self, wizard: Dict[str, Any]) -> List[OutgoingMessage]:
        base = self._base()
        problems = validate_snapshot(
            normalise_snapshot(self._apply_all(replace(base, fields=replace(base.fields, paused=True)), wizard)),
            None,
        )
        if problems:
            self._state.set_wizard(None)
            return [
                OutgoingMessage(
                    self._problem_text("Not applied:", problems, "Send /setup to start again."),
                )
            ]
        return self._advance(wizard, "confirm")

    def _apply(self, wizard: Dict[str, Any]) -> List[OutgoingMessage]:
        base = self._base()
        self._state.set_wizard(None)  # single use: a repeated press finds no wizard
        try:
            result = apply_changes(
                self._store,
                self._operations(wizard, base),
                label="setup wizard",
                expected_version=int(wizard["base_version"]),
            )
        except VersionConflict:
            return [OutgoingMessage(_CONFLICT)]
        except ProfileValidationError as exc:
            return [
                OutgoingMessage(self._problem_text("Not applied:", exc.messages, "Send /setup to start again."))
            ]
        extra = [_PAUSED_NOTE] if result.new is not None and result.new.fields.paused else []
        return [OutgoingMessage(render_apply(result, extra))]
