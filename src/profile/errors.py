"""Typed errors of the profile subsystem."""
from typing import Iterable, List


class ProfileError(Exception):
    """Base class of every profile error."""


class ProfileValidationError(ProfileError):
    """A snapshot, operation or import failed validation; ``messages`` lists every problem."""

    def __init__(self, messages: "Iterable[str] | str"):
        self.messages: List[str] = [messages] if isinstance(messages, str) else list(messages)
        super().__init__("; ".join(self.messages))


class VersionConflict(ProfileError):
    """The stored profile version differs from the version the caller based its change on."""


class ProfileCompileError(ProfileError):
    """A snapshot cannot be compiled (invalid regex, unknown board / kind / match mode)."""


class LegacyImportError(ProfileError):
    """The legacy queries.json / .candidate_profile.md files cannot be imported."""


class NothingToUndo(ProfileError):
    """There is no earlier profile version to return to."""


class ExportRefused(ProfileError):
    """A profile export was refused (for example a path that is not git-ignored)."""
