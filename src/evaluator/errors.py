"""Exception types shared by the evaluation engine (providers, chain, daemon)."""

from typing import Optional


class EvaluatorError(Exception):
    """Base class for all evaluation-engine errors."""


class TransientProviderError(EvaluatorError):
    """A provider call failed in a way that may succeed later (5xx, timeout, network)."""


class RateLimited(TransientProviderError):
    """The provider answered HTTP 429. ``retry_after`` is seconds, or None if no header."""

    def __init__(self, retry_after: Optional[float] = None, message: str = "rate limited"):
        super().__init__(message)
        self.retry_after = retry_after
        self.message = message

    def __reduce__(self):
        return (type(self), (self.retry_after, self.message))


class PermanentProviderError(EvaluatorError):
    """A provider call failed in a way retrying cannot fix (401/403/404/400/422)."""


class AuthenticationFailed(PermanentProviderError):
    """HTTP 401/403: the API key is bad or revoked; the provider is unusable until fixed."""


class ModelNotFound(PermanentProviderError):
    """HTTP 404: the configured model (or endpoint) does not exist any more."""


class BadRequest(PermanentProviderError):
    """HTTP 400/422 (or another unexpected 4xx/3xx): this request is at fault, not the provider."""


class UnexpectedStatus(PermanentProviderError):
    """An HTTP status outside the known classes (402, 409, 3xx, other 4xx): provider-wide, not the job's fault."""


class QuotaExhausted(EvaluatorError):
    """A provider's quota is spent until ``reset_at`` (epoch seconds, or None if unknown)."""

    def __init__(self, reset_at: Optional[float] = None, message: str = "quota exhausted"):
        super().__init__(message)
        self.reset_at = reset_at
        self.message = message

    def __reduce__(self):
        return (type(self), (self.reset_at, self.message))


class InvalidResponse(EvaluatorError):
    """HTTP 200 but the body was empty, unparseable, schema-invalid or truncated."""


class ResponseTruncated(InvalidResponse):
    """HTTP 200 with ``finish_reason == "length"``: the output hit ``max_tokens``."""


class ProvidersUnavailable(EvaluatorError):
    """Every provider was skipped or failed provider-wide; the job is not at fault."""


class EvaluationRejected(EvaluatorError):
    """Providers answered but every answer was invalid for this job (job-specific)."""


class StopRequested(EvaluatorError):
    """Shutdown was requested while waiting; abandon the call."""


class ConfigurationError(EvaluatorError, ValueError):
    """The evaluation engine cannot be built from the current settings (e.g. no provider enabled)."""
