import pickle

import pytest

from src.evaluator import errors


@pytest.mark.parametrize("name", [
    "RateLimited",
    "TransientProviderError",
    "PermanentProviderError",
    "QuotaExhausted",
    "InvalidResponse",
    "ProvidersUnavailable",
    "EvaluationRejected",
    "StopRequested",
])
def test_all_error_types_share_a_base(name):
    cls = getattr(errors, name)
    assert issubclass(cls, errors.EvaluatorError)
    assert issubclass(errors.EvaluatorError, Exception)


def test_rate_limited_carries_retry_after():
    assert errors.RateLimited(retry_after=12.5).retry_after == 12.5
    assert errors.RateLimited().retry_after is None
    assert errors.RateLimited(None).retry_after is None


def test_rate_limited_is_transient():
    assert issubclass(errors.RateLimited, errors.TransientProviderError)


def test_quota_exhausted_carries_reset_at():
    exc = errors.QuotaExhausted(reset_at=1234.0)
    assert exc.reset_at == 1234.0
    assert errors.QuotaExhausted().reset_at is None


def test_permanent_is_not_transient():
    assert not issubclass(errors.PermanentProviderError, errors.TransientProviderError)


def test_stop_requested_is_not_a_provider_error():
    assert not issubclass(errors.StopRequested, errors.TransientProviderError)
    assert not issubclass(errors.StopRequested, errors.PermanentProviderError)


@pytest.mark.parametrize("exc", [
    errors.RateLimited(retry_after=3.0),
    errors.QuotaExhausted(reset_at=99.0),
    errors.InvalidResponse("bad json"),
    errors.StopRequested("stop"),
])
def test_errors_are_picklable(exc):
    clone = pickle.loads(pickle.dumps(exc))
    assert type(clone) is type(exc)
    assert str(clone) == str(exc)
    assert getattr(clone, "retry_after", None) == getattr(exc, "retry_after", None)
    assert getattr(clone, "reset_at", None) == getattr(exc, "reset_at", None)
