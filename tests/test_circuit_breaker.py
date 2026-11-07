import threading

import pytest

from src.evaluator.circuit_breaker import CircuitBreaker


class FakeClock:
    def __init__(self, start: float = 100.0):
        self.now = start

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def breaker(clock):
    return CircuitBreaker(failure_threshold=3, cooldown_seconds=600, time_fn=clock)


def trip(breaker):
    for _ in range(3):
        assert breaker.allow()
        breaker.record_failure()


def test_defaults():
    b = CircuitBreaker()
    assert b.failure_threshold == 3
    assert b.cooldown_seconds == 600


def test_starts_closed_and_allows(breaker):
    assert breaker.state == "closed"
    assert breaker.allow() is True
    assert breaker.allow() is True  # closed allows any number of calls


def test_opens_exactly_at_threshold(breaker):
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state == "closed"
    breaker.record_failure()
    assert breaker.state == "open"
    assert breaker.allow() is False


def test_success_resets_failure_counter(breaker):
    breaker.record_failure()
    breaker.record_failure()
    breaker.record_success()
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state == "closed"
    breaker.record_failure()
    assert breaker.state == "open"


def test_blocks_during_cooldown(breaker, clock):
    trip(breaker)
    clock.now += 599.0
    assert breaker.state == "open"
    assert breaker.allow() is False


def test_half_open_after_cooldown_allows_single_probe(breaker, clock):
    trip(breaker)
    clock.now += 600.0
    assert breaker.state == "half_open"
    assert breaker.allow() is True  # claims the probe
    assert breaker.state == "half_open"
    assert breaker.allow() is False  # probe outstanding
    assert breaker.allow() is False


def test_probe_success_closes(breaker, clock):
    trip(breaker)
    clock.now += 600.0
    assert breaker.allow()
    breaker.record_success()
    assert breaker.state == "closed"
    assert breaker.allow() is True
    assert breaker.allow() is True


def test_probe_failure_reopens_with_fresh_cooldown(breaker, clock):
    trip(breaker)
    clock.now += 600.0
    assert breaker.allow()
    breaker.record_failure()
    assert breaker.state == "open"
    clock.now += 599.0
    assert breaker.allow() is False  # cooldown restarted at the probe failure
    clock.now += 1.0
    assert breaker.allow() is True


def test_release_probe_lets_the_next_caller_probe(breaker, clock):
    """Exception path: a probe abandoned without an outcome must not stick."""
    trip(breaker)
    clock.now += 600.0
    assert breaker.allow()
    assert breaker.allow() is False
    breaker.release_probe()
    assert breaker.state == "half_open"
    assert breaker.allow() is True


def test_probe_released_in_finally_on_exception(breaker, clock):
    trip(breaker)
    clock.now += 600.0

    def guarded_call():
        if not breaker.allow():
            return
        settled = False
        try:
            raise ValueError("unexpected failure mid-call")
        finally:
            if not settled:
                breaker.release_probe()

    with pytest.raises(ValueError):
        guarded_call()
    assert breaker.allow() is True


def test_release_probe_is_noop_when_closed(breaker):
    breaker.release_probe()
    assert breaker.state == "closed"
    assert breaker.allow() is True


def test_open_immediately(breaker, clock):
    breaker.open_immediately()
    assert breaker.state == "open"
    assert breaker.allow() is False
    clock.now += 600.0
    assert breaker.allow() is True


def test_open_immediately_clears_outstanding_probe(breaker, clock):
    trip(breaker)
    clock.now += 600.0
    assert breaker.allow()
    breaker.open_immediately()
    assert breaker.state == "open"
    clock.now += 600.0
    assert breaker.allow() is True


def test_failures_while_open_do_not_extend_cooldown(breaker, clock):
    trip(breaker)
    clock.now += 300.0
    breaker.record_failure()  # straggler from a call started before opening
    clock.now += 300.0
    assert breaker.allow() is True


def test_only_one_thread_wins_the_half_open_probe(breaker, clock):
    trip(breaker)
    clock.now += 600.0
    n = 16
    barrier = threading.Barrier(n)
    results = []
    lock = threading.Lock()

    def worker():
        barrier.wait(timeout=5)
        got = breaker.allow()
        with lock:
            results.append(got)

    threads = [threading.Thread(target=worker) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)
    assert len(results) == n
    assert results.count(True) == 1
