import pytest

from gateway.keypool import KeyPool

KEYS = ["sk-aaaa1111", "sk-bbbb2222"]


class Clock:
    def __init__(self):
        self.now = 1_000.0

    def __call__(self):
        return self.now


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def pool(tmp_path, clock):
    return KeyPool(str(tmp_path / "keys.db"), clock=clock)


def test_rotation_alternates_between_keys(pool):
    assert [pool.pick("groq", KEYS) for _ in range(4)] == KEYS * 2


def test_auth_failure_benches_a_key_for_a_day(pool, clock):
    pool.report("groq", KEYS[0], 401)
    assert {pool.pick("groq", KEYS) for _ in range(4)} == {KEYS[1]}
    clock.now += 24 * 3600 + 1
    assert KEYS[0] in {pool.pick("groq", KEYS) for _ in range(4)}


def test_rate_limit_honours_retry_after(pool, clock):
    pool.report("groq", KEYS[0], 429, retry_after=5)
    assert pool.pick("groq", KEYS[:1]) is None
    clock.now += 6
    assert pool.pick("groq", KEYS[:1]) == KEYS[0]


def test_success_clears_a_cooldown(pool):
    pool.report("groq", KEYS[0], 503)
    pool.report("groq", KEYS[0], 200)
    assert pool.pick("groq", KEYS[:1]) == KEYS[0]


def test_state_is_shared_through_the_file(tmp_path, clock):
    path = str(tmp_path / "keys.db")
    KeyPool(path, clock=clock).report("groq", KEYS[0], 401)
    assert KeyPool(path, clock=clock).pick("groq", KEYS[:1]) is None


def test_all_cooling_reports_the_soonest_recovery(pool):
    pool.report("groq", KEYS[0], 429, retry_after=30)
    pool.report("groq", KEYS[1], 429, retry_after=10)
    assert pool.pick("groq", KEYS) is None
    assert pool.soonest("groq", KEYS) == pytest.approx(10)


def test_states_never_expose_the_full_key(pool):
    pool.report("groq", KEYS[0], 401)
    states = pool.states("groq", KEYS)
    assert [s["state"] for s in states] == ["cooling", "ok"]
    assert all(k not in str(states) for k in KEYS)
    assert states[0]["key"].endswith("1111")
