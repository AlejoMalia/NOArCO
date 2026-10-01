"""TRIADA/MATE core: cache semantics, decorator protocol, projection certification, thread safety."""

import threading

import pytest

from noarco.mate.core import (
    MATEResult,
    MATEStatus,
    NOArCoCache,
    get_default_cache,
    mate_project,
    reset_default_cache,
    triada,
)


def _res(v=1):
    return MATEResult(value=v, status=MATEStatus.FULL_COMPUTE)


class TestCache:
    def test_miss_then_hit_counts(self):
        c = NOArCoCache(maxsize=4)
        assert c.get("a") is None
        c.set("a", _res(1))
        assert c.get("a").value == 1
        s = c.stats()
        assert (s["hits"], s["misses"], s["size"]) == (1, 1, 1)
        assert s["hit_rate"] == pytest.approx(0.5)

    def test_lru_eviction_order(self):
        c = NOArCoCache(maxsize=2)
        c.set("a", _res()); c.set("b", _res())
        c.get("a")              # a becomes most-recently-used
        c.set("c", _res())      # evicts b
        assert c.get("b") is None and c.get("a") is not None and c.get("c") is not None

    def test_overwrite_does_not_grow(self):
        c = NOArCoCache(maxsize=2)
        for _ in range(5):
            c.set("a", _res())
        assert c.stats()["size"] == 1

    def test_invalidate_all_and_prefix(self):
        c = NOArCoCache()
        for k in ("x1", "x2", "y1"):
            c.set(k, _res())
        assert c.invalidate("x") == 2
        assert c.stats()["size"] == 1
        assert c.invalidate() == 1 and c.stats()["size"] == 0

    def test_invalid_maxsize(self):
        with pytest.raises(ValueError):
            NOArCoCache(maxsize=0)

    def test_key_is_deterministic_and_argument_sensitive(self):
        c = NOArCoCache()
        assert c._make_key("f", (1, 2), {"k": 3}) == c._make_key("f", (1, 2), {"k": 3})
        assert c._make_key("f", (1, 2), {}) != c._make_key("f", (2, 1), {})
        assert c._make_key("f", (), {"a": 1, "b": 2}) == c._make_key("f", (), {"b": 2, "a": 1})
        assert len(c._make_key("f", (), {})) == 64  # sha256 hex

    def test_unserialisable_arguments_still_keyed(self):
        c = NOArCoCache()
        assert c._make_key("f", (object(),), {})

    def test_thread_safety_under_contention(self):
        c = NOArCoCache(maxsize=16)
        errors = []

        def work(i):
            try:
                for j in range(300):
                    k = f"k{(i * 7 + j) % 40}"
                    if c.get(k) is None:
                        c.set(k, _res(j))
                    if j % 50 == 0:
                        c.invalidate(f"k{j % 5}")
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        ts = [threading.Thread(target=work, args=(i,)) for i in range(8)]
        for t in ts: t.start()
        for t in ts: t.join()
        assert not errors
        s = c.stats()
        assert s["size"] <= 16 and len(c._store) == len(c._access_order)


class TestTriadaDecorator:
    def test_full_compute_then_cache_hit(self):
        cache = NOArCoCache()
        calls = []

        @triada(cache=cache)
        def f(x):
            calls.append(x)
            return x * 2

        a, b = f(3), f(3)
        assert (a.status, b.status) == (MATEStatus.FULL_COMPUTE, MATEStatus.CACHE_HIT)
        assert a.value == b.value == 6 and calls == [3]
        assert f(4).status == MATEStatus.FULL_COMPUTE and calls == [3, 4]

    def test_closed_form_short_circuits(self):
        cache = NOArCoCache()
        ran = []

        @triada(closed_form=lambda x: 10 if x > 0 else None, closed_form_eq="y=10", cache=cache)
        def f(x):
            ran.append(x)
            return -1

        r = f(1)
        assert r.status == MATEStatus.CLOSED_FORM and r.value == 10 and r.closed_form_equation == "y=10"
        assert not ran
        assert f(-1).status == MATEStatus.FULL_COMPUTE and f(-1).value == -1

    def test_failing_closed_form_falls_through(self):
        @triada(closed_form=lambda x: 1 / 0, cache=NOArCoCache())
        def f(x):
            return x + 1

        assert f(1).value == 2 and f(1).status == MATEStatus.CACHE_HIT

    def test_skip_cache_never_hits(self):
        n = []

        @triada(skip_cache=True, cache=NOArCoCache())
        def f():
            n.append(1)
            return len(n)

        assert f().value == 1 and f().value == 2

    def test_exceptions_propagate_and_are_not_cached(self):
        cache = NOArCoCache()

        @triada(cache=cache)
        def f():
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            f()
        assert cache.stats()["size"] == 0

    def test_preserves_metadata_and_default_cache_reset(self):
        @triada(cache=NOArCoCache())
        def documented():
            """doc"""
            return 1

        assert documented.__name__ == "documented" and documented.__doc__ == "doc"
        get_default_cache().set("z", _res())
        reset_default_cache()
        assert get_default_cache().stats()["size"] == 0


class TestProjectionAndResult:
    def test_mate_project_contract(self):
        r = mate_project(
            value={"feasible": False},
            projection_basis="Step 1 -> Step 2 -> Step 3",
            unexpanded_branches=["regolith: cannot close a 4x gap"],
            source_call="t", warnings=["w"],
        )
        assert r.status == MATEStatus.PROJECTED
        assert r.projection_basis and r.unexpanded_branches == ["regolith: cannot close a 4x gap"]
        assert r.warnings == ["w"] and r.unwrap() == {"feasible": False}
        assert "PROJECTED" in r.summary().upper()

    def test_result_summary_for_every_status(self):
        for st in MATEStatus:
            assert isinstance(MATEResult(value=1, status=st).summary(), str)
