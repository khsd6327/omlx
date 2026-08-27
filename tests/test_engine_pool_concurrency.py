# SPDX-License-Identifier: Apache-2.0
"""fork regression tests: engine pool lock restructure + lifecycle guards.

Covers:
- get_engine no longer holds the pool lock across the whole load
  (head-of-line blocking fix)
- concurrent get_engine callers share one load via the per-entry future
- load failures propagate to waiters
- discover_models preserves mid-load entries
- EngineCore.stop() releases hung waiters
"""

import asyncio
from pathlib import Path

import pytest

from omlx.engine_pool import EnginePool, EngineEntry


def _entry(model_id: str, model_path: str | None = None, **kw) -> EngineEntry:
    return EngineEntry(
        model_id=model_id,
        model_path=model_path or f"/tmp/{model_id}",
        model_type="llm",
        engine_type="batched",
        estimated_size=1024,
        **kw,
    )


def _make_pool(*model_ids: str, model_root: Path | None = None) -> EnginePool:
    pool = EnginePool()
    for mid in model_ids:
        model_path = None
        if model_root is not None:
            model_dir = model_root / mid
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "config.json").write_text("{}")
            model_path = str(model_dir)
        pool._entries[mid] = _entry(mid, model_path=model_path)
    return pool


class TestPoolLockRestructure:
    @pytest.mark.asyncio
    async def test_concurrent_get_engine_single_load(self, monkeypatch, tmp_path):
        pool = _make_pool("m1", model_root=tmp_path)
        started = asyncio.Event()
        release = asyncio.Event()
        loads: list[str] = []

        async def fake_load(model_id, force_lm=False):
            loads.append(model_id)
            started.set()
            await release.wait()
            entry = pool._entries[model_id]
            entry.engine = object()
            entry.is_loading = False

        monkeypatch.setattr(pool, "_load_engine", fake_load)

        t1 = asyncio.create_task(pool.get_engine("m1"))
        await asyncio.wait_for(started.wait(), 1.0)
        t2 = asyncio.create_task(pool.get_engine("m1"))
        await asyncio.sleep(0.01)
        assert not t2.done()

        release.set()
        e1, e2 = await asyncio.gather(t1, t2)
        assert e1 is e2
        assert loads == ["m1"]  # the waiter rode the first load

    @pytest.mark.asyncio
    async def test_loaded_model_not_blocked_by_other_load(
        self, monkeypatch, tmp_path
    ):
        """The core head-of-line fix: a request to an already-loaded model
        must not queue behind another model's cold load."""
        pool = _make_pool("slow", "fast", model_root=tmp_path)
        fast_engine = object()
        pool._entries["fast"].engine = fast_engine
        started = asyncio.Event()
        release = asyncio.Event()

        async def fake_load(model_id, force_lm=False):
            started.set()
            await release.wait()
            entry = pool._entries[model_id]
            entry.engine = object()
            entry.is_loading = False

        monkeypatch.setattr(pool, "_load_engine", fake_load)

        t_slow = asyncio.create_task(pool.get_engine("slow"))
        await asyncio.wait_for(started.wait(), 1.0)

        # Under the old code this timed out: the loader held the pool lock
        # for the full duration of _load_engine.
        got = await asyncio.wait_for(pool.get_engine("fast"), timeout=1.0)
        assert got is fast_engine

        release.set()
        await t_slow

    @pytest.mark.asyncio
    async def test_load_failure_propagates_to_waiters(
        self, monkeypatch, tmp_path
    ):
        pool = _make_pool("m1", model_root=tmp_path)
        started = asyncio.Event()
        release = asyncio.Event()

        async def fake_load(model_id, force_lm=False):
            started.set()
            await release.wait()
            pool._entries[model_id].is_loading = False
            raise RuntimeError("weights exploded")

        monkeypatch.setattr(pool, "_load_engine", fake_load)

        t1 = asyncio.create_task(pool.get_engine("m1"))
        await asyncio.wait_for(started.wait(), 1.0)
        t2 = asyncio.create_task(pool.get_engine("m1"))
        await asyncio.sleep(0.01)
        release.set()

        with pytest.raises(RuntimeError, match="weights exploded"):
            await t1
        with pytest.raises(RuntimeError, match="weights exploded"):
            await t2
        # Reservation refunded; entry usable for a retry.
        assert pool._loading_reserved_bytes == 0
        assert pool._entries["m1"].load_future is None

    @pytest.mark.asyncio
    async def test_lease_taken_after_load(self, monkeypatch, tmp_path):
        pool = _make_pool("m1", model_root=tmp_path)

        async def fake_load(model_id, force_lm=False):
            entry = pool._entries[model_id]
            entry.engine = object()
            entry.is_loading = False

        monkeypatch.setattr(pool, "_load_engine", fake_load)
        engine = await pool.get_engine("m1", _lease=True)
        assert engine is pool._entries["m1"].engine
        assert pool._entries["m1"].in_use == 1
        await pool.release_engine("m1")
        assert pool._entries["m1"].in_use == 0


class TestDiscoverPreservesLoading:
    def test_loading_entry_survives_rediscovery(self, tmp_path):
        pool = _make_pool("m1")
        pool._entries["m1"].is_loading = True
        original = pool._entries["m1"]

        # Empty dir: nothing discovered; stale removal must skip the
        # mid-load entry (deleting it orphans the in-flight load).
        pool.discover_models(str(tmp_path))
        assert pool._entries.get("m1") is original

    def test_unloaded_idle_entry_still_pruned(self, tmp_path):
        pool = _make_pool("m1")
        pool.discover_models(str(tmp_path))
        assert "m1" not in pool._entries


class TestEngineCoreStopReleasesWaiters:
    @pytest.mark.asyncio
    async def test_stop_fails_inflight_requests(self):
        from omlx.engine_core import EngineCore
        from omlx.output_collector import RequestOutputCollector

        core = EngineCore.__new__(EngineCore)
        core._running = True
        core._wake_event = None
        core._task = None
        core._loop = None
        collector = RequestOutputCollector(aggregate=True)
        event = asyncio.Event()
        core._output_collectors = {"r1": collector}
        core._finished_events = {"r1": event}

        await core.stop()

        assert event.is_set()
        out = await asyncio.wait_for(collector.get(), 1.0)
        assert out.finished
        assert out.finish_reason == "error"
        assert "stopped" in (out.error or "")
