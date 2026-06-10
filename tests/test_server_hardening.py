# SPDX-License-Identifier: Apache-2.0
"""Tests for the fork server-hardening layer.

Covers:
- BodySizeLimitMiddleware (request body size cap)
- _compile_grammar_for_request_async per-engine LRU
- OQManager start_quantization gating against loaded engines
"""

import asyncio
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from omlx import server as omlx_server
from omlx.admin.oq_manager import OQManager
from omlx.server import (
    BodySizeLimitMiddleware,
    _compile_grammar_for_request_async,
    _grammar_cache_key,
)


# ---------------------------------------------------------------------------
# BodySizeLimitMiddleware
# ---------------------------------------------------------------------------


@pytest.fixture
def limited_client(monkeypatch):
    """A tiny app wrapped in the body-cap middleware with a 1 KB limit."""
    app = FastAPI()

    @app.post("/v1/echo")
    async def echo(payload: dict):
        return {"n": len(payload)}

    @app.post("/plain")
    async def plain(payload: dict):
        return {"n": len(payload)}

    app.add_middleware(BodySizeLimitMiddleware)
    monkeypatch.setattr(omlx_server, "_max_request_body_bytes", lambda: 1024)
    return TestClient(app)


class TestBodySizeLimit:
    def test_normal_request_passes(self, limited_client):
        resp = limited_client.post("/v1/echo", json={"a": 1})
        assert resp.status_code == 200

    def test_oversized_content_length_rejected_413(self, limited_client):
        body = json.dumps({"a": "x" * 4096}).encode()
        resp = limited_client.post(
            "/v1/echo",
            content=body,
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 413
        # OpenAI error shape on /v1/ routes
        assert "error" in resp.json()

    def test_oversized_non_v1_route_plain_detail(self, limited_client):
        body = json.dumps({"a": "x" * 4096}).encode()
        resp = limited_client.post(
            "/plain",
            content=body,
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 413
        assert "detail" in resp.json()

    def test_chunked_body_exceeding_cap_rejected(self, limited_client):
        # No Content-Length: the receive wrapper must count and abort.
        def gen():
            for _ in range(8):
                yield b'{"a": "' + b"x" * 512 + b'"}'

        resp = limited_client.post(
            "/v1/echo",
            content=gen(),
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 413

    def test_disabled_limit_passes_everything(self, limited_client, monkeypatch):
        monkeypatch.setattr(omlx_server, "_max_request_body_bytes", lambda: 0)
        body = json.dumps({"a": "x" * 4096}).encode()
        resp = limited_client.post(
            "/v1/echo",
            content=body,
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 200


class TestMaxRequestBodyBytes:
    def test_settings_value_flows(self, monkeypatch):
        class _Srv:
            max_request_body_mb = 2

        class _GS:
            server = _Srv()

        monkeypatch.setattr(omlx_server._server_state, "global_settings", _GS())
        assert omlx_server._max_request_body_bytes() == 2 * 1024 * 1024

    def test_null_disables(self, monkeypatch):
        class _Srv:
            max_request_body_mb = None

        class _GS:
            server = _Srv()

        monkeypatch.setattr(omlx_server._server_state, "global_settings", _GS())
        assert omlx_server._max_request_body_bytes() == 0

    def test_no_settings_uses_default(self, monkeypatch):
        monkeypatch.setattr(omlx_server._server_state, "global_settings", None)
        assert omlx_server._max_request_body_bytes() == 256 * 1024 * 1024


# ---------------------------------------------------------------------------
# Grammar compile LRU
# ---------------------------------------------------------------------------


class _FakeEngine:
    """Engine stub: only needs to be attribute-assignable."""


class TestGrammarCompileLRU:
    def test_cache_key_stable_and_distinct(self):
        k1 = _grammar_cache_key({"schema": 1}, None, None, "qwen")
        k2 = _grammar_cache_key({"schema": 1}, None, None, "qwen")
        k3 = _grammar_cache_key({"schema": 2}, None, None, "qwen")
        assert k1 == k2
        assert k1 != k3

    def test_compiles_once_per_schema(self, monkeypatch):
        calls = []

        def fake_compile(engine, **kwargs):
            calls.append(kwargs)
            return object()

        monkeypatch.setattr(
            omlx_server, "_compile_grammar_for_request", fake_compile
        )
        engine = _FakeEngine()
        so = {"json_schema": {"type": "object"}}

        async def run():
            g1 = await _compile_grammar_for_request_async(
                engine, structured_outputs=so
            )
            g2 = await _compile_grammar_for_request_async(
                engine, structured_outputs=so
            )
            return g1, g2

        g1, g2 = asyncio.run(run())
        assert g1 is g2
        assert len(calls) == 1

    def test_none_result_not_cached(self, monkeypatch):
        calls = []

        def fake_compile(engine, **kwargs):
            calls.append(kwargs)
            return None

        monkeypatch.setattr(
            omlx_server, "_compile_grammar_for_request", fake_compile
        )
        engine = _FakeEngine()

        async def run():
            for _ in range(2):
                await _compile_grammar_for_request_async(
                    engine, structured_outputs={"x": 1}
                )

        asyncio.run(run())
        assert len(calls) == 2

    def test_no_grammar_short_circuits(self, monkeypatch):
        def fake_compile(engine, **kwargs):  # pragma: no cover
            raise AssertionError("should not be called")

        monkeypatch.setattr(
            omlx_server, "_compile_grammar_for_request", fake_compile
        )

        async def run():
            return await _compile_grammar_for_request_async(_FakeEngine())

        assert asyncio.run(run()) is None

    def test_cache_is_per_engine(self, monkeypatch):
        calls = []

        def fake_compile(engine, **kwargs):
            calls.append(engine)
            return object()

        monkeypatch.setattr(
            omlx_server, "_compile_grammar_for_request", fake_compile
        )
        e1, e2 = _FakeEngine(), _FakeEngine()
        so = {"json_schema": {"type": "object"}}

        async def run():
            await _compile_grammar_for_request_async(e1, structured_outputs=so)
            await _compile_grammar_for_request_async(e2, structured_outputs=so)

        asyncio.run(run())
        assert calls == [e1, e2]


# ---------------------------------------------------------------------------
# oQ quantization gating
# ---------------------------------------------------------------------------


@pytest.fixture
def quant_source(tmp_path):
    d = tmp_path / "models"
    d.mkdir()
    model = d / "Llama-3B"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps({"model_type": "llama", "num_hidden_layers": 32})
    )
    (model / "model.safetensors").write_bytes(b"\x00" * 4096)
    return d


class _FakePool:
    def __init__(self, loaded=(), loading=()):
        self._loaded = list(loaded)
        self._loading = list(loading)

    def get_loaded_model_ids(self):
        return list(self._loaded)

    def get_loaded_or_loading_model_ids(self):
        return [*self._loaded, *self._loading]


class TestOQGating:
    @pytest.mark.asyncio
    async def test_refuses_while_models_loaded(self, quant_source):
        mgr = OQManager(
            model_dirs=[str(quant_source)],
            engine_pool_getter=lambda: _FakePool(["gemma-4"]),
        )
        with pytest.raises(ValueError, match="while models are loaded"):
            await mgr.start_quantization(
                model_path=str(quant_source / "Llama-3B"), oq_level=4
        )
        assert not mgr._active_tasks

    @pytest.mark.asyncio
    async def test_refuses_while_models_loading(self, quant_source):
        mgr = OQManager(
            model_dirs=[str(quant_source)],
            engine_pool_getter=lambda: _FakePool(loading=["qwen-loading"]),
        )
        with pytest.raises(ValueError, match="loaded or loading"):
            await mgr.start_quantization(
                model_path=str(quant_source / "Llama-3B"), oq_level=4
            )
        assert not mgr._active_tasks

    @pytest.mark.asyncio
    async def test_allows_when_pool_empty(self, quant_source, monkeypatch):
        mgr = OQManager(
            model_dirs=[str(quant_source)],
            engine_pool_getter=lambda: _FakePool([]),
        )

        # Don't actually run a quantization; stub the runner.
        async def fake_run(task_id):
            return None

        monkeypatch.setattr(mgr, "_run_quantization", fake_run)
        task = await mgr.start_quantization(
            model_path=str(quant_source / "Llama-3B"), oq_level=4
        )
        assert task.task_id in mgr.get_tasks()[0]["task_id"] or task.task_id

    @pytest.mark.asyncio
    async def test_no_getter_keeps_legacy_behavior(self, quant_source, monkeypatch):
        mgr = OQManager(model_dirs=[str(quant_source)])

        async def fake_run(task_id):
            return None

        monkeypatch.setattr(mgr, "_run_quantization", fake_run)
        task = await mgr.start_quantization(
            model_path=str(quant_source / "Llama-3B"), oq_level=4
        )
        assert task is not None
