# SPDX-License-Identifier: Apache-2.0
"""fork regression tests: VLM media prep off the MLX executor + size caps."""

import asyncio
import base64
import io
import threading

import pytest

import omlx.utils.image as image_utils
from omlx.engine.vlm import VLMBatchedEngine
from omlx.exceptions import InvalidRequestError
from omlx.utils.image import load_image


class TestMediaExecutorSplit:
    def _bare_engine(self) -> VLMBatchedEngine:
        engine = VLMBatchedEngine.__new__(VLMBatchedEngine)
        engine._media_executor = None
        return engine

    def test_extraction_runs_on_media_thread(self):
        engine = self._bare_engine()
        seen = {}

        async def run():
            # Wrap to capture which thread the extraction runs on.
            loop = asyncio.get_running_loop()

            def probe(messages):
                seen["thread"] = threading.current_thread().name
                return ([], [], [])

            return await loop.run_in_executor(
                engine._get_media_executor(), probe, []
            )

        asyncio.run(run())
        assert seen["thread"].startswith("vlm-media")

    def test_extract_chat_media_async_returns_media_tuple(self):
        engine = self._bare_engine()
        messages = [{"role": "user", "content": "hello"}]

        text, images, audio = asyncio.run(engine._extract_chat_media_async(messages))
        assert text == [{"role": "user", "content": "hello"}]
        assert images == [] and audio == []

    def test_media_executor_recreated_after_shutdown(self):
        engine = self._bare_engine()
        first = engine._get_media_executor()
        first.shutdown(wait=False)
        second = engine._get_media_executor()
        assert second is not first
        # And it works.
        assert second.submit(lambda: 42).result() == 42


class TestImageSizeCaps:
    def test_oversized_data_uri_rejected_before_decode(self, monkeypatch):
        monkeypatch.setattr(image_utils, "MAX_IMAGE_BYTES", 16)
        big = "data:image/png;base64," + "A" * 64
        with pytest.raises(InvalidRequestError, match="exceeds the"):
            image_utils.load_image(big)

    def test_small_data_uri_still_loads(self):
        # 1x1 red PNG
        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (1, 1), (255, 0, 0)).save(buf, format="PNG")
        uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        img = load_image(uri)
        assert img.size == (1, 1)
        assert img.mode == "RGB"
