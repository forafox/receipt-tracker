from __future__ import annotations

import json
import urllib.error
from io import BytesIO

import cv2
import numpy as np
import pytest

from ocr.core.errors import OcrEngineUnavailableError
from ocr.engines.hunyuan_vision import HunyuanVisionEngine


def _image() -> bytes:
    success, encoded = cv2.imencode(".png", np.full((80, 40, 3), 255, dtype=np.uint8))
    assert success
    return encoded.tobytes()


def test_hunyuan_returns_one_synthetic_block_per_transcript_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = BytesIO(
        json.dumps({"choices": [{"message": {"content": "МАГАЗИН\n\nИТОГО 250,00"}}]}).encode()
    )
    captured: dict[str, object] = {}

    def open_request(request: object, **kwargs: object) -> BytesIO:
        captured["request"] = request
        captured["timeout"] = kwargs.get("timeout")
        return response

    monkeypatch.setattr("urllib.request.urlopen", open_request)

    result = HunyuanVisionEngine(timeout_seconds=35).recognize_blocks(_image())

    request = captured["request"]
    payload = json.loads(request.data)  # type: ignore[attr-defined]
    image_part = payload["messages"][0]["content"][0]
    assert [block.text for block in result.blocks] == ["МАГАЗИН", "ИТОГО 250,00"]
    assert result.image_width == 40
    assert result.image_height == 80
    assert result.blocks[0].bbox[0][1] < result.blocks[1].bbox[0][1]
    assert image_part["image_url"]["url"].startswith("data:image/png;base64,")
    assert payload["model"] == "HunyuanOCR"
    assert captured["timeout"] == 35


def test_hunyuan_reports_unavailable_endpoint_as_ocr_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_request(*args: object, **kwargs: object) -> None:
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", fail_request)

    with pytest.raises(OcrEngineUnavailableError, match="HunyuanOCR request failed"):
        HunyuanVisionEngine().recognize_blocks(_image())
