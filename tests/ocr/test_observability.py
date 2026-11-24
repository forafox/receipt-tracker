from __future__ import annotations

import logging
import sys
import types
from typing import Any

import cv2
import numpy as np
import pytest

from common.ocr import OcrResult, TextBlock
from ocr.core.observability import log_detected_text
from ocr.engines.easyocr import EasyOcrEngine
from ocr.engines.tesseract import TesseractEngine


def _image_bytes() -> bytes:
    success, buffer = cv2.imencode(".png", np.zeros((10, 10, 3), dtype=np.uint8))
    assert success
    return buffer.tobytes()


def _install_fake_easyocr(monkeypatch: pytest.MonkeyPatch, readtext: Any) -> None:
    class Reader:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def readtext(self, image: np.ndarray) -> Any:
            return readtext(image)

    fake_module: Any = types.ModuleType("easyocr")
    fake_module.Reader = Reader
    monkeypatch.setitem(sys.modules, "easyocr", fake_module)


def test_log_detected_text_renders_reading_order(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="ocr")
    result = OcrResult(
        blocks=[
            TextBlock(text="250,00", bbox=[(50, 0), (90, 0), (90, 10), (50, 10)]),
            TextBlock(text="ИТОГО", bbox=[(0, 0), (40, 0), (40, 10), (0, 10)]),
        ]
    )

    log_detected_text("fake", result)

    [record] = caplog.records
    assert record.name == "ocr"
    assert "fake" in record.getMessage()
    assert "ИТОГО 250,00" in record.getMessage()


def test_log_detected_text_marks_empty_result(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="ocr")

    log_detected_text("fake", OcrResult())

    [record] = caplog.records
    assert "<no text>" in record.getMessage()


def test_easyocr_logs_detected_text_once(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="ocr")
    quad = [[0, 0], [10, 0], [10, 5], [0, 5]]
    _install_fake_easyocr(monkeypatch, lambda image: [(quad, "ИТОГО", 0.9)])

    assert EasyOcrEngine().recognize(_image_bytes()) == "ИТОГО"

    [record] = caplog.records
    assert "easyocr" in record.getMessage()
    assert "ИТОГО" in record.getMessage()


def test_tesseract_logs_detected_text_once(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="ocr")
    fake_module: Any = types.ModuleType("pytesseract")
    fake_module.Output = types.SimpleNamespace(DICT="dict")
    fake_module.image_to_data = lambda *args, **kwargs: {
        "text": ["ИТОГО", "250,00"],
        "level": [5, 5],
        "conf": ["95", "90"],
        "left": [0, 50],
        "top": [0, 0],
        "width": [40, 40],
        "height": [10, 10],
        "block_num": [1, 1],
        "par_num": [1, 1],
        "line_num": [1, 1],
    }
    monkeypatch.setattr(TesseractEngine, "_module", lambda self: fake_module)

    assert TesseractEngine(languages="eng").recognize(_image_bytes()) == "ИТОГО 250,00"

    [record] = caplog.records
    assert "tesseract" in record.getMessage()
    assert "ИТОГО 250,00" in record.getMessage()
