from __future__ import annotations

import sys
import types
from typing import Any

import cv2
import numpy as np
import pytest

from ocr.core.errors import ImageDecodeError, OcrEngineUnavailableError
from ocr.engines.easyocr import EasyOcrEngine


def _image_bytes() -> bytes:
    success, buffer = cv2.imencode(".png", np.zeros((10, 10, 3), dtype=np.uint8))
    assert success
    return buffer.tobytes()


def _install_fake_easyocr(monkeypatch: pytest.MonkeyPatch, reader: Any) -> None:
    fake_module: Any = types.ModuleType("easyocr")
    fake_module.Reader = reader
    monkeypatch.setitem(sys.modules, "easyocr", fake_module)


def test_reader_initialization_failure_is_engine_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingReader:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("model load failed")

    _install_fake_easyocr(monkeypatch, FailingReader)

    with pytest.raises(OcrEngineUnavailableError) as raised:
        EasyOcrEngine().recognize_blocks(_image_bytes())

    assert isinstance(raised.value.__cause__, RuntimeError)


def test_recognition_failure_is_engine_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    class FailingReader:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def readtext(self, image: np.ndarray) -> list[tuple[Any, str, float]]:
            raise RuntimeError("inference failed")

    _install_fake_easyocr(monkeypatch, FailingReader)

    with pytest.raises(OcrEngineUnavailableError) as raised:
        EasyOcrEngine().recognize_blocks(_image_bytes())

    assert isinstance(raised.value.__cause__, RuntimeError)


def test_invalid_image_is_reported_before_loading_reader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader_created = False

    class Reader:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            nonlocal reader_created
            reader_created = True

        def readtext(self, image: np.ndarray) -> list[tuple[Any, str, float]]:
            return []

    _install_fake_easyocr(monkeypatch, Reader)

    with pytest.raises(ImageDecodeError):
        EasyOcrEngine().recognize_blocks(b"not an image")

    assert not reader_created
