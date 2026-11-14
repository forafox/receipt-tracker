from __future__ import annotations

import builtins
import sys
import threading
import time
import types
from typing import Any

import cv2
import numpy as np
import pytest

from ocr.engines.easyocr import EasyOcrEngine
from ocr.engines.tesseract import TesseractEngine

WORKER_COUNT = 8
TIMEOUT = 5.0


def _image_bytes() -> bytes:
    success, buffer = cv2.imencode(".png", np.zeros((10, 10, 3), dtype=np.uint8))
    assert success
    return buffer.tobytes()


def test_concurrent_first_use_creates_single_easyocr_reader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader_started = threading.Event()
    release_reader = threading.Event()
    created_readers = 0
    counter_lock = threading.Lock()

    class FakeReader:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            nonlocal created_readers
            with counter_lock:
                created_readers += 1
            reader_started.set()
            if not release_reader.wait(TIMEOUT):
                raise TimeoutError("reader construction was not released")

        def readtext(self, image: np.ndarray) -> list[tuple[Any, str, float]]:
            return []

    fake_module: Any = types.ModuleType("easyocr")
    fake_module.Reader = FakeReader
    monkeypatch.setitem(sys.modules, "easyocr", fake_module)

    engine = EasyOcrEngine()
    image = _image_bytes()
    start = threading.Barrier(WORKER_COUNT)
    errors: list[BaseException] = []

    def worker() -> None:
        start.wait(timeout=TIMEOUT)
        try:
            engine.recognize_blocks(image)
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=worker) for _ in range(WORKER_COUNT)]
    for thread in threads:
        thread.start()

    assert reader_started.wait(timeout=TIMEOUT)
    time.sleep(0.1)
    release_reader.set()
    for thread in threads:
        thread.join(timeout=TIMEOUT)

    assert not any(thread.is_alive() for thread in threads)
    assert errors == []
    assert created_readers == 1


def test_concurrent_first_use_imports_pytesseract_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import_started = threading.Event()
    release_import = threading.Event()
    import_calls = 0
    counter_lock = threading.Lock()
    original_import = builtins.__import__

    fake_module = types.ModuleType("pytesseract")

    def fake_import(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        nonlocal import_calls
        if name == "pytesseract":
            with counter_lock:
                import_calls += 1
            import_started.set()
            if not release_import.wait(TIMEOUT):
                raise TimeoutError("pytesseract import was not released")
            return fake_module
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    engine = TesseractEngine()
    start = threading.Barrier(WORKER_COUNT)
    errors: list[BaseException] = []

    def worker() -> None:
        start.wait(timeout=TIMEOUT)
        try:
            engine._module()
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=worker) for _ in range(WORKER_COUNT)]
    for thread in threads:
        thread.start()

    assert import_started.wait(timeout=TIMEOUT)
    time.sleep(0.1)
    release_import.set()
    for thread in threads:
        thread.join(timeout=TIMEOUT)

    assert not any(thread.is_alive() for thread in threads)
    assert errors == []
    assert import_calls == 1
