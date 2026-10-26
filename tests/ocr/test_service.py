from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from common.ocr import OcrEngine, OcrResult, TextBlock
from ocr.core.config import OcrSettings
from ocr.service import OcrService, build_default_ocr_service


class FakeEngine:
    def __init__(self) -> None:
        self.name = "fake"
        self.images: list[bytes] = []

    def recognize_blocks(self, image: bytes) -> OcrResult:
        self.images.append(image)
        block = TextBlock(text="ИТОГО", bbox=[(0, 0), (10, 0), (10, 5), (0, 5)])
        return OcrResult(blocks=[block])

    def recognize(self, image: bytes) -> str:
        self.images.append(image)
        return "ИТОГО"


def test_service_delegates_to_injected_engine() -> None:
    engine = FakeEngine()
    service = OcrService(engine=engine)

    result = service.process_receipt(b"image")

    assert result.blocks[0].text == "ИТОГО"
    assert service.recognize(b"image") == "ИТОГО"
    assert engine.images == [b"image", b"image"]


def test_service_satisfies_public_interfaces() -> None:
    engine: OcrEngine = OcrService(engine=FakeEngine())

    assert engine.name == "fake"
    assert engine.recognize_blocks(b"image").blocks
    assert engine.recognize(b"image") == "ИТОГО"


def test_engine_is_built_lazily(monkeypatch: pytest.MonkeyPatch) -> None:
    built: list[OcrSettings] = []

    def fake_build(settings: OcrSettings) -> FakeEngine:
        built.append(settings)
        return FakeEngine()

    monkeypatch.setattr("ocr.service.build_ocr_engine", fake_build)
    service = OcrService(OcrSettings(engine="fake"))

    assert built == []
    assert service.name == "fake"
    service.process_receipt(b"image")
    assert len(built) == 1


def test_concurrent_first_use_builds_engine_once(monkeypatch: pytest.MonkeyPatch) -> None:
    build_calls = 0
    build_entered = threading.Event()
    release_build = threading.Event()

    def fake_build(settings: OcrSettings) -> FakeEngine:
        nonlocal build_calls
        build_calls += 1
        build_entered.set()
        if not release_build.wait(timeout=5):
            raise TimeoutError("engine build was not released")
        return FakeEngine()

    monkeypatch.setattr("ocr.service.build_ocr_engine", fake_build)
    service = OcrService(OcrSettings(engine="fake"))
    worker_count = 8
    start = threading.Barrier(worker_count)
    results: list[OcrResult] = []
    errors: list[BaseException] = []
    results_lock = threading.Lock()

    def worker() -> None:
        start.wait(timeout=5)
        try:
            result = service.process_receipt(b"image")
        except BaseException as error:
            with results_lock:
                errors.append(error)
        else:
            with results_lock:
                results.append(result)

    threads = [threading.Thread(target=worker) for _ in range(worker_count)]
    for thread in threads:
        thread.start()

    assert build_entered.wait(timeout=5)
    time.sleep(0.1)
    release_build.set()
    for thread in threads:
        thread.join(timeout=5)

    assert not any(thread.is_alive() for thread in threads)
    assert errors == []
    assert len(results) == worker_count
    assert all(result.blocks[0].text == "ИТОГО" for result in results)
    assert build_calls == 1


def test_default_service_uses_default_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("OCR_ENGINE", raising=False)
    monkeypatch.chdir(tmp_path)

    service = build_default_ocr_service()

    assert isinstance(service, OcrService)
    assert service.settings.engine == "tesseract"
    assert service.name == "tesseract"
