from __future__ import annotations

from pathlib import Path

import pytest

from common.ocr import OcrEngine, OcrResult, TextBlock
from ocr.core.config import OcrSettings
from ocr.service import OcrService, build_default_ocr_service
from pipeline import TextRecognizer


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
    recognizer: TextRecognizer = OcrService(engine=FakeEngine())

    assert engine.name == "fake"
    assert engine.recognize_blocks(b"image").blocks
    assert recognizer.recognize(b"image") == "ИТОГО"


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


def test_default_service_uses_default_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("OCR_ENGINE", raising=False)
    monkeypatch.chdir(tmp_path)

    service = build_default_ocr_service()

    assert isinstance(service, OcrService)
    assert service.settings.engine == "tesseract"
    assert service.name == "tesseract"
