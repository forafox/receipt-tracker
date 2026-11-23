from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from common.ocr import OcrEngine, OcrResult, TextBlock
from ocr.core.config import OcrSettings
from ocr.core.ordering import blocks_to_text
from ocr.core.postprocess import PostprocessSettings
from ocr.core.quality import QualityThresholds
from ocr.engines.easyocr import EasyOcrEngine
from ocr.engines.factory import build_ocr_engine
from ocr.engines.tesseract import TesseractEngine

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class FakeEngine:
    def __init__(self, result: OcrResult) -> None:
        self.name = "fake"
        self._result = result
        self.received: bytes | None = None

    def recognize_blocks(self, image: bytes) -> OcrResult:
        self.received = image
        return self._result

    def recognize(self, image: bytes) -> str:
        return blocks_to_text(self._result)


def _result() -> OcrResult:
    block = TextBlock(text="ИТОГО 250,00", bbox=[(0, 0), (10, 0), (10, 5), (0, 5)])
    return OcrResult(blocks=[block], image_width=20, image_height=10)


def test_fake_engine_satisfies_ocr_engine_interface() -> None:
    engine: OcrEngine = FakeEngine(_result())

    assert engine.recognize_blocks(b"image") == _result()
    assert engine.recognize(b"image") == "ИТОГО 250,00"


def test_build_ocr_engine_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown OCR engine"):
        build_ocr_engine(OcrSettings(engine="missing"))


def test_factory_wires_quality_gate_from_settings() -> None:
    settings = OcrSettings(
        engine="tesseract",
        quality_min_sharpness=12.0,
        quality_min_contrast=3.0,
        quality_max_glare_ratio=0.1,
    )

    engine = build_ocr_engine(settings)

    assert isinstance(engine, TesseractEngine)
    assert engine._quality == QualityThresholds(12.0, 3.0, 0.1)


def test_factory_can_disable_quality_gate() -> None:
    engine = build_ocr_engine(OcrSettings(engine="easyocr", quality_check_enabled=False))

    assert isinstance(engine, EasyOcrEngine)
    assert engine._quality is None


def test_factory_wires_postprocess_settings() -> None:
    settings = OcrSettings(
        engine="tesseract",
        postprocess_min_box_area=4.0,
        postprocess_nms_iou=0.3,
    )

    engine = build_ocr_engine(settings)

    assert isinstance(engine, TesseractEngine)
    assert engine._postprocess == PostprocessSettings(4.0, 0.3)


def test_factory_can_disable_postprocess() -> None:
    engine = build_ocr_engine(OcrSettings(engine="easyocr", postprocess_enabled=False))

    assert isinstance(engine, EasyOcrEngine)
    assert engine._postprocess is None


def test_importing_ocr_does_not_import_engine_dependencies() -> None:
    code = (
        "import sys, ocr;"
        "assert 'easyocr' not in sys.modules, 'easyocr imported eagerly';"
        "assert 'pytesseract' not in sys.modules, 'pytesseract imported eagerly'"
    )

    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=PROJECT_ROOT, capture_output=True, text=True
    )

    assert completed.returncode == 0, completed.stderr
