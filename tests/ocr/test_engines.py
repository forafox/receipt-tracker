from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from common.ocr import OcrEngine, OcrResult, TextBlock
from ocr.core.config import OcrSettings
from ocr.core.ordering import blocks_to_text
from ocr.engines.factory import build_ocr_engine
from pipeline import TextRecognizer

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


def test_fake_engine_satisfies_both_interfaces() -> None:
    engine: OcrEngine = FakeEngine(_result())
    recognizer: TextRecognizer = FakeEngine(_result())

    assert engine.recognize_blocks(b"image") == _result()
    assert recognizer.recognize(b"image") == "ИТОГО 250,00"


def test_build_ocr_engine_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown OCR engine"):
        build_ocr_engine(OcrSettings(engine="missing"))


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
