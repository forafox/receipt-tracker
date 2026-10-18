from __future__ import annotations

import shutil

import cv2
import numpy as np
import pytest

pytest.importorskip("pytesseract")

from ocr.core.errors import OcrEngineUnavailableError  # noqa: E402
from ocr.datasets.recognition import DATASET_ROOT, EngRecognitionDataset  # noqa: E402
from ocr.engines.tesseract import TesseractEngine  # noqa: E402

pytestmark = pytest.mark.slow

_RECOGNITION_SPLIT = DATASET_ROOT / "recognition/eng_arab/test/test"

if shutil.which("tesseract") is None:
    pytest.skip("tesseract binary not installed", allow_module_level=True)
if not _RECOGNITION_SPLIT.is_dir():
    pytest.skip("dataset not downloaded", allow_module_level=True)


def _encode(crop: np.ndarray) -> bytes:
    scaled = cv2.resize(crop, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    success, buffer = cv2.imencode(".png", scaled)
    assert success
    return buffer.tobytes()


def test_tesseract_reads_english_crops() -> None:
    dataset = EngRecognitionDataset("test")
    engine = TesseractEngine(languages="eng")

    recognized = False
    for index in range(min(len(dataset), 5)):
        crop, _ = dataset[index]
        try:
            result = engine.recognize_blocks(_encode(crop))
        except OcrEngineUnavailableError as error:
            pytest.skip(str(error))
        assert result.image_width is not None
        assert result.image_height is not None
        for block in result.blocks:
            assert block.text
            xs = [point[0] for point in block.bbox]
            ys = [point[1] for point in block.bbox]
            assert 0 <= min(xs) and max(xs) <= result.image_width
            assert 0 <= min(ys) and max(ys) <= result.image_height
            recognized = True

    assert recognized, "tesseract returned no text for any crop"
