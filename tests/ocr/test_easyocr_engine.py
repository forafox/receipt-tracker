from __future__ import annotations

import os

import cv2
import numpy as np
import pytest

pytest.importorskip("easyocr")

from ocr.dataset_rec import DATASET_ROOT, EngRecognitionDataset  # noqa: E402
from ocr.easyocr_engine import EasyOcrEngine  # noqa: E402

pytestmark = pytest.mark.slow

_RECOGNITION_SPLIT = DATASET_ROOT / "recognition/eng_arab/test/test"

if os.environ.get("OCR_RUN_EASYOCR") != "1":
    pytest.skip("set OCR_RUN_EASYOCR=1 to run EasyOCR tests", allow_module_level=True)
if not _RECOGNITION_SPLIT.is_dir():
    pytest.skip("dataset not downloaded", allow_module_level=True)


def _encode(crop: np.ndarray) -> bytes:
    success, buffer = cv2.imencode(".png", crop)
    assert success
    return buffer.tobytes()


def test_easyocr_reads_english_crop() -> None:
    dataset = EngRecognitionDataset("test")
    engine = EasyOcrEngine(languages=["en"], gpu=False)

    crop, _ = dataset[0]
    result = engine.recognize_blocks(_encode(crop))

    assert result.image_width == crop.shape[1]
    assert result.image_height == crop.shape[0]
    assert result.image_width is not None
    assert result.image_height is not None
    for block in result.blocks:
        assert block.text
        xs = [point[0] for point in block.bbox]
        ys = [point[1] for point in block.bbox]
        assert 0 <= min(xs) and max(xs) <= result.image_width
        assert 0 <= min(ys) and max(ys) <= result.image_height
