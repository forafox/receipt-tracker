from __future__ import annotations

import numpy as np
import pytest

from ocr.dataset_det import (
    DATASET_ROOT,
    EngDetectionDataset,
    RuDetectionDataset,
    arabic_ratio,
    load_ocr_index,
    receipt_uuid,
)
from ocr.dataset_rec import (
    EngRecognitionDataset,
    RuRecognitionDataset,
    contains_arabic,
)

pytestmark = pytest.mark.skipif(not DATASET_ROOT.is_dir(), reason="dataset not downloaded")


def _check_boxes(image: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> None:
    height, width = image.shape[:2]
    for x1, y1, x2, y2 in boxes:
        assert 0 <= x1 < x2 <= width
        assert 0 <= y1 < y2 <= height


def test_eng_recognition_excludes_arabic() -> None:
    dataset = EngRecognitionDataset("test")
    assert len(dataset) > 0
    for index in range(min(len(dataset), 10)):
        image, text = dataset[index]
        assert image.size > 0
        assert text
        assert not contains_arabic(text)


def test_eng_detection_boxes() -> None:
    dataset = EngDetectionDataset("test")
    assert len(dataset) > 0
    for index in range(min(len(dataset), 5)):
        image, boxes = dataset[index]
        assert image.size > 0
        assert len(boxes) > 0
        _check_boxes(image, boxes)


def test_eng_detection_excludes_arabic() -> None:
    filtered = EngDetectionDataset("test", exclude_arabic=True)
    unfiltered = EngDetectionDataset("test", exclude_arabic=False)
    assert 0 < len(filtered) < len(unfiltered)

    index = load_ocr_index(DATASET_ROOT)
    kept = set(filtered.image_paths)
    for path in unfiltered.image_paths:
        texts = index.get(receipt_uuid(path.stem), [])
        if path in kept:
            assert arabic_ratio(texts) <= 0.25
        else:
            assert arabic_ratio(texts) > 0.25


def test_ru_detection_boxes() -> None:
    dataset = RuDetectionDataset("test")
    assert len(dataset) > 0
    for index in range(min(len(dataset), 5)):
        image, boxes = dataset[index]
        assert image.size > 0
        assert len(boxes) > 0
        _check_boxes(image, boxes)


def test_ru_recognition_crops() -> None:
    dataset = RuRecognitionDataset("test")
    assert len(dataset) > 0
    for index in range(min(len(dataset), 5)):
        image, text = dataset[index]
        assert image.size > 0
        assert text
        assert not contains_arabic(text)
