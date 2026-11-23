from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import pytest

from ocr.core.errors import ImageDecodeError, ImageQualityError
from ocr.core.quality import (
    ImageQualityReport,
    QualityThresholds,
    assess_quality,
    ensure_quality,
)
from ocr.engines.tesseract import TesseractEngine


def _receipt_like() -> np.ndarray:
    image = np.full((200, 400, 3), 180, dtype=np.uint8)
    for top in range(20, 180, 20):
        image[top : top + 8, 30:370] = 20
    return image


def _blurred() -> np.ndarray:
    noise = np.random.default_rng(0).integers(0, 256, (200, 400, 3), dtype=np.uint8)
    return cv2.GaussianBlur(noise, (51, 51), 0)


def _low_contrast() -> np.ndarray:
    return np.full((200, 400, 3), 128, dtype=np.uint8)


def _glare() -> np.ndarray:
    image = np.full((200, 400, 3), 255, dtype=np.uint8)
    image[:, :200] = 200
    return image


def test_sharp_receipt_like_image_is_accepted() -> None:
    report = assess_quality(_receipt_like(), QualityThresholds())

    assert isinstance(report, ImageQualityReport)
    assert report.accepted is True
    assert report.sharpness >= 50.0
    assert report.contrast >= 20.0
    assert report.glare_ratio <= 0.35


def test_blurred_image_is_rejected_for_low_sharpness() -> None:
    report = assess_quality(_blurred(), QualityThresholds())

    assert report.accepted is False
    assert report.sharpness < 50.0


def test_low_contrast_image_is_rejected() -> None:
    report = assess_quality(_low_contrast(), QualityThresholds())

    assert report.accepted is False
    assert report.contrast < 20.0


def test_glare_image_is_rejected() -> None:
    report = assess_quality(_glare(), QualityThresholds())

    assert report.accepted is False
    assert report.glare_ratio > 0.35


def test_ensure_quality_returns_report_for_accepted_image() -> None:
    report = ensure_quality(_receipt_like(), QualityThresholds())

    assert report.accepted is True


def test_ensure_quality_reports_measured_metrics() -> None:
    with pytest.raises(ImageQualityError) as raised:
        ensure_quality(_low_contrast(), QualityThresholds())

    message = str(raised.value)
    assert "sharpness=" in message
    assert "contrast=" in message
    assert "glare=" in message


def test_quality_error_is_reported_as_undecodable_image() -> None:
    assert issubclass(ImageQualityError, ImageDecodeError)


def test_permissive_thresholds_accept_featureless_image() -> None:
    thresholds = QualityThresholds(min_sharpness=0, min_contrast=0, max_glare_ratio=1)

    assert assess_quality(_low_contrast(), thresholds).accepted is True


def test_tesseract_decodes_before_loading_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    backend_calls: list[bool] = []

    def fake_module(self: TesseractEngine) -> Any:
        backend_calls.append(True)
        raise AssertionError("backend must not be loaded before decoding")

    monkeypatch.setattr(TesseractEngine, "_module", fake_module)
    engine = TesseractEngine(quality=QualityThresholds())

    with pytest.raises(ImageDecodeError):
        engine.recognize_blocks(b"not an image")

    assert backend_calls == []


def test_tesseract_rejects_poor_quality_before_loading_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend_calls: list[bool] = []

    def fake_module(self: TesseractEngine) -> Any:
        backend_calls.append(True)
        raise AssertionError("backend must not be loaded before the quality gate")

    monkeypatch.setattr(TesseractEngine, "_module", fake_module)
    engine = TesseractEngine(quality=QualityThresholds(min_sharpness=1e9))
    success, buffer = cv2.imencode(".png", _receipt_like())
    assert success

    with pytest.raises(ImageQualityError):
        engine.recognize_blocks(buffer.tobytes())

    assert backend_calls == []
