from pathlib import Path

import pytest
from pydantic import ValidationError

from ocr import OcrSettings


def test_settings_ignore_other_modules_variables(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("OCR_ENGINE=easyocr\nBOT_TOKEN=secret\n")

    settings = OcrSettings(_env_file=env_file)

    assert settings.engine == "easyocr"


def test_quality_settings_are_enabled_with_conservative_defaults() -> None:
    settings = OcrSettings(_env_file=None)

    assert settings.quality_check_enabled is True
    assert settings.quality_min_sharpness == 50.0
    assert settings.quality_min_contrast == 20.0
    assert settings.quality_max_glare_ratio == 0.35


def test_adaptive_mode_is_disabled_with_safe_defaults() -> None:
    settings = OcrSettings(_env_file=None)

    assert settings.adaptive is False
    assert settings.adaptive_psm_modes == [3, 4, 6]


def test_adaptive_mode_is_loaded_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OCR_ADAPTIVE", "true")
    monkeypatch.setenv("OCR_ADAPTIVE_PSM_MODES", "[4,6,4]")

    settings = OcrSettings(_env_file=None)

    assert settings.adaptive is True
    assert settings.adaptive_psm_modes == [4, 6]


@pytest.mark.parametrize("modes", ["[]", "[0]", "[14]"])
def test_adaptive_mode_rejects_invalid_psm_modes(
    monkeypatch: pytest.MonkeyPatch, modes: str
) -> None:
    monkeypatch.setenv("OCR_ADAPTIVE_PSM_MODES", modes)

    with pytest.raises(ValidationError):
        OcrSettings(_env_file=None)


def test_quality_settings_are_loaded_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OCR_QUALITY_CHECK_ENABLED", "false")
    monkeypatch.setenv("OCR_QUALITY_MIN_SHARPNESS", "12.5")

    settings = OcrSettings(_env_file=None)

    assert settings.quality_check_enabled is False
    assert settings.quality_min_sharpness == 12.5


def test_quality_settings_reject_invalid_glare_ratio() -> None:
    with pytest.raises(ValidationError):
        OcrSettings(_env_file=None, quality_max_glare_ratio=1.5)


def test_postprocess_is_enabled_with_defaults() -> None:
    settings = OcrSettings(_env_file=None)

    assert settings.postprocess_enabled is True
    assert settings.postprocess_min_box_area == 9.0
    assert settings.postprocess_nms_iou == 0.5


def test_postprocess_settings_are_loaded_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OCR_POSTPROCESS_ENABLED", "false")
    monkeypatch.setenv("OCR_POSTPROCESS_MIN_BOX_AREA", "4.0")
    monkeypatch.setenv("OCR_POSTPROCESS_NMS_IOU", "0.3")

    settings = OcrSettings(_env_file=None)

    assert settings.postprocess_enabled is False
    assert settings.postprocess_min_box_area == 4.0
    assert settings.postprocess_nms_iou == 0.3


@pytest.mark.parametrize("nms_iou", [0.0, 1.5])
def test_postprocess_rejects_invalid_nms_iou(nms_iou: float) -> None:
    with pytest.raises(ValidationError):
        OcrSettings(_env_file=None, postprocess_nms_iou=nms_iou)


def test_postprocess_rejects_negative_box_area() -> None:
    with pytest.raises(ValidationError):
        OcrSettings(_env_file=None, postprocess_min_box_area=-1.0)
