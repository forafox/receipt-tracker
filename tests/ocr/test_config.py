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
