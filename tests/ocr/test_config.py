from pathlib import Path

from ocr import OcrSettings


def test_settings_ignore_other_modules_variables(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("OCR_ENGINE=easyocr\nBOT_TOKEN=secret\n")

    settings = OcrSettings(_env_file=env_file)

    assert settings.engine == "easyocr"
