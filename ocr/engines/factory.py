from __future__ import annotations

from common.ocr import OcrEngine
from ocr.core.config import OcrSettings


def build_ocr_engine(settings: OcrSettings | None = None) -> OcrEngine:
    """Build the OCR engine selected by settings.

    Args:
        settings: OCR settings; defaults to values loaded from the environment.

    Returns:
        Configured OCR engine.

    Raises:
        ValueError: If the configured engine name is unknown.
    """
    resolved = settings if settings is not None else OcrSettings()
    if resolved.engine == "tesseract":
        from ocr.engines.tesseract import TesseractEngine

        return TesseractEngine(
            languages=resolved.tesseract_languages,
            min_confidence=resolved.min_confidence,
        )
    if resolved.engine == "easyocr":
        from ocr.engines.easyocr import EasyOcrEngine

        return EasyOcrEngine(
            languages=resolved.easyocr_languages,
            gpu=resolved.gpu,
            min_confidence=resolved.min_confidence,
        )
    raise ValueError(f"Unknown OCR engine: {resolved.engine!r}")
