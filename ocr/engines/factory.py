from __future__ import annotations

from common.ocr import OcrEngine
from ocr.core.config import OcrSettings
from ocr.core.postprocess import PostprocessSettings


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
            tesseract_config=resolved.tesseract_config,
            postprocess=_postprocess_settings(resolved),
        )
    if resolved.engine == "easyocr":
        from ocr.engines.easyocr import EasyOcrEngine

        return EasyOcrEngine(
            languages=resolved.easyocr_languages,
            gpu=resolved.gpu,
            min_confidence=resolved.min_confidence,
            postprocess=_postprocess_settings(resolved),
        )
    raise ValueError(f"Unknown OCR engine: {resolved.engine!r}")


def _postprocess_settings(settings: OcrSettings) -> PostprocessSettings | None:
    if not settings.postprocess_enabled:
        return None
    return PostprocessSettings(
        min_box_area=settings.postprocess_min_box_area,
        nms_iou_threshold=settings.postprocess_nms_iou,
    )
