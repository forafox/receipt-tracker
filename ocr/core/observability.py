from __future__ import annotations

import logging

from common.ocr import OcrResult
from ocr.core.ordering import blocks_to_text

logger = logging.getLogger("ocr")


def log_detected_text(engine_name: str, result: OcrResult) -> None:
    """Log the text an OCR engine detected in an image.

    Args:
        engine_name: Identifier of the engine that produced the result.
        result: Recognized blocks rendered into reading-order text.
    """
    text = blocks_to_text(result)
    logger.info(
        "OCR engine %s detected %d block(s):\n%s",
        engine_name,
        len(result.blocks),
        text or "<no text>",
    )
