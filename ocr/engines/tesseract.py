from __future__ import annotations

import threading
from typing import Any

import cv2

from common.ocr import OcrResult, TextBlock
from ocr.core.errors import OcrEngineUnavailableError
from ocr.core.image import decode_image
from ocr.core.observability import log_detected_text
from ocr.core.ordering import blocks_to_text
from ocr.core.postprocess import PostprocessSettings, postprocess_blocks
from ocr.core.quality import QualityThresholds, ensure_quality

_WORD_LEVEL = 5


class TesseractEngine:
    """OCR engine backed by the Tesseract binary through ``pytesseract``.

    Attributes:
        name: Engine identifier used by configuration and factories.
    """

    name = "tesseract"

    def __init__(
        self,
        *,
        languages: str = "rus+eng",
        min_confidence: float = 0.0,
        tesseract_config: str = "",
        quality: QualityThresholds | None = None,
        postprocess: PostprocessSettings | None = None,
    ) -> None:
        """Create a Tesseract engine.

        Args:
            languages: Tesseract language string, for example ``rus+eng``.
            min_confidence: Minimum block confidence in the ``0..1`` range to keep a block.
            tesseract_config: Extra Tesseract configuration flags.
            quality: Optional quality gate applied after decoding and before loading
                the backend; ``None`` disables the gate.
            postprocess: Optional filtering and reading-order settings; ``None`` returns
                engine blocks unchanged.
        """
        self._languages = languages
        self._min_confidence = min_confidence
        self._tesseract_config = tesseract_config
        self._quality = quality
        self._postprocess = postprocess
        self._pytesseract: Any | None = None
        self._module_lock = threading.Lock()

    def recognize_blocks(self, image: bytes) -> OcrResult:
        """Recognize receipt text with bounding boxes.

        Args:
            image: Raw receipt image bytes.

        Returns:
            Recognized line blocks with absolute-pixel bounding boxes.

        Raises:
            ImageQualityError: If the configured quality gate rejects the image.
            OcrEngineUnavailableError: If ``pytesseract`` or the Tesseract binary is missing.
        """
        array = decode_image(image)
        if self._quality is not None:
            ensure_quality(array, self._quality)
        height, width = array.shape[:2]
        pytesseract = self._module()
        rgb = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
        try:
            data = pytesseract.image_to_data(
                rgb,
                lang=self._languages,
                config=self._tesseract_config,
                output_type=pytesseract.Output.DICT,
            )
        except pytesseract.TesseractNotFoundError as exc:
            raise OcrEngineUnavailableError("tesseract binary not found") from exc
        except pytesseract.TesseractError as exc:
            raise OcrEngineUnavailableError(f"Tesseract failed: {exc}") from exc
        blocks = _line_blocks(data, self._min_confidence)
        if self._postprocess is not None:
            blocks = postprocess_blocks(blocks, self._postprocess)
        result = OcrResult(blocks=blocks, image_width=width, image_height=height)
        log_detected_text(self.name, result)
        return result

    def recognize(self, image: bytes) -> str:
        """Recognize receipt text as plain text.

        Args:
            image: Raw receipt image bytes.

        Returns:
            Recognized text with one receipt line per text line.
        """
        return blocks_to_text(self.recognize_blocks(image))

    def _module(self) -> Any:
        if self._pytesseract is not None:
            return self._pytesseract
        with self._module_lock:
            if self._pytesseract is None:
                try:
                    import pytesseract
                except ImportError as exc:
                    raise OcrEngineUnavailableError("pytesseract is not installed") from exc
                self._pytesseract = pytesseract
            return self._pytesseract


def _line_blocks(data: dict[str, list[Any]], min_confidence: float) -> list[TextBlock]:
    grouped: dict[tuple[int, int, int], list[tuple[str, float, int, int, int, int]]] = {}
    count = len(data.get("text", []))
    for index in range(count):
        if int(data["level"][index]) != _WORD_LEVEL:
            continue
        text = str(data["text"][index]).strip()
        confidence = _parse_confidence(data["conf"][index])
        if not text or confidence < min_confidence:
            continue
        left = int(data["left"][index])
        top = int(data["top"][index])
        width = int(data["width"][index])
        height = int(data["height"][index])
        key = (
            int(data["block_num"][index]),
            int(data["par_num"][index]),
            int(data["line_num"][index]),
        )
        grouped.setdefault(key, []).append((text, confidence, left, top, width, height))

    blocks: list[TextBlock] = []
    for words in grouped.values():
        left = min(word[2] for word in words)
        top = min(word[3] for word in words)
        right = max(word[2] + word[4] for word in words)
        bottom = max(word[3] + word[5] for word in words)
        mean_confidence = sum(word[1] for word in words) / len(words)
        blocks.append(
            TextBlock(
                text=" ".join(word[0] for word in words),
                bbox=[(left, top), (right, top), (right, bottom), (left, bottom)],
                confidence=min(1.0, mean_confidence),
            )
        )
    return blocks


def _parse_confidence(raw: Any) -> float:
    try:
        return max(0.0, float(raw) / 100.0)
    except (TypeError, ValueError):
        return 0.0
