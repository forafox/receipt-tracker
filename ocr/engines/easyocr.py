from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from common.ocr import OcrResult, TextBlock
from ocr.core.errors import OcrEngineUnavailableError
from ocr.core.image import decode_image
from ocr.core.ordering import blocks_to_text


class EasyOcrEngine:
    """OCR engine backed by EasyOCR.

    The underlying ``easyocr.Reader`` is created lazily on the first recognition call, so
    importing and constructing the engine never downloads model weights.

    Attributes:
        name: Engine identifier used by configuration and factories.
    """

    name = "easyocr"

    def __init__(
        self,
        *,
        languages: Sequence[str] | None = None,
        gpu: bool = False,
        min_confidence: float = 0.0,
    ) -> None:
        """Create an EasyOCR engine.

        Args:
            languages: EasyOCR language codes; defaults to Russian and English.
            gpu: Whether EasyOCR should use the GPU.
            min_confidence: Minimum block confidence in the ``0..1`` range to keep a block.
        """
        self._languages = list(languages) if languages is not None else ["ru", "en"]
        self._gpu = gpu
        self._min_confidence = min_confidence
        self._reader: Any | None = None

    def recognize_blocks(self, image: bytes) -> OcrResult:
        """Recognize receipt text with bounding boxes.

        Args:
            image: Raw receipt image bytes.

        Returns:
            Recognized blocks with absolute-pixel bounding boxes.

        Raises:
            OcrEngineUnavailableError: If ``easyocr`` is not installed.
        """
        reader = self._reader_instance()
        array = decode_image(image)
        height, width = array.shape[:2]

        blocks: list[TextBlock] = []
        for quad, text, confidence in reader.readtext(array):
            stripped = str(text).strip()
            score = min(1.0, max(0.0, float(confidence)))
            if not stripped or score < self._min_confidence:
                continue
            blocks.append(
                TextBlock(
                    text=stripped,
                    bbox=[(float(point[0]), float(point[1])) for point in quad],
                    confidence=score,
                )
            )
        return OcrResult(blocks=blocks, image_width=width, image_height=height)

    def recognize(self, image: bytes) -> str:
        """Recognize receipt text as plain text.

        Args:
            image: Raw receipt image bytes.

        Returns:
            Recognized text with one receipt line per text line.
        """
        return blocks_to_text(self.recognize_blocks(image))

    def _reader_instance(self) -> Any:
        if self._reader is None:
            try:
                import easyocr
            except ImportError as exc:
                raise OcrEngineUnavailableError("easyocr is not installed") from exc
            self._reader = easyocr.Reader(self._languages, gpu=self._gpu)
        return self._reader
