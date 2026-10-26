from __future__ import annotations

import threading

from common.ocr import OcrEngine, OcrResult
from ocr.core.config import OcrSettings
from ocr.engines.factory import build_ocr_engine


class OcrService:
    """Central OCR runtime selected by configuration.

    The service reads engine settings from :class:`~ocr.core.config.OcrSettings`
    (environment variables with the ``OCR_`` prefix by default) and creates the engine
    lazily on first use, so constructing the service does not import heavy OCR
    dependencies or download model weights. Lazy creation is serialized, so concurrent
    first calls share a single engine instance.

    Attributes:
        settings: Effective OCR settings used to build the engine.
        name: Selected engine name.
    """

    def __init__(
        self, settings: OcrSettings | None = None, *, engine: OcrEngine | None = None
    ) -> None:
        """Create an OCR service.

        Args:
            settings: OCR settings; defaults to values loaded from the environment.
            engine: Pre-built engine, mainly for dependency injection in tests.
        """
        self._settings = settings if settings is not None else OcrSettings()
        self._engine = engine
        self._engine_lock = threading.Lock()
        self.name = engine.name if engine is not None else self._settings.engine

    @property
    def settings(self) -> OcrSettings:
        """Return the effective OCR settings.

        Returns:
            Settings used to build the engine.
        """
        return self._settings

    def process_receipt(self, image: bytes) -> OcrResult:
        """Recognize a receipt image with bounding boxes.

        Args:
            image: Raw receipt image bytes.

        Returns:
            Recognized text blocks with absolute-pixel bounding boxes.
        """
        return self.recognize_blocks(image)

    def recognize_blocks(self, image: bytes) -> OcrResult:
        """Recognize receipt text with bounding boxes.

        Args:
            image: Raw receipt image bytes.

        Returns:
            Recognized text blocks with absolute-pixel bounding boxes.
        """
        return self._resolved_engine().recognize_blocks(image)

    def recognize(self, image: bytes) -> str:
        """Recognize receipt text as plain text.

        Args:
            image: Raw receipt image bytes.

        Returns:
            Recognized text with one receipt line per text line.
        """
        return self._resolved_engine().recognize(image)

    def _resolved_engine(self) -> OcrEngine:
        if self._engine is not None:
            return self._engine
        with self._engine_lock:
            if self._engine is None:
                self._engine = build_ocr_engine(self._settings)
            return self._engine


def build_default_ocr_service() -> OcrService:
    """Build the default OCR service from environment configuration.

    Returns:
        Service using default or environment-provided ``OcrSettings``.
    """
    return OcrService(OcrSettings())
