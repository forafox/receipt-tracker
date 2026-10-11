import asyncio
import logging
from datetime import tzinfo
from typing import Protocol

from common.contracts import Receipt
from extract import parse_receipt

logger = logging.getLogger(__name__)


class ReceiptUnreadable(Exception):
    """Raised when no useful receipt data can be extracted from a photo."""


class TextRecognizer(Protocol):
    """OCR step that turns a receipt image into plain text."""

    def recognize(self, image: bytes) -> str:
        """Recognize text on a receipt image.

        Args:
            image: Raw receipt image bytes.

        Returns:
            Recognized text with one receipt line per text line.
        """
        ...


class NullRecognizer:
    """Recognizer used until an OCR engine is connected."""

    def recognize(self, image: bytes) -> str:
        """Recognize nothing.

        Args:
            image: Raw receipt image bytes.

        Returns:
            Empty text.
        """
        return ""


class ReceiptPipeline:
    """Runs the receipt processing steps in order."""

    def __init__(self, recognizer: TextRecognizer, timezone: tzinfo) -> None:
        """Create a pipeline.

        Args:
            recognizer: OCR step.
            timezone: Timezone of the local time printed on receipts.
        """
        self._recognizer = recognizer
        self._timezone = timezone

    async def process_photo(self, user_id: int, photo: bytes) -> Receipt:
        """Recognize and parse a receipt photo.

        Args:
            user_id: Telegram user identifier.
            photo: Raw receipt image bytes.

        Returns:
            Parsed receipt.

        Raises:
            ReceiptUnreadable: If neither the date nor the total was found.
        """
        text = await asyncio.to_thread(self._recognizer.recognize, photo)
        receipt = parse_receipt(text, timezone=self._timezone)
        if receipt.datetime is None and receipt.total is None:
            raise ReceiptUnreadable
        logger.info("Parsed receipt for user %s, warnings: %s", user_id, receipt.warnings)
        return receipt
