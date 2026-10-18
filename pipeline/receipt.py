import asyncio
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta, tzinfo
from decimal import Decimal
from typing import Protocol

from common.contracts import Receipt
from extract import parse_receipt
from ocr import ImageDecodeError
from storage import ReceiptStorage

logger = logging.getLogger(__name__)

HISTORY_SIZE = 5


class ReceiptUnreadable(Exception):
    """Raised when no useful receipt data can be extracted from a photo."""


class ReceiptAlreadySaved(Exception):
    """Raised when the user has already saved the same receipt."""


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


@dataclass(frozen=True)
class MonthlySummary:
    """Spending of one user in one calendar month.

    Attributes:
        month: First day of the month.
        receipt_count: Number of saved receipts.
        total: Sum of receipt totals; receipts without a total are not counted.
    """

    month: date
    receipt_count: int
    total: Decimal


class ReceiptPipeline:
    """Runs the receipt processing steps in order."""

    def __init__(
        self, recognizer: TextRecognizer, storage: ReceiptStorage, timezone: tzinfo
    ) -> None:
        """Create a pipeline.

        Args:
            recognizer: OCR step.
            storage: Storage for parsed receipts.
            timezone: Timezone of the local time printed on receipts and of month boundaries.
        """
        self._recognizer = recognizer
        self._storage = storage
        self._timezone = timezone

    async def process_photo(self, user_id: int, photo: bytes, received_at: datetime) -> Receipt:
        """Recognize, parse and save a receipt photo.

        Args:
            user_id: Telegram user identifier.
            photo: Raw receipt image bytes.
            received_at: Time the photo was sent; used when the receipt has no date.

        Returns:
            Parsed receipt.

        Raises:
            ReceiptUnreadable: If the image cannot be decoded or neither the date nor the total
                was found.
            ReceiptAlreadySaved: If the user has already saved the same receipt.
        """
        try:
            text = await asyncio.to_thread(self._recognizer.recognize, photo)
        except ImageDecodeError as error:
            raise ReceiptUnreadable from error
        receipt = parse_receipt(text, timezone=self._timezone)
        if receipt.datetime is None and receipt.total is None:
            raise ReceiptUnreadable
        purchased_at = receipt.datetime or received_at
        saved = await asyncio.to_thread(self._storage.save, user_id, receipt, purchased_at)
        if not saved:
            raise ReceiptAlreadySaved
        logger.info("Saved receipt for user %s, warnings: %s", user_id, receipt.warnings)
        return receipt

    async def monthly_summary(self, user_id: int, moment: datetime) -> MonthlySummary:
        """Sum a user's spending for the month containing a moment.

        Args:
            user_id: Telegram user identifier.
            moment: Timezone-aware moment inside the month.

        Returns:
            Spending summary for the month.
        """
        start = moment.astimezone(self._timezone).replace(
            day=1, hour=0, minute=0, second=0, microsecond=0
        )
        end = (start + timedelta(days=32)).replace(day=1)
        receipts = await asyncio.to_thread(self._storage.list_between, user_id, start, end)
        total = sum(
            (receipt.total for receipt in receipts if receipt.total is not None), Decimal(0)
        )
        return MonthlySummary(month=start.date(), receipt_count=len(receipts), total=total)

    async def recent_receipts(self, user_id: int) -> list[Receipt]:
        """List a user's most recent receipts.

        Args:
            user_id: Telegram user identifier.

        Returns:
            Up to ``HISTORY_SIZE`` receipts ordered from the newest to the oldest.
        """
        return await asyncio.to_thread(self._storage.list_recent, user_id, HISTORY_SIZE)
