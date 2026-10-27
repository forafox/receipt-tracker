import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, tzinfo
from decimal import Decimal

from common.contracts import Receipt
from common.ocr import OcrEngine
from extract import parse_ocr_result
from ocr import ImageDecodeError, OcrError
from storage import ReceiptStorage

logger = logging.getLogger(__name__)

HISTORY_SIZE = 5


class ReceiptUnreadable(Exception):
    """Raised when no useful receipt data can be extracted from a photo."""


class RecognitionUnavailable(Exception):
    """Raised when the OCR engine fails for reasons unrelated to the photo."""


class ReceiptAlreadySaved(Exception):
    """Raised when the user has already saved the same receipt."""


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

    def __init__(self, recognizer: OcrEngine, storage: ReceiptStorage, timezone: tzinfo) -> None:
        """Create a pipeline.

        Args:
            recognizer: OCR step that returns text blocks with their positions.
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
            RecognitionUnavailable: If the OCR engine itself failed.
        """
        started = time.perf_counter()
        try:
            blocks = await asyncio.to_thread(self._recognizer.recognize_blocks, photo)
        except ImageDecodeError as error:
            raise ReceiptUnreadable from error
        except OcrError as error:
            logger.exception("OCR engine %s failed", self._recognizer.name)
            raise RecognitionUnavailable from error
        recognized = time.perf_counter()
        receipt = parse_ocr_result(blocks, timezone=self._timezone)
        parsed = time.perf_counter()
        if receipt.datetime is None and receipt.total is None:
            logger.info(
                "Unreadable receipt for user %s: ocr %.2fs, %d blocks",
                user_id,
                recognized - started,
                len(blocks.blocks),
            )
            raise ReceiptUnreadable
        purchased_at = receipt.datetime or received_at
        saved = await asyncio.to_thread(self._storage.save, user_id, receipt, purchased_at)
        finished = time.perf_counter()
        logger.info(
            "Processed receipt for user %s: ocr %.2fs, parse %.2fs, save %.2fs, "
            "items %d, warnings %s, duplicate %s",
            user_id,
            recognized - started,
            parsed - recognized,
            finished - parsed,
            len(receipt.items),
            receipt.warnings,
            not saved,
        )
        if not saved:
            raise ReceiptAlreadySaved
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

    async def undo_last(self, user_id: int) -> Receipt | None:
        """Delete the receipt the user saved most recently.

        Args:
            user_id: Telegram user identifier.

        Returns:
            Deleted receipt, or None if the user has no receipts.
        """
        return await asyncio.to_thread(self._storage.delete_last, user_id)
