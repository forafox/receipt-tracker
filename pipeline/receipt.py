import asyncio
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta, tzinfo
from decimal import Decimal
from functools import partial
from itertools import count

from common.contracts import Receipt, ReceiptWarning
from common.ocr import OcrEngine, OcrResult
from categorize.service import CategoryService
from extract.agent import parse_with_agent
from ocr import ImageDecodeError, ImageTooLargeError, OcrError
from storage import ReceiptStorage

logger = logging.getLogger(__name__)

HISTORY_SIZE = 5


class ReceiptUnreadable(Exception):
    """Raised when no useful receipt data can be extracted from a photo."""


class ReceiptImageTooLarge(Exception):
    """Raised when the photo is too large for text recognition."""


class RecognitionUnavailable(Exception):
    """Raised when the OCR engine fails for reasons unrelated to the photo."""


class ReceiptAlreadySaved(Exception):
    """Raised when the user has already saved the same receipt.

    Attributes:
        pending_id: Identifier of the rejected receipt kept until the user decides on it.
    """

    def __init__(self, pending_id: int) -> None:
        """Create the error.

        Args:
            pending_id: Identifier of the rejected receipt kept until the user decides on it.
        """
        super().__init__(pending_id)
        self.pending_id = pending_id


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
    category_totals: dict[str, Decimal] | None = None


def _score_receipt(receipt: Receipt, block_count: int) -> int:
    warnings = {str(warning) for warning in receipt.warnings}
    score = 0
    if receipt.datetime is not None:
        score += 1000
    if receipt.total is not None:
        score += 1000
    if receipt.store is not None:
        score += 50
    score += min(len(receipt.items), 50) * 5
    item_sum = sum((item.sum for item in receipt.items), Decimal(0))
    if receipt.total is not None and receipt.items:
        if item_sum == receipt.total:
            score += 200
        elif abs(item_sum - receipt.total) <= Decimal("0.01"):
            score += 100
    if ReceiptWarning.ITEMS_TOTAL_MISMATCH.value in warnings:
        score -= 50
    else:
        score += 30
    if ReceiptWarning.ITEMS_INCOMPLETE.value not in warnings:
        score += 20
    if ReceiptWarning.ITEM_AMOUNT_MISMATCH.value in warnings:
        score -= 20
    score += min(block_count, 200) // 20
    return score


class ReceiptPipeline:
    """Runs the receipt processing steps in order.

    A receipt rejected as a duplicate is kept in memory until the user decides whether to
    save it anyway. Only the latest such receipt per user is kept, and it is lost on restart.
    A parseable Hunyuan result is absolute priority. Other recognizers run only when
    Hunyuan fails; without Hunyuan, the highest-scoring parsed OCR result is saved.
    """

    def __init__(
        self,
        recognizer: OcrEngine,
        storage: ReceiptStorage,
        timezone: tzinfo,
        *,
        recognizers: Sequence[OcrEngine] | None = None,
        category_service: CategoryService | None = None,
    ) -> None:
        """Create a pipeline.

        Args:
            recognizer: Primary OCR step, also used for the engine name in logs.
            storage: Storage for parsed receipts.
            timezone: Timezone of the local time printed on receipts and of month boundaries.
            recognizers: Optional OCR candidates. When omitted, ``recognizer`` is the only one.

        Raises:
            ValueError: If an explicit empty recognizer sequence is supplied.
        """
        self._recognizer = recognizer
        self._recognizers = tuple(recognizers) if recognizers is not None else (recognizer,)
        if not self._recognizers:
            msg = "at least one recognizer is required"
            raise ValueError(msg)
        self._storage = storage
        self._category_service = category_service
        self._timezone = timezone
        self._pending_duplicates: dict[int, tuple[int, Receipt, datetime]] = {}
        self._pending_ids = count(1)

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
            ReceiptImageTooLarge: If the photo exceeds the OCR size limits.
            RecognitionUnavailable: If the OCR engine itself failed.
        """
        try:
            blocks, receipt, ocr_seconds, parse_seconds = await self._read_receipt(photo)
        except ImageTooLargeError as error:
            raise ReceiptImageTooLarge from error
        except ImageDecodeError as error:
            raise ReceiptUnreadable from error
        except OcrError as error:
            logger.exception("OCR engine %s failed", self._recognizer.name)
            raise RecognitionUnavailable from error
        if receipt.datetime is None and receipt.total is None:
            logger.info(
                "Unreadable receipt for user %s: ocr %.2fs, %d blocks",
                user_id,
                ocr_seconds,
                len(blocks.blocks),
            )
            raise ReceiptUnreadable
        if self._category_service is not None and receipt.items:
            try:
                classified = self._category_service.classify_receipt(receipt)
                receipt = receipt.model_copy(
                    update={
                        "items": [
                            item.model_copy(
                                update={"categories": [s.category for s in item.categories]}
                            )
                            for item in classified.items
                        ]
                    }
                )
            except Exception:
                logger.exception("M3 category classification failed; saving receipt without categories")
        purchased_at = receipt.datetime or received_at
        save_started = time.perf_counter()
        saved = await asyncio.to_thread(self._storage.save, user_id, receipt, purchased_at)
        save_seconds = time.perf_counter() - save_started
        logger.info(
            "Processed receipt for user %s: ocr %.2fs, parse %.2fs, save %.2fs, "
            "items %d, warnings %s, duplicate %s",
            user_id,
            ocr_seconds,
            parse_seconds,
            save_seconds,
            len(receipt.items),
            receipt.warnings,
            not saved,
        )
        if not saved:
            pending_id = next(self._pending_ids)
            self._pending_duplicates[user_id] = (pending_id, receipt, purchased_at)
            raise ReceiptAlreadySaved(pending_id)
        return receipt

    async def _read_receipt(self, photo: bytes) -> tuple[OcrResult, Receipt, float, float]:
        ocr_seconds = 0.0
        parse_seconds = 0.0
        errors: list[Exception] = []
        candidates: list[tuple[OcrEngine, OcrResult, Receipt]] = []
        for recognizer in self._recognizers:
            ocr_started = time.perf_counter()
            try:
                result = await asyncio.to_thread(recognizer.recognize_blocks, photo)
            except (ImageTooLargeError, ImageDecodeError):
                raise
            except OcrError as error:
                errors.append(error)
                continue
            ocr_seconds += time.perf_counter() - ocr_started

            parse_started = time.perf_counter()
            try:
                receipt = await asyncio.to_thread(
                    parse_with_agent, result, timezone=self._timezone
                )
            except ValueError as error:
                errors.append(error)
                continue
            parse_seconds += time.perf_counter() - parse_started
            candidates.append((recognizer, result, receipt))
            # Hunyuan is the primary recognizer. Once its text can be parsed, keep
            # its result without waiting for or comparing lower-priority OCR engines.
            if recognizer.name == "hunyuan_ocr":
                break

        if not candidates:
            if errors:
                raise errors[-1]
            msg = "no OCR candidate could be parsed"
            raise RecognitionUnavailable(msg)

        hunyuan = next(
            (candidate for candidate in candidates if candidate[0].name == "hunyuan_ocr"),
            None,
        )
        best = hunyuan or max(
            candidates,
            key=lambda candidate: _score_receipt(candidate[2], len(candidate[1].blocks)),
        )
        return best[1], best[2], ocr_seconds, parse_seconds

    async def save_duplicate(self, user_id: int, pending_id: int) -> Receipt | None:
        """Save a receipt rejected as a duplicate, after the user confirmed it.

        Args:
            user_id: Telegram user identifier.
            pending_id: Identifier from the ``ReceiptAlreadySaved`` error.

        Returns:
            Saved receipt, or None if this receipt is no longer pending.
        """
        pending = self._take_pending(user_id, pending_id)
        if pending is None:
            return None
        receipt, purchased_at = pending
        await asyncio.to_thread(
            partial(self._storage.save, allow_duplicate=True), user_id, receipt, purchased_at
        )
        return receipt

    def discard_duplicate(self, user_id: int, pending_id: int) -> None:
        """Forget a receipt rejected as a duplicate.

        Args:
            user_id: Telegram user identifier.
            pending_id: Identifier from the ``ReceiptAlreadySaved`` error.
        """
        self._take_pending(user_id, pending_id)

    def _take_pending(self, user_id: int, pending_id: int) -> tuple[Receipt, datetime] | None:
        # A newer duplicate replaces the older one, so an old question must not touch it.
        pending = self._pending_duplicates.get(user_id)
        if pending is None or pending[0] != pending_id:
            return None
        del self._pending_duplicates[user_id]
        return pending[1], pending[2]

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
        category_totals: dict[str, Decimal] = {}
        for receipt in receipts:
            for item in receipt.items:
                if not item.categories:
                    continue
                share = item.sum / len(item.categories)
                for category in item.categories:
                    category_totals[category] = category_totals.get(category, Decimal(0)) + share
        return MonthlySummary(
            month=start.date(),
            receipt_count=len(receipts),
            total=total,
            category_totals=category_totals or None,
        )

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
