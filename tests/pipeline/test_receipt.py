import logging
import re
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from common.contracts import Item, Receipt
from common.ocr import OcrResult, TextBlock
from ocr import ImageDecodeError, OcrEngineUnavailableError
from pipeline import (
    MonthlySummary,
    ReceiptAlreadySaved,
    ReceiptPipeline,
    ReceiptUnreadable,
    RecognitionUnavailable,
)
from storage import SqliteReceiptStorage

MOSCOW_TIMEZONE = ZoneInfo("Europe/Moscow")
RECEIVED_AT = datetime(2026, 10, 11, 12, 0, tzinfo=UTC)


CHARACTER_WIDTH = 20
LINE_HEIGHT = 40


def blocks_from_text(text: str) -> OcrResult:
    """Lay out text as OCR blocks: one row per line, columns split by two or more spaces."""
    blocks = []
    for row, line in enumerate(text.splitlines()):
        left = 0
        top = row * LINE_HEIGHT
        for part in re.split(r"(\s{2,})", line):
            right = left + CHARACTER_WIDTH * len(part)
            if part.strip():
                bbox = [(left, top), (right, top), (right, top + 30), (left, top + 30)]
                blocks.append(TextBlock(text=part, bbox=bbox))
            left = right
    return OcrResult(blocks=blocks)


class FakeRecognizer:
    name = "fake"

    def __init__(self, text: str) -> None:
        self.text = text
        self.received: bytes | None = None

    def recognize_blocks(self, image: bytes) -> OcrResult:
        self.received = image
        return blocks_from_text(self.text)

    def recognize(self, image: bytes) -> str:
        return self.text


class FakeStorage:
    def __init__(self) -> None:
        self.saved: list[tuple[int, Receipt, datetime]] = []

    def save(
        self,
        user_id: int,
        receipt: Receipt,
        purchased_at: datetime,
        *,
        allow_duplicate: bool = False,
    ) -> bool:
        is_duplicate = any(owner == user_id and saved == receipt for owner, saved, _ in self.saved)
        if is_duplicate and not allow_duplicate:
            return False
        self.saved.append((user_id, receipt, purchased_at))
        return True

    def list_between(self, user_id: int, start: datetime, end: datetime) -> list[Receipt]:
        return [
            receipt
            for owner, receipt, purchased_at in self.saved
            if owner == user_id and start <= purchased_at < end
        ]

    def delete_last(self, user_id: int) -> Receipt | None:
        for index in range(len(self.saved) - 1, -1, -1):
            if self.saved[index][0] == user_id:
                return self.saved.pop(index)[1]
        return None

    def list_recent(self, user_id: int, limit: int) -> list[Receipt]:
        receipts = [receipt for owner, receipt, _ in self.saved if owner == user_id]
        return receipts[::-1][:limit]


def make_pipeline(text: str, storage: FakeStorage | None = None) -> ReceiptPipeline:
    return ReceiptPipeline(FakeRecognizer(text), storage or FakeStorage(), MOSCOW_TIMEZONE)


async def test_pipeline_parses_and_saves_receipt() -> None:
    recognizer = FakeRecognizer("ПЯТЁРОЧКА\n05.10.2026 18:20\nИТОГО 250,00")
    storage = FakeStorage()
    pipeline = ReceiptPipeline(recognizer, storage, MOSCOW_TIMEZONE)

    receipt = await pipeline.process_photo(1, b"image", RECEIVED_AT)

    purchased_at = datetime(2026, 10, 5, 18, 20, tzinfo=MOSCOW_TIMEZONE)
    assert recognizer.received == b"image"
    assert receipt.store == "ПЯТЁРОЧКА"
    assert receipt.datetime == purchased_at
    assert receipt.total == Decimal("250.00")
    assert storage.saved == [(1, receipt, purchased_at)]


async def test_pipeline_extracts_items_from_layout() -> None:
    text = (
        "ПЯТЁРОЧКА\n"
        "05.10.2026 18:20\n"
        "МОЛОКО 3.2%      89.90\n"
        "ХЛЕБ             45.00\n"
        "ИТОГО           134.90"
    )

    receipt = await make_pipeline(text).process_photo(1, b"image", RECEIVED_AT)

    assert receipt.items == [
        Item(name="МОЛОКО 3.2%", sum=Decimal("89.90")),
        Item(name="ХЛЕБ", sum=Decimal("45.00")),
    ]
    assert receipt.warnings == []


async def test_receipt_without_date_is_saved_at_receive_time() -> None:
    storage = FakeStorage()

    receipt = await make_pipeline("ПЯТЁРОЧКА\nИТОГО 250,00", storage).process_photo(
        1, b"image", RECEIVED_AT
    )

    assert receipt.warnings == ["datetime_not_found"]
    assert storage.saved == [(1, receipt, RECEIVED_AT)]


@pytest.mark.parametrize("text", ["", "ПЯТЁРОЧКА\nСПАСИБО ЗА ПОКУПКУ"])
async def test_unreadable_receipt_is_rejected_and_not_saved(text: str) -> None:
    storage = FakeStorage()

    with pytest.raises(ReceiptUnreadable):
        await make_pipeline(text, storage).process_photo(1, b"image", RECEIVED_AT)

    assert storage.saved == []


class FailingRecognizer:
    name = "failing"

    def __init__(self, error: Exception) -> None:
        self.error = error

    def recognize_blocks(self, image: bytes) -> OcrResult:
        raise self.error

    def recognize(self, image: bytes) -> str:
        raise self.error


async def test_undecodable_image_is_unreadable() -> None:
    storage = FakeStorage()
    pipeline = ReceiptPipeline(FailingRecognizer(ImageDecodeError()), storage, MOSCOW_TIMEZONE)

    with pytest.raises(ReceiptUnreadable):
        await pipeline.process_photo(1, b"not an image", RECEIVED_AT)

    assert storage.saved == []


async def test_engine_failure_is_reported_as_temporary() -> None:
    error = OcrEngineUnavailableError("tesseract binary not found")
    storage = FakeStorage()
    pipeline = ReceiptPipeline(FailingRecognizer(error), storage, MOSCOW_TIMEZONE)

    with pytest.raises(RecognitionUnavailable) as raised:
        await pipeline.process_photo(1, b"image", RECEIVED_AT)

    assert raised.value.__cause__ is error
    assert storage.saved == []


async def test_monthly_summary_uses_local_month_boundaries() -> None:
    storage = FakeStorage()
    storage.save(
        1, Receipt(total=Decimal("100.00")), datetime(2026, 10, 1, 0, 30, tzinfo=MOSCOW_TIMEZONE)
    )
    storage.save(
        1, Receipt(total=Decimal("50.50")), datetime(2026, 10, 31, 23, 0, tzinfo=MOSCOW_TIMEZONE)
    )
    storage.save(1, Receipt(store="без суммы"), datetime(2026, 10, 15, tzinfo=MOSCOW_TIMEZONE))
    storage.save(
        1, Receipt(total=Decimal("7.00")), datetime(2026, 9, 30, 23, 0, tzinfo=MOSCOW_TIMEZONE)
    )
    storage.save(
        1, Receipt(total=Decimal("7.00")), datetime(2026, 11, 1, 0, 0, tzinfo=MOSCOW_TIMEZONE)
    )
    storage.save(2, Receipt(total=Decimal("7.00")), datetime(2026, 10, 15, tzinfo=MOSCOW_TIMEZONE))

    summary = await make_pipeline("", storage).monthly_summary(1, RECEIVED_AT)

    assert summary == MonthlySummary(
        month=date(2026, 10, 1), receipt_count=3, total=Decimal("150.50")
    )


async def test_monthly_summary_converts_moment_to_local_time() -> None:
    late_september_utc = datetime(2026, 9, 30, 22, 30, tzinfo=UTC)

    summary = await make_pipeline("").monthly_summary(1, late_september_utc)

    assert summary == MonthlySummary(month=date(2026, 10, 1), receipt_count=0, total=Decimal(0))


async def test_same_receipt_is_rejected_on_second_upload() -> None:
    storage = FakeStorage()
    pipeline = make_pipeline("ПЯТЁРОЧКА\n05.10.2026 18:20\nИТОГО 250,00", storage)
    await pipeline.process_photo(1, b"image", RECEIVED_AT)

    with pytest.raises(ReceiptAlreadySaved):
        await pipeline.process_photo(1, b"image", RECEIVED_AT)

    assert len(storage.saved) == 1


async def test_recent_receipts_are_limited_to_history_size() -> None:
    storage = FakeStorage()
    for number in range(7):
        storage.save(1, Receipt(store=f"store {number}"), RECEIVED_AT)

    receipts = await make_pipeline("", storage).recent_receipts(1)

    assert [receipt.store for receipt in receipts] == [f"store {n}" for n in range(6, 1, -1)]


async def test_undo_deletes_last_receipt_of_user() -> None:
    storage = FakeStorage()
    storage.save(1, Receipt(store="first"), RECEIVED_AT)
    storage.save(1, Receipt(store="second"), RECEIVED_AT)
    storage.save(2, Receipt(store="other user"), RECEIVED_AT)

    deleted = await make_pipeline("", storage).undo_last(1)

    assert deleted == Receipt(store="second")
    assert [receipt.store for _, receipt, _ in storage.saved] == ["first", "other user"]


@pytest.mark.parametrize(
    ("timezone", "inside", "outside"),
    [
        # Year rollover: the last minute of December belongs to December, midnight to January.
        (
            "Europe/Moscow",
            datetime(2026, 12, 31, 23, 59),
            datetime(2027, 1, 1, 0, 0),
        ),
        # Clocks go back on the last Sunday of October, so October and November have
        # different UTC offsets in Berlin.
        (
            "Europe/Berlin",
            datetime(2026, 10, 31, 23, 59),
            datetime(2026, 11, 1, 0, 0),
        ),
        # Ahead of UTC by ten hours: local midnight of the 1st is still the previous day in UTC.
        (
            "Asia/Vladivostok",
            datetime(2026, 11, 1, 0, 0),
            datetime(2026, 10, 31, 23, 59),
        ),
    ],
)
async def test_monthly_summary_boundaries_with_sqlite(
    tmp_path: Path, timezone: str, inside: datetime, outside: datetime
) -> None:
    zone = ZoneInfo(timezone)
    inside, outside = inside.replace(tzinfo=zone), outside.replace(tzinfo=zone)
    storage = SqliteReceiptStorage(tmp_path / "receipts.sqlite")
    storage.save(1, Receipt(store="inside", total=Decimal("10.00")), inside)
    storage.save(1, Receipt(store="outside", total=Decimal("99.00")), outside)
    pipeline = ReceiptPipeline(FakeRecognizer(""), storage, zone)

    summary = await pipeline.monthly_summary(1, inside.astimezone(UTC))

    assert summary == MonthlySummary(
        month=inside.date().replace(day=1), receipt_count=1, total=Decimal("10.00")
    )


async def test_receipt_without_date_sent_after_local_midnight_counts_in_new_month() -> None:
    storage = FakeStorage()
    pipeline = make_pipeline("ПЯТЁРОЧКА\nИТОГО 250,00", storage)
    sent_at = datetime(2026, 10, 31, 21, 30, tzinfo=UTC)  # 1 November 00:30 in Moscow

    await pipeline.process_photo(1, b"image", sent_at)
    november = await pipeline.monthly_summary(1, sent_at)
    october = await pipeline.monthly_summary(1, datetime(2026, 10, 31, 20, 59, tzinfo=UTC))

    assert november.month == date(2026, 11, 1)
    assert november.receipt_count == 1
    assert october.receipt_count == 0


async def test_processing_time_of_each_step_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="pipeline.receipt")

    await make_pipeline("ПЯТЁРОЧКА\n05.10.2026 18:20\nИТОГО 250,00").process_photo(
        7, b"image", RECEIVED_AT
    )

    [record] = caplog.records
    assert record.getMessage().startswith("Processed receipt for user 7: ocr ")
    assert ", parse " in record.getMessage()
    assert ", save " in record.getMessage()


DUPLICATE_TEXT = "ПЯТЁРОЧКА\n05.10.2026 18:20\nИТОГО 250,00"


async def test_confirmed_duplicate_is_saved_again() -> None:
    storage = FakeStorage()
    pipeline = make_pipeline(DUPLICATE_TEXT, storage)
    receipt = await pipeline.process_photo(1, b"image", RECEIVED_AT)
    with pytest.raises(ReceiptAlreadySaved):
        await pipeline.process_photo(1, b"image", RECEIVED_AT)

    saved = await pipeline.save_duplicate(1)

    assert saved == receipt
    assert len(storage.saved) == 2
    assert await pipeline.save_duplicate(1) is None


async def test_discarded_duplicate_cannot_be_saved() -> None:
    storage = FakeStorage()
    pipeline = make_pipeline(DUPLICATE_TEXT, storage)
    await pipeline.process_photo(1, b"image", RECEIVED_AT)
    with pytest.raises(ReceiptAlreadySaved):
        await pipeline.process_photo(1, b"image", RECEIVED_AT)

    pipeline.discard_duplicate(1)

    assert await pipeline.save_duplicate(1) is None
    assert len(storage.saved) == 1


async def test_pending_duplicate_belongs_to_its_user() -> None:
    storage = FakeStorage()
    pipeline = make_pipeline(DUPLICATE_TEXT, storage)
    await pipeline.process_photo(1, b"image", RECEIVED_AT)
    with pytest.raises(ReceiptAlreadySaved):
        await pipeline.process_photo(1, b"image", RECEIVED_AT)

    assert await pipeline.save_duplicate(2) is None
    assert await pipeline.save_duplicate(1) is not None
