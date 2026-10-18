from datetime import UTC, date, datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from common.contracts import Receipt
from ocr import ImageDecodeError, OcrEngineUnavailableError
from pipeline import MonthlySummary, ReceiptPipeline, ReceiptUnreadable

MOSCOW_TIMEZONE = ZoneInfo("Europe/Moscow")
RECEIVED_AT = datetime(2026, 10, 11, 12, 0, tzinfo=UTC)


class FakeRecognizer:
    def __init__(self, text: str) -> None:
        self.text = text
        self.received: bytes | None = None

    def recognize(self, image: bytes) -> str:
        self.received = image
        return self.text


class FakeStorage:
    def __init__(self) -> None:
        self.saved: list[tuple[int, Receipt, datetime]] = []

    def save(self, user_id: int, receipt: Receipt, purchased_at: datetime) -> None:
        self.saved.append((user_id, receipt, purchased_at))

    def list_between(self, user_id: int, start: datetime, end: datetime) -> list[Receipt]:
        return [
            receipt
            for owner, receipt, purchased_at in self.saved
            if owner == user_id and start <= purchased_at < end
        ]


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
    def __init__(self, error: Exception) -> None:
        self.error = error

    def recognize(self, image: bytes) -> str:
        raise self.error


async def test_undecodable_image_is_unreadable() -> None:
    storage = FakeStorage()
    pipeline = ReceiptPipeline(FailingRecognizer(ImageDecodeError()), storage, MOSCOW_TIMEZONE)

    with pytest.raises(ReceiptUnreadable):
        await pipeline.process_photo(1, b"not an image", RECEIVED_AT)

    assert storage.saved == []


async def test_unavailable_engine_is_not_reported_as_unreadable() -> None:
    error = OcrEngineUnavailableError("tesseract binary not found")
    pipeline = ReceiptPipeline(FailingRecognizer(error), FakeStorage(), MOSCOW_TIMEZONE)

    with pytest.raises(OcrEngineUnavailableError):
        await pipeline.process_photo(1, b"image", RECEIVED_AT)


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
