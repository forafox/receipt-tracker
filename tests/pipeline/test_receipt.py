from datetime import datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from pipeline import NullRecognizer, ReceiptPipeline, ReceiptUnreadable

MOSCOW_TIMEZONE = ZoneInfo("Europe/Moscow")


class FakeRecognizer:
    def __init__(self, text: str) -> None:
        self.text = text
        self.received: bytes | None = None

    def recognize(self, image: bytes) -> str:
        self.received = image
        return self.text


async def test_pipeline_parses_recognized_text() -> None:
    recognizer = FakeRecognizer("ПЯТЁРОЧКА\n05.10.2026 18:20\nИТОГО 250,00")

    receipt = await ReceiptPipeline(recognizer, MOSCOW_TIMEZONE).process_photo(1, b"image")

    assert recognizer.received == b"image"
    assert receipt.store == "ПЯТЁРОЧКА"
    assert receipt.datetime == datetime(2026, 10, 5, 18, 20, tzinfo=MOSCOW_TIMEZONE)
    assert receipt.total == Decimal("250.00")


async def test_pipeline_keeps_partial_receipt() -> None:
    recognizer = FakeRecognizer("ПЯТЁРОЧКА\nИТОГО 250,00")

    receipt = await ReceiptPipeline(recognizer, MOSCOW_TIMEZONE).process_photo(1, b"image")

    assert receipt.total == Decimal("250.00")
    assert receipt.warnings == ["datetime_not_found"]


@pytest.mark.parametrize("text", ["", "ПЯТЁРОЧКА\nСПАСИБО ЗА ПОКУПКУ"])
async def test_pipeline_rejects_receipt_without_date_and_total(text: str) -> None:
    pipeline = ReceiptPipeline(FakeRecognizer(text), MOSCOW_TIMEZONE)

    with pytest.raises(ReceiptUnreadable):
        await pipeline.process_photo(1, b"image")


async def test_null_recognizer_makes_every_photo_unreadable() -> None:
    pipeline = ReceiptPipeline(NullRecognizer(), MOSCOW_TIMEZONE)

    with pytest.raises(ReceiptUnreadable):
        await pipeline.process_photo(1, b"image")
