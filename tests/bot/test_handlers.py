from datetime import date, datetime
from decimal import Decimal
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock
from zoneinfo import ZoneInfo

from bot.handlers import handle_image_document, handle_month, handle_photo, handle_start
from bot.texts import EN, RU
from common.contracts import Receipt
from pipeline import MonthlySummary, ReceiptUnreadable

SENT_AT = datetime(2026, 10, 11, 9, 0, tzinfo=ZoneInfo("UTC"))
RECEIPT = Receipt(
    store="ПЯТЁРОЧКА",
    datetime=datetime(2026, 10, 5, 18, 20, tzinfo=ZoneInfo("Europe/Moscow")),
    total=Decimal("250.00"),
)


class FakeService:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.received: tuple[int, bytes, datetime] | None = None
        self.received_month: tuple[int, datetime] | None = None

    async def process_photo(self, user_id: int, photo: bytes, received_at: datetime) -> Receipt:
        if self.error is not None:
            raise self.error
        self.received = (user_id, photo, received_at)
        return RECEIPT

    async def monthly_summary(self, user_id: int, moment: datetime) -> MonthlySummary:
        self.received_month = (user_id, moment)
        return MonthlySummary(month=date(2026, 10, 1), receipt_count=2, total=Decimal("300"))


def make_message() -> MagicMock:
    message = MagicMock()
    message.from_user.id = 42
    message.date = SENT_AT
    message.photo = [MagicMock(), MagicMock()]
    message.answer = AsyncMock()
    return message


def make_bot(content: bytes) -> MagicMock:
    async def download(file: object, destination: BytesIO) -> None:
        destination.write(content)

    bot = MagicMock()
    bot.download = AsyncMock(side_effect=download)
    bot.send_chat_action = AsyncMock()
    return bot


async def test_start_sends_greeting() -> None:
    message = make_message()

    await handle_start(message, RU)

    message.answer.assert_awaited_once_with(RU.start)


async def test_month_sends_summary_for_message_time() -> None:
    message = make_message()
    service = FakeService()

    await handle_month(message, service, RU)

    assert service.received_month == (42, SENT_AT)
    message.answer.assert_awaited_once_with("Расходы за 10.2026: 300.00\nЧеков: 2")


async def test_photo_passes_largest_size_to_service() -> None:
    message = make_message()
    bot = make_bot(b"image")
    service = FakeService()

    await handle_photo(message, bot, service, EN)

    assert bot.download.await_args.args[0] is message.photo[-1]
    assert service.received == (42, b"image", SENT_AT)
    message.answer.assert_awaited_once_with(
        "Store: ПЯТЁРОЧКА\nDate: 05.10.2026 18:20\nTotal: 250.00"
    )


async def test_photo_reports_error_when_service_fails() -> None:
    message = make_message()

    await handle_photo(message, make_bot(b"image"), FakeService(RuntimeError("boom")), EN)

    message.answer.assert_awaited_once_with(EN.error)


async def test_photo_reports_unreadable_receipt() -> None:
    message = make_message()

    await handle_photo(message, make_bot(b"image"), FakeService(ReceiptUnreadable()), RU)

    message.answer.assert_awaited_once_with(RU.unreadable)


async def test_image_document_is_processed_like_photo() -> None:
    message = make_message()
    message.document = MagicMock()
    bot = make_bot(b"file")
    service = FakeService()

    await handle_image_document(message, bot, service, EN)

    assert bot.download.await_args.args[0] is message.document
    assert service.received == (42, b"file", SENT_AT)
    message.answer.assert_awaited_once()
