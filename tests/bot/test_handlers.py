from io import BytesIO
from unittest.mock import AsyncMock, MagicMock

from bot.handlers import (
    ERROR_TEXT,
    START_TEXT,
    handle_month,
    handle_photo,
    handle_start,
)


class FakeService:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.received: tuple[int, bytes] | None = None

    async def process_photo(self, user_id: int, photo: bytes) -> str:
        if self.error is not None:
            raise self.error
        self.received = (user_id, photo)
        return "receipt saved"

    async def monthly_report(self, user_id: int) -> str:
        return f"report for {user_id}"


def make_message() -> MagicMock:
    message = MagicMock()
    message.from_user.id = 42
    message.photo = [MagicMock(), MagicMock()]
    message.answer = AsyncMock()
    return message


def make_bot(content: bytes) -> MagicMock:
    async def download(file: object, destination: BytesIO) -> None:
        destination.write(content)

    bot = MagicMock()
    bot.download = AsyncMock(side_effect=download)
    return bot


async def test_start_sends_greeting() -> None:
    message = make_message()

    await handle_start(message)

    message.answer.assert_awaited_once_with(START_TEXT)


async def test_month_returns_report_for_user() -> None:
    message = make_message()

    await handle_month(message, FakeService())

    message.answer.assert_awaited_once_with("report for 42")


async def test_photo_passes_largest_size_to_service() -> None:
    message = make_message()
    bot = make_bot(b"image")
    service = FakeService()

    await handle_photo(message, bot, service)

    assert bot.download.await_args.args[0] is message.photo[-1]
    assert service.received == (42, b"image")
    message.answer.assert_awaited_once_with("receipt saved")


async def test_photo_reports_error_when_service_fails() -> None:
    message = make_message()

    await handle_photo(message, make_bot(b"image"), FakeService(RuntimeError("boom")))

    message.answer.assert_awaited_once_with(ERROR_TEXT)
