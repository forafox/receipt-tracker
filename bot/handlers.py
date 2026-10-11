import logging
from io import BytesIO

from aiogram import Bot, F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import Message

from bot.service import ReceiptService
from bot.texts import Texts, format_receipt
from pipeline import ReceiptUnreadable

logger = logging.getLogger(__name__)

router = Router()


@router.message(CommandStart())
async def handle_start(message: Message, texts: Texts) -> None:
    """Send the localized greeting.

    Args:
        message: Incoming Telegram start-command message.
        texts: Localized response texts.
    """
    await message.answer(texts.start)


@router.message(Command("help"))
async def handle_help(message: Message, texts: Texts) -> None:
    """Send localized usage guidance.

    Args:
        message: Incoming Telegram help-command message.
        texts: Localized response texts.
    """
    await message.answer(texts.help)


@router.message(Command("month"))
async def handle_month(message: Message, texts: Texts) -> None:
    """Send the current user's monthly expense report.

    Args:
        message: Incoming Telegram month-command message.
        texts: Localized response texts.
    """
    await message.answer(texts.month_unavailable)


@router.message(F.photo)
async def handle_photo(message: Message, bot: Bot, service: ReceiptService, texts: Texts) -> None:
    """Download the largest receipt photo and reply with the parsed receipt.

    Args:
        message: Incoming Telegram message containing receipt photos.
        bot: Telegram bot used to download the selected photo.
        service: Receipt service that processes the downloaded image.
        texts: Localized response texts.
    """
    if message.from_user is None or not message.photo:
        return
    buffer = BytesIO()
    await bot.download(message.photo[-1], destination=buffer)
    try:
        receipt = await service.process_photo(message.from_user.id, buffer.getvalue())
        reply = format_receipt(receipt, texts)
    except ReceiptUnreadable:
        reply = texts.unreadable
    except Exception:
        logger.exception("Failed to process receipt photo")
        reply = texts.error
    await message.answer(reply)


@router.message()
async def handle_unknown(message: Message, texts: Texts) -> None:
    """Reply to an unsupported Telegram message.

    Args:
        message: Incoming unsupported Telegram message.
        texts: Localized response texts.
    """
    await message.answer(texts.unknown)
