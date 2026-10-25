import logging
from io import BytesIO

from aiogram import Bot, F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import Downloadable, Message
from aiogram.utils.chat_action import ChatActionSender

from bot.service import ReceiptService
from bot.texts import Texts, format_history, format_monthly_summary, format_receipt
from pipeline import ReceiptAlreadySaved, ReceiptUnreadable, RecognitionUnavailable

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
async def handle_month(message: Message, service: ReceiptService, texts: Texts) -> None:
    """Send the current user's spending for the current month.

    Args:
        message: Incoming Telegram month-command message.
        service: Receipt service that sums monthly spending.
        texts: Localized response texts.
    """
    if message.from_user is None:
        return
    summary = await service.monthly_summary(message.from_user.id, message.date)
    await message.answer(format_monthly_summary(summary, texts))


@router.message(Command("history"))
async def handle_history(message: Message, service: ReceiptService, texts: Texts) -> None:
    """Send the current user's most recent receipts.

    Args:
        message: Incoming Telegram history-command message.
        service: Receipt service that lists saved receipts.
        texts: Localized response texts.
    """
    if message.from_user is None:
        return
    receipts = await service.recent_receipts(message.from_user.id)
    await message.answer(format_history(receipts, texts))


@router.message(Command("undo"))
async def handle_undo(message: Message, service: ReceiptService, texts: Texts) -> None:
    """Delete the current user's most recently saved receipt.

    Args:
        message: Incoming Telegram undo-command message.
        service: Receipt service that deletes saved receipts.
        texts: Localized response texts.
    """
    if message.from_user is None:
        return
    receipt = await service.undo_last(message.from_user.id)
    if receipt is None:
        await message.answer(texts.undo_empty)
        return
    await message.answer(f"{texts.undo_done}\n{format_receipt(receipt, texts)}")


@router.message(F.photo)
async def handle_photo(message: Message, bot: Bot, service: ReceiptService, texts: Texts) -> None:
    """Process the largest size of a receipt photo.

    Args:
        message: Incoming Telegram message containing receipt photos.
        bot: Telegram bot used to download the photo.
        service: Receipt service that processes the downloaded image.
        texts: Localized response texts.
    """
    if message.photo:
        await _process_image(message, bot, message.photo[-1], service, texts)


@router.message(F.document.mime_type.startswith("image/"))
async def handle_image_document(
    message: Message, bot: Bot, service: ReceiptService, texts: Texts
) -> None:
    """Process a receipt image sent as an uncompressed file.

    Args:
        message: Incoming Telegram message containing an image file.
        bot: Telegram bot used to download the file.
        service: Receipt service that processes the downloaded image.
        texts: Localized response texts.
    """
    if message.document:
        await _process_image(message, bot, message.document, service, texts)


async def _process_image(
    message: Message, bot: Bot, image: Downloadable, service: ReceiptService, texts: Texts
) -> None:
    if message.from_user is None:
        return
    async with ChatActionSender.typing(bot=bot, chat_id=message.chat.id):
        buffer = BytesIO()
        await bot.download(image, destination=buffer)
        try:
            receipt = await service.process_photo(
                message.from_user.id, buffer.getvalue(), message.date
            )
            reply = format_receipt(receipt, texts)
        except ReceiptUnreadable:
            reply = texts.unreadable
        except ReceiptAlreadySaved:
            reply = texts.duplicate
        except RecognitionUnavailable:
            reply = texts.recognition_unavailable
        except Exception:
            logger.exception("Failed to process receipt image")
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
