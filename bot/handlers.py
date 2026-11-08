import logging
from io import BytesIO

from aiogram import Bot, F, Router
from aiogram.filters import Command, CommandStart
from aiogram.filters.callback_data import CallbackData
from aiogram.types import (
    CallbackQuery,
    Downloadable,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)
from aiogram.utils.chat_action import ChatActionSender

from bot.service import ReceiptService
from bot.texts import Texts, format_history, format_monthly_summary, format_receipt
from pipeline import (
    ReceiptAlreadySaved,
    ReceiptImageTooLarge,
    ReceiptUnreadable,
    RecognitionUnavailable,
)

logger = logging.getLogger(__name__)

router = Router()

# Bot API refuses to send files larger than this to bots.
MAX_DOWNLOAD_BYTES = 20 * 1024 * 1024


class DuplicateDecision(CallbackData, prefix="duplicate"):
    """Button data of the duplicate question.

    Attributes:
        save: Whether the user chose to save the receipt again.
        pending_id: Identifier of the rejected receipt the question is about.
    """

    save: bool
    pending_id: int


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
    if not message.document:
        return
    if (message.document.file_size or 0) > MAX_DOWNLOAD_BYTES:
        await message.answer(texts.file_too_large)
        return
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
        except ReceiptImageTooLarge:
            reply = texts.image_too_large
        except ReceiptAlreadySaved as duplicate:
            keyboard = _duplicate_keyboard(texts, duplicate.pending_id)
            await message.answer(texts.duplicate, reply_markup=keyboard)
            return
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


@router.callback_query(DuplicateDecision.filter(F.save))
async def handle_save_duplicate(
    callback: CallbackQuery,
    callback_data: DuplicateDecision,
    service: ReceiptService,
    texts: Texts,
) -> None:
    """Save a suspected duplicate after the user confirmed it.

    Args:
        callback: Button press on the duplicate question.
        callback_data: Decision and the receipt it applies to.
        service: Receipt service that keeps the pending duplicate.
        texts: Localized response texts.
    """
    receipt = await service.save_duplicate(callback.from_user.id, callback_data.pending_id)
    if receipt is None:
        reply = texts.duplicate_expired
    else:
        reply = f"{texts.duplicate_saved}\n{format_receipt(receipt, texts)}"
    await _replace_question(callback, reply)


@router.callback_query(DuplicateDecision.filter(~F.save))
async def handle_skip_duplicate(
    callback: CallbackQuery,
    callback_data: DuplicateDecision,
    service: ReceiptService,
    texts: Texts,
) -> None:
    """Drop a suspected duplicate.

    Args:
        callback: Button press on the duplicate question.
        callback_data: Decision and the receipt it applies to.
        service: Receipt service that keeps the pending duplicate.
        texts: Localized response texts.
    """
    service.discard_duplicate(callback.from_user.id, callback_data.pending_id)
    await _replace_question(callback, texts.duplicate_skipped)


def _duplicate_keyboard(texts: Texts, pending_id: int) -> InlineKeyboardMarkup:
    save = DuplicateDecision(save=True, pending_id=pending_id).pack()
    skip = DuplicateDecision(save=False, pending_id=pending_id).pack()
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text=texts.duplicate_save, callback_data=save),
                InlineKeyboardButton(text=texts.duplicate_skip, callback_data=skip),
            ]
        ]
    )


async def _replace_question(callback: CallbackQuery, reply: str) -> None:
    # Editing the question removes its buttons, so they cannot be pressed twice.
    if isinstance(callback.message, Message):
        await callback.message.edit_text(reply)
    await callback.answer()
