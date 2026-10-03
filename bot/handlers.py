import logging
from io import BytesIO

from aiogram import Bot, F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import Message

from bot.service import ReceiptService
from bot.texts import Texts

logger = logging.getLogger(__name__)

router = Router()


@router.message(CommandStart())
async def handle_start(message: Message, texts: Texts) -> None:
    await message.answer(texts.start)


@router.message(Command("help"))
async def handle_help(message: Message, texts: Texts) -> None:
    await message.answer(texts.help)


@router.message(Command("month"))
async def handle_month(message: Message, service: ReceiptService) -> None:
    if message.from_user is None:
        return
    await message.answer(await service.monthly_report(message.from_user.id))


@router.message(F.photo)
async def handle_photo(message: Message, bot: Bot, service: ReceiptService, texts: Texts) -> None:
    if message.from_user is None or not message.photo:
        return
    buffer = BytesIO()
    await bot.download(message.photo[-1], destination=buffer)
    try:
        reply = await service.process_photo(message.from_user.id, buffer.getvalue())
    except Exception:
        logger.exception("Failed to process receipt photo")
        reply = texts.error
    await message.answer(reply)


@router.message()
async def handle_unknown(message: Message, texts: Texts) -> None:
    await message.answer(texts.unknown)
