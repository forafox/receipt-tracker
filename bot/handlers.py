import logging
from io import BytesIO

from aiogram import Bot, F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import Message

from bot.service import ReceiptService

logger = logging.getLogger(__name__)

router = Router()

START_TEXT = "Привет! Пришлите фото чека, и я посчитаю покупки по категориям.\nСписок команд: /help"
HELP_TEXT = (
    "Отправьте фото чека, чтобы сохранить покупку.\n"
    "/month — расходы за текущий месяц\n"
    "/help — эта справка"
)
UNKNOWN_TEXT = "Я понимаю только фото чеков. Список команд: /help"
ERROR_TEXT = "Не получилось обработать чек. Попробуйте ещё раз позже."


@router.message(CommandStart())
async def handle_start(message: Message) -> None:
    await message.answer(START_TEXT)


@router.message(Command("help"))
async def handle_help(message: Message) -> None:
    await message.answer(HELP_TEXT)


@router.message(Command("month"))
async def handle_month(message: Message, service: ReceiptService) -> None:
    if message.from_user is None:
        return
    await message.answer(await service.monthly_report(message.from_user.id))


@router.message(F.photo)
async def handle_photo(message: Message, bot: Bot, service: ReceiptService) -> None:
    if message.from_user is None or not message.photo:
        return
    buffer = BytesIO()
    await bot.download(message.photo[-1], destination=buffer)
    try:
        reply = await service.process_photo(message.from_user.id, buffer.getvalue())
    except Exception:
        logger.exception("Failed to process receipt photo")
        reply = ERROR_TEXT
    await message.answer(reply)


@router.message()
async def handle_unknown(message: Message) -> None:
    await message.answer(UNKNOWN_TEXT)
