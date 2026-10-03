import asyncio
import logging

from aiogram import Bot, Dispatcher

from bot.config import BotSettings
from bot.handlers import router
from bot.service import StubReceiptService


async def main() -> None:
    settings = BotSettings()
    bot = Bot(token=settings.token.get_secret_value())
    dispatcher = Dispatcher(service=StubReceiptService())
    dispatcher.include_router(router)
    await dispatcher.start_polling(bot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
