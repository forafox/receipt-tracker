import asyncio
import logging

from aiogram import Bot, Dispatcher
from aiogram.types import BotCommand

from bot.config import BotSettings
from bot.handlers import router
from bot.middleware import LanguageMiddleware
from bot.service import StubReceiptService
from bot.texts import DEFAULT_TEXTS, TEXTS_BY_LANGUAGE, Texts


def commands(texts: Texts) -> list[BotCommand]:
    """Build the bot command menu for one locale.

    Args:
        texts: Localized command descriptions.

    Returns:
        Commands suitable for registration with Telegram.
    """
    return [
        BotCommand(command="month", description=texts.month_command),
        BotCommand(command="help", description=texts.help_command),
    ]


async def set_commands(bot: Bot) -> None:
    """Register default and localized command menus.

    Args:
        bot: Telegram bot whose command menus are configured.
    """
    await bot.set_my_commands(commands(DEFAULT_TEXTS))
    for language, texts in TEXTS_BY_LANGUAGE.items():
        await bot.set_my_commands(commands(texts), language_code=language)


async def main() -> None:
    """Configure and start the Telegram bot polling loop."""
    settings = BotSettings()
    bot = Bot(token=settings.token.get_secret_value())
    dispatcher = Dispatcher(service=StubReceiptService())
    dispatcher.message.middleware(LanguageMiddleware())
    dispatcher.include_router(router)
    await set_commands(bot)
    await dispatcher.start_polling(bot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
