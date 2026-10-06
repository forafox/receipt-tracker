from collections.abc import Awaitable, Callable
from typing import Any

from aiogram import BaseMiddleware
from aiogram.types import TelegramObject, User

from bot.texts import texts_for


class LanguageMiddleware(BaseMiddleware):
    """Add localized response texts to each Telegram handler context."""

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        """Select texts from the sender's language and invoke the handler.

        Args:
            handler: Next Telegram event handler in the middleware chain.
            event: Telegram event being processed.
            data: Mutable handler context populated by aiogram.

        Returns:
            Result produced by the next handler.
        """
        user: User | None = data.get("event_from_user")
        data["texts"] = texts_for(user.language_code if user else None)
        return await handler(event, data)
