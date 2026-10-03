from typing import Any
from unittest.mock import AsyncMock, MagicMock

from bot.middleware import LanguageMiddleware
from bot.texts import EN, RU


async def run_middleware(data: dict[str, Any]) -> dict[str, Any]:
    handler = AsyncMock()
    await LanguageMiddleware()(handler, MagicMock(), data)
    handler.assert_awaited_once()
    result: dict[str, Any] = handler.await_args.args[1]
    return result


async def test_middleware_uses_user_language() -> None:
    user = MagicMock(language_code="ru")

    data = await run_middleware({"event_from_user": user})

    assert data["texts"] is RU


async def test_middleware_falls_back_without_user() -> None:
    data = await run_middleware({})

    assert data["texts"] is EN
