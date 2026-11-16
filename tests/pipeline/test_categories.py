from collections.abc import AsyncIterator
from typing import Any

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from pipeline import CategoryApiClient, CategoryUnavailable


async def start(handler: Any) -> AsyncIterator[str]:
    app = web.Application()
    app.router.add_post("/classify/items", handler)
    async with TestServer(app) as server:
        yield str(server.make_url("/"))


async def test_client_returns_most_likely_category_per_item() -> None:
    received: list[Any] = []

    async def classify(request: web.Request) -> web.Response:
        received.append(await request.json())
        return web.json_response(
            {
                "items": [
                    {"name": "МОЛОКО", "categories": [{"category": "food", "confidence": 0.9}]},
                    {"name": "???", "categories": []},
                ]
            }
        )

    async for url in start(classify):
        categories = await CategoryApiClient(url).categorize(["МОЛОКО", "???"])

    assert categories == ["food", None]
    assert received == [
        {"items": [{"name": "МОЛОКО", "total": "0"}, {"name": "???", "total": "0"}]}
    ]


async def test_client_reports_server_error_as_unavailable() -> None:
    async def classify(request: web.Request) -> web.Response:
        return web.json_response({"detail": "Service not initialized"}, status=503)

    async for url in start(classify):
        with pytest.raises(CategoryUnavailable):
            await CategoryApiClient(url).categorize(["МОЛОКО"])


async def test_client_reports_unexpected_answer_as_unavailable() -> None:
    async def classify(request: web.Request) -> web.Response:
        return web.json_response({"items": []})

    async for url in start(classify):
        with pytest.raises(CategoryUnavailable):
            await CategoryApiClient(url).categorize(["МОЛОКО"])


async def test_client_reports_unreachable_api_as_unavailable() -> None:
    with pytest.raises(CategoryUnavailable):
        await CategoryApiClient("http://127.0.0.1:9", timeout=2).categorize(["МОЛОКО"])
