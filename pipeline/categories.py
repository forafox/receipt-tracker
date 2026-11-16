from typing import Any, Protocol

import aiohttp


class CategoryUnavailable(Exception):
    """Raised when item categories cannot be obtained."""


class ItemCategorizer(Protocol):
    """Step that assigns a category to each receipt item."""

    async def categorize(self, names: list[str]) -> list[str | None]:
        """Find the most likely category of each item.

        Args:
            names: Item names as recognized on the receipt.

        Returns:
            One category code or None per name, in the same order.

        Raises:
            CategoryUnavailable: If the categories cannot be obtained.
        """
        ...


class CategoryApiClient:
    """Client of the category model API (``POST /classify/items``)."""

    def __init__(self, base_url: str, timeout: float = 10.0) -> None:
        """Create a client.

        Args:
            base_url: Address of the category API, for example ``http://127.0.0.1:8000``.
            timeout: Limit in seconds for the whole request.
        """
        self._url = f"{base_url.rstrip('/')}/classify/items"
        self._timeout = aiohttp.ClientTimeout(total=timeout)

    async def categorize(self, names: list[str]) -> list[str | None]:
        """Find the most likely category of each item.

        Args:
            names: Item names as recognized on the receipt.

        Returns:
            One category code or None per name, in the same order.

        Raises:
            CategoryUnavailable: If the API cannot be reached or answers unexpectedly.
        """
        # The API item requires a non-negative total, but the model reads only the name.
        payload = {"items": [{"name": name, "total": "0"} for name in names]}
        try:
            async with (
                aiohttp.ClientSession(timeout=self._timeout) as session,
                session.post(self._url, json=payload) as response,
            ):
                response.raise_for_status()
                data: Any = await response.json()
            items = data["items"]
            if len(items) != len(names):
                raise ValueError(f"expected {len(names)} items, got {len(items)}")
            return [_top_category(item) for item in items]
        except (aiohttp.ClientError, TimeoutError, KeyError, TypeError, ValueError) as error:
            raise CategoryUnavailable(str(error)) from error


def _top_category(item: Any) -> str | None:
    categories = item["categories"]
    if not categories:
        return None
    # The API sorts categories by confidence, the most likely first.
    category: str = categories[0]["category"]
    return category
