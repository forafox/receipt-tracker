from typing import Protocol

from common.contracts import Receipt


class ReceiptService(Protocol):
    """Receipt processing used by Telegram handlers."""

    async def process_photo(self, user_id: int, photo: bytes) -> Receipt:
        """Process a receipt photo for a user.

        Args:
            user_id: Telegram user identifier.
            photo: Raw receipt image bytes.

        Returns:
            Parsed receipt.
        """
        ...
