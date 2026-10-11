from datetime import datetime
from typing import Protocol

from common.contracts import Receipt


class ReceiptService(Protocol):
    """Receipt processing used by Telegram handlers."""

    async def process_photo(self, user_id: int, photo: bytes, received_at: datetime) -> Receipt:
        """Process and save a receipt photo for a user.

        Args:
            user_id: Telegram user identifier.
            photo: Raw receipt image bytes.
            received_at: Time the photo was sent.

        Returns:
            Parsed receipt.
        """
        ...
