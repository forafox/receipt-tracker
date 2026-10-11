from datetime import datetime
from typing import Protocol

from common.contracts import Receipt
from pipeline import MonthlySummary


class ReceiptService(Protocol):
    """Receipt processing and reports used by Telegram handlers."""

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

    async def monthly_summary(self, user_id: int, moment: datetime) -> MonthlySummary:
        """Sum a user's spending for the month containing a moment.

        Args:
            user_id: Telegram user identifier.
            moment: Timezone-aware moment inside the month.

        Returns:
            Spending summary for the month.
        """
        ...
