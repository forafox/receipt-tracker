from typing import Protocol


class ReceiptService(Protocol):
    """Operations used by Telegram handlers to process receipts and reports."""

    async def process_photo(self, user_id: int, photo: bytes) -> str:
        """Process a receipt photo for a user.

        Args:
            user_id: Telegram user identifier.
            photo: Raw receipt image bytes.

        Returns:
            User-facing processing result.
        """
        ...

    async def monthly_report(self, user_id: int) -> str:
        """Build a user's monthly expense report.

        Args:
            user_id: Telegram user identifier.

        Returns:
            User-facing monthly report.
        """
        ...


class StubReceiptService:
    """Placeholder receipt service used before the processing pipeline is connected."""

    async def process_photo(self, user_id: int, photo: bytes) -> str:
        """Return a placeholder response for an uploaded receipt.

        Args:
            user_id: Telegram user identifier.
            photo: Raw receipt image bytes.

        Returns:
            Message explaining that receipt processing is unavailable.
        """
        return "Чек получен. Распознавание пока не подключено."

    async def monthly_report(self, user_id: int) -> str:
        """Return a placeholder monthly report response.

        Args:
            user_id: Telegram user identifier.

        Returns:
            Message explaining that monthly reports are unavailable.
        """
        return "Отчёт за месяц пока недоступен."
