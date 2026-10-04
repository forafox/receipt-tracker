from typing import Protocol


class ReceiptService(Protocol):
    async def process_photo(self, user_id: int, photo: bytes) -> str: ...

    async def monthly_report(self, user_id: int) -> str: ...


class StubReceiptService:
    async def process_photo(self, user_id: int, photo: bytes) -> str:
        return "Чек получен. Распознавание пока не подключено."

    async def monthly_report(self, user_id: int) -> str:
        return "Отчёт за месяц пока недоступен."
