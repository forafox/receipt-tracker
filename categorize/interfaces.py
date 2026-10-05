from typing import Protocol

from common import ClassifiedReceipt, Receipt


class ReceiptCategorizer(Protocol):
    """Boundary used by M4 after M2 has extracted a structured receipt."""

    def classify_receipt(self, receipt: Receipt) -> ClassifiedReceipt:
        """Return a receipt with category and confidence on every item."""
