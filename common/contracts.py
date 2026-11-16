from decimal import Decimal
from enum import StrEnum

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field


class ReceiptWarning(StrEnum):
    """Codes of non-fatal problems found while extracting a receipt.

    The extraction module adds these codes to ``Receipt.warnings``, other modules read them.
    The string values are stored with saved receipts, so an existing value must not change.

    Attributes:
        STORE_NOT_FOUND: The merchant name was not found.
        DATETIME_NOT_FOUND: The purchase date and time were not found.
        TOTAL_NOT_FOUND: The receipt total was not found.
        ITEMS_INCOMPLETE: Some lines looked like items but could not be parsed.
        ITEM_AMOUNT_MISMATCH: Quantity times price differs from the line sum for some item.
        ITEMS_TOTAL_MISMATCH: Item sums do not add up to the receipt total.
    """

    STORE_NOT_FOUND = "store_not_found"
    DATETIME_NOT_FOUND = "datetime_not_found"
    TOTAL_NOT_FOUND = "total_not_found"
    ITEMS_INCOMPLETE = "items_incomplete"
    ITEM_AMOUNT_MISMATCH = "item_amount_mismatch"
    ITEMS_TOTAL_MISMATCH = "items_total_mismatch"


class Item(BaseModel):
    """A purchase item extracted from a receipt.

    Attributes:
        name: Product name as recognized on the receipt.
        quantity: Number of units or weight, if available.
        price: Price per unit, if available.
        sum: Final amount for the receipt line.
        category: Category code from the category model (``food``, ``household``,
            ``personal_care``, ``medicine``, ``cafe``), or None when the item was not
            classified.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    quantity: Decimal | None = None
    price: Decimal | None = None
    sum: Decimal
    category: str | None = None


class Receipt(BaseModel):
    """Structured receipt produced by the extraction module.

    Attributes:
        store: Recognized merchant name, or None when unavailable.
        datetime: Timezone-aware receipt date and time, if available.
        total: Final receipt amount, if available.
        items: Purchase items extracted from the receipt.
        warnings: Non-fatal extraction warnings, values of ``ReceiptWarning``.
    """

    model_config = ConfigDict(extra="forbid")

    store: str | None = None
    datetime: AwareDatetime | None = None
    total: Decimal | None = None
    items: list[Item] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
