from decimal import Decimal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field


class Item(BaseModel):
    """A purchase item extracted from a receipt.

    Attributes:
        name: Product name as recognized on the receipt.
        quantity: Number of units or weight, if available.
        price: Price per unit, if available.
        sum: Final amount for the receipt line.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    quantity: Decimal | None = None
    price: Decimal | None = None
    sum: Decimal


class Receipt(BaseModel):
    """Structured receipt produced by the extraction module.

    Attributes:
        store: Recognized merchant name, or None when unavailable.
        datetime: Timezone-aware receipt date and time, if available.
        total: Final receipt amount, if available.
        items: Purchase items extracted from the receipt.
        warnings: Machine-readable non-fatal extraction warnings.
    """

    model_config = ConfigDict(extra="forbid")

    store: str | None = None
    datetime: AwareDatetime | None = None
    total: Decimal | None = None
    items: list[Item] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
