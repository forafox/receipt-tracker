from decimal import Decimal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field


class Item(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    quantity: Decimal | None = None
    price: Decimal | None = None
    sum: Decimal


class Receipt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    store: str | None = None
    datetime: AwareDatetime | None = None
    total: Decimal | None = None
    items: list[Item] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
