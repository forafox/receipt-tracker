from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field


class Item(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    name: str = Field(min_length=1)
    quantity: Decimal = Field(default=Decimal("1"), gt=0)
    total: Decimal = Field(ge=0)


class Receipt(BaseModel):
    receipt_id: str = Field(min_length=1)
    purchased_at: datetime
    items: list[Item] = Field(min_length=1)


class CategoryScore(BaseModel):
    category: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)


class CategoryPrediction(BaseModel):
    categories: list[CategoryScore] = Field(min_length=1)


class ClassifiedItem(Item):
    categories: list[CategoryScore] = Field(min_length=1)


class ClassifiedReceipt(BaseModel):
    receipt_id: str = Field(min_length=1)
    purchased_at: datetime
    items: list[ClassifiedItem] = Field(min_length=1)
