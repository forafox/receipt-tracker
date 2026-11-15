"""M3 output models; M2 owns the single Item/Receipt input contract.

The aliases below are kept for backwards compatibility with existing M3 imports.
There is no second definition of Item or Receipt.
"""

from pydantic import BaseModel, Field

from common.contracts import Item, Receipt


class CategoryScore(BaseModel):
    category: str = Field(min_length=1)
    # A sigmoid score for this category, not a calibrated probability of correctness.
    confidence: float = Field(ge=0, le=1)


class CategoryPrediction(BaseModel):
    categories: list[CategoryScore] = Field(min_length=1)
    # Single threshold shared by training metrics and inference (see transformer).
    threshold: float = Field(default=0.5, ge=0, le=1)
    # True when nothing reached the threshold and the most confident label was
    # returned as an explicit fallback instead of failing.
    fallback: bool = False


class ClassifiedItem(Item):
    categories: list[CategoryScore] = Field(min_length=1)


class ClassifiedReceipt(Receipt):
    """Original M2 receipt fields plus per-item M3 predictions."""

    # Narrowing Receipt.items is intentional; Pydantic still builds the parent
    # fields. mypy cannot express covariant field narrowing.
    items: list[ClassifiedItem] = Field(default_factory=list)  # type: ignore[assignment]
