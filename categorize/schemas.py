"""Pydantic schemas for the categorize module.

Contracts from architecture.pdf:
- TrainingExample(text, category): both min_length=1, used for training data
- CategoryScore(category, confidence): category is str, confidence is float in [0,1]
- CategoryPrediction(categories): list of CategoryScore, min_length=1
- ClassifiedItem(Item): Item + categories
- ClassifiedReceipt(receipt_id, purchased_at, items)
"""
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from common.models import Item


class TrainingExample(BaseModel):
    """A training example consisting of text and its category label."""
    model_config = ConfigDict(frozen=False)
    
    text: str = Field(min_length=1, description="Product name or description text")
    category: str = Field(min_length=1, description="Category label (can be multi-label separated by |)")


class CategoryScore(BaseModel):
    """A category with an associated confidence score."""
    category: str = Field(min_length=1, description="Category name")
    confidence: float = Field(ge=0, le=1, description="Confidence score in [0, 1]")


class CategoryPrediction(BaseModel):
    """Prediction result containing categories with confidences."""
    categories: list[CategoryScore] = Field(
        min_length=1, 
        description="List of categories with confidence scores, sorted by confidence descending"
    )


class ClassifiedItem(Item):
    """An item with its predicted categories appended."""
    categories: list[CategoryScore] = Field(
        min_length=1, 
        description="Predicted categories for this item"
    )


class ClassifiedReceipt(BaseModel):
    """A receipt with all items classified."""
    receipt_id: str = Field(min_length=1, description="Unique receipt identifier")
    purchased_at: str = Field(description="Timezone-aware purchase datetime")
    items: list[ClassifiedItem] = Field(min_length=1, description=" classified items")