"""M3 training schema and canonical M3 prediction model re-exports."""

from pydantic import BaseModel, ConfigDict, Field

from common.models import CategoryPrediction, CategoryScore, ClassifiedItem, ClassifiedReceipt


class TrainingExample(BaseModel):
    """Training line, with one or more pipe-separated category labels."""

    model_config = ConfigDict(frozen=False)
    text: str = Field(min_length=1, description="Product name or description text")
    category: str = Field(min_length=1, description="Category labels separated by |")
    source: str | None = Field(default=None, description="Dataset or augmentation source")
    group_id: str | None = Field(
        default=None,
        description="Split group; variants of the same product share it",
    )


__all__ = [
    "TrainingExample",
    "CategoryScore",
    "CategoryPrediction",
    "ClassifiedItem",
    "ClassifiedReceipt",
]
