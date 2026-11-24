"""Receipt item categorization. Heavy transformer dependencies load on demand."""

from typing import TYPE_CHECKING

from categorize.classifier import CategoryClassifier
from categorize.service import CategoryService

if TYPE_CHECKING:
    from categorize.transformer import TransformerCategoryClassifier

__all__ = ["CategoryClassifier", "CategoryService", "TransformerCategoryClassifier"]


def __getattr__(name: str) -> object:
    if name == "TransformerCategoryClassifier":
        from categorize.transformer import TransformerCategoryClassifier

        return TransformerCategoryClassifier
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
