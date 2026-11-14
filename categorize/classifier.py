"""CategoryClassifier protocol matching architecture.pdf specification.

The PDF defines the contract: predict(item_name) -> CategoryPrediction.
This module provides the Protocol and the Pydantic models used for the contract.
"""

from typing import Protocol

from common.models import CategoryPrediction


class CategoryClassifier(Protocol):
    """Protocol for category classification per architecture.pdf.

    PDF requirement: CategoryClassifier with predict(item_name) -> CategoryPrediction.
    """

    def predict(self, item_name: str) -> CategoryPrediction:
        """Classify a product item name into categories.

        Args:
            item_name: Full or partial product name as it appears on a receipt.

        Returns:
            CategoryPrediction containing categories with confidence scores.

        Raises:
            ValueError: If item_name is empty or whitespace-only.
        """
        ...
