from typing import Protocol

from common import CategoryPrediction


class CategoryClassifier(Protocol):
    def predict(self, item_name: str) -> CategoryPrediction:
        """Return category prediction for one receipt item."""
