from typing import Protocol

from common import CategoryPrediction


class CategoryClassifier(Protocol):
    def predict(self, item_name: str) -> CategoryPrediction: ...
