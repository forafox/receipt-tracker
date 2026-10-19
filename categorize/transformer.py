"""Transformer-based CategoryClassifier implementation.

PDF requirements:
- Uses cointegrated/rubert-tiny as base model
- problem_type="multi_label_classification"
- Tokenization with max_length=64
- sigmoid on logits, threshold 0.5
- If no category passes threshold, return argmax
- Sort by confidence descending
- Empty string → ValueError
- Returns CategoryPrediction with CategoryScore list
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common.models import CategoryScore, CategoryPrediction


class TransformerCategoryClassifier:
    """Category classifier using a fine-tuned transformer model.

    PDF specifications:
    - Base model: cointegrated/rubert-tiny
    - problem_type: multi_label_classification
    - max_length: 64 (configurable)
    - Threshold: 0.5 on sigmoid probabilities
    - Empty item name raises ValueError
    """
    
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        id_to_label: dict[int, str],
        max_length: int = 64,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._id_to_label = id_to_label
        self._max_length = max_length
        self._model.eval()

    @classmethod
    def load(cls, path: Path) -> TransformerCategoryClassifier:
        """Load classifier from a saved model directory.
        
        Expected directory structure:
        - category_metadata.json (with id_to_label and max_length)
        - tokenizer/ and model/ from transformers save_pretrained
        """
        with (path / "category_metadata.json").open(encoding="utf-8") as file:
            metadata = json.load(file)
        id_to_label = {int(key): value for key, value in metadata["id_to_label"].items()}
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path)
        return cls(
            model=model,
            tokenizer=tokenizer,
            id_to_label=id_to_label,
            max_length=int(metadata.get("max_length", 64)),
        )

    @staticmethod
    def save_metadata(path: Path, id_to_label: dict[int, str], max_length: int) -> None:
        """Save category metadata to the model directory."""
        path.mkdir(parents=True, exist_ok=True)
        with (path / "category_metadata.json").open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "id_to_label": {str(key): value for key, value in id_to_label.items()},
                    "max_length": max_length,
                },
                file,
                ensure_ascii=False,
                indent=2,
            )

    def predict(self, item_name: str) -> CategoryPrediction:
        """Classify the given item name and return categories with confidences.

        PDF behavior:
        1. Tokenize with max_length (default 64)
        2. Forward pass through model, get logits
        3. Apply sigmoid to get probabilities per category
        4. Filter categories with probability >= 0.5 threshold
        5. If none pass threshold, select argmax (most confident)
        6. Sort selected categories by confidence descending
        7. Return CategoryPrediction with CategoryScore objects

        Raises:
            ValueError: If item_name is empty or whitespace-only.
        """
        if not item_name.strip():
            msg = "item name cannot be empty"
            raise ValueError(msg)

        # Tokenize the item name
        encoded = self._tokenizer(
            item_name,
            truncation=True,
            padding=True,
            max_length=self._max_length,
            return_tensors="pt",
        )

        with torch.no_grad():
            logits = self._model(**encoded).logits[0]

        # Apply sigmoid to get probabilities for each category
        probabilities = torch.sigmoid(logits)

        # Build list of (category, confidence) pairs
        scores = [
            CategoryScore(
                category=self._id_to_label[idx],
                confidence=round(float(prob), 4),
            )
            for idx, prob in enumerate(probabilities.tolist())
        ]

        # Filter by threshold (0.5)
        selected = [score for score in scores if score.confidence >= 0.5]

        # If no category passes threshold, fall back to argmax
        if not selected:
            max_score = max(scores, key=lambda s: s.confidence)
            selected = [max_score]

        # Sort by confidence descending
        selected.sort(key=lambda score: score.confidence, reverse=True)

        return CategoryPrediction(categories=selected)