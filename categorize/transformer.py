from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common import CategoryPrediction


class TransformerCategoryClassifier:
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
        if not item_name.strip():
            msg = "item name cannot be empty"
            raise ValueError(msg)

        encoded = self._tokenizer(
            item_name,
            truncation=True,
            padding=True,
            max_length=self._max_length,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = self._model(**encoded).logits[0]
            probabilities = torch.softmax(logits, dim=-1)
        confidence, label_id = torch.max(probabilities, dim=-1)
        return CategoryPrediction(
            category=self._id_to_label[int(label_id.item())],
            confidence=round(float(confidence.item()), 4),
        )
