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
import logging
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from categorize.manifest import load_manifest
from categorize.text import denoise
from common.models import CategoryPrediction, CategoryScore

logger = logging.getLogger(__name__)

# One threshold shared by inference and offline metrics.
DEFAULT_THRESHOLD = 0.5


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
        thresholds: dict[str, float] | None = None,
        temperature: float = 1.0,
    ) -> None:
        # Fail on corrupt/mismatched metadata at loading time rather than at predict().
        num_labels = getattr(model.config, "num_labels", None)
        if not isinstance(num_labels, int) or num_labels < 1:
            raise ValueError("model.config.num_labels must be a positive integer")
        if set(id_to_label) != set(range(num_labels)):
            raise ValueError(
                f"id_to_label must contain indices 0..{num_labels - 1}; got {sorted(id_to_label)}"
            )
        if any(not isinstance(label, str) or not label.strip() for label in id_to_label.values()):
            raise ValueError("id_to_label contains an empty/non-string category")
        if len(set(id_to_label.values())) != num_labels:
            raise ValueError("id_to_label contains duplicate category names")
        config_labels = getattr(model.config, "id2label", None)
        if (
            config_labels is not None
            and {int(index): label for index, label in config_labels.items()} != id_to_label
        ):
            raise ValueError("model config id2label and metadata id_to_label disagree")
        self._model = model
        self._tokenizer = tokenizer
        self._id_to_label = id_to_label
        self._max_length = max_length
        self._thresholds = thresholds or {}
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self._temperature = temperature
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
        manifest = load_manifest(path)
        if manifest is not None and manifest.label_names:
            expected = [id_to_label[index] for index in sorted(id_to_label)]
            if manifest.label_names != expected:
                raise ValueError(
                    "model_manifest.json label_names disagree with category_metadata.json"
                )
            logger.info("Loaded model version %s", manifest.model_version)
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path)
        return cls(
            model=model,
            tokenizer=tokenizer,
            id_to_label=id_to_label,
            max_length=int(metadata.get("max_length", 64)),
            thresholds={
                str(label): float(value) for label, value in metadata.get("thresholds", {}).items()
            },
            temperature=float(metadata.get("temperature", 1.0)),
        )

    @staticmethod
    def save_metadata(
        path: Path,
        id_to_label: dict[int, str],
        max_length: int,
        thresholds: dict[str, float] | None = None,
        temperature: float = 1.0,
    ) -> None:
        """Save category metadata, calibrated thresholds and temperature."""
        path.mkdir(parents=True, exist_ok=True)
        with (path / "category_metadata.json").open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "id_to_label": {str(key): value for key, value in id_to_label.items()},
                    "max_length": max_length,
                    "thresholds": thresholds or {},
                    "temperature": temperature,
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

        # OCR artifacts (VAT prefixes, homoglyphs, trailing quantities) are
        # removed before tokenization; the model was trained on both forms.
        cleaned = denoise(item_name) or item_name.strip()

        encoded = self._tokenizer(
            cleaned,
            truncation=True,
            padding=True,
            max_length=self._max_length,
            return_tensors="pt",
        )

        with torch.no_grad():
            logits = self._model(**encoded).logits[0]

        # Sigmoid returns independent scores in [0, 1], not calibrated confidence.
        # Per-label thresholds are tuned on the validation split; DEFAULT_THRESHOLD
        # remains the fallback for labels absent from the metadata.
        probabilities = torch.sigmoid(logits / self._temperature)

        # Build list of (category, confidence) pairs
        scores = [
            CategoryScore(
                category=self._id_to_label[idx],
                confidence=round(float(prob), 4),
            )
            for idx, prob in enumerate(probabilities.tolist())
        ]

        selected = [
            score
            for score in scores
            if score.confidence >= self._thresholds.get(score.category, DEFAULT_THRESHOLD)
        ]

        # No category passed: return argmax as an explicit, visible fallback.
        fallback = not selected
        if fallback:
            selected = [max(scores, key=lambda score: score.confidence)]

        # Sort by confidence descending
        selected.sort(key=lambda score: score.confidence, reverse=True)

        return CategoryPrediction(
            categories=selected,
            threshold=DEFAULT_THRESHOLD,
            fallback=fallback,
        )
