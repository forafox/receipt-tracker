"""Behavioural tests for M3 contracts and inference (no model download)."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest
import torch

from categorize.manifest import (
    MANIFEST_FILENAME,
    ModelManifest,
    load_manifest,
    save_manifest,
    sha256_file,
)
from categorize.transformer import TransformerCategoryClassifier
from common.contracts import Item, Receipt
from common.models import CategoryScore, ClassifiedItem, ClassifiedReceipt

LABELS = {0: "food", 1: "household", 2: "personal_care", 3: "medicine", 4: "cafe"}


class _FakeConfig:
    def __init__(self, num_labels: int, id2label: dict[int, str]) -> None:
        self.num_labels = num_labels
        self.id2label = id2label


class _FakeOutput:
    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits


class _FakeModel:
    def __init__(self, weights: list[float], id2label: dict[int, str]) -> None:
        self._logits = torch.tensor([weights])
        self.config = _FakeConfig(len(weights), id2label)

    def eval(self) -> _FakeModel:
        return self

    def __call__(self, **_: object) -> _FakeOutput:
        batch_size = len(_.get("input_ids", [0]))
        return _FakeOutput(self._logits.repeat(batch_size, 1))


class _FakeTokenizer:
    def __call__(self, texts: object, **__: object) -> dict[str, torch.Tensor]:
        batch_size = len(texts) if isinstance(texts, list) else 1
        return {
            "input_ids": torch.tensor([[1, 2, 3]] * batch_size),
            "attention_mask": torch.tensor([[1, 1, 1]] * batch_size),
        }


def _classifier(
    weights: list[float], thresholds: dict[str, float] | None = None
) -> TransformerCategoryClassifier:
    model = _FakeModel(weights, LABELS)
    return TransformerCategoryClassifier(model, _FakeTokenizer(), LABELS, thresholds=thresholds)


def test_classified_models_extend_canonical_contracts() -> None:
    assert issubclass(ClassifiedItem, Item)
    assert issubclass(ClassifiedReceipt, Receipt)
    item = ClassifiedItem(
        name="milk",
        quantity=Decimal("1"),
        price=Decimal("1.00"),
        sum=Decimal("1.00"),
        categories=[CategoryScore(category="food", confidence=0.9)],
    )
    assert isinstance(item, Item)
    receipt = ClassifiedReceipt(
        store="shop",
        datetime=datetime(2026, 1, 1, tzinfo=UTC),
        total=Decimal("1.00"),
        items=[item],
    )
    assert isinstance(receipt, Receipt)
    assert isinstance(receipt.items[0], ClassifiedItem)


def test_threshold_keeps_only_confident_labels() -> None:
    prediction = _classifier([4.0, -4.0, -4.0, -4.0, -4.0]).predict("milk")
    assert [score.category for score in prediction.categories] == ["food"]
    assert prediction.fallback is False
    assert prediction.threshold == 0.5


def test_independent_labels_above_threshold_are_all_returned() -> None:
    prediction = _classifier([4.0, 4.0, -4.0, -4.0, -4.0]).predict("soap")
    assert {score.category for score in prediction.categories} == {"food", "household"}
    assert prediction.fallback is False


def test_per_label_thresholds_are_applied() -> None:
    classifier = _classifier(
        [0.0, 0.0, -9.0, -9.0, -9.0],
        thresholds={"food": 0.6, "household": 0.4},
    )
    prediction = classifier.predict("item")
    assert {score.category for score in prediction.categories} == {"household"}


def test_low_confidence_falls_back_to_argmax_visibly() -> None:
    prediction = _classifier([-4.0, -4.0, -4.0, -4.0, -3.0]).predict("unknown")
    assert [score.category for score in prediction.categories] == ["cafe"]
    assert prediction.fallback is True


def test_batch_predictions_match_single_predictions() -> None:
    classifier = _classifier([4.0, 4.0, -4.0, -4.0, -4.0])

    batched = classifier.predict_many(["milk", "soap"])

    assert batched == [classifier.predict("milk"), classifier.predict("soap")]
    assert classifier.predict_many([]) == []


def test_empty_name_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        _classifier([0.0] * 5).predict("   ")


def test_missing_label_index_is_rejected() -> None:
    model = _FakeModel([0.0] * 5, LABELS)
    with pytest.raises(ValueError, match="indices"):
        TransformerCategoryClassifier(model, _FakeTokenizer(), {0: "food"})


def test_duplicate_label_names_are_rejected() -> None:
    model = _FakeModel([0.0] * 5, {0: "a", 1: "a", 2: "b", 3: "c", 4: "d"})
    with pytest.raises(ValueError, match="duplicate"):
        TransformerCategoryClassifier(
            model, _FakeTokenizer(), {0: "a", 1: "a", 2: "b", 3: "c", 4: "d"}
        )


def test_config_and_metadata_disagreement_is_rejected() -> None:
    model = _FakeModel([0.0] * 5, {index: f"other_{index}" for index in range(5)})
    with pytest.raises(ValueError, match="disagree"):
        TransformerCategoryClassifier(model, _FakeTokenizer(), LABELS)


def test_manifest_roundtrip(tmp_path: Path) -> None:
    data = tmp_path / "dataset.csv"
    data.write_text("item,category\nmilk,food\n", encoding="utf-8")
    manifest = ModelManifest(
        model_version="1.1.0",
        base_model="cointegrated/rubert-tiny",
        created_at="2026-01-01T00:00:00+00:00",
        dataset_path=str(data),
        dataset_sha256=sha256_file(data),
        hyperparameters={"epochs": 1},
        thresholds={"default": 0.5},
        metrics={"micro_f1": 0.9},
        label_names=list(LABELS.values()),
    )
    save_manifest(tmp_path, manifest)

    loaded = load_manifest(tmp_path)
    assert loaded == manifest
    assert (tmp_path / MANIFEST_FILENAME).exists()
    assert len(manifest.dataset_sha256 or "") == 64


def test_missing_manifest_returns_none(tmp_path: Path) -> None:
    assert load_manifest(tmp_path) is None
