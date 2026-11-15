from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from transformers.trainer_utils import EvalPrediction

from categorize.prepare_dataset import build
from categorize.training import (
    _compute_metrics,
    _compute_pos_weight,
    _oversample_training_dataset,
    _stratified_split,
)


def _write(path: Path, rows: list[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["text", "category"])
        writer.writerows(rows)


def test_build_merges_curated_labels_and_reports_duplicates(tmp_path: Path) -> None:
    base = tmp_path / "base.csv"
    curated = tmp_path / "curated.csv"
    output = tmp_path / "out.csv"
    _write(
        base,
        [
            ("Сода", "household"),
            ("сода", "food"),
            ("Хлеб", "food"),
            ("Хлеб", "food"),
            ("Мыло", "household"),
        ],
    )
    _write(curated, [("сода", "food|household")])

    stats = build(base, curated, output)

    rows = list(csv.DictReader(output.open(encoding="utf-8")))
    assert stats["rows"] == 3
    assert stats["duplicate_texts_removed"] == 2
    assert stats["multilabel_rows"] == 1
    assert {"text": "Сода", "category": "food|household"} in rows
    assert {"text": "Мыло", "category": "household"} in rows
    assert stats["missing_labels"] == ["cafe", "medicine", "personal_care"]


def _dataset() -> Dataset:
    return Dataset.from_dict(
        {
            "text": ["a", "b", "c", "d", "e"],
            "labels": [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
            ],
        }
    )


def test_stratified_split_keeps_every_label_in_test() -> None:
    train, test = _stratified_split(_dataset(), test_size=0.2)
    test_vectors = np.asarray(test["labels"])
    assert test_vectors.sum(axis=0).min() > 0
    assert len(train) + len(test) == 5


def test_oversample_caps_rare_label_repetition() -> None:
    balanced = _oversample_training_dataset(_dataset())
    texts = balanced["text"]
    assert texts.count("d") <= 4
    assert texts.count("c") <= 4
    assert texts.count("e") <= 4


def test_oversample_keeps_near_equal_labels_untouched() -> None:
    dataset = Dataset.from_dict(
        {"text": ["a", "b", "c", "d"], "labels": [[1, 0], [1, 0], [0, 1], [0, 1]]}
    )
    balanced = _oversample_training_dataset(dataset)
    assert len(balanced) == 4


def test_pos_weight_is_clipped() -> None:
    weight = _compute_pos_weight(_dataset(), cap=3.0)
    assert torch.all(weight >= 1.0)
    assert torch.all(weight <= 3.0)


def test_compute_metrics_reports_per_label_scores() -> None:
    labels = np.array([[1, 0], [0, 1]])
    logits = np.array([[10.0, -10.0], [-10.0, 10.0]])
    metrics = _compute_metrics(
        EvalPrediction(predictions=logits, label_ids=labels),
        label_names=["food", "cafe"],
    )
    assert metrics["exact_match"] == 1.0
    assert metrics["micro_f1"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["f1_food"] == 1.0
    assert metrics["f1_cafe"] == 1.0
