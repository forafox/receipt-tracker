from __future__ import annotations

import csv
from pathlib import Path

from datasets import Dataset

from categorize.augment import add_noise_variants, assign_groups, compose_multilabel
from categorize.external_multilabel import _labels_for
from categorize.prepare_dataset import build
from categorize.schemas import TrainingExample
from categorize.training import _stratified_split


def _seed_examples() -> list[TrainingExample]:
    return [
        TrainingExample(text="молоко", category="food"),
        TrainingExample(text="шампунь", category="personal_care"),
        TrainingExample(text="аспирин", category="medicine"),
        TrainingExample(text="порошок", category="household"),
        TrainingExample(text="капучино", category="cafe"),
    ]


def test_compose_multilabel_covers_every_pair() -> None:
    rows = compose_multilabel(_seed_examples(), pairs_per_pair=2, triples=5, seed=1)
    pairs = {row.category for row in rows if row.source == "synthetic_pair"}
    assert len(pairs) == 10
    assert all(row.group_id for row in rows)
    assert any("|" in row.category for row in rows)


def test_noise_variants_keep_label_and_group() -> None:
    rows = assign_groups([TrainingExample(text="молоко", category="food")])
    noisy = add_noise_variants(rows, fraction=1.0, seed=1)
    assert noisy
    assert noisy[0].category == "food"
    assert noisy[0].group_id == rows[0].group_id


def test_grouped_split_does_not_leak_variants() -> None:
    dataset = Dataset.from_dict(
        {
            "text": ["a", "a noisy", "b", "b noisy", "c", "d", "e", "f"],
            "labels": [[1, 0], [1, 0], [0, 1], [0, 1], [1, 1], [1, 0], [0, 1], [1, 1]],
            "group_id": ["a", "a", "b", "b", "c", "d", "e", "f"],
        }
    )
    train, test = _stratified_split(dataset, test_size=0.25)
    train_groups = set(train["group_id"])
    test_groups = set(test["group_id"])
    assert not train_groups & test_groups


def test_external_taxonomy_mapping_finds_cross_group_rows() -> None:
    assert _labels_for(["en:vitamins"], "food") == {"food", "medicine"}
    assert _labels_for(["en:dietary-supplements"], "food") == {"food"}
    assert _labels_for(["en:protein-bars", "en:dietary-supplements"], "food") == {"food"}
    assert _labels_for(["en:cereal-bars", "en:vitamins"], "food") == {"food"}
    assert _labels_for(["en:vinegars"], "food") == {"food", "household"}
    assert _labels_for(["en:soaps"], "beauty") == {"personal_care", "household"}
    assert _labels_for(["en:sandwiches"], "food") == {"food", "cafe"}
    assert _labels_for(["en:sandwich-biscuits"], "food") == {"food"}
    assert _labels_for(["en:toothpastes"], "food") == {"personal_care"}


def test_build_merges_counterexamples_and_groups(tmp_path: Path) -> None:
    base = tmp_path / "base.csv"
    curated = tmp_path / "curated.csv"
    hard = tmp_path / "hard.csv"
    output = tmp_path / "out.csv"
    for path, rows in (
        (base, [("молоко", "food"), ("шампунь", "personal_care")]),
        (curated, [("сода", "food|household")]),
        (hard, [("филе куриное охлажденное", "food")]),
    ):
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["text", "category"])
            writer.writerows(rows)

    stats = build(
        base,
        curated,
        output,
        hard_negatives=hard,
        pairs_per_pair=1,
        triples=0,
        noise_fraction=0.0,
    )

    rows = list(csv.DictReader(output.open(encoding="utf-8")))
    assert stats["multilabel_rows"]
    assert all(row["group_id"] for row in rows)
    assert {"text", "category", "source", "group_id"} == set(rows[0])


def test_build_adds_bpe_noise_variants(tmp_path: Path) -> None:
    base = tmp_path / "base.csv"
    curated = tmp_path / "curated.csv"
    output = tmp_path / "out.csv"
    with base.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["text", "category"])
        writer.writerows(
            [
                ("молоко 3,2%", "food"),
                ("шампунь для волос", "personal_care"),
                ("стиральный порошок", "household"),
            ]
        )
    with curated.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["text", "category"])
        writer.writerow(["сода", "food|household"])

    stats = build(
        base,
        curated,
        output,
        pairs_per_pair=0,
        triples=0,
        noise_fraction=1.0,
        noise_variants=2,
        use_bpe=True,
        bpe_vocab_size=100,
    )

    assert stats["noisy_rows"]
    assert stats["noise_variants"] == 2
    assert stats["bpe"] is True
