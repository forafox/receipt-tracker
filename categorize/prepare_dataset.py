"""Build the final categorization dataset.

The Wildberries/Open Food Facts dump is single-label.  This script merges:

* a curated file of genuinely ambiguous receipt names (``сода`` is food and a
  household cleaner);
* cross-group rows extracted from Open Food Facts taxonomy tags;
* manually curated counterexamples from the error analysis;
* deterministic multi-item lines composed from real single-label names;
* OCR-noise variants of a fraction of all rows.

Every row gets a ``group_id`` so clean, short, composed and noisy variants of
the same product never leak between train and test.

Usage:
    python -m categorize.prepare_dataset
    python -m categorize.prepare_dataset --base a.csv --output b.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from categorize.augment import add_noise_variants, assign_groups, compose_multilabel
from categorize.bpe import train_bpe
from categorize.dataset import load_training_examples
from categorize.schemas import TrainingExample

LABELS = ("cafe", "food", "household", "medicine", "personal_care")

DEFAULT_BASE = Path("categorize/data/training_wb.csv")
DEFAULT_MULTILABEL = Path("categorize/data/multilabel_curated.csv")
DEFAULT_EXTERNAL = Path("categorize/data/multilabel_external.csv")
DEFAULT_HARD_NEGATIVES = Path("categorize/data/hard_negatives.csv")
DEFAULT_OUTPUT = Path("categorize/data/training_final.csv")

DEFAULT_PAIRS_PER_PAIR = 500
DEFAULT_TRIPLES = 1000
DEFAULT_NOISE_FRACTION = 0.25
DEFAULT_NOISE_VARIANTS = 2
DEFAULT_BPE_VOCAB_SIZE = 2000
DEFAULT_BPE_FRACTION = 0.35


def normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


def canonical(category: str) -> str:
    parts = {part.strip() for part in category.split("|") if part.strip()}
    unknown = parts - set(LABELS)
    if unknown:
        raise ValueError(f"unknown categories: {sorted(unknown)}")
    return "|".join(label for label in LABELS if label in parts)


def _read_examples(path: Path) -> list[TrainingExample]:
    return load_training_examples(path)


def _source_priority(source: str | None) -> int:
    if source is None:
        return 0
    if source.startswith("hard_negative"):
        return 3
    if source.startswith("openfoodfacts"):
        return 2
    if source.startswith("curated"):
        return 4
    return 1


def _merge_overrides(
    base: list[TrainingExample],
    overrides: list[TrainingExample],
) -> list[TrainingExample]:
    by_key: dict[str, TrainingExample] = {}
    for example in overrides:
        key = normalize(example.text)
        current = by_key.get(key)
        if current is None or _source_priority(example.source) >= _source_priority(current.source):
            by_key[key] = example

    merged: list[TrainingExample] = []
    seen: set[str] = set()
    for example in base:
        key = normalize(example.text)
        if key in by_key:
            merged.append(by_key[key])
            seen.add(key)
        else:
            merged.append(example)
    for key, example in by_key.items():
        if key not in seen:
            merged.append(example)
    return merged


def _dedupe_rows(examples: list[TrainingExample]) -> list[TrainingExample]:
    best: dict[str, TrainingExample] = {}
    for example in examples:
        key = normalize(example.text)
        current = best.get(key)
        if current is None:
            best[key] = example
            continue
        current_priority = _source_priority(current.source)
        new_priority = _source_priority(example.source)
        current_labels = len(current.category.split("|"))
        new_labels = len(example.category.split("|"))
        if (new_priority, new_labels) > (current_priority, current_labels):
            best[key] = example
    return list(best.values())


def build(
    base: Path,
    multilabel: Path,
    output: Path,
    *,
    external: Path | None = None,
    hard_negatives: Path | None = None,
    pairs_per_pair: int = 0,
    triples: int = 0,
    noise_fraction: float = 0.0,
    seed: int = 42,
    noise_variants: int = DEFAULT_NOISE_VARIANTS,
    use_bpe: bool = True,
    bpe_vocab_size: int = DEFAULT_BPE_VOCAB_SIZE,
    bpe_fraction: float = DEFAULT_BPE_FRACTION,
) -> dict[str, object]:
    """Merge all dataset sources and write ``text,category,source,group_id``.

    Args:
        base: Single-label Russian product names.
        multilabel: Curated ambiguous names.
        output: Destination CSV.
        external: Rows extracted from Open Food Facts, if available.
        hard_negatives: Counterexamples collected from error analysis.
        pairs_per_pair: Synthetic two-item rows per label pair.
        triples: Synthetic three-item rows.
        noise_fraction: Fraction of rows that get OCR-corrupted copies.
        seed: Deterministic augmentation seed.
        noise_variants: Number of corrupted copies per selected row.
        use_bpe: Train a BPE tokenizer and use BPE-dropout for some variants.
        bpe_vocab_size: Vocabulary size of the augmentation tokenizer.
        bpe_fraction: Share of noise variants that use BPE-dropout.

    Returns:
        Dataset statistics.
    """
    base_examples = _read_examples(base)
    curated = [
        example.model_copy(update={"source": example.source or "curated"})
        for example in _read_examples(multilabel)
    ]
    overrides: list[TrainingExample] = []
    if external is not None and external.exists():
        overrides.extend(_read_examples(external))
    if hard_negatives is not None and hard_negatives.exists():
        overrides.extend(
            example.model_copy(update={"source": example.source or "hard_negative"})
            for example in _read_examples(hard_negatives)
        )
    overrides.extend(curated)

    merged = _merge_overrides(base_examples, overrides)
    merged = assign_groups(merged)

    synthetic: list[TrainingExample] = []
    if pairs_per_pair or triples:
        synthetic = compose_multilabel(
            merged,
            pairs_per_pair=pairs_per_pair,
            triples=triples,
            seed=seed,
        )

    rows = merged + synthetic
    bpe_tokenizer = (
        train_bpe((row.text for row in rows), vocab_size=bpe_vocab_size)
        if use_bpe and noise_fraction > 0
        else None
    )
    noisy = add_noise_variants(
        rows,
        fraction=noise_fraction,
        seed=seed,
        variants=noise_variants,
        bpe_tokenizer=bpe_tokenizer,
        bpe_fraction=bpe_fraction,
    )
    rows = _dedupe_rows(rows + noisy)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["text", "category", "source", "group_id"])
        writer.writeheader()
        for example in rows:
            writer.writerow(
                {
                    "text": example.text,
                    "category": canonical(example.category),
                    "source": example.source or "base",
                    "group_id": example.group_id or normalize(example.text),
                }
            )

    label_counts: Counter[str] = Counter()
    pair_counts: Counter[str] = Counter()
    multi = 0
    for example in rows:
        parts = example.category.split("|")
        label_counts.update(parts)
        if len(parts) > 1:
            multi += 1
            pair_counts["|".join(parts)] += 1

    stats: dict[str, object] = {
        "rows": len(rows),
        "multilabel_rows": multi,
        "synthetic_rows": len(synthetic),
        "noisy_rows": len(noisy),
        "noise_variants": noise_variants,
        "bpe": bpe_tokenizer is not None,
        "duplicate_texts_removed": len(merged) + len(synthetic) + len(noisy) - len(rows),
        "label_counts": dict(label_counts),
        "labelsets": dict(pair_counts.most_common(20)),
        "source_counts": dict(Counter(example.source or "base" for example in rows)),
        "missing_labels": sorted(set(LABELS) - set(label_counts)),
    }
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--multilabel", type=Path, default=DEFAULT_MULTILABEL)
    parser.add_argument("--external", type=Path, default=DEFAULT_EXTERNAL)
    parser.add_argument("--hard-negatives", type=Path, default=DEFAULT_HARD_NEGATIVES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pairs-per-pair", type=int, default=DEFAULT_PAIRS_PER_PAIR)
    parser.add_argument("--triples", type=int, default=DEFAULT_TRIPLES)
    parser.add_argument("--noise-fraction", type=float, default=DEFAULT_NOISE_FRACTION)
    parser.add_argument("--noise-variants", type=int, default=DEFAULT_NOISE_VARIANTS)
    parser.add_argument("--no-bpe", action="store_true")
    parser.add_argument("--bpe-vocab-size", type=int, default=DEFAULT_BPE_VOCAB_SIZE)
    parser.add_argument("--bpe-fraction", type=float, default=DEFAULT_BPE_FRACTION)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    stats = build(
        args.base,
        args.multilabel,
        args.output,
        external=args.external,
        hard_negatives=args.hard_negatives,
        pairs_per_pair=args.pairs_per_pair,
        triples=args.triples,
        noise_fraction=args.noise_fraction,
        seed=args.seed,
        noise_variants=args.noise_variants,
        use_bpe=not args.no_bpe,
        bpe_vocab_size=args.bpe_vocab_size,
        bpe_fraction=args.bpe_fraction,
    )
    print(f"wrote {args.output}")
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
