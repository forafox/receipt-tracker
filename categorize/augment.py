"""Build realistic multilabel rows and OCR-noise variants from single-label data.

Genuine multilabel supervision for the five M3 categories is almost absent in
public Russian datasets.  Two reproducible sources close that gap:

* composition of real single-label item names into a multi-item receipt line
  (``молоко, шампунь``), which keeps every label real and mirrors what M2 can
  produce when OCR merges two lines;
* Open Food Facts rows whose own taxonomy tags span several M3 groups.

Composed rows carry a ``group_id`` so the same source name cannot leak between
train and test through a clean and corrupted variant.
"""

from __future__ import annotations

import random
from collections import defaultdict
from hashlib import sha1
from itertools import combinations

from tokenizers import Tokenizer

from categorize.bpe import bpe_dropout
from categorize.schemas import TrainingExample
from categorize.text import add_noise, shorten

LABELS = ("cafe", "food", "household", "medicine", "personal_care")
SEPARATORS = (", ", " + ", " и ", "; ", " / ", "\n")


def _group(prefix: str, value: str) -> str:
    return f"{prefix}:{sha1(value.encode('utf-8')).hexdigest()[:16]}"


def _labels(example: TrainingExample) -> set[str]:
    return {label.strip() for label in example.category.split("|") if label.strip()}


def _single_label_bank(examples: list[TrainingExample]) -> dict[str, list[str]]:
    bank: dict[str, list[str]] = defaultdict(list)
    for example in examples:
        labels = _labels(example)
        if len(labels) == 1:
            bank[next(iter(labels))].append(shorten(example.text))
    for values in bank.values():
        values.sort()
    return bank


def compose_multilabel(
    examples: list[TrainingExample],
    pairs_per_pair: int = 500,
    triples: int = 1000,
    seed: int = 42,
) -> list[TrainingExample]:
    """Compose multi-item receipt lines from real single-label names.

    Args:
        examples: Single- and multi-label seed rows.
        pairs_per_pair: Number of two-item rows for each of the ten label pairs.
        triples: Number of three-item rows sampled across distinct labels.
        seed: Deterministic seed for sampling and separator choice.

    Returns:
        Synthetic rows labelled with the union of their parts.
    """
    rng = random.Random(seed)
    bank = _single_label_bank(examples)
    available = [label for label in LABELS if bank.get(label)]
    rows: list[TrainingExample] = []

    for left, right in combinations(available, 2):
        for _ in range(pairs_per_pair):
            first = rng.choice(bank[left])
            second = rng.choice(bank[right])
            separator = rng.choice(SEPARATORS)
            text = f"{first}{separator}{second}"
            rows.append(
                TrainingExample(
                    text=text,
                    category="|".join(sorted((left, right))),
                    source="synthetic_pair",
                    group_id=_group("combo", "\x1f".join(sorted((first, second)))),
                )
            )

    for _ in range(triples):
        labels = rng.sample(available, 3)
        parts = [rng.choice(bank[label]) for label in labels]
        rng.shuffle(parts)
        text = rng.choice(SEPARATORS).join(parts)
        rows.append(
            TrainingExample(
                text=text,
                category="|".join(sorted(labels)),
                source="synthetic_triple",
                group_id=_group("combo", "\x1f".join(sorted(parts))),
            )
        )
    return rows


def add_noise_variants(
    examples: list[TrainingExample],
    fraction: float = 0.25,
    level: float = 0.4,
    seed: int = 42,
    *,
    variants: int = 1,
    profiles: tuple[str, ...] = ("ocr", "layout", "receipt", "words", "severe"),
    bpe_tokenizer: Tokenizer | None = None,
    bpe_fraction: float = 0.35,
    bpe_drop_prob: float = 0.15,
) -> list[TrainingExample]:
    """Add deterministic OCR-corrupted copies for a fraction of the rows.

    The copy keeps the original label and ``group_id`` so it always lands in
    the same train/test split as its clean counterpart.  Several noise
    profiles are mixed because the test errors are dominated by combined
    character corruption, lost word boundaries, truncation and receipt
    artifacts.  When a BPE tokenizer is supplied, a share of the variants is
    additionally passed through BPE-dropout to imitate subword loss.
    """
    rng = random.Random(seed)
    profiles = profiles or ("mixed",)
    variants_list: list[TrainingExample] = []
    for example in examples:
        if rng.random() >= fraction:
            continue
        for _ in range(variants):
            profile = rng.choice(profiles)
            if profile == "words" and len(example.text.split()) < 3:
                profile = "ocr"
            if bpe_tokenizer is not None and rng.random() < bpe_fraction:
                corrupted = bpe_dropout(
                    example.text,
                    bpe_tokenizer,
                    rng,
                    drop_prob=bpe_drop_prob,
                )
                if rng.random() < 0.5:
                    corrupted = add_noise(
                        corrupted,
                        rng,
                        level=min(level, 0.3),
                        profile="ocr",
                    )
            else:
                corrupted = add_noise(example.text, rng, level=level, profile=profile)
            if corrupted == example.text:
                continue
            variants_list.append(
                TrainingExample(
                    text=corrupted,
                    category=example.category,
                    source=f"{example.source or 'base'}+noise",
                    group_id=example.group_id or _group("text", example.text.casefold().strip()),
                )
            )
    return variants_list


def assign_groups(examples: list[TrainingExample]) -> list[TrainingExample]:
    """Attach a stable group id to every row that does not have one."""
    result: list[TrainingExample] = []
    for example in examples:
        group_id = example.group_id or _group("text", example.text.casefold().strip())
        result.append(example.model_copy(update={"group_id": group_id}))
    return result
