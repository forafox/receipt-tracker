from __future__ import annotations

import random

from categorize.bpe import bpe_dropout, train_bpe
from categorize.text import add_noise

TEXTS = [
    "молоко 3,2%",
    "шампунь для волос",
    "туалетная бумага 3-сл",
    "кофе в зернах",
    "стиральный порошок",
]


def test_bpe_dropout_is_deterministic_and_non_empty() -> None:
    tokenizer = train_bpe(TEXTS, vocab_size=300)
    first = bpe_dropout("стиральный порошок автомат", tokenizer, random.Random(7), 0.2)
    second = bpe_dropout("стиральный порошок автомат", tokenizer, random.Random(7), 0.2)
    assert first == second
    assert first
    assert "\ufffd" not in first


def test_noise_profiles_are_deterministic_and_non_empty() -> None:
    text = "стиральный порошок автомат"
    for profile in ("ocr", "layout", "receipt", "words", "severe"):
        first = add_noise(text, random.Random(7), profile=profile)
        second = add_noise(text, random.Random(7), profile=profile)
        assert first == second, profile
        assert first, profile
