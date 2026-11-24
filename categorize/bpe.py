"""Byte-level BPE tokenizer and BPE-dropout augmentation.

The production classifier is a WordPiece model (``cointegrated/rubert-tiny``),
so BPE is not used at inference.  This module trains a small byte-level BPE
tokenizer on the current dataset and uses it only as a *noise generator*:
dropping random BPE tokens imitates OCR that loses or merges subword fragments.
"""

from __future__ import annotations

import random
from collections.abc import Iterable

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

SPECIAL_TOKENS = ["[UNK]", "[PAD]", "[BOS]", "[EOS]"]


def train_bpe(texts: Iterable[str], vocab_size: int = 2000) -> Tokenizer:
    """Train a deterministic byte-level BPE tokenizer on product names.

    Args:
        texts: Clean product names used to learn subword units.
        vocab_size: BPE vocabulary size.  Small vocabularies produce more
            aggressive fragmentation, which is useful for OCR noise.

    Returns:
        A trained ``tokenizers.Tokenizer``.
    """
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=1,
        special_tokens=SPECIAL_TOKENS,
        show_progress=False,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    return tokenizer


def bpe_dropout(
    text: str,
    tokenizer: Tokenizer,
    rng: random.Random,
    drop_prob: float = 0.15,
) -> str:
    """Drop random BPE tokens to imitate subword-level OCR loss.

    Args:
        text: Clean item name.
        tokenizer: Tokenizer returned by :func:`train_bpe`.
        rng: Deterministic random generator.
        drop_prob: Probability of dropping each BPE token.

    Returns:
        A decoded string.  Falls back to ``text`` when dropping would produce
        invalid UTF-8 or an empty string.
    """
    ids = tokenizer.encode(text).ids
    if not ids:
        return text
    kept = [token_id for token_id in ids if rng.random() >= drop_prob]
    if not kept:
        return text
    value = tokenizer.decode(kept, skip_special_tokens=True).strip()
    if not value or "\ufffd" in value:
        return text
    return value


__all__ = ["bpe_dropout", "train_bpe"]
