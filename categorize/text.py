"""Text normalization and OCR-like augmentation for receipt item names.

Receipt lines arrive from OCR, so the same product can be written as
``Молоко 3,2%``, ``HДC 20% Mолоко 3,2%`` or ``MOnOKO 3.2%``.  Training only on
clean marketplace titles leaves the classifier fragile.  This module provides
two deterministic operations:

* :func:`denoise` normalizes an item name before inference: homoglyphs, VAT
  prefixes, trailing quantities and duplicated whitespace are removed.
* :func:`add_noise` imitates the most common OCR corruptions so the training
  set can contain realistic variants of every product.
"""

from __future__ import annotations

import random
import re
import unicodedata

LATIN_TO_CYRILLIC = str.maketrans(
    {
        "a": "а",
        "b": "в",
        "c": "с",
        "e": "е",
        "h": "н",
        "k": "к",
        "m": "м",
        "o": "о",
        "p": "р",
        "t": "т",
        "x": "х",
        "y": "у",
        "A": "А",
        "B": "В",
        "C": "С",
        "E": "Е",
        "H": "Н",
        "K": "К",
        "M": "М",
        "O": "О",
        "P": "Р",
        "T": "Т",
        "X": "Х",
        "Y": "У",
    }
)
CYRILLIC_TO_LATIN = str.maketrans(
    {
        "а": "a",
        "в": "b",
        "е": "e",
        "к": "k",
        "м": "m",
        "н": "h",
        "о": "o",
        "р": "p",
        "с": "c",
        "т": "t",
        "у": "y",
        "х": "x",
        "А": "A",
        "В": "B",
        "Е": "E",
        "К": "K",
        "М": "M",
        "Н": "H",
        "О": "O",
        "Р": "P",
        "С": "C",
        "Т": "T",
        "У": "Y",
        "Х": "X",
    }
)
CYRILLIC_LOOKALIKE_CHARS = set("авекмнорстухАВЕКМНОРСТУХ")
DIGIT_LOOKALIKE_CHARS = set("оОзЗчЧбБ")
DIGIT_LOOKALIKES = str.maketrans(
    {"о": "0", "О": "0", "з": "3", "З": "3", "ч": "4", "Ч": "4", "б": "6", "Б": "6"}
)
OCR_CONFUSIONS: dict[str, tuple[str, ...]] = {
    "н": ("п", "м", "и"),
    "п": ("н", "л"),
    "ш": ("щ", "и"),
    "щ": ("ш", "ц"),
    "и": ("й", "н"),
    "й": ("и",),
    "л": ("1", "п"),
    "1": ("л",),
    "о": ("0", "а"),
    "0": ("о",),
    "з": ("3", "в"),
    "3": ("з",),
    "ч": ("4",),
    "4": ("ч",),
    "б": ("6", "в"),
    "6": ("б",),
    "с": ("е", "о"),
    "е": ("с", "ё"),
    "а": ("о",),
    "в": ("ь", "б"),
    "ь": ("в", "ъ"),
    "к": ("х", "н"),
    "х": ("к", "ж"),
    "м": ("н", "л"),
    "т": ("г", "п"),
    "г": ("т", "р"),
    "р": ("г", "ф"),
    "у": ("ч",),
    "ц": ("щ", "ч"),
    "ы": ("и",),
}
CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")
VAT_PREFIX_RE = re.compile(
    r"^(?:(?:в\s*т\.?\s*ч\.?\s*)?ндс\s*\d+(?:[.,]\d+)?\s*%?\s*[:\-]?\s*)+",
    re.IGNORECASE,
)
VAT_INLINE_RE = re.compile(r"\(?\s*в\s*т\.?\s*ч\.?\s*ндс[^)]*\)?", re.IGNORECASE)
TRAILING_QUANTITY_RE = re.compile(
    r"(?:\s*[x×х]\s*\d+|\s*\d+(?:[.,]\d+)?\s*(?:шт|л|мл|кг|г|гр|уп|пач)\.?)\s*$",
    re.IGNORECASE,
)
TRAILING_PUNCTUATION_RE = re.compile(r"[\s*|,;:/\\-]+$")
LEADING_PUNCTUATION_RE = re.compile(r"^[\s*|,;:/\\-]+")
MULTISPACE_RE = re.compile(r"\s+")


def _translate_homoglyphs(text: str) -> str:
    """Translate Latin lookalikes only inside tokens that already contain Cyrillic."""
    words: list[str] = []
    for word in text.split():
        if CYRILLIC_RE.search(word) and re.search(r"[A-Za-z]", word):
            word = word.translate(LATIN_TO_CYRILLIC)
        words.append(word)
    return " ".join(words)


def denoise(text: str) -> str:
    """Normalize a receipt item name before classification.

    The function is intentionally conservative: it only removes artifacts that
    OCR engines reliably add (VAT prefixes, trailing quantities, mixed
    alphabets) and never drops meaningful words such as ``1%``.

    Args:
        text: Raw item name, possibly corrupted by OCR.

    Returns:
        Lowercased, space-normalized item name.
    """
    value = unicodedata.normalize("NFKC", text).replace("\u00a0", " ")
    value = value.replace("\n", " ").replace("\t", " ")
    value = VAT_PREFIX_RE.sub("", value)
    value = VAT_INLINE_RE.sub(" ", value)
    value = _translate_homoglyphs(value)
    value = value.strip().lower().replace("ё", "е")
    value = TRAILING_QUANTITY_RE.sub("", value)
    value = TRAILING_PUNCTUATION_RE.sub("", value)
    value = LEADING_PUNCTUATION_RE.sub("", value)
    value = MULTISPACE_RE.sub(" ", value)
    return value.strip()


def _drop_char(value: str, rng: random.Random) -> str:
    if len(value) <= 3:
        return value
    index = rng.randrange(len(value))
    return value[:index] + value[index + 1 :]


def _duplicate_char(value: str, rng: random.Random) -> str:
    index = rng.randrange(len(value))
    return value[:index] + value[index] + value[index:]


def _swap_chars(value: str, rng: random.Random) -> str:
    if len(value) <= 2:
        return value
    index = rng.randrange(len(value) - 1)
    return value[:index] + value[index + 1] + value[index] + value[index + 2 :]


def _remove_space(value: str, rng: random.Random) -> str:
    positions = [index for index, char in enumerate(value) if char == " "]
    if not positions:
        return value
    index = rng.choice(positions)
    return value[:index] + value[index + 1 :]


def _insert_space(value: str, rng: random.Random) -> str:
    if len(value) <= 2:
        return value
    index = rng.randrange(1, len(value) - 1)
    return value[:index] + " " + value[index:]


def _upper_word(value: str, rng: random.Random) -> str:
    words = value.split()
    if not words:
        return value
    index = rng.randrange(len(words))
    words[index] = words[index].upper()
    return " ".join(words)


def _truncate(value: str, rng: random.Random) -> str:
    if len(value) <= 12:
        return value
    cut = rng.randrange(max(4, len(value) // 2), len(value))
    return value[:cut].rstrip()


def _homoglyph(value: str, rng: random.Random) -> str:
    candidates = [index for index, char in enumerate(value) if char in CYRILLIC_LOOKALIKE_CHARS]
    if not candidates:
        return value
    index = rng.choice(candidates)
    return value[:index] + value[index].translate(CYRILLIC_TO_LATIN) + value[index + 1 :]


def _digit_lookalike(value: str, rng: random.Random) -> str:
    candidates = [index for index, char in enumerate(value) if char in DIGIT_LOOKALIKE_CHARS]
    if not candidates:
        return value
    index = rng.choice(candidates)
    return value[:index] + value[index].translate(DIGIT_LOOKALIKES) + value[index + 1 :]


def _vat_prefix(value: str, rng: random.Random) -> str:
    rate = rng.choice(("10", "20", "10,0", "20.0"))
    return f"НДС {rate}% {value}"


def _append_quantity(value: str, rng: random.Random) -> str:
    quantity = rng.choice(("1 шт", "2 шт", "x2", "0,5 л", "1 кг", "100 г"))
    return f"{value} {quantity}"


def _separator(value: str, rng: random.Random) -> str:
    if len(value) <= 2:
        return value
    index = rng.randrange(1, len(value) - 1)
    return value[:index] + rng.choice(("|", "*", "/")) + value[index:]


def _ocr_confusion(value: str, rng: random.Random) -> str:
    candidates = [index for index, char in enumerate(value) if char.lower() in OCR_CONFUSIONS]
    if not candidates:
        return value
    index = rng.choice(candidates)
    original = value[index]
    replacement = rng.choice(OCR_CONFUSIONS[original.lower()])
    if original.isupper():
        replacement = replacement.upper()
    return value[:index] + replacement + value[index + 1 :]


def _drop_word(value: str, rng: random.Random) -> str:
    words = value.split()
    if len(words) <= 1:
        return value
    index = rng.randrange(len(words))
    return " ".join(words[:index] + words[index + 1 :])


def _duplicate_word(value: str, rng: random.Random) -> str:
    words = value.split()
    if not words:
        return value
    index = rng.randrange(len(words))
    return " ".join(words[: index + 1] + [words[index]] + words[index + 1 :])


def _merge_words(value: str, rng: random.Random) -> str:
    words = value.split()
    if len(words) <= 1:
        return value
    index = rng.randrange(len(words) - 1)
    merged = words[:index] + [words[index] + words[index + 1]] + words[index + 2 :]
    return " ".join(merged)


def _split_word(value: str, rng: random.Random) -> str:
    words = value.split()
    candidates = [index for index, word in enumerate(words) if len(word) > 4]
    if not candidates:
        return value
    index = rng.choice(candidates)
    word = words[index]
    cut = rng.randrange(2, len(word))
    words[index : index + 1] = [word[:cut], word[cut:]]
    return " ".join(words)


def _truncate_word(value: str, rng: random.Random) -> str:
    words = value.split()
    candidates = [index for index, word in enumerate(words) if len(word) > 4]
    if not candidates:
        return value
    index = rng.choice(candidates)
    word = words[index]
    words[index] = word[: rng.randrange(2, len(word))]
    return " ".join(words)


def _swap_words(value: str, rng: random.Random) -> str:
    words = value.split()
    if len(words) <= 1:
        return value
    index = rng.randrange(len(words) - 1)
    words[index], words[index + 1] = words[index + 1], words[index]
    return " ".join(words)


def _alternate_case(value: str, rng: random.Random) -> str:
    words = value.split()
    if not words:
        return value
    index = rng.randrange(len(words))
    words[index] = "".join(
        char.upper() if position % 2 else char.lower() for position, char in enumerate(words[index])
    )
    return " ".join(words)


def _punctuation(value: str, rng: random.Random) -> str:
    candidates = [index for index, char in enumerate(value) if char in ".,;:-"]
    if candidates:
        index = rng.choice(candidates)
        return value[:index] + rng.choice(".,;:—_-|*") + value[index + 1 :]
    if len(value) <= 2:
        return value
    index = rng.randrange(1, len(value) - 1)
    return value[:index] + rng.choice(".,;:—_-|*") + value[index:]


_CHAR_OPERATIONS = (
    _drop_char,
    _duplicate_char,
    _swap_chars,
    _homoglyph,
    _digit_lookalike,
    _ocr_confusion,
)
_WORD_OPERATIONS = (
    _remove_space,
    _insert_space,
    _upper_word,
    _alternate_case,
    _drop_word,
    _duplicate_word,
    _merge_words,
    _split_word,
    _truncate_word,
    _swap_words,
)
_ARTIFACT_OPERATIONS = (
    _vat_prefix,
    _append_quantity,
    _separator,
    _punctuation,
    _truncate,
)
_NOISE_OPERATIONS = (
    _drop_char,
    _duplicate_char,
    _swap_chars,
    _remove_space,
    _insert_space,
    _upper_word,
    _truncate,
    _homoglyph,
    _digit_lookalike,
    _vat_prefix,
    _append_quantity,
    _separator,
)
_PROFILE_OPERATIONS = {
    "mixed": _NOISE_OPERATIONS,
    "ocr": _CHAR_OPERATIONS,
    "layout": (
        _remove_space,
        _insert_space,
        _merge_words,
        _split_word,
        _upper_word,
        _alternate_case,
        _swap_words,
    ),
    "receipt": _ARTIFACT_OPERATIONS,
    "words": (
        _drop_word,
        _duplicate_word,
        _truncate_word,
        _merge_words,
        _split_word,
        _swap_words,
    ),
}


def add_noise(
    text: str,
    rng: random.Random,
    level: float = 0.4,
    profile: str = "mixed",
) -> str:
    """Return an OCR-like corrupted variant of ``text``.

    Args:
        text: Clean item name.
        rng: Deterministic random generator.
        level: Fraction of the available corruption operations to apply.
        profile: ``mixed``, ``ocr``, ``layout``, ``receipt``, ``words`` or
            ``severe``.  Profiles target different error families found in
            the test errors: character confusion, lost word boundaries,
            receipt artifacts and truncation.

    Returns:
        A non-empty corrupted string.
    """
    if profile == "severe":
        value = text
        if rng.random() < 0.6:
            value = _vat_prefix(value, rng)
        if rng.random() < 0.6:
            value = _append_quantity(value, rng)
        for _ in range(rng.randint(2, 4)):
            value = rng.choice(_CHAR_OPERATIONS)(value, rng)
        value = rng.choice(_WORD_OPERATIONS)(value, rng)
        if rng.random() < 0.5:
            value = rng.choice(_ARTIFACT_OPERATIONS)(value, rng)
        value = MULTISPACE_RE.sub(" ", value).strip()
        return value or text
    if profile not in _PROFILE_OPERATIONS:
        raise ValueError(f"unknown noise profile: {profile}")
    operations = _PROFILE_OPERATIONS[profile]
    count = min(len(operations), max(1, round(level * len(operations))))
    for operation in rng.sample(operations, k=count):
        text = operation(text, rng)
    value = MULTISPACE_RE.sub(" ", text).strip()
    return value or text


def shorten(text: str, max_words: int = 5) -> str:
    """Keep the first ``max_words`` words of a product title."""
    words = denoise(text).split()
    return " ".join(words[:max_words])
