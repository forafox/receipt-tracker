from __future__ import annotations

from collections.abc import Sequence


def levenshtein(source: Sequence[object], target: Sequence[object]) -> int:
    """Compute the Levenshtein edit distance between two sequences.

    Args:
        source: Sequence compared from.
        target: Sequence compared to.

    Returns:
        Minimum number of insertions, deletions and substitutions.
    """
    if not source:
        return len(target)
    if not target:
        return len(source)

    previous = list(range(len(target) + 1))
    for source_index, source_item in enumerate(source, start=1):
        current = [source_index]
        for target_index, target_item in enumerate(target, start=1):
            cost = 0 if source_item == target_item else 1
            current.append(
                min(
                    previous[target_index] + 1,
                    current[target_index - 1] + 1,
                    previous[target_index - 1] + cost,
                )
            )
        previous = current
    return previous[-1]


def character_error_rate(reference: str, hypothesis: str) -> float:
    """Compute the character error rate between two strings.

    Args:
        reference: Ground-truth text.
        hypothesis: Recognized text.

    Returns:
        Edit distance divided by reference length, or ``0.0`` when both are empty and ``1.0``
        when only the reference is empty.
    """
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return levenshtein(reference, hypothesis) / len(reference)


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Compute the word error rate between two strings.

    Args:
        reference: Ground-truth text.
        hypothesis: Recognized text.

    Returns:
        Word-level edit distance divided by reference word count, or ``0.0`` when both are
        empty and ``1.0`` when only the reference is empty.
    """
    reference_words = reference.split()
    if not reference_words:
        return 0.0 if not hypothesis.split() else 1.0
    return levenshtein(reference_words, hypothesis.split()) / len(reference_words)
