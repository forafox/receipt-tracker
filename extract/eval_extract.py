"""Evaluate receipt extraction on independently annotated OCR/reference JSONL."""

import argparse
from collections.abc import Sequence
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from extract.evaluation import evaluate_ocr_results
from extract.samples import load_evaluation_samples


def main(argv: Sequence[str] | None = None) -> None:
    """Load annotations, evaluate extraction and print metrics and mismatching IDs.

    Args:
        argv: Command-line arguments, or None to use the process arguments.

    Raises:
        SystemExit: With code 2 for invalid arguments, input or an empty dataset;
            with code 0 for help.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to annotated samples in UTF-8 JSONL.")
    parser.add_argument("--timezone", required=True, help="Receipt timezone, e.g. Europe/Moscow.")
    args = parser.parse_args(argv)

    try:
        timezone = ZoneInfo(args.timezone)
    except (ZoneInfoNotFoundError, ValueError) as error:
        parser.error(f"invalid timezone {args.timezone!r}: {error}")

    try:
        samples = load_evaluation_samples(args.input)
    except (OSError, ValueError) as error:
        parser.error(f"cannot load {args.input}: {error}")

    try:
        result = evaluate_ocr_results(
            ((sample.ocr, sample.expected) for sample in samples), timezone=timezone
        )
    except ValueError as error:
        parser.error(f"cannot evaluate {args.input}: {error}")

    print(f"Receipts: {result.receipt_count}")
    for name, metric in (
        ("store", result.store),
        ("datetime", result.datetime),
        ("total", result.total),
    ):
        accuracy = metric.accuracy
        score = "N/A" if accuracy is None else f"{accuracy:.4f}"
        print(f"{name} accuracy: {score} ({metric.matches}/{metric.reference_count})")
    print(f"item precision: {result.items.precision:.4f}")
    print(f"item recall: {result.items.recall:.4f}")
    print(f"item F1: {result.items.f1:.4f}")
    print(f"Mismatches: {len(result.mismatches)}")
    for mismatch in result.mismatches:
        sample_id = samples[mismatch.index].id
        fields = ", ".join(mismatch.fields) or "none"
        print(
            f"  {sample_id!r}: fields={fields}; "
            f"missing_items={mismatch.missing_items}; extra_items={mismatch.extra_items}"
        )

    if not samples:
        parser.error(f"empty dataset: {args.input}")


if __name__ == "__main__":
    main()
