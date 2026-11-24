"""Audit text/category CSV for genuine multi-label training and data leakage.

Run: python -m categorize.data_audit categorize/data/training_wb.csv
Does not require PyTorch or transformers.
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

EXPECTED = {"food", "household", "personal_care", "medicine", "cafe"}


def audit(path: Path) -> dict[str, object]:
    categories: Counter[str] = Counter()
    labelsets: Counter[str] = Counter()
    by_text: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    invalid: list[int] = []
    count = 0
    multi_count = 0
    with path.open(encoding="utf-8-sig", newline="") as infile:
        reader = csv.DictReader(infile)
        if not {"text", "category"}.issubset(reader.fieldnames or []):
            raise ValueError("CSV must contain text,category columns")
        for lineno, row in enumerate(reader, 2):
            count += 1
            labels: tuple[str, ...] = tuple(
                sorted({x.strip() for x in row["category"].split("|") if x.strip()})
            )
            text = row["text"].strip().casefold()
            if not text or not labels or not set(labels).issubset(EXPECTED):
                invalid.append(lineno)
                continue
            if len(labels) > 1:
                multi_count += 1
            categories.update(labels)
            labelsets["|".join(labels)] += 1
            by_text[text].add(labels)
    return {
        "rows": count,
        "genuine_multilabel_rows": multi_count,
        "genuine_multilabel_fraction": round(multi_count / count, 4) if count else 0,
        "label_counts": dict(categories),
        "labelset_counts": dict(labelsets),
        "invalid_line_numbers_first_30": invalid[:30],
        "same_text_multiple_labelsets": sum(len(v) > 1 for v in by_text.values()),
        "duplicate_text_rows": count - len(by_text),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.csv), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
