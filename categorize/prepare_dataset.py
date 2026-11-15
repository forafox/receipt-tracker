"""Build the final categorization dataset.

The Wildberries/Open Food Facts dump is single-label.  A small curated file adds
*genuine* multi-label items: short receipt names whose meaning depends on
context (e.g. "сода" is both a food product and a household cleaner).  This
script merges that file, removes duplicate texts and reports the resulting
distribution so training never silently consumes a dirty dataset.

Usage:
    python -m categorize.prepare_dataset
    python -m categorize.prepare_dataset --base a.csv --output b.csv
"""

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

LABELS = ("food", "household", "personal_care", "medicine", "cafe")

DEFAULT_BASE = Path("categorize/data/training_wb.csv")
DEFAULT_MULTILABEL = Path("categorize/data/multilabel_curated.csv")
DEFAULT_OUTPUT = Path("categorize/data/training_final.csv")


def normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


def canonical(category: str) -> str:
    parts = {part.strip() for part in category.split("|") if part.strip()}
    unknown = parts - set(LABELS)
    if unknown:
        raise ValueError(f"unknown categories: {sorted(unknown)}")
    return "|".join(label for label in LABELS if label in parts)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        return [row for row in csv.DictReader(file) if row.get("text") and row.get("category")]


def build(base: Path, multilabel: Path, output: Path) -> dict[str, object]:
    curated = {normalize(row["text"]): canonical(row["category"]) for row in _read_rows(multilabel)}

    base_rows = _read_rows(base)
    buckets: defaultdict[str, Counter[str]] = defaultdict(Counter)
    display: dict[str, str] = {}
    for row in base_rows:
        key = normalize(row["text"])
        display.setdefault(key, row["text"].strip())
        buckets[key][curated.get(key, canonical(row["category"]))] += 1
    base_unique = len(buckets)

    for key in curated:
        buckets[key]
        display.setdefault(key, key)

    rows: list[tuple[str, str]] = []
    conflicts = 0
    for key in sorted(buckets):
        counters = buckets[key]
        if key in curated:
            label = curated[key]
        else:
            label = counters.most_common(1)[0][0]
            conflicts += len(counters) - 1
        rows.append((display[key], label))

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["text", "category"])
        writer.writerows(rows)

    label_counts: Counter[str] = Counter()
    multi = 0
    for _, label in rows:
        parts = label.split("|")
        label_counts.update(parts)
        if len(parts) > 1:
            multi += 1

    stats: dict[str, object] = {
        "rows": len(rows),
        "multilabel_rows": multi,
        "duplicate_texts_removed": len(base_rows) - base_unique,
        "label_conflicts_resolved": conflicts,
        "label_counts": dict(label_counts),
        "missing_labels": sorted(set(LABELS) - set(label_counts)),
    }
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--multilabel", type=Path, default=DEFAULT_MULTILABEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    stats = build(args.base, args.multilabel, args.output)
    print(f"wrote {args.output}")
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
