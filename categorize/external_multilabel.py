"""Extract genuine multilabel rows for the five M3 categories from Open Food Facts.

Open Food Facts (and its beauty split) is one of the few public sources that
contains Russian product names together with a hierarchical taxonomy.  A
product is kept only when its own tags span more than one M3 group, for
example a dietary supplement (``food|medicine``), vinegar
(``food|household``) or a cleansing wipe (``household|personal_care``).

The raw parquet files are large (about 8 GB for food) and are not committed.
This module turns them into a small CSV that is safe to keep in git.

Usage:
    python -m categorize.external_multilabel \
        --food /data/food.parquet --beauty /data/beauty.parquet \
        --output categorize/data/multilabel_external.csv
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from categorize.text import CYRILLIC_RE, denoise, shorten

LABELS = ("cafe", "food", "household", "medicine", "personal_care")
DEFAULT_OUTPUT = Path("categorize/data/multilabel_external.csv")
PER_LABELSET_LIMIT = 400

TAG_PATTERNS: dict[str, tuple[str, ...]] = {
    "medicine": (
        "vitamin",
        "mineral-supplement",
        "medicines",
        "medical",
        "antiseptic",
        "first-aid",
        "pharmacy",
    ),
    "household": (
        "cleaning",
        "detergent",
        "laundry",
        "dishwash",
        "surface-cleaner",
        "household",
        "bleach",
        "disinfectant",
        "vinegar",
        "baking-soda",
        "citric-acid",
        "soap",
    ),
    "personal_care": (
        "hygiene",
        "cosmetic",
        "body-care",
        "hair-care",
        "shampoo",
        "soap",
        "deodorant",
        "oral-hygiene",
        "toothpaste",
        "shaving",
        "skin-care",
        "sunscreen",
        "wipes",
        "cotton",
        "beauty",
    ),
    "cafe": (),
}

CAFE_TAGS = {
    "cafe",
    "prepared-meals",
    "ready-to-eat",
    "restaurant",
    "fast-food",
    "sandwiches",
    "pizzas",
    "hamburgers",
    "burgers",
    "sushi",
    "coffee-drinks",
    "tea-drinks",
}

FOOD_PATTERNS = (
    "plant-based-foods",
    "foods",
    "beverages",
    "snacks",
    "meals",
    "condiments",
    "vinegars",
    "baking-soda",
    "citric-acid",
    "dairy",
    "meat",
    "seafood",
    "cereals",
    "fruits",
    "vegetables",
    "legumes",
    "bakery",
    "confectioneries",
    "sweet-snacks",
    "salty-snacks",
    "protein-bars",
    "dietary-supplements",
    "cereal-bars",
    "chocolate-bars",
    "prepared-salads",
    "pizzas",
    "sandwiches",
    "biscuits",
    "chocolates",
    "coffee",
    "teas",
    "spices",
    "sauces",
    "soups",
    "cheeses",
    "yogurts",
    "milks",
    "waters",
    "juices",
    "breakfasts",
    "spreads",
)


def _labels_for(tags: list[str], split: str) -> set[str]:
    names = {tag.split(":", 1)[1].lower() for tag in tags if ":" in tag}
    labels: set[str] = set()
    is_supplement_bar = any("bars" in name or "snacks" in name for name in names)
    for label, patterns in TAG_PATTERNS.items():
        if label == "cafe":
            if names & CAFE_TAGS:
                labels.add(label)
        elif label == "medicine" and is_supplement_bar:
            continue
        elif any(pattern in name for name in names for pattern in patterns):
            labels.add(label)

    food_like = any(pattern in name for name in names for pattern in FOOD_PATTERNS)
    if split == "food":
        if "medicine" in labels or food_like:
            labels.add("food")
    elif not labels & {"food", "household", "medicine"}:
        labels.add("personal_care")
    return labels


def _russian_name(product_name: Any) -> str | None:
    if not isinstance(product_name, list):
        return None
    by_language: dict[str, list[str]] = defaultdict(list)
    for item in product_name:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        language = item.get("lang")
        if isinstance(text, str) and text.strip() and isinstance(language, str):
            by_language[language].append(text.strip())
    for language in ("ru", "main"):
        for text in by_language.get(language, []):
            if CYRILLIC_RE.search(text):
                return text
    for values in by_language.values():
        for text in values:
            if CYRILLIC_RE.search(text):
                return text
    return None


def _rows_from_parquet(path: Path, split: str) -> list[dict[str, str]]:
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    columns = ["product_name", "categories_tags", "categories", "code"]
    by_text: dict[str, set[str]] = {}
    codes: dict[str, str] = {}
    for group in range(parquet.metadata.num_row_groups):
        table = parquet.read_row_group(group, columns=columns)
        for row in table.to_pylist():
            name = _russian_name(row.get("product_name"))
            if name is None:
                continue
            text = denoise(name)
            if len(text) < 3:
                continue
            tags = [tag for tag in (row.get("categories_tags") or []) if isinstance(tag, str)]
            labels = _labels_for(tags, split)
            if len(labels) < 2:
                continue
            by_text.setdefault(text, set()).update(labels)
            if row.get("code"):
                codes.setdefault(text, str(row["code"]))
    rows: list[dict[str, str]] = []
    for text, labels in by_text.items():
        rows.append(
            {
                "text": text,
                "category": "|".join(label for label in LABELS if label in labels),
                "source": f"openfoodfacts:{split}",
                "group_id": f"off:{codes.get(text) or text}",
            }
        )
    return rows


def _cap_by_labelset(
    rows: list[dict[str, str]], limit: int, seed: int = 42
) -> list[dict[str, str]]:
    grouped: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["category"]].append(row)
    rng = random.Random(seed)
    result: list[dict[str, str]] = []
    for values in (item[1] for item in sorted(grouped.items())):
        values.sort(key=lambda row: row["text"])
        if len(values) > limit:
            values = rng.sample(values, limit)
        result.extend(values)
    return result


def _add_short_forms(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for row in rows:
        result.append(row)
        short = shorten(row["text"], max_words=4)
        if short and short != row["text"]:
            result.append({**row, "text": short, "source": f"{row['source']}:short"})
    return result


def extract(food: Path | None, beauty: Path | None, output: Path) -> dict[str, object]:
    """Extract, cap and write multilabel CSV rows.

    Args:
        food: Open Food Facts food parquet, or None.
        beauty: Open Food Facts beauty parquet, or None.
        output: Destination CSV.

    Returns:
        Statistics about the generated file.
    """
    rows: list[dict[str, str]] = []
    if food is not None:
        rows.extend(_rows_from_parquet(food, "food"))
    if beauty is not None:
        rows.extend(_rows_from_parquet(beauty, "beauty"))
    rows = _cap_by_labelset(rows, PER_LABELSET_LIMIT)
    rows = _add_short_forms(rows)
    rows.sort(key=lambda row: (row["category"], row["text"]))

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["text", "category", "source", "group_id"])
        writer.writeheader()
        writer.writerows(rows)

    counts: Counter[str] = Counter(row["category"] for row in rows)
    return {
        "rows": len(rows),
        "labelsets": dict(counts),
        "sources": dict(Counter(row["source"] for row in rows)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--food", type=Path, default=None)
    parser.add_argument("--beauty", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.food is None and args.beauty is None:
        parser.error("at least one of --food/--beauty is required")
    stats = extract(args.food, args.beauty, args.output)
    print(f"wrote {args.output}")
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
