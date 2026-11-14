"""Build training CSV from Hugging Face dataset + seed examples.

PDF requirements:
- build_training_csv(output_path, seed_path, limit, dataset_name, max_per_category)
  - Load seed examples from local CSV
  - Load external rows from IvanTatarkin/ru-product-taxonomy HF dataset
  - Filter by lang_hint == "ru"
  - Map Russian product labels to receipt categories using RECEIPT_CATEGORY_MAP
  - Enforce max_per_category (default 30)
  - Balanced seed + external rows
- RECEIPT_CATEGORY_MAP: mapping from HF label prefixes to receipt categories
- _map_row(row): map a HF dataset row to (text, category) or None
- _extract_category(row): extract category label from HF row
- _to_receipt_category(raw_category): normalize HF category to receipt category"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

from datasets import load_dataset

from categorize.dataset import load_training_examples

HF_DATASET = "IvanTatarkin/ru-product-taxonomy"

# Mapping from normalized HF category names to receipt category names
RECEIPT_CATEGORY_MAP = {
    "продукты": "food",
    "еда": "food",
    "напитки": "food",
    "молочные": "food",
    "мясо": "food",
    "рыба": "food",
    "овощи": "food",
    "фрукты": "food",
    "бакалея": "food",
    "бытовая химия": "household",
    "товары для дома": "household",
    "уборка": "household",
    "косметика": "personal_care",
    "гигиена": "personal_care",
    "уход": "personal_care",
    "аптека": "medicine",
    "лекарства": "medicine",
    "медицина": "medicine",
    "кафе": "cafe",
    "ресторан": "cafe",
}


def build_training_csv(
    output_path: Path,
    seed_path: Path,
    limit: int = 2000,
    dataset_name: str = HF_DATASET,
    max_per_category: int = 30,
) -> None:
    """Build a balanced training CSV from seed examples + HF dataset.

    PDF behavior:
    - Load seed examples from seed_path (CSV with text,category)
    - Load up to `limit` rows from HF dataset (filtered by lang_hint=="ru")
    - Map HF labels to receipt categories using RECEIPT_CATEGORY_MAP
    - Enforce max_per_category per receipt category
    - Combine and write as CSV: text,category

    Args:
        output_path: Path to write the output CSV
        seed_path: Path to seed CSV (text,category)
        limit: Max number of external rows from HF dataset
        dataset_name: HF dataset name (default: IvanTatarkin/ru-product-taxonomy)
        max_per_category: Max examples per category (default 30)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seed_examples = load_training_examples(seed_path)
    external_rows = _load_external_rows(dataset_name, limit)
    rows = _balanced_rows(
        [{"text": example.text, "category": example.category} for example in seed_examples],
        external_rows,
        max_per_category,
    )
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["text", "category"])
        writer.writeheader()
        writer.writerows(rows)


def _balanced_rows(
    seed_rows: list[dict[str, str]],
    external_rows: list[dict[str, str]],
    max_per_category: int,
) -> list[dict[str, str]]:
    """Balance seed and external rows, max_per_category per category."""
    rows_by_category: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in seed_rows + external_rows:
        category = row["category"]
        if len(rows_by_category[category]) < max_per_category:
            rows_by_category[category].append(row)

    balanced: list[dict[str, str]] = []
    for category in sorted(rows_by_category):
        balanced.extend(rows_by_category[category])
    return balanced


def _load_external_rows(dataset_name: str, limit: int) -> list[dict[str, str]]:
    """Load external rows from Hugging Face dataset."""
    dataset = load_dataset(dataset_name, split="train", verification_mode="no_checks")
    rows: list[dict[str, str]] = []
    sorted_dataset = dataset.filter(lambda row: row.get("lang_hint") == "ru")
    if len(sorted_dataset) < limit:
        sorted_dataset = dataset
    for row in sorted_dataset:
        mapped = _map_row(row)
        if mapped is not None:
            rows.append(mapped)
        if len(rows) >= limit:
            break
    return rows


def _map_row(row: dict[str, Any]) -> dict[str, str] | None:
    """Map a single HF dataset row to (text, category) or None."""
    text = str(
        row.get("product_name")
        or row.get("title")
        or row.get("name")
        or row.get("product")
        or row.get("description")
        or row.get("text")
        or ""
    ).strip()
    raw_category = _extract_category(row)
    if not text or raw_category is None:
        return None
    category = _to_receipt_category(raw_category)
    if category is None:
        return None
    return {"text": text[:200], "category": category}


def _extract_category(row: dict[str, Any]) -> str | None:
    """Extract the category label from a HF dataset row."""
    value = (
        row.get("label")
        or row.get("label_name_ru")
        or row.get("categories")
        or row.get("category_name")
        or row.get("category")
        or row.get("taxonomy")
        or row.get("path")
    )
    if isinstance(value, list):
        return " ".join(str(part) for part in value)
    if value is None:
        return None
    return str(value)


def _to_receipt_category(raw_category: str) -> str | None:
    """Normalize a raw category string to a receipt category.

    PDF behavior:
    - Lowercase and normalize (replace _ with space, / with space)
    - If starts with "food" -> "food"
    - Check RECEIPT_CATEGORY_MAP for needle matches
    - Return None if no match found
    """
    normalized = raw_category.lower().replace("_", " ").replace("/", " ")
    if normalized.startswith("food"):
        return "food"
    for needle, category in RECEIPT_CATEGORY_MAP.items():
        if needle in normalized:
            return category
    return None
