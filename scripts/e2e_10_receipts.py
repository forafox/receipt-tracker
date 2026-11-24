"""Run the first ten annotated receipt images through OCR, M2 and M3."""

import json
import os
import time
import urllib.request
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

from categorize.classifier import CategoryClassifier
from categorize.ollama import OllamaCategoryValidator
from categorize.service import CategoryService
from categorize.transformer import TransformerCategoryClassifier
from common.contracts import Item
from extract.agent import parse_with_agent
from extract.layout import reconstruct_layout
from extract.parser import parse_ocr_result
from ocr.engines.tesseract import TesseractEngine


def main() -> None:
    """Fetch ten annotated test receipts and report M1–M3 results."""
    tesseract_cmd = os.getenv("TESSERACT_CMD")
    import pytesseract

    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    rows_url = (
        "https://datasets-server.huggingface.co/rows?dataset=cdek-ocr/receipt-ocr-ru"
        "&config=default&split=test&offset=0&length=30"
    )
    with urllib.request.urlopen(rows_url, timeout=30) as response:
        rows = json.loads(response.read())["rows"]
    rows = [
        wrapped
        for wrapped in rows
        if wrapped["row"].get("date_text")
        and wrapped["row"].get("total_text")
        and wrapped["row"].get("total_text").replace(".", "", 1).isdigit()
    ][: int(os.getenv("E2E_LIMIT", "10"))]
    engine = TesseractEngine(languages="rus+eng")
    classifier = TransformerCategoryClassifier.load(Path("categorize/models/category_transformer"))
    category_service = CategoryService(classifier, OllamaCategoryValidator())
    timezone = ZoneInfo("Europe/Moscow")
    results = []
    image_cache = Path(os.getenv("E2E_IMAGE_CACHE", "/tmp/receipt-e2e10-images"))
    image_cache.mkdir(parents=True, exist_ok=True)
    output_path = Path(os.getenv("E2E_OUTPUT", "/tmp/receipt-e2e10-results.json"))

    for wrapped in rows:
        row = wrapped["row"]
        image_path = image_cache / f"{row['image_id']}.jpg"
        if image_path.exists():
            image = image_path.read_bytes()
        else:
            image = _download(row["image"]["src"])
            image_path.write_bytes(image)
        ocr = engine.recognize_blocks(image)
        baseline = parse_ocr_result(ocr, timezone=timezone)
        receipt = parse_with_agent(ocr, timezone=timezone)
        target_total = Decimal(row["total_text"])
        target_date = _parse_target_date(row["date_text"])
        store_target = row["seller_text"].lower().replace("ооо", "").replace('"', "").strip()
        base_store = (baseline.store or "").lower().replace("ооо", "").replace('"', "").strip()
        final_store = (receipt.store or "").lower().replace("ооо", "").replace('"', "").strip()
        base_categories = _classify_items(classifier, baseline.items)
        categories = _classify_validated_items(category_service, receipt.items)
        results.append(
            {
                "id": row["image_id"],
                "ocr_blocks": len(ocr.blocks),
                "ocr_text": "\n".join(line.text for line in reconstruct_layout(ocr)),
                "target": {
                    "store": row["seller_text"],
                    "date": row["date_text"],
                    "total": str(target_total),
                    "items": row["item_texts"],
                    "item_sums": row["item_sums"],
                },
                "base": {
                    "store": baseline.store,
                    "date": str(baseline.datetime.date()) if baseline.datetime else None,
                    "total": str(baseline.total) if baseline.total is not None else None,
                    "items": [{"name": item.name, "sum": str(item.sum)} for item in baseline.items],
                    "warnings": baseline.warnings,
                },
                "final": {
                    "store": receipt.store,
                    "date": str(receipt.datetime.date()) if receipt.datetime else None,
                    "total": str(receipt.total) if receipt.total is not None else None,
                    "items": [{"name": item.name, "sum": str(item.sum)} for item in receipt.items],
                    "warnings": receipt.warnings,
                },
                "field_match": {
                    "store": bool(store_target and store_target in final_store),
                    "date": bool(receipt.datetime and receipt.datetime.date() == target_date),
                    "total": receipt.total == target_total,
                },
                "base_match": {
                    "store": bool(store_target and store_target in base_store),
                    "date": bool(baseline.datetime and baseline.datetime.date() == target_date),
                    "total": baseline.total == target_total,
                },
                "m3_regex": base_categories,
                "m3": categories,
            }
        )
        print(json.dumps(results[-1], ensure_ascii=False))
        output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


def _download(url: str) -> bytes:
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=45) as response:
                return response.read()
        except OSError:
            if attempt == 2:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("image download retries exhausted")


def _parse_target_date(value: str) -> date:
    date_text = value.split()[0]
    for date_format in ("%d.%m.%y", "%d.%m.%Y"):
        try:
            return datetime.strptime(date_text, date_format).date()
        except ValueError:
            continue
    raise ValueError(f"unsupported annotated receipt date: {value}")


def _classify_items(classifier: CategoryClassifier, items: list[Item]) -> list[dict]:
    results = []
    for item in items:
        prediction = classifier.predict(item.name)
        results.append(
            {
                "name": item.name,
                "categories": [score.category for score in prediction.categories],
            }
        )
    return results


def _classify_validated_items(service: CategoryService, items: list[Item]) -> list[dict]:
    results = []
    for item in items:
        classified = service.classify_item(item)
        results.append(
            {
                "name": item.name,
                "categories": [score.category for score in classified.categories],
                "validation": classified.category_validation,
                "suggested": classified.suggested_categories,
            }
        )
    return results


if __name__ == "__main__":
    main()
