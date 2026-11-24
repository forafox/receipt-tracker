"""Run the first ten annotated receipt images through OCR, M2 and M3."""

import json
import os
import urllib.request
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

from categorize.ollama import OllamaCategoryValidator
from categorize.service import CategoryService
from categorize.transformer import TransformerCategoryClassifier
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

    for wrapped in rows:
        row = wrapped["row"]
        with urllib.request.urlopen(row["image"]["src"], timeout=30) as response:
            image = response.read()
        ocr = engine.recognize_blocks(image)
        baseline = parse_ocr_result(ocr, timezone=timezone)
        receipt = parse_with_agent(ocr, timezone=timezone)
        target_total = Decimal(row["total_text"])
        target_date = datetime.strptime(row["date_text"], "%d.%m.%y").date()
        store_target = row["seller_text"].lower().replace("ооо", "").replace('"', "").strip()
        store_found = (receipt.store or "").lower().replace("ооо", "").replace('"', "").strip()
        categories = []
        for item in receipt.items:
            classified = category_service.classify_item(item)
            categories.append(
                {
                    "name": item.name,
                    "categories": [score.category for score in classified.categories],
                    "validation": classified.category_validation,
                    "suggested": classified.suggested_categories,
                }
            )
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
                    "store": bool(store_target and store_target in store_found),
                    "date": bool(receipt.datetime and receipt.datetime.date() == target_date),
                    "total": receipt.total == target_total,
                },
                "m3": categories,
            }
        )
        print(json.dumps(results[-1], ensure_ascii=False))

    output_path = Path(os.getenv("E2E_OUTPUT", "/tmp/receipt-e2e10-results.json"))
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
