# E2E: regex on main vs Ollama fallback

Run date: 2026-11-24. Dataset: `cdek-ocr/receipt-ocr-ru`, `test` split. The run
used the first ten rows among the first thirty with a numeric annotated total and
date. Each photo was recognized once by Tesseract; those same OCR blocks were
parsed by the regex version from updated `main` and by regex plus the local Ollama
fallback. The existing M3 classifier and Ollama validator ran on fallback items.

## Results

| Measure | Result |
|---|---:|
| Store name exact/normalized match | regex 0/10 → fallback 2/10 |
| Receipt date match | regex 2/10 → fallback 2/10 |
| Total match | regex 2/10 → fallback 3/10 |
| Annotated item rows | 43 |
| Item rows extracted | regex 8 → fallback 17 |
| Extracted sums matching any annotated item sum | regex 2/8 → fallback 8/17 |
| Receipts with at least one warning | regex 10/10 → fallback 10/10 |
| M3 outputs rechecked by Ollama | 17 |
| M3 checks: agreed / disagreed / uncertain | 9 / 7 / 1 |

The local fallback improves store and total matches slightly and more than
doubles the number of extracted item rows. Date accuracy is unchanged. It still
misses most of the 43 annotated rows; 8 of its 17 extracted sums match any
annotated sum, and every receipt retains at least one warning. Some accepted
item rows are false positives or have sums that do not reconcile. `IMG_2026_enh`
has ten annotated items; regex finds three rows, and the fallback also finds
three. The fallback adds the correct store while retaining the correct date and
total, but reports an arithmetic mismatch.

The dataset has no product-category annotations, so M3 category accuracy cannot
be measured on these rows. On the 17 extracted rows Ollama agreed with nine M3
predictions, disagreed with six, and returned `uncertain` for two. The M3 labels
remain the selected categories when the validator is uncertain; suggestions are
reported separately. Agreement is not proof of correctness.

## Example

On `IMG_2026_enh`, OCR found `10000702652 CYMMA (РУБ.); 1678.38`. Regex returns
date `2025-11-05`, total `1678.38`, but no store. The fallback adds store
`ЛЕНТА`; it still extracts only three of ten annotated items. For
`САРДЕЛЬКИ N3 ПОМОНОСОВ`, M3 selected `household` and the local check suggested
`food`:

```json
{
  "name": "САРДЕЛЬКИ N3 ПОМОНОСОВ",
  "categories": ["household"],
  "category_validation": "disagreed",
  "suggested_categories": ["food"]
}
```

This disagreement is exposed for review; it does not silently overwrite the
classifier's selected category. The category annotations needed to decide which
model is correct are absent from this dataset.

## Reproduce

Install the project OCR dependencies and Tesseract with Russian and English data,
pull the local model, then run from the repository root:

```bash
uv sync --group dev --group ocr
ollama pull qwen2.5:1.5b-instruct-q4_K_M
PYTHONPATH=. python scripts/e2e_10_receipts.py
```

If Tesseract is outside `PATH`, set `TESSERACT_CMD`. The script fetches images and
annotations from the Hugging Face dataset server, caches images in
`/tmp/receipt-e2e10-images`, and writes full outputs to
`/tmp/receipt-e2e10-results.json` by default.
