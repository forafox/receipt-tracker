# E2E: ten real receipts

Run date: 2026-11-24. Dataset: `cdek-ocr/receipt-ocr-ru`, `test` split. The run
used the first ten rows among the first thirty with a numeric annotated total and
date. Each image passed through Tesseract M1, deterministic M2 plus the Ollama
repair fallback, the existing M3 classifier, and the Ollama category check.

## Results

| Measure | Result |
|---|---:|
| Store name exact/normalized match | 3/10 |
| Receipt date match | 2/10 |
| Total match | 3/10 |
| Annotated item rows | 43 |
| Item rows extracted by regex parser | 8 |
| Item rows extracted with Ollama fallback | 17 |
| Extracted sums matching any annotated item sum | 8/17 |
| M3 outputs rechecked by Ollama | 17 |
| M3 checks: agreed / disagreed / uncertain | 9 / 6 / 2 |

Compared with the regex-only parse, the accepted Ollama fallback raised store
matches from 0 to 3, total matches from 2 to 3, and extracted item rows from 8 to
17. Date matches stayed at 2. The fallback still missed most of the 43 annotated
items, and only 8 of its 17 amounts matched any annotated item sum. It also
retained arithmetic warnings where evidence did not reconcile. `IMG_2026_enh`
has ten annotated items; the fallback found three, while retaining its correct
date and total.

The dataset has no product-category annotations, so M3 category accuracy cannot
be measured on these rows. On the 17 extracted rows Ollama agreed with nine M3
predictions, disagreed with six, and returned `uncertain` for two. The M3 labels
remain the selected categories when the validator is uncertain; suggestions are
reported separately. Agreement is not proof of correctness.

## Example

On `IMG_2026_enh`, OCR found `10000702652 CYMMA (РУБ.); 1678.38`. M2 returned
store `ЛЕНТА`, date `2025-11-05`, total `1678.38`, and three item rows (the
annotation has ten). For `САРДЕЛЬКИ N3 ПОМОНОСОВ`, M3 selected `household` and
the local check suggested `food`:

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
annotations from the Hugging Face dataset server and writes full outputs to
`/tmp/receipt-e2e10-results.json` by default.
