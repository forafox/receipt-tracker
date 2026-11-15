# M3: dataset decision and error analysis (2026-11-14)

## Current evidence from checked-in CSV files

- `training_wb.csv`: 24,219 rows, 0 genuine multi-label examples; class distribution 6000/6000/6000/6000/219.
- `training_full.csv`: 5,080 rows, 0 genuine multi-label examples; 5,000 `food` rows and 20 from each other category. Not suitable for replacement.
- `training.csv`: 110 rows, 0 genuine multi-label examples; smoke tests only.
- `PROGRESS.md` reports 0.8907 micro-F1 from an internal holdout, and `cafe` F1=0.7273 (44 examples).
- Do not interpret 0.8907 as a verified multi-label F1 on independent real OCR receipts.

## Source candidates (not yet downloaded or trained)

1. **Open Products Facts bulk JSONL** — category tags and product name, broad non-food products; ODbL. https://www.data.gouv.fr/datasets/open-products-facts . Requires language filtering, taxonomy mapping and manual verification.
2. **Open Beauty Facts** — hygiene and cosmetics with multiple `categories_tags`. https://github.com/openfoodfacts/openbeautyfacts . Primarily for `personal_care`; license/usage terms must be respected.
3. **Open Food Facts product-database** — large food product dataset with `categories_tags` arrays; contains mostly food items and cannot train the five broad labels alone. https://huggingface.co/datasets/openfoodfacts/product-database
4. **nyuuzyou/wb-products** — Russian item names and category metadata, already partially used in current corpus; single-category supervision and source quality limitations. https://huggingface.co/datasets/nyuuzyou/wb-products
5. **cdek-ocr/receipt-ocr-ru** — receipt OCR data useful for domain evaluation; no verified labels for the five M3 categories. https://huggingface.co/datasets/cdek-ocr/receipt-ocr-ru

**Important:** Having multiple hierarchical `categories_tags` does *not* prove a product belongs to several of M3's five groups. A mapping of nested tags that all mean `food` still produces exactly one M3 label. Use explicit multi-label human adjudication for cross-group cases. No single public dataset was verified to be simultaneously Russian receipt text + the exact five labels + genuine multi-label annotations.

## Proposed reproducible training data

- Base: the reviewed current Russian WB names; supplement with checked Russian-language product names from OPF, OBF and OFF. Do not blindly map English or French items into Russian training examples.
- Create a separate, human-labeled `receipt_gold.csv` from M2-extracted real receipt lines with `text,category,source,receipt_group_id`; category is pipe-separated among five labels. Keep group IDs so the same receipt and variations of the same product cannot leak between splits.
- Deliberately annotate real cross-group ambiguities (`food|medicine` for some supplements if policy permits, `household|personal_care` for dual-use hygiene/cleaning products). Never synthesize multi-label examples by randomly combining unrelated categories.
- Hold out entire receipts / normalized product groups; do not hold out generated abbreviations after splitting long versions across train/test.
- Measure exact match, micro-F1, macro-F1, per-label precision/recall/F1, multi-label subset recall and confidence calibration; compare to single-label baseline.
- Underrepresented `cafe` needs verified real receipt lines rather than repeating 219 seeds.

## Error taxonomy to investigate

| Problem | Evidence in PROGRESS.md | Action |
|---|---|---|
| Food confused with medicine | `food -> medicine`: 81/1200 | review vitamins, supplements, nutrition products and labeling policy |
| Medicine confused with food | `medicine -> food`: 60/1200 | review supplements vs food boundaries |
| Household vs personal care | 51/62 reported confusions | review soaps, antiseptics, dual-use products |
| Cafe too sparse | F1 0.7273, n=44 | collect genuine cafe receipt examples and evaluate on a separate cafe set |
| Short OCR lines | examples `сахар`, `соль`, `батарейки`, `лампочка` misclassified | build gold test set with truncated names and OCR corruption |

**Threshold note:** sigmoid returns a score between 0 and 1 but does not guarantee calibration; hardcoded 0.5 plus an argmax fallback can force false positives. Tune thresholds on a separate validation set once true multilabel examples exist.

## Useful commands

```bash
python -m categorize.data_audit categorize/data/training_wb.csv
python -m categorize.data_audit categorize/data/training_full.csv
python -m pytest -q tests/categorize/test_m2_integration.py tests/categorize/test_classifier.py
```
