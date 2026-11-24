# HunyuanOCR receipt examples

These are ten real test-split receipt images from `cdek-ocr/receipt-ocr-ru`, with their HunyuanOCR transcripts. Filenames pair each `.jpg` with the corresponding `.txt` transcript. The samples were used to refine the deterministic M2 parser; OCR is not rerun by parser regression tests.

| ID | Notes |
|---|---|
| IMG_2040_enh | Lenta receipt with full bank/payment footer |
| IMG_2026_enh | Lenta table with ten products |
| IMG_0750_enh | Lenta Markdown-like table and totals in separate rows |
| IMG_1134_enh | Canteen table, tax rows, sole proprietor in footer |
| IMG_0929_enh | Tobacco/sweets receipt, totals near receipt footer |
| IMG_1202_enh | Canteen table with separate total |
| IMG_1369_enh | Canteen table with missing totals in OCR output |
| IMG_1987_enh | SPAR barcode-style item lines |
| IMG_0799_enh | Lenta receipt with duplicated OCR transcript |
| IMG_1685_enh | Lenta zero-total receipt with loyalty balance footer |

The image and transcript are original evaluation artifacts. Transcripts may contain OCR errors, omissions, or repeated output; they are preserved verbatim for reproducibility.

## Category spending end-to-end run

`category_spending_e2e.json` records an in-process end-to-end run over all ten images. The
photo handler received each image through the bot's download path, replayed its saved HunyuanOCR
transcript, parsed M2, classified items with the checked-in M3 transformer, saved receipts to
temporary SQLite, and answered `/month`. The runner independently checks receipt counts, receipt
totals, and every category amount in each response against the saved records. OCR inference
itself is not repeated, and no Telegram network messages are sent by this test.

Run it from the repository root in WSL with:

```bash
PYTHONPATH=. .venv/bin/python scripts/e2e_bot_category_examples.py \
  --report docs/examples/category_spending_e2e.json
```
