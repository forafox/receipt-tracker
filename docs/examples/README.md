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

## November 2025 receipts

The newest November receipts in this example set are:

| Date | Example | Store | Total | Parsed items |
|---|---|---|---:|---:|
| 03.11.2025 | [IMG_2040_enh.jpg](IMG_2040_enh.jpg) | Lenta | 214.98 | 2 |
| 05.11.2025 | [IMG_2026_enh.jpg](IMG_2026_enh.jpg) | Lenta | 1678.38 | 10 |
| 06.11.2025 | [IMG_1987_enh.jpg](IMG_1987_enh.jpg) | SPAR | 694.70 | 3 |

Their JSON sidecars now contain the current M2 parse and M3 categories from the bot-handler
end-to-end run. In particular, the SPAR example now has its three item rows instead of the
older empty parse. The full bot responses are in `category_spending_e2e.json`.

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
