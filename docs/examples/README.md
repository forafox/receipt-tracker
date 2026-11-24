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
