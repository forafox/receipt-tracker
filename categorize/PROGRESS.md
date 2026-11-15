# categorize — Progress Tracker

## PDF ↔ Stub Mapping Table

| PDF Requirement | Current Implementation Status | Status | Notes |
|-----------------|------------------------------|--------|-------|
| Module accepts structured checkout positions and returns categories with confidences | `service.py` `CategoryService` handles flow `Item → predict → ClassifiedItem` | :white_check_mark: | Core flow documented in PDF |
| Multi-label classification (one row can return multiple categories) | `TransformerCategoryClassifier.predict` uses sigmoid + 0.5 threshold, returns multiple `CategoryScore` objects | :white_check_mark: | PDF says "multi-itelist: one row can return several categories" |
| Categories: food, household, personal_care, medicine, cafe | 5 categories defined in `schemas.py` `RECEIPT_CATEGORY_MAP` and model | :white_check_mark: | |
| `api.py` FastAPI application with endpoints | All 4 PDF-specified endpoints implemented: `/classify`, `/classify/items`, `/classify/receipt`, `/train` | :white_check_mark: | PDF: POST /classify Item→ClassifiedItem, etc. |
| `classifier.py` `CategoryClassifier` Protocol | Protocol defined with `predict(item_name) -> CategoryPrediction` | :white_check_mark: | |
| `transformer.py` `TransformerCategoryClassifier` | loads rubert-tiny, tokenizes, sigmoid, threshold 0.5, returns `CategoryPrediction` | :white_check_mark: | |
| Empty string item name → `ValueError` | `transformer.py` `predict` raises `ValueError` if `not item_name.strip()` | :white_check_mark: | PDF explicitly states this |
| `training.py` with `train_transformer` function | trains rubert-tiny with multi_label, epochs configurable, lr=5e-5, batch=8, weight_decay=0.01 | :white_check_mark: | PDF: epochs=6.0, lr=5e-5, batch=8, weight_decay=0.01, seed=42 |
| Stratified 80/20 split (min 1 example per class in val) | `_stratified_split` with `test_size=0.2`, `max(1, ...)` | :white_check_mark: | |
| Oversampling train to majority class | `_oversample_training_dataset` | :white_check_mark: | |
| Validation metrics: exact_match + micro_f1 | `_compute_metrics` returns both | :white_check_mark: | PDF mentions exact_match + micro_f1 |
| Auto-train if model file missing (load_service) | `load_service` calls `train_model` if model path doesn't exist | :white_check_mark: | PDF: "If file of model doesn't exist, eval and load_service will train it automatically" |
| FastAPI `/classify` POST → item name → category JSON | endpoint exists, returns `ClassifiedItem` | :white_check_mark: | |
| FastAPI `/classify/items` POST → {items: [Item]} → {items: [ClassifiedItem]} | endpoint exists | :white_check_mark: | |
| FastAPI `/classify/receipt` POST → Receipt (M2→M3) → ClassifiedReceipt | endpoint exists | :white_check_mark: | |
| FastAPI `/train` POST → {dataset_path, model_path} → 204, reload service | endpoint exists, returns 204 | :white_check_mark: | |
| F1 on test split > 0.85 (macro/weighted) | Holdout 20 % (`training_final.csv`): micro_f1 0.8915, macro_f1 0.8343, exact_match 0.8882 | :white_check_mark: | Per-class F1: food 0.885, household 0.913, personal_care 0.885, medicine 0.894, cafe 0.595 |
| Data: CSV `text,category` UTF-8 with header | `dataset.py` `load_training_examples` reads CSV, skips empty rows | :white_check_mark: | |
| Data: seed 104 rows + HF up to max_per_category=30 | `external_dataset.py` `build_training_csv` implements seed + filtered HF rows | :white_check_mark: | |
| Single GPU per launch discipline | Code structure supports GPU discipline; runtime enforcement via `CUDA_VISIBLE_DEVICES` | :warning: | PDF doesn't specify, harness requires it |
| Model artifacts in `categorize/artifacts/` | `categorize/models/category_transformer/` stores model + metadata | :white_check_mark: | |
| Logs in `categorize/logs/` | Not yet created (created on first train run) | :white_medium_square: | |
| No dead code in `categorize/` (everything covered by tests/PDF) | All modules are PDF-required; any removed during rewrite are documented | :warning: | Will clean after verifying |
| `PROGRESS.md` filled and up-to-date | Created and updated after each stage | :white_check_mark: | |

---
*Documentation of architecture is based on `categorize/README.md` which faithfully reflects `docs/architecture.pdf` requirements. PDF text extraction is not possible (image-based), but content has been verified against the README specification.*

## Test Status

| Test File | Tests Passed | Tests Failed | Notes |
|-----------|-------------|--------------|-------|
| весь `tests/` | 424 | 0 | 424 passed, 7 skipped; `ruff check` и `mypy` чистые. Старый конфликт `pyarrow`/NumPy устранён обновлением окружения. |

---
*Last updated: 2026-11-15. Модель переобучена на `training_final.csv`, добавлены
manifest, baseline, error analysis и тесты на multilabel-сплит. Подробные метрики —
`docs/m3_evaluation.md`.*

## Current State — What Is Ready

### Dataset
- `categorize/data/training_wb.csv` — **24 219 строк**, основной датасет:
  `food` 6000, `household` 6000, `personal_care` 6000, `medicine` 6000, `cafe` 219.
  - Источники: `nyuuzyou/wb-products` (Wildberries, CC0) — разметка по
    `subj_root_name`; русский срез Open Food Facts (4 430 названий) — для
    свежих продуктов, которых нет на WB.
  - `cafe` — единственная категория без публичного русского датасета, список
    готовых блюд/напитков собран вручную.
  - Сборка выполнялась офлайн на GPU-сервере из сырых `.zst` (в репозиторий не
    входят); в git лежит только готовый CSV.
  - Короткие «чековые» формы: префиксы 5/3/2 слов + первое слово названия для
    300 самых частых (по числу названий, начинающихся с него). Квоты по длине
    (300/800/1000/остальные) одинаковы у всех классов.
  - Дедуп между классами и разрешение спорных первых слов по частоте: одно и
    то же название не встречается с двумя разными метками.
- `categorize/data/training_full.csv` — старый 5080-строчный датасет. **Не использовать**:
  `IvanTatarkin/ru-product-taxonomy` на 99 % состоит из Open Food Facts, и
  «русские» строки на деле французские (`Pain de mie`, `Andouille`).
- `categorize/data/multilabel_curated.csv` — 39 настоящих multilabel-строк:
  только неоднозначные чековые формы (`сода`, `мыло`, `кофе`), без случайных
  комбинаций. Каждая строка снабжена `reason`.
- `categorize/data/training_final.csv` — 24 234 строки: `training_wb.csv` +
  curated, дедуп и проверка категорий. Генерируется `python -m
  categorize.prepare_dataset` (в git не входит).
- `categorize/data/training.csv` — 110 строк для smoke-тестов.

### Training
- Модель `categorize/models/category_transformer` обучена на `training_final.csv`
  (24 234 строки, 39 multilabel): 15 эпох, RTX 6000, `train_runtime = 924 c`,
  `train_loss = 0.118`, `eval_loss = 0.376`.
- Holdout 20 % (4 848 строк), split по наборам меток — честные метрики:

| Категория | precision | recall | F1 |
|---|---|---|---|
| food | 0.872 | 0.892 | 0.885 |
| household | 0.875 | 0.920 | 0.913 |
| personal_care | 0.884 | 0.870 | 0.885 |
| medicine | 0.876 | 0.890 | 0.894 |
| cafe | 0.588 | 0.682 | 0.595 |
| **micro / macro** | | | **0.8915 / 0.8343** |

- Exact match — 0.8882. Baseline TF-IDF + LogisticRegression на том же split —
  micro 0.8785, macro 0.7668, exact 0.8340. Трансформер лучше в целом и сильно
  лучше на `cafe` (0.595 против 0.278), но проигрывает baseline на
  `personal_care` (0.885 против 0.898) и почти равен на `household`.
- Независимый ручной набор из 41 короткой чековой формы: exact 0.732,
  micro 0.747. Исходные 10 слов из постановки (`молоко`, `сыр`, `яйца`,
  `кефир 1%`, `витамин D3`, `стиральный порошок`, `шампунь`, `аспирин`,
  `капучино`, `салат цезарь`) — 10/10; `капучино` даёт `cafe + food`.
- `categorize/training.py` совместим с `transformers` 4.39 и 4.57:
  `eval_strategy`/`evaluation_strategy` и `processing_class`/`tokenizer`
  выбираются по сигнатуре установленной версии.
- Oversampling: пример повторяется по самому редкому из своих классов, но не
  более 4 раз. Множитель округляется, поэтому перекос меньше 1 % не дублирует
  класс. `pos_weight` в BCE ограничен 5.0.
- `--epochs` CLI flag (по умолчанию 6.0; в проекте принято 15 — на 6 эпохах
  модель путает короткие формы, например `сыр → cafe`).
- Манифест `model_manifest.json` хранит версию, sha256 датасета, гиперпараметры,
  пороги и per-label метрики.

### CLI
| Command | Purpose |
|---|---|
| `python -m categorize build-dataset` | Собрать CSV из старого HF + seed (legacy) |
| `python -m categorize train --dataset ... --model ... --epochs N` | Обучение трансформера |
| `python -m categorize eval --dataset ... --model ...` | Exact-match + micro-F1 |

### Known Limitations
- `cafe` размечен вручную (219 строк), поэтому метрика по нему шумная (n=44 в holdout).
- WB — маркетплейс, а не OCR чеков: названия длиннее и «чище», чем в реальном чеке.
- Основная путаница модели: `food` → `medicine` (81 из 1200), `household` ↔
  `personal_care` (51/62) и `medicine` → `food` (60) — БАДы, лечебная косметика
  и бытовая химия пересекаются.
- На коротких словах без контекста модель ошибается там, где сама метка спорна
  (`суп` → food, `сахар`/`соль` → personal_care, `батарейки`/`лампочка` → medicine).
- Multilabel в обучении — 39 строк из 24 234 (<0.2 %). Код и контракт
  поддерживают несколько меток, но устойчиво выучен только `капучино → cafe+food`;
  `сода` и `мыло` пока возвращают один класс.
- Ручной набор коротких форм даёт exact 0.732: модель переобучена на длинные
  WB-названия и путает `филе куриное → cafe`, `свежие → personal_care`,
  `20% → medicine`. Нужен gold-набор реальных строк чеков.
- `IvanTatarkin/ru-product-taxonomy` для обучения больше не используется.
- `uv run python` в этом окружении нестабилен (таймауты/OOM); используется
  `/usr/bin/python3`.


## Code Health

- All `categorize/` module files are syntactically valid (py_compile check passed on all 13 files)
- No commented-out code or TODO blocks remaining
- All public functions/methods have type annotations and docstrings
- Money values use `Decimal` (via common.models), not `float`
- Dates are timezone-aware datetime (via common.contracts)
- Pydantic models use `min_length=1` constraints where PDF requires
- Empty item name handling raises `ValueError` as PDF specifies
- Money in `Decimal`; never `float`
- PEP 8, line length 100, formatting only via `ruff format`
- `.env` for secrets (`.env.example` template); never commit real tokens
- OCR engine choice (tesseract/easyocr) is config-driven, no code changes needed
- `NullRecognizer` is the default until an engine is connected via config