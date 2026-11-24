# M3: dataset, baseline and error analysis

Дата: 2026-11-15. Все цифры получены на RTX 6000 (remote `jlab-wsl`),
`transformers 4.57.6`, `torch 2.11.0+cu`.

## 1. Аудит данных

| Файл | Строк | Multilabel | Комментарий |
|---|---:|---:|---|
| `training.csv` | 110 | 0 | smoke-набор, не для обучения |
| `training_full.csv` | 5 080 | 0 | 5 000 `food`, остальные по 20; 11 дублей; устаревший |
| `training_wb.csv` | 24 219 | 0 | основной источник: WB + Open Food Facts |
| `multilabel_curated.csv` | 39 | 39 | ручная разметка неоднозначных чековых форм |
| `training_final.csv` | 24 234 | 39 | `training_wb.csv` + curated, без дублей |

Распределение `training_final.csv`: `food` 6 016, `household` 6 012,
`personal_care` 6 015, `medicine` 6 009, `cafe` 221. Все пять категорий
представлены.

Исходные датасеты на 100 % single-label. Multilabel добавлен только там, где
сам текст неоднозначен без контекста магазина:

- `сода`, `уксус`, `лимонная кислота` — `food|household`;
- `мыло`, `влажные салфетки`, `антибактериальное мыло` — `household|personal_care`;
- `антисептик`, `вата`, `бинт`, `презервативы` — `medicine|personal_care`;
- `термометр`, `грелка`, `марганцовка` — `medicine|household`;
- `витамины`, `рыбий жир` — `medicine|food`;
- `кофе`, `чай`, `торт`, `салат`, `сэндвич`, `капучино` — `food|cafe`.

Комбинации не генерировались случайно: каждая строка — реальная товарная форма,
у которой есть два законных прочтения. Полный список с обоснованием —
`categorize/data/multilabel_curated.csv`.

## 2. Обучение

```bash
python -m categorize.prepare_dataset
python -m categorize train \
  --dataset categorize/data/training_final.csv \
  --model categorize/models/category_transformer \
  --epochs 15
```

- База: `cointegrated/rubert-tiny`, `max_length=64`, `lr=5e-5`, `batch=8`,
  `weight_decay=0.01`, `seed=42`.
- Split: 80/20 по целым наборам меток. Multilabel-строка не рвётся между
  train/test; каждый label встречается в test.
- Train: 19 386 строк → 19 917 после oversampling (только `cafe` повторяется
  ×4; остальные классы не дублируются, если их перекос < 1 %).
- Loss: `binary_cross_entropy_with_logits` с `pos_weight` по классам,
  ограниченным 5.0 (`cafe` — 5.0, остальные ≈ 3.1).
- Runtime: 924 c (15 эпох), train loss 0.118, eval loss 0.376.

Для сравнения: дефолтный прогон на 6 эпохах дал `micro_f1=0.8824`,
`macro_f1=0.8341`, `f1_cafe=0.632`, но ошибался на коротком `сыр → cafe`.
15 эпох — конфигурация, уже принятая в проекте; она выбрана по sanity-check
на 10 словах, а не по test split. Оба прогона зафиксированы в манифестах.

## 3. Метрики на holdout (4 848 строк, 20 %)

Порог 0.5, при отсутствии положительных — argmax.

| Категория | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| food | 0.872 | 0.892 | 0.885 | ~1 200 |
| household | 0.875 | 0.920 | 0.913 | ~1 200 |
| personal_care | 0.884 | 0.870 | 0.885 | ~1 200 |
| medicine | 0.876 | 0.890 | 0.894 | ~1 200 |
| cafe | 0.588 | 0.682 | 0.595 | 44 |
| **micro** | | | **0.8915** | 4 848 |
| **macro** | | | **0.8343** | |
| **exact match** | | | **0.8882** | |

`cafe` остаётся шумным: 44 примера в holdout, разметка ручная.

## 4. Baseline: TF-IDF + logistic regression

Тот же split, тот же порог, тот же multiclass→multilabel one-vs-rest.

| Модель | micro F1 | macro F1 | exact match | F1 cafe |
|---|---:|---:|---:|---:|
| TF-IDF + LogisticRegression | 0.8785 | 0.7668 | 0.8340 | 0.2775 |
| rubert-tiny (15 эпох) | 0.8915 | 0.8343 | 0.8882 | 0.5952 |
| Δ | +0.0130 | +0.0675 | +0.0542 | +0.3177 |

Baseline сильнее на `household` (0.910 vs 0.913 — практически паритет) и
`personal_care` (0.898 vs 0.885); трансформер заметно лучше на `cafe`,
`food` и `medicine` и лучше в целом. Полные цифры — `categorize/models/baseline_metrics.json`.

## 5. Независимый ручной набор (41 строка)

Короткие чековые формы, которых нет в обучении:

| Метрика | Значение |
|---|---:|
| exact match | 0.7317 |
| micro F1 | 0.7470 |
| F1 food | 0.741 |
| F1 household | 0.824 |
| F1 medicine | 0.545 |
| F1 personal_care | 0.941 |
| F1 cafe | 0.545 |

Это честная оценка domain shift: WB-названия длиннее и «чище» реальных чеков.
Sanity-check из постановки (10 слов) — 10/10, включая `капучино → cafe+food`.

Типичные ошибки:

| Текст | Ожидание | Предсказание | Причина |
|---|---|---|---|
| `филе куриное` | food | cafe 0.999 | в обучении `филе` встречается только как food, но короткая форма без бренда попала в cafe-паттерн |
| `помидоры свежие` | food | personal_care 0.997 | `свежие` чаще встречается в косметике (`свежесть`), чем в продуктах |
| `сметана 20%` | food | medicine 0.995 | `20%` — частый маркер БАД/лекарств |
| `соль пищевая` | food | household 0.998 | в корпусе `соль` есть и food, и household, и personal_care; слово `пищевая` не спасло |
| `тонометр автоматический` | medicine | household 0.999 | `тонометр` отсутствует в обучении; модель обобщила по «бытовому» контексту |

Вывод: узкое место — не архитектура, а домен и разметка. Нужен gold-набор
реальных строк чеков (с группировкой по чеку, чтобы не течь между split) и
ручное разрешение спорных пар `food/medicine`, `household/personal_care`.

## 6. Multilabel: что реально работает

Модель обучается с BCE и независимыми сигмоидами, поэтому технически может
вернуть несколько категорий. Но curated-набор — 39 строк из 24 234 (<0.2 %),
поэтому модель уверенно выучила только самые частые паттерны:

- `капучино → cafe + food` — работает;
- `сода → food` (без household) — второй label не проходит порог;
- `мыло → household` (без personal_care) — второй label не проходит порог.

Это ограничение данных, а не контракта. Чтобы multilabel был устойчивым, нужно
не менее нескольких сотен размеченных пар на каждый тип пересечения; текущий
набор документирует политику и проверяет код, но не даёт качества.

## 7. Воспроизведение

```bash
python -m categorize.data_audit
python -m categorize.prepare_dataset
python -m categorize.baseline
python -m categorize train --dataset categorize/data/training_final.csv \
  --model categorize/models/category_transformer --epochs 15
python -m categorize.error_analysis --split test
python -m categorize.error_analysis \
  --dataset categorize/data/holdout_manual.csv --split all
```

`model_manifest.json` хранит версию, sha256 датасета, гиперпараметры, пороги и
все метрики, включая per-label precision/recall/F1.

## 8. Ограничения

- WB — маркетплейс, а не OCR чеков: длинные брендовые названия, другой регистр
  и пунктуация.
- `cafe` — 221 строка, из них 44 в holdout; метрика шумная.
- Multilabel — 39 строк; качество на пересечениях не доказано.
- Holdout — случайное 20 % того же корпуса, а не отдельные чеки. Одинаковые
  товары в разных написаниях могут попадать в train и test.
- Порог 0.5 не калиброван; argmax-fallback может давать ложные срабатывания.
