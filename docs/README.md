# Документация

## С чего начать

| Документ | О чём |
|---|---|
| [../README.md](../README.md) | обзор проекта, модули, структура репозитория, проверки |
| [RUN.md](RUN.md) | полная инструкция по установке и запуску на одной машине |
| [../CONTRIBUTING.md](../CONTRIBUTING.md) | правила кода, веток, коммитов и PR |

## Модули

| Модуль | Документ |
|---|---|
| M1, распознавание | [../ocr/README.md](../ocr/README.md) |
| M2, разбор чека | [../extract/README.md](../extract/README.md) |
| M3, категории | [../categorize/README.md](../categorize/README.md) |
| M4, бот, пайплайн, хранилище | [../bot/README.md](../bot/README.md) |

## Остальные документы

| Документ | Владелец | О чём | Состояние |
|---|---|---|---|
| `architecture.jpg`, `architecture.pdf` | команда | первая схема системы (итерация 0) | исторический, сравнивать с текущим устройством |
| [m4/README.md](m4/README.md) | M4 | журнал разработки M4 по итерациям | ведётся |
| [m3_evaluation.md](m3_evaluation.md) | M3 | данные, обучение и метрики классификатора | отчёт |
| [m3_categorize.md](m3_categorize.md) | M3 | описание модуля M3 | во многом повторяет `categorize/README.md` |
| [hunyuanocr_evaluation.md](hunyuanocr_evaluation.md) | M3 | сравнение Hunyuan OCR и Tesseract на 10 чеках | отчёт |
| [e2e_10_receipts.md](e2e_10_receipts.md) | M3 | разбор правилами против Ollama на 10 чеках | отчёт |
| [examples/](examples/README.md) | M3 | 10 фото чеков с результатами распознавания | данные к отчётам |
| [HANDOFF.md](HANDOFF.md) | M3 | передача контекста по ветке `uvusibuneka` | частично устарел: числа тестов и состояние на момент написания |
| [bot.md](bot.md) | M3 | запуск конкретного экземпляра бота в WSL с Hunyuan | личная заметка, общая инструкция в `RUN.md` |

Рабочие заметки M3 лежат рядом с кодом: `categorize/PROGRESS.md` и `categorize/DATASET_RESEARCH_M3.md`.
