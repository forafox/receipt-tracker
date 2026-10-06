from __future__ import annotations

import re

from common import CategoryScore

CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "food": (
        "молоко",
        "кефир",
        "йогурт",
        "сыр",
        "творог",
        "масло",
        "хлеб",
        "батон",
        "булка",
        "гречка",
        "рис",
        "макароны",
        "яйцо",
        "курица",
        "говядина",
        "свинина",
        "рыба",
        "колбаса",
        "сосиск",
        "яблок",
        "банан",
        "томат",
        "огур",
        "картоф",
        "сок",
        "вода",
        "чай",
        "сахар",
        "соль",
        "печенье",
        "шоколад",
    ),
    "household": (
        "порошок",
        "гель для стир",
        "кондиционер для белья",
        "средство для мытья",
        "fairy",
        "губк",
        "тряпк",
        "пакет",
        "фольга",
        "пергамент",
        "салфетк",
        "бумага туалет",
        "полотенц",
        "мыло хозяй",
        "чистящ",
        "отбелив",
        "освежитель",
        "лампоч",
        "батарей",
    ),
    "personal_care": (
        "шампун",
        "бальзам",
        "гель для душ",
        "дезодорант",
        "паста зуб",
        "зубная паст",
        "щетка зуб",
        "крем",
        "бритв",
        "проклад",
        "тампон",
        "ватн",
        "лосьон",
        "маска для лица",
        "мыло туалет",
        "head shoulders",
    ),
    "medicine": (
        "парацетамол",
        "ибупрофен",
        "аспирин",
        "цитрамон",
        "таблет",
        "капсул",
        "сироп",
        "мазь",
        "спрей",
        "капли",
        "бинт",
        "пластыр",
        "витамин",
        "аптеч",
        "термометр",
        "антисептик",
        "хлоргексидин",
    ),
    "cafe": (
        "латте",
        "капучино",
        "эспрессо",
        "американо",
        "раф",
        "бургер",
        "картофель фри",
        "фри",
        "пицца",
        "ролл",
        "шаурм",
        "сендвич",
        "сэндвич",
        "комбо",
        "кофейня",
        "кафе",
        "бизнес ланч",
        "десерт",
        "пирожное",
    ),
}


def predict_by_keywords(item_name: str) -> list[CategoryScore]:
    normalized = _normalize(item_name)
    scores: list[CategoryScore] = []
    for category, keywords in CATEGORY_KEYWORDS.items():
        matches = sum(1 for keyword in keywords if keyword in normalized)
        if matches:
            confidence = min(0.98, 0.72 + matches * 0.08)
            scores.append(CategoryScore(category=category, confidence=round(confidence, 4)))
    scores.sort(key=lambda score: score.confidence, reverse=True)
    return scores


def merge_category_scores(
    model_scores: list[CategoryScore],
    lexical_scores: list[CategoryScore],
) -> list[CategoryScore]:
    if lexical_scores:
        model_scores = [score for score in model_scores if score.confidence >= 0.5]
    merged: dict[str, CategoryScore] = {score.category: score for score in model_scores}
    for score in lexical_scores:
        current = merged.get(score.category)
        if current is None or score.confidence > current.confidence:
            merged[score.category] = score
    scores = sorted(merged.values(), key=lambda score: score.confidence, reverse=True)
    return scores


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold().replace("ё", "е")).strip()
