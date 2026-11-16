import pytest

from categorize.eval import evaluate
from categorize.schemas import TrainingExample
from categorize.transformer import normalize_case
from common import CategoryPrediction, CategoryScore


class FakeClassifier:
    def predict(self, item_name: str) -> CategoryPrediction:
        if "молоко" in item_name:
            return CategoryPrediction(categories=[CategoryScore(category="food", confidence=0.9)])
        return CategoryPrediction(categories=[CategoryScore(category="household", confidence=0.8)])


def test_classifier_protocol_predicts() -> None:
    prediction = FakeClassifier().predict("молоко ультрапастеризованное")

    assert prediction.categories[0].category == "food"
    assert 0 <= prediction.categories[0].confidence <= 1


def test_classifier_eval_reports_accuracy() -> None:
    examples = [
        TrainingExample(text="молоко", category="food"),
        TrainingExample(text="порошок", category="household"),
    ]

    result = evaluate(FakeClassifier(), examples)

    assert result.accuracy == 1
    assert result.examples == len(examples)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("МОЛОКО 3.2%", "молоко 3.2%"),
        ("ХЛЕБ", "хлеб"),
        ("Молоко 3,2% пастеризованное", "Молоко 3,2% пастеризованное"),
        ("капучино", "капучино"),
        ("123 456", "123 456"),
    ],
)
def test_normalize_case_lowercases_only_all_caps_names(name: str, expected: str) -> None:
    assert normalize_case(name) == expected
