from categorize.eval import evaluate
from categorize.schemas import TrainingExample
from common import CategoryPrediction


class FakeClassifier:
    def predict(self, item_name: str) -> CategoryPrediction:
        if "молоко" in item_name:
            return CategoryPrediction(category="food", confidence=0.9)
        return CategoryPrediction(category="household", confidence=0.8)


def test_classifier_protocol_predicts() -> None:
    prediction = FakeClassifier().predict("молоко ультрапастеризованное")

    assert prediction.category == "food"
    assert 0 <= prediction.confidence <= 1


def test_classifier_eval_reports_accuracy() -> None:
    examples = [
        TrainingExample(text="молоко", category="food"),
        TrainingExample(text="порошок", category="household"),
    ]

    result = evaluate(FakeClassifier(), examples)

    assert result.accuracy == 1
    assert result.examples == len(examples)
