from pydantic import BaseModel, Field

from categorize.classifier import CategoryClassifier
from categorize.schemas import TrainingExample


class EvalResult(BaseModel):
    """Evaluation metrics for a set of labelled examples."""

    accuracy: float = Field(ge=0, le=1, description="Exact-match accuracy")
    examples: int = Field(ge=0, description="Number of evaluated examples")
    micro_f1: float = Field(default=0.0, ge=0, le=1, description="Micro-averaged F1")


def evaluate(classifier: CategoryClassifier, examples: list[TrainingExample]) -> EvalResult:
    """Evaluate ``classifier`` with exact-match accuracy and micro F1.

    Args:
        classifier: Category classifier to evaluate.
        examples: Labelled examples (``text`` + pipe-separated ``category``).

    Returns:
        Metrics for the provided examples. Empty input yields zeros.
    """
    if not examples:
        return EvalResult(accuracy=0, examples=0, micro_f1=0)

    exact_matches = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0

    for example in examples:
        expected = {label.strip() for label in example.category.split("|") if label.strip()}
        predicted = {score.category for score in classifier.predict(example.text).categories}
        if expected == predicted:
            exact_matches += 1
        true_positive += len(expected & predicted)
        false_positive += len(predicted - expected)
        false_negative += len(expected - predicted)

    denominator = 2 * true_positive + false_positive + false_negative
    micro_f1 = 2 * true_positive / denominator if denominator else 0.0
    return EvalResult(
        accuracy=exact_matches / len(examples),
        examples=len(examples),
        micro_f1=micro_f1,
    )
