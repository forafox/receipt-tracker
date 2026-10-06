from pydantic import BaseModel, Field

from categorize.classifier import CategoryClassifier
from categorize.schemas import TrainingExample


class EvalResult(BaseModel):
    accuracy: float = Field(ge=0, le=1)
    examples: int = Field(ge=0)


def evaluate(classifier: CategoryClassifier, examples: list[TrainingExample]) -> EvalResult:
    if not examples:
        return EvalResult(accuracy=0, examples=0)

    correct = 0
    for example in examples:
        expected = {label.strip() for label in example.category.split("|") if label.strip()}
        predicted = {score.category for score in classifier.predict(example.text).categories}
        if expected == predicted:
            correct += 1
    return EvalResult(accuracy=correct / len(examples), examples=len(examples))
