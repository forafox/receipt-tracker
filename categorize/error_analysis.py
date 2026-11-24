"""Inspect categorization errors on a held-out or manual dataset.

Usage:
    python -m categorize.error_analysis --model DIR --dataset CSV --split test
    python -m categorize.error_analysis --model DIR --dataset holdout_manual.csv
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from datasets import Dataset

from categorize.dataset import load_training_examples
from categorize.schemas import TrainingExample
from categorize.training import _stratified_split
from categorize.transformer import TransformerCategoryClassifier

DEFAULT_MODEL = Path("categorize/models/category_transformer")
DEFAULT_DATASET = Path("categorize/data/training_final.csv")
DEFAULT_OUTPUT = Path("categorize/models/error_analysis.json")


def _test_fold(dataset_path: Path) -> list[TrainingExample]:
    examples = load_training_examples(dataset_path)
    labels = sorted({label for example in examples for label in example.category.split("|")})
    dataset = Dataset.from_dict(
        {
            "text": [example.text for example in examples],
            "labels": [
                [1.0 if label.strip() in example.category.split("|") else 0.0 for label in labels]
                for example in examples
            ],
            "group_id": [
                example.group_id or example.text.casefold().strip() for example in examples
            ],
        }
    )
    _, test = _stratified_split(dataset, test_size=0.2)
    wanted = set(test["text"])
    return [example for example in examples if example.text in wanted]


def run(
    model_path: Path, dataset_path: Path, output: Path, split: str, limit: int
) -> dict[str, object]:
    classifier = TransformerCategoryClassifier.load(model_path)
    examples = _test_fold(dataset_path) if split == "test" else load_training_examples(dataset_path)

    true_positive: defaultdict[str, int] = defaultdict(int)
    false_positive: defaultdict[str, int] = defaultdict(int)
    false_negative: defaultdict[str, int] = defaultdict(int)
    errors: list[dict[str, object]] = []
    exact = 0

    for example in examples:
        expected = {label.strip() for label in example.category.split("|") if label.strip()}
        prediction = classifier.predict(example.text)
        predicted = {score.category for score in prediction.categories}
        exact += expected == predicted
        for label in expected & predicted:
            true_positive[label] += 1
        for label in predicted - expected:
            false_positive[label] += 1
        for label in expected - predicted:
            false_negative[label] += 1
        if expected != predicted:
            errors.append(
                {
                    "text": example.text,
                    "expected": sorted(expected),
                    "predicted": sorted(predicted),
                    "fallback": prediction.fallback,
                    "scores": {score.category: score.confidence for score in prediction.categories},
                }
            )

    labels = sorted({label for example in examples for label in example.category.split("|")})
    per_label: dict[str, dict[str, float]] = {}
    for label in labels:
        tp = true_positive[label]
        fp = false_positive[label]
        fn = false_negative[label]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {"precision": precision, "recall": recall, "f1": f1}

    total_tp = sum(true_positive.values())
    total_fp = sum(false_positive.values())
    total_fn = sum(false_negative.values())
    denominator = 2 * total_tp + total_fp + total_fn
    result: dict[str, object] = {
        "examples": len(examples),
        "exact_match": exact / len(examples) if examples else 0.0,
        "micro_f1": 2 * total_tp / denominator if denominator else 0.0,
        "per_label": per_label,
        "error_count": len(errors),
        "errors": errors[:limit],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split", choices=["all", "test"], default="test")
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()
    result = run(args.model, args.dataset, args.output, args.split, args.limit)
    print(
        json.dumps({key: result[key] for key in ("examples", "exact_match", "micro_f1")}, indent=2)
    )
    print(json.dumps(result["per_label"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
