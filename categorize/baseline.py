"""TF-IDF + logistic regression baseline for the categorization task.

The transformer is only useful if it beats a cheap linear model on the same
split.  This script reuses the training split and the training metrics so the
comparison is apples-to-apples.

Usage:
    python -m categorize.baseline --dataset categorize/data/training_final.csv
"""

import argparse
import json
from pathlib import Path

import numpy as np
from datasets import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from transformers.trainer_utils import EvalPrediction

from categorize.dataset import load_training_examples
from categorize.text import denoise
from categorize.training import _compute_metrics, _stratified_split, tune_thresholds

DEFAULT_DATASET = Path("categorize/data/training_final.csv")
DEFAULT_OUTPUT = Path("categorize/models/baseline_metrics.json")


def _label_vector(category: str, label_to_id: dict[str, int]) -> list[float]:
    vector = [0.0] * len(label_to_id)
    for label in category.split("|"):
        vector[label_to_id[label.strip()]] = 1.0
    return vector


def run(dataset_path: Path, output: Path) -> dict[str, float]:
    examples = load_training_examples(dataset_path)
    labels = sorted({label for example in examples for label in example.category.split("|")})
    label_to_id = {label: index for index, label in enumerate(labels)}
    dataset = Dataset.from_dict(
        {
            "text": [example.text for example in examples],
            "labels": [_label_vector(example.category, label_to_id) for example in examples],
            "group_id": [
                example.group_id or example.text.casefold().strip() for example in examples
            ],
        }
    )
    train_pool, test = _stratified_split(dataset, test_size=0.2)
    train, validation = _stratified_split(train_pool, test_size=0.1)

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    train_features = vectorizer.fit_transform([denoise(text) for text in train["text"]])
    validation_features = vectorizer.transform([denoise(text) for text in validation["text"]])
    test_features = vectorizer.transform([denoise(text) for text in test["text"]])

    classifier = OneVsRestClassifier(
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    )
    classifier.fit(train_features, np.asarray(train["labels"]))
    validation_scores = classifier.decision_function(validation_features)
    validation_probabilities = 1 / (1 + np.exp(-validation_scores))
    validation_labels = np.asarray(validation["labels"])
    thresholds = tune_thresholds(validation_probabilities, validation_labels, labels)
    scores = classifier.decision_function(test_features)

    metrics = _compute_metrics(
        EvalPrediction(predictions=scores, label_ids=np.asarray(test["labels"])),
        label_names=labels,
        thresholds=[thresholds[label] for label in labels],
    )
    metrics.update({f"threshold_{label}": value for label, value in thresholds.items()})
    metrics["train_size"] = float(len(train))
    metrics["validation_size"] = float(len(validation))
    metrics["test_size"] = float(len(test))

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.dataset, args.output), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
