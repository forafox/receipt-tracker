"""Build training artifacts, plots and an error report for M3.

The report is reproducible: it re-creates the exact group-aware train/test split
used by ``train_transformer`` and evaluates the saved model on the held-out fold.

Usage:
    python -m categorize.report --model categorize/models/category_transformer \
        --dataset categorize/data/training_final.csv \
        --log reports/m3/raw/train.log \
        --output reports/m3
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from categorize.dataset import load_training_examples
from categorize.text import denoise
from categorize.training import _stratified_split

THRESHOLD = 0.5
LABELS = ("cafe", "food", "household", "medicine", "personal_care")

DEFAULT_MODEL = Path("categorize/models/category_transformer")
DEFAULT_DATASET = Path("categorize/data/training_final.csv")
DEFAULT_LOG = Path("reports/m3/raw/train.log")
DEFAULT_OUTPUT = Path("reports/m3")


@dataclass
class Prediction:
    text: str
    expected: set[str]
    predicted: set[str]
    probabilities: dict[str, float]
    fallback: bool

    @property
    def top_label(self) -> str:
        return max(self.probabilities, key=lambda label: self.probabilities[label])

    @property
    def top_confidence(self) -> float:
        return self.probabilities[self.top_label]


def _test_examples(dataset_path: Path) -> list[Any]:
    examples = load_training_examples(dataset_path)
    labels = sorted({label for example in examples for label in example.category.split("|")})
    dataset = Dataset.from_dict(
        {
            "text": [example.text for example in examples],
            "labels": [
                [1.0 if label in example.category.split("|") else 0.0 for label in labels]
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


def _predict(model: Any, tokenizer: Any, texts: list[str], max_length: int) -> np.ndarray:
    output: list[np.ndarray] = []
    batch_size = 64
    for start in range(0, len(texts), batch_size):
        batch = [denoise(text) for text in texts[start : start + batch_size]]
        encoded = tokenizer(
            batch, truncation=True, padding=True, max_length=max_length, return_tensors="pt"
        )
        with torch.no_grad():
            logits = model(**encoded).logits
        output.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(output) if output else np.zeros((0, len(LABELS)))


def _build_predictions(
    examples: list[Any],
    probabilities: np.ndarray,
    labels: list[str],
    thresholds: dict[str, float],
) -> list[Prediction]:
    predictions: list[Prediction] = []
    for example, row in zip(examples, probabilities, strict=True):
        expected = {label.strip() for label in example.category.split("|") if label.strip()}
        scores = {label: float(row[index]) for index, label in enumerate(labels)}
        predicted = {
            label for label, score in scores.items() if score >= thresholds.get(label, THRESHOLD)
        }
        fallback = not predicted
        if fallback:
            predicted = {max(scores, key=lambda label: scores[label])}
        predictions.append(
            Prediction(
                text=example.text,
                expected=expected,
                predicted=predicted,
                probabilities=scores,
                fallback=fallback,
            )
        )
    return predictions


def _metrics(predictions: list[Prediction], labels: list[str]) -> dict[str, Any]:
    tp: defaultdict[str, int] = defaultdict(int)
    fp: defaultdict[str, int] = defaultdict(int)
    fn: defaultdict[str, int] = defaultdict(int)
    support: defaultdict[str, int] = defaultdict(int)
    subset_errors: list[Prediction] = []

    for prediction in predictions:
        for label in prediction.expected & prediction.predicted:
            tp[label] += 1
        for label in prediction.predicted - prediction.expected:
            fp[label] += 1
        for label in prediction.expected - prediction.predicted:
            fn[label] += 1
        for label in prediction.expected:
            support[label] += 1
        if prediction.expected != prediction.predicted:
            subset_errors.append(prediction)

    per_label: dict[str, dict[str, float]] = {}
    f1_values: list[float] = []
    for label in labels:
        precision = tp[label] / (tp[label] + fp[label]) if tp[label] + fp[label] else 0.0
        recall = tp[label] / support[label] if support[label] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": float(support[label]),
        }

    total_tp = sum(tp.values())
    total_fp = sum(fp.values())
    total_fn = sum(fn.values())
    micro_denominator = 2 * total_tp + total_fp + total_fn
    micro_f1 = 2 * total_tp / micro_denominator if micro_denominator else 0.0
    exact = 1 - len(subset_errors) / len(predictions) if predictions else 0.0
    hamming = (
        float(
            np.mean([len(prediction.expected ^ prediction.predicted) for prediction in predictions])
        )
        / len(labels)
        if predictions and labels
        else 0.0
    )
    mean_label_recall = (
        float(
            np.mean(
                [
                    len(prediction.expected & prediction.predicted) / len(prediction.expected)
                    for prediction in predictions
                    if prediction.expected
                ]
            )
        )
        if predictions
        else 0.0
    )
    multilabel = [prediction for prediction in predictions if len(prediction.expected) > 1]
    multilabel_exact = (
        sum(prediction.expected == prediction.predicted for prediction in multilabel)
        / len(multilabel)
        if multilabel
        else 0.0
    )

    return {
        "examples": len(predictions),
        "exact_match": exact,
        "micro_f1": micro_f1,
        "macro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "hamming_loss": hamming,
        "mean_label_recall": mean_label_recall,
        "multilabel_examples": len(multilabel),
        "multilabel_exact_match": multilabel_exact,
        "per_label": per_label,
        "subset_errors": len(subset_errors),
        "fallbacks": sum(1 for prediction in predictions if prediction.fallback),
    }


def _confusion(predictions: list[Prediction], labels: list[str]) -> list[list[int]]:
    index = {label: position for position, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for prediction in predictions:
        if len(prediction.expected) != 1:
            continue
        expected = next(iter(prediction.expected))
        matrix[index[expected]][index[prediction.top_label]] += 1
    return matrix


def _one_vs_rest(predictions: list[Prediction], labels: list[str]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for label in labels:
        tp = fp = fn = tn = 0
        for prediction in predictions:
            positive_expected = label in prediction.expected
            positive_predicted = label in prediction.predicted
            if positive_expected and positive_predicted:
                tp += 1
            elif positive_expected:
                fn += 1
            elif positive_predicted:
                fp += 1
            else:
                tn += 1
        result[label] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
    return result


def _error_taxonomy(predictions: list[Prediction]) -> dict[str, int]:
    counts = Counter({"wrong_single": 0, "missed_label": 0, "extra_label": 0, "partial": 0})
    for prediction in predictions:
        if prediction.expected == prediction.predicted:
            continue
        if len(prediction.expected) == 1 and len(prediction.predicted) == 1:
            counts["wrong_single"] += 1
        elif (
            prediction.expected - prediction.predicted
            and not prediction.predicted - prediction.expected
        ):
            counts["missed_label"] += 1
        elif (
            prediction.predicted - prediction.expected
            and not prediction.expected - prediction.predicted
        ):
            counts["extra_label"] += 1
        else:
            counts["partial"] += 1
    return dict(counts)


def _confidence_stats(predictions: list[Prediction]) -> dict[str, float]:
    correct = [p.top_confidence for p in predictions if p.expected == p.predicted]
    wrong = [p.top_confidence for p in predictions if p.expected != p.predicted]
    return {
        "correct_mean": float(np.mean(correct)) if correct else 0.0,
        "correct_median": float(np.median(correct)) if correct else 0.0,
        "wrong_mean": float(np.mean(wrong)) if wrong else 0.0,
        "wrong_median": float(np.median(wrong)) if wrong else 0.0,
        "wrong_high_confidence": float(sum(1 for value in wrong if value >= 0.9)),
    }


def _pair_counts(predictions: list[Prediction]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for prediction in predictions:
        if len(prediction.expected) == 1 and len(prediction.predicted) == 1:
            expected = next(iter(prediction.expected))
            predicted = next(iter(prediction.predicted))
            if expected != predicted:
                counts[f"{expected} -> {predicted}"] += 1
    return counts


def _write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


def _save_plots(
    output: Path,
    labels: list[str],
    metrics: dict[str, Any],
    matrix: list[list[int]],
    predictions: list[Prediction],
    pairs: Counter[str],
    training: dict[str, list[tuple[float, float]]],
) -> None:
    plots = output / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = [epoch for epoch, _ in training["eval_loss"]]
    axes[0].plot(epochs, [loss for _, loss in training["eval_loss"]], marker="o")
    axes[0].set_title("Validation loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)
    axes[1].plot(
        epochs, [value for _, value in training["eval_micro"]], marker="o", label="micro F1"
    )
    axes[1].plot(
        epochs, [value for _, value in training["eval_macro"]], marker="s", label="macro F1"
    )
    axes[1].plot(
        epochs, [value for _, value in training["eval_exact"]], marker="^", label="exact match"
    )
    axes[1].set_title("Validation metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0.7, 0.95)
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(plots / "validation_curves.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 4))
    steps = [step for step, _ in training["train_loss"]]
    losses = [loss for _, loss in training["train_loss"]]
    axis.plot(steps, losses, linewidth=0.6, alpha=0.35, label="train loss (per log)")
    window = 50
    if len(losses) >= window:
        smoothed = np.convolve(losses, np.ones(window) / window, mode="valid")
        axis.plot(steps[window - 1 :], smoothed, linewidth=2, label=f"moving average ({window})")
    axis.set_title("Training loss")
    axis.set_xlabel("Step")
    axis.set_ylabel("Loss")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(plots / "training_loss.png", dpi=150)
    plt.close(figure)

    names = [label for label in labels]
    values = [metrics["per_label"][label]["f1"] for label in names]
    figure, axis = plt.subplots(figsize=(8, 4))
    bars = axis.bar(names, values, color="#4c72b0")
    axis.bar_label(bars, fmt="%.3f", padding=2)
    axis.set_ylim(0, 1.05)
    axis.set_title("Per-label F1 (held-out)")
    axis.set_ylabel("F1")
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(plots / "per_label_f1.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xlabel("Predicted (top-1)")
    axis.set_ylabel("Expected")
    axis.set_title("Confusion matrix (single-label rows)")
    for row in range(len(labels)):
        for column in range(len(labels)):
            value = matrix[row][column]
            if value:
                axis.text(
                    column,
                    row,
                    str(value),
                    ha="center",
                    va="center",
                    color="white" if value > max(max(r) for r in matrix) / 2 else "black",
                )
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(plots / "confusion_matrix.png", dpi=150)
    plt.close(figure)

    normalized = np.asarray(matrix, dtype=float)
    row_sums = normalized.sum(axis=1, keepdims=True)
    normalized = np.divide(normalized, row_sums, out=np.zeros_like(normalized), where=row_sums > 0)
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xlabel("Predicted (top-1)")
    axis.set_ylabel("Expected")
    axis.set_title("Confusion matrix, row-normalized")
    for row in range(len(labels)):
        for column in range(len(labels)):
            value = normalized[row][column]
            if value:
                axis.text(
                    column,
                    row,
                    f"{value:.1%}",
                    ha="center",
                    va="center",
                    color="white" if value > 0.5 else "black",
                )
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(plots / "confusion_matrix_normalized.png", dpi=150)
    plt.close(figure)

    error_rates = [
        1 - (matrix[index][index] / sum(matrix[index])) if sum(matrix[index]) else 0.0
        for index in range(len(labels))
    ]
    figure, axis = plt.subplots(figsize=(8, 4))
    bars = axis.bar(labels, [rate * 100 for rate in error_rates], color="#c44e52")
    axis.bar_label(bars, fmt="%.1f%%", padding=2)
    axis.set_ylim(0, max(error_rates) * 120 + 1)
    axis.set_title("Top-1 error rate by expected category")
    axis.set_ylabel("Error rate, %")
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(plots / "error_rate_by_category.png", dpi=150)
    plt.close(figure)

    correct = [p.top_confidence for p in predictions if p.expected == p.predicted]
    wrong = [p.top_confidence for p in predictions if p.expected != p.predicted]
    figure, axis = plt.subplots(figsize=(8, 4))
    bins = list(np.linspace(0, 1, 21))
    axis.hist(correct, bins=bins, alpha=0.6, label=f"correct (n={len(correct)})")
    axis.hist(wrong, bins=bins, alpha=0.7, label=f"error (n={len(wrong)})")
    axis.set_title("Top-1 confidence")
    axis.set_xlabel("Confidence")
    axis.set_ylabel("Examples")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(plots / "confidence_distribution.png", dpi=150)
    plt.close(figure)

    top_pairs = pairs.most_common(10)
    if top_pairs:
        figure, axis = plt.subplots(figsize=(9, 4))
        names = [name for name, _ in reversed(top_pairs)]
        counts = [count for _, count in reversed(top_pairs)]
        axis.barh(names, counts, color="#dd8452")
        axis.set_title("Top single-label confusions")
        axis.set_xlabel("Examples")
        axis.grid(axis="x", alpha=0.3)
        figure.tight_layout()
        figure.savefig(plots / "top_confusions.png", dpi=150)
        plt.close(figure)


_LOSS_PATTERN = re.compile(
    r"\{'loss': (?P<loss>[\d.]+), 'grad_norm': [\d.]+, "
    r"'learning_rate': [\d.e-]+, 'epoch': (?P<epoch>[\d.]+)\}"
)
_EVAL_PATTERN = re.compile(
    r"\{'eval_loss': (?P<loss>[\d.]+), 'eval_exact_match': (?P<exact>[\d.]+), "
    r"'eval_micro_f1': (?P<micro>[\d.]+),.*?'eval_macro_f1': (?P<macro>[\d.]+),"
    r".*?'epoch': (?P<epoch>[\d.]+)\}"
)


def _parse_training_log(path: Path) -> dict[str, list[tuple[float, float]]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    train_loss = [
        (float(m.group("epoch")), float(m.group("loss"))) for m in _LOSS_PATTERN.finditer(text)
    ]
    eval_rows = list(_EVAL_PATTERN.finditer(text))
    return {
        "train_loss": train_loss,
        "eval_loss": [(float(m.group("epoch")), float(m.group("loss"))) for m in eval_rows],
        "eval_micro": [(float(m.group("epoch")), float(m.group("micro"))) for m in eval_rows],
        "eval_macro": [(float(m.group("epoch")), float(m.group("macro"))) for m in eval_rows],
        "eval_exact": [(float(m.group("epoch")), float(m.group("exact"))) for m in eval_rows],
    }


def run(
    model_path: Path,
    dataset_path: Path,
    log_path: Path,
    output: Path,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).eval()
    labels = [model.config.id2label[index] for index in range(model.config.num_labels)]
    metadata = json.loads((model_path / "category_metadata.json").read_text(encoding="utf-8"))
    max_length = int(metadata.get("max_length", 64))
    thresholds = {
        str(label): float(value) for label, value in metadata.get("thresholds", {}).items()
    }

    examples = _test_examples(dataset_path)
    probabilities = _predict(model, tokenizer, [example.text for example in examples], max_length)
    predictions = _build_predictions(examples, probabilities, labels, thresholds)
    metrics = _metrics(predictions, labels)
    matrix = _confusion(predictions, labels)
    one_vs_rest = _one_vs_rest(predictions, labels)
    taxonomy = _error_taxonomy(predictions)
    confidence = _confidence_stats(predictions)
    pairs = _pair_counts(predictions)

    training: dict[str, list[tuple[float, float]]] = {
        "train_loss": [],
        "eval_loss": [],
        "eval_micro": [],
        "eval_macro": [],
        "eval_exact": [],
    }
    if log_path.exists():
        training = _parse_training_log(log_path)

    _write_csv(
        output / "per_label.csv",
        ["label", "precision", "recall", "f1", "support", "tp", "fp", "fn", "tn"],
        [
            [
                label,
                metrics["per_label"][label]["precision"],
                metrics["per_label"][label]["recall"],
                metrics["per_label"][label]["f1"],
                metrics["per_label"][label]["support"],
                one_vs_rest[label]["tp"],
                one_vs_rest[label]["fp"],
                one_vs_rest[label]["fn"],
                one_vs_rest[label]["tn"],
            ]
            for label in labels
        ],
    )
    _write_csv(
        output / "confusion_matrix.csv",
        ["expected"] + labels,
        [[label] + row for label, row in zip(labels, matrix, strict=True)],
    )
    _write_csv(
        output / "top_confusions.csv",
        ["expected_to_predicted", "count"],
        [[name, count] for name, count in pairs.most_common()],
    )
    _write_csv(
        output / "errors.csv",
        ["text", "expected", "predicted", "top1", "top1_confidence", "fallback", "scores"],
        [
            [
                prediction.text,
                "|".join(sorted(prediction.expected)),
                "|".join(sorted(prediction.predicted)),
                prediction.top_label,
                f"{prediction.top_confidence:.4f}",
                prediction.fallback,
                ";".join(
                    f"{label}:{score:.4f}"
                    for label, score in sorted(prediction.probabilities.items())
                ),
            ]
            for prediction in predictions
            if prediction.expected != prediction.predicted
        ],
    )
    _write_csv(
        output / "training_metrics.csv",
        ["series", "epoch", "value"],
        [[series, epoch, value] for series, values in training.items() for epoch, value in values],
    )
    pair_stats: defaultdict[str, dict[str, int]] = defaultdict(
        lambda: {"support": 0, "recalled": 0, "exact": 0}
    )
    for prediction in predictions:
        if len(prediction.expected) < 2:
            continue
        key = "|".join(sorted(prediction.expected))
        pair_stats[key]["support"] += 1
        pair_stats[key]["recalled"] += len(prediction.expected & prediction.predicted)
        pair_stats[key]["exact"] += int(prediction.expected == prediction.predicted)
    _write_csv(
        output / "multilabel_pairs.csv",
        ["expected_labels", "support", "label_recall", "exact_match"],
        [
            [
                key,
                stats["support"],
                stats["recalled"] / (stats["support"] * len(key.split("|"))),
                stats["exact"] / stats["support"],
            ]
            for key, stats in sorted(pair_stats.items())
        ],
    )

    baseline_path = Path("categorize/models/baseline_metrics.json")
    baseline = (
        json.loads(baseline_path.read_text(encoding="utf-8")) if baseline_path.exists() else None
    )
    summary: dict[str, Any] = {
        "model": str(model_path),
        "dataset": str(dataset_path),
        "labels": labels,
        "thresholds": thresholds,
        "metrics": metrics,
        "one_vs_rest": one_vs_rest,
        "error_taxonomy": taxonomy,
        "confidence": confidence,
        "top_confusions": pairs.most_common(20),
        "baseline": baseline,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _save_plots(output, labels, metrics, matrix, predictions, pairs, training)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = run(args.model, args.dataset, args.log, args.output)
    print(json.dumps({key: summary[key] for key in ("metrics", "error_taxonomy")}, indent=2))


if __name__ == "__main__":
    main()
