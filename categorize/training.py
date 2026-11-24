from __future__ import annotations

import functools
import inspect
import random
import shutil
from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset
from torch.nn import functional as F
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import EvalPrediction

from categorize.dataset import load_training_examples
from categorize.manifest import (
    MODEL_VERSION,
    ModelManifest,
    save_manifest,
    sha256_file,
)
from categorize.schemas import TrainingExample
from categorize.text import denoise
from categorize.transformer import DEFAULT_THRESHOLD, TransformerCategoryClassifier

# Compatibility shim: transformers 4.39 passes arguments such as
# ``dispatch_batches``/``even_batches`` to ``accelerate.Accelerator``; newer
# accelerate releases moved them into ``DataLoaderConfiguration``. Dropping the
# unsupported keywords keeps Trainer usable on that version pair.
try:  # pragma: no cover - trivial import guard
    from accelerate import Accelerator as _Accelerator

    _accelerator_init = _Accelerator.__init__
    _accelerator_params = inspect.signature(_accelerator_init).parameters
    _accelerator_accepts_kwargs = "kwargs" in _accelerator_params

    def _accelerator_init_without_unsupported(
        self: object, *args: object, **kwargs: object
    ) -> None:
        if not _accelerator_accepts_kwargs:
            for key in list(kwargs):
                if key not in _accelerator_params:
                    kwargs.pop(key)
        _accelerator_init(self, *args, **kwargs)

    _Accelerator.__init__ = _accelerator_init_without_unsupported
except ImportError:  # pragma: no cover - accelerate is a transformers dependency
    pass

DEFAULT_BASE_MODEL = "cointegrated/rubert-tiny"
DEFAULT_MAX_LENGTH = 64
MAX_OVERSAMPLE_TARGET = 5000
MAX_OVERSAMPLE_REPEAT = 4
_TRAINER_PARAMS = inspect.signature(Trainer.__init__).parameters
_TRAINING_ARGUMENT_PARAMS = inspect.signature(TrainingArguments.__init__).parameters


def train_transformer(
    dataset_path: Path,
    output_dir: Path,
    base_model: str = DEFAULT_BASE_MODEL,
    epochs: float = 6.0,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> None:
    examples = load_training_examples(dataset_path)
    labels = sorted({label for example in examples for label in _example_labels(example)})
    if len(labels) < 2:
        msg = "at least two categories are required"
        raise ValueError(msg)

    label_to_id = {label: index for index, label in enumerate(labels)}
    id_to_label = {index: label for label, index in label_to_id.items()}
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    dataset = _build_dataset(examples, label_to_id)
    train_pool, _test_dataset = _stratified_split(dataset, test_size=0.2)
    train_dataset, validation_dataset = _stratified_split(train_pool, test_size=0.1)
    train_dataset = _oversample_training_dataset(train_dataset)
    pos_weight = _compute_pos_weight(train_dataset)

    tokenized_train = train_dataset.map(
        lambda batch: tokenizer(
            [denoise(text) for text in batch["text"]],
            truncation=True,
            max_length=max_length,
        ),
        batched=True,
    )
    tokenized_validation = validation_dataset.map(
        lambda batch: tokenizer(
            [denoise(text) for text in batch["text"]],
            truncation=True,
            max_length=max_length,
        ),
        batched=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=len(labels),
        id2label=id_to_label,
        label2id=label_to_id,
        problem_type="multi_label_classification",
    )
    checkpoint_dir = output_dir / "_trainer"
    args = _build_training_arguments(checkpoint_dir, epochs)
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": args,
        "train_dataset": tokenized_train,
        "eval_dataset": tokenized_validation,
        "data_collator": DataCollatorWithPadding(tokenizer=tokenizer),
        "compute_metrics": functools.partial(_compute_metrics, label_names=labels),
    }
    if "processing_class" in _TRAINER_PARAMS:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = _WeightedTrainer(**trainer_kwargs, pos_weight=pos_weight)
    trainer.train()

    evaluation = trainer.evaluate()
    prediction = trainer.predict(tokenized_validation)
    probabilities = 1 / (1 + np.exp(-np.asarray(prediction.predictions)))
    label_ids = prediction.label_ids
    if label_ids is None:
        raise RuntimeError("trainer.predict returned no labels")
    thresholds = tune_thresholds(probabilities, np.asarray(label_ids), labels)
    metrics = _compute_metrics(
        EvalPrediction(predictions=prediction.predictions, label_ids=label_ids),
        label_names=labels,
        thresholds=[thresholds[label] for label in labels],
    )
    metrics["eval_loss"] = float(evaluation.get("eval_loss", 0.0))

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(output_dir)
    shutil.rmtree(checkpoint_dir, ignore_errors=True)
    TransformerCategoryClassifier.save_metadata(
        output_dir,
        id_to_label,
        max_length,
        thresholds=thresholds,
    )

    manifest = ModelManifest(
        model_version=MODEL_VERSION,
        base_model=base_model,
        created_at=datetime.now(UTC).isoformat(),
        dataset_path=str(dataset_path),
        dataset_sha256=sha256_file(dataset_path),
        hyperparameters={
            "epochs": epochs,
            "max_length": max_length,
            "learning_rate": 5e-5,
            "per_device_train_batch_size": 8,
            "seed": 42,
            "best_checkpoint": "eval_loss",
        },
        thresholds=thresholds,
        metrics={key: float(value) for key, value in metrics.items()},
        label_names=labels,
    )
    save_manifest(output_dir, manifest)


def _build_training_arguments(output_dir: Path, epochs: float) -> TrainingArguments:
    """Build ``TrainingArguments`` across transformers versions.

    ``evaluation_strategy`` was renamed to ``eval_strategy`` in transformers
    4.41, so the argument name is chosen by inspecting the installed version.
    """
    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "learning_rate": 5e-5,
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 8,
        "num_train_epochs": epochs,
        "weight_decay": 0.01,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "save_strategy": "epoch",
        "save_total_limit": 1,
        "warmup_ratio": 0.06,
        "logging_steps": 5,
        "report_to": [],
        "seed": 42,
    }
    if "eval_strategy" in _TRAINING_ARGUMENT_PARAMS:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"
    return TrainingArguments(**kwargs)


class _WeightedTrainer(Trainer):
    """Trainer with per-label positive weighting for imbalanced multilabel data."""

    def __init__(self, *args: Any, pos_weight: torch.Tensor | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight

    def compute_loss(  # type: ignore[override]
        self,
        model: Any,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        **kwargs: Any,
    ) -> Any:
        labels = inputs.pop("labels").float()
        outputs = model(**inputs)
        weight = self.pos_weight
        loss = F.binary_cross_entropy_with_logits(
            outputs.logits,
            labels,
            pos_weight=weight.to(outputs.logits.device) if weight is not None else None,
        )
        return (loss, outputs) if return_outputs else loss


def _compute_pos_weight(dataset: Dataset, cap: float = 5.0) -> torch.Tensor:
    labels = np.asarray(dataset["labels"], dtype=float)
    positive = labels.sum(axis=0)
    negative = labels.shape[0] - positive
    weight = np.divide(negative, positive, out=np.ones_like(positive), where=positive > 0)
    return torch.tensor(np.clip(weight, 1.0, cap), dtype=torch.float32)


def _build_dataset(examples: list[TrainingExample], label_to_id: dict[str, int]) -> Dataset:
    return Dataset.from_dict(
        {
            "text": [example.text for example in examples],
            "labels": [_multi_hot(_example_labels(example), label_to_id) for example in examples],
            "group_id": [
                example.group_id or example.text.casefold().strip() for example in examples
            ],
        }
    )


def _oversample_training_dataset(dataset: Dataset) -> Dataset:
    """Balance every label without over-repeating a tiny class.

    Each example is repeated by the factor needed to bring its rarest label up
    to the majority count, capped at ``MAX_OVERSAMPLE_REPEAT`` passes.  The cap
    prevents a class such as ``cafe`` (a few hundred texts) from being copied
    thousands of times and memorised verbatim.
    """
    vectors = np.asarray(dataset["labels"], dtype=float)
    counts = vectors.sum(axis=0)
    target = min(float(counts.max()), MAX_OVERSAMPLE_TARGET)
    indices: list[int] = []
    for index, vector in enumerate(vectors):
        present = counts[vector >= 0.5]
        factor = int(round(target / present.min())) if present.size else 1
        repeats = max(1, min(MAX_OVERSAMPLE_REPEAT, factor))
        indices.extend([index] * repeats)
    return dataset.select(indices)


def _stratified_split(dataset: Dataset, test_size: float) -> tuple[Dataset, Dataset]:
    """Split by groups and whole label sets.

    Rows that share a ``group_id`` (clean/noisy/short variants of the same
    product) always land in the same fold.  Groups are stratified by the union
    of their label sets so a multilabel row is never torn apart and every label
    is represented in the evaluation fold.
    """
    vectors = [tuple(row) for row in dataset["labels"]]
    label_sets = [
        tuple(index for index, value in enumerate(vector) if value >= 0.5) for vector in vectors
    ]
    group_keys = (
        list(dataset["group_id"])
        if "group_id" in dataset.column_names
        else [str(index) for index in range(len(dataset))]
    )
    groups: defaultdict[str, list[int]] = defaultdict(list)
    for index, key in enumerate(group_keys):
        groups[str(key)].append(index)

    signatures: dict[str, tuple[int, ...]] = {}
    for key, indices in groups.items():
        union = {label for index in indices for label in label_sets[index]}
        signatures[key] = tuple(sorted(union))

    by_signature: defaultdict[tuple[int, ...], list[str]] = defaultdict(list)
    for key, signature in signatures.items():
        by_signature[signature].append(key)

    rng = random.Random(42)
    train_groups: set[str] = set()
    test_groups: set[str] = set()
    for signature in sorted(by_signature):
        keys = sorted(by_signature[signature])
        rng.shuffle(keys)
        if len(keys) < 2:
            train_groups.update(keys)
            continue
        test_count = max(1, int(round(len(keys) * test_size)))
        test_groups.update(keys[:test_count])
        train_groups.update(keys[test_count:])

    # A rare label may sit in a one-group signature; make sure it is evaluated.
    test_labels = {label for key in test_groups for label in signatures[key]}
    all_labels = {label for signature in signatures.values() for label in signature}
    for label in sorted(all_labels - test_labels):
        candidates = [key for key in sorted(train_groups) if label in signatures[key]]
        if len(candidates) > 1:
            moved = candidates[0]
            train_groups.remove(moved)
            test_groups.add(moved)

    train_indices = [index for key in sorted(train_groups) for index in groups[key]]
    test_indices = [index for key in sorted(test_groups) for index in groups[key]]
    return dataset.select(train_indices), dataset.select(test_indices)


def _compute_metrics(
    eval_pred: EvalPrediction,
    label_names: Sequence[str] = (),
    thresholds: Sequence[float] | None = None,
) -> dict[str, float]:
    logits = np.asarray(eval_pred.predictions)
    labels = np.asarray(eval_pred.label_ids)
    probabilities = 1 / (1 + np.exp(-logits))
    cutoffs = np.asarray(thresholds if thresholds is not None else [DEFAULT_THRESHOLD])
    if cutoffs.size == 1:
        cutoffs = np.full(labels.shape[1], float(cutoffs[0]))
    predictions = (probabilities >= cutoffs).astype(int)
    for index, row in enumerate(predictions):
        if not row.any():
            row[int(probabilities[index].argmax())] = 1
    exact_match = (predictions == labels).all(axis=1).mean()
    true_positive = float((predictions * labels).sum())
    predicted_positive = float(predictions.sum())
    actual_positive = float(labels.sum())
    precision = true_positive / predicted_positive if predicted_positive else 0.0
    recall = true_positive / actual_positive if actual_positive else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    hamming = float((predictions != labels).mean())
    mean_label_recall = float(
        sum(
            float((row * target).sum() / target.sum()) if target.sum() else 0.0
            for row, target in zip(predictions, labels, strict=True)
        )
        / max(1, len(predictions))
    )

    per_label_f1: list[float] = []
    metrics: dict[str, float] = {
        "exact_match": float(exact_match),
        "micro_f1": float(f1),
        "hamming_loss": hamming,
        "mean_label_recall": mean_label_recall,
    }
    for column in range(labels.shape[1]):
        tp = float((predictions[:, column] * labels[:, column]).sum())
        predicted = float(predictions[:, column].sum())
        actual = float(labels[:, column].sum())
        column_precision = tp / predicted if predicted else 0.0
        column_recall = tp / actual if actual else 0.0
        column_f1 = (
            2 * column_precision * column_recall / (column_precision + column_recall)
            if column_precision + column_recall
            else 0.0
        )
        per_label_f1.append(column_f1)
        name = label_names[column] if column < len(label_names) else str(column)
        metrics[f"precision_{name}"] = float(column_precision)
        metrics[f"recall_{name}"] = float(column_recall)
        metrics[f"f1_{name}"] = float(column_f1)
    metrics["macro_f1"] = float(np.mean(per_label_f1)) if per_label_f1 else 0.0
    return metrics


def tune_thresholds(
    probabilities: np.ndarray,
    labels: np.ndarray,
    label_names: Sequence[str],
    minimum_support: int = 5,
) -> dict[str, float]:
    """Pick a per-label sigmoid cutoff that maximises F1 on validation rows."""
    thresholds: dict[str, float] = {}
    for column, name in enumerate(label_names):
        target = labels[:, column] >= 0.5
        if int(target.sum()) < minimum_support:
            thresholds[name] = DEFAULT_THRESHOLD
            continue
        best_threshold = DEFAULT_THRESHOLD
        best_f1 = -1.0
        for cutoff in np.arange(0.1, 0.901, 0.02):
            predicted = probabilities[:, column] >= cutoff
            true_positive = float((predicted & target).sum())
            predicted_positive = float(predicted.sum())
            actual_positive = float(target.sum())
            precision = true_positive / predicted_positive if predicted_positive else 0.0
            recall = true_positive / actual_positive if actual_positive else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = float(cutoff)
        thresholds[name] = round(best_threshold, 2)
    return thresholds


def _example_labels(example: TrainingExample) -> list[str]:
    return [label.strip() for label in example.category.split("|") if label.strip()]


def _multi_hot(labels: list[str], label_to_id: dict[str, int]) -> list[float]:
    encoded = [0.0] * len(label_to_id)
    for label in labels:
        encoded[label_to_id[label]] = 1.0
    return encoded
