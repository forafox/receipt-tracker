from __future__ import annotations

import functools
import inspect
import random
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
    train_dataset, eval_dataset = _stratified_split(dataset, test_size=0.2)
    train_dataset = _oversample_training_dataset(train_dataset)
    pos_weight = _compute_pos_weight(train_dataset)

    tokenized_train = train_dataset.map(
        lambda batch: tokenizer(batch["text"], truncation=True, max_length=max_length),
        batched=True,
    )
    tokenized_eval = eval_dataset.map(
        lambda batch: tokenizer(batch["text"], truncation=True, max_length=max_length),
        batched=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=len(labels),
        id2label=id_to_label,
        label2id=label_to_id,
        problem_type="multi_label_classification",
    )
    args = _build_training_arguments(output_dir, epochs)
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": args,
        "train_dataset": tokenized_train,
        "eval_dataset": tokenized_eval,
        "data_collator": DataCollatorWithPadding(tokenizer=tokenizer),
        "compute_metrics": functools.partial(_compute_metrics, label_names=labels),
    }
    if "processing_class" in _TRAINER_PARAMS:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = _WeightedTrainer(**trainer_kwargs, pos_weight=pos_weight)
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(output_dir)
    TransformerCategoryClassifier.save_metadata(output_dir, id_to_label, max_length)

    metrics = trainer.evaluate()
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
        },
        thresholds={"default": DEFAULT_THRESHOLD},
        metrics={
            key.removeprefix("eval_"): float(value)
            for key, value in metrics.items()
            if isinstance(value, (int, float))
        },
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
        "save_strategy": "no",
        "learning_rate": 5e-5,
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 8,
        "num_train_epochs": epochs,
        "weight_decay": 0.01,
        "load_best_model_at_end": False,
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
    """Split by whole label sets so multilabel rows are not torn apart."""
    label_sets = [
        tuple(index for index, value in enumerate(vector) if value >= 0.5)
        for vector in dataset["labels"]
    ]
    groups: defaultdict[tuple[int, ...], list[int]] = defaultdict(list)
    for index, label_set in enumerate(label_sets):
        groups[label_set].append(index)

    rng = random.Random(42)
    train_indices: list[int] = []
    test_indices: list[int] = []
    for label_set in sorted(groups):
        indices = groups[label_set]
        rng.shuffle(indices)
        if len(indices) < 2:
            train_indices.extend(indices)
            continue
        test_count = max(1, int(round(len(indices) * test_size)))
        test_indices.extend(indices[:test_count])
        train_indices.extend(indices[test_count:])

    # A rare label may sit in a one-example group; make sure it is evaluated.
    test_labels = {label for index in test_indices for label in label_sets[index]}
    all_labels = {label for label_set in groups for label in label_set}
    train_counts: dict[int, int] = defaultdict(int)
    for index in train_indices:
        for label in label_sets[index]:
            train_counts[label] += 1
    for label in sorted(all_labels - test_labels):
        movable = next(
            (index for index in train_indices if label in label_sets[index]),
            None,
        )
        if movable is not None and train_counts[label] > 1:
            train_indices.remove(movable)
            test_indices.append(movable)
            for moved_label in label_sets[movable]:
                train_counts[moved_label] -= 1
    return dataset.select(train_indices), dataset.select(test_indices)


def _compute_metrics(
    eval_pred: EvalPrediction, label_names: Sequence[str] = ()
) -> dict[str, float]:
    logits = np.asarray(eval_pred.predictions)
    labels = np.asarray(eval_pred.label_ids)
    probabilities = 1 / (1 + np.exp(-logits))
    predictions = (probabilities >= DEFAULT_THRESHOLD).astype(int)
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

    per_label_f1: list[float] = []
    metrics: dict[str, float] = {"exact_match": float(exact_match), "micro_f1": float(f1)}
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


def _example_labels(example: TrainingExample) -> list[str]:
    return [label.strip() for label in example.category.split("|") if label.strip()]


def _multi_hot(labels: list[str], label_to_id: dict[str, int]) -> list[float]:
    encoded = [0.0] * len(label_to_id)
    for label in labels:
        encoded[label_to_id[label]] = 1.0
    return encoded
