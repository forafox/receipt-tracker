from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import EvalPrediction

from categorize.dataset import load_training_examples
from categorize.schemas import TrainingExample
from categorize.transformer import TransformerCategoryClassifier

DEFAULT_BASE_MODEL = "cointegrated/rubert-tiny"
DEFAULT_MAX_LENGTH = 64
DEFAULT_THRESHOLD = 0.5


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
    args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="no",
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=epochs,
        weight_decay=0.01,
        load_best_model_at_end=False,
        logging_steps=5,
        report_to=[],
        seed=42,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=_compute_metrics,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(output_dir)
    TransformerCategoryClassifier.save_metadata(output_dir, id_to_label, max_length)


def _build_dataset(examples: list[TrainingExample], label_to_id: dict[str, int]) -> Dataset:
    return Dataset.from_dict(
        {
            "text": [example.text for example in examples],
            "labels": [_multi_hot(_example_labels(example), label_to_id) for example in examples],
        }
    )


def _oversample_training_dataset(dataset: Dataset) -> Dataset:
    labels = [_primary_label(label_vector) for label_vector in dataset["labels"]]
    counts = Counter(labels)
    target_count = max(counts.values())
    indices_by_label: defaultdict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        indices_by_label[label].append(index)

    balanced_indices: list[int] = []
    for label in sorted(indices_by_label):
        indices = indices_by_label[label]
        repeats = [indices[index % len(indices)] for index in range(target_count)]
        balanced_indices.extend(repeats)
    return dataset.select(balanced_indices)


def _stratified_split(dataset: Dataset, test_size: float) -> tuple[Dataset, Dataset]:
    labels = [_primary_label(label_vector) for label_vector in dataset["labels"]]
    indices_by_label: defaultdict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        indices_by_label[label].append(index)

    train_indices: list[int] = []
    test_indices: list[int] = []
    for label in sorted(indices_by_label):
        indices = indices_by_label[label]
        random.Random(42 + label).shuffle(indices)
        test_count = max(1, int(round(len(indices) * test_size)))
        test_indices.extend(indices[:test_count])
        train_indices.extend(indices[test_count:])
    return dataset.select(train_indices), dataset.select(test_indices)


def _compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
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
    return {
        "exact_match": float(exact_match),
        "micro_f1": float(f1),
    }


def _example_labels(example: TrainingExample) -> list[str]:
    return [label.strip() for label in example.category.split("|") if label.strip()]


def _multi_hot(labels: list[str], label_to_id: dict[str, int]) -> list[float]:
    encoded = [0.0] * len(label_to_id)
    for label in labels:
        encoded[label_to_id[label]] = 1.0
    return encoded


def _primary_label(label_vector: Iterable[float | int]) -> int:
    values = [float(value) for value in label_vector]
    return values.index(max(values))
