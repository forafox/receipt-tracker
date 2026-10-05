from __future__ import annotations

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


def train_transformer(
    dataset_path: Path,
    output_dir: Path,
    base_model: str = DEFAULT_BASE_MODEL,
    epochs: float = 6.0,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> None:
    examples = load_training_examples(dataset_path)
    labels = sorted({example.category for example in examples})
    if len(labels) < 2:
        msg = "at least two categories are required"
        raise ValueError(msg)

    label_to_id = {label: index for index, label in enumerate(labels)}
    id_to_label = {index: label for label, index in label_to_id.items()}
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenized = _build_dataset(examples, label_to_id).map(
        lambda batch: tokenizer(batch["text"], truncation=True, max_length=max_length),
        batched=True,
    )
    split = tokenized.train_test_split(test_size=0.2, seed=42)

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=len(labels),
        id2label=id_to_label,
        label2id=label_to_id,
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
        train_dataset=split["train"],
        eval_dataset=split["test"],
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
            "label": [label_to_id[example.category] for example in examples],
        }
    )


def _compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
    logits = np.asarray(eval_pred.predictions)
    labels = np.asarray(eval_pred.label_ids)
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": float((predictions == labels).mean())}
