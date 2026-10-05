import csv
from pathlib import Path

from categorize.schemas import TrainingExample


def load_training_examples(path: Path) -> list[TrainingExample]:
    with path.open(encoding="utf-8", newline="") as file:
        rows = csv.DictReader(file)
        return [
            TrainingExample(text=row["text"], category=row["category"])
            for row in rows
            if row.get("text") and row.get("category")
        ]
