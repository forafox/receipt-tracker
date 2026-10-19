"""Dataset loading and parsing for the categorize module.

PDF requirements:
- load_training_examples(path): reads CSV 'text,category' in UTF-8 with header
- Empty rows are skipped
- Returns list of TrainingExample
- category field can contain multi-label separated by '|' """
import csv
from pathlib import Path

from categorize.schemas import TrainingExample


def load_training_examples(path: Path) -> list[TrainingExample]:
    """Load training examples from a CSV file.
    
    CSV format (per PDF spec):
    text,category
    
    Multi-label categories are separated by '|' in the category field.
    Empty rows (missing text or category) are skipped.
    
    Args:
        path: Path to the CSV file (UTF-8 encoded, with header)
    
    Returns:
        List of TrainingExample instances
    """
    with path.open(encoding="utf-8", newline="") as file:
        rows = csv.DictReader(file)
        examples = [
            TrainingExample(
                text=row["text"].strip(),
                category=row["category"],
            )
            for row in rows
            if row.get("text") and row.get("category")
        ]
    return examples