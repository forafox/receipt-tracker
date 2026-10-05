import argparse
from pathlib import Path

from categorize.api import DEFAULT_DATASET_PATH, DEFAULT_MODEL_PATH, train_model
from categorize.classifier import CategoryClassifier
from categorize.dataset import load_training_examples
from categorize.eval import evaluate
from categorize.external_dataset import build_training_csv
from categorize.transformer import TransformerCategoryClassifier


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m categorize")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    train_parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)

    dataset_parser = subparsers.add_parser("build-dataset")
    dataset_parser.add_argument("--output", type=Path, default=DEFAULT_DATASET_PATH)
    dataset_parser.add_argument("--seed", type=Path, default=Path("dataset/category/items.csv"))
    dataset_parser.add_argument("--limit", type=int, default=2000)

    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    eval_parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)

    args = parser.parse_args()
    if args.command == "build-dataset":
        build_training_csv(args.output, args.seed, args.limit)
        return

    if args.command == "train":
        train_model(args.dataset, args.model)
        return

    if not args.model.exists():
        train_model(args.dataset, args.model)
    classifier: CategoryClassifier = TransformerCategoryClassifier.load(args.model)
    result = evaluate(classifier, load_training_examples(args.dataset))
    print(f"accuracy={result.accuracy:.3f} examples={result.examples}")


if __name__ == "__main__":
    main()
