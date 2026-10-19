"""FastAPI application for the categorize module.

PDF-specified endpoints:
- POST /classify         -> Item -> ClassifiedItem
- POST /classify/items   -> {items: [Item]} -> {items: [ClassifiedItem]}
- POST /classify/receipt -> Receipt (M2->M3) -> ClassifiedReceipt
- POST /train            -> {dataset_path, model_path} -> 204, service reloads

PDF contract table (from README):
| Method | Path | Body | Response |
| ----- | ---- | ---- | ----- |
| POST | /classify | Item | ClassifiedItem |
| POST | /classify/items | {items: [Item]} | {items: [ClassifiedItem]} |
| POST | /classify/receipt | Receipt (M2->M3) | ClassifiedReceipt |
| POST | /train | {dataset_path, model_path} | 204, service reloads |
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, cast

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from categorize.service import CategoryService
from categorize.training import train_transformer
from categorize.transformer import TransformerCategoryClassifier
from common import ClassifiedItem, ClassifiedReceipt, Item, Receipt

DEFAULT_DATASET_PATH = Path("categorize/data/training.csv")
DEFAULT_MODEL_PATH = Path("categorize/models/category_transformer")


class ClassifyItemsRequest(BaseModel):
    items: list[Item] = Field(min_length=1, description="List of product items to classify")


class ClassifyItemsResponse(BaseModel):
    items: list[ClassifiedItem]


class TrainRequest(BaseModel):
    dataset_path: Path = DEFAULT_DATASET_PATH
    model_path: Path = DEFAULT_MODEL_PATH


class EvalResponse(BaseModel):
    accuracy: float = Field(ge=0, le=1, description="Exact-match accuracy")
    examples: int = Field(ge=0, description="Number of examples")


def train_model(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
    epochs: float = 6.0,
) -> None:
    """Train the category transformer model.

    PDF: Trains and saves model to model_path. If model already exists,
    re-trains from the given dataset.

    Args:
        dataset_path: CSV with ``text,category`` rows.
        model_path: Directory where the trained model is saved.
        epochs: Number of training epochs.
    """
    train_transformer(dataset_path=dataset_path, output_dir=model_path, epochs=epochs)


def load_service(model_path: Path = DEFAULT_MODEL_PATH) -> CategoryService:
    """Load or train the category service model.

    PDF behavior:
    - If model_path exists, load TransformerCategoryClassifier from it
    - If model_path does not exist, train from DEFAULT_DATASET_PATH automatically
    - Return CategoryService wrapping the classifier
    """
    if not model_path.exists():
        train_model(model_path=model_path)
    if not model_path.exists():
        msg = "model could not be trained or loaded"
        raise RuntimeError(msg)
    return CategoryService(TransformerCategoryClassifier.load(model_path))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize the category service on startup."""
    app.state.category_service = load_service()
    yield


app = FastAPI(title="M3 Category Classifier", version="0.1.0", lifespan=lifespan)


def get_category_service() -> CategoryService:
    """Get the category service from app state."""
    service = getattr(app.state, "category_service", None)
    if service is None:
        msg = "category service is not initialized"
        raise HTTPException(status_code=503, detail=msg)
    return cast(CategoryService, service)


CategoryServiceDependency = Annotated[CategoryService, Depends(get_category_service)]


@app.post("/classify", response_model=ClassifiedItem)
def classify_item(item: Item, service: CategoryServiceDependency) -> ClassifiedItem:
    """Classify a single product item.

    PDF: POST /classify with Item body → ClassifiedItem response
    """
    return service.classify_item(item)


@app.post("/classify/items", response_model=ClassifyItemsResponse)
def classify_items(
    request: ClassifyItemsRequest,
    service: CategoryServiceDependency,
) -> ClassifyItemsResponse:
    """Classify multiple product items.

    PDF: POST /classify/items with {items: [Item]} → {items: [ClassifiedItem]}
    """
    return ClassifyItemsResponse(items=service.classify_items(request.items))


@app.post("/classify/receipt", response_model=ClassifiedReceipt)
def classify_receipt(
    receipt: Receipt,
    service: CategoryServiceDependency,
) -> ClassifiedReceipt:
    """Classify all items in a receipt (M2→M3 transformation).

    PDF: POST /classify/receipt with Receipt (extraction module output) → ClassifiedReceipt
    """
    return service.classify_receipt(receipt)


@app.post("/train", status_code=204)
def train(request: TrainRequest) -> None:
    """Train the category classifier and reload the service.

    PDF: POST /train with {dataset_path, model_path} → 204 status,
    service reloads automatically.

    Raises:
        400 if training fails (OSError, ValueError, KeyError)
    """
    try:
        train_model(dataset_path=request.dataset_path, model_path=request.model_path)
        app.state.category_service = load_service(request.model_path)
    except (OSError, ValueError, KeyError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
