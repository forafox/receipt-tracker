from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, cast

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from categorize.service import CategoryService
from categorize.training import train_transformer
from categorize.transformer import TransformerCategoryClassifier
from common import ClassifiedItem, Item, MonthlyReport, Receipt
from reports.monthly import build_monthly_report

DEFAULT_DATASET_PATH = Path("dataset/category/training.csv")
DEFAULT_MODEL_PATH = Path("models/category_transformer")


class ClassifyItemsRequest(BaseModel):
    items: list[Item] = Field(min_length=1)


class ClassifyItemsResponse(BaseModel):
    items: list[ClassifiedItem]


class TrainRequest(BaseModel):
    dataset_path: Path = DEFAULT_DATASET_PATH
    model_path: Path = DEFAULT_MODEL_PATH


class EvalResponse(BaseModel):
    accuracy: float
    examples: int


def train_model(
    dataset_path: Path = DEFAULT_DATASET_PATH, model_path: Path = DEFAULT_MODEL_PATH
) -> None:
    train_transformer(dataset_path=dataset_path, output_dir=model_path)


def load_service(model_path: Path = DEFAULT_MODEL_PATH) -> CategoryService:
    if not model_path.exists():
        train_model(model_path=model_path)
    return CategoryService(TransformerCategoryClassifier.load(model_path))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.category_service = load_service()
    yield


app = FastAPI(title="M3 Category Classifier", version="0.1.0", lifespan=lifespan)


def get_category_service() -> CategoryService:
    service = getattr(app.state, "category_service", None)
    if service is None:
        msg = "category service is not initialized"
        raise HTTPException(status_code=503, detail=msg)
    return cast(CategoryService, service)


CategoryServiceDependency = Annotated[CategoryService, Depends(get_category_service)]


@app.post("/classify", response_model=ClassifiedItem)
def classify_item(item: Item, service: CategoryServiceDependency) -> ClassifiedItem:
    return service.classify_item(item)


@app.post("/classify/items", response_model=ClassifyItemsResponse)
def classify_items(
    request: ClassifyItemsRequest,
    service: CategoryServiceDependency,
) -> ClassifyItemsResponse:
    return ClassifyItemsResponse(items=service.classify_items(request.items))


@app.post("/reports/monthly", response_model=MonthlyReport)
def monthly_report(
    receipts: list[Receipt],
    service: CategoryServiceDependency,
) -> MonthlyReport:
    classified_receipts = [
        receipt.model_copy(update={"items": service.classify_items(receipt.items)})
        for receipt in receipts
    ]
    return build_monthly_report(classified_receipts)


@app.post("/train", status_code=204)
def train(request: TrainRequest) -> None:
    try:
        train_model(request.dataset_path, request.model_path)
        app.state.category_service = load_service(request.model_path)
    except (OSError, ValueError, KeyError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
