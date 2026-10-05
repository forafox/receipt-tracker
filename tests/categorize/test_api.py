from fastapi.testclient import TestClient
from pytest import MonkeyPatch

import categorize.api
from categorize.api import app, get_category_service
from categorize.service import CategoryService
from common import CategoryPrediction


class FakeClassifier:
    def predict(self, item_name: str) -> CategoryPrediction:
        if "шампунь" in item_name:
            return CategoryPrediction(category="personal_care", confidence=0.9)
        return CategoryPrediction(category="food", confidence=0.9)


def test_classify_endpoint_returns_category(monkeypatch: MonkeyPatch) -> None:
    service = CategoryService(FakeClassifier())
    monkeypatch.setattr(categorize.api, "load_service", lambda: service)
    app.dependency_overrides[get_category_service] = lambda: service

    try:
        with TestClient(app) as client:
            response = client.post(
                "/classify",
                json={"name": "молоко цельное", "quantity": "1", "total": "89.90"},
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["category"] == "food"
    assert 0 <= body["confidence"] <= 1


def test_monthly_report_endpoint_classifies_and_aggregates(monkeypatch: MonkeyPatch) -> None:
    service = CategoryService(FakeClassifier())
    monkeypatch.setattr(categorize.api, "load_service", lambda: service)
    app.dependency_overrides[get_category_service] = lambda: service

    try:
        with TestClient(app) as client:
            response = client.post(
                "/reports/monthly",
                json=[
                    {
                        "receipt_id": "r-1",
                        "purchased_at": "2026-10-03T10:00:00+03:00",
                        "items": [
                            {"name": "шампунь", "quantity": "1", "total": "210"},
                            {"name": "хлеб", "quantity": "1", "total": "55"},
                        ],
                    }
                ],
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["year"] == 2026
    assert body["month"] == 10
    assert body["total"] == "265"
    assert {category["category"] for category in body["categories"]} == {"food", "personal_care"}


def test_classify_receipt_endpoint_is_m2_to_m3_contract(monkeypatch: MonkeyPatch) -> None:
    service = CategoryService(FakeClassifier())
    monkeypatch.setattr(categorize.api, "load_service", lambda: service)
    app.dependency_overrides[get_category_service] = lambda: service

    try:
        with TestClient(app) as client:
            response = client.post(
                "/classify/receipt",
                json={
                    "receipt_id": "r-1",
                    "purchased_at": "2026-10-03T10:00:00+03:00",
                    "items": [{"name": "молоко цельное", "quantity": "1", "total": "89.90"}],
                },
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["receipt_id"] == "r-1"
    assert body["items"][0]["category"] == "food"
    assert body["items"][0]["confidence"] == 0.9
