from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from common.contracts import Receipt
from storage import SqliteReceiptStorage

OCTOBER = datetime(2026, 10, 1, tzinfo=UTC)
NOVEMBER = datetime(2026, 11, 1, tzinfo=UTC)


def test_saved_receipt_is_listed_unchanged(tmp_path: Path) -> None:
    storage = SqliteReceiptStorage(tmp_path / "receipts.sqlite")
    receipt = Receipt(store="ЛЕНТА", total=Decimal("99.90"), warnings=["datetime_not_found"])

    storage.save(1, receipt, datetime(2026, 10, 5, tzinfo=UTC))

    assert storage.list_between(1, OCTOBER, NOVEMBER) == [receipt]


def test_list_filters_by_user_and_range(tmp_path: Path) -> None:
    storage = SqliteReceiptStorage(tmp_path / "receipts.sqlite")
    storage.save(1, Receipt(store="first"), OCTOBER)
    storage.save(1, Receipt(store="next month"), NOVEMBER)
    storage.save(2, Receipt(store="other user"), OCTOBER)

    receipts = storage.list_between(1, OCTOBER, NOVEMBER)

    assert [receipt.store for receipt in receipts] == ["first"]


def test_data_survives_reopening(tmp_path: Path) -> None:
    path = tmp_path / "receipts.sqlite"
    SqliteReceiptStorage(path).save(1, Receipt(store="ЛЕНТА"), OCTOBER)

    assert len(SqliteReceiptStorage(path).list_between(1, OCTOBER, NOVEMBER)) == 1
