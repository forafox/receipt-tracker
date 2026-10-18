import sqlite3
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


FULL_RECEIPT = Receipt(
    store="ЛЕНТА", datetime=datetime(2026, 10, 5, 12, tzinfo=UTC), total=Decimal("99.90")
)


def test_same_receipt_is_saved_once_per_user(tmp_path: Path) -> None:
    storage = SqliteReceiptStorage(tmp_path / "receipts.sqlite")

    assert storage.save(1, FULL_RECEIPT, OCTOBER) is True
    assert storage.save(1, FULL_RECEIPT, OCTOBER) is False
    assert storage.save(2, FULL_RECEIPT, OCTOBER) is True
    assert len(storage.list_between(1, OCTOBER, NOVEMBER)) == 1


def test_receipts_with_missing_fields_are_never_duplicates(tmp_path: Path) -> None:
    storage = SqliteReceiptStorage(tmp_path / "receipts.sqlite")
    receipt = Receipt(store="ЛЕНТА", total=Decimal("99.90"))

    assert storage.save(1, receipt, OCTOBER) is True
    assert storage.save(1, receipt, OCTOBER) is True


def test_existing_database_is_migrated(tmp_path: Path) -> None:
    path = tmp_path / "receipts.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE receipts (id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL,"
            " purchased_at INTEGER NOT NULL, payload TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO receipts (user_id, purchased_at, payload) VALUES (1, ?, ?)",
            (int(OCTOBER.timestamp()), Receipt(store="old").model_dump_json()),
        )
    connection.close()

    storage = SqliteReceiptStorage(path)

    assert storage.save(1, FULL_RECEIPT, OCTOBER) is True
    assert storage.save(1, FULL_RECEIPT, OCTOBER) is False
    assert len(storage.list_between(1, OCTOBER, NOVEMBER)) == 2


def test_list_recent_returns_newest_first(tmp_path: Path) -> None:
    storage = SqliteReceiptStorage(tmp_path / "receipts.sqlite")
    storage.save(1, Receipt(store="old"), OCTOBER)
    storage.save(1, Receipt(store="new"), NOVEMBER)
    storage.save(1, Receipt(store="same time, saved later"), NOVEMBER)
    storage.save(2, Receipt(store="other user"), NOVEMBER)

    receipts = storage.list_recent(1, limit=2)

    assert [receipt.store for receipt in receipts] == ["same time, saved later", "new"]
