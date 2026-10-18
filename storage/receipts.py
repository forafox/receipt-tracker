import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from datetime import datetime
from pathlib import Path
from typing import Protocol

from common.contracts import Receipt

SCHEMA = """
CREATE TABLE IF NOT EXISTS receipts (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    purchased_at INTEGER NOT NULL,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS receipts_user_time ON receipts (user_id, purchased_at);
"""
FINGERPRINT_MIGRATION = """
ALTER TABLE receipts ADD COLUMN fingerprint TEXT;
CREATE UNIQUE INDEX receipts_user_fingerprint ON receipts (user_id, fingerprint);
"""


class ReceiptStorage(Protocol):
    """Persistent storage of users' receipts."""

    def save(self, user_id: int, receipt: Receipt, purchased_at: datetime) -> bool:
        """Save a receipt unless the user already saved the same one.

        Args:
            user_id: Telegram user identifier.
            receipt: Parsed receipt.
            purchased_at: Timezone-aware time the receipt is counted at.

        Returns:
            True if the receipt was saved, False if it is a duplicate.
        """
        ...

    def list_between(self, user_id: int, start: datetime, end: datetime) -> list[Receipt]:
        """List a user's receipts counted within a time range.

        Args:
            user_id: Telegram user identifier.
            start: Inclusive timezone-aware range start.
            end: Exclusive timezone-aware range end.

        Returns:
            Receipts ordered by purchase time.
        """
        ...


class SqliteReceiptStorage:
    """Receipt storage backed by an SQLite file."""

    def __init__(self, path: Path) -> None:
        """Open the database and create the schema if needed.

        Args:
            path: Path to the SQLite database file.
        """
        self._path = path
        with self._connect() as connection:
            connection.executescript(SCHEMA)
            columns = {row[1] for row in connection.execute("PRAGMA table_info(receipts)")}
            if "fingerprint" not in columns:
                connection.executescript(FINGERPRINT_MIGRATION)

    def save(self, user_id: int, receipt: Receipt, purchased_at: datetime) -> bool:
        """Save a receipt unless the user already saved the same one.

        Receipts are considered the same when their store, date and total match. Receipts
        missing any of these fields are always saved.

        Args:
            user_id: Telegram user identifier.
            receipt: Parsed receipt.
            purchased_at: Timezone-aware time the receipt is counted at.

        Returns:
            True if the receipt was saved, False if it is a duplicate.
        """
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT OR IGNORE INTO receipts (user_id, purchased_at, payload, fingerprint)"
                " VALUES (?, ?, ?, ?)",
                (
                    user_id,
                    int(purchased_at.timestamp()),
                    receipt.model_dump_json(),
                    _fingerprint(receipt),
                ),
            )
        return cursor.rowcount == 1

    def list_between(self, user_id: int, start: datetime, end: datetime) -> list[Receipt]:
        """List a user's receipts counted within a time range.

        Args:
            user_id: Telegram user identifier.
            start: Inclusive timezone-aware range start.
            end: Exclusive timezone-aware range end.

        Returns:
            Receipts ordered by purchase time.
        """
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload FROM receipts"
                " WHERE user_id = ? AND purchased_at >= ? AND purchased_at < ?"
                " ORDER BY purchased_at, id",
                (user_id, int(start.timestamp()), int(end.timestamp())),
            ).fetchall()
        return [Receipt.model_validate_json(payload) for (payload,) in rows]

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        with closing(sqlite3.connect(self._path)) as connection, connection:
            yield connection


def _fingerprint(receipt: Receipt) -> str | None:
    if receipt.store is None or receipt.datetime is None or receipt.total is None:
        return None
    return f"{receipt.store}|{int(receipt.datetime.timestamp())}|{receipt.total:.2f}"
