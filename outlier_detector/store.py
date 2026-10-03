"""SQLite log of fired alerts (not every scan), in table `alerts`.

The pre-1.0 engine wrote an incompatible `signals` table to the same default
file; using a new table lets both coexist instead of failing on upgrade.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import fields
from pathlib import Path
from types import TracebackType

from .models import Direction, Signal, Stage

LOGGER = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY,
    detected_at_ms INTEGER NOT NULL,
    bar_start_ms INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    stage TEXT NOT NULL,
    direction TEXT NOT NULL,
    price REAL NOT NULL,
    move_pct REAL NOT NULL,
    market_move_pct REAL NOT NULL,
    zscore REAL NOT NULL,
    rvol REAL NOT NULL,
    level REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS alerts_key_time ON alerts (symbol, direction, stage, detected_at_ms);
"""

# Table columns are named after the Signal fields, so the SQL follows the dataclass.
_COLUMNS = tuple(field.name for field in fields(Signal))
_INSERT = f"INSERT INTO alerts ({', '.join(_COLUMNS)}) VALUES ({', '.join(':' + name for name in _COLUMNS)})"
_SELECT_RECENT = f"SELECT {', '.join(_COLUMNS)} FROM alerts ORDER BY detected_at_ms DESC, id DESC LIMIT ?"
_SELECT_LAST_ALERTS = """
SELECT symbol, direction, stage, MAX(detected_at_ms)
FROM alerts
WHERE detected_at_ms >= ?
GROUP BY symbol, direction, stage
"""

AlertKey = tuple[str, Direction, Stage]


class SignalStore:
    def __init__(self, path: str | Path) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)
        LOGGER.debug("signal store opened at %s", path)

    def record(self, signal: Signal) -> None:
        params = {name: getattr(signal, name) for name in _COLUMNS}
        params["stage"] = signal.stage.value
        params["direction"] = signal.direction.value
        with self._conn:
            self._conn.execute(_INSERT, params)

    def last_alerts(self, since_ms: int) -> dict[AlertKey, int]:
        """Latest alert time per (symbol, direction, stage) at or after `since_ms`."""
        rows = self._conn.execute(_SELECT_LAST_ALERTS, (since_ms,)).fetchall()
        return {
            (symbol, Direction(direction), Stage(stage)): last_ms
            for symbol, direction, stage, last_ms in rows
        }

    def recent(self, limit: int = 50) -> list[Signal]:
        """Most recent signals, newest first."""
        rows = self._conn.execute(_SELECT_RECENT, (limit,)).fetchall()
        return [_to_signal(row) for row in rows]

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> SignalStore:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def _to_signal(row: sqlite3.Row) -> Signal:
    values = dict(zip(row.keys(), row, strict=True))
    values["stage"] = Stage(values["stage"])
    values["direction"] = Direction(values["direction"])
    return Signal(**values)
