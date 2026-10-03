from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest

from outlier_detector.models import Direction, Signal, Stage
from outlier_detector.store import SignalStore

BASE = Signal(
    symbol="SOLUSDT",
    stage=Stage.EARLY,
    direction=Direction.UP,
    bar_start_ms=1_700_000_000_000,
    detected_at_ms=1_700_000_100_000,
    price=152.34,
    move_pct=3.2,
    market_move_pct=0.4,
    zscore=4.1,
    rvol=5.3,
    level=150.1,
)


def at(ms: int, **changes: object) -> Signal:
    return replace(BASE, detected_at_ms=ms, **changes)


@pytest.fixture
def store() -> Iterator[SignalStore]:
    with SignalStore(":memory:") as signal_store:
        yield signal_store


def test_creates_schema_and_index_with_wal(tmp_path: Path) -> None:
    path = tmp_path / "signals.sqlite3"
    SignalStore(path).close()

    conn = sqlite3.connect(path)
    try:
        columns = [row[1] for row in conn.execute("PRAGMA table_info(alerts)")]
        indexes = conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'").fetchall()
        index_columns = [row[2] for row in conn.execute("PRAGMA index_info(alerts_key_time)")]
        journal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    finally:
        conn.close()

    assert columns == [
        "id",
        "detected_at_ms",
        "bar_start_ms",
        "symbol",
        "stage",
        "direction",
        "price",
        "move_pct",
        "market_move_pct",
        "zscore",
        "rvol",
        "level",
    ]
    assert ("alerts_key_time",) in indexes
    assert index_columns == ["symbol", "direction", "stage", "detected_at_ms"]
    assert journal_mode == "wal"


def test_creates_missing_parent_directory(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "dir" / "signals.sqlite3"
    with SignalStore(path) as store:
        store.record(BASE)
    assert path.is_file()


def test_record_and_recent_round_trip(store: SignalStore) -> None:
    older = at(1_000)
    newer = at(2_000, symbol="WIFUSDT", stage=Stage.CONFIRMED, direction=Direction.DOWN, zscore=-3.6)
    store.record(older)
    store.record(newer)

    recent = store.recent()

    assert recent == [newer, older]
    assert type(recent[0].stage) is Stage
    assert type(recent[0].direction) is Direction
    assert recent[0].stage is Stage.CONFIRMED
    assert recent[0].direction is Direction.DOWN


def test_recent_respects_limit_and_breaks_ties_by_insertion(store: SignalStore) -> None:
    first = at(5_000, symbol="AAAUSDT")
    second = at(5_000, symbol="BBBUSDT")
    store.record(at(1_000))
    store.record(first)
    store.record(second)

    assert store.recent(limit=2) == [second, first]


def test_recent_on_empty_store(store: SignalStore) -> None:
    assert store.recent() == []
    assert store.last_alerts(0) == {}


def test_last_alerts_returns_max_per_key(store: SignalStore) -> None:
    store.record(at(1_000))
    store.record(at(3_000))
    store.record(at(2_000))
    store.record(at(2_500, stage=Stage.CONFIRMED))
    store.record(at(4_000, direction=Direction.DOWN))
    store.record(at(1_500, symbol="ETHUSDT"))

    assert store.last_alerts(0) == {
        ("SOLUSDT", Direction.UP, Stage.EARLY): 3_000,
        ("SOLUSDT", Direction.UP, Stage.CONFIRMED): 2_500,
        ("SOLUSDT", Direction.DOWN, Stage.EARLY): 4_000,
        ("ETHUSDT", Direction.UP, Stage.EARLY): 1_500,
    }
    key = next(iter(store.last_alerts(0)))
    assert type(key[1]) is Direction
    assert type(key[2]) is Stage


def test_last_alerts_filters_by_since(store: SignalStore) -> None:
    store.record(at(1_000))
    store.record(at(2_000, symbol="ETHUSDT"))
    store.record(at(3_000, symbol="ETHUSDT"))

    assert store.last_alerts(2_000) == {("ETHUSDT", Direction.UP, Stage.EARLY): 3_000}
    assert store.last_alerts(3_001) == {}


def test_reopening_file_keeps_data(tmp_path: Path) -> None:
    path = tmp_path / "signals.sqlite3"
    with SignalStore(path) as store:
        store.record(at(1_000))
        store.record(at(2_000, stage=Stage.CONFIRMED))

    with SignalStore(str(path)) as reopened:
        assert reopened.recent() == [at(2_000, stage=Stage.CONFIRMED), at(1_000)]
        assert reopened.last_alerts(0)[("SOLUSDT", Direction.UP, Stage.CONFIRMED)] == 2_000


def test_context_manager_closes_connection() -> None:
    with SignalStore(":memory:") as store:
        store.record(BASE)
    with pytest.raises(sqlite3.ProgrammingError):
        store.recent()


def test_coexists_with_legacy_signals_table(tmp_path: Path) -> None:
    path = tmp_path / "signals.sqlite3"
    with sqlite3.connect(path) as conn:  # the pre-1.0 engine's schema
        conn.execute("CREATE TABLE signals (timestamp TEXT, ticker TEXT, composite_score REAL)")
        conn.execute("INSERT INTO signals VALUES ('2026-04-05', 'SOLUSDT', 1.2)")
    conn.close()
    with SignalStore(path) as store:
        store.record(BASE)
        assert store.last_alerts(since_ms=0)
        assert len(store.recent()) == 1
