from __future__ import annotations

import asyncio
import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class SignalRecord:
    timestamp: str
    stage: str
    signal_kind: str
    ticker: str
    momentum_z: float
    curvature: float
    hurst: float
    regime_score: int
    composite_score: float
    alerted: bool
    price: float
    rank: int
    persistence_hits: int
    dom_falling: bool
    dom_state: str
    dom_change_pct: float


class SignalDatabase:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.execute("PRAGMA busy_timeout=5000;")

    async def initialize(self) -> None:
        await asyncio.to_thread(self._initialize_sync)

    def _initialize_sync(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS signals (
                timestamp TEXT NOT NULL,
                stage TEXT NOT NULL DEFAULT 'confirmed',
                signal_kind TEXT NOT NULL DEFAULT 'confirmed',
                ticker TEXT NOT NULL,
                momentum_z REAL NOT NULL,
                curvature REAL NOT NULL,
                hurst REAL NOT NULL,
                regime_score INTEGER NOT NULL,
                composite_score REAL NOT NULL,
                alerted INTEGER NOT NULL,
                price REAL NOT NULL DEFAULT 0,
                rank INTEGER NOT NULL DEFAULT 0,
                persistence_hits INTEGER NOT NULL DEFAULT 0,
                dom_falling INTEGER NOT NULL DEFAULT 0,
                dom_state TEXT NOT NULL DEFAULT 'neutral',
                dom_change_pct REAL NOT NULL DEFAULT 0
            )
            """
        )
        existing_columns = {
            row[1] for row in self._conn.execute("PRAGMA table_info(signals)")
        }
        if "stage" not in existing_columns:
            self._conn.execute(
                "ALTER TABLE signals ADD COLUMN stage TEXT NOT NULL DEFAULT 'confirmed'"
            )
        if "signal_kind" not in existing_columns:
            self._conn.execute(
                "ALTER TABLE signals ADD COLUMN signal_kind TEXT NOT NULL DEFAULT 'confirmed'"
            )
        if "price" not in existing_columns:
            self._conn.execute("ALTER TABLE signals ADD COLUMN price REAL NOT NULL DEFAULT 0")
        if "rank" not in existing_columns:
            self._conn.execute("ALTER TABLE signals ADD COLUMN rank INTEGER NOT NULL DEFAULT 0")
        if "persistence_hits" not in existing_columns:
            self._conn.execute(
                "ALTER TABLE signals ADD COLUMN persistence_hits INTEGER NOT NULL DEFAULT 0"
            )
        if "dom_falling" not in existing_columns:
            self._conn.execute(
                "ALTER TABLE signals ADD COLUMN dom_falling INTEGER NOT NULL DEFAULT 0"
            )
        if "dom_state" not in existing_columns:
            self._conn.execute(
                "ALTER TABLE signals ADD COLUMN dom_state TEXT NOT NULL DEFAULT 'neutral'"
            )
        if "dom_change_pct" not in existing_columns:
            self._conn.execute(
                "ALTER TABLE signals ADD COLUMN dom_change_pct REAL NOT NULL DEFAULT 0"
            )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS summary_cycles (
                stage TEXT NOT NULL,
                cycle_time_ms INTEGER NOT NULL,
                PRIMARY KEY (stage, cycle_time_ms)
            )
            """
        )
        self._conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_signals_ticker_timestamp
            ON signals (ticker, timestamp)
            """
        )
        self._conn.commit()

    async def log_signal(self, record: SignalRecord) -> None:
        await asyncio.to_thread(self._log_signal_sync, record)

    async def log_signals(self, records: list[SignalRecord]) -> None:
        if not records:
            return
        await asyncio.to_thread(self._log_signals_sync, records)

    def _log_signal_sync(self, record: SignalRecord) -> None:
        self._log_signals_sync([record])

    def _log_signals_sync(self, records: list[SignalRecord]) -> None:
        self._conn.execute(
            "BEGIN"
        )
        self._conn.executemany(
            """
            INSERT INTO signals (
                timestamp,
                stage,
                signal_kind,
                ticker,
                momentum_z,
                curvature,
                hurst,
                regime_score,
                composite_score,
                alerted,
                price,
                rank,
                persistence_hits,
                dom_falling,
                dom_state,
                dom_change_pct
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    record.timestamp,
                    record.stage,
                    record.signal_kind,
                    record.ticker,
                    record.momentum_z,
                    record.curvature,
                    record.hurst,
                    record.regime_score,
                    record.composite_score,
                    int(record.alerted),
                    record.price,
                    record.rank,
                    record.persistence_hits,
                    int(record.dom_falling),
                    record.dom_state,
                    record.dom_change_pct,
                )
                for record in records
            ],
        )
        self._conn.commit()

    async def record_summary_cycle(self, stage: str, cycle_time_ms: int) -> bool:
        return await asyncio.to_thread(self._record_summary_cycle_sync, stage, cycle_time_ms)

    def _record_summary_cycle_sync(self, stage: str, cycle_time_ms: int) -> bool:
        cursor = self._conn.execute(
            """
            INSERT OR IGNORE INTO summary_cycles (stage, cycle_time_ms)
            VALUES (?, ?)
            """,
            (stage, cycle_time_ms),
        )
        self._conn.commit()
        return cursor.rowcount == 1

    def close(self) -> None:
        self._conn.close()
