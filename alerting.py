from __future__ import annotations

from dataclasses import dataclass

import aiohttp


@dataclass(slots=True)
class AlertPayload:
    stage: str
    signal_kind: str
    ticker: str
    composite_score: float
    momentum_z: float
    curvature: float
    hurst: float
    current_price: float
    regime_score: int
    rank: int | None = None
    persistence_hits: int | None = None
    persistence_window: int | None = None


@dataclass(slots=True)
class SummaryEntry:
    ticker: str
    rank: int
    composite_score: float
    momentum_z: float
    curvature_z: float
    hurst: float


@dataclass(slots=True)
class SummaryPayload:
    stage: str
    cycle_time_ms: int
    regime_score: int
    dom_falling: bool
    top_rankings: list[SummaryEntry]
    bottom_rankings: list[SummaryEntry]
    qualified_signals: list[str]


class TelegramNotifier:
    def __init__(
        self,
        session: aiohttp.ClientSession,
        bot_token: str | None,
        chat_id: str | None,
    ) -> None:
        self.session = session
        self.bot_token = bot_token
        self.chat_id = chat_id

    @property
    def enabled(self) -> bool:
        return bool(self.bot_token and self.chat_id)

    async def _send_text(self, text: str) -> bool:
        if not self.enabled:
            return False
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        async with self.session.post(
            url,
            json={"chat_id": self.chat_id, "text": text},
            timeout=aiohttp.ClientTimeout(total=10),
        ) as response:
            response.raise_for_status()
        return True

    async def send(self, payload: AlertPayload) -> bool:
        labels = {
            "watchlist": "watchlist candidate",
            "emerging": "emerging breakout candidate",
            "confirmed": "confirmed breakout candidate",
            "confirmed_strong": "persistent confirmed breakout candidate",
        }
        label = labels.get(payload.signal_kind, "signal candidate")
        persistence_line = ""
        if payload.persistence_hits is not None and payload.persistence_window is not None:
            persistence_line = (
                f"Persistence: {payload.persistence_hits}/{payload.persistence_window}\n"
            )
        text = (
            f"{payload.ticker} {label}\n"
            f"Stage: {payload.stage.upper()}\n"
            f"Signal: {payload.signal_kind.upper()}\n"
            f"Composite: {payload.composite_score:.2f}\n"
            f"Momentum z: {payload.momentum_z:.2f}\n"
            f"Curvature: {payload.curvature:.6f}\n"
            f"Hurst: {payload.hurst:.3f}\n"
            f"Price: {payload.current_price:.6f}\n"
            f"BTC regime: {payload.regime_score}\n"
            f"{persistence_line}"
            f"Rank: {payload.rank if payload.rank is not None else 'n/a'}"
        )
        return await self._send_text(text)

    async def send_summary(self, payload: SummaryPayload) -> bool:
        top_lines = "\n".join(
            f"#{entry.rank} {entry.ticker} score={entry.composite_score:.2f} "
            f"mom_z={entry.momentum_z:.2f} cur_z={entry.curvature_z:.2f} hurst={entry.hurst:.2f}"
            for entry in payload.top_rankings
        ) or "none"
        bottom_lines = "\n".join(
            f"#{entry.rank} {entry.ticker} score={entry.composite_score:.2f} "
            f"mom_z={entry.momentum_z:.2f} cur_z={entry.curvature_z:.2f} hurst={entry.hurst:.2f}"
            for entry in payload.bottom_rankings
        ) or "none"
        qualified_lines = "\n".join(payload.qualified_signals) or "none"
        text = (
            f"15m confirmed summary\n"
            f"Cycle time: {payload.cycle_time_ms}\n"
            f"BTC regime: {payload.regime_score}\n"
            f"Dominance falling: {'yes' if payload.dom_falling else 'no'}\n"
            f"Qualified signals:\n{qualified_lines}\n"
            f"Top {len(payload.top_rankings)}:\n{top_lines}\n"
            f"Bottom {len(payload.bottom_rankings)}:\n{bottom_lines}"
        )
        return await self._send_text(text)
