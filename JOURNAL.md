# Journal

## 2026-04-03

- Initialized the repository from a blank directory.
- Implemented the signal engine end to end: config, state manager, Bybit REST bootstrapper, WebSocket ingestion, indicator math, ranking, SQLite logging, Telegram alerting, async supervision, and tests.
- Resolved two spec contradictions explicitly instead of hiding them:
  - BTC `EMA200` forced the BTC daily lookback to increase from 30 to 220.
  - BTC dominance was replaced with a BTC-vs-alt-basket proxy so the system can run on Bybit market data alone.
- Raised the volatility safety floor to `1e-5` after tests showed that a near-zero denominator let smooth low-vol drifts dominate the composite score unrealistically.
- Added cycle settling, candle-gap detection, and batched SQLite writes to keep rankings coherent and per-cycle processing cheap.
- Moved cycle timestamps onto the queue/engine path and fixed cooldown so failed alert sends do not poison future alerts.
- Added a replay runner so recent Bybit history can be pushed through the production engine path for faster validation.
- Added a live universe validator against Bybit instruments metadata so dead symbols can be caught before deployment.
- Added a SQLite report CLI so replay and live logs can be inspected quickly without manual SQL.
- Ran the live universe validator and removed `FETUSDT` and `FTMUSDT` from the default universe after Bybit reported them unavailable in the current linear market.
- Added a one-command smoke runner that chains live universe validation, replay, and log reporting.
- Ran a live smoke check successfully: Bybit universe validation passed for the shipped 59-symbol tracked set, and a 4-cycle replay wrote 348 rows to `/tmp/outlier-smoke.sqlite3` with no runtime errors.
- Ran the actual service against live Bybit endpoints and confirmed the full startup path: REST bootstrap completed, BTC macro refresh completed, the WebSocket connected and subscribed, and the engine processed the bootstrap cycle cleanly.
- Added a benchmark CLI and measured the current engine path at roughly 36ms average for 100 synthetic tickers on this machine.
- Added bounded-runtime and runtime-summary support to `main.py` so the live loop can be soak-tested without manual shutdown.
- Added a deployment-oriented env template and soak-run guide so VPS rollout is scripted instead of improvised.
- Refactored the live path so the WebSocket now feeds two engine stages instead of only closed candles:
  - `emerging` intrabar ranking off provisional candle prices
  - `confirmed` ranking on 15m candle close using confirmed history only
- Added isolated provisional state in `state.py`, stage-aware SQLite logging, stage-aware Telegram messages, and separate emerging-vs-confirmed cooldown handling.
- Ran a live bounded service check after the refactor and confirmed the real process now emits repeated `emerging` cycles during the open candle while keeping the close-confirmed cycle intact, with zero websocket failures in the test window.
- Tightened the intrabar path from simple provisional top-rank polling into a real state machine: tickers now move through `WATCHLIST -> EMERGING -> CONFIRMED`, with `EMERGING` requiring recent rank improvement plus rising composite score across successive intrabar observations.
- Added `signal_kind` logging so SQLite and Telegram distinguish `watchlist`, `emerging`, `confirmed`, and `none` rows instead of hiding everything under the broader processing stage.
- Added confirmed rank persistence tracking so close-confirmed signals can upgrade to `confirmed_strong` when a ticker has held leadership across recent confirmed bars, without blocking the first confirmed breakout signal.
- Verified the implementation locally with `pytest` and `python3 -m py_compile`.
