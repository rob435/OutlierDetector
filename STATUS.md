# Status

## Current

- Core implementation exists.
- The live engine now supports both intrabar and close-confirmed cycles, with intrabar signals differentiated into `watchlist` and `emerging`, and confirmed signals able to upgrade into `confirmed_strong`.
- Tests exist for indicator math, gap detection, and signal-engine behavior.
- Replay tooling exists for recent-candle validation through the production engine path.
- Universe validation tooling exists for checking the manual symbol list against Bybit’s current instruments.
- SQLite reporting tooling exists for quick inspection of replay and live signal logs.
- One-command smoke validation exists to chain universe validation, replay, and reporting.
- Benchmark tooling exists to measure cycle latency against the spec target.
- `main.py` supports bounded live runs with runtime counters for soak validation.
- Deployment docs now include a dedicated soak-run guide and production env template.
- Deployment scaffold exists for `systemd`.
- Local verification is green: `22` tests passing.
- Cycle processing now batches DB writes and waits briefly for the WebSocket close wave before scoring.
- Confirmed logs and cooldown are tied to candle event time; emerging alerts use wall-clock detection time because they fire before candle close.
- Default universe was live-validated against Bybit; `FETUSDT` and `FTMUSDT` were removed after failing validation.
- A live smoke run completed successfully against Bybit from this environment using `smoke.py --cycles 4 --strict-universe`.
- The actual service startup path was exercised against live Bybit endpoints: REST bootstrap, BTC macro refresh, WebSocket subscribe, and bootstrap-cycle processing all completed without runtime errors.
- A post-refactor live run completed successfully against Bybit with both stage paths active: one bootstrap `confirmed` cycle followed by repeated intrabar `emerging` cycles, with `websocket_failures=0`.
- The intrabar path now requires strengthening before promotion to `emerging`; it no longer equates “currently top-ranked” with “emerging breakout.”
- Confirmed-bar persistence is now tracked so repeated confirmed leadership is visible as `confirmed_strong` instead of being buried inside generic confirmed rows.
- Confirmed 15m cycles now emit a routine Telegram summary with the top and bottom ranked names so operators can distinguish "no alerts" from "no activity."

## Remaining risks

- No sustained live WebSocket soak across multiple candle closes or systemd deployment run has been executed yet.
- The BTC dominance component is a proxy, not true market-cap dominance.
- The manual universe may include symbols that are not currently listed on the chosen Bybit environment.
- Intrabar `watchlist` and `emerging` processing are intentionally noisier than the close-confirmed path and still need real-world observation before anyone should trust their alert quality.

## Next validation steps

- Run the engine on testnet or a compliant mainnet host for 5 to 7 days.
- Review `watchlist`, `emerging`, `confirmed`, and `confirmed_strong` logged signals for false positives, timing benefit, and missed breakouts before tuning thresholds.
- Trim or refresh the manual universe against the exact Bybit market you intend to trade.
