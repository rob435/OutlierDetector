# CHANGELOG

All notable changes to this project will be documented in this file.

Format: Date, change description, rationale, impact

---

## [2025-10-17] Statistical Arbitrage Configuration

### Changed
- **Z_SCORE_WINDOW**: 288 bars (24h) → 4032 bars (14 days)
  - **Rationale**: 24h window was catching microstructure noise, not genuine mispricings. Renaissance-style stat-arb uses weeks-to-months lookback.
  - **Impact**: Signals will be more stable, fewer false positives, longer hold times (days not hours)

- **Z_SCORE_VELOCITY_WINDOW**: 24 bars (2h) → 288 bars (24h)
  - **Rationale**: Align velocity measurement with stat-arb timeframe, not HFT noise
  - **Impact**: Velocity now measures meaningful z-score momentum

- **PRICE_CHANGE_PERIOD**: 12 bars (1h) → 288 bars (24h)
  - **Rationale**: Consistent with stat-arb approach
  - **Impact**: Price change measures full-day moves, not intraday volatility

- **DATA_DOWNLOAD_DAYS**: 3 days → 30 days
  - **Rationale**: Need sufficient history for 14-day z-score calculation + buffer for half-life
  - **Impact**: Larger initial download, but necessary for proper statistical analysis

### Disabled
- **Z_SCORE_ACCELERATION_ENABLED**: True → False
  - **Rationale**: Third derivative over short windows = pure noise with no predictive value. Jim Simons avoided higher-order derivatives.
  - **Impact**: Removes noisy signal that was likely causing false entries

### Strategy Decision
- **Confirmed stat-arb approach** (mean reversion, NOT momentum)
  - Fade extremes (z-score >2 or <-2)
  - Hold for days/weeks until reversion
  - Target 55-60% win rate with asymmetric payoffs

---

## [Previous - Before Oct 2025]

### Initial System
- 24-hour z-score window
- 2-hour velocity window
- 1-hour acceleration window
- MTF confirmation (1h/4h/1d)
- Half-life calculation for position sizing
- Volume surge detection

**Note**: This configuration was more suitable for HFT/microstructure trading, not stat-arb
