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

- **OI** WE ADDED OI CHANGE 1 HOUR AND 24 HOUR
- **VOLZ** ADDED Volume Z SCORE

### Strategy Decision
- **Confirmed stat-arb approach** (mean reversion, NOT momentum)
  - Fade extremes (z-score >2 or <-2)
  - Hold for days/weeks until reversion
  - Target 55-60% win rate with asymmetric payoffs

---

## [2025-10-18] Paper Trading System Implementation

### Added
- **Delta-Neutral Paper Trading Engine**
  - Automatic pair execution: LONG coin + SHORT BTC hedge
  - 2% per leg position sizing ($2k + $2k = $4k per pair on $100k capital)
  - Entry: |z| > 2.0, Exit: z → 0, Stop: z > 5.0
  - Max portfolio heat: 40% deployed (20% net exposure, 10 pairs max)

- **Paper Trading Infrastructure**
  - `paper_trading_config.py`: Configuration (thresholds, fees, capital)
  - `signal_logger.py`: Entry/exit/stop signal tracking
  - `position_manager.py`: Position lifecycle with P&L calculation
  - `performance_tracker.py`: Sharpe, Sortino, drawdown metrics
  - `paper_trading_engine.py`: Main orchestration
  - `view_positions.py`: Standalone position/P&L viewer

- **Transaction Cost Modeling**
  - Binance futures fees: 0.02% maker, 0.05% taker
  - Slippage: 0.05% per trade
  - Funding rate cost: 0.01% per 8h (modeled)
  - Total round-trip cost: ~0.2-0.3% per position

### Changed
- **ENABLE_DATA_DOWNLOAD**: False → True
  - **Rationale**: Live/paper trading requires fresh data each run (rolling 30-day window)
  - **Impact**: Positions managed on current data, exits triggered when z-score reverts
  - **Note**: NOT lookahead bias - using only data available at execution time

### Configuration Decision: Option 1 (Fixed Thresholds)

**Decision**: Keep simple fixed thresholds, NO variable stops

**Entry/Exit Settings** (FINAL - No changes until 50+ trades):
```
ENTRY_Z_THRESHOLD = 2.0  # Fixed entry signal
EXIT_Z_THRESHOLD = 0.0   # Mean reversion target
STOP_LOSS_Z_THRESHOLD = 5.0  # Fixed statistical boundary
POSITION_SIZE_PCT = 0.02  # 2% per leg (fixed)
```

**Risk-Reward Analysis**:
- Naive RR: 1:3.5 (2.0 reward / 7.0 risk)
- Probability-weighted RR: ~2.5:1 (accounting for 80% win rate)
- Expected value: ~4% per trade (before costs)

**Rationale (Jim Simons Philosophy)**:
1. **Simplicity over optimization** - Fewer parameters = less overfitting
2. **Statistical boundaries** - z=5.0 is a fixed statistical threshold (extreme outlier)
3. **Evidence-based** - Gather 50+ trades before adjusting thresholds
4. **Portfolio-level risk** - Manage via portfolio heat limits, not individual stops

**Alternatives Rejected**:
- ❌ **Variable stop loss** - Overfitting, adds complexity
- ❌ **Coin-specific thresholds** - Cannot aggregate learnings
- ❌ **Dynamic exits** - Curve-fitting to past regimes

**What Could Change (After 50+ Trades)**:
- If stop-out rate >15%: Consider ENTRY_Z = 2.5 (not tighter stops)
- If win rate <55%: Signal quality issue, not threshold problem
- If Sharpe <1.0: Fundamental strategy issue

**Success Metrics** (2-3 Month Target):
- Win rate: 55-60% minimum
- Profit factor: >1.5
- Sharpe ratio: >1.0
- Max drawdown: <15%
- Stop-out rate: <15%

**Next Steps**:
1. Run daily: `python3 src/main.py`
2. Monitor: `python3 src/view_positions.py`
3. Collect 50 closed trades (2-3 months)
4. Analyze metrics
5. THEN optimize (if needed)

---

## [2025-10-18] Documentation Updates

### Added
- **TRADINGVIEW_INTEGRATION.md**
  - Pine Script v5 indicator for BTC-relative z-score visualization
  - Explanation of TradingView limitations (cannot execute delta-neutral pairs)
  - Recommendation: Use for visualization only, not backtesting

### Changed
- **STRATEGY.md**
  - Added delta-neutral implementation details
  - Updated position sizing: 2% per leg (Option 2 - aggressive)
  - Added portfolio limits: 40% deployed, 20% net, 10 pairs max
  - Added paper trading system status






