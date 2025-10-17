# STRATEGY DECISIONS - STATISTICAL ARBITRAGE SYSTEM

## Strategy Type: Mean Reversion Stat-Arb (Renaissance-style)

This is NOT a momentum system. We fade extremes, we don't chase trends.

**Core bet:** "When a coin deviates significantly from its historical relationship with BTC, it will revert to that relationship over days/weeks"

---

## Key Configuration Decisions (Oct 2025)

### Z-Score Window: 14 days (4032 bars on 5m)
- Captures genuine structural mispricings, not intraday noise
- Renaissance used weeks-to-months lookback for stat-arb
- Previous 24h window was too short (HFT territory, not stat-arb)

### Velocity Window: 24 hours (288 bars)
- Measures meaningful z-score momentum
- Aligned with stat-arb timeframe (not microstructure)

### Acceleration: DISABLED
- Third derivative over short windows = pure noise
- Jim Simons avoided higher-order derivatives at ultra-short timeframes
- No predictive value, just overfitting

### Data Requirements: 30 days history
- Need sufficient lookback for 14-day window + buffer for half-life calculations

### Primary Timeframe: 5m bars
- All core analysis runs on 5m (z-score, velocity, half-life, volume)
- MTF confirmation uses 1h/4h/1d for structural trend validation
- 1m and 15m downloaded but NOT used in calculations

---

## Signal Hierarchy (Order of Importance)

1. **BTC-Relative Z-Score (14-day)** - PRIMARY EDGE
   - Entry: |Z| > 2.0 (2 std dev from mean)
   - Exit: Z crosses back to 0 (mean reversion complete)

2. **MTF Confirmation (1h/4h/1d)** - STRUCTURAL VALIDATION
   - Equal-weight voting system
   - 2+ aligned timeframes = 20% score boost
   - Filter: Don't fade strong aligned trends (regime change detection)

3. **Half-Life** - RISK MANAGEMENT & POSITION SIZING
   - Faster mean reversion (low half-life) = larger positions
   - Filter: Skip trades with half-life > 72h (too slow, trend risk)

4. **Volume Surge** - ENTRY TIMING CONFIRMATION
   - 2 std dev threshold (meaningful surges only)
   - 30% score boost when volume confirms move

5. **Derivatives Data (Optional)** - NICE TO HAVE
   - Binance funding rate, OI, liquidations (FREE API, sufficient)
   - Coinglass premium NOT needed (Binance alone = 40%+ of market)
   - Confirmation signal, not primary edge

---

## Risk Management Principles (CRITICAL)

### Position Sizing
- Max 2% of capital per trade
- Never risk more than you can afford on one outlier

### Stop Loss (MANDATORY)
- Exit if z-score > 5.0 (admit you're wrong, regime change)
- "It keeps going" can happen (LTCM lesson: stat-arb can blow up)

### Portfolio Limits
- Max 20% of capital in active stat-arb trades
- Correlation limit: Don't short 5 memecoins simultaneously (same bet)

### Drawdown Circuit Breaker
- >10% drawdown: Cut position sizes 50%
- >20% drawdown: Pause trading, review system

### Expected Win Rate: 55-60%
- You WILL be wrong 40-45% of the time
- Edge comes from asymmetry (bigger wins than losses) + sample size

---

## What Makes This Different From CoinMarketCap "Top Movers"

**CMC:** "What moved the most?" (reactive, no prediction)
**This System:** "What's statistically overextended and likely to revert?" (predictive)

We're not chasing what already moved. We're betting on statistical reversion of extremes.

**Example Trade:**
- Day 1: PEPE z-score = +3.2 → SHORT PEPE, LONG BTC (delta-neutral pair)
- Day 5: PEPE z-score = +0.8 → Reversion happening
- Day 9: PEPE z-score = -0.1 → Mean reversion complete, CLOSE POSITION

---

## Multi-Timeframe System Details

### Active Timeframes
- **5m:** PRIMARY - all core analysis happens here
- **1h/4h/1d:** MTF confirmation (equal-weight voting, 20% boost)

### Dormant Timeframes
- **1m/15m:** Downloaded but NOT used in calculations (potential future use)

### MTF Logic
- No individual timeframe weights
- Binary confirmation: 2+ aligned = boost, otherwise neutral
- Used for trend validation, not score calculation

---

## Data Sources & Quality

### OHLCV Data
- Binance Futures API (free, sufficient)
- 30 days historical required
- Quality: Production-grade, no fancy cleaning needed yet

### Derivatives Data
- Binance API (free): Funding rate, OI, liquidations, long/short ratio
- Coinglass premium: NOT needed (expensive, diminishing returns)
- Role: Confirmation only, not primary edge

**Renaissance didn't have derivatives data in the 90s** - they made billions on pure price/volume stat-arb. We have MORE data than they did.

---

## Backtesting Reality Check

### Current system is NOT ready for backtesting

**Critical problems:**
- Survivorship bias (coin list = today's survivors, missing dead coins)
- Look-ahead bias (using current market cap on historical data)
- No transaction costs (slippage, fees, funding rate costs)
- No execution modeling (assume trade at close = fantasy)
- Data quality issues (gaps, delistings, maintenance periods)
- Regime changes (2021 bull ≠ 2022 bear ≠ 2024 ETF inflows)

**Jim Simons would say:** "Come back when you have clean data, realistic costs, and proper validation."

**Better approach for now:** Paper trade forward for 2-3 months to collect real execution metrics. Worth 10x more than flawed backtest.

---

## NO EWMA Smoothing

**Question raised:** Should we smooth signals with EWMA to reduce jumpiness?
**Answer:** NO - Jim Simons wouldn't do it.

### Why not
- EWMA introduces lag (death in stat-arb)
- Smoothing obscures the microstructure patterns we're extracting
- If signals are jumpy, that's information (volatility clustering)
- Renaissance traded the noise, didn't smooth it away

### Better approaches
- Position sizing based on signal confidence (not smoothing)
- Ensemble models (combine uncorrelated signals)
- If forced to denoise: Kalman filters (adaptive, not backward-looking EWMA)

---

## Future Considerations

### Momentum Strategy (Later)
- Opposite of mean reversion: ride winners instead of fading them
- Timeframe: Months/years (not days/weeks)
- Example funds: AQR, Winton, Man Group
- Can combine with stat-arb on different timeframes
  - Short-term: Mean reversion (days/weeks)
  - Long-term: Momentum (months/years)
- They're complementary, not contradictory

### Multi-Strategy Future
- Mean reversion on 14-day extremes (current system)
- Momentum on 90-day trends (future addition)
- Example: Fade 2-day spike while holding 6-month uptrend position

---

## Open Questions & Next Steps

### Before Live Trading
1. Download 30 days of OHLCV data (when internet access available)
2. Run system with 14-day windows, observe signal stability
3. Paper trade for 2-3 months to validate edge
4. Implement risk management (position sizing, stop losses, portfolio heat limits)
5. Add Binance derivatives data (funding rate confirmation)

### Before Backtesting
1. Source historical market cap data (not current values on historical dates)
2. Build survivorship-bias-free coin universe (include delisted/dead coins)
3. Model transaction costs (slippage, fees, funding rate costs)
4. Implement execution simulation (bid-ask spread, order fill modeling)
5. Design walk-forward validation (train/test splits, out-of-sample testing)

### Critical Backtesting Implementation Detail

**ALWAYS use correct resampling to avoid lookahead bias:**

```python
# WRONG (lookahead bias - timestamp is START of period):
df.resample("1h").last()  # Default: label="left", closed="left"

# CORRECT (timestamp is END of period):
df.resample("1h", label="right", closed="right").last()
```

**Why this matters:**
- `label="left"` puts timestamp at period START (you're seeing future data)
- `label="right"` puts timestamp at period END (you only know what happened)
- Most LLMs and code examples get this wrong
- This single mistake can make a losing strategy look profitable

**Apply to all timeframe conversions:**
- 5m → 15m aggregation
- 5m → 1h aggregation
- 5m → 4h aggregation
- 5m → 1d aggregation

**Source:** Friend's advice - common backtesting pitfall

If we are backtesting we should download historical data and store them in parquet files