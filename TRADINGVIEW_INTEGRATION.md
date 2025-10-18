# TradingView Pine Script Integration

Guide for visualizing stat-arb signals and exploring backtesting on TradingView.

---

## Overview

**TradingView limitations for this strategy:**
- Cannot access multiple symbols simultaneously (no AAVE/BTC pair analysis in single script)
- No direct Python integration
- Limited to TradingView's historical data (may differ from Binance)
- Cannot execute delta-neutral pairs (would need two separate charts)

**What TradingView CAN do:**
- Visualize BTC-relative z-scores on coin charts
- Display entry/exit signals for manual validation
- Simple backtests (without proper delta-neutral hedging)
- Help identify if signals make visual sense

---

## Approach 1: Visual Signal Validator (RECOMMENDED)

Display z-scores and signals on TradingView charts for manual inspection.

### Pine Script v5 - BTC-Relative Z-Score Indicator

```pine
//@version=5
indicator("BTC-Relative Z-Score (14-day)", overlay=false)

// Configuration
zscore_window = input.int(288, "Z-Score Window (bars)", minval=10)  // 14 days on 5m = 4032, but TV may be slow
btc_symbol = input.symbol("BINANCE:BTCUSDT", "BTC Symbol")

// Get BTC price
btc_close = request.security(btc_symbol, timeframe.period, close)

// Calculate coin/BTC ratio
ratio = close / btc_close

// Calculate z-score (14-day rolling)
ratio_mean = ta.sma(ratio, zscore_window)
ratio_std = ta.stdev(ratio, zscore_window)
z_score = (ratio - ratio_mean) / ratio_std

// Entry/Exit levels
entry_long_level = -2.0
entry_short_level = 2.0
exit_level = 0.0
stop_loss_level = 5.0

// Plot z-score
plot(z_score, "Z-Score", color=color.blue, linewidth=2)
hline(entry_long_level, "Entry LONG", color=color.green, linestyle=hline.style_dashed)
hline(entry_short_level, "Entry SHORT", color=color.red, linestyle=hline.style_dashed)
hline(exit_level, "Exit (Mean)", color=color.gray, linestyle=hline.style_solid)
hline(stop_loss_level, "Stop Loss", color=color.orange, linestyle=hline.style_dotted)
hline(-stop_loss_level, "Stop Loss", color=color.orange, linestyle=hline.style_dotted)

// Background coloring for zones
bgcolor(z_score > entry_short_level ? color.new(color.red, 90) : na, title="Overvalued Zone")
bgcolor(z_score < entry_long_level ? color.new(color.green, 90) : na, title="Undervalued Zone")

// Signals
long_signal = ta.crossunder(z_score, entry_long_level)
short_signal = ta.crossover(z_score, entry_short_level)
exit_signal = math.abs(z_score) < 0.2 and math.abs(z_score[1]) >= 0.2

plotshape(long_signal, "LONG Signal", shape.triangleup, location.bottom, color.green, size=size.small)
plotshape(short_signal, "SHORT Signal", shape.triangledown, location.top, color.red, size=size.small)
plotshape(exit_signal, "EXIT Signal", shape.circle, location.absolute, color.gray, size=size.tiny)
```

**How to use:**
1. Open TradingView → Select AAVE/USDT chart
2. Add indicator → Paste code above
3. Observe z-score behavior visually
4. Validate: Do extreme z-scores revert to mean?

**Limitations:**
- Uses SMA/stdev (we use MAD in Python for robustness)
- Slower lookback window on TradingView (288 bars instead of 4032 due to performance)
- No delta-neutral hedging visualization

---

## Approach 2: Simplified Backtest (LIMITED VALUE)

**WARNING:** This is NOT true delta-neutral backtesting. Use only for rough validation.

### Pine Script v5 - Simple Backtest Strategy

```pine
//@version=5
strategy("BTC-Relative Mean Reversion (SIMPLIFIED)", overlay=true, default_qty_type=strategy.percent_of_equity, default_qty_value=2)

// Config
zscore_window = input.int(288, "Z-Score Window")
btc_symbol = input.symbol("BINANCE:BTCUSDT", "BTC Symbol")
entry_threshold = input.float(2.0, "Entry Threshold (|z|)")
exit_threshold = input.float(0.2, "Exit Threshold (|z|)")

// Get BTC price
btc_close = request.security(btc_symbol, timeframe.period, close)

// Calculate z-score
ratio = close / btc_close
ratio_mean = ta.sma(ratio, zscore_window)
ratio_std = ta.stdev(ratio, zscore_window)
z_score = (ratio - ratio_mean) / ratio_std

// Entry conditions
long_entry = z_score < -entry_threshold
short_entry = z_score > entry_threshold

// Exit conditions
exit_condition = math.abs(z_score) < exit_threshold
stop_loss = math.abs(z_score) > 5.0

// Execute trades (DIRECTIONAL, NOT DELTA-NEUTRAL)
if long_entry
    strategy.entry("LONG", strategy.long)
if short_entry
    strategy.entry("SHORT", strategy.short)
if exit_condition or stop_loss
    strategy.close_all()

// Plot
plot(z_score, "Z-Score", color=color.blue)
hline(-entry_threshold, "Entry LONG", color=color.green)
hline(entry_threshold, "Entry SHORT", color=color.red)
```

**CRITICAL LIMITATIONS:**
- This is **directional** trading (not delta-neutral pairs)
- No BTC hedge (you're exposed to market direction)
- Results will NOT match Python paper trading system
- Use only to validate: "Do z-scores >2 tend to revert?"

**Why results differ from your system:**
- Python: LONG AAVE + SHORT BTC (market-neutral)
- TradingView: LONG AAVE only (directional)
- Python wins if AAVE/BTC spread narrows (regardless of absolute prices)
- TradingView wins only if AAVE price increases

---

## Approach 3: Data Export for External Backtesting

**Best approach for accurate backtesting:**

1. **Export TradingView data:**
   - Right-click chart → Export chart data → CSV
   - Download OHLCV for both coin and BTC
   - Store in `backtest_data/` folder

2. **Run Python backtest:**
   - Use your existing `outlier_detector.py` logic
   - Calculate z-scores on exported data
   - Simulate delta-neutral pair entries/exits
   - Calculate realistic P&L with fees/slippage/funding

3. **Benefits:**
   - Accurate delta-neutral simulation
   - Your exact z-score calculation (MAD-based)
   - Realistic transaction costs
   - Full control over execution logic

---

## Recommended Workflow

**For visual validation:**
1. Use Approach 1 (Visual Indicator) on TradingView
2. Manually inspect: Do extreme z-scores revert?
3. Check: Are entry signals reasonable?

**For backtesting:**
1. ❌ DON'T use TradingView strategy backtest (not delta-neutral)
2. ✅ DO use Python paper trading forward (current approach)
3. ✅ LATER: Export TV data → backtest in Python with proper pairs

**Current recommendation:**
- Stick with Python paper trading (2-3 months forward)
- Use TradingView only for visual confirmation
- If you want historical backtesting later, export TV data → backtest in Python

---

## TradingView Limitations Summary

| Feature | Your Python System | TradingView |
|---------|-------------------|-------------|
| Delta-neutral pairs | ✅ Automatic | ❌ Not possible |
| BTC hedge | ✅ Simultaneous | ❌ Separate charts |
| MAD-based z-score | ✅ Robust | ❌ Only SMA/stdev |
| 14-day window (4032 bars) | ✅ Fast | ⚠️ Slow/laggy |
| Transaction costs | ✅ Realistic | ⚠️ Simplified |
| Funding rate costs | ✅ Modeled | ❌ Not available |
| Multi-coin analysis | ✅ 62 coins | ❌ One at a time |

**Verdict:** TradingView is useful for visualization, but NOT for accurate backtesting of delta-neutral stat-arb.

---

## Alternative: Build Custom Backtest in Python

If you want historical backtesting (better than TradingView):

**Requirements:**
1. Download historical OHLCV (Binance API, all coins, 6+ months)
2. Calculate z-scores on historical data
3. Simulate pair entries (LONG coin + SHORT BTC)
4. Calculate pair P&L (combined leg performance)
5. Apply realistic costs (fees, slippage, funding)
6. Avoid lookahead bias (use `label="right"` on resampling)

**Advantage over TradingView:**
- True delta-neutral simulation
- Exact same logic as live system
- Realistic transaction costs
- Can test 6-12 months of history

**Disadvantage:**
- Still has survivorship bias (missing delisted coins)
- Regime changes (bull ≠ bear ≠ ETF era)
- Data quality issues

**Jim Simons would say:** Paper trading forward > flawed backtest. Your current approach (2-3 months forward) is correct.
