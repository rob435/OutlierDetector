# Quick Start Guide - PairsAlgo

One-page reference for running your stat-arb system.

---

## Daily Operations

### Run Paper Trading
```bash
cd /Applications/PairsAlgo
python3 src/main.py
```

### Check Positions
```bash
python3 src/view_positions.py
```

### Run Automation Script
```bash
./run_daily.sh
```

---

## Automation Setup (One-Time)

### Option 1: launchd (Recommended for macOS)
```bash
# 1. Create plist file (see AUTOMATION.md for template)
nano ~/Library/LaunchAgents/com.pairsalgo.papertrading.plist

# 2. Load the job
launchctl load ~/Library/LaunchAgents/com.pairsalgo.papertrading.plist

# 3. Verify
launchctl list | grep pairsalgo
```

### Option 2: Cron (Alternative)
```bash
# Edit crontab
crontab -e

# Add this line (runs 3x daily: 9am, 3pm, 9pm)
0 9,15,21 * * * /Applications/PairsAlgo/run_daily.sh >> /Applications/PairsAlgo/logs/cron.log 2>&1

# Verify
crontab -l
```

---

## Key Files

| File | Purpose |
|------|---------|
| `src/main.py` | Main execution script |
| `src/view_positions.py` | Position viewer |
| `paper_trades/positions.parquet` | All positions (open + closed) |
| `paper_trades/signals.parquet` | All signals logged |
| `logs/` | Automation logs |

---

## Configuration (Don't Change Until 50 Trades)

```python
# src/main.py
ENABLE_DATA_DOWNLOAD = True  # Fresh data each run
ENABLE_PAPER_TRADING = True  # Paper trading on

# src/paper_trading_config.py
ENTRY_Z_THRESHOLD = 2.0      # Entry signal
EXIT_Z_THRESHOLD = 0.0       # Mean reversion target
STOP_LOSS_Z_THRESHOLD = 5.0  # Statistical boundary
POSITION_SIZE_PCT = 0.02     # 2% per leg (fixed)
STARTING_CAPITAL = 100000    # $100k paper capital
```

---

## Success Metrics (Target After 50 Trades)

| Metric | Target | Current |
|--------|--------|---------|
| Win Rate | 55-60% | TBD |
| Profit Factor | >1.5 | TBD |
| Sharpe Ratio | >1.0 | TBD |
| Max Drawdown | <15% | TBD |
| Stop-Out Rate | <15% | TBD |

---

## Weekly Checklist

**Monday**: Check automation logs, review open positions
**Wednesday**: Spot-check signals
**Friday**: Weekly P&L review
**Monthly**: Performance analysis (after 15+ trades)

---

## Troubleshooting

### System not running?
```bash
# Check if automation is active
launchctl list | grep pairsalgo  # launchd
crontab -l                       # cron

# Check logs
tail -f logs/launchd.log
tail -f logs/cron.log

# Run manually to debug
python3 src/main.py
```

### No positions opening?
- Check if any coins have |z| > 2.0 (view outlier rankings)
- Check portfolio heat (max 40% deployed)
- Check max positions (max 10 pairs)

### Positions not closing?
- Check if z-score crossed back to 0
- Ensure `ENABLE_DATA_DOWNLOAD = True` (need fresh data)
- Check stop loss threshold (z > 5.0)

---

## Important Reminders

1. **Don't change thresholds** until 50+ closed trades
2. **Run 3x per day** minimum (9am, 3pm, 9pm recommended)
3. **Let it run for 2-3 months** before judging performance
4. **Keep ENABLE_DATA_DOWNLOAD = True** for live trading
5. **Review CHANGELOG.md** for all configuration decisions

---

## Next Steps

1. Set up automation (launchd or cron)
2. Let system run for 2-3 months
3. Collect 50 closed trades
4. Analyze performance metrics
5. Read LEARNING_PLAN.md for what to study

---

## Emergency Contacts

**Documentation**:
- STRATEGY.md - Strategy decisions
- CHANGELOG.md - All changes with rationale
- AUTOMATION.md - Full automation guide
- LEARNING_PLAN.md - What to learn while waiting
- TRADINGVIEW_INTEGRATION.md - Visualization options

**Support**: This is a solo project - debug using logs and documentation

---

**Remember**: Jim Simons took 10 years to build Medallion Fund. You're in Week 1. Be patient.
