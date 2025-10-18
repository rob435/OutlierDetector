# ============================================================================
# PAPER TRADING CONFIGURATION
# ============================================================================

import os

# Capital Configuration
STARTING_CAPITAL = 100_000  # $100k starting capital (realistic for stat-arb)
POSITION_SIZE_PCT = 0.02  # 2% of capital per LEG ($2k LONG + $2k SHORT = $4k total per pair)
MAX_PORTFOLIO_HEAT = 0.40  # Max 40% of capital deployed (20% net exposure, 10 pairs max)

# Delta-Neutral Configuration (Renaissance-style)
DELTA_NEUTRAL_ENABLED = True  # Enable delta-neutral hedging with BTC
HEDGE_RATIO = 1.0  # 1:1 hedge ratio (equal notional)

# Signal Thresholds (from STRATEGY.md)
ENTRY_Z_THRESHOLD = 2.0  # Enter when |z-score| > 2.0 (2 std dev)
EXIT_Z_THRESHOLD = 0.0  # Exit when z-score crosses back to 0 (mean reversion complete)
STOP_LOSS_Z_THRESHOLD = 5.0  # Stop loss when z > 5.0 (regime change)

# Transaction Costs (Binance Futures - realistic)
MAKER_FEE = 0.0002  # 0.02% maker fee
TAKER_FEE = 0.0005  # 0.05% taker fee
SLIPPAGE = 0.0005  # 0.05% slippage estimate for liquid perpetuals
FUNDING_RATE_COST_PER_8H = 0.0001  # Average funding rate cost (0.01% per 8h)

# Risk Management
MAX_CONCURRENT_POSITIONS = 10  # Max 10 positions (20% heat / 2% per position)
CORRELATION_THRESHOLD = 0.7  # Don't open correlated positions (>0.7 correlation)

# Data Storage
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER_TRADING_DIR = os.path.join(BASE_DIR, "paper_trades")
SIGNALS_FILE = os.path.join(PAPER_TRADING_DIR, "signals.parquet")
POSITIONS_FILE = os.path.join(PAPER_TRADING_DIR, "positions.parquet")
DAILY_PNL_FILE = os.path.join(PAPER_TRADING_DIR, "daily_pnl.parquet")
METRICS_FILE = os.path.join(PAPER_TRADING_DIR, "metrics.json")

# Performance Tracking
RISK_FREE_RATE = 0.05  # 5% annual risk-free rate for Sharpe calculation
TRADING_DAYS_PER_YEAR = 365  # Crypto trades 24/7

# Position Sizing Method
POSITION_SIZING_METHOD = "fixed_pct"  # Options: "fixed_pct", "volatility_adjusted", "half_life_adjusted"

# Exit Strategy
EXIT_STRATEGY = "mean_reversion"  # Options: "mean_reversion" (z=0), "opposite_extreme" (z crosses opposite threshold)

# Ensure paper trading directory exists
os.makedirs(PAPER_TRADING_DIR, exist_ok=True)
