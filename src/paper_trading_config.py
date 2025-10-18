# ============================================================================
# PAPER TRADING CONFIGURATION
# ============================================================================

import os

# Capital Configuration
STARTING_CAPITAL = 100_000  # $100k starting capital (realistic for stat-arb)
POSITION_SIZE_PCT = 0.01  # 1% of capital per LEG ($1k LONG + $1k SHORT = $2k total per pair)
MAX_PORTFOLIO_HEAT = 1.00  # Max 100% of capital deployed (data collection phase - maximize trades)

# Delta-Neutral Configuration (Renaissance-style)
DELTA_NEUTRAL_ENABLED = True  # Enable delta-neutral hedging with BTC
HEDGE_RATIO = 1.0  # 1:1 hedge ratio (equal notional)

# Signal Thresholds (PHASE 1: Data Collection)
ENTRY_Z_THRESHOLD = 1.5  # Enter when |z-score| > 1.5 (lowered from 2.0 for more data)
EXIT_Z_THRESHOLD = 0.3  # Exit when |z-score| < 0.3 (tighter profit taking)
STOP_LOSS_Z_THRESHOLD = 5.0  # Stop loss when |z| > 5.0 (regime change)

# Multi-Factor Entry Filters (PHASE 1: Quality filters)
USE_MULTI_FACTOR_FILTERS = True
MIN_Z_VELOCITY = 0.02  # Minimum momentum toward mean reversion
MAX_HALF_LIFE_HOURS = 48  # Maximum half-life (prefer faster mean reversion)
MIN_VOLUME_SURGE_Z = -1.0  # Avoid collapsing volume (not below -1 std dev)
MAX_ABS_FUNDING_RATE = 0.05  # Avoid extreme funding rate environments (5%)

# Time-Based Stops (using half-life)
USE_HALF_LIFE_STOP = True
HALF_LIFE_STOP_MULTIPLIER = 2.0  # Exit if hold_time > 2 × half_life (not reverting)

# Transaction Costs (Binance Futures - realistic modeling)
APPLY_TRANSACTION_COSTS = True  # Enable realistic cost modeling
TAKER_FEE = 0.0004  # 0.04% taker fee (assume market orders)
SLIPPAGE = 0.0005  # 0.05% slippage estimate for liquid perpetuals
TOTAL_ENTRY_COST = TAKER_FEE + SLIPPAGE  # 0.09% per entry
TOTAL_EXIT_COST = TAKER_FEE + SLIPPAGE  # 0.09% per exit
TOTAL_ROUND_TRIP_COST = TOTAL_ENTRY_COST + TOTAL_EXIT_COST  # 0.18% total
USE_ACTUAL_FUNDING_RATES = True  # Use actual funding rate data from derivatives

# Risk Management
MAX_CONCURRENT_POSITIONS = 50  # Max 50 positions (100% heat / 1% per position)
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
