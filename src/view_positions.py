"""
Position Viewer - View current open positions and P&L

Run this anytime to see your paper trading status without running the full system
"""

import pandas as pd
import os
from datetime import datetime
import ccxt

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER_TRADING_DIR = os.path.join(BASE_DIR, "paper_trades")
POSITIONS_FILE = os.path.join(PAPER_TRADING_DIR, "positions.parquet")

def load_positions():
    """Load all positions from file"""
    if not os.path.exists(POSITIONS_FILE):
        print("No positions file found. Run main.py first to create positions.")
        return pd.DataFrame()

    return pd.read_parquet(POSITIONS_FILE)

def view_open_positions():
    """Display all open positions with live P&L"""
    positions = load_positions()

    if positions.empty:
        print("No positions data available")
        return

    open_positions = positions[positions['status'] == 'OPEN']

    if len(open_positions) == 0:
        print("\nNo open positions")
        return

    # Fetch current prices
    exchange = ccxt.binance()
    current_prices = {}

    print("\nFetching current prices...")
    for coin in open_positions['coin'].unique():
        try:
            ticker = exchange.fetch_ticker(f'{coin}/USDT')
            current_prices[coin] = ticker['last']
        except Exception as e:
            print(f"Warning: Could not fetch price for {coin}: {e}")
            current_prices[coin] = None

    print("\n" + "=" * 140)
    print("OPEN POSITIONS WITH LIVE P&L")
    print("=" * 140)
    print(f"{'ID':<20} {'Pair':<15} {'Coin':<6} {'Dir':<6} {'Entry $':<12} {'Current $':<12} {'Size $':<10} {'P&L $':<10} {'P&L %':<8} {'Entry Z':<8} {'Hedge':<6}")
    print("-" * 140)

    total_pnl = 0

    for _, pos in open_positions.iterrows():
        pos_id = pos['position_id'][-12:]
        pair_id = pos.get('pair_id', 'N/A')
        pair_id_short = pair_id[-8:] if pair_id and pair_id != 'N/A' else 'N/A'
        coin = pos['coin']
        direction = pos['direction']
        entry_price = pos['entry_price']
        size_usd = pos['position_size_usd']
        entry_z = pos['entry_z_score']
        entry_time = pd.to_datetime(pos['entry_timestamp']).strftime('%Y-%m-%d %H:%M')
        is_hedge = 'YES' if pos.get('is_hedge', False) else 'NO'

        # Calculate P&L
        current_price = current_prices.get(coin)
        if current_price is None:
            pnl_usd = 0
            pnl_pct = 0
            current_price_str = "N/A"
        else:
            if direction == 'LONG':
                pnl_pct = ((current_price - entry_price) / entry_price) * 100
            else:  # SHORT
                pnl_pct = ((entry_price - current_price) / entry_price) * 100

            pnl_usd = size_usd * (pnl_pct / 100)
            total_pnl += pnl_usd
            current_price_str = f"${current_price:,.2f}"

        pnl_sign = "+" if pnl_usd > 0 else ""

        print(f"{pos_id:<20} {pair_id_short:<15} {coin:<6} {direction:<6} ${entry_price:<11,.2f} {current_price_str:<12} ${size_usd:<9.0f} {pnl_sign}${pnl_usd:<9.2f} {pnl_sign}{pnl_pct:<7.2f}% {entry_z:<7.2f} {is_hedge:<6}")

    # Calculate portfolio metrics
    total_deployed = open_positions['position_size_usd'].sum()
    capital = 100000
    portfolio_heat = (total_deployed / capital) * 100
    total_pnl_pct = (total_pnl / capital) * 100

    print("-" * 140)
    print(f"\nOpen Positions: {len(open_positions)}")
    print(f"Total Deployed: ${total_deployed:,.0f}")
    print(f"Portfolio Heat: {portfolio_heat:.1f}% (max 40%)")

    # Group by pair to show delta-neutral pairs
    if 'pair_id' in open_positions.columns:
        pairs = open_positions.groupby('pair_id')
        print(f"Active Pairs: {len(pairs)}")

    print(f"\nUnrealized P&L: ${total_pnl:+,.2f} ({total_pnl_pct:+.3f}%)")
    print(f"Portfolio Value: $100,000 → ${capital + total_pnl:,.2f}")

def view_closed_positions():
    """Display all closed positions with P&L"""
    positions = load_positions()

    if positions.empty:
        print("No positions data available")
        return

    closed_positions = positions[positions['status'] == 'CLOSED']

    if len(closed_positions) == 0:
        print("\nNo closed positions yet")
        return

    print("\n" + "=" * 140)
    print("CLOSED POSITIONS")
    print("=" * 140)
    print(f"{'ID':<20} {'Coin':<6} {'Dir':<6} {'Entry $':<10} {'Exit $':<10} {'Entry Z':<8} {'Exit Z':<8} {'Hold(h)':<8} {'P&L $':<10} {'P&L %':<8} {'Reason':<15}")
    print("-" * 140)

    for _, pos in closed_positions.iterrows():
        pos_id = pos['position_id'][-12:]  # Last 12 chars
        coin = pos['coin']
        direction = pos['direction']
        entry_price = pos['entry_price']
        exit_price = pos.get('exit_price', 0)
        entry_z = pos['entry_z_score']
        exit_z = pos.get('exit_z_score', 0)
        hold_time = pos.get('hold_time_hours', 0)
        pnl_usd = pos.get('pnl_usd', 0)
        pnl_pct = pos.get('pnl_pct', 0)
        exit_reason = pos.get('exit_reason', 'N/A')

        # Color code P&L
        pnl_symbol = "+" if pnl_usd > 0 else ""

        print(f"{pos_id:<20} {coin:<6} {direction:<6} ${entry_price:<9.2f} ${exit_price:<9.2f} {entry_z:<7.2f} {exit_z:<7.2f} {hold_time:<7.1f} {pnl_symbol}${pnl_usd:<9.2f} {pnl_symbol}{pnl_pct:<7.2f}% {exit_reason:<15}")

    # Summary stats
    total_pnl = closed_positions['pnl_usd'].sum()
    avg_pnl = closed_positions['pnl_usd'].mean()
    wins = len(closed_positions[closed_positions['pnl_usd'] > 0])
    losses = len(closed_positions[closed_positions['pnl_usd'] <= 0])
    win_rate = (wins / len(closed_positions)) * 100 if len(closed_positions) > 0 else 0

    avg_hold_time = closed_positions['hold_time_hours'].mean()

    print("-" * 140)
    print(f"\nClosed Positions: {len(closed_positions)}")
    print(f"Total P&L: ${total_pnl:,.2f}")
    print(f"Average P&L: ${avg_pnl:,.2f}")
    print(f"Win Rate: {win_rate:.1f}% ({wins}W / {losses}L)")
    print(f"Average Hold Time: {avg_hold_time:.1f} hours")

def view_performance_summary():
    """Display performance metrics"""
    positions = load_positions()

    if positions.empty:
        print("No positions data available")
        return

    closed_positions = positions[positions['status'] == 'CLOSED']

    if len(closed_positions) < 5:
        print(f"\nNeed at least 5 closed trades for performance metrics (currently {len(closed_positions)})")
        return

    print("\n" + "=" * 80)
    print("PERFORMANCE METRICS")
    print("=" * 80)

    # Calculate metrics
    total_trades = len(closed_positions)
    total_pnl = closed_positions['pnl_usd'].sum()

    wins = closed_positions[closed_positions['pnl_usd'] > 0]
    losses = closed_positions[closed_positions['pnl_usd'] <= 0]

    win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0
    avg_win = wins['pnl_usd'].mean() if len(wins) > 0 else 0
    avg_loss = losses['pnl_usd'].mean() if len(losses) > 0 else 0

    profit_factor = abs(wins['pnl_usd'].sum() / losses['pnl_usd'].sum()) if len(losses) > 0 and losses['pnl_usd'].sum() != 0 else 0

    # Sharpe ratio (simplified)
    returns = closed_positions['pnl_pct'] / 100
    sharpe = (returns.mean() / returns.std()) * (252 ** 0.5) if returns.std() > 0 else 0

    # Sortino ratio (downside deviation only)
    negative_returns = returns[returns < 0]
    downside_std = negative_returns.std()
    sortino = (returns.mean() / downside_std) * (252 ** 0.5) if downside_std > 0 else 0

    # Max drawdown (cumulative)
    cumulative_pnl = closed_positions['pnl_usd'].cumsum()
    running_max = cumulative_pnl.expanding().max()
    drawdown = cumulative_pnl - running_max
    max_drawdown = drawdown.min()
    max_drawdown_pct = (max_drawdown / 100000) * 100  # 100k capital

    print(f"\nTotal Trades: {total_trades}")
    print(f"Total P&L: ${total_pnl:,.2f}")
    print(f"\nWin Rate: {win_rate:.1f}% ({len(wins)}W / {len(losses)}L)")
    print(f"Average Win: ${avg_win:,.2f}")
    print(f"Average Loss: ${avg_loss:,.2f}")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"\nSharpe Ratio: {sharpe:.2f}")
    print(f"Sortino Ratio: {sortino:.2f}")
    print(f"Max Drawdown: ${max_drawdown:,.2f} ({max_drawdown_pct:.2f}%)")

def main():
    """Main function"""
    print("\n" + "=" * 80)
    print("PAPER TRADING POSITION VIEWER")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    view_open_positions()
    view_closed_positions()
    view_performance_summary()

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
