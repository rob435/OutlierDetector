"""
Position Manager - Track open positions and calculate P&L

Manages position lifecycle: OPEN → MONITORING → CLOSED
Validates risk rules: 2% size, z>5 stop, 20% portfolio heat
"""

import pandas as pd
import os
from datetime import datetime
from paper_trading_config import (
    POSITIONS_FILE, PAPER_TRADING_DIR, STARTING_CAPITAL,
    POSITION_SIZE_PCT, MAX_CONCURRENT_POSITIONS, MAX_PORTFOLIO_HEAT,
    TAKER_FEE, SLIPPAGE, TOTAL_ENTRY_COST, TOTAL_EXIT_COST,
    USE_ACTUAL_FUNDING_RATES, APPLY_TRANSACTION_COSTS,
    STOP_LOSS_Z_THRESHOLD, DELTA_NEUTRAL_ENABLED, HEDGE_RATIO
)

class PositionManager:
    def __init__(self):
        self.positions_file = POSITIONS_FILE
        self.capital = STARTING_CAPITAL
        os.makedirs(PAPER_TRADING_DIR, exist_ok=True)

    def open_position(self, coin, direction, entry_price, z_score, btc_price=None, **kwargs):
        """
        Open a new position with risk validation
        If delta-neutral enabled, opens a paired position (LONG coin + SHORT BTC)

        Args:
            coin: Coin symbol
            direction: "LONG" or "SHORT"
            entry_price: Entry price
            z_score: Entry z-score
            btc_price: BTC price for hedge (required if delta_neutral_enabled)
            **kwargs: Additional metadata (velocity, half_life, etc.)

        Returns:
            dict: Position data (or pair_id if delta-neutral) or None if rejected
        """
        # Check if position can be opened
        open_positions = self.get_open_positions()

        # Risk check 1: Max concurrent positions
        if len(open_positions) >= MAX_CONCURRENT_POSITIONS:
            return None  # Portfolio heat limit reached

        # Risk check 2: Portfolio heat
        current_heat = self._calculate_portfolio_heat(open_positions)
        if current_heat >= MAX_PORTFOLIO_HEAT:
            return None  # Portfolio heat limit reached

        # Risk check 3: Don't open duplicate position for same coin
        if len(open_positions) > 0 and coin in open_positions['coin'].values:
            return None  # Already have position in this coin

        # Calculate position size (2% of capital per LEG)
        position_size_usd = self.capital * POSITION_SIZE_PCT
        position_size_coins = position_size_usd / entry_price

        # Calculate entry cost (fees + slippage)
        if APPLY_TRANSACTION_COSTS:
            entry_cost_pct = TOTAL_ENTRY_COST
            entry_cost = entry_price * entry_cost_pct
            effective_entry_price = entry_price + entry_cost if direction == "LONG" else entry_price - entry_cost
        else:
            entry_cost = 0
            effective_entry_price = entry_price

        # Generate pair ID for delta-neutral positions
        pair_id = self._generate_pair_id() if DELTA_NEUTRAL_ENABLED else None

        position_data = {
            'position_id': self._generate_position_id(),
            'pair_id': pair_id,
            'coin': coin,
            'direction': direction,
            'entry_timestamp': datetime.now(),
            'entry_price': entry_price,
            'effective_entry_price': effective_entry_price,
            'entry_z_score': z_score,
            'position_size_usd': position_size_usd,
            'position_size_coins': position_size_coins,
            'status': 'OPEN',
            'exit_timestamp': None,
            'exit_price': None,
            'exit_z_score': None,
            'exit_reason': None,
            'pnl_usd': None,
            'pnl_pct': None,
            'fees_paid': entry_cost * position_size_coins,
            'funding_cost': 0.0,
            'hold_time_hours': None,
            **kwargs
        }

        # Save primary position
        self._save_position(position_data)

        # If delta-neutral enabled, open BTC hedge
        if DELTA_NEUTRAL_ENABLED and btc_price:
            hedge_direction = "SHORT" if direction == "LONG" else "LONG"
            hedge_size_usd = position_size_usd * HEDGE_RATIO
            hedge_size_btc = hedge_size_usd / btc_price

            if APPLY_TRANSACTION_COSTS:
                hedge_entry_cost = btc_price * TOTAL_ENTRY_COST
                hedge_effective_price = btc_price + hedge_entry_cost if hedge_direction == "LONG" else btc_price - hedge_entry_cost
            else:
                hedge_entry_cost = 0
                hedge_effective_price = btc_price

            hedge_data = {
                'position_id': self._generate_position_id(),
                'pair_id': pair_id,
                'coin': 'BTC',
                'direction': hedge_direction,
                'entry_timestamp': datetime.now(),
                'entry_price': btc_price,
                'effective_entry_price': hedge_effective_price,
                'entry_z_score': z_score,
                'position_size_usd': hedge_size_usd,
                'position_size_coins': hedge_size_btc,
                'status': 'OPEN',
                'exit_timestamp': None,
                'exit_price': None,
                'exit_z_score': None,
                'exit_reason': None,
                'pnl_usd': None,
                'pnl_pct': None,
                'fees_paid': hedge_entry_cost * hedge_size_btc,
                'funding_cost': 0.0,
                'hold_time_hours': None,
                'is_hedge': True
            }

            # Save hedge position
            self._save_position(hedge_data)

            return {'pair_id': pair_id, 'primary': position_data, 'hedge': hedge_data}

        return position_data

    def close_position(self, position_id, exit_price, z_score, exit_reason):
        """
        Close an open position and calculate P&L

        Args:
            position_id: Position ID to close
            exit_price: Exit price
            z_score: Exit z-score
            exit_reason: "MEAN_REVERSION", "STOP_LOSS", "MANUAL"

        Returns:
            dict: Updated position data with P&L
        """
        positions = self._load_positions()
        position_idx = positions[positions['position_id'] == position_id].index

        if len(position_idx) == 0:
            return None  # Position not found

        position = positions.loc[position_idx[0]].to_dict()

        if position['status'] != 'OPEN':
            return None  # Position already closed

        # Calculate hold time
        entry_time = pd.to_datetime(position['entry_timestamp'])
        exit_time = datetime.now()
        hold_time_hours = (exit_time - entry_time).total_seconds() / 3600

        # Calculate exit cost (fees + slippage)
        if APPLY_TRANSACTION_COSTS:
            exit_cost_pct = TOTAL_EXIT_COST
            exit_cost = exit_price * exit_cost_pct

            # Calculate funding cost (use actual funding rate if available)
            if USE_ACTUAL_FUNDING_RATES and 'funding_rate' in position and position['funding_rate'] is not None:
                funding_periods = hold_time_hours / 8  # Funding every 8 hours
                funding_cost = abs(position['position_size_usd'] * position['funding_rate'] * funding_periods)
            else:
                funding_cost = 0  # No funding cost if data unavailable
        else:
            exit_cost = 0
            funding_cost = 0

        # Calculate P&L
        if position['direction'] == "LONG":
            price_pnl = (exit_price - position['effective_entry_price']) * position['position_size_coins']
        else:  # SHORT
            price_pnl = (position['effective_entry_price'] - exit_price) * position['position_size_coins']

        # Subtract exit fees and funding
        exit_fees = exit_cost * position['position_size_coins']
        total_fees = position['fees_paid'] + exit_fees + funding_cost
        net_pnl = price_pnl - total_fees

        pnl_pct = (net_pnl / position['position_size_usd']) * 100

        # Update position
        positions.loc[position_idx[0], 'status'] = 'CLOSED'
        positions.loc[position_idx[0], 'exit_timestamp'] = exit_time
        positions.loc[position_idx[0], 'exit_price'] = exit_price
        positions.loc[position_idx[0], 'exit_z_score'] = z_score
        positions.loc[position_idx[0], 'exit_reason'] = exit_reason
        positions.loc[position_idx[0], 'pnl_usd'] = net_pnl
        positions.loc[position_idx[0], 'pnl_pct'] = pnl_pct
        positions.loc[position_idx[0], 'fees_paid'] = total_fees
        positions.loc[position_idx[0], 'funding_cost'] = funding_cost
        positions.loc[position_idx[0], 'hold_time_hours'] = hold_time_hours

        # Save updated positions
        positions.to_parquet(self.positions_file, index=False)

        return positions.loc[position_idx[0]].to_dict()

    def get_open_positions(self):
        """Get all open positions"""
        if not os.path.exists(self.positions_file):
            return pd.DataFrame()

        positions = pd.read_parquet(self.positions_file)
        return positions[positions['status'] == 'OPEN']

    def get_closed_positions(self):
        """Get all closed positions"""
        if not os.path.exists(self.positions_file):
            return pd.DataFrame()

        positions = pd.read_parquet(self.positions_file)
        return positions[positions['status'] == 'CLOSED']

    def check_stop_loss(self, coin, current_z_score):
        """
        Check if any open position should be stopped out

        Args:
            coin: Coin symbol
            current_z_score: Current z-score

        Returns:
            bool: True if stop loss triggered
        """
        open_positions = self.get_open_positions()

        if len(open_positions) == 0 or coin not in open_positions['coin'].values:
            return False

        position = open_positions[open_positions['coin'] == coin].iloc[0]

        # Stop loss: z-score > 5 (regime change)
        if abs(current_z_score) > STOP_LOSS_Z_THRESHOLD:
            return True

        return False

    def _calculate_portfolio_heat(self, open_positions):
        """Calculate current portfolio heat (% of capital deployed)"""
        if len(open_positions) == 0:
            return 0.0

        total_deployed = open_positions['position_size_usd'].sum()
        return total_deployed / self.capital

    def _generate_position_id(self):
        """Generate unique position ID"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        return f"POS_{timestamp}"

    def _generate_pair_id(self):
        """Generate unique pair ID for delta-neutral positions"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        return f"PAIR_{timestamp}"

    def _save_position(self, position_data):
        """Save position to parquet file"""
        position_df = pd.DataFrame([position_data])

        if os.path.exists(self.positions_file):
            existing_positions = pd.read_parquet(self.positions_file)
            all_positions = pd.concat([existing_positions, position_df], ignore_index=True)
        else:
            all_positions = position_df

        all_positions.to_parquet(self.positions_file, index=False)

    def _load_positions(self):
        """Load all positions from parquet file"""
        if not os.path.exists(self.positions_file):
            return pd.DataFrame()

        return pd.read_parquet(self.positions_file)

    def get_position_summary(self):
        """Get summary of all positions"""
        positions = self._load_positions()

        if len(positions) == 0:
            return {}

        open_positions = positions[positions['status'] == 'OPEN']
        closed_positions = positions[positions['status'] == 'CLOSED']

        summary = {
            'total_positions': len(positions),
            'open_positions': len(open_positions),
            'closed_positions': len(closed_positions),
            'current_portfolio_heat': self._calculate_portfolio_heat(open_positions),
            'total_pnl': closed_positions['pnl_usd'].sum() if len(closed_positions) > 0 else 0.0,
            'avg_pnl': closed_positions['pnl_usd'].mean() if len(closed_positions) > 0 else 0.0,
            'win_rate': (closed_positions['pnl_usd'] > 0).sum() / len(closed_positions) if len(closed_positions) > 0 else 0.0,
        }

        return summary
