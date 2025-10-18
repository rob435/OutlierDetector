"""
Paper Trading Engine - Main orchestration layer

Integrates signal detection → position management → performance tracking
Runs automated paper trading based on outlier signals
"""

from signal_logger import SignalLogger
from position_manager import PositionManager
from performance_tracker import PerformanceTracker
from paper_trading_config import (
    ENTRY_Z_THRESHOLD, EXIT_Z_THRESHOLD, STOP_LOSS_Z_THRESHOLD,
    USE_MULTI_FACTOR_FILTERS, MIN_Z_VELOCITY, MAX_HALF_LIFE_HOURS,
    MIN_VOLUME_SURGE_Z, MAX_ABS_FUNDING_RATE,
    USE_HALF_LIFE_STOP, HALF_LIFE_STOP_MULTIPLIER,
    ENABLE_AB_TESTING, AB_TEST_BASELINE_PCT
)
import random

class PaperTradingEngine:
    def __init__(self):
        self.signal_logger = SignalLogger()
        self.position_manager = PositionManager()
        self.performance_tracker = PerformanceTracker()

    def process_signals(self, outlier_scores):
        """
        Process outlier signals and manage paper trading positions

        Args:
            outlier_scores: DataFrame from outlier_detector with columns:
                coin, z_score, close_price, z_velocity, half_life, volume_surge_z,
                funding_rate, long_short_ratio, oi_change_1h, oi_change_24h, etc.
        """
        if outlier_scores.empty:
            return

        # Get current open positions
        open_positions = self.position_manager.get_open_positions()

        # Track positions to close
        positions_to_close = []

        # Check existing positions for exits or stop losses
        for _, position in open_positions.iterrows():
            coin = position['coin']

            # Get current signal for this coin
            current_signal = outlier_scores[outlier_scores['coin'] == coin]

            if current_signal.empty:
                continue  # No signal for this coin

            current_z = current_signal.iloc[0]['z_score']
            current_price = current_signal.iloc[0]['close_price']

            # Check stop loss (z > 5 = regime change)
            if self.position_manager.check_stop_loss(coin, current_z):
                positions_to_close.append({
                    'position_id': position['position_id'],
                    'exit_price': current_price,
                    'z_score': current_z,
                    'exit_reason': 'STOP_LOSS'
                })

                # Log stop loss signal
                self.signal_logger.log_signal(
                    'STOP_LOSS',
                    coin,
                    current_z,
                    current_price,
                    position_id=position['position_id']
                )

                continue

            # Check half-life based time stop (position not reverting)
            if USE_HALF_LIFE_STOP and 'half_life' in position and position['half_life'] is not None:
                from datetime import datetime
                import pandas as pd
                entry_time = pd.to_datetime(position['entry_timestamp'])
                hold_time_hours = (datetime.now() - entry_time).total_seconds() / 3600
                max_hold_time = position['half_life'] * HALF_LIFE_STOP_MULTIPLIER

                if hold_time_hours > max_hold_time:
                    positions_to_close.append({
                        'position_id': position['position_id'],
                        'exit_price': current_price,
                        'z_score': current_z,
                        'exit_reason': 'HALF_LIFE_STOP'
                    })

                    # Log time-based stop
                    self.signal_logger.log_signal(
                        'HALF_LIFE_STOP',
                        coin,
                        current_z,
                        current_price,
                        position_id=position['position_id'],
                        hold_time=hold_time_hours,
                        max_hold_time=max_hold_time
                    )

                    continue

            # Check mean reversion exit (z crosses back to 0)
            entry_direction = position['direction']
            entry_z = position['entry_z_score']

            # LONG position: entered when z < -2, exit when z > 0
            if entry_direction == 'LONG' and entry_z < 0 and current_z >= EXIT_Z_THRESHOLD:
                positions_to_close.append({
                    'position_id': position['position_id'],
                    'exit_price': current_price,
                    'z_score': current_z,
                    'exit_reason': 'MEAN_REVERSION'
                })

                # Log exit signal
                self.signal_logger.log_signal(
                    'EXIT',
                    coin,
                    current_z,
                    current_price,
                    position_id=position['position_id']
                )

            # SHORT position: entered when z > +2, exit when z < 0
            elif entry_direction == 'SHORT' and entry_z > 0 and current_z <= -EXIT_Z_THRESHOLD:
                positions_to_close.append({
                    'position_id': position['position_id'],
                    'exit_price': current_price,
                    'z_score': current_z,
                    'exit_reason': 'MEAN_REVERSION'
                })

                # Log exit signal
                self.signal_logger.log_signal(
                    'EXIT',
                    coin,
                    current_z,
                    current_price,
                    position_id=position['position_id']
                )

        # Close positions
        for close_data in positions_to_close:
            self.position_manager.close_position(
                close_data['position_id'],
                close_data['exit_price'],
                close_data['z_score'],
                close_data['exit_reason']
            )

        # Check for new entry signals
        for _, signal in outlier_scores.iterrows():
            coin = signal['coin']
            z_score = signal['z_score']
            price = signal['close_price']

            # Skip if already have position in this coin
            if len(open_positions) > 0 and coin in open_positions['coin'].values:
                continue

            # Entry signal: |z| > threshold
            if abs(z_score) < ENTRY_Z_THRESHOLD:
                continue

            # A/B Testing: Randomly assign to BASELINE or FILTERED variant
            if ENABLE_AB_TESTING:
                use_filters = random.random() > AB_TEST_BASELINE_PCT
                strategy_variant = "FILTERED" if use_filters else "BASELINE"
            else:
                use_filters = USE_MULTI_FACTOR_FILTERS
                strategy_variant = "FILTERED" if USE_MULTI_FACTOR_FILTERS else "BASELINE"

            # Multi-factor quality filters (PHASE 1: Data Collection)
            if use_filters:
                # Filter 1: Z-velocity (momentum toward mean reversion)
                z_velocity = signal.get('z_velocity', 0)
                if abs(z_velocity) < MIN_Z_VELOCITY:
                    self.signal_logger.log_signal(
                        'SKIPPED',
                        coin,
                        z_score,
                        price,
                        reason='LOW_Z_VELOCITY',
                        z_velocity=z_velocity
                    )
                    continue

                # Filter 2: Half-life (prefer faster mean reversion)
                half_life = signal.get('half_life', 0)
                if half_life > MAX_HALF_LIFE_HOURS:
                    self.signal_logger.log_signal(
                        'SKIPPED',
                        coin,
                        z_score,
                        price,
                        reason='SLOW_HALF_LIFE',
                        half_life=half_life
                    )
                    continue

                # Filter 3: Volume surge (avoid collapsing volume)
                volume_surge_z = signal.get('volume_surge_z', 0)
                if volume_surge_z < MIN_VOLUME_SURGE_Z:
                    self.signal_logger.log_signal(
                        'SKIPPED',
                        coin,
                        z_score,
                        price,
                        reason='COLLAPSING_VOLUME',
                        volume_surge_z=volume_surge_z
                    )
                    continue

                # Filter 4: Funding rate (avoid extreme environments)
                funding_rate = signal.get('funding_rate', 0)
                if abs(funding_rate) > MAX_ABS_FUNDING_RATE:
                    self.signal_logger.log_signal(
                        'SKIPPED',
                        coin,
                        z_score,
                        price,
                        reason='EXTREME_FUNDING',
                        funding_rate=funding_rate
                    )
                    continue

            # Determine direction
            direction = 'SHORT' if z_score > 0 else 'LONG'

            # Prepare signal metadata (store ALL signals for offline analysis)
            signal_metadata = {
                'strategy_variant': strategy_variant,  # BASELINE or FILTERED
                'z_velocity': signal.get('z_velocity', 0),
                'half_life': signal.get('half_life', 0),
                'volume_surge_z': signal.get('volume_surge_z', 0),
                'funding_rate': signal.get('funding_rate', 0),
                'long_short_ratio': signal.get('long_short_ratio', 0),
                'oi_change_1h': signal.get('oi_change_1h', 0),
                'oi_change_24h': signal.get('oi_change_24h', 0)
            }

            # Get BTC price for delta-neutral hedge
            btc_signal = outlier_scores[outlier_scores['coin'] == 'BTC']
            btc_price = btc_signal.iloc[0]['close_price'] if not btc_signal.empty else None

            # Try to open position (will validate risk limits)
            position = self.position_manager.open_position(
                coin,
                direction,
                price,
                z_score,
                btc_price=btc_price,
                **signal_metadata
            )

            if position:
                # Position opened successfully
                self.signal_logger.log_signal(
                    f'ENTRY_{direction}',
                    coin,
                    z_score,
                    price,
                    **signal_metadata
                )
            else:
                # Position rejected (risk limits reached)
                self.signal_logger.log_signal(
                    'SKIPPED',
                    coin,
                    z_score,
                    price,
                    reason='RISK_LIMIT',
                    **signal_metadata
                )

    def get_status(self):
        """Get current paper trading status"""
        signal_summary = self.signal_logger.get_signal_summary()
        position_summary = self.position_manager.get_position_summary()

        return {
            'signals': signal_summary,
            'positions': position_summary
        }

    def display_status(self):
        """Display paper trading status"""
        status = self.get_status()

        print("\n" + "=" * 60)
        print("PAPER TRADING STATUS")
        print("=" * 60)

        # Signal summary
        if status['signals']:
            print(f"\nSignals:")
            print(f"  Total: {status['signals']['total_signals']}")
            print(f"  Entries: {status['signals']['entry_signals']}")
            print(f"  Exits: {status['signals']['exit_signals']}")
            print(f"  Stop Losses: {status['signals']['stop_loss_signals']}")
            print(f"  Skipped: {status['signals']['skipped_signals']}")

        # Position summary
        if status['positions']:
            print(f"\nPositions:")
            print(f"  Open: {status['positions']['open_positions']}")
            print(f"  Closed: {status['positions']['closed_positions']}")
            print(f"  Portfolio Heat: {status['positions']['current_portfolio_heat']:.1%}")
            if status['positions']['closed_positions'] > 0:
                print(f"  Total P&L: ${status['positions']['total_pnl']:.2f}")
                print(f"  Win Rate: {status['positions']['win_rate']:.1%}")

        print("=" * 60)
