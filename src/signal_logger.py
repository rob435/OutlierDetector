"""
Signal Logger - Track all entry/exit signals for paper trading

Records every signal (taken or skipped) with complete context for analysis.
Renaissance principle: Record everything, analyze later.
"""

import pandas as pd
import os
from datetime import datetime
from paper_trading_config import SIGNALS_FILE, PAPER_TRADING_DIR

class SignalLogger:
    def __init__(self):
        self.signals_file = SIGNALS_FILE
        os.makedirs(PAPER_TRADING_DIR, exist_ok=True)

    def log_signal(self, signal_type, coin, z_score, price, **kwargs):
        """
        Log a trading signal with full context

        Args:
            signal_type: "ENTRY_LONG", "ENTRY_SHORT", "EXIT", "STOP_LOSS", "SKIPPED"
            coin: Coin symbol
            z_score: Current z-score
            price: Current price
            **kwargs: Additional signal metadata (velocity, half_life, volume_z, etc.)
        """
        signal_data = {
            'timestamp': datetime.now(),
            'signal_type': signal_type,
            'coin': coin,
            'z_score': z_score,
            'price': price,
            **kwargs  # velocity, half_life, volume_z, funding_rate, oi_change, etc.
        }

        # Convert to DataFrame
        signal_df = pd.DataFrame([signal_data])

        # Append to existing signals file
        if os.path.exists(self.signals_file):
            existing_signals = pd.read_parquet(self.signals_file)
            all_signals = pd.concat([existing_signals, signal_df], ignore_index=True)
        else:
            all_signals = signal_df

        # Save to parquet
        all_signals.to_parquet(self.signals_file, index=False)

        return signal_data

    def get_signals(self, coin=None, signal_type=None, start_date=None, end_date=None):
        """Retrieve signals with optional filtering"""
        if not os.path.exists(self.signals_file):
            return pd.DataFrame()

        signals = pd.read_parquet(self.signals_file)

        # Apply filters
        if coin:
            signals = signals[signals['coin'] == coin]
        if signal_type:
            signals = signals[signals['signal_type'] == signal_type]
        if start_date:
            signals = signals[signals['timestamp'] >= start_date]
        if end_date:
            signals = signals[signals['timestamp'] <= end_date]

        return signals

    def get_signal_summary(self):
        """Get summary statistics of all signals"""
        if not os.path.exists(self.signals_file):
            return {}

        signals = pd.read_parquet(self.signals_file)

        return {
            'total_signals': len(signals),
            'entry_signals': len(signals[signals['signal_type'].isin(['ENTRY_LONG', 'ENTRY_SHORT'])]),
            'exit_signals': len(signals[signals['signal_type'] == 'EXIT']),
            'stop_loss_signals': len(signals[signals['signal_type'] == 'STOP_LOSS']),
            'skipped_signals': len(signals[signals['signal_type'] == 'SKIPPED']),
            'unique_coins': signals['coin'].nunique(),
            'date_range': (signals['timestamp'].min(), signals['timestamp'].max()) if len(signals) > 0 else None
        }
