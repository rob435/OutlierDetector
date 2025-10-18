"""
Performance Tracker - Calculate trading metrics

Metrics calculated:
- Win rate, avg win, avg loss, profit factor
- Max drawdown, recovery time
- Sharpe ratio, Sortino ratio
- Trade duration distribution
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from paper_trading_config import (
    POSITIONS_FILE, METRICS_FILE, DAILY_PNL_FILE,
    RISK_FREE_RATE, TRADING_DAYS_PER_YEAR
)

class PerformanceTracker:
    def __init__(self):
        self.positions_file = POSITIONS_FILE
        self.metrics_file = METRICS_FILE
        self.daily_pnl_file = DAILY_PNL_FILE

    def calculate_metrics(self):
        """Calculate all performance metrics from closed positions"""
        if not os.path.exists(self.positions_file):
            return {}

        positions = pd.read_parquet(self.positions_file)
        closed_positions = positions[positions['status'] == 'CLOSED']

        if len(closed_positions) == 0:
            return {'error': 'No closed positions yet'}

        metrics = {}

        # Basic metrics
        metrics['total_trades'] = len(closed_positions)
        metrics['winning_trades'] = (closed_positions['pnl_usd'] > 0).sum()
        metrics['losing_trades'] = (closed_positions['pnl_usd'] <= 0).sum()
        metrics['win_rate'] = metrics['winning_trades'] / metrics['total_trades']

        # P&L metrics
        metrics['total_pnl_usd'] = closed_positions['pnl_usd'].sum()
        metrics['avg_pnl_usd'] = closed_positions['pnl_usd'].mean()
        metrics['avg_pnl_pct'] = closed_positions['pnl_pct'].mean()

        # Win/Loss distribution
        wins = closed_positions[closed_positions['pnl_usd'] > 0]['pnl_usd']
        losses = closed_positions[closed_positions['pnl_usd'] <= 0]['pnl_usd']

        metrics['avg_win_usd'] = wins.mean() if len(wins) > 0 else 0.0
        metrics['avg_loss_usd'] = losses.mean() if len(losses) > 0 else 0.0
        metrics['largest_win_usd'] = wins.max() if len(wins) > 0 else 0.0
        metrics['largest_loss_usd'] = losses.min() if len(losses) > 0 else 0.0

        # Profit factor (total wins / total losses)
        total_wins = wins.sum() if len(wins) > 0 else 0.0
        total_losses = abs(losses.sum()) if len(losses) > 0 else 1.0  # Avoid division by zero
        metrics['profit_factor'] = total_wins / total_losses if total_losses > 0 else 0.0

        # Expectancy (average profit per trade)
        metrics['expectancy_usd'] = (metrics['win_rate'] * metrics['avg_win_usd']) - ((1 - metrics['win_rate']) * abs(metrics['avg_loss_usd']))

        # Trade duration
        metrics['avg_hold_time_hours'] = closed_positions['hold_time_hours'].mean()
        metrics['median_hold_time_hours'] = closed_positions['hold_time_hours'].median()
        metrics['min_hold_time_hours'] = closed_positions['hold_time_hours'].min()
        metrics['max_hold_time_hours'] = closed_positions['hold_time_hours'].max()

        # Drawdown analysis
        drawdown_metrics = self._calculate_drawdown(closed_positions)
        metrics.update(drawdown_metrics)

        # Risk-adjusted returns (if enough data)
        if len(closed_positions) >= 10:
            risk_metrics = self._calculate_risk_adjusted_returns(closed_positions)
            metrics.update(risk_metrics)

        # Exit reason breakdown
        exit_reasons = closed_positions['exit_reason'].value_counts().to_dict()
        metrics['exit_reasons'] = exit_reasons

        # Save metrics
        self._save_metrics(metrics)

        return metrics

    def _calculate_drawdown(self, closed_positions):
        """Calculate maximum drawdown and recovery metrics"""
        # Sort by exit timestamp
        positions_sorted = closed_positions.sort_values('exit_timestamp')

        # Calculate cumulative P&L
        cumulative_pnl = positions_sorted['pnl_usd'].cumsum()

        # Calculate running maximum (peak)
        running_max = cumulative_pnl.expanding().max()

        # Calculate drawdown from peak
        drawdown = cumulative_pnl - running_max

        max_drawdown = drawdown.min()
        max_drawdown_pct = (max_drawdown / running_max.max() * 100) if running_max.max() > 0 else 0.0

        # Find drawdown recovery time
        if max_drawdown < 0:
            drawdown_start_idx = drawdown.idxmin()
            recovery_positions = positions_sorted.loc[drawdown_start_idx:]

            if len(recovery_positions) > 0:
                recovery_pnl = recovery_positions['pnl_usd'].cumsum()
                recovered = recovery_pnl >= abs(max_drawdown)

                if recovered.any():
                    recovery_idx = recovered.idxmax()
                    drawdown_start_time = positions_sorted.loc[drawdown_start_idx, 'exit_timestamp']
                    recovery_time = positions_sorted.loc[recovery_idx, 'exit_timestamp']
                    recovery_days = (recovery_time - drawdown_start_time).total_seconds() / 86400
                else:
                    recovery_days = None  # Still in drawdown
            else:
                recovery_days = None
        else:
            recovery_days = 0.0

        return {
            'max_drawdown_usd': max_drawdown,
            'max_drawdown_pct': max_drawdown_pct,
            'recovery_time_days': recovery_days
        }

    def _calculate_risk_adjusted_returns(self, closed_positions):
        """Calculate Sharpe and Sortino ratios"""
        # Daily returns (assume each trade represents a day for simplicity)
        returns = closed_positions['pnl_pct'] / 100  # Convert to decimal

        # Sharpe ratio
        avg_return = returns.mean()
        std_return = returns.std()
        risk_free_daily = RISK_FREE_RATE / TRADING_DAYS_PER_YEAR

        sharpe_ratio = (avg_return - risk_free_daily) / std_return if std_return > 0 else 0.0
        sharpe_ratio_annualized = sharpe_ratio * np.sqrt(TRADING_DAYS_PER_YEAR)

        # Sortino ratio (only downside volatility)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else std_return

        sortino_ratio = (avg_return - risk_free_daily) / downside_std if downside_std > 0 else 0.0
        sortino_ratio_annualized = sortino_ratio * np.sqrt(TRADING_DAYS_PER_YEAR)

        return {
            'sharpe_ratio': sharpe_ratio_annualized,
            'sortino_ratio': sortino_ratio_annualized,
            'avg_return_pct': avg_return * 100,
            'return_volatility_pct': std_return * 100
        }

    def _save_metrics(self, metrics):
        """Save metrics to JSON file"""
        # Convert numpy types to native Python types for JSON serialization
        metrics_serializable = {}
        for key, value in metrics.items():
            if isinstance(value, (np.integer, np.floating)):
                metrics_serializable[key] = float(value)
            elif pd.isna(value):
                metrics_serializable[key] = None
            else:
                metrics_serializable[key] = value

        # Add timestamp
        metrics_serializable['last_updated'] = datetime.now().isoformat()

        with open(self.metrics_file, 'w') as f:
            json.dump(metrics_serializable, f, indent=2)

    def get_metrics(self):
        """Load metrics from JSON file"""
        if not os.path.exists(self.metrics_file):
            return {}

        with open(self.metrics_file, 'r') as f:
            return json.load(f)

    def display_metrics(self):
        """Display metrics in human-readable format"""
        metrics = self.calculate_metrics()

        if 'error' in metrics:
            print(metrics['error'])
            return

        print("\n" + "=" * 60)
        print("PAPER TRADING PERFORMANCE METRICS")
        print("=" * 60)

        print(f"\nTotal Trades: {metrics['total_trades']}")
        print(f"Win Rate: {metrics['win_rate']:.1%}")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")

        print(f"\nP&L:")
        print(f"  Total P&L: ${metrics['total_pnl_usd']:,.2f}")
        print(f"  Avg P&L: ${metrics['avg_pnl_usd']:,.2f} ({metrics['avg_pnl_pct']:.2f}%)")
        print(f"  Avg Win: ${metrics['avg_win_usd']:,.2f}")
        print(f"  Avg Loss: ${metrics['avg_loss_usd']:,.2f}")
        print(f"  Largest Win: ${metrics['largest_win_usd']:,.2f}")
        print(f"  Largest Loss: ${metrics['largest_loss_usd']:,.2f}")
        print(f"  Expectancy: ${metrics['expectancy_usd']:,.2f}/trade")

        print(f"\nDrawdown:")
        print(f"  Max Drawdown: ${metrics['max_drawdown_usd']:,.2f} ({metrics['max_drawdown_pct']:.2f}%)")
        if metrics['recovery_time_days']:
            print(f"  Recovery Time: {metrics['recovery_time_days']:.1f} days")
        else:
            print(f"  Recovery Time: Still in drawdown")

        print(f"\nHold Times:")
        print(f"  Average: {metrics['avg_hold_time_hours']:.1f} hours ({metrics['avg_hold_time_hours']/24:.1f} days)")
        print(f"  Median: {metrics['median_hold_time_hours']:.1f} hours")
        print(f"  Range: {metrics['min_hold_time_hours']:.1f}h - {metrics['max_hold_time_hours']:.1f}h")

        if 'sharpe_ratio' in metrics:
            print(f"\nRisk-Adjusted Returns:")
            print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            print(f"  Sortino Ratio: {metrics['sortino_ratio']:.2f}")

        print(f"\nExit Reasons:")
        for reason, count in metrics['exit_reasons'].items():
            print(f"  {reason}: {count} trades ({count/metrics['total_trades']:.1%})")

        print("\n" + "=" * 60)
