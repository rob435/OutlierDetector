import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from datetime import datetime, timedelta
import os

from main import (
    COINS, DATA_FOLDER, Z_SCORE_WINDOW, PRICE_CHANGE_PERIOD,
    PRIMARY_TIMEFRAME, MULTI_TIMEFRAMES_ADVANCED, MTF_REQUIRE_ALIGNMENT,
    HALF_LIFE_ENABLED, HALF_LIFE_MIN_PERIODS, VOLUME_SURGE_THRESHOLD, VOLUME_SURGE_MULTIPLIER,
    USE_MODIFIED_ZSCORE, MAD_SCALE_FACTOR, USE_REAL_MARKET_CAP, VOLUME_MCAP_NORMALIZATION,
    FUNDING_RATE_ENABLED, FUNDING_RATE_EXTREME_THRESHOLD, FUNDING_RATE_MULTIPLIER,
    OI_ENABLED, OI_THRESHOLD_USD,
    LONG_SHORT_RATIO_ENABLED, LS_RATIO_EXTREME_LONG, LS_RATIO_EXTREME_SHORT, LS_RATIO_MULTIPLIER,
    Z_SCORE_VELOCITY_ENABLED, Z_SCORE_VELOCITY_WINDOW, Z_SCORE_VELOCITY_THRESHOLD,
    PREDICTIVE_SIGNAL_ENABLED, PREDICTIVE_EARLY_ENTRY_Z, PREDICTIVE_MOMENTUM_EXHAUSTION_Z,
    PREDICTIVE_VELOCITY_REVERSAL_THRESHOLD
)
from coinmarketcap_client import CoinMarketCapClient
from binance_futures_client import BinanceFuturesClient

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OutlierDetector:
    def __init__(self):
        self.data_folder = DATA_FOLDER
        self.primary_timeframe = PRIMARY_TIMEFRAME
        self.z_score_window = Z_SCORE_WINDOW
        self.price_change_period = PRICE_CHANGE_PERIOD
        self.use_real_market_cap = USE_REAL_MARKET_CAP
        self.volume_mcap_normalization = VOLUME_MCAP_NORMALIZATION

        # Core features
        self.half_life_enabled = HALF_LIFE_ENABLED
        self.half_life_min_periods = HALF_LIFE_MIN_PERIODS
        self.volume_surge_threshold = VOLUME_SURGE_THRESHOLD
        self.volume_surge_multiplier = VOLUME_SURGE_MULTIPLIER
        self.use_modified_zscore = USE_MODIFIED_ZSCORE
        self.mad_scale_factor = MAD_SCALE_FACTOR

        # Advanced MTF settings
        self.multi_timeframes_advanced = MULTI_TIMEFRAMES_ADVANCED
        self.mtf_require_alignment = MTF_REQUIRE_ALIGNMENT

        # Derivatives signals
        self.funding_rate_enabled = FUNDING_RATE_ENABLED
        self.funding_rate_extreme_threshold = FUNDING_RATE_EXTREME_THRESHOLD
        self.funding_rate_multiplier = FUNDING_RATE_MULTIPLIER
        self.oi_enabled = OI_ENABLED
        self.oi_threshold_usd = OI_THRESHOLD_USD
        self.long_short_ratio_enabled = LONG_SHORT_RATIO_ENABLED
        self.ls_ratio_extreme_long = LS_RATIO_EXTREME_LONG
        self.ls_ratio_extreme_short = LS_RATIO_EXTREME_SHORT
        self.ls_ratio_multiplier = LS_RATIO_MULTIPLIER

        # Predictive Z-score settings
        self.z_score_velocity_enabled = Z_SCORE_VELOCITY_ENABLED
        self.z_score_velocity_window = Z_SCORE_VELOCITY_WINDOW
        self.z_score_velocity_threshold = Z_SCORE_VELOCITY_THRESHOLD
        self.predictive_signal_enabled = PREDICTIVE_SIGNAL_ENABLED
        self.predictive_early_entry_z = PREDICTIVE_EARLY_ENTRY_Z
        self.predictive_momentum_exhaustion_z = PREDICTIVE_MOMENTUM_EXHAUSTION_Z
        self.predictive_velocity_reversal_threshold = PREDICTIVE_VELOCITY_REVERSAL_THRESHOLD

        # Initialize Binance Futures derivatives client (PUBLIC API, no auth required)
        self.derivatives_client = BinanceFuturesClient()
        
        # Initialize CoinMarketCap client and fetch market cap data once
        if self.use_real_market_cap:
            try:
                self.cmc_client = CoinMarketCapClient()
                self.market_caps = self.cmc_client.get_market_data()
                logger.info(f"Loaded market cap data for {len(self.market_caps)} coins")
            except Exception as e:
                logger.warning(f"Failed to load market cap data, falling back to proxy: {e}")
                self.use_real_market_cap = False
                self.market_caps = {}
        else:
            self.market_caps = {}
        
    def load_coin_data(self, coin: str, timeframe: str = None) -> pd.DataFrame:
        """Load OHLCV data for a specific coin and timeframe"""
        if timeframe is None:
            timeframe = self.primary_timeframe
        filename = f"{self.data_folder}/{coin}_{timeframe}_ohlcv.parquet"
        
        if not os.path.exists(filename):
            logger.warning(f"Data file not found for {coin}: {filename}")
            return pd.DataFrame()
        
        try:
            df = pd.read_parquet(filename)
            df = df.sort_values('open_time').reset_index(drop=True)
            logger.info(f"Loaded {len(df)} records for {coin}")
            return df
        except Exception as e:
            logger.error(f"Error loading data for {coin}: {e}")
            return pd.DataFrame()
    
    def calculate_rolling_z_score(self, series: pd.Series, window: int) -> pd.Series:
        """Calculate rolling Z-score with configurable window"""
        if self.use_modified_zscore:
            return self.calculate_modified_z_score(series, window)
        else:
            rolling_mean = series.rolling(window=window, min_periods=1).mean()
            rolling_std = series.rolling(window=window, min_periods=1).std()
            
            # Avoid division by zero
            rolling_std = rolling_std.fillna(1.0)
            rolling_std = np.where(rolling_std == 0, 1.0, rolling_std)
            
            z_score = (series - rolling_mean) / rolling_std
            return z_score
    
    def calculate_modified_z_score(self, series: pd.Series, window: int) -> pd.Series:
        """Calculate Modified Z-score using MAD (Median Absolute Deviation) - robust to fat tails"""
        rolling_median = series.rolling(window=window, min_periods=1).median()
        
        # Calculate MAD (Median Absolute Deviation)
        abs_dev = np.abs(series - rolling_median)
        rolling_mad = abs_dev.rolling(window=window, min_periods=1).median()
        
        # Avoid division by zero
        rolling_mad = rolling_mad.fillna(1.0)
        rolling_mad = np.where(rolling_mad == 0, 1.0, rolling_mad)
        
        # Modified Z-score formula: 0.6745 * (X - median) / MAD
        modified_z_score = self.mad_scale_factor * (series - rolling_median) / rolling_mad
        return modified_z_score
    
    def calculate_btc_relative_z_score(self, coin_prices: pd.Series, btc_prices: pd.Series, window: int) -> pd.Series:
        """Calculate Z-score of coin price relative to BTC"""
        # Calculate coin/BTC ratio
        coin_btc_ratio = coin_prices / btc_prices
        
        # Calculate rolling Z-score of the ratio
        rolling_mean = coin_btc_ratio.rolling(window=window, min_periods=1).mean()
        rolling_std = coin_btc_ratio.rolling(window=window, min_periods=1).std()
        
        # Avoid division by zero
        rolling_std = rolling_std.fillna(1.0)
        rolling_std = np.where(rolling_std == 0, 1.0, rolling_std)
        
        z_score = (coin_btc_ratio - rolling_mean) / rolling_std
        return z_score
    
    def calculate_price_change(self, close_prices: pd.Series, period: int) -> pd.Series:
        """Calculate percentage price change over specified period"""
        price_change = close_prices.pct_change(periods=period) * 100
        return price_change.fillna(0)
    
    def get_market_cap(self, coin: str, close_prices: pd.Series, volumes: pd.Series) -> pd.Series:
        """Get market cap - either real from CMC or calculated proxy"""
        if self.use_real_market_cap and coin in self.market_caps and self.market_caps[coin] is not None:
            # Use real market cap from CoinMarketCap
            real_market_cap = self.market_caps[coin]
            # Return series with constant market cap value
            market_cap_series = pd.Series(real_market_cap, index=close_prices.index)
            return market_cap_series
        else:
            # Fall back to proxy calculation
            volume_24h = volumes.rolling(window=288, min_periods=1).mean()  # 24h average volume
            market_cap_proxy = close_prices * volume_24h
            return market_cap_proxy
    
    def calculate_volume_mcap_ratio(self, volumes: pd.Series, market_cap_proxy: pd.Series) -> pd.Series:
        """Calculate volume to market cap ratio using Binance futures volume and CoinMarketCap market cap"""
        # volumes: Binance futures trading volume
        # market_cap_proxy: Real market cap from CoinMarketCap API
        # Avoid division by zero
        market_cap_proxy_safe = np.where(market_cap_proxy == 0, 1.0, market_cap_proxy)
        volume_mcap_ratio = volumes / market_cap_proxy_safe
        return volume_mcap_ratio
    
    def normalize_volume_mcap_ratio(self, ratio_series: pd.Series, method: str = 'log') -> pd.Series:
        """Normalize volume/market cap ratio using log or min-max scaling"""
        if method == 'log':
            # Log normalization - better for highly skewed data
            # Add small constant to avoid log(0)
            ratio_safe = np.maximum(ratio_series, 1e-10)
            normalized = np.log10(ratio_safe)
            # Scale to 0-1 range
            min_val = normalized.min()
            max_val = normalized.max()
            if max_val > min_val:
                normalized = (normalized - min_val) / (max_val - min_val)
            else:
                normalized = pd.Series(0.5, index=ratio_series.index)  # Default to middle value
        else:  # min-max scaling
            min_val = ratio_series.min()
            max_val = ratio_series.max()
            if max_val > min_val:
                normalized = (ratio_series - min_val) / (max_val - min_val)
            else:
                normalized = pd.Series(0.5, index=ratio_series.index)  # Default to middle value
        
        return normalized
    
    
    def calculate_half_life(self, z_scores: pd.Series) -> float:
        """Calculate half-life of mean reversion in hours"""
        if not self.half_life_enabled or len(z_scores) < self.half_life_min_periods:
            return 999.0  # Return high value if insufficient data
        
        try:
            # Use AR(1) model to estimate half-life
            # ΔZ_t = α + β*Z_{t-1} + ε_t
            z_lagged = z_scores.shift(1).dropna()
            z_diff = z_scores.diff().dropna()
            
            # Align series
            common_index = z_lagged.index.intersection(z_diff.index)
            z_lagged = z_lagged.loc[common_index]
            z_diff = z_diff.loc[common_index]
            
            if len(z_lagged) < 20:
                return 0.0
            
            # Simple linear regression
            x_mean = z_lagged.mean()
            y_mean = z_diff.mean()
            
            numerator = np.sum((z_lagged - x_mean) * (z_diff - y_mean))
            denominator = np.sum((z_lagged - x_mean) ** 2)
            
            if denominator == 0:
                return 0.0
            
            beta = numerator / denominator
            
            # Half-life = -ln(2) / ln(1 + β)
            if beta >= 0:
                return 999.0  # No mean reversion
            
            half_life_periods = -np.log(2) / np.log(1 + beta)
            half_life_hours = half_life_periods / 12  # Convert 5m periods to hours (12 periods per hour)
            
            # Cap at reasonable values
            half_life_hours = np.clip(half_life_hours, 0.1, 999.0)
            
            return half_life_hours
            
        except Exception as e:
            logger.warning(f"Half-life calculation failed: {e}")
            return 0.0
    
    def calculate_volume_surge_score(self, volumes: pd.Series) -> float:
        """Calculate volume surge Z-score (abnormal volume detection)"""
        if len(volumes) < self.z_score_window:
            return 0.0

        # Calculate rolling Z-score of volume
        volume_mean = volumes.rolling(window=self.z_score_window, min_periods=1).mean()
        volume_std = volumes.rolling(window=self.z_score_window, min_periods=1).std()

        # Avoid division by zero
        volume_std = volume_std.fillna(1.0)
        volume_std = np.where(volume_std == 0, 1.0, volume_std)

        volume_z_score = (volumes - volume_mean) / volume_std

        # Get latest volume Z-score
        latest_volume_z = volume_z_score.iloc[-1] if len(volume_z_score) > 0 else 0

        return latest_volume_z

    def calculate_z_score_velocity(self, z_scores: pd.Series) -> pd.Series:
        """Calculate Z-score velocity (rate of change) - PREDICTIVE SIGNAL"""
        if not self.z_score_velocity_enabled or len(z_scores) < self.z_score_velocity_window:
            return pd.Series(0, index=z_scores.index)

        # Calculate rolling velocity (change per period)
        velocity = z_scores.diff(self.z_score_velocity_window) / self.z_score_velocity_window

        return velocity.fillna(0)

    def classify_predictive_signal(self, z_score: float, velocity: float,
                                   volume_surge_z: float) -> Dict[str, any]:
        """Classify predictive signal type using velocity only - STAT-ARB FOCUSED"""
        if not self.predictive_signal_enabled:
            return {
                'signal': 'NEUTRAL',
                'confidence': 0,
                'action': 'HOLD',
                'reason': 'Predictive signals disabled'
            }

        abs_z = abs(z_score)

        # MOMENTUM CONTINUATION: Strong Z-score + positive velocity
        if (abs_z >= self.predictive_momentum_exhaustion_z * 0.7 and
              velocity * np.sign(z_score) > self.z_score_velocity_threshold):

            confidence = min(abs(velocity) / 1.0, 1.0)

            return {
                'signal': 'MOMENTUM',
                'confidence': confidence,
                'action': 'HOLD',
                'reason': f'Strong momentum (vel={velocity:.3f})'
            }

        # REVERSAL: Velocity turning negative
        elif velocity * np.sign(z_score) < self.predictive_velocity_reversal_threshold:

            confidence = min(abs(velocity) / 0.5, 1.0)

            return {
                'signal': 'REVERSAL',
                'confidence': confidence,
                'action': 'EXIT',
                'reason': f'Reversal detected (vel={velocity:.3f})'
            }

        # MEAN REVERSION SETUP: Extreme Z-score + negative velocity
        elif (abs_z >= self.predictive_momentum_exhaustion_z and
              velocity * np.sign(z_score) < 0):

            confidence = min(abs(velocity) / 1.0, 1.0)

            return {
                'signal': 'MEAN_REVERT',
                'confidence': confidence,
                'action': 'FADE',
                'reason': f'Mean reversion (vel={velocity:.3f})'
            }

        # NEUTRAL: No clear signal
        else:
            return {
                'signal': 'NEUTRAL',
                'confidence': 0,
                'action': 'HOLD',
                'reason': 'No strong signal'
            }
    
    
    def calculate_relative_score(self, z_score: float) -> float:
        """Calculate a score from -5 to 5 based on Z-score relative to BTC"""
        # Cap the Z-score to reasonable bounds and scale to -5 to 5 range
        # Clamp Z-score between -5 and 5, then use it directly as the score
        clamped_z_score = max(-5.0, min(5.0, z_score))
        return clamped_z_score
    
    
    def calculate_advanced_mtf_confirmation(self, coin: str) -> Dict[str, any]:
        """Calculate advanced multi-timeframe confirmation using 1h/4h/1d for structural trends"""
        timeframe_signals = {}
        
        for timeframe in self.multi_timeframes_advanced:
            try:
                # Load data for this timeframe
                df = self.load_coin_data(coin, timeframe)
                btc_df = self.load_coin_data('BTC', timeframe)
                
                if df.empty or btc_df.empty:
                    timeframe_signals[timeframe] = {'z_score': 0, 'signal': 'neutral'}
                    continue
                
                # Align timestamps
                coin_df = df.set_index('open_time')
                btc_df_aligned = btc_df.set_index('open_time')
                
                merged = coin_df.join(btc_df_aligned[['close']], how='inner', rsuffix='_btc')
                
                if merged.empty or len(merged) < 50:
                    timeframe_signals[timeframe] = {'z_score': 0, 'signal': 'neutral'}
                    continue
                
                # Calculate Z-score for this timeframe
                window_periods = min(50, len(merged) // 2)  # Adaptive window
                z_score = self.calculate_btc_relative_z_score(
                    merged['close'], 
                    merged['close_btc'], 
                    window_periods
                )
                
                latest_z = z_score.iloc[-1] if len(z_score) > 0 else 0
                
                # Classify signal direction
                if latest_z > 1.5:
                    signal = 'bullish'
                elif latest_z < -1.5:
                    signal = 'bearish'
                else:
                    signal = 'neutral'
                
                timeframe_signals[timeframe] = {'z_score': latest_z, 'signal': signal}
                
            except Exception as e:
                logger.warning(f"Advanced MTF failed for {coin} on {timeframe}: {e}")
                timeframe_signals[timeframe] = {'z_score': 0, 'signal': 'neutral'}
        
        # Count aligned timeframes (same direction)
        bullish_count = sum(1 for tf in timeframe_signals.values() if tf['signal'] == 'bullish')
        bearish_count = sum(1 for tf in timeframe_signals.values() if tf['signal'] == 'bearish')
        
        aligned_count = max(bullish_count, bearish_count)
        alignment_score = aligned_count / len(self.multi_timeframes_advanced) if self.multi_timeframes_advanced else 0
        
        # Determine if requirement met
        meets_requirement = aligned_count >= self.mtf_require_alignment
        
        return {
            'aligned': aligned_count,
            'score': alignment_score,
            'meets_requirement': meets_requirement,
            'timeframes': timeframe_signals
        }
    
    def detect_outliers_single_coin(self, coin: str) -> pd.DataFrame:
        """Detect outliers for a single coin"""
        df = self.load_coin_data(coin)

        if df.empty:
            return pd.DataFrame()

        # Calculate rolling Z-score for close prices
        z_score = self.calculate_rolling_z_score(df['close'], self.z_score_window)

        # Calculate predictive signals: velocity only
        z_score_velocity = self.calculate_z_score_velocity(z_score)

        # Calculate price change using configured period
        price_change = self.calculate_price_change(df['close'], self.price_change_period)

        # Calculate volume/market cap components
        market_cap = self.get_market_cap(coin, df['close'], df['volume'])
        volume_mcap_ratio = self.calculate_volume_mcap_ratio(df['volume'], market_cap)
        volume_mcap_normalized = self.normalize_volume_mcap_ratio(volume_mcap_ratio, self.volume_mcap_normalization)

        # Calculate half-life
        half_life = self.calculate_half_life(z_score)

        # Calculate relative score (-5 to 5)
        latest_z_score = z_score.iloc[-1] if not z_score.empty else 0
        relative_score = self.calculate_relative_score(latest_z_score)

        # Fetch Binance Futures derivatives data (PUBLIC API)
        funding_rate = 0
        oi_change_1h = 0
        oi_change_24h = 0
        long_short_ratio = 0

        try:
            # Funding rate (8h intervals on Binance)
            if self.funding_rate_enabled:
                fr = self.derivatives_client.get_funding_rate(coin)
                if fr is not None:
                    funding_rate = fr

            # Open interest change % (1H and 24H)
            if self.oi_enabled:
                oi_1h = self.derivatives_client.get_open_interest_change(coin, lookback_hours=1)
                if oi_1h is not None:
                    oi_change_1h = oi_1h['change_pct']

                oi_24h = self.derivatives_client.get_open_interest_change(coin, lookback_hours=24)
                if oi_24h is not None:
                    oi_change_24h = oi_24h['change_pct']

            # Long/Short ratio (top traders by positions, 5m period)
            if self.long_short_ratio_enabled:
                ls_ratio = self.derivatives_client.get_long_short_ratio(coin, period='5m')
                if ls_ratio is not None:
                    long_short_ratio = ls_ratio

        except Exception as e:
            logger.warning(f"Failed to fetch Binance Futures derivatives data for {coin}: {e}")

        # Create results dataframe
        results = pd.DataFrame({
            'timestamp': df['open_time'],
            'coin': coin,
            'close_price': df['close'],
            'volume': df['volume'],
            'market_cap': market_cap,
            'volume_mcap_ratio': volume_mcap_ratio,
            'volume_mcap_normalized': volume_mcap_normalized,
            'z_score': z_score,
            'z_velocity': z_score_velocity,
            'price_change': price_change,
            'relative_score': relative_score,
            'half_life': half_life,
            'funding_rate': funding_rate,
            'oi_change_1h': oi_change_1h,
            'oi_change_24h': oi_change_24h,
            'long_short_ratio': long_short_ratio
        })

        return results
    
    def detect_outliers_single_coin_vs_btc(self, coin: str, btc_prices: pd.Series, btc_timestamps: pd.Series, 
                                           btc_prices_dict: Dict[str, pd.Series] = None, 
                                           btc_timestamps_dict: Dict[str, pd.Series] = None,
                                           all_coin_returns: Dict[str, pd.Series] = None) -> pd.DataFrame:
        """Detect outliers for a single coin relative to BTC with advanced features"""
        df = self.load_coin_data(coin)
        
        if df.empty:
            return pd.DataFrame()
        
        # Align timestamps with BTC data
        coin_df = df.set_index('open_time')
        btc_df_aligned = pd.DataFrame({'btc_close': btc_prices.values, 'timestamp': btc_timestamps.values}).set_index('timestamp')
        
        # Merge on timestamp
        merged = coin_df.join(btc_df_aligned, how='inner')
        
        if merged.empty:
            logger.warning(f"No matching timestamps between {coin} and BTC")
            return pd.DataFrame()
        
        # Calculate BTC-relative Z-score
        z_score = self.calculate_btc_relative_z_score(merged['close'], merged['btc_close'], self.z_score_window)

        # Calculate predictive signals: velocity only
        z_score_velocity = self.calculate_z_score_velocity(z_score)

        # Calculate price change using configured period
        price_change = self.calculate_price_change(merged['close'], self.price_change_period)
        
        # Calculate volume/market cap components
        market_cap = self.get_market_cap(coin, merged['close'], merged['volume'])
        volume_mcap_ratio = self.calculate_volume_mcap_ratio(merged['volume'], market_cap)
        volume_mcap_normalized = self.normalize_volume_mcap_ratio(volume_mcap_ratio, self.volume_mcap_normalization)
        
        # Core features
        half_life = self.calculate_half_life(z_score)
        volume_surge_z = self.calculate_volume_surge_score(merged['volume'])
        advanced_mtf = self.calculate_advanced_mtf_confirmation(coin)
        
        # Fetch Binance Futures derivatives data (40%+ of market, PUBLIC API)
        funding_rate = 0
        oi_change_1h = 0
        oi_change_24h = 0
        long_short_ratio = 0

        try:
            # Funding rate (8h intervals on Binance)
            if self.funding_rate_enabled:
                fr = self.derivatives_client.get_funding_rate(coin)
                if fr is not None:
                    funding_rate = fr

            # Open interest change % (1H and 24H)
            if self.oi_enabled:
                oi_1h = self.derivatives_client.get_open_interest_change(coin, lookback_hours=1)
                if oi_1h is not None:
                    oi_change_1h = oi_1h['change_pct']

                oi_24h = self.derivatives_client.get_open_interest_change(coin, lookback_hours=24)
                if oi_24h is not None:
                    oi_change_24h = oi_24h['change_pct']

            # Long/Short ratio (top traders by positions, 5m period)
            if self.long_short_ratio_enabled:
                ls_ratio = self.derivatives_client.get_long_short_ratio(coin, period='5m')
                if ls_ratio is not None:
                    long_short_ratio = ls_ratio

        except Exception as e:
            logger.warning(f"Failed to fetch Binance Futures derivatives data for {coin}: {e}")

        # ============================================================================
        # SCORING - ORTHOGONAL SIGNALS + PREDICTIVE CLASSIFICATION
        # ============================================================================

        latest_z_score = z_score.iloc[-1] if not z_score.empty else 0
        latest_velocity = z_score_velocity.iloc[-1] if not z_score_velocity.empty else 0

        # Classify predictive signal
        predictive_signal = self.classify_predictive_signal(
            latest_z_score,
            latest_velocity,
            volume_surge_z
        )

        base_score = self.calculate_relative_score(latest_z_score)
        
        # PRIMARY SIGNAL: BTC-relative Z-score
        final_score = base_score
        
        # VOLUME CONFIRMATION: Only boost if significant volume surge
        if volume_surge_z > self.volume_surge_threshold:
            final_score *= self.volume_surge_multiplier

        # MTF CONFIRMATION: Boost if multiple timeframes aligned
        mtf_aligned = advanced_mtf['aligned']
        if mtf_aligned >= self.mtf_require_alignment:
            final_score *= 1.2  # 20% boost for structural confirmation

        # BINANCE FUNDING RATE: Extreme funding = overleveraged positions
        if self.funding_rate_enabled and abs(funding_rate) > self.funding_rate_extreme_threshold:
            # Extreme positive funding = overleveraged longs = bearish reversal
            # Extreme negative funding = overleveraged shorts = bullish reversal
            if (latest_z_score > 0 and funding_rate > self.funding_rate_extreme_threshold) or \
               (latest_z_score < 0 and funding_rate < -self.funding_rate_extreme_threshold):
                final_score *= self.funding_rate_multiplier

        # LONG/SHORT RATIO: Binance top trader positioning
        if self.long_short_ratio_enabled:
            # Overleveraged longs = bearish reversal signal
            if long_short_ratio > self.ls_ratio_extreme_long and latest_z_score > 0:
                final_score *= self.ls_ratio_multiplier
            # Overleveraged shorts = bullish reversal signal
            elif long_short_ratio < self.ls_ratio_extreme_short and latest_z_score < 0:
                final_score *= self.ls_ratio_multiplier

        final_score = np.clip(final_score, -10, 10)

        # Create results dataframe
        results = pd.DataFrame({
            'timestamp': merged.index,
            'coin': coin,
            'close_price': merged['close'],
            'btc_price': merged['btc_close'],
            'coin_btc_ratio': merged['close'] / merged['btc_close'],
            'volume': merged['volume'],
            'market_cap': market_cap,
            'volume_mcap_ratio': volume_mcap_ratio,
            'volume_mcap_normalized': volume_mcap_normalized,
            'z_score': z_score,
            'z_velocity': z_score_velocity,
            'price_change': price_change,
            'relative_score': final_score,
            'half_life': half_life,
            'volume_surge_z': volume_surge_z,
            'mtf_aligned': mtf_aligned,
            'funding_rate': funding_rate,
            'oi_change_1h': oi_change_1h,
            'oi_change_24h': oi_change_24h,
            'long_short_ratio': long_short_ratio,
            'pred_signal': predictive_signal['signal'],
            'pred_action': predictive_signal['action'],
            'pred_confidence': predictive_signal['confidence']
        })
        
        return results.reset_index(drop=True)
    
    def detect_outliers_all_coins(self) -> pd.DataFrame:
        """Detect outliers for all coins relative to BTC"""
        all_results = []

        # Load BTC data for primary timeframe
        logger.info("Loading BTC data as reference")
        btc_df = self.load_coin_data('BTC', self.primary_timeframe)

        if btc_df.empty:
            logger.error("Cannot load BTC data for primary timeframe - required as reference")
            return pd.DataFrame()

        btc_prices = btc_df['close']
        btc_timestamps = btc_df['open_time']

        for coin in COINS:
            logger.info(f"Processing outlier detection for {coin}")

            if coin == 'BTC':
                coin_results = self.detect_outliers_single_coin(coin)
            else:
                coin_results = self.detect_outliers_single_coin_vs_btc(
                    coin, btc_prices, btc_timestamps, None, None, None
                )

            if not coin_results.empty:
                all_results.append(coin_results)

        if not all_results:
            logger.warning("No outlier results generated")
            return pd.DataFrame()

        combined_results = pd.concat(all_results, ignore_index=True)
        combined_results = combined_results.sort_values(['timestamp', 'relative_score'], ascending=[True, False])

        logger.info(f"Outlier detection completed for {len(COINS)} coins")
        return combined_results
    
    def get_current_outliers(self, top_n: int = 10) -> pd.DataFrame:
        """Get current outliers (most recent data points above threshold)"""
        all_results = self.detect_outliers_all_coins()
        
        if all_results.empty:
            return pd.DataFrame()
        
        # Get the most recent timestamp for each coin
        latest_results = all_results.groupby('coin').tail(1).reset_index(drop=True)
        
        # Filter for outliers (relative_score above 1.5 or below -1.5) and sort by score
        current_outliers = latest_results[abs(latest_results['relative_score']) >= 1.5].copy()
        current_outliers = current_outliers.sort_values('relative_score', ascending=False)
        
        return current_outliers.head(top_n)
    
    def get_all_current_scores(self) -> pd.DataFrame:
        """Get all current scores for all coins, ranked by relative score"""
        all_results = self.detect_outliers_all_coins()
        
        if all_results.empty:
            return pd.DataFrame()
        
        # Get the most recent timestamp for each coin
        latest_results = all_results.groupby('coin').tail(1).reset_index(drop=True)
        
        # Sort by relative score (highest first)
        latest_results = latest_results.sort_values('relative_score', ascending=False)
        
        return latest_results
    
    def get_outlier_summary(self) -> Dict:
        """Get summary statistics for outlier detection"""
        all_results = self.detect_outliers_all_coins()
        
        if all_results.empty:
            return {}
        
        # Get latest data for each coin
        latest_results = all_results.groupby('coin').tail(1).reset_index(drop=True)
        
        outliers = latest_results[abs(latest_results['relative_score']) >= 1.5]
        
        summary = {
            'total_coins_analyzed': len(latest_results),
            'current_outliers': len(outliers),
            'outlier_percentage': (len(outliers) / len(latest_results)) * 100 if len(latest_results) > 0 else 0,
            'avg_relative_score': latest_results['relative_score'].mean(),
            'max_relative_score': latest_results['relative_score'].max(),
            'min_relative_score': latest_results['relative_score'].min(),
            'top_outlier': latest_results.loc[latest_results['relative_score'].idxmax(), 'coin'] if not latest_results.empty else None
        }
        
        return summary

def main():
    """Main function to run outlier detection"""
    detector = OutlierDetector()
    
    # Get all current scores
    all_scores = detector.get_all_current_scores()
    
    if not all_scores.empty:
        print("\n=== ALL COINS RANKED BY RELATIVE SCORE ===")
        print(f"{'Rank':<4} {'Coin':<8} {'Price':<12} {'BTC Ratio':<12} {'Z-Score':<8} {'Price Change%':<14} {'Vol/MCap':<11} {'Score':<8}")
        print("-" * 95)
        
        for idx, (_, row) in enumerate(all_scores.iterrows(), 1):
            btc_ratio = row.get('coin_btc_ratio', 0)
            price_change = row.get('price_change', 0)
            volume_mcap_norm = row.get('volume_mcap_normalized', 0)
            relative_score = row.get('relative_score', 0)
            
            print(f"{idx:<4} {row['coin']:<8} {row['close_price']:<12.4f} {btc_ratio:<12.6f} {row['z_score']:<8.2f} {price_change:<14.2f} {volume_mcap_norm:<11.4f} {relative_score:<8.2f}")
    else:
        print("No data available")
    
    # Get summary
    summary = detector.get_outlier_summary()
    if summary:
        print(f"\n=== SUMMARY ===")
        print(f"Total coins analyzed: {summary['total_coins_analyzed']}")
        print(f"Current outliers: {summary['current_outliers']}")
        print(f"Outlier percentage: {summary['outlier_percentage']:.1f}%")
        print(f"Average relative score: {summary['avg_relative_score']:.2f}")
        print(f"Max relative score: {summary['max_relative_score']:.2f}")
        print(f"Min relative score: {summary['min_relative_score']:.2f}")
        if summary['top_outlier']:
            print(f"Top performing coin: {summary['top_outlier']}")

if __name__ == "__main__":
    main()