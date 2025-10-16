import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from datetime import datetime, timedelta
import os

from main import (
    COINS, DATA_FOLDER, Z_SCORE_WINDOW, PRICE_CHANGE_PERIOD,
    PRIMARY_TIMEFRAME, MULTI_TIMEFRAMES, MULTI_TIMEFRAMES_ADVANCED, MTF_REQUIRE_ALIGNMENT,
    HALF_LIFE_ENABLED, HALF_LIFE_MIN_PERIODS, VOLUME_SURGE_THRESHOLD, VOLUME_SURGE_MULTIPLIER,
    USE_MODIFIED_ZSCORE, MAD_SCALE_FACTOR, USE_REAL_MARKET_CAP, VOLUME_MCAP_NORMALIZATION,
    MTF_WEIGHTS,
    FUNDING_RATE_ENABLED, FUNDING_RATE_THRESHOLD, FUNDING_RATE_MULTIPLIER,
    OI_CHANGE_ENABLED, PERP_SPOT_BASIS_ENABLED, PERP_SPOT_THRESHOLD, PERP_SPOT_PENALTY,
    ORDER_BOOK_ENABLED, ORDER_BOOK_DEPTH_LEVELS, BID_ASK_IMBALANCE_THRESHOLD, ORDER_BOOK_MULTIPLIER
)
from coinmarketcap_client import CoinMarketCapClient
from binance_derivatives import BinanceDerivatives

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OutlierDetector:
    def __init__(self):
        self.data_folder = DATA_FOLDER
        self.primary_timeframe = PRIMARY_TIMEFRAME
        self.multi_timeframes = MULTI_TIMEFRAMES
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
        
        # MTF settings
        self.multi_timeframes_advanced = MULTI_TIMEFRAMES_ADVANCED
        self.mtf_require_alignment = MTF_REQUIRE_ALIGNMENT
        
        # Derivatives signals
        self.funding_rate_enabled = FUNDING_RATE_ENABLED
        self.funding_rate_threshold = FUNDING_RATE_THRESHOLD
        self.funding_rate_multiplier = FUNDING_RATE_MULTIPLIER
        self.oi_change_enabled = OI_CHANGE_ENABLED
        self.perp_spot_basis_enabled = PERP_SPOT_BASIS_ENABLED
        self.perp_spot_threshold = PERP_SPOT_THRESHOLD
        self.perp_spot_penalty = PERP_SPOT_PENALTY
        
        # Order book signals
        self.order_book_enabled = ORDER_BOOK_ENABLED
        self.order_book_depth_levels = ORDER_BOOK_DEPTH_LEVELS
        self.bid_ask_imbalance_threshold = BID_ASK_IMBALANCE_THRESHOLD
        self.order_book_multiplier = ORDER_BOOK_MULTIPLIER
        
        # MTF weights
        self.mtf_weights = MTF_WEIGHTS
        
        # Initialize derivatives client
        self.derivatives_client = BinanceDerivatives()
        
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
    
    def apply_ewma(self, series: pd.Series, alpha: float) -> pd.Series:
        """Apply Exponentially Weighted Moving Average"""
        return series.ewm(alpha=alpha, adjust=False).mean()
    
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
            half_life_hours = half_life_periods / 60  # Convert 1m periods to hours
            
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
    
    def calculate_multi_timeframe_confirmation(self, coin: str, btc_prices_dict: Dict[str, pd.Series], btc_timestamps_dict: Dict[str, pd.Series]) -> float:
        """Calculate multi-timeframe confirmation score"""
        confirmation_scores = {}
        
        for timeframe in self.multi_timeframes:
            try:
                df = self.load_coin_data(coin, timeframe)
                if df.empty:
                    continue
                
                # Align with BTC data for this timeframe
                coin_df = df.set_index('open_time')
                btc_df_aligned = pd.DataFrame({
                    'btc_close': btc_prices_dict[timeframe].values, 
                    'timestamp': btc_timestamps_dict[timeframe].values
                }).set_index('timestamp')
                
                merged = coin_df.join(btc_df_aligned, how='inner')
                
                if merged.empty:
                    continue
                
                # Calculate Z-score for this timeframe
                z_score = self.calculate_btc_relative_z_score(
                    merged['close'], 
                    merged['btc_close'], 
                    int(self.z_score_window / (self._timeframe_to_minutes(timeframe) / self._timeframe_to_minutes(self.primary_timeframe)))
                )
                
                # Get latest z-score
                latest_z = z_score.iloc[-1] if len(z_score) > 0 else 0
                
                # Normalize to 0-1 where 1 = strong outlier signal
                confirmation_score = np.clip(np.abs(latest_z) / 5.0, 0, 1)
                confirmation_scores[timeframe] = confirmation_score
                
            except Exception as e:
                logger.warning(f"Multi-timeframe analysis failed for {coin} on {timeframe}: {e}")
                continue
        
        # Calculate weighted confirmation
        weighted_confirmation = 0
        total_weight = 0
        for timeframe, score in confirmation_scores.items():
            weight = self.mtf_weights.get(timeframe, 0)
            weighted_confirmation += score * weight
            total_weight += weight
        
        if total_weight > 0:
            final_confirmation = weighted_confirmation / total_weight
        else:
            final_confirmation = 0
        
        return final_confirmation
    
    def _timeframe_to_minutes(self, timeframe: str) -> int:
        """Convert timeframe string to minutes"""
        unit = timeframe[-1]
        value = int(timeframe[:-1])
        
        if unit == 'm':
            return value
        elif unit == 'h':
            return value * 60
        elif unit == 'd':
            return value * 1440
        else:
            return 1
    
    def calculate_weighted_outlier_score(self, z_score: pd.Series, price_change: pd.Series, 
                                       volume_mcap_normalized: pd.Series = None) -> pd.Series:
        """Calculate weighted outlier score combining Z-score, price change, and volume/mcap ratio"""
        # Normalize price change to similar scale as Z-score
        price_change_abs = np.abs(price_change)
        price_change_normalized = (price_change_abs - price_change_abs.rolling(window=self.z_score_window, min_periods=1).mean()) / price_change_abs.rolling(window=self.z_score_window, min_periods=1).std().fillna(1.0)
        
        # Calculate base weighted score
        base_score = (self.z_score_weight * np.abs(z_score) + 
                     self.price_change_weight * np.abs(price_change_normalized))
        
        if volume_mcap_normalized is not None:
            # Add volume/mcap component as a multiplier (not additive)
            # Higher volume/mcap ratio amplifies the signal
            volume_multiplier = 1 + (self.volume_mcap_weight * volume_mcap_normalized)
            weighted_score = base_score * volume_multiplier
        else:
            weighted_score = base_score
        
        return weighted_score
    
    def detect_outliers_single_coin(self, coin: str) -> pd.DataFrame:
        """Detect outliers for a single coin"""
        df = self.load_coin_data(coin)
        
        if df.empty:
            return pd.DataFrame()
        
        # Calculate rolling Z-score for close prices
        z_score = self.calculate_rolling_z_score(df['close'], self.z_score_window)
        
        # Calculate price change using configured period
        price_change = self.calculate_price_change(df['close'], self.price_change_period)
        
        # Calculate volume/market cap components
        market_cap = self.get_market_cap(coin, df['close'], df['volume'])
        volume_mcap_ratio = self.calculate_volume_mcap_ratio(df['volume'], market_cap)
        volume_mcap_normalized = self.normalize_volume_mcap_ratio(volume_mcap_ratio, self.volume_mcap_normalization)
        
        # Calculate relative score (-5 to 5)
        latest_z_score = z_score.iloc[-1] if not z_score.empty else 0
        relative_score = self.calculate_relative_score(latest_z_score)
        
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
            'price_change': price_change,
            'relative_score': relative_score
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
        
        # Fetch derivatives data (live)
        funding_rate = 0
        funding_rate_z = 0
        oi_change_pct = 0
        oi_change_z = 0
        perp_spot_basis = 0
        bid_ask_imbalance = 0
        
        try:
            # Funding rate
            if self.funding_rate_enabled:
                fr = self.derivatives_client.get_funding_rate(coin)
                if fr is not None:
                    funding_rate = fr
            
            # Open interest (fetch current, calculate change later if we store historical)
            if self.oi_change_enabled:
                oi = self.derivatives_client.get_open_interest(coin)
                # For now, we don't have historical OI, so oi_change = 0
                # TODO: Store OI history for proper change calculation
            
            # Perp-spot basis
            if self.perp_spot_basis_enabled:
                basis = self.derivatives_client.get_perp_spot_basis(coin)
                if basis is not None:
                    perp_spot_basis = basis
            
            # Order book imbalance
            if self.order_book_enabled:
                ob = self.derivatives_client.get_order_book_depth(coin, self.order_book_depth_levels)
                if ob is not None:
                    bid_ask_imbalance = ob['imbalance']
        
        except Exception as e:
            logger.warning(f"Failed to fetch derivatives data for {coin}: {e}")
        
        # Calculate multi-timeframe confirmation if data available (legacy)
        mtf_confirmation = 0.0
        if btc_prices_dict and btc_timestamps_dict:
            mtf_confirmation = self.calculate_multi_timeframe_confirmation(coin, btc_prices_dict, btc_timestamps_dict)
        
        # ============================================================================
        # SCORING - ORTHOGONAL SIGNALS
        # ============================================================================
        
        latest_z_score = z_score.iloc[-1] if not z_score.empty else 0
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
        
        # FUNDING RATE: Extreme funding = contrarian reversal signal
        if self.funding_rate_enabled and abs(funding_rate) > 0.1:  # >0.1% is significant
            # Positive funding = longs paying shorts = overleveraged longs = bearish
            # Negative funding = shorts paying longs = bullish
            # If Z-score and funding are opposite signs = reversal confirmation
            if (latest_z_score > 0 and funding_rate > 0.2) or (latest_z_score < 0 and funding_rate < -0.1):
                final_score *= self.funding_rate_multiplier
        
        # PERP-SPOT BASIS: Excessive premium/discount = speculation warning
        if self.perp_spot_basis_enabled and abs(perp_spot_basis) > self.perp_spot_threshold:
            # High premium = excessive speculation = reduce score
            final_score *= self.perp_spot_penalty
        
        # ORDER BOOK IMBALANCE: Strong directional bias
        if self.order_book_enabled and abs(bid_ask_imbalance) > self.bid_ask_imbalance_threshold:
            # Imbalance in same direction as Z-score = confirmation
            if (latest_z_score > 0 and bid_ask_imbalance > 0) or (latest_z_score < 0 and bid_ask_imbalance < 0):
                final_score *= self.order_book_multiplier
        
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
            'price_change': price_change,
            'relative_score': final_score,
            'half_life': half_life,
            'volume_surge_z': volume_surge_z,
            'mtf_aligned': mtf_aligned,
            'funding_rate': funding_rate,
            'perp_spot_basis': perp_spot_basis,
            'bid_ask_imbalance': bid_ask_imbalance
        })
        
        return results.reset_index(drop=True)
    
    def detect_outliers_all_coins(self) -> pd.DataFrame:
        """Detect outliers for all coins relative to BTC with multi-timeframe analysis"""
        all_results = []
        
        # Load BTC data for all timeframes
        logger.info("Loading BTC data as reference for all timeframes")
        btc_prices_dict = {}
        btc_timestamps_dict = {}
        
        for timeframe in self.multi_timeframes:
            btc_df = self.load_coin_data('BTC', timeframe)
            if not btc_df.empty:
                btc_prices_dict[timeframe] = btc_df['close']
                btc_timestamps_dict[timeframe] = btc_df['open_time']
        
        # Check if primary timeframe loaded
        if self.primary_timeframe not in btc_prices_dict or btc_prices_dict[self.primary_timeframe].empty:
            logger.error("Cannot load BTC data for primary timeframe - required as reference")
            return pd.DataFrame()
        
        btc_prices = btc_prices_dict[self.primary_timeframe]
        btc_timestamps = btc_timestamps_dict[self.primary_timeframe]
        
        for coin in COINS:
            logger.info(f"Processing outlier detection for {coin}")
            
            if coin == 'BTC':
                coin_results = self.detect_outliers_single_coin(coin)
            else:
                coin_results = self.detect_outliers_single_coin_vs_btc(
                    coin, btc_prices, btc_timestamps, 
                    btc_prices_dict, btc_timestamps_dict,
                    None
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