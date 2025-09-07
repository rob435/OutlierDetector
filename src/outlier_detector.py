import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from datetime import datetime, timedelta
import os

from main import (
    COINS, DATA_FOLDER, Z_SCORE_WINDOW, PRICE_CHANGE_PERIOD, 
    EWMA_ALPHA, Z_SCORE_WEIGHT, PRICE_CHANGE_WEIGHT, VOLUME_MCAP_WEIGHT,
    VOLUME_MCAP_NORMALIZATION, HIGH_LIQUIDITY_THRESHOLD, LOW_LIQUIDITY_THRESHOLD, USE_REAL_MARKET_CAP
)
from coinmarketcap_client import CoinMarketCapClient

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OutlierDetector:
    def __init__(self):
        self.data_folder = DATA_FOLDER
        self.z_score_window = Z_SCORE_WINDOW
        self.price_change_period = PRICE_CHANGE_PERIOD
        self.ewma_alpha = EWMA_ALPHA
        self.z_score_weight = Z_SCORE_WEIGHT
        self.price_change_weight = PRICE_CHANGE_WEIGHT
        self.volume_mcap_weight = VOLUME_MCAP_WEIGHT
        self.volume_mcap_normalization = VOLUME_MCAP_NORMALIZATION
        self.high_liquidity_threshold = HIGH_LIQUIDITY_THRESHOLD
        self.low_liquidity_threshold = LOW_LIQUIDITY_THRESHOLD
        self.use_real_market_cap = USE_REAL_MARKET_CAP
        
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
        
    def load_coin_data(self, coin: str) -> pd.DataFrame:
        """Load 5m OHLCV data for a specific coin"""
        filename = f"{self.data_folder}/{coin}_5m_ohlcv.parquet"
        
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
        rolling_mean = series.rolling(window=window, min_periods=1).mean()
        rolling_std = series.rolling(window=window, min_periods=1).std()
        
        # Avoid division by zero
        rolling_std = rolling_std.fillna(1.0)
        rolling_std = np.where(rolling_std == 0, 1.0, rolling_std)
        
        z_score = (series - rolling_mean) / rolling_std
        return z_score
    
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
    
    def calculate_relative_score(self, z_score: float) -> float:
        """Calculate a score from -5 to 5 based on Z-score relative to BTC"""
        # Cap the Z-score to reasonable bounds and scale to -5 to 5 range
        # Clamp Z-score between -5 and 5, then use it directly as the score
        clamped_z_score = max(-5.0, min(5.0, z_score))
        return clamped_z_score
    
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
    
    def detect_outliers_single_coin_vs_btc(self, coin: str, btc_prices: pd.Series, btc_timestamps: pd.Series) -> pd.DataFrame:
        """Detect outliers for a single coin relative to BTC"""
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
        
        # Calculate relative score (-5 to 5)
        latest_z_score = z_score.iloc[-1] if not z_score.empty else 0
        relative_score = self.calculate_relative_score(latest_z_score)
        
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
            'relative_score': relative_score
        })
        
        return results.reset_index(drop=True)
    
    def detect_outliers_all_coins(self) -> pd.DataFrame:
        """Detect outliers for all coins relative to BTC and return combined results"""
        all_results = []
        
        # Load BTC data first as reference
        logger.info("Loading BTC data as reference")
        btc_df = self.load_coin_data('BTC')
        if btc_df.empty:
            logger.error("Cannot load BTC data - required as reference")
            return pd.DataFrame()
        
        btc_prices = btc_df['close']
        btc_timestamps = btc_df['open_time']
        
        for coin in COINS:
            logger.info(f"Processing outlier detection for {coin}")
            
            if coin == 'BTC':
                # For BTC, use regular Z-score
                coin_results = self.detect_outliers_single_coin(coin)
            else:
                # For other coins, use BTC-relative Z-score
                coin_results = self.detect_outliers_single_coin_vs_btc(coin, btc_prices, btc_timestamps)
            
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