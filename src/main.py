import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path
from numba import jit, prange
import numba
import multiprocessing as mp

DATA_FOLDER = "/Applications/PairsAlgo/data"
WINDOW_HOURS = 24
Z_SCORE_THRESHOLD = 3.0
POSITIVE_ONLY = False
BTC_MOVE_THRESHOLD = 0.3
ENABLE_EWMA_SMOOTHING = True
EWMA_ALPHA = 0.000001
TOP_N_OUTLIERS = 10
TARGET_TIMESTAMP = "2025-08-09 09:45:00"
LOG_LEVEL = logging.INFO


COINS = [
    "BTC", "ETH", "BNB", "XRP", "ADA", "SOL", "DOGE", "DOT", "PEPE",
    "LTC", "AVAX", "UNI", "LINK", "ATOM", "ETC", "BCH", "ALGO", "XLM", "VET",
    "FIL", "ICP", "INJ", "TRX", "AAVE", "GRT", "CAKE", "RUNE", "XTZ", "MKR",
    "SUSHI", "CRV", "ENJ", "SEI", "NEAR", "TAO", "HBAR", "FLOW", "ENS", "IMX",
    "APT", "LDO", "APE", "GALA", "ARB", "SUI", "FLOKI", "WLD", "DYDX",
    "TIA", "WIF", "BONK", "PYTH", "JUP", "TON", "ETHFI"
]

logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@jit(nopython=True, cache=True, fastmath=True)
def _ewma_smooth_jit(values, alpha):
    n = len(values)
    smoothed = np.full(n, 0.0)
    
    first_valid_idx = 0
    for i in range(n):
        if not np.isnan(values[i]):
            smoothed[i] = values[i]
            first_valid_idx = i
            break
    
    prev_val = smoothed[first_valid_idx]
    one_minus_alpha = 1.0 - alpha
    
    for i in range(first_valid_idx + 1, n):
        if not np.isnan(values[i]):
            prev_val = alpha * values[i] + one_minus_alpha * prev_val
        smoothed[i] = prev_val
    
    return smoothed

@jit(nopython=True, parallel=True, cache=True, fastmath=True)
def _batch_ewma_smooth_jit(data_batch, alpha):
    n_series, n_points = data_batch.shape
    result = np.zeros_like(data_batch)
    
    for series_idx in numba.prange(n_series):
        result[series_idx] = _ewma_smooth_jit(data_batch[series_idx], alpha)
    
    return result

@jit(nopython=True, parallel=True, cache=True, fastmath=True)
def _batch_calculate_returns(price_matrix):
    n_series, n_points = price_matrix.shape
    returns = np.zeros_like(price_matrix)
    
    for series_idx in numba.prange(n_series):
        prices = price_matrix[series_idx]
        for i in range(1, n_points):
            if not np.isnan(prices[i]) and not np.isnan(prices[i-1]) and prices[i-1] != 0:
                returns[series_idx, i] = (prices[i] - prices[i-1]) / prices[i-1]
            else:
                returns[series_idx, i] = 0.0
    
    return returns

@jit(nopython=True, cache=True, fastmath=True)
def _calculate_rolling_stats_with_ewma_jit(values, window, ewma_alpha, enable_ewma):
    n = len(values)
    rolling_means = np.full(n, 0.0)
    rolling_stds = np.full(n, 0.0)
    min_periods = window // 2
    
    for i in range(window-1, n):
        start_idx = i - window + 1
        window_data = values[start_idx:i + 1]
        
        valid_count = 0
        sum_val = 0.0
        for j in range(window):
            if not np.isnan(window_data[j]):
                valid_count += 1
                sum_val += window_data[j]
        
        if valid_count >= min_periods:
            mean_val = sum_val / valid_count
            rolling_means[i] = mean_val
            
            sum_sq_diff = 0.0
            for j in range(window):
                if not np.isnan(window_data[j]):
                    diff = window_data[j] - mean_val
                    sum_sq_diff += diff * diff
            
            rolling_stds[i] = np.sqrt(sum_sq_diff / valid_count)
    
    if enable_ewma:
        smoothed_means = np.copy(rolling_means)
        for i in range(window, n):
            if rolling_means[i] != 0.0 and rolling_means[i-1] != 0.0:
                smoothed_means[i] = ewma_alpha * rolling_means[i] + (1 - ewma_alpha) * smoothed_means[i-1]
        
        smoothed_stds = np.copy(rolling_stds)
        for i in range(window, n):
            if rolling_stds[i] != 0.0 and rolling_stds[i-1] != 0.0:
                smoothed_stds[i] = ewma_alpha * rolling_stds[i] + (1 - ewma_alpha) * smoothed_stds[i-1]
        
        return smoothed_means, smoothed_stds
    else:
        return rolling_means, rolling_stds

@jit(nopython=True, cache=True, fastmath=True)
def _calculate_filtered_z_score_jit(values, btc_values, window, btc_threshold, ewma_alpha, enable_ewma):
    n = len(values)
    filtered_z_scores = np.full(n, 0.0)
    last_valid_z = 0.0
    
    min_periods = window // 2
    btc_threshold_scaled = btc_threshold / 100.0
    
    rolling_means, rolling_stds = _calculate_rolling_stats_with_ewma_jit(
        values, window, ewma_alpha, enable_ewma
    )
    
    for i in range(window-1, n):
        if np.isnan(values[i]) or np.isnan(btc_values[i]):
            filtered_z_scores[i] = last_valid_z
            continue
            
        if rolling_stds[i] > 1e-8:
            current_z = (values[i] - rolling_means[i]) / rolling_stds[i]
        else:
            current_z = 0.0
        
        btc_move = np.abs(btc_values[i]) >= btc_threshold_scaled
        
        update_condition = (
            btc_move or
            i == window-1 or
            (i > 0 and i % 12 == 0)
        )
        
        if update_condition:
            last_valid_z = current_z
        
        filtered_z_scores[i] = last_valid_z
    
    for i in range(window-1):
        filtered_z_scores[i] = 0.0
    
    return filtered_z_scores

class PairsTradingAlgorithm:
    def __init__(self, data_folder: str = DATA_FOLDER, window_hours: int = WINDOW_HOURS):
        self.data_folder = Path(data_folder)
        self.window_hours = window_hours
        self.window_periods = window_hours * 12
        self.data_cache = {}
        self.memory_limit = 1024 * 1024 * 1024
        self.chunk_size = 10000
        
    def load_coin_data(self, coin: str) -> Optional[pd.DataFrame]:
        if coin in self.data_cache:
            return self.data_cache[coin]
            
        file_path = self.data_folder / f"{coin}_5m_ohlcv.parquet"
        
        if not file_path.exists():
            return None
            
        try:
            self.manage_memory_cache()
            
            df = pd.read_parquet(
                file_path, 
                columns=['open_time', 'close'], 
                use_pandas_metadata=True
            )
            
            df['close'] = pd.to_numeric(df['close'], downcast='float')
            df = df.sort_values('open_time').reset_index(drop=True)
            
            self.data_cache[coin] = df
            return df
        except Exception as e:
            return None
    
    def batch_load_coin_data(self, coins: List[str]) -> Dict[str, pd.DataFrame]:
        data_dict = {}
        
        with mp.Pool(processes=min(mp.cpu_count(), len(coins))) as pool:
            tasks = [(coin, str(self.data_folder / f"{coin}_5m_ohlcv.parquet")) for coin in coins]
            
            results = pool.starmap(self._load_single_coin_worker, tasks)
            
            for coin, df in results:
                if df is not None:
                    data_dict[coin] = df
                    self.data_cache[coin] = df
        
        return data_dict
    
    @staticmethod
    def _load_single_coin_worker(coin: str, file_path: str) -> Tuple[str, Optional[pd.DataFrame]]:
        if not Path(file_path).exists():
            return coin, None
            
        try:
            df = pd.read_parquet(file_path, columns=['open_time', 'close'], use_pandas_metadata=True)
            df = df.sort_values('open_time').reset_index(drop=True)
            return coin, df
        except Exception as e:
            return coin, None
    
    def calculate_returns(self, df: pd.DataFrame) -> pd.Series:
        """Calculate percentage returns from close prices with memory optimization"""
        if len(df) > self.chunk_size:
            chunks = []
            for i in range(0, len(df), self.chunk_size):
                chunk = df.iloc[i:i + self.chunk_size]
                chunk_returns = chunk['close'].pct_change()
                chunks.append(chunk_returns)
            return pd.concat(chunks, ignore_index=False)
        else:
            return df['close'].pct_change()
    
    def process_data_in_chunks(self, data: pd.DataFrame, processing_func, chunk_size: int = None) -> pd.DataFrame:
        """
        Process large datasets in chunks to optimize memory usage
        """
        if chunk_size is None:
            chunk_size = self.chunk_size
            
        if len(data) <= chunk_size:
            return processing_func(data)
        
        results = []
        for i in range(0, len(data), chunk_size):
            chunk = data.iloc[i:i + chunk_size].copy()
            chunk_result = processing_func(chunk)
            results.append(chunk_result)
            
            del chunk
            import gc
            gc.collect()
        
        return pd.concat(results, ignore_index=False)
    
    def manage_memory_cache(self):
        """
        Manage memory usage by clearing old cache entries when memory limit is exceeded
        """
        import sys
        
        total_memory = sum(
            sys.getsizeof(df) + df.memory_usage(deep=True).sum() 
            for df in self.data_cache.values()
        )
        
        if total_memory > self.memory_limit:
            cache_items = list(self.data_cache.items())
            for i in range(len(cache_items) // 2):
                del self.data_cache[cache_items[i][0]]
            
            import gc
            gc.collect()
    
    def calculate_relative_returns(self, alt_returns: pd.Series, btc_returns: pd.Series) -> pd.Series:
        """Calculate relative returns: alt_return - btc_return with proper alignment"""
        common_index = alt_returns.index.intersection(btc_returns.index)
        
        if len(common_index) == 0:
            if len(alt_returns) <= len(btc_returns):
                aligned_alt = alt_returns
                aligned_btc = btc_returns.reindex(alt_returns.index, method='ffill')
            else:
                aligned_alt = alt_returns.reindex(btc_returns.index, method='ffill') 
                aligned_btc = btc_returns
        else:
            aligned_alt = alt_returns.reindex(common_index, method='ffill')
            aligned_btc = btc_returns.reindex(common_index, method='ffill')
        
        relative_returns = aligned_alt - aligned_btc
        return relative_returns.dropna()
    
    def calculate_rolling_z_score(self, series: pd.Series, window: int) -> pd.Series:
        """
        Calculate rolling z-score using optimized batch processing
        """
        values = series.values
        n = len(values)
        
        rolling_means = np.full(n, np.nan)
        rolling_stds = np.full(n, np.nan)
        z_scores = np.full(n, 0.0)
        
        min_periods = window // 2
        
        for i in range(window-1, n):
            start_idx = max(0, i - window + 1)
            window_data = values[start_idx:i + 1]
            
            valid_data = window_data[~np.isnan(window_data)]
            if len(valid_data) >= min_periods:
                rolling_means[i] = np.mean(valid_data)
                rolling_stds[i] = np.std(valid_data, ddof=0)
        
        valid_mask = (~np.isnan(rolling_stds)) & (rolling_stds > 1e-8)
        z_scores[valid_mask] = (values[valid_mask] - rolling_means[valid_mask]) / rolling_stds[valid_mask]
        
        return pd.Series(z_scores, index=series.index)
    
    @staticmethod
    @jit(nopython=True, parallel=True, cache=True, fastmath=True)
    def _batch_rolling_stats(data_matrix, window):
        """
        JIT-compiled batch rolling statistics calculation for multiple series
        """
        n_series, n_points = data_matrix.shape
        means = np.zeros((n_series, n_points))
        stds = np.zeros((n_series, n_points))
        
        for series_idx in numba.prange(n_series):
            values = data_matrix[series_idx]
            min_periods = window // 2
            
            for i in range(window-1, n_points):
                start_idx = i - window + 1
                window_data = values[start_idx:i + 1]
                
                valid_count = 0
                sum_val = 0.0
                for j in range(window):
                    if not np.isnan(window_data[j]):
                        valid_count += 1
                        sum_val += window_data[j]
                
                if valid_count >= min_periods:
                    mean_val = sum_val / valid_count
                    means[series_idx, i] = mean_val
                    
                    sum_sq_diff = 0.0
                    for j in range(window):
                        if not np.isnan(window_data[j]):
                            diff = window_data[j] - mean_val
                            sum_sq_diff += diff * diff
                    
                    stds[series_idx, i] = np.sqrt(sum_sq_diff / valid_count)
        
        return means, stds
    
    def apply_ewma_smoothing(self, series: pd.Series) -> pd.Series:
        """
        Apply EWMA smoothing to reduce high-frequency noise
        """
        if not ENABLE_EWMA_SMOOTHING:
            return series
        
        smoothed_values = _ewma_smooth_jit(series.values, EWMA_ALPHA)
        return pd.Series(smoothed_values, index=series.index)
    
    def batch_apply_ewma_smoothing(self, series_dict: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        Apply EWMA smoothing to multiple series in batch for better performance
        """
        if not ENABLE_EWMA_SMOOTHING or not series_dict:
            return series_dict
        
        keys = list(series_dict.keys())
        max_len = max(len(series) for series in series_dict.values())
        
        data_matrix = np.full((len(keys), max_len), np.nan)
        for i, key in enumerate(keys):
            series = series_dict[key]
            data_matrix[i, :len(series)] = series.values
        
        smoothed_matrix = _batch_ewma_smooth_jit(data_matrix, EWMA_ALPHA)
        
        result = {}
        for i, key in enumerate(keys):
            series = series_dict[key]
            smoothed_values = smoothed_matrix[i, :len(series)]
            result[key] = pd.Series(smoothed_values, index=series.index)
        
        return result

    def calculate_filtered_z_score(self, series: pd.Series, btc_returns: pd.Series, window: int) -> pd.Series:
        """
        Calculate rolling z-score with BTC filtering and EWMA smoothing integration
        """
        common_index = series.index.intersection(btc_returns.index)
        
        if len(common_index) == 0:
            aligned_btc = btc_returns.reindex(series.index, method='ffill')
            aligned_series = series
        else:
            aligned_series = series.reindex(common_index, method='ffill')
            aligned_btc = btc_returns.reindex(common_index, method='ffill')
        
        aligned_series = aligned_series.ffill()
        aligned_btc = aligned_btc.ffill()
        
        min_len = min(len(aligned_series), len(aligned_btc))
        if min_len == 0:
            return pd.Series([], dtype=float)
        
        series_values = aligned_series.values[:min_len]
        btc_values = aligned_btc.values[:min_len]
        
        filtered_z_scores = _calculate_filtered_z_score_jit(
            series_values, 
            btc_values, 
            window, 
            BTC_MOVE_THRESHOLD,
            EWMA_ALPHA,
            ENABLE_EWMA_SMOOTHING
        )
        
        return pd.Series(filtered_z_scores, index=aligned_series.index[:min_len])
    
    def identify_outliers(self, z_scores: pd.Series, threshold: float = Z_SCORE_THRESHOLD) -> pd.Series:
        """Identify outliers based on z-score threshold"""
        return (z_scores > threshold) | (z_scores < -threshold)
    

    def process_pairs_data(self, target_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Process all pairs data and calculate relative returns and z-scores
        Uses optimized batch loading and parallel processing for speed
        """
        coins_to_load = [coin for coin in COINS if coin not in self.data_cache]
        if coins_to_load:
            batch_data = self.batch_load_coin_data(coins_to_load)
            self.data_cache.update(batch_data)
        
        btc_data = self.load_coin_data("BTC")
        if btc_data is None:
            raise ValueError("BTC data not found - cannot proceed")
        
        if target_date:
            target_dt = pd.to_datetime(target_date)
            start_dt = target_dt - pd.Timedelta(hours=12)
            btc_data = btc_data[btc_data['open_time'] >= start_dt]
        
        btc_returns = self.calculate_returns(btc_data)
        results = {}
        
        max_workers = min(mp.cpu_count(), len(COINS))
        
        with mp.Pool(processes=max_workers) as pool:
            tasks = [
                (coin, btc_data.copy(), btc_returns.copy(), target_date, self.window_periods, EWMA_ALPHA, ENABLE_EWMA_SMOOTHING)
                for coin in COINS if coin != "BTC" and coin in self.data_cache
            ]
            
            results_list = pool.starmap(self._process_single_coin_worker, tasks)
            
            for result in results_list:
                if result is not None:
                    coin, result_df = result
                    results[coin] = result_df
        
        return results
    
    @staticmethod
    def _process_single_coin_worker(coin: str, btc_data: pd.DataFrame, btc_returns: pd.Series, 
                                   target_date: Optional[str], window_periods: int, 
                                   ewma_alpha: float, enable_ewma: bool) -> Optional[Tuple[str, pd.DataFrame]]:
        """Static worker function for parallel coin processing"""
        file_path = Path(DATA_FOLDER) / f"{coin}_5m_ohlcv.parquet"
        
        if not file_path.exists():
            return None
            
        try:
            alt_data = pd.read_parquet(file_path, columns=['open_time', 'close'])
            alt_data = alt_data.sort_values('open_time').reset_index(drop=True)
        except:
            return None
        
        if target_date:
            target_dt = pd.to_datetime(target_date) 
            start_dt = target_dt - pd.Timedelta(hours=12)
            extended_data = alt_data[alt_data['open_time'] >= start_dt]
            
            alt_returns_full = extended_data['close'].pct_change()
            
            common_index = alt_returns_full.index.intersection(btc_returns.index)
            if len(common_index) > 0:
                aligned_alt = alt_returns_full.reindex(common_index).ffill()
                aligned_btc = btc_returns.reindex(common_index).ffill()
            else:
                min_len = min(len(alt_returns_full), len(btc_returns))
                aligned_alt = alt_returns_full.iloc[:min_len]
                aligned_btc = btc_returns.iloc[:min_len].reindex(aligned_alt.index).ffill()
            relative_returns_full = (aligned_alt - aligned_btc).dropna()
            
            z_scores_full = pd.Series(
                _calculate_filtered_z_score_jit(
                    relative_returns_full.values, 
                    aligned_btc.values, 
                    window_periods, 
                    BTC_MOVE_THRESHOLD,
                    ewma_alpha,
                    enable_ewma
                ), 
                index=relative_returns_full.index
            )
            
            target_mask = extended_data['open_time'].dt.date == target_dt.date()
            alt_data = extended_data[target_mask]
            alt_returns = alt_returns_full[target_mask]
            relative_returns = relative_returns_full[target_mask]
            z_scores = z_scores_full[target_mask]
        else:
            if target_date:
                alt_data = alt_data[alt_data['open_time'].dt.date == pd.to_datetime(target_date).date()]
            
            if len(alt_data) == 0:
                return None
            
            alt_returns = alt_data['close'].pct_change()
            common_index = alt_returns.index.intersection(btc_returns.index)
            if len(common_index) > 0:
                aligned_alt = alt_returns.reindex(common_index).ffill()
                aligned_btc = btc_returns.reindex(common_index).ffill()
            else:
                min_len = min(len(alt_returns), len(btc_returns))
                aligned_alt = alt_returns.iloc[:min_len]
                aligned_btc = btc_returns.iloc[:min_len].reindex(aligned_alt.index).ffill()
            relative_returns = (aligned_alt - aligned_btc).dropna()
            
            z_scores = pd.Series(
                _calculate_filtered_z_score_jit(
                    relative_returns.values, 
                    aligned_btc.values, 
                    window_periods, 
                    BTC_MOVE_THRESHOLD,
                    EWMA_ALPHA,
                    ENABLE_EWMA_SMOOTHING
                ), 
                index=relative_returns.index
            )
        
        if len(alt_data) == 0:
            return None
        
        outliers = (z_scores > Z_SCORE_THRESHOLD) | (z_scores < -Z_SCORE_THRESHOLD)
        
        result_df = pd.DataFrame({
            'timestamp': alt_data['open_time'],
            'alt_return': alt_returns,
            'btc_return': btc_returns.reindex(alt_returns.index, method='ffill'),
            'relative_return': relative_returns,
            'z_score': z_scores,
            'is_outlier': outliers,
            'close_price': alt_data['close']
        })
        
        result_df = result_df.dropna()
        
        return (coin, result_df)
    
    def get_top_coins_by_z_score(self, results: Dict[str, pd.DataFrame], n: int = 10) -> List[Dict]:
        """
        Get top N coins by highest z-scores
        
        Args:
            results: Dictionary of processed data
            n: Number of top coins to return
                     
        Returns:
            List of dictionaries with coin information
        """
        coin_max_scores = []
        
        for coin, df in results.items():
            if not df.empty and 'z_score' in df.columns:
                max_score = df['z_score'].max()
                max_row = df.loc[df['z_score'].idxmax()]
                
                coin_max_scores.append({
                    'coin': coin,
                    'timestamp': max_row['timestamp'],
                    'z_score': max_score,
                    'relative_return': max_row['relative_return'],
                    'alt_return': max_row['alt_return'],
                    'btc_return': max_row['btc_return'],
                    'close_price': max_row['close_price']
                })
        
        coin_max_scores.sort(key=lambda x: x['z_score'], reverse=True)
        
        return coin_max_scores[:n]
    
    def get_bottom_coins_by_z_score(self, results: Dict[str, pd.DataFrame], n: int = 10) -> List[Dict]:
        """
        Get bottom N coins by lowest z-scores
        
        Args:
            results: Dictionary of processed data
            n: Number of bottom coins to return
                     
        Returns:
            List of dictionaries with coin information
        """
        coin_min_scores = []
        
        for coin, df in results.items():
            if not df.empty and 'z_score' in df.columns:
                min_score = df['z_score'].min()
                min_row = df.loc[df['z_score'].idxmin()]
                
                coin_min_scores.append({
                    'coin': coin,
                    'timestamp': min_row['timestamp'],
                    'z_score': min_score,
                    'relative_return': min_row['relative_return'],
                    'alt_return': min_row['alt_return'],
                    'btc_return': min_row['btc_return'],
                    'close_price': min_row['close_price']
                })
        
        coin_min_scores.sort(key=lambda x: x['z_score'])
        
        return coin_min_scores[:n]
    

    
    def run_daily_analysis(self, target_date: str = None) -> Dict:
        """
        Run daily analysis for a specific date
        
        Args:
            target_date: Date to analyze (format: 'YYYY-MM-DD'), defaults to config value
            
        Returns:
            Dictionary with analysis results
        """
        if target_date is None:
            target_date = TARGET_TIMESTAMP.split()[0]
        
        results = self.process_pairs_data(target_date)
        
        top_coins = self.get_top_coins_by_z_score(results, n=TOP_N_OUTLIERS)
        bottom_coins = self.get_bottom_coins_by_z_score(results, n=TOP_N_OUTLIERS)
        
        total_outliers = sum(len(df[df['is_outlier']]) for df in results.values())
        positive_outliers = sum(len(df[df['z_score'] > Z_SCORE_THRESHOLD]) for df in results.values())
        negative_outliers = sum(len(df[df['z_score'] < -Z_SCORE_THRESHOLD]) for df in results.values())
        
        return {
            'date': target_date,
            'total_coins_analyzed': len(results),
            'total_outliers': total_outliers,
            'positive_outliers': positive_outliers,
            'negative_outliers': negative_outliers,
            'top_10_coins': top_coins,
            'bottom_10_coins': bottom_coins,
            'window_hours': self.window_hours
        }
    
    def analyze_time_period_movements(self, start_time: str, end_time: str, coins_to_analyze: List[str] = None) -> Dict:
        """
        Analyze price movements and returns for specific coins during a time period
        
        Args:
            start_time: Start time (format: 'YYYY-MM-DD HH:MM:SS')
            end_time: End time (format: 'YYYY-MM-DD HH:MM:SS')
            coins_to_analyze: List of coins to analyze (defaults to ["BTC", "TIA", "APT"])
            
        Returns:
            Dictionary with price movements and returns for each coin
        """
        if coins_to_analyze is None:
            coins_to_analyze = ["BTC", "TIA", "APT", "ETH", "RUNE"]
            
        start_dt = pd.to_datetime(start_time)
        end_dt = pd.to_datetime(end_time)
        
        results = {}
        
        for coin in coins_to_analyze:
            coin_data = self.load_coin_data(coin)
            if coin_data is None:
                continue
                
            mask = (coin_data['open_time'] >= start_dt) & (coin_data['open_time'] <= end_dt)
            period_data = coin_data[mask].copy()
            
            if len(period_data) == 0:
                continue
                
            period_data['returns'] = period_data['close'].pct_change()
            
            start_price = period_data['close'].iloc[0] if len(period_data) > 0 else None
            end_price = period_data['close'].iloc[-1] if len(period_data) > 0 else None
            total_return = (end_price - start_price) / start_price if start_price and end_price else 0
            
            results[coin] = {
                'start_time': start_dt,
                'end_time': end_dt,
                'start_price': start_price,
                'end_price': end_price,
                'total_return_pct': total_return * 100,
                'price_change': end_price - start_price if start_price and end_price else 0,
                'num_periods': len(period_data),
                'period_returns': period_data[['open_time', 'close', 'returns']].to_dict('records')
            }
            
        return results

    def run_multi_day_analysis(self, start_date: str = None, end_date: str = None) -> Dict:
        """
        Run analysis across multiple days to get better outlier detection
        
        Args:
            start_date: Start date (format: 'YYYY-MM-DD'), defaults to config value
            end_date: End date (format: 'YYYY-MM-DD'), defaults to config value
            
        Returns:
            Dictionary with analysis results
        """
        if start_date is None:
            start_date = TARGET_TIMESTAMP.split()[0]
        if end_date is None:
            end_date = TARGET_TIMESTAMP.split()[0]
        
        results = self.process_pairs_data()
        
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        filtered_results = {}
        for coin, df in results.items():
            mask = (df['timestamp'] >= start_dt) & (df['timestamp'] <= end_dt)
            filtered_df = df[mask].copy()
            if len(filtered_df) > 0:
                filtered_results[coin] = filtered_df
        
        top_coins = self.get_top_coins_by_z_score(filtered_results, n=TOP_N_OUTLIERS)
        bottom_coins = self.get_bottom_coins_by_z_score(filtered_results, n=TOP_N_OUTLIERS)
        
        total_outliers = sum(len(df[df['is_outlier']]) for df in filtered_results.values())
        positive_outliers = sum(len(df[df['z_score'] > Z_SCORE_THRESHOLD]) for df in filtered_results.values())
        negative_outliers = sum(len(df[df['z_score'] < -Z_SCORE_THRESHOLD]) for df in filtered_results.values())
        
        result = {
            'start_date': start_date,
            'end_date': end_date,
            'total_coins_analyzed': len(filtered_results),
            'total_outliers': total_outliers,
            'positive_outliers': positive_outliers,
            'negative_outliers': negative_outliers,
            'top_10_coins': top_coins,
            'bottom_10_coins': bottom_coins,
            'window_hours': self.window_hours
        }
        
        return result

    def analyze_z_scores_at_timestamp(self, target_timestamp: str) -> List[Dict]:
        """
        Analyze z-scores for all coins at a specific timestamp
        
        Args:
            target_timestamp: Target timestamp (format: 'YYYY-MM-DD HH:MM:SS')
            
        Returns:
            List of dictionaries with z-score data for all coins at the timestamp
        """
        target_dt = pd.to_datetime(target_timestamp)
        
        results = self.process_pairs_data()
        
        coin_z_scores = []
        
        for coin, df in results.items():
            if df.empty:
                continue
                
            df['time_diff'] = abs(df['timestamp'] - target_dt)
            closest_idx = df['time_diff'].idxmin()
            closest_row = df.loc[closest_idx]
            
            if closest_row['time_diff'] <= pd.Timedelta(minutes=5):
                coin_z_scores.append({
                    'coin': coin,
                    'timestamp': closest_row['timestamp'],
                    'z_score': closest_row['z_score'],
                    'relative_return': closest_row['relative_return'],
                    'alt_return': closest_row['alt_return'],
                    'btc_return': closest_row['btc_return'],
                    'close_price': closest_row['close_price'],
                    'is_outlier': closest_row['is_outlier'],
                    'time_diff_minutes': closest_row['time_diff'].total_seconds() / 60
                })
        
        coin_z_scores.sort(key=lambda x: x['z_score'], reverse=True)
        
        return coin_z_scores


def main():
    """Main function to run the pairs trading algorithm on real data"""
    algo = PairsTradingAlgorithm()
    
    print("=" * 60)
    print("PAIRS TRADING ALGORITHM - REAL DATA ANALYSIS")
    print("=" * 60)
    print(f"Configuration:")
    print(f"- Data folder: {DATA_FOLDER}")
    print(f"- Rolling window: {WINDOW_HOURS} hours ({WINDOW_HOURS * 12} periods)")
    print(f"- Z-score threshold: {Z_SCORE_THRESHOLD}")
    print(f"- BTC move threshold: {BTC_MOVE_THRESHOLD}% (noise filtering)")
    print(f"- EWMA smoothing: {'Enabled' if ENABLE_EWMA_SMOOTHING else 'Disabled'} (α={EWMA_ALPHA})")
    print(f"- Target timestamp: {TARGET_TIMESTAMP}")
    print(f"- Top N outliers: {TOP_N_OUTLIERS}")
    print("=" * 60)
    
    try:
        print(f"\n{'='*60}")
        print(f"Z-SCORES AT TIMESTAMP: {TARGET_TIMESTAMP}")
        print(f"{'='*60}")
        
        timestamp_results = algo.analyze_z_scores_at_timestamp(TARGET_TIMESTAMP)
        
        if timestamp_results:
            print(f"Found z-scores for {len(timestamp_results)} coins:")
            for i, coin_data in enumerate(timestamp_results, 1):
                outlier_marker = " *" if coin_data['is_outlier'] else ""
                print(f"{i:2d}. {coin_data['coin']:6s}: Z-score = {coin_data['z_score']:7.3f}, "
                      f"Relative Return = {coin_data['relative_return']:8.4f}, "
                      f"Price = ${coin_data['close_price']:8.2f}{outlier_marker}")
        else:
            print(f"No data found for timestamp {TARGET_TIMESTAMP}")
        
    except Exception as e:
        print(f"Error running analysis: {e}")
        print("Make sure you have downloaded the data using binance_downloader.py")
        print("Run: python src/binance_downloader.py to download data first")



if __name__ == "__main__":
    main()


