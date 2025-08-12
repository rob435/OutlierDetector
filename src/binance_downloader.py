import requests
import pandas as pd
import time
from datetime import datetime, timedelta
import os
from typing import List, Dict, Any
import logging

# Import the COINS list from main.py
from main import COINS

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BinanceDataDownloader:
    def __init__(self, data_folder: str = "../data"):
        self.base_url = "https://api.binance.com/api/v3"
        self.data_folder = data_folder
        self.ensure_data_folder()
    
    def ensure_data_folder(self):
        """Ensure the data folder exists"""
        os.makedirs(self.data_folder, exist_ok=True)
        logger.info(f"Data folder ensured: {self.data_folder}")
    
    def get_klines(self, symbol: str, interval: str = "5m", start_time: int = None, end_time: int = None) -> List[List]:
        """
        Get kline/candlestick data from Binance API
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            interval: Kline interval (e.g., '5m')
            start_time: Start time in milliseconds
            end_time: End time in milliseconds
        
        Returns:
            List of kline data
        """
        url = f"{self.base_url}/klines"
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': 1000  # Maximum limit per request
        }
        
        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return []
    
    def download_coin_data(self, coin: str, days_back: int = 365) -> bool:
        """
        Download OHLCV data for a specific coin
        
        Args:
            coin: Coin symbol (e.g., 'BTC')
            days_back: Number of days to go back from now
        
        Returns:
            bool: True if successful, False otherwise
        """
        symbol = f"{coin}USDT"
        
        # Calculate time range
        end_time = int(time.time() * 1000)  # Current time in milliseconds
        start_time = end_time - (days_back * 24 * 60 * 60 * 1000)  # days_back days ago
        
        logger.info(f"Downloading data for {symbol} from {datetime.fromtimestamp(start_time/1000)} to {datetime.fromtimestamp(end_time/1000)}")
        
        all_klines = []
        current_start = start_time
        
        # Download data in chunks (1000 candles per request)
        while current_start < end_time:
            current_end = min(current_start + (1000 * 5 * 60 * 1000), end_time)  # 1000 * 5 minutes in milliseconds
            
            klines = self.get_klines(symbol, start_time=current_start, end_time=current_end)
            
            if not klines:
                logger.warning(f"No data received for {symbol} in time range {current_start} to {current_end}")
                break
            
            all_klines.extend(klines)
            current_start = current_end
            
            # Small delay to be respectful to the API
            time.sleep(0.1)
        
        if not all_klines:
            logger.error(f"No data downloaded for {symbol}")
            return False
        
        # Convert to DataFrame
        df = pd.DataFrame(all_klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Convert types
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume', 
                          'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
        df[numeric_columns] = df[numeric_columns].astype(float)
        df[['open_time', 'close_time', 'number_of_trades']] = df[['open_time', 'close_time', 'number_of_trades']].astype(int)
        
        # Convert timestamps to datetime
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        
        # Drop unnecessary columns
        df = df.drop(['ignore'], axis=1)
        
        # Save as parquet
        filename = f"{self.data_folder}/{coin}_5m_ohlcv.parquet"
        df.to_parquet(filename, index=False)
        
        logger.info(f"Saved {len(df)} records for {symbol} to {filename}")
        return True
    
    def download_all_coins(self, days_back: int = 365):
        """
        Download data for all coins in the COINS list
        
        Args:
            days_back: Number of days to go back from now
        """
        logger.info(f"Starting download for {len(COINS)} coins, going back {days_back} days")
        
        successful_downloads = 0
        failed_downloads = 0
        
        for i, coin in enumerate(COINS, 1):
            logger.info(f"Processing {coin} ({i}/{len(COINS)})")
            
            try:
                if self.download_coin_data(coin, days_back):
                    successful_downloads += 1
                else:
                    failed_downloads += 1
            except Exception as e:
                logger.error(f"Unexpected error downloading {coin}: {e}")
                failed_downloads += 1
            
            # Small delay between coins
            time.sleep(0.5)
        
        logger.info(f"Download completed. Successful: {successful_downloads}, Failed: {failed_downloads}")


def main():
    """Main function to run the downloader"""
    downloader = BinanceDataDownloader()
    downloader.download_all_coins(days_back=365)


if __name__ == "__main__":
    main()
