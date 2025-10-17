import requests
import pandas as pd
import time
from datetime import datetime, timedelta
import os
from typing import List, Dict, Any
import logging

from main import COINS

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BinanceDataDownloader:
    def __init__(self, data_folder: str = "../data"):
        self.spot_base_url = "https://api.binance.com/api/v3"
        self.futures_base_url = "https://fapi.binance.com/fapi/v1"
        self.data_folder = data_folder
        
        # Special symbol mappings for Binance futures
        self.symbol_mappings = {
            'PEPE': '1000PEPEUSDT',
            'BONK': '1000BONKUSDT', 
            'FLOKI': '1000FLOKIUSDT'
        }
        
        self.ensure_data_folder()
    
    def ensure_data_folder(self):
        os.makedirs(self.data_folder, exist_ok=True)
        logger.info(f"Data folder ensured: {self.data_folder}")
    
    def get_klines(self, symbol: str, interval: str = "5m", start_time: int = None, end_time: int = None) -> List[List]:
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': 1000
        }
        
        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time
        
        # Use futures market only
        try:
            url = f"{self.futures_base_url}/klines"
            response = requests.get(url, params=params)
            response.raise_for_status()
            logger.info(f"Successfully fetched {symbol} from futures market")
            return response.json()
        except requests.exceptions.RequestException as futures_error:
            logger.error(f"Futures market failed for {symbol}: {futures_error}")
            return []
    
    def download_coin_data(self, coin: str, days_back: int = 365, interval: str = "5m") -> bool:
        # Use special mapping if available, otherwise default format
        if coin in self.symbol_mappings:
            symbol = self.symbol_mappings[coin]
        else:
            symbol = f"{coin}USDT"

        end_time = int(time.time() * 1000)
        start_time = end_time - (days_back * 24 * 60 * 60 * 1000)

        logger.info(f"Downloading data for {symbol} ({interval}) from {datetime.fromtimestamp(start_time/1000)} to {datetime.fromtimestamp(end_time/1000)}")

        # Calculate interval multiplier for pagination
        interval_minutes = {
            "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440
        }
        minutes = interval_minutes.get(interval, 5)

        all_klines = []
        current_start = start_time

        while current_start < end_time:
            current_end = min(current_start + (1000 * minutes * 60 * 1000), end_time)

            klines = self.get_klines(symbol, interval=interval, start_time=current_start, end_time=current_end)

            if not klines:
                logger.warning(f"No data received for {symbol} ({interval}) in time range {current_start} to {current_end}")
                break

            all_klines.extend(klines)
            current_start = current_end

        if not all_klines:
            logger.error(f"No data downloaded for {symbol} ({interval})")
            return False

        df = pd.DataFrame(all_klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])

        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume',
                          'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
        df[numeric_columns] = df[numeric_columns].astype(float)
        df[['open_time', 'close_time', 'number_of_trades']] = df[['open_time', 'close_time', 'number_of_trades']].astype(int)

        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')

        df = df.drop(['ignore'], axis=1)

        filename = f"{self.data_folder}/{coin}_{interval}_ohlcv.parquet"
        df.to_parquet(filename, index=False)

        logger.info(f"Saved {len(df)} records for {symbol} ({interval}) to {filename}")
        return True
    
    def download_all_coins(self, days_back: int = 365, intervals: List[str] = None):
        if intervals is None:
            intervals = ["5m"]

        logger.info(f"Starting download for {len(COINS)} coins across {len(intervals)} intervals, going back {days_back} days")

        successful_downloads = 0
        failed_downloads = 0

        for i, coin in enumerate(COINS, 1):
            logger.info(f"Processing {coin} ({i}/{len(COINS)})")

            coin_success = True
            for interval in intervals:
                try:
                    if not self.download_coin_data(coin, days_back, interval):
                        coin_success = False
                        failed_downloads += 1
                except Exception as e:
                    logger.error(f"Unexpected error downloading {coin} ({interval}): {e}")
                    coin_success = False
                    failed_downloads += 1

            if coin_success:
                successful_downloads += 1

        logger.info(f"Download completed. Successful coins: {successful_downloads}, Failed operations: {failed_downloads}")


def main():
    from main import DATA_FOLDER
    downloader = BinanceDataDownloader(DATA_FOLDER)
    downloader.download_all_coins(days_back=7)  # Get 7 days for better z-scores


if __name__ == "__main__":
    main()
