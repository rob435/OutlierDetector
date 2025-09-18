import requests
import pandas as pd
import time
from datetime import datetime, timedelta
import os
from typing import List, Dict, Any
import logging

from main import COINS, NARRATIVE_DATA_FOLDER, NARRATIVE_TIMEFRAME, NARRATIVE_HISTORY_DAYS

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NarrativeDataDownloader:
    def __init__(self, data_folder: str = None):
        self.spot_base_url = "https://api.binance.com/api/v3"
        self.futures_base_url = "https://fapi.binance.com/fapi/v1"
        self.data_folder = data_folder or NARRATIVE_DATA_FOLDER
        
        # Special symbol mappings for Binance futures
        self.symbol_mappings = {
            'PEPE': '1000PEPEUSDT',
            'BONK': '1000BONKUSDT', 
            'FLOKI': '1000FLOKIUSDT'
        }
        
        self.ensure_data_folder()
    
    def ensure_data_folder(self):
        os.makedirs(self.data_folder, exist_ok=True)
        logger.info(f"Narrative data folder ensured: {self.data_folder}")
    
    def get_klines(self, symbol: str, interval: str = "1h", start_time: int = None, end_time: int = None) -> List[List]:
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
    
    def download_coin_data(self, coin: str, days_back: int = NARRATIVE_HISTORY_DAYS) -> bool:
        # Use special mapping if available, otherwise default format
        if coin in self.symbol_mappings:
            symbol = self.symbol_mappings[coin]
        else:
            symbol = f"{coin}USDT"
        
        end_time = int(time.time() * 1000)
        start_time = end_time - (days_back * 24 * 60 * 60 * 1000)
        
        logger.info(f"Downloading narrative data for {symbol} from {datetime.fromtimestamp(start_time/1000)} to {datetime.fromtimestamp(end_time/1000)}")
        
        all_klines = []
        current_start = start_time
        
        while current_start < end_time:
            # 1000 hours = ~41 days per request
            current_end = min(current_start + (1000 * 60 * 60 * 1000), end_time)
            
            klines = self.get_klines(symbol, interval=NARRATIVE_TIMEFRAME, start_time=current_start, end_time=current_end)
            
            if not klines:
                logger.warning(f"No data received for {symbol} in time range {current_start} to {current_end}")
                break
            
            all_klines.extend(klines)
            current_start = current_end
            
            time.sleep(0.1)
        
        if not all_klines:
            logger.error(f"No data downloaded for {symbol}")
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
        
        filename = f"{self.data_folder}/{coin}_1h_ohlcv.parquet"
        df.to_parquet(filename, index=False)
        
        logger.info(f"Saved {len(df)} records for {symbol} to {filename}")
        return True
    
    def download_all_coins(self, days_back: int = NARRATIVE_HISTORY_DAYS):
        logger.info(f"Starting narrative data download for {len(COINS)} coins, going back {days_back} days")
        
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
            
            time.sleep(0.5)
        
        logger.info(f"Narrative data download completed. Successful: {successful_downloads}, Failed: {failed_downloads}")


def main():
    downloader = NarrativeDataDownloader()
    downloader.download_all_coins(days_back=NARRATIVE_HISTORY_DAYS)


if __name__ == "__main__":
    main()