import requests
import pandas as pd
import logging
from typing import Dict, List
import time

from main import CMC_API_KEY, CMC_BASE_URL, COINS

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CoinMarketCapClient:
    def __init__(self):
        self.api_key = CMC_API_KEY
        self.base_url = CMC_BASE_URL
        self.headers = {
            'Accepts': 'application/json',
            'X-CMC_PRO_API_KEY': self.api_key,
        }
        
        # Mapping from our coin symbols to CoinMarketCap symbols
        self.symbol_mapping = {
            'BTC': 'BTC',
            'ETH': 'ETH', 
            'BNB': 'BNB',
            'ADA': 'ADA',
            'XRP': 'XRP',
            'SOL': 'SOL',
            'DOGE': 'DOGE',
            'DOT': 'DOT',
            'AVAX': 'AVAX',
            'LINK': 'LINK',
            'LTC': 'LTC',
            'UNI': 'UNI',
            'ALGO': 'ALGO',
            'BCH': 'BCH',
            'XLM': 'XLM',
            'VET': 'VET',
            'FIL': 'FIL',
            'ETC': 'ETC',
            'ATOM': 'ATOM',
            'HBAR': 'HBAR',
            'ICP': 'ICP',
            'APT': 'APT',
            'ARB': 'ARB',
            'CRV': 'CRV',
            'GRT': 'GRT',
            'RUNE': 'RUNE',
            'INJ': 'INJ',
            'MKR': 'MKR',
            'NEAR': 'NEAR',
            'LDO': 'LDO',
            'SUI': 'SUI',
            'PEPE': 'PEPE',
            'BONK': 'BONK',
            'WIF': 'WIF',
            'FLOKI': 'FLOKI',
            'JUP': 'JUP',
            'PYTH': 'PYTH',
            'TIA': 'TIA',
            'SEI': 'SEI',
            'CAKE': 'CAKE',
            'SUSHI': 'SUSHI',
            'TAO': 'TAO',
            'VIRTUAL': 'VIRTUAL',
            'ENS': 'ENS',
            'GALA': 'GALA',
            'SPX': 'SPX',
            'FLOW': 'FLOW',
            'DYDX': 'DYDX',
            'ETHFI': 'ETHFI',
            'FARTCOIN': 'FARTCOIN',
            'WLD': 'WLD',
            'TON': 'TON',
            'XTZ': 'XTZ'
        }
    
    def get_market_data(self) -> Dict[str, float]:
        """Fetch current market cap data for all coins"""
        try:
            # Get symbols for API call
            symbols = ','.join([self.symbol_mapping.get(coin, coin) for coin in COINS])
            
            url = f"{self.base_url}/cryptocurrency/quotes/latest"
            parameters = {
                'symbol': symbols,
                'convert': 'USD'
            }
            
            response = requests.get(url, headers=self.headers, params=parameters)
            response.raise_for_status()
            
            data = response.json()
            
            if data['status']['error_code'] != 0:
                logger.error(f"CMC API Error: {data['status']['error_message']}")
                return {}
            
            market_caps = {}
            for coin in COINS:
                cmc_symbol = self.symbol_mapping.get(coin, coin)
                try:
                    if cmc_symbol in data['data']:
                        market_cap = data['data'][cmc_symbol]['quote']['USD']['market_cap']
                        market_caps[coin] = market_cap
                        logger.info(f"{coin}: Market Cap = ${market_cap:,.0f}")
                    else:
                        logger.warning(f"No market cap data found for {coin} ({cmc_symbol})")
                        market_caps[coin] = None
                except KeyError as e:
                    logger.warning(f"Error parsing market cap for {coin}: {e}")
                    market_caps[coin] = None
            
            return market_caps
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching CMC data: {e}")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error in CMC API call: {e}")
            return {}
    
    def get_market_cap_for_coin(self, coin: str) -> float:
        """Get market cap for a single coin"""
        market_caps = self.get_market_data()
        return market_caps.get(coin, None)

def main():
    """Test the CoinMarketCap client"""
    client = CoinMarketCapClient()
    market_caps = client.get_market_data()
    
    print("\n=== MARKET CAP DATA ===")
    for coin, market_cap in market_caps.items():
        if market_cap:
            print(f"{coin}: ${market_cap:,.0f}")
        else:
            print(f"{coin}: No data")

if __name__ == "__main__":
    main()