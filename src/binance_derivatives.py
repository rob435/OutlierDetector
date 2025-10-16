import requests
import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional
from main import COINS

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

class BinanceDerivatives:
    """Fetch derivatives data: funding rates, open interest, perpetual-spot basis"""
    
    def __init__(self):
        self.futures_base_url = "https://fapi.binance.com/fapi/v1"
        self.spot_base_url = "https://api.binance.com/api/v3"
        
        self.symbol_mappings = {
            'PEPE': '1000PEPEUSDT',
            'BONK': '1000BONKUSDT',
            'FLOKI': '1000FLOKIUSDT'
        }
    
    def _get_symbol(self, coin: str) -> str:
        """Get Binance symbol with special mappings"""
        return self.symbol_mappings.get(coin, f"{coin}USDT")
    
    def get_funding_rate(self, coin: str) -> Optional[float]:
        """Get current funding rate for perpetual futures"""
        symbol = self._get_symbol(coin)
        
        try:
            url = f"{self.futures_base_url}/premiumIndex"
            params = {'symbol': symbol}
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            funding_rate = float(data.get('lastFundingRate', 0))
            return funding_rate * 100  # Convert to percentage
            
        except Exception as e:
            logger.warning(f"Failed to get funding rate for {coin}: {e}")
            return None
    
    def get_open_interest(self, coin: str) -> Optional[float]:
        """Get current open interest in USDT"""
        symbol = self._get_symbol(coin)
        
        try:
            url = f"{self.futures_base_url}/openInterest"
            params = {'symbol': symbol}
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            # Get OI in contracts
            oi_contracts = float(data.get('openInterest', 0))
            
            # Get current price to convert to USDT
            price = self._get_futures_price(symbol)
            if price is None:
                return None
            
            oi_usdt = oi_contracts * price
            return oi_usdt
            
        except Exception as e:
            logger.warning(f"Failed to get open interest for {coin}: {e}")
            return None
    
    def _get_futures_price(self, symbol: str) -> Optional[float]:
        """Get current futures price"""
        try:
            url = f"{self.futures_base_url}/ticker/price"
            params = {'symbol': symbol}
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            return float(data.get('price', 0))
        except:
            return None
    
    def _get_spot_price(self, symbol: str) -> Optional[float]:
        """Get current spot price"""
        try:
            url = f"{self.spot_base_url}/ticker/price"
            params = {'symbol': symbol}
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            return float(data.get('price', 0))
        except:
            return None
    
    def get_perp_spot_basis(self, coin: str) -> Optional[float]:
        """Get perpetual-spot premium (basis)"""
        symbol = self._get_symbol(coin)
        
        try:
            perp_price = self._get_futures_price(symbol)
            spot_price = self._get_spot_price(symbol)
            
            if perp_price is None or spot_price is None or spot_price == 0:
                return None
            
            # Calculate basis as percentage
            basis = ((perp_price - spot_price) / spot_price) * 100
            return basis
            
        except Exception as e:
            logger.warning(f"Failed to get perp-spot basis for {coin}: {e}")
            return None
    
    def get_order_book_depth(self, coin: str, limit: int = 10) -> Optional[Dict]:
        """Get order book depth (bids/asks)"""
        symbol = self._get_symbol(coin)
        
        try:
            url = f"{self.futures_base_url}/depth"
            params = {'symbol': symbol, 'limit': limit * 2}  # Get more than needed
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            bids = data.get('bids', [])[:limit]  # Top limit bids
            asks = data.get('asks', [])[:limit]  # Top limit asks
            
            if not bids or not asks:
                return None
            
            # Calculate total volume at each side
            bid_volume = sum(float(price) * float(qty) for price, qty in bids)
            ask_volume = sum(float(price) * float(qty) for price, qty in asks)
            
            total_volume = bid_volume + ask_volume
            if total_volume == 0:
                return None
            
            # Calculate imbalance: positive = more bids (support), negative = more asks (resistance)
            imbalance = (bid_volume - ask_volume) / total_volume
            
            return {
                'bid_volume': bid_volume,
                'ask_volume': ask_volume,
                'imbalance': imbalance,
                'total_volume': total_volume
            }
            
        except Exception as e:
            logger.warning(f"Failed to get order book for {coin}: {e}")
            return None
    
    def get_all_derivatives_data(self, coins: list = None) -> pd.DataFrame:
        """Get all derivatives data for all coins"""
        if coins is None:
            coins = COINS
        
        results = []
        
        for coin in coins:
            logger.info(f"Fetching derivatives data for {coin}")
            
            funding_rate = self.get_funding_rate(coin)
            open_interest = self.get_open_interest(coin)
            perp_spot_basis = self.get_perp_spot_basis(coin)
            order_book = self.get_order_book_depth(coin)
            
            results.append({
                'coin': coin,
                'funding_rate': funding_rate if funding_rate is not None else 0,
                'open_interest': open_interest if open_interest is not None else 0,
                'perp_spot_basis': perp_spot_basis if perp_spot_basis is not None else 0,
                'bid_ask_imbalance': order_book['imbalance'] if order_book else 0,
                'order_book_volume': order_book['total_volume'] if order_book else 0
            })
        
        return pd.DataFrame(results)


def main():
    """Test derivatives data fetching"""
    client = BinanceDerivatives()
    
    test_coins = ['BTC', 'ETH', 'SOL', 'PEPE']
    
    print("\n=== DERIVATIVES DATA TEST ===\n")
    
    for coin in test_coins:
        print(f"\n{coin}:")
        
        funding = client.get_funding_rate(coin)
        print(f"  Funding Rate: {funding:.4f}%" if funding else "  Funding Rate: N/A")
        
        oi = client.get_open_interest(coin)
        print(f"  Open Interest: ${oi:,.0f}" if oi else "  Open Interest: N/A")
        
        basis = client.get_perp_spot_basis(coin)
        print(f"  Perp-Spot Basis: {basis:.3f}%" if basis else "  Perp-Spot Basis: N/A")
        
        order_book = client.get_order_book_depth(coin)
        if order_book:
            print(f"  Order Book Imbalance: {order_book['imbalance']:.3f}")
            print(f"  Total OB Volume: ${order_book['total_volume']:,.0f}")
        else:
            print("  Order Book: N/A")


if __name__ == "__main__":
    main()
