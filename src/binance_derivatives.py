import requests
import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional
import time

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

class AggregatedDerivativesClient:
    """
    Coinglass API client for aggregated derivatives data across ALL major exchanges.

    Provides:
    - Volume-weighted average funding rates (10+ exchanges)
    - Aggregated open interest
    - Liquidation data (whale hunting signals)
    - Exchange-level OI distribution
    """

    def __init__(self):
        self.base_url = "https://open-api.coinglass.com/public/v2"
        self.session = requests.Session()
        self.cache = {}
        self.cache_ttl = 60  # Cache for 60 seconds

    def _get_cached(self, key: str):
        """Check cache for recent data"""
        if key in self.cache:
            data, timestamp = self.cache[key]
            if time.time() - timestamp < self.cache_ttl:
                return data
        return None

    def _set_cache(self, key: str, data):
        """Store data in cache"""
        self.cache[key] = (data, time.time())

    def _make_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make API request with error handling"""
        try:
            url = f"{self.base_url}/{endpoint}"
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.warning(f"Coinglass API request failed for {endpoint}: {e}")
            return None

    def get_funding_rate(self, coin: str) -> Optional[float]:
        """
        Get volume-weighted average funding rate across all exchanges.

        Returns: Funding rate as percentage (e.g., 0.01 = 0.01%)
        """
        cache_key = f"funding_{coin}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        try:
            # Try multiple symbol formats
            symbols_to_try = [
                f"{coin}USDT",
                f"{coin}USD",
                coin
            ]

            for symbol in symbols_to_try:
                data = self._make_request("indicator/funding-rates", params={
                    'symbol': symbol,
                    'exchange': 'all'
                })

                if data and 'data' in data and data['data']:
                    funding_data = data['data']

                    # Calculate volume-weighted average
                    total_rate = 0
                    total_volume = 0

                    if isinstance(funding_data, list):
                        for item in funding_data:
                            try:
                                rate = float(item.get('rate', item.get('fundingRate', 0)))
                                volume = float(item.get('openInterestUsd', item.get('volUsd', 1)))
                                total_rate += rate * volume
                                total_volume += volume
                            except:
                                continue

                        if total_volume > 0:
                            avg_rate = (total_rate / total_volume) * 100
                            self._set_cache(cache_key, avg_rate)
                            return avg_rate

            return None

        except Exception as e:
            logger.warning(f"Failed to get funding rate for {coin}: {e}")
            return None

    def get_open_interest(self, coin: str) -> Optional[float]:
        """
        Get aggregated open interest in USD across all exchanges.

        Returns: Total OI in USD
        """
        cache_key = f"oi_{coin}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        try:
            symbol = f"{coin}USDT"
            data = self._make_request("openInterest", params={'symbol': symbol})

            if not data or 'data' not in data:
                return None

            oi_data = data['data']

            # Get total OI across all exchanges
            if 'totalOpenInterest' in oi_data:
                total_oi = float(oi_data['totalOpenInterest'])
                self._set_cache(cache_key, total_oi)
                return total_oi

            return None

        except Exception as e:
            logger.warning(f"Failed to get open interest for {coin}: {e}")
            return None

    def get_liquidations(self, coin: str, timeframe: str = '1h') -> Optional[Dict]:
        """
        Get liquidation data (long/short liquidations).

        High liquidations = overleveraged positions = potential reversal signal.

        Args:
            coin: Coin symbol (e.g., 'BTC')
            timeframe: '5m', '15m', '1h', '4h', '12h', '24h'

        Returns: Dict with long_liq, short_liq, total_liq in USD
        """
        cache_key = f"liq_{coin}_{timeframe}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        try:
            symbol = f"{coin}USDT"
            data = self._make_request("liquidation_history", params={
                'symbol': symbol,
                'timeType': timeframe
            })

            if not data or 'data' not in data:
                return None

            liq_data = data['data']

            # Sum recent liquidations
            long_liq = 0
            short_liq = 0

            if isinstance(liq_data, list) and len(liq_data) > 0:
                # Get most recent data point
                recent = liq_data[0]
                long_liq = float(recent.get('longLiquidationUsd', 0))
                short_liq = float(recent.get('shortLiquidationUsd', 0))

            result = {
                'long_liq': long_liq,
                'short_liq': short_liq,
                'total_liq': long_liq + short_liq,
                'liq_ratio': long_liq / short_liq if short_liq > 0 else 0
            }

            self._set_cache(cache_key, result)
            return result

        except Exception as e:
            logger.warning(f"Failed to get liquidations for {coin}: {e}")
            return None

    def get_oi_distribution(self, coin: str) -> Optional[Dict]:
        """
        Get open interest distribution across exchanges.

        Useful for identifying concentration risk.
        """
        try:
            symbol = f"{coin}USDT"
            data = self._make_request("openInterest", params={'symbol': symbol})

            if not data or 'data' not in data:
                return None

            oi_data = data['data']

            distribution = {}
            if 'dataMap' in oi_data:
                for exchange, value in oi_data['dataMap'].items():
                    try:
                        distribution[exchange] = float(value)
                    except:
                        continue

            return distribution

        except Exception as e:
            logger.warning(f"Failed to get OI distribution for {coin}: {e}")
            return None

    def get_long_short_ratio(self, coin: str) -> Optional[float]:
        """
        Get aggregated long/short ratio across exchanges.

        Ratio > 1 = more longs than shorts (potential reversal down)
        Ratio < 1 = more shorts than longs (potential reversal up)
        """
        cache_key = f"ls_ratio_{coin}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        try:
            symbol = f"{coin}USDT"
            data = self._make_request("longShortRate", params={'symbol': symbol})

            if not data or 'data' not in data:
                return None

            ls_data = data['data']

            # Get aggregated ratio
            if 'ratio' in ls_data:
                ratio = float(ls_data['ratio'])
                self._set_cache(cache_key, ratio)
                return ratio

            return None

        except Exception as e:
            logger.warning(f"Failed to get long/short ratio for {coin}: {e}")
            return None

    def get_funding_rate_chart(self, coin: str, interval: str = '1h') -> Optional[pd.DataFrame]:
        """
        Get historical funding rate data for analysis.

        Args:
            coin: Coin symbol
            interval: '8h' (default funding period), '1h', '4h', '12h'
        """
        try:
            symbol = f"{coin}USDT"
            data = self._make_request("funding_usd_history", params={
                'symbol': symbol,
                'interval': interval
            })

            if not data or 'data' not in data:
                return None

            df = pd.DataFrame(data['data'])
            return df

        except Exception as e:
            logger.warning(f"Failed to get funding rate chart for {coin}: {e}")
            return None


def main():
    """Test aggregated derivatives data"""
    client = AggregatedDerivativesClient()

    test_coins = ['BTC', 'ETH', 'SOL']

    print("\n=== AGGREGATED DERIVATIVES DATA (COINGLASS) ===\n")

    for coin in test_coins:
        print(f"\n{coin}:")

        # Funding rate
        funding = client.get_funding_rate(coin)
        if funding is not None:
            print(f"  Funding Rate (Aggregated): {funding:.4f}%")
        else:
            print(f"  Funding Rate: N/A")

        # Open interest
        oi = client.get_open_interest(coin)
        if oi is not None:
            print(f"  Open Interest (Total): ${oi:,.0f}")
        else:
            print(f"  Open Interest: N/A")

        # Liquidations
        liq = client.get_liquidations(coin, '1h')
        if liq:
            print(f"  Liquidations (1h):")
            print(f"    - Longs: ${liq['long_liq']:,.0f}")
            print(f"    - Shorts: ${liq['short_liq']:,.0f}")
            print(f"    - Total: ${liq['total_liq']:,.0f}")
            print(f"    - Ratio: {liq['liq_ratio']:.2f}")
        else:
            print(f"  Liquidations: N/A")

        # Long/Short ratio
        ls_ratio = client.get_long_short_ratio(coin)
        if ls_ratio is not None:
            print(f"  Long/Short Ratio: {ls_ratio:.2f}")
            if ls_ratio > 1.5:
                print(f"    → Overleveraged longs (bearish)")
            elif ls_ratio < 0.67:
                print(f"    → Overleveraged shorts (bullish)")

        # OI distribution
        oi_dist = client.get_oi_distribution(coin)
        if oi_dist:
            print(f"  OI Distribution:")
            sorted_exchanges = sorted(oi_dist.items(), key=lambda x: x[1], reverse=True)[:3]
            for exchange, amount in sorted_exchanges:
                print(f"    - {exchange}: ${amount:,.0f}")


if __name__ == "__main__":
    main()
