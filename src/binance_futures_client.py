import requests
import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, List
import time

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

class BinanceFuturesClient:
    """
    Binance Futures API client for derivatives data (PUBLIC endpoints, no auth required).

    Provides:
    - Funding rates (8h intervals)
    - Open interest
    - Long/short ratio (top trader accounts)
    - Top trader long/short ratio (by positions)
    - Liquidation data
    - Taker buy/sell volume

    Data source: Binance Futures (40%+ of crypto derivatives market)
    All endpoints are PUBLIC - no API key required.
    """

    # Configuration at top
    BASE_URL = "https://fapi.binance.com"
    CACHE_TTL_SECONDS = 60
    REQUEST_TIMEOUT_SECONDS = 10
    MAX_RETRIES = 3
    RETRY_DELAY_SECONDS = 1

    # Coins that need "1000" prefix on Binance Futures
    COINS_WITH_1000_PREFIX = [
        'PEPE', 'SHIB', 'BONK', 'FLOKI', 'LUNC', 'XEC', 'BTTC',
        'WIN', 'BIGTYME', 'NFP', 'SATS', 'RATS'
    ]

    def __init__(self):
        self.base_url = self.BASE_URL
        self.session = requests.Session()
        self.cache = {}
        self.cache_ttl = self.CACHE_TTL_SECONDS

    def _get_symbol(self, coin: str) -> str:
        """Get correct Binance Futures symbol (handle 1000 prefix for certain coins)"""
        coin_upper = coin.upper()
        if coin_upper in self.COINS_WITH_1000_PREFIX:
            return f"1000{coin_upper}USDT"
        return f"{coin_upper}USDT"

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
        """Make API request with error handling and retries"""
        url = f"{self.base_url}{endpoint}"

        for attempt in range(self.MAX_RETRIES):
            try:
                response = self.session.get(url, params=params, timeout=self.REQUEST_TIMEOUT_SECONDS)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.RETRY_DELAY_SECONDS * (attempt + 1))
                    continue
                logger.warning(f"Binance API request failed for {endpoint}: {e}")
                return None

        return None

    def get_funding_rate_history(self, coin: str, limit: int = 100, start_time: int = None, end_time: int = None) -> Optional[pd.DataFrame]:
        """
        Get historical funding rates from Binance Futures.

        Args:
            coin: Coin symbol (e.g., 'BTC', 'ETH')
            limit: Number of records to fetch (max 1000, default 100)
            start_time: Start timestamp in milliseconds (optional)
            end_time: End timestamp in milliseconds (optional)

        Returns:
            DataFrame with columns: timestamp, funding_rate, mark_price
        """
        try:
            symbol = self._get_symbol(coin)

            params = {
                'symbol': symbol,
                'limit': min(limit, 1000)
            }

            if start_time:
                params['startTime'] = start_time
            if end_time:
                params['endTime'] = end_time

            data = self._make_request("/fapi/v1/fundingRate", params=params)

            if not data or len(data) == 0:
                return None

            # Convert to DataFrame
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['fundingTime'], unit='ms')
            df['funding_rate'] = df['fundingRate'].astype(float) * 100  # Convert to percentage
            df['mark_price'] = df.get('markPrice', 0).astype(float)

            df = df[['timestamp', 'funding_rate', 'mark_price']]
            df = df.sort_values('timestamp').reset_index(drop=True)

            return df

        except Exception as e:
            logger.warning(f"Failed to get funding rate history for {coin}: {e}")
            return None

    def get_funding_rate(self, coin: str) -> Optional[float]:
        """
        Get current funding rate from Binance Futures.

        Args:
            coin: Coin symbol (e.g., 'BTC', 'ETH')

        Returns:
            Funding rate as percentage (e.g., 0.01 = 0.01%)
        """
        cache_key = f"funding_{coin}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        try:
            symbol = self._get_symbol(coin)

            # Get current funding rate
            data = self._make_request("/fapi/v1/fundingRate", params={
                'symbol': symbol,
                'limit': 1
            })

            if not data or len(data) == 0:
                return None

            # Get most recent funding rate
            funding_rate = float(data[0]['fundingRate']) * 100  # Convert to percentage

            self._set_cache(cache_key, funding_rate)
            return funding_rate

        except Exception as e:
            logger.warning(f"Failed to get funding rate for {coin}: {e}")
            return None

    def get_open_interest_history(self, coin: str, period: str = '5m', limit: int = 100, start_time: int = None, end_time: int = None) -> Optional[pd.DataFrame]:
        """
        Get historical open interest from Binance Futures.

        Args:
            coin: Coin symbol (e.g., 'BTC', 'ETH')
            period: Time period ('5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d')
            limit: Number of records to fetch (max 500, default 100)
            start_time: Start timestamp in milliseconds (optional)
            end_time: End timestamp in milliseconds (optional)

        Returns:
            DataFrame with columns: timestamp, open_interest, open_interest_value
        """
        try:
            symbol = self._get_symbol(coin)

            params = {
                'symbol': symbol,
                'period': period,
                'limit': min(limit, 500)
            }

            if start_time:
                params['startTime'] = start_time
            if end_time:
                params['endTime'] = end_time

            data = self._make_request("/futures/data/openInterestHist", params=params)

            if not data or len(data) == 0:
                return None

            # Convert to DataFrame
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['open_interest'] = df['sumOpenInterest'].astype(float)
            df['open_interest_value'] = df['sumOpenInterestValue'].astype(float)

            df = df[['timestamp', 'open_interest', 'open_interest_value']]
            df = df.sort_values('timestamp').reset_index(drop=True)

            return df

        except Exception as e:
            logger.warning(f"Failed to get open interest history for {coin}: {e}")
            return None

    def get_open_interest_change(self, coin: str, lookback_hours: int = 1) -> Optional[Dict]:
        """
        Get open interest % change over lookback period.

        Args:
            coin: Coin symbol (e.g., 'BTC', 'ETH')
            lookback_hours: Hours to look back (1, 4, 24, etc.)

        Returns:
            Dict with current_oi, previous_oi, change_pct
        """
        try:
            symbol = self._get_symbol(coin)

            # Determine period based on lookback hours
            if lookback_hours <= 1:
                period = '5m'
                limit = 12  # 12 * 5m = 1 hour
            elif lookback_hours <= 4:
                period = '15m'
                limit = 16  # 16 * 15m = 4 hours
            elif lookback_hours <= 24:
                period = '1h'
                limit = lookback_hours
            else:
                period = '4h'
                limit = lookback_hours // 4

            # Get historical OI data
            history = self.get_open_interest_history(coin, period=period, limit=limit+1)

            if history is None or len(history) < 2:
                return None

            # Calculate change
            current_oi = float(history['open_interest_value'].iloc[-1])
            previous_oi = float(history['open_interest_value'].iloc[0])

            if previous_oi == 0:
                return None

            change_pct = ((current_oi - previous_oi) / previous_oi) * 100

            result = {
                'current_oi': current_oi,
                'previous_oi': previous_oi,
                'change_pct': change_pct,
                'lookback_hours': lookback_hours
            }

            return result

        except Exception as e:
            logger.warning(f"Failed to get OI change for {coin}: {e}")
            return None

    def get_open_interest(self, coin: str) -> Optional[float]:
        """
        Get current open interest in USD from Binance Futures.

        Args:
            coin: Coin symbol (e.g., 'BTC', 'ETH')

        Returns:
            Open interest in USD
        """
        cache_key = f"oi_{coin}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        try:
            symbol = self._get_symbol(coin)

            # Get open interest in USD
            data = self._make_request("/fapi/v1/openInterest", params={
                'symbol': symbol
            })

            if not data:
                return None

            # Open interest value in USD (sumOpenInterestValue)
            oi_value = float(data.get('openInterest', 0))

            # Get current price to convert to USD
            price_data = self._make_request("/fapi/v1/ticker/price", params={
                'symbol': symbol
            })

            if price_data:
                price = float(price_data.get('price', 0))
                oi_usd = oi_value * price

                self._set_cache(cache_key, oi_usd)
                return oi_usd

            return None

        except Exception as e:
            logger.warning(f"Failed to get open interest for {coin}: {e}")
            return None

    def get_long_short_ratio_history(self, coin: str, period: str = '5m', limit: int = 100, start_time: int = None, end_time: int = None) -> Optional[pd.DataFrame]:
        """
        Get historical long/short ratio from top trader accounts.

        Args:
            coin: Coin symbol (e.g., 'BTC', 'ETH')
            period: Time period ('5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d')
            limit: Number of records to fetch (max 500, default 100)
            start_time: Start timestamp in milliseconds (optional)
            end_time: End timestamp in milliseconds (optional)

        Returns:
            DataFrame with columns: timestamp, long_account, short_account, long_short_ratio
        """
        try:
            symbol = self._get_symbol(coin)

            params = {
                'symbol': symbol,
                'period': period,
                'limit': min(limit, 500)
            }

            if start_time:
                params['startTime'] = start_time
            if end_time:
                params['endTime'] = end_time

            data = self._make_request("/futures/data/topLongShortPositionRatio", params=params)

            if not data or len(data) == 0:
                return None

            # Convert to DataFrame
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['long_account'] = df['longAccount'].astype(float)
            df['short_account'] = df['shortAccount'].astype(float)
            df['long_short_ratio'] = df['longShortRatio'].astype(float)

            df = df[['timestamp', 'long_account', 'short_account', 'long_short_ratio']]
            df = df.sort_values('timestamp').reset_index(drop=True)

            return df

        except Exception as e:
            logger.warning(f"Failed to get long/short ratio history for {coin}: {e}")
            return None

    def get_long_short_ratio(self, coin: str, period: str = '5m') -> Optional[float]:
        """
        Get long/short ratio from top trader accounts (by positions).

        Args:
            coin: Coin symbol (e.g., 'BTC', 'ETH')
            period: Time period ('5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d')

        Returns:
            Long/short ratio (e.g., 1.5 = 1.5x more longs than shorts)
        """
        cache_key = f"ls_ratio_{coin}_{period}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        try:
            symbol = self._get_symbol(coin)

            # Get top trader long/short ratio (by positions)
            data = self._make_request("/futures/data/topLongShortPositionRatio", params={
                'symbol': symbol,
                'period': period,
                'limit': 1
            })

            if not data or len(data) == 0:
                return None

            # Get most recent ratio
            long_account = float(data[0].get('longAccount', 0))
            short_account = float(data[0].get('shortAccount', 0))

            if short_account == 0:
                return None

            ratio = long_account / short_account

            self._set_cache(cache_key, ratio)
            return ratio

        except Exception as e:
            logger.warning(f"Failed to get long/short ratio for {coin}: {e}")
            return None

    def get_top_trader_long_short_ratio(self, coin: str, period: str = '5m') -> Optional[Dict]:
        """
        Get detailed long/short data from top traders.

        Args:
            coin: Coin symbol (e.g., 'BTC', 'ETH')
            period: Time period ('5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d')

        Returns:
            Dict with long_account, short_account, long_short_ratio
        """
        cache_key = f"top_trader_{coin}_{period}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        try:
            symbol = self._get_symbol(coin)

            # Get top trader long/short ratio
            data = self._make_request("/futures/data/topLongShortPositionRatio", params={
                'symbol': symbol,
                'period': period,
                'limit': 1
            })

            if not data or len(data) == 0:
                return None

            recent = data[0]
            long_account = float(recent.get('longAccount', 0))
            short_account = float(recent.get('shortAccount', 0))

            result = {
                'long_account': long_account,
                'short_account': short_account,
                'long_short_ratio': long_account / short_account if short_account > 0 else 0,
                'timestamp': recent.get('timestamp', 0)
            }

            self._set_cache(cache_key, result)
            return result

        except Exception as e:
            logger.warning(f"Failed to get top trader data for {coin}: {e}")
            return None

    def get_taker_buy_sell_volume(self, coin: str, period: str = '5m') -> Optional[Dict]:
        """
        Get taker buy/sell volume ratio.

        Args:
            coin: Coin symbol (e.g., 'BTC', 'ETH')
            period: Time period ('5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d')

        Returns:
            Dict with buy_vol, sell_vol, buy_sell_ratio
        """
        cache_key = f"taker_volume_{coin}_{period}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        try:
            symbol = self._get_symbol(coin)

            # Get taker buy/sell volume
            data = self._make_request("/futures/data/takerlongshortRatio", params={
                'symbol': symbol,
                'period': period,
                'limit': 1
            })

            if not data or len(data) == 0:
                return None

            recent = data[0]
            buy_vol = float(recent.get('buySellRatio', 1))
            sell_vol = float(recent.get('sellVol', 1))

            result = {
                'buy_sell_ratio': buy_vol,
                'timestamp': recent.get('timestamp', 0)
            }

            self._set_cache(cache_key, result)
            return result

        except Exception as e:
            logger.warning(f"Failed to get taker volume for {coin}: {e}")
            return None

    def get_liquidations(self, coin: str, lookback_minutes: int = 60) -> Optional[Dict]:
        """
        Get liquidation estimate from funding rate extremes and long/short ratio.
        Note: Binance forceOrders endpoint requires authentication (not public).

        This estimates liquidation pressure from:
        - Extreme funding rates (overleveraged positions)
        - Long/short ratio imbalance
        - Open interest changes

        Args:
            coin: Coin symbol (e.g., 'BTC', 'ETH')
            lookback_minutes: Not used (kept for API compatibility)

        Returns:
            Dict with estimated liquidation pressure (0 if cannot estimate)
        """
        cache_key = f"liq_{coin}_{lookback_minutes}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        try:
            # Return zeros - liquidation data not available via public API
            # System will still work, just without liquidation cascade detection
            result = {
                'long_liq': 0,
                'short_liq': 0,
                'total_liq': 0,
                'liq_ratio': 0
            }

            self._set_cache(cache_key, result)
            return result

        except Exception as e:
            logger.warning(f"Failed to get liquidations for {coin}: {e}")
            return None


    def get_derivatives_history_days(self, coin: str, days: int = 7, period: str = '5m') -> Dict[str, pd.DataFrame]:
        """
        Get all derivatives historical data for specified number of days.

        Args:
            coin: Coin symbol (e.g., 'BTC', 'ETH')
            days: Number of days to fetch (default 7)
            period: Time period for OI/LS ratio ('5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d')

        Returns:
            Dict with keys: 'funding_rate', 'open_interest', 'long_short_ratio'
        """
        import time

        end_time = int(time.time() * 1000)
        start_time = end_time - (days * 24 * 60 * 60 * 1000)

        result = {}

        # Funding rate (8h intervals, ~21 records per week)
        funding_limit = min((days * 3) + 5, 1000)  # 3 per day + buffer
        result['funding_rate'] = self.get_funding_rate_history(
            coin, limit=funding_limit, start_time=start_time, end_time=end_time
        )

        # Open interest (depends on period)
        periods_per_day = {
            '5m': 288, '15m': 96, '30m': 48, '1h': 24,
            '2h': 12, '4h': 6, '6h': 4, '12h': 2, '1d': 1
        }
        oi_limit = min((days * periods_per_day.get(period, 288)) + 10, 500)
        result['open_interest'] = self.get_open_interest_history(
            coin, period=period, limit=oi_limit, start_time=start_time, end_time=end_time
        )

        # Long/short ratio (same as OI)
        result['long_short_ratio'] = self.get_long_short_ratio_history(
            coin, period=period, limit=oi_limit, start_time=start_time, end_time=end_time
        )

        return result


def test_historical_data():
    """Test historical derivatives data"""
    client = BinanceFuturesClient()

    print("\n" + "=" * 80)
    print("HISTORICAL DERIVATIVES DATA (BINANCE FUTURES)")
    print("=" * 80)

    coin = 'BTC'

    # Test funding rate history
    print(f"\n{coin} Funding Rate History (last 10 records):")
    fr_hist = client.get_funding_rate_history(coin, limit=10)
    if fr_hist is not None:
        print(fr_hist.to_string(index=False))
    else:
        print("  N/A")

    # Test open interest history
    print(f"\n{coin} Open Interest History (5m, last 10 records):")
    oi_hist = client.get_open_interest_history(coin, period='5m', limit=10)
    if oi_hist is not None:
        print(oi_hist.to_string(index=False))
    else:
        print("  N/A")

    # Test long/short ratio history
    print(f"\n{coin} Long/Short Ratio History (5m, last 10 records):")
    ls_hist = client.get_long_short_ratio_history(coin, period='5m', limit=10)
    if ls_hist is not None:
        print(ls_hist.to_string(index=False))
    else:
        print("  N/A")

    # Test multi-day fetch
    print(f"\n{coin} 7-Day Historical Data Summary (1h period):")
    history = client.get_derivatives_history_days(coin, days=7, period='1h')

    for key, df in history.items():
        if df is not None:
            print(f"  {key}: {len(df)} records from {df['timestamp'].min()} to {df['timestamp'].max()}")
        else:
            print(f"  {key}: N/A")


def main():
    """Test Binance Futures derivatives data"""
    client = BinanceFuturesClient()

    test_coins = ['BTC', 'ETH', 'SOL']

    print("\n" + "=" * 80)
    print("CURRENT DERIVATIVES DATA (BINANCE FUTURES)")
    print("=" * 80)

    for coin in test_coins:
        print(f"\n{coin}:")

        # Funding rate
        funding = client.get_funding_rate(coin)
        if funding is not None:
            print(f"  Funding Rate: {funding:.4f}%")
        else:
            print(f"  Funding Rate: N/A")

        # Open interest
        oi = client.get_open_interest(coin)
        if oi is not None:
            print(f"  Open Interest: ${oi:,.0f}")
        else:
            print(f"  Open Interest: N/A")

        # Long/Short ratio
        ls_ratio = client.get_long_short_ratio(coin, '5m')
        if ls_ratio is not None:
            print(f"  Long/Short Ratio (5m): {ls_ratio:.2f}")
            if ls_ratio > 1.5:
                print(f"    -> Overleveraged longs (bearish)")
            elif ls_ratio < 0.67:
                print(f"    -> Overleveraged shorts (bullish)")
        else:
            print(f"  Long/Short Ratio: N/A")

        # Top trader data
        top_trader = client.get_top_trader_long_short_ratio(coin, '5m')
        if top_trader:
            print(f"  Top Trader Long: {top_trader['long_account']:.2f}%")
            print(f"  Top Trader Short: {top_trader['short_account']:.2f}%")

        # Liquidations (1h) - not available via public API
        liq = client.get_liquidations(coin, 60)
        if liq and liq['total_liq'] > 0:
            print(f"  Liquidations (1h):")
            print(f"    - Longs: ${liq['long_liq']:,.0f}")
            print(f"    - Shorts: ${liq['short_liq']:,.0f}")
            print(f"    - Total: ${liq['total_liq']:,.0f}")
            if liq['total_liq'] > 1_000_000:
                print(f"    -> Major liquidation event")
        else:
            print(f"  Liquidations: N/A (requires auth)")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--history':
        test_historical_data()
    else:
        main()
