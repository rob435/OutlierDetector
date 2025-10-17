# ============================================================================
# SYSTEM CONFIGURATION VARIABLES
# ============================================================================

import os

# System Control Configuration
ENABLE_DATA_DOWNLOAD = False  # Enable automatic data downloading (set to False for faster testing)
ENABLE_OUTLIER_DETECTION = True  # Enable outlier detection system
ENABLE_FULL_SYSTEM_STATUS = True  # Enable full system status reporting

# Data Download Configuration - STAT-ARB REQUIRES MORE HISTORY
DATA_DOWNLOAD_DAYS = 30  # Days of OHLCV data to download for 14-day z-score window + buffer
FORCE_DATA_REFRESH = True  # Force re-download of existing data
DOWNLOAD_INTERVALS = ["5m", "15m", "1h", "4h", "1d"]  # Timeframes to download

# Derivatives Historical Data Configuration
DOWNLOAD_DERIVATIVES_HISTORY = True # Download historical funding/OI/LS data for backtesting
DERIVATIVES_DOWNLOAD_DAYS = 20  # Days of derivatives history (limited by Binance retention)
DERIVATIVES_DOWNLOAD_PERIOD = "4h"  # Period for OI/LS data (4h allows 30 days, 1h only allows 20 days)

# System Display Configuration
MAX_OUTLIER_DISPLAY = 999  # Maximum outliers to display in ranking (999 = all coins)
DISPLAY_SYSTEM_METRICS = True  # Show system performance metrics

# Advanced System Configuration
VERBOSE_OUTPUT = False  # Enable detailed logging output
SYSTEM_HEALTH_CHECK = True  # Enable system health monitoring

# Coin Configuration - Simple List
COINS = [
    # MAJOR ASSETS
    'BTC', 'ETH', 'BNB', 'ADA', 'XRP', 'SOL', 'DOT', 'AVAX', 'LINK', 'LTC',

    # ESTABLISHED ALTCOINS
    'UNI', 'ALGO', 'BCH', 'XLM', 'VET', 'FIL', 'ETC', 'ATOM', 'HBAR', 'ICP',

    # LAYER 1 & LAYER 2
    'APT', 'ARB', 'SUI', 'NEAR', 'TIA', 'SEI', 'TON', 'INJ',

    # DEFI PROTOCOLS
    'CRV', 'GRT', 'RUNE', 'LDO', 'CAKE', 'SUSHI', 'ENS', 'AAVE', 'PENDLE', 'ENA', 'MORPHO',

    # MEMECOINS
    'DOGE', 'PEPE', 'BONK', 'WIF', 'FLOKI', 'FARTCOIN',

    # GAMING & NFTS
    'GALA', 'ENJ', 'IMX', 'FLOW',

    # AI & EMERGING
    'TAO', 'WLD', 'VIRTUAL', 'HYPE',

    # SPECIALIZED
    'DYDX', 'PYTH', 'JUP', 'SPX', 'ETHFI', 'XTZ', 'PUMP', 'S', 'TRX'
]

# Data configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FOLDER = os.path.join(BASE_DIR, "data")
PRIMARY_TIMEFRAME = "5m"  # Primary timeframe for analysis

# Outlier Detection Configuration - STATISTICAL ARBITRAGE APPROACH
# Renaissance-style: Longer lookback windows to capture genuine mispricings, not noise
Z_SCORE_WINDOW = 4032  # Number of bars for Z-score calculation (14 days of 5m bars)
PRICE_CHANGE_PERIOD = 288  # Number of bars for price change calculation (24 hours of 5m bars)

# ============================================================================
# CORE FEATURES - ORTHOGONAL SIGNALS
# ============================================================================

# Core Signal: BTC-Relative Z-Score (PRIMARY EDGE)
USE_MODIFIED_ZSCORE = True  # MAD-based Z-score (fat-tail robust)
MAD_SCALE_FACTOR = 0.6745  # Standard MAD scaling factor

# Volume Confirmation (ORTHOGONAL - not price-based)
VOLUME_SURGE_THRESHOLD = 2.0  # Only significant surges (>2 std devs)
VOLUME_SURGE_MULTIPLIER = 1.3  # 30% boost for volume-confirmed moves

# Half-Life (useful for position sizing, not scoring)
HALF_LIFE_ENABLED = True  # Calculate mean reversion speed
HALF_LIFE_MIN_PERIODS = 100  # Minimum data for reliable estimate

# Advanced Multi-Timeframe (1h/4h/1d structural confirmation)
MULTI_TIMEFRAMES_ADVANCED = ["1h", "4h", "1d"]
MTF_REQUIRE_ALIGNMENT = 2  # Minimum aligned timeframes for confirmation

# ============================================================================
# PREDICTIVE Z-SCORE SIGNALS - TRANSFORM FROM REACTIVE TO PREDICTIVE
# ============================================================================

# Z-Score Velocity (Rate of change in Z-score) - STAT-ARB CONFIGURATION
# Measure z-score momentum over meaningful timeframes, not microstructure noise
Z_SCORE_VELOCITY_ENABLED = True
Z_SCORE_VELOCITY_WINDOW = 288  # Periods to calculate velocity (24 hours on 5m)
Z_SCORE_VELOCITY_THRESHOLD = 0.3  # Minimum velocity for momentum signal
Z_SCORE_VELOCITY_DISPLAY_MULTIPLIER = 100  # Scale velocity for display (100x)


# Predictive Signal Classification
PREDICTIVE_SIGNAL_ENABLED = True
PREDICTIVE_EARLY_ENTRY_Z = 1.5  # Z-score threshold for early entry (before peak)
PREDICTIVE_MOMENTUM_EXHAUSTION_Z = 3.0  # Z-score threshold for exhaustion
PREDICTIVE_VELOCITY_REVERSAL_THRESHOLD = -0.2  # Velocity turning negative = reversal

# ============================================================================
# BINANCE FUTURES DERIVATIVES SIGNALS (PUBLIC API - NO AUTH REQUIRED)
# ============================================================================

# Funding Rate (8h intervals on Binance)
FUNDING_RATE_ENABLED = True  # FREE - Binance public API
FUNDING_RATE_EXTREME_THRESHOLD = 0.15  # 0.15% = extreme overleveraged
FUNDING_RATE_MULTIPLIER = 1.25  # 25% boost for extreme funding

# Open Interest (Binance Futures OI in USD)
OI_ENABLED = True  # FREE - Binance public API
OI_THRESHOLD_USD = 100_000_000  # $100M minimum for liquidity filter


# Long/Short Ratio (Top trader positions on Binance)
LONG_SHORT_RATIO_ENABLED = True  # FREE - Binance public API
LS_RATIO_EXTREME_LONG = 1.8  # >1.8 = overleveraged longs (bearish)
LS_RATIO_EXTREME_SHORT = 0.5  # <0.5 = overleveraged shorts (bullish)
LS_RATIO_MULTIPLIER = 1.2  # 20% boost for extreme positioning

# NOTE: Binance Futures = 40%+ of crypto derivatives market (sufficient for stat-arb)
# All endpoints are PUBLIC - no API key required, no rate limit issues

# Volume/Market Cap Configuration
VOLUME_MCAP_NORMALIZATION = 'log'  # 'log' or 'minmax' normalization

# CoinMarketCap API Configuration
CMC_API_KEY = "1f95581d-5beb-4bf5-985e-cb8fac961084"
CMC_BASE_URL = "https://pro-api.coinmarketcap.com/v1"
USE_REAL_MARKET_CAP = True  # Set to False to use proxy method

def download_derivatives_history():
    """Download historical derivatives data (funding rate, OI, long/short ratio) for all coins"""
    if not DOWNLOAD_DERIVATIVES_HISTORY:
        print("Derivatives history download disabled in configuration")
        return False

    print("\n" + "=" * 60)
    print("DERIVATIVES HISTORICAL DATA DOWNLOAD")
    print("=" * 60)

    success = True

    try:
        from binance_futures_client import BinanceFuturesClient
        import time

        client = BinanceFuturesClient()

        print(f"Downloading {DERIVATIVES_DOWNLOAD_DAYS} days of derivatives data for {len(COINS)} coins...")
        print(f"Period: {DERIVATIVES_DOWNLOAD_PERIOD} (funding=8h, OI/LS={DERIVATIVES_DOWNLOAD_PERIOD})")

        start_time = time.time()
        downloaded_count = 0
        failed_count = 0

        for coin in COINS:
            try:
                # Get all derivatives history
                history = client.get_derivatives_history_days(
                    coin,
                    days=DERIVATIVES_DOWNLOAD_DAYS,
                    period=DERIVATIVES_DOWNLOAD_PERIOD
                )

                # Save to parquet files
                if history['funding_rate'] is not None:
                    history['funding_rate'].to_parquet(f'{DATA_FOLDER}/{coin}_funding_{DERIVATIVES_DOWNLOAD_DAYS}d.parquet')

                if history['open_interest'] is not None:
                    history['open_interest'].to_parquet(f'{DATA_FOLDER}/{coin}_oi_{DERIVATIVES_DOWNLOAD_PERIOD}_{DERIVATIVES_DOWNLOAD_DAYS}d.parquet')

                if history['long_short_ratio'] is not None:
                    history['long_short_ratio'].to_parquet(f'{DATA_FOLDER}/{coin}_ls_{DERIVATIVES_DOWNLOAD_PERIOD}_{DERIVATIVES_DOWNLOAD_DAYS}d.parquet')

                downloaded_count += 1

            except Exception as e:
                print(f"Failed to download derivatives for {coin}: {e}")
                failed_count += 1
                continue

        elapsed = time.time() - start_time

        print(f"\nDerivatives download completed in {elapsed:.1f}s")
        print(f"Success: {downloaded_count}/{len(COINS)} coins")

        if failed_count > 0:
            print(f"Failed: {failed_count} coins")
            success = False

    except Exception as e:
        print(f"Derivatives download failed: {e}")
        success = False

    return success


def run_data_download():
    """Download and update all required data"""
    if not ENABLE_DATA_DOWNLOAD:
        print("Data download disabled in configuration")
        return False

    print("=" * 60)
    print("DATA DOWNLOAD SYSTEM")
    print("=" * 60)

    success = True

    try:
        from binance_downloader import BinanceDataDownloader
        import time

        # Download OHLCV data
        print(f"Downloading {DATA_DOWNLOAD_DAYS} days of OHLCV data for {len(DOWNLOAD_INTERVALS)} timeframes: {DOWNLOAD_INTERVALS}...")
        start_time = time.time()
        downloader = BinanceDataDownloader(DATA_FOLDER)
        downloader.download_all_coins(days_back=DATA_DOWNLOAD_DAYS, intervals=DOWNLOAD_INTERVALS)
        elapsed = time.time() - start_time
        print(f"OHLCV download completed in {elapsed:.1f}s")

        # Download derivatives historical data
        if DOWNLOAD_DERIVATIVES_HISTORY:
            derivatives_success = download_derivatives_history()
            success = success and derivatives_success

    except Exception as e:
        print(f"Data download failed: {e}")
        success = False

    return success


def run_outlier_analysis():
    """Run the outlier detection system"""
    if not ENABLE_OUTLIER_DETECTION:
        print("Outlier detection disabled in configuration")
        return {}
        
    print("\n" + "=" * 60)
    print("OUTLIER DETECTION SYSTEM")
    print("=" * 60)
    
    try:
        import time
        from outlier_detector import OutlierDetector
        
        start_time = time.time()
        detector = OutlierDetector()
        all_scores = detector.get_all_current_scores()
        elapsed = time.time() - start_time
        
        if not all_scores.empty:
            print(f"\nAnalyzed {len(all_scores)} coins in {elapsed:.2f}s")
            print(f"\n=== TOP {min(MAX_OUTLIER_DISPLAY, len(all_scores))} OUTLIERS WITH PREDICTIVE SIGNALS ===")
            print(f"{'Rank':<4} {'Coin':<8} {'Price':<12} {'Z-Score':<10} {'Vel':<8} {'Score':<8} {'HL(h)':<6} {'VolZ':<6} {'FR%':<8} {'LS':<6} {'OI1h%':<8} {'OI24h%':<9}")
            print("-" * 115)

            display_count = min(MAX_OUTLIER_DISPLAY, len(all_scores))
            for idx, (_, row) in enumerate(all_scores.head(display_count).iterrows(), 1):
                z_score = row.get('z_score', 0)
                z_velocity = row.get('z_velocity', 0) * Z_SCORE_VELOCITY_DISPLAY_MULTIPLIER
                score = row.get('relative_score', 0)
                half_life = row.get('half_life', 999)
                vol_surge_z = row.get('volume_surge_z', 0)
                funding_rate = row.get('funding_rate', 0)
                ls_ratio = row.get('long_short_ratio', 0)
                oi_1h = row.get('oi_change_1h', 0)
                oi_24h = row.get('oi_change_24h', 0)

                print(f"{idx:<4} {row['coin']:<8} {row['close_price']:<12.4f} {z_score:<10.2f} {z_velocity:<8.2f} {score:<8.2f} {half_life:<6.1f} {vol_surge_z:<6.1f} {funding_rate:<8.3f} {ls_ratio:<6.2f} {oi_1h:<8.1f} {oi_24h:<9.1f}")

            return {'total_coins': len(all_scores), 'top_outliers': all_scores.head(10), 'execution_time': elapsed}
        else:
            print("No outlier data available")
            return {'total_coins': 0, 'top_outliers': None}
            
    except Exception as e:
        print(f"Outlier analysis failed: {e}")
        return {'error': str(e)}
    

def run_system_health_check():
    """Check system health and display metrics"""
    if not SYSTEM_HEALTH_CHECK:
        return {}
        
    print("\n" + "=" * 60)
    print("SYSTEM HEALTH CHECK")
    print("=" * 60)
    
    health_status = {}
    
    try:
        import os
        
        # Check data availability
        data_files = len([f for f in os.listdir(DATA_FOLDER) if f.endswith('.parquet')]) if os.path.exists(DATA_FOLDER) else 0
        
        expected_coins = len(COINS)
        data_status = "OK" if data_files >= expected_coins else "INCOMPLETE"
        
        print(f"Outlier data: {data_files} files [{data_status}]")
        
        health_status = {
            'data_coverage': min(1.0, data_files / expected_coins) if expected_coins else 0
        }
        
        # Overall health score
        health_score = health_status['data_coverage']
        
        health_status_text = "EXCELLENT" if health_score >= 0.9 else "GOOD" if health_score >= 0.7 else "DEGRADED" if health_score >= 0.5 else "POOR"
        print(f"\nSystem health: {health_score:.1%} [{health_status_text}]")
        health_status['health_score'] = health_score
        
        if health_score < 0.9:
            print("System recommendations:")
            if health_status['data_coverage'] < 0.9:
                print("- Run data download to update missing coin data")
        
        return health_status
        
    except Exception as e:
        print(f"Health check failed: {e}")
        return {'error': str(e)}

def main():
    """Main function - run complete analysis system"""
    import time
    start_time = time.time()
    
    print("=" * 80)
    print("CRYPTO OUTLIER DETECTION SYSTEM")
    print("=" * 80)
    print(f"System initialized with {len(COINS)} cryptocurrencies")
    
    if ENABLE_FULL_SYSTEM_STATUS:
        print("\nSystem Configuration:")
        print(f"- Data Download: {'Enabled' if ENABLE_DATA_DOWNLOAD else 'Disabled'}")
        print(f"- Outlier Detection: {'Enabled' if ENABLE_OUTLIER_DETECTION else 'Disabled'}")
    
    # Store results from each system component
    system_results = {}
    
    # 1. System Health Check
    if SYSTEM_HEALTH_CHECK:
        health_results = run_system_health_check()
        system_results['health'] = health_results
    
    # 2. Data Download System
    if ENABLE_DATA_DOWNLOAD:
        download_success = run_data_download()
        system_results['data_download'] = {'success': download_success}
    
    # 3. Outlier Detection System
    if ENABLE_OUTLIER_DETECTION:
        outlier_results = run_outlier_analysis()
        system_results['outlier_detection'] = outlier_results
    
    # 4. Final System Summary
    if ENABLE_FULL_SYSTEM_STATUS:
        end_time = time.time()
        execution_time = end_time - start_time
        
        print("\n" + "=" * 80)
        print("SYSTEM EXECUTION SUMMARY")
        print("=" * 80)
        
        print(f"Total execution time: {execution_time:.2f} seconds")
        
        # Component status summary
        for component, results in system_results.items():
            if isinstance(results, dict) and 'error' in results:
                print(f"- {component.replace('_', ' ').title()}: FAILED ({results['error']})")
            elif isinstance(results, dict) and results:
                print(f"- {component.replace('_', ' ').title()}: SUCCESS")
            else:
                print(f"- {component.replace('_', ' ').title()}: COMPLETED")
        
        # Key metrics summary
        if 'outlier_detection' in system_results and 'total_coins' in system_results['outlier_detection']:
            total_outliers = system_results['outlier_detection']['total_coins']
            print(f"\nKey Metrics:")
            print(f"- Total coins analyzed: {total_outliers}")
        
        if 'health' in system_results and 'health_score' in system_results['health']:
            health_score = system_results['health']['health_score']
            print(f"- System health: {health_score:.1%}")
        
        print("\nSystem ready for continuous monitoring.")
    
    return system_results

if __name__ == "__main__":
    main()
