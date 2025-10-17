# ============================================================================
# SYSTEM CONFIGURATION VARIABLES
# ============================================================================

import os

# System Control Configuration
ENABLE_DATA_DOWNLOAD = True  # Enable automatic data downloading (set to False for faster testing)
ENABLE_OUTLIER_DETECTION = True  # Enable outlier detection system
ENABLE_FULL_SYSTEM_STATUS = True  # Enable full system status reporting

# Data Download Configuration - STAT-ARB REQUIRES MORE HISTORY
DATA_DOWNLOAD_DAYS = 30  # Days of data to download for 14-day z-score window + buffer
FORCE_DATA_REFRESH = True  # Force re-download of existing data
DOWNLOAD_INTERVALS = ["5m", "15m", "1h", "4h", "1d"]  # Timeframes to download

# System Display Configuration
MAX_OUTLIER_DISPLAY = 999  # Maximum outliers to display in ranking (999 = all coins)
DISPLAY_SYSTEM_METRICS = True  # Show system performance metrics

# Advanced System Configuration
VERBOSE_OUTPUT = False  # Enable detailed logging output
SYSTEM_HEALTH_CHECK = True  # Enable system health monitoring

# Coin Configuration
COINS_DATA = {
    # MAJOR ASSETS
    'BTC': {'narratives': ['store_of_value', 'digital_gold', 'bitcoin']},
    'ETH': {'narratives': ['smart_contracts', 'defi', 'ethereum_ecosystem', 'layer1']},
    'BNB': {'narratives': ['exchange_token', 'bsc_ecosystem', 'centralized_exchange']},
    'ADA': {'narratives': ['layer1', 'proof_of_stake', 'academic']},
    'XRP': {'narratives': ['payments', 'enterprise', 'banking']},
    'SOL': {'narratives': ['layer1', 'high_performance', 'solana_ecosystem', 'defi', 'nft']},
    'DOT': {'narratives': ['interoperability', 'layer1', 'polkadot_ecosystem']},
    'AVAX': {'narratives': ['layer1', 'ethereum_alternative', 'enterprise']},
    'LINK': {'narratives': ['oracle', 'defi', 'infrastructure']},
    'LTC': {'narratives': ['payments', 'bitcoin_fork', 'store_of_value']},
    
    # ESTABLISHED ALTCOINS
    'UNI': {'narratives': ['defi', 'dex', 'ethereum_ecosystem']},
    'ALGO': {'narratives': ['layer1', 'proof_of_stake', 'enterprise']},
    'BCH': {'narratives': ['payments', 'bitcoin_fork', 'scaling']},
    'XLM': {'narratives': ['payments', 'banking', 'enterprise']},
    'VET': {'narratives': ['enterprise', 'supply_chain', 'real_world_utility']},
    'FIL': {'narratives': ['storage', 'web3', 'infrastructure']},
    'ETC': {'narratives': ['layer1', 'proof_of_work', 'ethereum_classic']},
    'ATOM': {'narratives': ['interoperability', 'layer1', 'cosmos_ecosystem']},
    'HBAR': {'narratives': ['enterprise', 'layer1', 'hashgraph']},
    'ICP': {'narratives': ['web3', 'layer1', 'decentralized_internet']},
    
    # LAYER 1 & LAYER 2
    'APT': {'narratives': ['layer1', 'high_performance', 'move_language']},
    'ARB': {'narratives': ['layer2', 'ethereum_ecosystem', 'scaling']},
    'SUI': {'narratives': ['layer1', 'high_performance', 'move_language']},
    'NEAR': {'narratives': ['layer1', 'developer_friendly', 'web3']},
    'TIA': {'narratives': ['modular_blockchain', 'layer1', 'infrastructure']},
    'SEI': {'narratives': ['layer1', 'trading_focused', 'high_performance']},
    'TON': {'narratives': ['layer1', 'telegram', 'messaging']},
    'INJ': {'narratives': ['defi', 'derivatives', 'layer1']},
    
    # DEFI PROTOCOLS
    'CRV': {'narratives': ['defi', 'dex', 'ethereum_ecosystem', 'stablecoins']},
    'GRT': {'narratives': ['infrastructure', 'web3', 'indexing']},
    'RUNE': {'narratives': ['defi', 'cross_chain', 'dex']},
    'LDO': {'narratives': ['defi', 'staking', 'ethereum_ecosystem']},
    'CAKE': {'narratives': ['defi', 'dex', 'bsc_ecosystem']},
    'SUSHI': {'narratives': ['defi', 'dex', 'ethereum_ecosystem']},
    'ENS': {'narratives': ['web3', 'ethereum_ecosystem', 'infrastructure']},
    'AAVE': {'narratives': ['defi', 'lending', 'ethereum_ecosystem']},
    'PENDLE': {'narratives': ['defi', 'yield', 'ethereum_ecosystem']},
    'ENA': {'narratives': ['defi', 'stablecoins', 'ethereum_ecosystem']},
    'MORPHO': {'narratives': ['defi', 'lending', 'ethereum_ecosystem']},
    
    # MEMECOINS
    'DOGE': {'narratives': ['meme', 'payments', 'elon', 'community']},
    'PEPE': {'narratives': ['meme', 'ethereum_ecosystem', 'community']},
    'BONK': {'narratives': ['meme', 'solana_ecosystem', 'community']},
    'WIF': {'narratives': ['meme', 'solana_ecosystem', 'community']},
    'FLOKI': {'narratives': ['meme', 'elon', 'community']},
    'FARTCOIN': {'narratives': ['meme', 'high_volatility', 'speculative']},
    
    # GAMING & NFTS
    'GALA': {'narratives': ['gaming', 'nft', 'metaverse']},
    'ENJ': {'narratives': ['gaming', 'nft', 'metaverse']},
    'IMX': {'narratives': ['gaming', 'nft', 'layer2', 'ethereum_ecosystem']},
    'FLOW': {'narratives': ['nft', 'gaming', 'layer1']},
    
    # AI & EMERGING
    'TAO': {'narratives': ['ai', 'infrastructure', 'decentralized_compute']},
    'WLD': {'narratives': ['ai', 'identity', 'sam_altman']},
    'VIRTUAL': {'narratives': ['ai', 'metaverse', 'agents']},
    'HYPE': {'narratives': ['derivatives', 'defi', 'high_performance']},
    
    # SPECIALIZED
    'DYDX': {'narratives': ['derivatives', 'defi', 'trading']},
    'PYTH': {'narratives': ['oracle', 'infrastructure', 'trading']},
    'JUP': {'narratives': ['defi', 'dex', 'solana_ecosystem']},
    'SPX': {'narratives': ['meme', 'tradfi', 'speculative']},
    'ETHFI': {'narratives': ['defi', 'staking', 'ethereum_ecosystem']},
    'XTZ': {'narratives': ['layer1', 'governance', 'enterprise']},
    'PUMP': {'narratives': ['meme', 'solana_ecosystem', 'platform']},
    'S': {'narratives': ['unknown']},
    'TRX': {'narratives': ['layer1', 'entertainment', 'payments']}
}

# Extract coin symbols
COINS = list(COINS_DATA.keys())

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

# Z-Score Acceleration (Rate of change in velocity) - DISABLED FOR STAT-ARB
# Third derivative over short windows = noise. Renaissance avoided this.
Z_SCORE_ACCELERATION_ENABLED = False  # DISABLED - too noisy for stat-arb
Z_SCORE_ACCELERATION_WINDOW = 12  # Periods to calculate acceleration (1 hour on 5m)
Z_SCORE_ACCELERATION_THRESHOLD = 0.1  # Minimum acceleration for early entry
Z_SCORE_ACCELERATION_DISPLAY_MULTIPLIER = 1000  # Scale acceleration for display (1000x)

# Predictive Signal Classification
PREDICTIVE_SIGNAL_ENABLED = True
PREDICTIVE_EARLY_ENTRY_Z = 1.5  # Z-score threshold for early entry (before peak)
PREDICTIVE_MOMENTUM_EXHAUSTION_Z = 3.0  # Z-score threshold for exhaustion
PREDICTIVE_VELOCITY_REVERSAL_THRESHOLD = -0.2  # Velocity turning negative = reversal

# ============================================================================
# AGGREGATED DERIVATIVES SIGNALS - COINGLASS (10+ EXCHANGES)
# ============================================================================

# Funding Rate (Volume-weighted across all exchanges)
FUNDING_RATE_ENABLED = False  # Disabled - API rate limits
FUNDING_RATE_EXTREME_THRESHOLD = 0.15  # 0.15% = extreme overleveraged
FUNDING_RATE_MULTIPLIER = 1.25  # 25% boost for extreme funding

# Open Interest (Aggregated total OI)
OI_ENABLED = False  # Disabled - API rate limits
OI_THRESHOLD_USD = 100_000_000  # $100M minimum for liquidity filter

# Liquidations (Cascade detection)
LIQUIDATIONS_ENABLED = False  # Disabled - API rate limits
LIQUIDATION_THRESHOLD_USD = 10_000_000  # $10M in 1h = cascade
LIQUIDATION_MULTIPLIER = 1.3  # 30% boost for liquidation events

# Long/Short Ratio (Aggregated positioning)
LONG_SHORT_RATIO_ENABLED = False  # Disabled - API rate limits
LS_RATIO_EXTREME_LONG = 1.8  # >1.8 = overleveraged longs (bearish)
LS_RATIO_EXTREME_SHORT = 0.5  # <0.5 = overleveraged shorts (bullish)
LS_RATIO_MULTIPLIER = 1.2  # 20% boost for extreme positioning

# NOTE: Coinglass free API has rate limits. Enable these when you have API access.
# Alternative: Use paid Coinglass API key or implement multi-exchange fetching yourself.

# Volume/Market Cap Configuration
VOLUME_MCAP_NORMALIZATION = 'log'  # 'log' or 'minmax' normalization

# CoinMarketCap API Configuration
CMC_API_KEY = "1f95581d-5beb-4bf5-985e-cb8fac961084"
CMC_BASE_URL = "https://pro-api.coinmarketcap.com/v1"
USE_REAL_MARKET_CAP = True  # Set to False to use proxy method

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
        
        # Download outlier detection data
        print(f"Downloading {DATA_DOWNLOAD_DAYS} days of data for {len(DOWNLOAD_INTERVALS)} timeframes: {DOWNLOAD_INTERVALS}...")
        start_time = time.time()
        downloader = BinanceDataDownloader(DATA_FOLDER)
        downloader.download_all_coins(days_back=DATA_DOWNLOAD_DAYS, intervals=DOWNLOAD_INTERVALS)
        elapsed = time.time() - start_time
        print(f"Data download completed in {elapsed:.1f}s")
            
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
            print(f"{'Rank':<4} {'Coin':<8} {'Price':<12} {'Z-Score':<10} {'Vel':<8} {'Accel':<8} {'Score':<8} {'HL(h)':<6}")
            print("-" * 70)

            display_count = min(MAX_OUTLIER_DISPLAY, len(all_scores))
            for idx, (_, row) in enumerate(all_scores.head(display_count).iterrows(), 1):
                z_score = row.get('z_score', 0)
                z_velocity = row.get('z_velocity', 0) * Z_SCORE_VELOCITY_DISPLAY_MULTIPLIER
                z_accel = row.get('z_accel', 0) * Z_SCORE_ACCELERATION_DISPLAY_MULTIPLIER
                score = row.get('relative_score', 0)
                half_life = row.get('half_life', 999)

                print(f"{idx:<4} {row['coin']:<8} {row['close_price']:<12.4f} {z_score:<10.2f} {z_velocity:<8.2f} {z_accel:<8.2f} {score:<8.2f} {half_life:<6.1f}")
                
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
