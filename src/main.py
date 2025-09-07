# Configuration Variables
COINS = [
    'BTC', 'ETH', 'BNB', 'ADA', 'XRP', 'SOL', 'DOGE', 'DOT', 'AVAX', 'LINK',
    'LTC', 'UNI', 'ALGO', 'BCH', 'XLM', 'VET', 'FIL', 'ETC', 'ATOM',
    'HBAR', 'ICP', 'APT', 'ARB', 'CRV', 'GRT', 'RUNE', 'INJ', 'MKR', 'NEAR',
    'LDO', 'SUI', 'PEPE', 'BONK', 'WIF', 'FLOKI', 'JUP', 'PYTH', 'TIA', 'SEI',
    'CAKE', 'SUSHI', 'TAO', 'VIRTUAL', 'ENS', 'GALA', 'SPX', 'FLOW', 'CRV',
    'ETHFI', 'FARTCOIN', 'WLD', 'TON', 'XTZ' , 'PUMP', "PENDLE", "ENA", "AAVE", "HYPE", "S", "MORPHO"
]

# Data configuration
DATA_FOLDER = "/Applications/PairsAlgo/data"
TIMEFRAME = "5m"

# Outlier Detection Configuration
Z_SCORE_WINDOW = 288  # Number of bars for Z-score calculation (24 hours of 5m bars)
PRICE_CHANGE_PERIOD = 144  # Number of 5m bars for 5m price change calculation
EWMA_ALPHA = 0.005  # EWMA smoothing factor
Z_SCORE_WEIGHT = 0.74  # Weight for Z-score in final outlier score
PRICE_CHANGE_WEIGHT = 0.03  # Weight for 5m price change in final outlier score
VOLUME_MCAP_WEIGHT = 0.23  # Weight for volume/market cap ratio in final outlier score


# Volume/Market Cap Configuration
VOLUME_MCAP_NORMALIZATION = 'log'  # 'log' or 'minmax' normalization
HIGH_LIQUIDITY_THRESHOLD = 0.8  # Threshold for high significance events
LOW_LIQUIDITY_THRESHOLD = 0.2  # Threshold for potential manipulation detection

# CoinMarketCap API Configuration
CMC_API_KEY = "1f95581d-5beb-4bf5-985e-cb8fac961084"
CMC_BASE_URL = "https://pro-api.coinmarketcap.com/v1"
USE_REAL_MARKET_CAP = True  # Set to False to use proxy method

def main():
    """Main function to download data and run outlier detection"""
    from binance_downloader import BinanceDataDownloader
    from outlier_detector import OutlierDetector
    
    # Download latest 3 days of data
    print("Downloading latest crypto data...")
    downloader = BinanceDataDownloader(DATA_FOLDER)
    downloader.download_all_coins(days_back=3)
    
    # Run outlier detection
    print("\nRunning outlier detection...")
    detector = OutlierDetector()
    
    # Get all current scores
    all_scores = detector.get_all_current_scores()
    
    if not all_scores.empty:
        print("\n=== ALL COINS RANKED BY RELATIVE SCORE ===")
        print(f"{'Rank':<4} {'Coin':<8} {'Price':<12} {'BTC Ratio':<12} {'Z-Score':<8} {'Price Change%':<14} {'Vol/MCap':<11} {'Score':<8}")
        print("-" * 95)
        
        for idx, (_, row) in enumerate(all_scores.iterrows(), 1):
            btc_ratio = row.get('coin_btc_ratio', 0)
            price_change = row.get('price_change', 0)
            volume_mcap_norm = row.get('volume_mcap_normalized', 0)
            relative_score = row.get('relative_score', 0)
            
            print(f"{idx:<4} {row['coin']:<8} {row['close_price']:<12.4f} {btc_ratio:<12.6f} {row['z_score']:<8.2f} {price_change:<14.2f} {volume_mcap_norm:<11.4f} {relative_score:<8.2f}")
    else:
        print("No data available")
    
    # Get summary
    summary = detector.get_outlier_summary()
    if summary:
        print(f"\n=== SUMMARY ===")
        print(f"Total coins analyzed: {summary['total_coins_analyzed']}")
        print(f"Current outliers: {summary['current_outliers']}")
        print(f"Outlier percentage: {summary['outlier_percentage']:.1f}%")
        print(f"Average relative score: {summary['avg_relative_score']:.2f}")
        print(f"Max relative score: {summary['max_relative_score']:.2f}")
        print(f"Min relative score: {summary['min_relative_score']:.2f}")
        if summary['top_outlier']:
            print(f"Top performing coin: {summary['top_outlier']}")

if __name__ == "__main__":
    main()
