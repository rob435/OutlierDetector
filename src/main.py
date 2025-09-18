# Configuration Variables
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
    'MKR': {'narratives': ['defi', 'stablecoins', 'ethereum_ecosystem', 'dao']},
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

# Extract coin symbols for backward compatibility
COINS = list(COINS_DATA.keys())

# Utility functions for narrative analysis
def get_coin_narratives(coin: str) -> list:
    """Get narratives for a specific coin"""
    return COINS_DATA.get(coin, {}).get('narratives', [])

def get_coins_by_narrative(narrative: str) -> list:
    """Get all coins that have a specific narrative"""
    return [coin for coin, data in COINS_DATA.items() if narrative in data.get('narratives', [])]

def get_narrative_groups() -> dict:
    """Get grouped coins by their primary narratives"""
    narrative_groups = {}
    for coin, data in COINS_DATA.items():
        for narrative in data.get('narratives', []):
            if narrative not in narrative_groups:
                narrative_groups[narrative] = []
            narrative_groups[narrative].append(coin)
    return narrative_groups

# Data configuration
DATA_FOLDER = "C:/Users/user/Desktop/strength/realalgo/data"
TIMEFRAME = "5m"

# Narrative Analysis Configuration
NARRATIVE_DATA_FOLDER = "C:/Users/user/Desktop/strength/realalgo/narrative_data"
NARRATIVE_TIMEFRAME = "1h"
NARRATIVE_HISTORY_DAYS = 7  # 1 week of data for narrative analysis

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

def run_outlier_analysis():
    """Run the outlier detection system"""
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
    

def run_narrative_analysis():
    """Run the narrative prediction system"""
    from narrative_downloader import NarrativeDataDownloader
    from narrative_analyzer import NarrativeAnalyzer
    
    print("\n" + "="*50)
    print("NARRATIVE ANALYSIS SYSTEM")
    print("="*50)
    
    # Download narrative data if needed
    print("Checking narrative data...")
    downloader = NarrativeDataDownloader()
    # Only download if we don't have recent data
    # downloader.download_all_coins(days_back=NARRATIVE_HISTORY_DAYS)
    
    # Run narrative analysis
    analyzer = NarrativeAnalyzer()
    summary = analyzer.get_narrative_summary()
    
    print(f"\n=== NARRATIVE SUMMARY ===")
    print(f"Total narratives tracked: {summary['total_narratives']}")
    print(f"Currently pumping narrative: {summary['current_pumping_narrative'] or 'None detected'}")
    
    if summary['next_predicted_narrative']:
        print(f"Predicted next narrative: {summary['next_predicted_narrative']}")
        print(f"Confidence: {summary['prediction_confidence']:.2%}")
        
        print(f"\n=== TOP NARRATIVE PREDICTIONS ===")
        for i, pred in enumerate(summary['top_predictions'][:3], 1):
            print(f"{i}. {pred['narrative']} ({pred['probability']:.2%})")
            print(f"   Coins: {', '.join(pred['coins'][:5])}{'...' if len(pred['coins']) > 5 else ''}")
    else:
        print("No predictions available (model needs training)")

def main():
    """Main function - run complete analysis system"""
    print("=== CRYPTO ANALYSIS SYSTEM ===")
    
    # Run outlier analysis first (fast)
    run_outlier_analysis()
    
    # Then handle narrative analysis setup (slow on first run)
    import os
    narrative_data_exists = os.path.exists(NARRATIVE_DATA_FOLDER) and len(os.listdir(NARRATIVE_DATA_FOLDER)) > 10
    
    if not narrative_data_exists:
        print("\n" + "="*50)
        print("FIRST RUN SETUP - DOWNLOADING NARRATIVE DATA")
        print("="*50)
        print("Downloading 1-year of 1h data for narrative analysis...")
        print("This will take several minutes on first run...")
        from narrative_downloader import NarrativeDataDownloader
        downloader = NarrativeDataDownloader()
        downloader.download_all_coins(days_back=NARRATIVE_HISTORY_DAYS)
        narrative_data_exists = True
    
    # Check if model exists, train if needed
    model_exists = os.path.exists(f"{NARRATIVE_DATA_FOLDER}/narrative_model.pkl")
    
    if not model_exists and narrative_data_exists:
        print("\nTraining narrative prediction model...")
        from narrative_analyzer import NarrativeAnalyzer
        analyzer = NarrativeAnalyzer()
        analyzer.train_narrative_prediction_model()
    
    # Run narrative analysis
    run_narrative_analysis()

if __name__ == "__main__":
    main()
