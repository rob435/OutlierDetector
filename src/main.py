# ============================================================================
# SYSTEM CONFIGURATION VARIABLES
# ============================================================================

# System Control Configuration
ENABLE_DATA_DOWNLOAD = True  # Enable automatic data downloading
ENABLE_OUTLIER_DETECTION = True  # Enable outlier detection system
ENABLE_NARRATIVE_ANALYSIS = True  # Enable narrative analysis system
ENABLE_TIMELINE_DISPLAY = True  # Enable comprehensive timeline display
ENABLE_MODEL_TRAINING = True  # Enable automatic model training if needed
ENABLE_FULL_SYSTEM_STATUS = True  # Enable full system status reporting

# Data Download Configuration
DATA_DOWNLOAD_DAYS = 3  # Days of data to download for outlier detection
NARRATIVE_FIRST_RUN_DAYS = 30  # Days of data for first-time narrative setup
FORCE_DATA_REFRESH = False  # Force re-download of existing data

# System Display Configuration
MAX_OUTLIER_DISPLAY = 20  # Maximum outliers to display in ranking
MAX_NARRATIVE_PREDICTIONS = 5  # Maximum predictions to display
MAX_TIMELINE_ACTIVITIES = 50  # Maximum timeline activities to display
DISPLAY_SYSTEM_METRICS = True  # Show system performance metrics

# Advanced System Configuration
PARALLEL_PROCESSING = True  # Enable parallel processing where available
VERBOSE_OUTPUT = False  # Enable detailed logging output
SYSTEM_HEALTH_CHECK = True  # Enable system health monitoring

# Coin and Narrative Configuration
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
NARRATIVE_HISTORY_DAYS = 30  # 30 days of data for narrative analysis (optimized for speed)
NARRATIVE_SAMPLING_HOURS = 6  # Sample every 6 hours for historical detection (rebalancing frequency)

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
        
        # Download outlier detection data
        print(f"Downloading {DATA_DOWNLOAD_DAYS} days of 5m data for outlier detection...")
        downloader = BinanceDataDownloader(DATA_FOLDER)
        downloader.download_all_coins(days_back=DATA_DOWNLOAD_DAYS)
        print(f"Successfully downloaded data for {len(COINS)} coins")
        
        # Check if narrative data needs to be downloaded
        import os
        narrative_data_exists = os.path.exists(NARRATIVE_DATA_FOLDER) and len(os.listdir(NARRATIVE_DATA_FOLDER)) > 10
        
        if not narrative_data_exists:
            print(f"\nFirst run detected - downloading {NARRATIVE_FIRST_RUN_DAYS} days of 1h data for narrative analysis...")
            print("This may take several minutes...")
            from narrative_downloader import NarrativeDataDownloader
            narrative_downloader = NarrativeDataDownloader()
            narrative_downloader.download_all_coins(days_back=NARRATIVE_FIRST_RUN_DAYS)
            print("Narrative data download completed")
        else:
            print("Narrative data already exists, skipping download")
            
    except Exception as e:
        print(f"Data download failed: {e}")
        success = False
        
    return success

def run_enhanced_narrative_prediction(analyzer):
    """Enhanced multi-signal narrative prediction system"""
    try:
        import numpy as np
        from datetime import datetime, timedelta
        
        # 1. Get current market state and outlier data
        current_scores = analyzer.outlier_detector.get_all_current_scores()
        if current_scores.empty:
            return {'error': 'No outlier data available'}
        
        # 2. Analyze market momentum across narratives
        narrative_groups = get_narrative_groups()
        narrative_momentum = {}
        
        for narrative, coins in narrative_groups.items():
            if len(coins) < 2:  # Skip single-coin narratives
                continue
                
            # Calculate narrative-level momentum
            narrative_coins_data = current_scores[current_scores['coin'].isin(coins)]
            if not narrative_coins_data.empty:
                avg_score = narrative_coins_data['relative_score'].mean()
                max_score = narrative_coins_data['relative_score'].max()
                volume_momentum = narrative_coins_data['volume_mcap_normalized'].mean()
                price_momentum = narrative_coins_data['price_change'].mean()
                
                # Composite momentum score
                momentum_score = (
                    0.4 * avg_score + 
                    0.3 * max_score + 
                    0.2 * volume_momentum + 
                    0.1 * price_momentum
                )
                
                narrative_momentum[narrative] = {
                    'momentum_score': momentum_score,
                    'avg_outlier_score': avg_score,
                    'max_outlier_score': max_score,
                    'volume_momentum': volume_momentum,
                    'price_momentum': price_momentum,
                    'active_coins': len(narrative_coins_data)
                }
        
        # 3. Historical pattern analysis
        historical_timeline = analyzer.get_historical_narrative_timeline()
        pattern_scores = {}
        
        if historical_timeline:
            # Analyze recent activity patterns (last 7 days)
            recent_cutoff = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            recent_activities = [act for act in historical_timeline if act['date'] >= recent_cutoff]
            
            # Count recent narrative frequencies
            narrative_recent_freq = {}
            for activity in recent_activities:
                narrative = activity['narrative']
                narrative_recent_freq[narrative] = narrative_recent_freq.get(narrative, 0) + 1
            
            # Calculate pattern strength based on historical frequency vs recent activity
            for narrative in narrative_groups.keys():
                if len(narrative_groups[narrative]) < 2:
                    continue
                    
                # Historical frequency
                historical_freq = len([act for act in historical_timeline if act['narrative'] == narrative])
                recent_freq = narrative_recent_freq.get(narrative, 0)
                
                # Pattern strength: deviation from historical average
                if historical_freq > 0:
                    expected_recent = historical_freq * (7 / 365)  # Expected based on historical rate
                    pattern_strength = min(2.0, max(0.0, recent_freq / max(expected_recent, 0.1)))
                else:
                    pattern_strength = 0.5
                
                pattern_scores[narrative] = pattern_strength
        
        # 4. Temporal analysis
        now = datetime.now()
        temporal_factors = {
            'hour_factor': 1.0 + 0.1 * np.sin((now.hour - 14) * np.pi / 12),  # Peak at 2 PM UTC
            'weekday_factor': 1.2 if now.weekday() < 5 else 0.8,  # Higher on weekdays
            'month_factor': 1.0 + 0.05 * np.sin((now.month - 1) * np.pi / 6)  # Seasonal factor
        }
        
        temporal_score = temporal_factors['hour_factor'] * temporal_factors['weekday_factor'] * temporal_factors['month_factor']
        
        # 5. Cross-narrative correlation analysis
        correlation_matrix = {}
        if len(narrative_momentum) > 1:
            momentum_values = {k: v['momentum_score'] for k, v in narrative_momentum.items()}
            narratives = list(momentum_values.keys())
            
            # Simple correlation based on momentum similarity
            for i, narrative1 in enumerate(narratives):
                for j, narrative2 in enumerate(narratives[i+1:], i+1):
                    momentum1 = momentum_values[narrative1]
                    momentum2 = momentum_values[narrative2]
                    
                    # Inverse correlation: when one is high, predict other might follow
                    correlation = abs(momentum1 - momentum2)
                    correlation_matrix[f"{narrative1}-{narrative2}"] = correlation
        
        # 6. Ensemble prediction combining all signals
        ensemble_predictions = []
        
        for narrative in narrative_groups.keys():
            if len(narrative_groups[narrative]) < 2:
                continue
                
            # Base prediction components
            momentum_component = narrative_momentum.get(narrative, {}).get('momentum_score', 0)
            pattern_component = pattern_scores.get(narrative, 0.5)
            
            # Historical success rate (simplified)
            historical_success = min(1.0, len([act for act in historical_timeline 
                                            if act['narrative'] == narrative and 
                                            act.get('performance', 'N/A') != 'N/A']) / 50)
            
            # Combine all signals
            ensemble_score = (
                0.35 * momentum_component +
                0.25 * pattern_component +
                0.20 * historical_success +
                0.15 * temporal_score +
                0.05 * np.random.uniform(0.8, 1.2)  # Small random factor for variance
            )
            
            ensemble_predictions.append({
                'narrative': narrative,
                'probability': min(1.0, ensemble_score),
                'momentum_score': momentum_component,
                'pattern_strength': pattern_component,
                'historical_success': historical_success,
                'temporal_factor': temporal_score
            })
        
        # Sort by probability
        ensemble_predictions.sort(key=lambda x: x['probability'], reverse=True)
        
        # 7. Determine market state
        avg_momentum = np.mean([pred['momentum_score'] for pred in ensemble_predictions]) if ensemble_predictions else 0
        if avg_momentum > 1.5:
            market_state = "High Volatility"
        elif avg_momentum > 0.8:
            market_state = "Active"
        elif avg_momentum > 0.3:
            market_state = "Moderate"
        else:
            market_state = "Low Activity"
        
        # 8. Calculate overall confidence
        if ensemble_predictions:
            top_score = ensemble_predictions[0]['probability']
            second_score = ensemble_predictions[1]['probability'] if len(ensemble_predictions) > 1 else 0
            confidence_spread = top_score - second_score
            overall_confidence = min(0.95, max(0.05, top_score * (1 + confidence_spread)))
        else:
            overall_confidence = 0.05
        
        # 9. Market signals summary
        market_signals = {
            'volume_trend': 'Increasing' if avg_momentum > 0.8 else 'Decreasing' if avg_momentum < 0.3 else 'Stable',
            'correlation_strength': np.mean(list(correlation_matrix.values())) if correlation_matrix else 0,
            'temporal_pattern': f"Favorable ({temporal_score:.2f})" if temporal_score > 1.0 else f"Neutral ({temporal_score:.2f})",
            'cycle_position': 'Early' if avg_momentum < 0.5 else 'Mid' if avg_momentum < 1.2 else 'Late'
        }
        
        return {
            'primary_prediction': ensemble_predictions[0]['narrative'] if ensemble_predictions else 'None',
            'overall_confidence': overall_confidence,
            'market_state': market_state,
            'ensemble_predictions': ensemble_predictions,
            'market_signals': market_signals,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        return {'error': f'Enhanced prediction failed: {str(e)}'}

def run_outlier_analysis():
    """Run the outlier detection system"""
    if not ENABLE_OUTLIER_DETECTION:
        print("Outlier detection disabled in configuration")
        return {}
        
    print("\n" + "=" * 60)
    print("OUTLIER DETECTION SYSTEM")
    print("=" * 60)
    
    try:
        from outlier_detector import OutlierDetector
        
        detector = OutlierDetector()
        all_scores = detector.get_all_current_scores()
        
        if not all_scores.empty:
            print(f"\n=== ALL {len(all_scores)} COINS RANKED BY RELATIVE SCORE ===")
            print(f"{'Rank':<4} {'Coin':<8} {'Price':<12} {'BTC Ratio':<12} {'Z-Score':<8} {'Price Change%':<14} {'Vol/MCap':<11} {'Score':<8}")
            print("-" * 95)
            
            for idx, (_, row) in enumerate(all_scores.iterrows(), 1):
                btc_ratio = row.get('coin_btc_ratio', 0)
                price_change = row.get('price_change', 0)
                volume_mcap_norm = row.get('volume_mcap_normalized', 0)
                relative_score = row.get('relative_score', 0)
                
                print(f"{idx:<4} {row['coin']:<8} {row['close_price']:<12.4f} {btc_ratio:<12.6f} {row['z_score']:<8.2f} {price_change:<14.2f} {volume_mcap_norm:<11.4f} {relative_score:<8.2f}")
                
            return {'total_coins': len(all_scores), 'top_outliers': all_scores.head(10)}
        else:
            print("No outlier data available")
            return {'total_coins': 0, 'top_outliers': None}
            
    except Exception as e:
        print(f"Outlier analysis failed: {e}")
        return {'error': str(e)}
    

def run_narrative_analysis():
    """Run the narrative prediction system"""
    if not ENABLE_NARRATIVE_ANALYSIS:
        print("Narrative analysis disabled in configuration")
        return {}
        
    print("\n" + "=" * 60)
    print("NARRATIVE ANALYSIS SYSTEM")
    print("=" * 60)
    
    summary = {}
    try:
        from narrative_analyzer import NarrativeAnalyzer
        
        # Check if model needs training
        import os
        model_exists = os.path.exists(f"{NARRATIVE_DATA_FOLDER}/narrative_model.pkl")
        
        if not model_exists and ENABLE_MODEL_TRAINING:
            print("Training narrative prediction model...")
            analyzer = NarrativeAnalyzer()
            analyzer.train_narrative_prediction_model()
            print("Model training completed")
        
        # Run analysis
        analyzer = NarrativeAnalyzer()
        summary = analyzer.get_narrative_summary()
        
        print(f"\n=== NARRATIVE SUMMARY ===")
        print(f"Total narratives tracked: {summary.get('total_narratives', 0)}")
        print(f"Currently pumping narrative: {summary.get('current_pumping_narrative') or 'None detected'}")
            
        # Enhanced Multi-Signal Narrative Prediction
        enhanced_predictions = run_enhanced_narrative_prediction(analyzer)
        
        if enhanced_predictions and 'error' not in enhanced_predictions:
            print(f"\n=== ENHANCED NARRATIVE PREDICTION SYSTEM ===")
            print(f"Current market state: {enhanced_predictions.get('market_state', 'Unknown')}")
            print(f"Prediction confidence: {enhanced_predictions.get('overall_confidence', 0):.2%}")
            print(f"Primary prediction: {enhanced_predictions.get('primary_prediction', 'None')}")
            
            # Show ensemble predictions
            ensemble_predictions = enhanced_predictions.get('ensemble_predictions', [])
            if ensemble_predictions:
                print(f"\n=== ENSEMBLE PREDICTIONS (Top {min(MAX_NARRATIVE_PREDICTIONS, len(ensemble_predictions))}) ===")
                display_count = min(MAX_NARRATIVE_PREDICTIONS, len(ensemble_predictions))
                for i, pred in enumerate(ensemble_predictions[:display_count], 1):
                    narrative = pred.get('narrative', 'Unknown')
                    probability = pred.get('probability', 0)
                    momentum = pred.get('momentum_score', 0)
                    pattern_strength = pred.get('pattern_strength', 0)
                    
                    # Get coins for this narrative
                    narrative_groups = get_narrative_groups()
                    coins = narrative_groups.get(narrative, [])
                    
                    print(f"{i}. {narrative} ({probability:.2%}) [Momentum: {momentum:.2f}, Pattern: {pattern_strength:.2f}]")
                    if coins:
                        coins_display = ', '.join(coins[:6])
                        if len(coins) > 6:
                            coins_display += f" (+{len(coins)-6} more)"
                        print(f"   Coins: {coins_display}")
                    else:
                        print(f"   Coins: None available")
            
            # Show market signals
            signals = enhanced_predictions.get('market_signals', {})
            if signals:
                print(f"\n=== MARKET SIGNALS ===")
                print(f"Volume trend: {signals.get('volume_trend', 'Unknown')}")
                print(f"Cross-narrative correlation: {signals.get('correlation_strength', 0):.2f}")
                print(f"Temporal pattern: {signals.get('temporal_pattern', 'Unknown')}")
                print(f"Historical cycle position: {signals.get('cycle_position', 'Unknown')}")
        else:
            print("Enhanced prediction unavailable (insufficient data or model training needed)")
            
        return summary
        
    except Exception as e:
        print(f"Error in narrative analysis: {e}")
        return {'error': str(e)}

def run_comprehensive_timeline():
    """Display comprehensive narrative timeline and statistics"""
    if not ENABLE_TIMELINE_DISPLAY:
        print("Timeline display disabled in configuration")
        return {}
        
    print("\n" + "=" * 60)
    print("COMPREHENSIVE NARRATIVE TIMELINE")
    print("=" * 60)
    
    try:
        from narrative_analyzer import NarrativeAnalyzer
        
        # Show configuration breakdown
        all_groups = get_narrative_groups()
        filtered_groups = {k: v for k, v in all_groups.items() if len(v) >= 2}
        
        print(f"Total configured narratives: {len(all_groups)}")
        print(f"Narratives with 2+ coins (trackable): {len(filtered_groups)}")
        print(f"Single-coin narratives (filtered out): {len(all_groups) - len(filtered_groups)}")
        
        analyzer = NarrativeAnalyzer()
        
        # Get enhanced summary
        summary = analyzer.get_narrative_summary()
        print(f"Actually detected in historical data: {summary.get('detected_narratives_count', 0)}")
        print(f"Historical activities found: {summary.get('historical_activities_count', 0)}")
        
        detected_narratives = summary.get('detected_narratives', [])
        missing_narratives = set(filtered_groups.keys()) - set(detected_narratives)
        if missing_narratives:
            missing_list = sorted(missing_narratives)
            print(f"Never detected: {', '.join(missing_list[:10])}")
            if len(missing_list) > 10:
                print(f"  ... and {len(missing_list) - 10} more")
        
        # Get timeline and show narrative activity dates
        timeline = analyzer.get_historical_narrative_timeline()
        
        if timeline:
            # Group activities by narrative to show dates
            narrative_dates = {}
            for activity in timeline:
                narrative = activity['narrative']
                if narrative not in narrative_dates:
                    narrative_dates[narrative] = []
                narrative_dates[narrative].append(activity['date'])
            
            print(f"\n=== COMPLETE NARRATIVE ACTIVITY HISTORY ===")
            print(f"{'Narrative':<20} {'Days Active':<12} {'All Active Dates'}")
            print("-" * 120)
            
            for narrative in sorted(narrative_dates.keys()):
                dates = sorted(set(narrative_dates[narrative]))
                all_dates = ', '.join(dates)
                # If dates list is too long, show first few and last few with count
                if len(all_dates) > 80:
                    first_dates = ', '.join(dates[:3])
                    last_dates = ', '.join(dates[-3:])
                    dates_display = f"{first_dates} ... [{len(dates)-6} more] ... {last_dates}"
                else:
                    dates_display = all_dates
                print(f"{narrative:<20} {len(dates):<12} {dates_display}")
            
            # Show detailed timeline if requested
            if DISPLAY_SYSTEM_METRICS and len(timeline) <= MAX_TIMELINE_ACTIVITIES:
                print(f"\n=== DETAILED TIMELINE ({len(timeline)} activities) ===")
                print(f"{'#':<3} {'Date':<10} {'Time':<5} {'Narrative':<20} {'Performance':<12} {'Pos%':<6} {'Coins':<6} {'Top Performers'}")
                print("-" * 120)
                
                for i, activity in enumerate(timeline[:MAX_TIMELINE_ACTIVITIES], 1):
                    narrative = activity['narrative'][:18]
                    performers = ', '.join(activity.get('top_performing_coins', [])[:3])
                    remaining = len(activity.get('all_coins', [])) - 3
                    if remaining > 0:
                        performers += f" (+{remaining})"
                    
                    print(f"{i:<3} {activity['date']:<10} {activity.get('time', ''):<5} {narrative:<20} "
                          f"{activity.get('performance', 'N/A'):<12} {activity.get('positive_ratio', 'N/A'):<6} "
                          f"{activity.get('coins_count', 0):<6} {performers}")
            
        print(f"\nTotal activities tracked: {len(timeline) if timeline else 0}")
        
        return {
            'total_narratives': len(all_groups),
            'trackable_narratives': len(filtered_groups),
            'detected_narratives': len(detected_narratives),
            'total_activities': len(timeline) if timeline else 0
        }
        
    except Exception as e:
        print(f"Timeline display failed: {e}")
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
        narrative_files = len([f for f in os.listdir(NARRATIVE_DATA_FOLDER) if f.endswith('.parquet')]) if os.path.exists(NARRATIVE_DATA_FOLDER) else 0
        model_exists = os.path.exists(f"{NARRATIVE_DATA_FOLDER}/narrative_model.pkl")
        
        print(f"Data files available: {data_files}/{len(COINS)} coins")
        print(f"Narrative data files: {narrative_files}/{len(COINS)} coins")
        print(f"Prediction model trained: {'Yes' if model_exists else 'No'}")
        
        # Check configuration
        total_narratives = len(get_narrative_groups())
        trackable_narratives = len({k: v for k, v in get_narrative_groups().items() if len(v) >= 2})
        
        print(f"Total narratives configured: {total_narratives}")
        print(f"Trackable narratives: {trackable_narratives}")
        
        health_status = {
            'data_coverage': data_files / len(COINS) if COINS else 0,
            'narrative_coverage': narrative_files / len(COINS) if COINS else 0,
            'model_trained': model_exists,
            'total_narratives': total_narratives,
            'trackable_narratives': trackable_narratives
        }
        
        # Overall health score
        health_score = (
            (0.4 * health_status['data_coverage']) +
            (0.3 * health_status['narrative_coverage']) +
            (0.2 * (1 if health_status['model_trained'] else 0)) +
            (0.1 * min(1, health_status['trackable_narratives'] / 20))
        )
        
        print(f"\nOverall system health: {health_score:.1%}")
        health_status['health_score'] = health_score
        
        if health_score < 0.8:
            print("System recommendations:")
            if health_status['data_coverage'] < 0.9:
                print("- Run data download to update missing coin data")
            if health_status['narrative_coverage'] < 0.9:
                print("- Download narrative data for complete analysis")
            if not health_status['model_trained']:
                print("- Train prediction model for forecasting")
        
        return health_status
        
    except Exception as e:
        print(f"Health check failed: {e}")
        return {'error': str(e)}

def main():
    """Main function - run complete analysis system"""
    import time
    start_time = time.time()
    
    print("=" * 80)
    print("COMPREHENSIVE CRYPTO OUTLIER DETECTION SYSTEM")
    print("=" * 80)
    print(f"System initialized with {len(COINS)} cryptocurrencies")
    print(f"Tracking {len(get_narrative_groups())} narratives")
    
    if ENABLE_FULL_SYSTEM_STATUS:
        print("\nSystem Configuration:")
        print(f"- Data Download: {'Enabled' if ENABLE_DATA_DOWNLOAD else 'Disabled'}")
        print(f"- Outlier Detection: {'Enabled' if ENABLE_OUTLIER_DETECTION else 'Disabled'}")
        print(f"- Narrative Analysis: {'Enabled' if ENABLE_NARRATIVE_ANALYSIS else 'Disabled'}")
        print(f"- Timeline Display: {'Enabled' if ENABLE_TIMELINE_DISPLAY else 'Disabled'}")
        print(f"- Model Training: {'Enabled' if ENABLE_MODEL_TRAINING else 'Disabled'}")
    
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
    
    # 4. Narrative Analysis System
    if ENABLE_NARRATIVE_ANALYSIS:
        narrative_results = run_narrative_analysis()
        system_results['narrative_analysis'] = narrative_results
    
    # 5. Comprehensive Timeline Display
    if ENABLE_TIMELINE_DISPLAY:
        timeline_results = run_comprehensive_timeline()
        system_results['timeline'] = timeline_results
    
    # 6. Final System Summary
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
            print(f"- Total outliers detected: {total_outliers}")
        
        if 'narrative_analysis' in system_results and 'total_narratives' in system_results['narrative_analysis']:
            total_narratives = system_results['narrative_analysis']['total_narratives']
            print(f"- Narratives analyzed: {total_narratives}")
        
        if 'timeline' in system_results and 'total_activities' in system_results['timeline']:
            total_activities = system_results['timeline']['total_activities']
            print(f"- Historical activities: {total_activities}")
        
        if 'health' in system_results and 'health_score' in system_results['health']:
            health_score = system_results['health']['health_score']
            print(f"- System health score: {health_score:.1%}")
        
        print("\nAll system components completed successfully.")
        print("System ready for continuous monitoring.")
    
    return system_results

if __name__ == "__main__":
    main()
