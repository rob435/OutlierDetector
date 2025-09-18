import pandas as pd
import numpy as np
from main import EWMA_ALPHA, DATA_FOLDER
from outlier_detector import OutlierDetector

def test_ewma_effect():
    """Test the actual effect of EWMA smoothing on outlier scores"""
    detector = OutlierDetector()
    
    # Test with a few sample coins
    test_coins = ['BTC', 'ETH', 'SOL', 'DOGE']
    
    print(f"=== EWMA ALPHA TEST (alpha = {EWMA_ALPHA}) ===")
    print("Testing effect of EWMA smoothing on outlier detection...")
    
    for coin in test_coins:
        df = detector.load_coin_data(coin)
        if df.empty:
            print(f"No data for {coin}")
            continue
            
        # Calculate raw Z-score
        z_score_raw = detector.calculate_rolling_z_score(df['close'], detector.z_score_window)
        
        # Apply EWMA smoothing
        z_score_ewma = detector.apply_ewma(z_score_raw, EWMA_ALPHA)
        
        # Calculate differences
        recent_data = df.tail(100)  # Last 100 points
        z_raw_recent = z_score_raw.tail(100)
        z_ewma_recent = z_score_ewma.tail(100)
        
        # Calculate metrics
        raw_std = z_raw_recent.std()
        ewma_std = z_ewma_recent.std()
        max_diff = abs(z_raw_recent - z_ewma_recent).max()
        avg_diff = abs(z_raw_recent - z_ewma_recent).mean()
        
        # Current values
        current_raw = z_raw_recent.iloc[-1]
        current_ewma = z_ewma_recent.iloc[-1]
        current_diff = abs(current_raw - current_ewma)
        
        print(f"\n{coin}:")
        print(f"  Current Z-score (raw): {current_raw:.3f}")
        print(f"  Current Z-score (EWMA): {current_ewma:.3f}")
        print(f"  Current difference: {current_diff:.3f}")
        print(f"  Average difference: {avg_diff:.3f}")
        print(f"  Max difference: {max_diff:.3f}")
        print(f"  Std reduction: {((raw_std - ewma_std) / raw_std * 100):.1f}%")
        
        # Check if EWMA is actually smoothing
        if avg_diff < 0.01:
            print(f"  WARNING: MINIMAL EFFECT - EWMA barely changes values")
        elif avg_diff < 0.1:
            print(f"  INFO: SMALL EFFECT - EWMA provides minor smoothing")
        else:
            print(f"  SUCCESS: SIGNIFICANT EFFECT - EWMA provides meaningful smoothing")

def test_different_alphas():
    """Test different EWMA alpha values to see their effects"""
    detector = OutlierDetector()
    
    # Test with BTC data
    df = detector.load_coin_data('BTC')
    if df.empty:
        print("No BTC data for alpha testing")
        return
    
    z_score_raw = detector.calculate_rolling_z_score(df['close'], detector.z_score_window)
    recent_raw = z_score_raw.tail(100)
    
    alphas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
    
    print(f"\n=== ALPHA COMPARISON TEST ===")
    print(f"Testing different EWMA alpha values on BTC...")
    print(f"{'Alpha':<8} {'Current':<10} {'Avg Diff':<10} {'Max Diff':<10} {'Effect':<15}")
    print("-" * 60)
    
    for alpha in alphas:
        z_ewma = detector.apply_ewma(z_score_raw, alpha)
        recent_ewma = z_ewma.tail(100)
        
        current_diff = abs(recent_raw.iloc[-1] - recent_ewma.iloc[-1])
        avg_diff = abs(recent_raw - recent_ewma).mean()
        max_diff = abs(recent_raw - recent_ewma).max()
        
        if avg_diff < 0.01:
            effect = "Minimal"
        elif avg_diff < 0.1:
            effect = "Small"
        elif avg_diff < 0.5:
            effect = "Moderate"
        else:
            effect = "Strong"
            
        print(f"{alpha:<8} {recent_ewma.iloc[-1]:<10.3f} {avg_diff:<10.3f} {max_diff:<10.3f} {effect:<15}")

def recommend_alpha():
    """Recommend optimal alpha value"""
    print(f"\n=== ALPHA RECOMMENDATIONS ===")
    print(f"Current alpha: {EWMA_ALPHA}")
    print(f"")
    print(f"Alpha effects:")
    print(f"  0.001-0.005: Very smooth, slow to respond, minimal noise reduction")
    print(f"  0.01-0.05:   Moderate smoothing, good balance")
    print(f"  0.1-0.2:     Strong smoothing, fast response, good for real-time")
    print(f"  0.5+:        Very responsive, minimal smoothing")
    print(f"")
    print(f"For outlier detection:")
    print(f"  - If you want to catch quick spikes: Use alpha = 0.1-0.2")
    print(f"  - If you want to reduce noise: Use alpha = 0.01-0.05")
    print(f"  - Current alpha = {EWMA_ALPHA} is very conservative (minimal effect)")

if __name__ == "__main__":
    test_ewma_effect()
    test_different_alphas()
    recommend_alpha()