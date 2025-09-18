import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

from main import (
    COINS_DATA, NARRATIVE_DATA_FOLDER, get_narrative_groups, get_coins_by_narrative
)

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Narrative Analysis Configuration
PUMP_THRESHOLD = 0.15  # 15% gain in narrative group
PUMP_TIMEFRAME_HOURS = 24  # Hours to look for pump
MIN_COINS_IN_NARRATIVE = 2  # Minimum coins required for narrative analysis
SEQUENCE_LOOKBACK_DAYS = 7  # Days to look back for narrative sequences

class NarrativeAnalyzer:
    def __init__(self):
        self.data_folder = NARRATIVE_DATA_FOLDER
        self.narrative_groups = get_narrative_groups()
        # Filter narratives with minimum coin count
        self.narrative_groups = {k: v for k, v in self.narrative_groups.items() 
                               if len(v) >= MIN_COINS_IN_NARRATIVE}
        self.pump_history = []
        self.model = None
        self.label_encoder = LabelEncoder()
        
    def load_coin_data(self, coin: str) -> pd.DataFrame:
        """Load 1h OHLCV data for a specific coin"""
        filename = f"{self.data_folder}/{coin}_1h_ohlcv.parquet"
        
        if not os.path.exists(filename):
            return pd.DataFrame()
        
        try:
            df = pd.read_parquet(filename)
            df = df.sort_values('open_time').reset_index(drop=True)
            return df
        except Exception as e:
            return pd.DataFrame()
    
    def calculate_narrative_performance(self, narrative: str, start_time: datetime, end_time: datetime) -> Dict:
        """Calculate performance of all coins in a narrative during a time period"""
        coins = self.narrative_groups.get(narrative, [])
        if not coins:
            return {}
        
        performances = {}
        valid_coins = 0
        
        for coin in coins:
            df = self.load_coin_data(coin)
            if df.empty:
                continue
                
            # Filter data for time period
            mask = (df['open_time'] >= start_time) & (df['open_time'] <= end_time)
            period_data = df[mask]
            
            if len(period_data) < 2:
                continue
                
            start_price = period_data.iloc[0]['close']
            end_price = period_data.iloc[-1]['close']
            performance = (end_price - start_price) / start_price
            
            performances[coin] = performance
            valid_coins += 1
        
        if valid_coins < MIN_COINS_IN_NARRATIVE:
            return {}
        
        # Calculate narrative aggregate performance
        avg_performance = np.mean(list(performances.values()))
        median_performance = np.median(list(performances.values()))
        positive_coins = sum(1 for p in performances.values() if p > 0)
        positive_ratio = positive_coins / len(performances)
        
        return {
            'narrative': narrative,
            'start_time': start_time,
            'end_time': end_time,
            'avg_performance': avg_performance,
            'median_performance': median_performance,
            'positive_ratio': positive_ratio,
            'coin_performances': performances,
            'coin_count': len(performances)
        }
    
    def detect_narrative_pumps(self, start_date: datetime = None, end_date: datetime = None) -> List[Dict]:
        """Detect historical narrative pumps"""
        if not start_date:
            start_date = datetime.now() - timedelta(days=365)
        if not end_date:
            end_date = datetime.now()
        
        pumps = []
        current_time = start_date
        
        logger.info(f"Detecting narrative pumps from {start_date} to {end_date}")
        
        while current_time < end_date:
            period_end = current_time + timedelta(hours=PUMP_TIMEFRAME_HOURS)
            
            for narrative in self.narrative_groups.keys():
                performance = self.calculate_narrative_performance(narrative, current_time, period_end)
                
                if (performance and 
                    performance['avg_performance'] >= PUMP_THRESHOLD and 
                    performance['positive_ratio'] >= 0.6):  # At least 60% of coins positive
                    
                    pump_data = {
                        'timestamp': current_time,
                        'narrative': narrative,
                        'performance': performance['avg_performance'],
                        'positive_ratio': performance['positive_ratio'],
                        'coin_count': performance['coin_count'],
                        'coins': list(performance['coin_performances'].keys())
                    }
                    pumps.append(pump_data)
                    logger.info(f"Detected {narrative} pump at {current_time}: {performance['avg_performance']:.2%}")
            
            current_time += timedelta(hours=1)  # Move forward by 1 hour
        
        self.pump_history = pumps
        return pumps
    
    def create_narrative_sequences(self, pumps: List[Dict]) -> List[Dict]:
        """Create sequences of narrative pumps for ML training"""
        sequences = []
        
        # Sort pumps by timestamp
        sorted_pumps = sorted(pumps, key=lambda x: x['timestamp'])
        
        for i in range(len(sorted_pumps) - 1):
            current_pump = sorted_pumps[i]
            
            # Find next pump within reasonable timeframe (up to 7 days)
            for j in range(i + 1, len(sorted_pumps)):
                next_pump = sorted_pumps[j]
                time_diff = (next_pump['timestamp'] - current_pump['timestamp']).total_seconds() / 3600  # hours
                
                if time_diff <= SEQUENCE_LOOKBACK_DAYS * 24:  # Within lookback period
                    sequence = {
                        'current_narrative': current_pump['narrative'],
                        'next_narrative': next_pump['narrative'],
                        'time_gap_hours': time_diff,
                        'current_performance': current_pump['performance'],
                        'next_performance': next_pump['performance'],
                        'timestamp': current_pump['timestamp']
                    }
                    sequences.append(sequence)
                    break  # Only take the first next pump
        
        return sequences
    
    def prepare_ml_features(self, sequences: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for ML model"""
        if not sequences:
            return np.array([]), np.array([])
        
        # Create feature matrix
        narratives = list(self.narrative_groups.keys())
        narrative_to_idx = {narrative: idx for idx, narrative in enumerate(narratives)}
        
        X = []
        y = []
        
        for seq in sequences:
            # Features: current narrative (one-hot), time features, performance features
            features = [0] * len(narratives)  # One-hot for current narrative
            if seq['current_narrative'] in narrative_to_idx:
                features[narrative_to_idx[seq['current_narrative']]] = 1
            
            # Add time-based features
            timestamp = seq['timestamp']
            features.extend([
                timestamp.hour,  # Hour of day
                timestamp.weekday(),  # Day of week
                timestamp.day,  # Day of month
                seq['time_gap_hours'],  # Time to next pump
                seq['current_performance']  # Current pump strength
            ])
            
            X.append(features)
            
            # Target: next narrative
            if seq['next_narrative'] in narrative_to_idx:
                y.append(seq['next_narrative'])
        
        return np.array(X), np.array(y)
    
    def train_narrative_prediction_model(self, sequences: List[Dict] = None):
        """Train ML model to predict next narrative"""
        if sequences is None:
            pumps = self.detect_narrative_pumps()
            sequences = self.create_narrative_sequences(pumps)
        
        if len(sequences) < 10:  # Need minimum sequences for training
            logger.warning("Not enough narrative sequences for training")
            return False
        
        X, y = self.prepare_ml_features(sequences)
        
        if len(X) == 0:
            logger.warning("No valid features extracted for training")
            return False
        
        # Encode target labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
        
        # Train Random Forest model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Evaluate
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        logger.info(f"Model trained - Train Score: {train_score:.3f}, Test Score: {test_score:.3f}")
        
        # Save model
        self.save_model()
        
        return True
    
    def predict_next_narrative(self, current_narrative: str = None) -> Dict:
        """Predict the next narrative to pump"""
        if self.model is None:
            self.load_model()
            if self.model is None:
                return {"error": "No trained model available"}
        
        # If no current narrative specified, detect current pumping narrative
        if current_narrative is None:
            current_narrative = self.detect_current_pumping_narrative()
            if not current_narrative:
                return {"error": "No current narrative pump detected"}
        
        # Prepare features for prediction
        narratives = list(self.narrative_groups.keys())
        narrative_to_idx = {narrative: idx for idx, narrative in enumerate(narratives)}
        
        if current_narrative not in narrative_to_idx:
            return {"error": f"Unknown narrative: {current_narrative}"}
        
        # Create feature vector
        features = [0] * len(narratives)
        features[narrative_to_idx[current_narrative]] = 1
        
        # Add current time features
        now = datetime.now()
        features.extend([
            now.hour,
            now.weekday(),
            now.day,
            12,  # Average time gap (placeholder)
            0.2  # Average performance (placeholder)
        ])
        
        # Make prediction
        prediction_proba = self.model.predict_proba([features])[0]
        predicted_idx = np.argmax(prediction_proba)
        predicted_narrative = self.label_encoder.inverse_transform([predicted_idx])[0]
        confidence = prediction_proba[predicted_idx]
        
        # Get top 3 predictions
        top_indices = np.argsort(prediction_proba)[-3:][::-1]
        top_predictions = []
        for idx in top_indices:
            narrative = self.label_encoder.inverse_transform([idx])[0]
            prob = prediction_proba[idx]
            coins = self.narrative_groups.get(narrative, [])
            top_predictions.append({
                'narrative': narrative,
                'probability': prob,
                'coins': coins
            })
        
        return {
            'current_narrative': current_narrative,
            'predicted_narrative': predicted_narrative,
            'confidence': confidence,
            'top_predictions': top_predictions
        }
    
    def detect_current_pumping_narrative(self) -> Optional[str]:
        """Detect which narrative is currently pumping"""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=PUMP_TIMEFRAME_HOURS)
        
        best_narrative = None
        best_performance = 0
        
        for narrative in self.narrative_groups.keys():
            performance = self.calculate_narrative_performance(narrative, start_time, end_time)
            
            if (performance and 
                performance['avg_performance'] > best_performance and
                performance['avg_performance'] >= PUMP_THRESHOLD and
                performance['positive_ratio'] >= 0.6):
                
                best_performance = performance['avg_performance']
                best_narrative = narrative
        
        return best_narrative
    
    def save_model(self):
        """Save trained model and label encoder"""
        model_path = f"{self.data_folder}/narrative_model.pkl"
        encoder_path = f"{self.data_folder}/label_encoder.pkl"
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        logger.info("Model and encoder saved")
    
    def load_model(self):
        """Load trained model and label encoder"""
        model_path = f"{self.data_folder}/narrative_model.pkl"
        encoder_path = f"{self.data_folder}/label_encoder.pkl"
        
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            with open(encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            
            logger.info("Model and encoder loaded")
        except FileNotFoundError:
            logger.warning("No saved model found")
    
    def get_narrative_summary(self) -> Dict:
        """Get summary of narrative analysis"""
        current_pumping = self.detect_current_pumping_narrative()
        prediction = self.predict_next_narrative(current_pumping)
        
        return {
            'total_narratives': len(self.narrative_groups),
            'current_pumping_narrative': current_pumping,
            'next_predicted_narrative': prediction.get('predicted_narrative'),
            'prediction_confidence': prediction.get('confidence'),
            'top_predictions': prediction.get('top_predictions', [])
        }


def main():
    """Main function to run narrative analysis"""
    analyzer = NarrativeAnalyzer()
    
    print("=== NARRATIVE ANALYSIS ===")
    
    # Detect historical pumps
    print("Detecting historical narrative pumps...")
    pumps = analyzer.detect_narrative_pumps()
    print(f"Found {len(pumps)} historical narrative pumps")
    
    # Train model
    print("Training narrative prediction model...")
    if analyzer.train_narrative_prediction_model():
        print("Model training completed successfully")
        
        # Get current analysis
        summary = analyzer.get_narrative_summary()
        
        print(f"\n=== CURRENT NARRATIVE ANALYSIS ===")
        print(f"Total narratives tracked: {summary['total_narratives']}")
        print(f"Currently pumping narrative: {summary['current_pumping_narrative']}")
        print(f"Predicted next narrative: {summary['next_predicted_narrative']}")
        print(f"Confidence: {summary['prediction_confidence']:.2%}")
        
        print(f"\n=== TOP PREDICTIONS ===")
        for i, pred in enumerate(summary['top_predictions'][:3], 1):
            print(f"{i}. {pred['narrative']} ({pred['probability']:.2%}) - {len(pred['coins'])} coins")
            print(f"   Coins: {pred['coins'][:5]}{'...' if len(pred['coins']) > 5 else ''}")
    else:
        print("Model training failed - not enough data")


if __name__ == "__main__":
    main()