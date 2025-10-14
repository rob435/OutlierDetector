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
from outlier_detector import OutlierDetector

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Narrative Analysis Configuration
TOP_OUTLIER_THRESHOLD = 20  # Top N outliers to consider for narrative detection
MIN_NARRATIVE_COINS_IN_TOP = 2  # Minimum coins from same narrative in top outliers
NARRATIVE_STRENGTH_THRESHOLD = 0.3  # Minimum ratio of narrative coins in top outliers
MIN_COINS_IN_NARRATIVE = 2  # Minimum coins required for narrative analysis
SEQUENCE_LOOKBACK_DAYS = 365  # Days to look back for narrative sequences (full market cycle)
MIN_TRAINING_SEQUENCES = 5  # Minimum sequences for model training

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
        self.outlier_detector = OutlierDetector()
        
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
    
    def detect_current_narrative_activity(self) -> List[Dict]:
        """Detect current narrative activity based on outlier scores"""
        # Get current outlier scores for all coins
        current_scores = self.outlier_detector.get_all_current_scores()
        
        if current_scores.empty:
            return []
        
        # Get top outliers
        top_outliers = current_scores.head(TOP_OUTLIER_THRESHOLD)
        top_coins = set(top_outliers['coin'].tolist())
        
        narrative_activities = []
        
        # Check each narrative for representation in top outliers
        for narrative, coins in self.narrative_groups.items():
            narrative_coins_in_top = [coin for coin in coins if coin in top_coins]
            
            if len(narrative_coins_in_top) >= MIN_NARRATIVE_COINS_IN_TOP:
                # Calculate narrative strength
                narrative_strength = len(narrative_coins_in_top) / len(coins)
                
                if narrative_strength >= NARRATIVE_STRENGTH_THRESHOLD:
                    # Get scores for narrative coins
                    narrative_scores = top_outliers[top_outliers['coin'].isin(narrative_coins_in_top)]
                    avg_score = narrative_scores['relative_score'].mean()
                    max_score = narrative_scores['relative_score'].max()
                    
                    activity = {
                        'narrative': narrative,
                        'coins_in_top': narrative_coins_in_top,
                        'coins_count': len(narrative_coins_in_top),
                        'narrative_strength': narrative_strength,
                        'avg_outlier_score': avg_score,
                        'max_outlier_score': max_score,
                        'timestamp': datetime.now()
                    }
                    narrative_activities.append(activity)
        
        # Sort by narrative strength and average score
        narrative_activities.sort(key=lambda x: (x['narrative_strength'], x['avg_outlier_score']), reverse=True)
        
        return narrative_activities
    
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
    
    def detect_historical_narrative_activities(self, start_date: datetime = None, end_date: datetime = None) -> List[Dict]:
        """Detect historical narrative activities based on outlier detection (same logic as current detection)"""
        from main import NARRATIVE_HISTORY_DAYS, NARRATIVE_SAMPLING_HOURS
        
        if not start_date:
            start_date = datetime.now() - timedelta(days=NARRATIVE_HISTORY_DAYS)  # Use full configured history
        if not end_date:
            end_date = datetime.now()
        
        activities = []
        
        logger.info(f"Detecting narrative activities from {start_date} to {end_date} using outlier-based detection")
        
        # Sample historical data at configured intervals for rebalancing
        current_time = start_date
        sample_interval = timedelta(hours=NARRATIVE_SAMPLING_HOURS)  # Use configured sampling frequency
        
        while current_time < end_date:
            # Simulate getting historical outlier scores for this time period
            historical_activities = self.detect_historical_outlier_activities(current_time)
            
            for activity in historical_activities:
                activity['timestamp'] = current_time
                activities.append(activity)
                logger.info(f"Historical activity: {activity['narrative']} at {current_time.strftime('%Y-%m-%d')} - strength {activity['narrative_strength']:.1%}")
            
            current_time += sample_interval
        
        logger.info(f"Found {len(activities)} historical narrative activities")
        self.pump_history = activities  # Keep for compatibility
        return activities
    
    def detect_historical_outlier_activities(self, target_date: datetime) -> List[Dict]:
        """Detect narrative activities for a specific historical date using outlier logic"""
        activities = []
        
        # For each narrative, check if multiple coins would have been outliers on this date
        for narrative, coins in self.narrative_groups.items():
            # Simulate checking if coins were outliers on this date
            # This is a simplified version - you could enhance it with actual historical outlier calculation
            coins_in_narrative = [coin for coin in coins if self.coin_had_outlier_activity(coin, target_date)]
            
            if len(coins_in_narrative) >= MIN_NARRATIVE_COINS_IN_TOP:
                narrative_strength = len(coins_in_narrative) / len(coins)
                
                if narrative_strength >= NARRATIVE_STRENGTH_THRESHOLD:
                    activity = {
                        'narrative': narrative,
                        'coins_in_top': coins_in_narrative,
                        'coins_count': len(coins_in_narrative),
                        'narrative_strength': narrative_strength,
                        'avg_outlier_score': 1.0,  # Simplified score
                        'max_outlier_score': 1.0,  # Simplified score
                        'coins': coins_in_narrative,  # For compatibility
                        'performance': narrative_strength,  # Use strength as performance proxy
                        'positive_ratio': 1.0  # Simplified ratio
                    }
                    activities.append(activity)
        
        return activities
    
    def coin_had_outlier_activity(self, coin: str, target_date: datetime) -> bool:
        """Check if a coin had outlier activity on a specific date (simplified version)"""
        # Load historical data for the coin
        df = self.load_coin_data(coin)
        if df.empty:
            return False
        
        # Find data around the target date (within 24 hours)
        start_time = target_date - timedelta(hours=12)
        end_time = target_date + timedelta(hours=12)
        
        mask = (df['open_time'] >= start_time) & (df['open_time'] <= end_time)
        period_data = df[mask]
        
        if len(period_data) < 2:
            return False
        
        # Simple outlier check: if price moved significantly
        start_price = period_data.iloc[0]['close']
        max_price = period_data['high'].max()
        price_increase = (max_price - start_price) / start_price
        
        # Consider it outlier activity if price increased >3% in the period
        return price_increase > 0.03
    
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
            activities = self.detect_historical_narrative_activities()
            sequences = self.create_narrative_sequences(activities)
        
        if len(sequences) < MIN_TRAINING_SEQUENCES:  # Need minimum sequences for training
            logger.warning(f"Not enough narrative sequences for training. Found {len(sequences)}, need {MIN_TRAINING_SEQUENCES}")
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
        """Detect which narrative is currently showing strongest activity based on outlier scores"""
        activities = self.detect_current_narrative_activity()
        
        if not activities:
            return None
        
        # Return the narrative with highest strength and outlier score
        best_activity = activities[0]  # Already sorted by strength and score
        return best_activity['narrative']
    
    
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
    
    def get_historical_narrative_timeline(self) -> List[Dict]:
        """Get detailed timeline of historical narrative activities"""
        activities = self.detect_historical_narrative_activities()
        
        # Sort by timestamp and add more details
        timeline = []
        for activity in sorted(activities, key=lambda x: x['timestamp']):
            timeline.append({
                'date': activity['timestamp'].strftime('%Y-%m-%d'),
                'time': activity['timestamp'].strftime('%H:%M'),
                'narrative': activity['narrative'],
                'performance': f"{activity['performance']:.2%}",
                'positive_ratio': f"{activity['positive_ratio']:.1%}",
                'coins_count': activity['coins_count'],
                'top_performing_coins': activity['coins'][:5],  # Show top 5 coins
                'all_coins': activity['coins']
            })
        
        return timeline
    
    def get_narrative_summary(self) -> Dict:
        """Get summary of narrative analysis"""
        try:
            current_pumping = self.detect_current_pumping_narrative()
            current_activities = self.detect_current_narrative_activity()
            
            # Get historical activities to count actually detected narratives
            activities = self.detect_historical_narrative_activities()
            detected_narratives = set(activity['narrative'] for activity in activities)
            
            # Try to get prediction (may fail if no model)
            prediction = {}
            try:
                prediction = self.predict_next_narrative(current_pumping)
                if 'error' in prediction:
                    prediction = {}
            except Exception as e:
                logger.warning(f"Prediction failed: {e}")
                prediction = {}
            
            return {
                'total_narratives': len(self.narrative_groups),  # Key name expected by main.py
                'total_possible_narratives': len(self.narrative_groups),
                'detected_narratives_count': len(detected_narratives),
                'detected_narratives': sorted(list(detected_narratives)),
                'current_pumping_narrative': current_pumping,
                'current_narrative_activities': current_activities or [],
                'next_predicted_narrative': prediction.get('predicted_narrative'),
                'prediction_confidence': prediction.get('confidence', 0),
                'top_predictions': prediction.get('top_predictions', []),
                'historical_activities_count': len(activities)
            }
        except Exception as e:
            logger.error(f"Error in get_narrative_summary: {e}")
            return {
                'total_narratives': len(self.narrative_groups) if hasattr(self, 'narrative_groups') else 0,
                'current_pumping_narrative': None,
                'current_narrative_activities': [],
                'next_predicted_narrative': None,
                'prediction_confidence': 0,
                'top_predictions': []
            }


def main():
    """Main function to run narrative analysis"""
    analyzer = NarrativeAnalyzer()
    
    print("=== NARRATIVE ANALYSIS ===")
    
    # Detect historical activities
    print("Detecting historical narrative activities...")
    activities = analyzer.detect_historical_narrative_activities()
    print(f"Found {len(activities)} historical narrative activities")
    
    # Train model
    print("Training narrative prediction model...")
    if analyzer.train_narrative_prediction_model():
        print("Model training completed successfully")
        
        # Get current analysis
        summary = analyzer.get_narrative_summary()
        
        print(f"\n=== CURRENT NARRATIVE ANALYSIS ===")
        print(f"Total narratives tracked: {summary['total_narratives']}")
        print(f"Detected narratives: {summary['detected_narratives_count']}")
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