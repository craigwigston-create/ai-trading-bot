"""Machine Learning Model Module

Trains and evaluates ML models for trading predictions.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class TradingModel:
    """Machine Learning Model for Trading Predictions"""

    def __init__(self, model_type='random_forest', random_state=42):
        """
        Initialize trading model.
        
        Args:
            model_type (str): Type of model ('random_forest', 'xgboost')
            random_state (int): Random seed for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=random_state,
                n_jobs=-1,  # Use all CPU cores
                class_weight='balanced'  # Handle imbalanced data
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_cols = None
        logger.info(f"TradingModel initialized with {model_type}")

    def prepare_data(self, df, test_size=0.2, val_size=0.1):
        """Prepare and split data for training.
        
        Args:
            df (pd.DataFrame): DataFrame with features and Target column
            test_size (float): Proportion for testing
            val_size (float): Proportion for validation
            
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # Remove NaN
        df = df.dropna()
        logger.info(f"Data shape after removing NaN: {df.shape}")
        
        # Get feature columns (all except Target)
        feature_cols = [col for col in df.columns if col != 'Target']
        self.feature_cols = feature_cols
        
        X = df[feature_cols].values
        y = df['Target'].values
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, 
            random_state=self.random_state, stratify=y_temp
        )
        
        # Normalize data
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)
        
        logger.info(f"✓ Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        logger.info(f"  Features: {len(feature_cols)}")
        logger.info(f"  Class distribution (Train): {np.bincount(y_train)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test

    def train(self, X_train, y_train):
        """Train the model.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
        """
        logger.info(f"Training {self.model_type} model...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        logger.info("✓ Model trained!")

    def evaluate(self, X_test, y_test, name="Test"):
        """Evaluate model on test set.
        
        Args:
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test labels
            name (str): Name for logging
            
        Returns:
            dict: Evaluation metrics
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        y_pred = self.model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        logger.info(f"\n{name} Set Results:")
        logger.info(f"  Accuracy:  {accuracy:.2%}")
        logger.info(f"  Precision: {precision:.2%}")
        logger.info(f"  Recall:    {recall:.2%}")
        logger.info(f"  F1 Score:  {f1:.2%}")
        
        if accuracy > 0.53:
            logger.info(f"  ✓ Model performs better than random (50%)")
        else:
            logger.warning(f"  ⚠ Model is not better than random guessing")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def predict(self, features):
        """Make prediction on new data.
        
        Args:
            features (list or np.ndarray): Feature values
            
        Returns:
            dict: {'signal': 'BUY'/'SELL', 'confidence': float}
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        # Handle single prediction
        if isinstance(features, list):
            features = np.array(features).reshape(1, -1)
        elif isinstance(features, np.ndarray) and features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Normalize
        features_scaled = self.scaler.transform(features)
        
        # Predict
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        return {
            'signal': 'BUY' if prediction == 1 else 'SELL',
            'confidence': max(probabilities),
            'probabilities': {'SELL': probabilities[0], 'BUY': probabilities[1]}
        }

    def save(self, filepath='data/models/trading_model.pkl'):
        """Save trained model to disk.
        
        Args:
            filepath (str): Path to save model
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, filepath)
        joblib.dump(self.scaler, filepath.replace('.pkl', '_scaler.pkl'))
        logger.info(f"✓ Model saved to {filepath}")

    def load(self, filepath='data/models/trading_model.pkl'):
        """Load trained model from disk.
        
        Args:
            filepath (str): Path to load model from
        """
        self.model = joblib.load(filepath)
        self.scaler = joblib.load(filepath.replace('.pkl', '_scaler.pkl'))
        self.is_trained = True
        logger.info(f"✓ Model loaded from {filepath}")

    def get_feature_importance(self, top_n=10):
        """Get most important features.
        
        Args:
            top_n (int): Number of top features to return
            
        Returns:
            pd.DataFrame: Features ranked by importance
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info(f"\nTop {top_n} Important Features:")
        for idx, row in importance.head(top_n).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        return importance


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.insert(0, '.')
    from src.data_collector import DataCollector
    from src.features import FeatureEngineer
    
    logging.basicConfig(level=logging.INFO)
    
    # Get data
    collector = DataCollector("AAPL", "2023-01-01", "2024-06-01")
    data = collector.fetch_data()
    
    # Engineer features
    engineer = FeatureEngineer(data)
    data = engineer.add_technical_indicators()
    data = engineer.create_target()
    
    # Train model
    model = TradingModel()
    X_train, X_val, X_test, y_train, y_val, y_test = model.prepare_data(data)
    model.train(X_train, y_train)
    model.evaluate(X_val, y_val, "Validation")
    model.evaluate(X_test, y_test, "Test")
    
    # Feature importance
    model.get_feature_importance()
    
    # Make a prediction
    sample_features = X_test[0]
    prediction = model.predict(sample_features)
    print(f"\nSample prediction: {prediction}")
