"""Feature Engineering Module

Creates technical indicators and features for machine learning model.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Builds technical indicators and features"""

    def __init__(self, df):
        """
        Initialize with OHLCV data.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV columns
        """
        self.df = df.copy()
        logger.info(f"FeatureEngineer initialized with {len(df)} rows")

    def add_technical_indicators(self):
        """Add all technical indicators to dataframe.
        
        Returns:
            pd.DataFrame: DataFrame with added indicators
        """
        logger.info("Adding technical indicators...")
        
        # Simple Moving Averages
        self.df['SMA_20'] = self.df['Close'].rolling(window=20).mean()
        self.df['SMA_50'] = self.df['Close'].rolling(window=50).mean()
        self.df['SMA_200'] = self.df['Close'].rolling(window=200).mean()
        
        # Exponential Moving Averages
        self.df['EMA_12'] = self.df['Close'].ewm(span=12).mean()
        self.df['EMA_26'] = self.df['Close'].ewm(span=26).mean()
        
        # RSI (Relative Strength Index)
        self.df['RSI'] = self._calculate_rsi(self.df['Close'], period=14)
        
        # MACD (Moving Average Convergence Divergence)
        self.df['MACD'], self.df['Signal_Line'], self.df['MACD_Histogram'] = \
            self._calculate_macd(self.df['Close'])
        
        # Bollinger Bands
        self.df['BB_Upper'], self.df['BB_Middle'], self.df['BB_Lower'] = \
            self._calculate_bollinger_bands(self.df['Close'], period=20)
        
        # Volatility
        self.df['Volatility'] = self.df['Close'].rolling(window=20).std()
        
        # Daily Returns
        self.df['Daily_Return'] = self.df['Close'].pct_change()
        
        # Volume indicators
        self.df['Volume_MA'] = self.df['Volume'].rolling(window=20).mean()
        self.df['Volume_Ratio'] = self.df['Volume'] / self.df['Volume_MA']
        
        # ATR (Average True Range) - volatility
        self.df['ATR'] = self._calculate_atr(self.df, period=14)
        
        logger.info(f"✓ Added {len(self.df.columns)} columns (including original OHLCV)")
        return self.df

    def _calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index.
        
        RSI measures momentum (0-100). >70 = overbought, <30 = oversold
        
        Args:
            prices (pd.Series): Close prices
            period (int): Period for calculation
            
        Returns:
            pd.Series: RSI values
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD (Moving Average Convergence Divergence).
        
        MACD is a trend-following momentum indicator.
        
        Args:
            prices (pd.Series): Close prices
            fast (int): Fast EMA period
            slow (int): Slow EMA period
            signal (int): Signal line period
            
        Returns:
            tuple: (MACD, Signal Line, Histogram)
        """
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram

    def _calculate_bollinger_bands(self, prices, period=20, num_std=2):
        """Calculate Bollinger Bands.
        
        Bollinger Bands show volatility and support/resistance levels.
        
        Args:
            prices (pd.Series): Close prices
            period (int): Period for SMA
            num_std (int): Number of standard deviations
            
        Returns:
            tuple: (Upper Band, Middle Band, Lower Band)
        """
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        return upper, sma, lower

    def _calculate_atr(self, df, period=14):
        """Calculate Average True Range.
        
        ATR measures volatility.
        
        Args:
            df (pd.DataFrame): OHLC data
            period (int): Period for calculation
            
        Returns:
            pd.Series: ATR values
        """
        high_low = df['High'] - df['Low']
        high_close = abs(df['High'] - df['Close'].shift())
        low_close = abs(df['Low'] - df['Close'].shift())
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        return atr

    def create_target(self, lookahead=1):
        """Create target variable: 1 if price goes up, 0 if down.
        
        Args:
            lookahead (int): Days ahead to predict
            
        Returns:
            pd.DataFrame: DataFrame with 'Target' column
        """
        self.df['Target'] = (self.df['Close'].shift(-lookahead) > self.df['Close']).astype(int)
        logger.info(f"✓ Target variable created")
        return self.df

    def get_features_and_target(self):
        """Get clean features and target for ML model.
        
        Returns:
            tuple: (features DataFrame, target Series)
        """
        # Remove rows with NaN
        df_clean = self.df.dropna()
        
        feature_cols = [
            'SMA_20', 'SMA_50', 'SMA_200', 'EMA_12', 'EMA_26',
            'RSI', 'MACD', 'Signal_Line', 'MACD_Histogram',
            'BB_Upper', 'BB_Middle', 'BB_Lower',
            'Volatility', 'Daily_Return', 'Volume_Ratio', 'ATR'
        ]
        
        X = df_clean[feature_cols]
        y = df_clean['Target']
        
        logger.info(f"✓ Extracted {len(X)} samples with {len(feature_cols)} features")
        return X, y


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.insert(0, '.')
    from src.data_collector import DataCollector
    
    logging.basicConfig(level=logging.INFO)
    
    collector = DataCollector("AAPL", "2023-01-01", "2024-06-01")
    data = collector.fetch_data()
    
    engineer = FeatureEngineer(data)
    data_with_features = engineer.add_technical_indicators()
    data_with_features = engineer.create_target()
    
    X, y = engineer.get_features_and_target()
    print(f"\nFeatures shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"\nClass distribution:")
    print(y.value_counts())
