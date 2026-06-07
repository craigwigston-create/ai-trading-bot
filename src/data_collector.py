"""Market Data Collection Module

Handles downloading historical and real-time market data from various sources.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class DataCollector:
    """Collects market data from Yahoo Finance"""

    def __init__(self, symbol, start_date=None, end_date=None):
        """
        Initialize data collector.
        
        Args:
            symbol (str): Stock symbol (e.g., 'AAPL', 'GOOGL')
            start_date (str): Start date in format 'YYYY-MM-DD'
            end_date (str): End date in format 'YYYY-MM-DD'
        """
        self.symbol = symbol
        self.start_date = start_date or (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        logger.info(f"DataCollector initialized for {symbol}")

    def fetch_data(self):
        """Download OHLCV data from Yahoo Finance
        
        Returns:
            pd.DataFrame: DataFrame with columns [Open, High, Low, Close, Volume, Adj Close]
        """
        try:
            logger.info(f"Fetching {self.symbol} data from {self.start_date} to {self.end_date}")
            data = yf.download(self.symbol, start=self.start_date, end=self.end_date, progress=False)
            logger.info(f"✓ Downloaded {len(data)} rows of data")
            return data
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            raise

    def get_last_n_days(self, n=100):
        """Get the last N days of data.
        
        Args:
            n (int): Number of days to retrieve
            
        Returns:
            pd.DataFrame: Last N days of OHLCV data
        """
        end = datetime.now()
        start = end - timedelta(days=n)
        return yf.download(self.symbol, start=start, end=end, progress=False)

    def get_realtime_price(self):
        """Get current market price.
        
        Returns:
            float: Current closing price
        """
        try:
            data = yf.download(self.symbol, period='1d', progress=False)
            current_price = data['Close'].iloc[-1]
            logger.info(f"Current price of {self.symbol}: ${current_price:.2f}")
            return current_price
        except Exception as e:
            logger.error(f"Error getting realtime price: {e}")
            return None

    def save_data(self, data, filepath='data/raw_data.csv'):
        """Save data to CSV.
        
        Args:
            data (pd.DataFrame): Data to save
            filepath (str): Where to save the file
        """
        try:
            data.to_csv(filepath)
            logger.info(f"✓ Data saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving data: {e}")

    def load_data(self, filepath='data/raw_data.csv'):
        """Load data from CSV.
        
        Args:
            filepath (str): Path to CSV file
            
        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            data = pd.read_csv(filepath, index_col=0, parse_dates=True)
            logger.info(f"✓ Loaded data from {filepath}")
            return data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    collector = DataCollector("AAPL", "2023-01-01", "2024-06-01")
    data = collector.fetch_data()
    print(data.head())
    print(f"\nShape: {data.shape}")
    print(f"\nData types:\n{data.dtypes}")
