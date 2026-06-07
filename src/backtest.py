"""Backtesting Engine Module

Simulates trading strategy on historical data to evaluate performance.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class Backtester:
    """Backtests trading strategy on historical data"""

    def __init__(self, initial_capital=10000, position_size=10, 
                 stop_loss_percent=0.05, confidence_threshold=0.7):
        """
        Initialize backtester.
        
        Args:
            initial_capital (float): Starting capital
            position_size (int): Shares to trade per signal
            stop_loss_percent (float): Stop loss percentage (0.05 = 5%)
            confidence_threshold (float): Min confidence to trade (0.7 = 70%)
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position = 0  # Current position size
        self.entry_price = 0  # Price at which we entered position
        self.position_size = position_size
        self.stop_loss_percent = stop_loss_percent
        self.confidence_threshold = confidence_threshold
        
        self.trades = []  # Track all trades
        self.portfolio_values = []  # Track portfolio value over time
        self.dates = []
        
        logger.info(f"Backtester initialized with ${initial_capital} capital")

    def run_backtest(self, df, model, start_idx=100):
        """Run backtest simulation.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV and features
            model: Trained ML model with predict method
            start_idx (int): Row to start trading (after enough history)
            
        Returns:
            list: Portfolio values over time
        """
        logger.info(f"Running backtest from row {start_idx}...")
        
        self.capital = self.initial_capital
        self.position = 0
        self.trades = []
        self.portfolio_values = []
        self.dates = []
        
        feature_cols = [col for col in df.columns if col not in 
                       ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 'Target']]
        
        for idx in range(start_idx, len(df)):
            current_price = df.iloc[idx]['Close']
            current_date = df.index[idx]
            
            # Get features for this row
            features = df.iloc[idx][feature_cols].values.reshape(1, -1)
            
            # Make prediction
            try:
                prediction = model.predict(features)
                signal = prediction['signal']
                confidence = prediction['confidence']
            except:
                continue
            
            # Check stop loss
            if self.position > 0:
                loss_percent = (self.entry_price - current_price) / self.entry_price
                if loss_percent >= self.stop_loss_percent:
                    # Stop loss triggered
                    revenue = self.position * current_price
                    self.capital += revenue
                    self.trades.append({
                        'date': current_date,
                        'type': 'SELL (Stop Loss)',
                        'price': current_price,
                        'shares': self.position,
                        'total': revenue
                    })
                    logger.info(f"  🛑 STOP LOSS @ {current_date}: Sold {self.position} @ ${current_price:.2f}")
                    self.position = 0
            
            # Execute trades based on signal
            if confidence > self.confidence_threshold:
                if signal == 'BUY' and self.position == 0:
                    # Buy
                    cost = self.position_size * current_price
                    if cost <= self.capital:
                        self.capital -= cost
                        self.position = self.position_size
                        self.entry_price = current_price
                        self.trades.append({
                            'date': current_date,
                            'type': 'BUY',
                            'price': current_price,
                            'shares': self.position_size,
                            'total': cost,
                            'confidence': confidence
                        })
                        logger.info(f"  💰 BUY @ {current_date}: {self.position_size} @ ${current_price:.2f} (confidence: {confidence:.0%})")
                
                elif signal == 'SELL' and self.position > 0:
                    # Sell
                    revenue = self.position * current_price
                    profit_loss = revenue - (self.position * self.entry_price)
                    self.capital += revenue
                    self.trades.append({
                        'date': current_date,
                        'type': 'SELL',
                        'price': current_price,
                        'shares': self.position,
                        'total': revenue,
                        'profit_loss': profit_loss,
                        'confidence': confidence
                    })
                    logger.info(f"  🎯 SELL @ {current_date}: {self.position} @ ${current_price:.2f} (P/L: ${profit_loss:.2f})")
                    self.position = 0
            
            # Calculate portfolio value
            portfolio_value = self.capital + (self.position * current_price)
            self.portfolio_values.append(portfolio_value)
            self.dates.append(current_date)
        
        # Close any remaining position at end
        if self.position > 0:
            final_price = df.iloc[-1]['Close']
            revenue = self.position * final_price
            self.capital += revenue
            self.position = 0
            logger.info(f"  📊 Closed position at end: ${revenue:.2f}")
        
        logger.info(f"✓ Backtest completed with {len(self.trades)} trades")
        return self.portfolio_values

    def print_results(self):
        """Print backtest results."""
        if not self.portfolio_values:
            logger.warning("No backtest results to print")
            return
        
        final_value = self.portfolio_values[-1]
        profit = final_value - self.initial_capital
        roi = (profit / self.initial_capital) * 100
        
        # Calculate drawdown
        cummax = np.maximum.accumulate(self.portfolio_values)
        drawdown = (np.array(self.portfolio_values) - cummax) / cummax
        max_drawdown = np.min(drawdown) * 100
        
        # Win rate
        closed_trades = [t for t in self.trades if t['type'] == 'SELL']
        winning_trades = [t for t in closed_trades if t.get('profit_loss', 0) > 0]
        win_rate = len(winning_trades) / len(closed_trades) * 100 if closed_trades else 0
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"  BACKTEST RESULTS")
        print(f"{'='*60}")
        print(f"Initial Capital:      ${self.initial_capital:,.2f}")
        print(f"Final Value:          ${final_value:,.2f}")
        print(f"Profit/Loss:          ${profit:,.2f}")
        print(f"ROI:                  {roi:+.2f}%")
        print(f"\nTotal Trades:         {len(self.trades)}")
        print(f"Buy Signals:          {len([t for t in self.trades if t['type'] == 'BUY'])}")
        print(f"Sell Signals:         {len([t for t in self.trades if t['type'] == 'SELL'])}")
        print(f"\nWin Rate:             {win_rate:.1f}%")
        print(f"Max Drawdown:         {max_drawdown:.2f}%")
        print(f"{'='*60}\n")

    def get_trades_df(self):
        """Get trades as DataFrame.
        
        Returns:
            pd.DataFrame: All trades with details
        """
        return pd.DataFrame(self.trades)

    def save_results(self, filepath='data/backtest_results.csv'):
        """Save backtest results to CSV.
        
        Args:
            filepath (str): Where to save
        """
        df_trades = self.get_trades_df()
        df_trades.to_csv(filepath, index=False)
        logger.info(f"✓ Results saved to {filepath}")


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.insert(0, '.')
    from src.data_collector import DataCollector
    from src.features import FeatureEngineer
    from src.model import TradingModel
    
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
    
    # Run backtest
    backtester = Backtester(initial_capital=10000, position_size=10)
    portfolio_values = backtester.run_backtest(data, model)
    backtester.print_results()
