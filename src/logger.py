"""Logging Configuration Module

Sets up logging for the trading bot.
"""

import logging
import json
from datetime import datetime
from pathlib import Path


def setup_logging(log_file='trading.log', level=logging.INFO):
    """Configure logging for the bot.
    
    Args:
        log_file (str): Path to log file
        level: Logging level
        
    Returns:
        logging.Logger: Configured logger
    """
    # Create logs directory
    Path('logs').mkdir(exist_ok=True)
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # File handler
    fh = logging.FileHandler(f'logs/{log_file}')
    fh.setLevel(level)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    logger.info("Logging configured")
    return logger


class TradeLogger:
    """Logs trades to a structured format"""
    
    def __init__(self, trades_file='trades.json'):
        self.trades_file = f'logs/{trades_file}'
        Path('logs').mkdir(exist_ok=True)
    
    def log_trade(self, trade_data):
        """Log a trade.
        
        Args:
            trade_data (dict): Trade details
        """
        trade_data['timestamp'] = datetime.now().isoformat()
        
        # Append to JSON file
        trades = []
        if Path(self.trades_file).exists():
            with open(self.trades_file, 'r') as f:
                trades = json.load(f)
        
        trades.append(trade_data)
        
        with open(self.trades_file, 'w') as f:
            json.dump(trades, f, indent=2)
    
    def get_trades(self):
        """Get all logged trades.
        
        Returns:
            list: Trade records
        """
        if Path(self.trades_file).exists():
            with open(self.trades_file, 'r') as f:
                return json.load(f)
        return []


if __name__ == "__main__":
    logger = setup_logging()
    logger.info("Test log message")
    logger.warning("Test warning")
    logger.error("Test error")
    
    trader_logger = TradeLogger()
    trader_logger.log_trade({
        'symbol': 'AAPL',
        'type': 'BUY',
        'price': 150.00,
        'qty': 10,
        'signal_confidence': 0.85
    })
    
    print(f"\nTrades logged: {trader_logger.get_trades()}")
