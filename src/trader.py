"""Live Trading Module

Executes real or paper trades via Alpaca API.
"""

import time
import logging
from datetime import datetime
import os
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class LiveTrader:
    """Executes live trades via Alpaca API"""

    def __init__(self, symbol='AAPL', paper_trading=True):
        """
        Initialize live trader.
        
        Args:
            symbol (str): Stock symbol to trade
            paper_trading (bool): Use simulated money (recommended)
        """
        load_dotenv()
        
        self.symbol = symbol
        self.paper_trading = paper_trading
        self.position = 0
        self.entry_price = 0
        
        # Import here to handle missing dependencies gracefully
        try:
            from alpaca_trade_api import REST
            
            api_key = os.getenv('ALPACA_API_KEY')
            secret_key = os.getenv('ALPACA_SECRET_KEY')
            base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
            
            if not api_key or not secret_key:
                raise ValueError("ALPACA_API_KEY or ALPACA_SECRET_KEY not set in .env")
            
            self.api = REST(api_key, secret_key, base_url=base_url)
            self.model = None
            
            mode = "PAPER" if paper_trading else "LIVE"
            logger.info(f"LiveTrader initialized ({mode} mode) for {symbol}")
        
        except ImportError:
            logger.warning("Alpaca API not installed. Install with: pip install alpaca-trade-api")
            self.api = None

    def get_account_info(self):
        """Get account details.
        
        Returns:
            dict: Account information
        """
        if not self.api:
            return None
        
        try:
            account = self.api.get_account()
            return {
                'buying_power': float(account.buying_power),
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value),
                'equity': float(account.equity)
            }
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return None

    def get_current_price(self):
        """Get current market price.
        
        Returns:
            float: Current price or None
        """
        if not self.api:
            return None
        
        try:
            bar = self.api.get_latest_bar(self.symbol)
            return float(bar.c) if bar else None
        except Exception as e:
            logger.error(f"Error getting price: {e}")
            return None

    def execute_buy(self, qty, price=None):
        """Place a buy order.
        
        Args:
            qty (int): Number of shares
            price (float): Limit price (None for market order)
            
        Returns:
            dict: Order details or None
        """
        if not self.api:
            logger.error("API not initialized")
            return None
        
        try:
            order_type = 'limit' if price else 'market'
            order = self.api.submit_order(
                symbol=self.symbol,
                qty=qty,
                side='buy',
                type=order_type,
                time_in_force='day',
                limit_price=price
            )
            
            logger.info(f"✓ BUY order placed: {qty} shares @ ${price:.2f}" if price 
                       else f"✓ BUY order placed: {qty} shares (market)")
            
            self.position += qty
            self.entry_price = price or self.get_current_price()
            
            return {
                'order_id': order.id,
                'symbol': order.symbol,
                'qty': order.qty,
                'side': order.side,
                'status': order.status
            }
        
        except Exception as e:
            logger.error(f"Buy order failed: {e}")
            return None

    def execute_sell(self, qty, price=None):
        """Place a sell order.
        
        Args:
            qty (int): Number of shares
            price (float): Limit price (None for market order)
            
        Returns:
            dict: Order details or None
        """
        if not self.api:
            logger.error("API not initialized")
            return None
        
        try:
            order_type = 'limit' if price else 'market'
            order = self.api.submit_order(
                symbol=self.symbol,
                qty=qty,
                side='sell',
                type=order_type,
                time_in_force='day',
                limit_price=price
            )
            
            logger.info(f"✓ SELL order placed: {qty} shares @ ${price:.2f}" if price
                       else f"✓ SELL order placed: {qty} shares (market)")
            
            self.position -= qty
            
            return {
                'order_id': order.id,
                'symbol': order.symbol,
                'qty': order.qty,
                'side': order.side,
                'status': order.status
            }
        
        except Exception as e:
            logger.error(f"Sell order failed: {e}")
            return None

    def close_position(self):
        """Close any open position.
        
        Returns:
            bool: True if successful
        """
        if self.position <= 0:
            return True
        
        result = self.execute_sell(self.position)
        return result is not None

    def run_trading_loop(self, model, interval_minutes=60, max_iterations=None):
        """Run trading bot continuously.
        
        Args:
            model: Trained ML model with predict method
            interval_minutes (int): Minutes between checks
            max_iterations (int): Max iterations before stopping (None = infinite)
        """
        if not self.api:
            logger.error("Cannot run trading: API not initialized")
            return
        
        self.model = model
        iteration = 0
        
        logger.info(f"Trading loop started. Interval: {interval_minutes} minutes")
        logger.info(f"Symbol: {self.symbol}, Paper Trading: {self.paper_trading}")
        
        try:
            while True:
                if max_iterations and iteration >= max_iterations:
                    break
                
                iteration += 1
                logger.info(f"\n--- Trading Iteration {iteration} @ {datetime.now()} ---")
                
                try:
                    # Get account info
                    account = self.get_account_info()
                    if account:
                        logger.info(f"Account: ${account['equity']:,.2f} equity, ${account['buying_power']:,.2f} buying power")
                    
                    # Get current price
                    current_price = self.get_current_price()
                    if not current_price:
                        logger.warning("Could not get price, skipping iteration")
                        time.sleep(interval_minutes * 60)
                        continue
                    
                    logger.info(f"{self.symbol}: ${current_price:.2f}")
                    
                    # TODO: Get features from market data
                    # For now, this is a placeholder
                    # In production, you would:
                    # 1. Fetch latest OHLCV data
                    # 2. Calculate technical indicators
                    # 3. Get features
                    # 4. Make prediction
                    
                    logger.info("Waiting for next iteration...")
                
                except Exception as e:
                    logger.error(f"Error in trading loop: {e}", exc_info=True)
                
                # Wait before next iteration
                time.sleep(interval_minutes * 60)
        
        except KeyboardInterrupt:
            logger.info("\n⏹️  Trading stopped by user")
            self.close_position()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example usage
    trader = LiveTrader(symbol='AAPL', paper_trading=True)
    
    # Get account info
    account = trader.get_account_info()
    if account:
        print(f"\nAccount Info:")
        for key, value in account.items():
            print(f"  {key}: ${value:,.2f}" if 'value' in key or 'cash' in key or 'power' in key else f"  {key}: {value}")
    
    # Get current price
    price = trader.get_current_price()
    if price:
        print(f"\nCurrent price: ${price:.2f}")
    
    # Uncomment to run trading loop:
    # trader.run_trading_loop(model, interval_minutes=60, max_iterations=5)
