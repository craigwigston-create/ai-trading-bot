"""Risk Management Module

Handles position sizing, stop losses, and risk controls.
"""

import logging

logger = logging.getLogger(__name__)


class RiskManager:
    """Manages trading risk and position sizing"""

    def __init__(self, account_size=10000, max_risk_per_trade=0.02,
                 max_daily_loss=0.10, max_position_size=100):
        """
        Initialize risk manager.
        
        Args:
            account_size (float): Total account value
            max_risk_per_trade (float): Max risk per trade (2% = 0.02)
            max_daily_loss (float): Max daily loss before stopping (10% = 0.10)
            max_position_size (int): Max shares per trade
        """
        self.account_size = account_size
        self.max_risk_per_trade = max_risk_per_trade
        self.max_daily_loss = max_daily_loss
        self.max_position_size = max_position_size
        self.peak_account_value = account_size
        self.daily_loss = 0
        
        logger.info(f"RiskManager initialized")
        logger.info(f"  Account Size: ${account_size:,.2f}")
        logger.info(f"  Max Risk/Trade: {max_risk_per_trade*100:.1f}%")
        logger.info(f"  Max Daily Loss: {max_daily_loss*100:.1f}%")

    def calculate_position_size(self, entry_price, stop_loss_price):
        """Calculate position size based on risk.
        
        Risk Management Formula:
        Position Size = (Account Risk) / (Entry Price - Stop Loss Price)
        
        Args:
            entry_price (float): Price at which to enter
            stop_loss_price (float): Price at which to exit (loss)
            
        Returns:
            int: Number of shares to buy
        """
        # Amount willing to risk
        risk_amount = self.account_size * self.max_risk_per_trade
        
        # Risk per share
        risk_per_share = abs(entry_price - stop_loss_price)
        
        if risk_per_share == 0:
            return 0
        
        # Calculate position size
        position_size = int(risk_amount / risk_per_share)
        
        # Cap at max position size
        position_size = min(position_size, self.max_position_size)
        
        logger.info(f"Position Size Calculation:")
        logger.info(f"  Entry: ${entry_price:.2f}, Stop: ${stop_loss_price:.2f}")
        logger.info(f"  Risk Amount: ${risk_amount:.2f}")
        logger.info(f"  Risk Per Share: ${risk_per_share:.2f}")
        logger.info(f"  → Position Size: {position_size} shares")
        
        return position_size

    def should_stop_loss(self, entry_price, current_price, stop_loss_percent=None):
        """Check if stop loss should be triggered.
        
        Args:
            entry_price (float): Entry price
            current_price (float): Current price
            stop_loss_percent (float): Stop loss % (default: 5%)
            
        Returns:
            bool: True if stop loss triggered
        """
        if stop_loss_percent is None:
            stop_loss_percent = self.max_risk_per_trade
        
        loss_percent = (entry_price - current_price) / entry_price
        
        if loss_percent >= stop_loss_percent:
            logger.warning(f"STOP LOSS triggered: {loss_percent*100:.2f}% loss")
            return True
        
        return False

    def can_trade(self, current_account_value):
        """Check if daily loss limit has been reached.
        
        Args:
            current_account_value (float): Current account value
            
        Returns:
            bool: True if can still trade
        """
        # Update peak
        if current_account_value > self.peak_account_value:
            self.peak_account_value = current_account_value
        
        # Calculate drawdown
        drawdown = (self.peak_account_value - current_account_value) / self.peak_account_value
        
        if drawdown >= self.max_daily_loss:
            logger.error(f"MAX DRAWDOWN ({drawdown*100:.2f}%) reached. STOPPING ALL TRADES.")
            return False
        
        return True

    def get_risk_metrics(self, current_price, position_size, entry_price):
        """Get risk metrics for current position.
        
        Args:
            current_price (float): Current market price
            position_size (int): Number of shares held
            entry_price (float): Price at entry
            
        Returns:
            dict: Risk metrics
        """
        unrealized_pnl = (current_price - entry_price) * position_size
        unrealized_pnl_percent = ((current_price - entry_price) / entry_price) * 100
        position_value = current_price * position_size
        percent_of_account = (position_value / self.account_size) * 100
        
        return {
            'unrealized_pnl': unrealized_pnl,
            'unrealized_pnl_percent': unrealized_pnl_percent,
            'position_value': position_value,
            'percent_of_account': percent_of_account
        }

    def validate_trade(self, trade_value, account_value):
        """Validate if trade is safe.
        
        Args:
            trade_value (float): Total value of trade
            account_value (float): Current account value
            
        Returns:
            bool: True if trade is valid
        """
        # Check if enough capital
        if trade_value > account_value:
            logger.warning(f"Insufficient capital: ${trade_value:.2f} > ${account_value:.2f}")
            return False
        
        # Check if within daily loss limit
        if not self.can_trade(account_value):
            return False
        
        return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    risk_mgr = RiskManager(account_size=10000, max_risk_per_trade=0.02)
    
    # Calculate position size
    position_size = risk_mgr.calculate_position_size(entry_price=150, stop_loss_price=145)
    print(f"\nBuy {position_size} shares")
    
    # Check stop loss
    should_exit = risk_mgr.should_stop_loss(entry_price=150, current_price=142, 
                                            stop_loss_percent=0.05)
    print(f"Exit trade: {should_exit}")
    
    # Get risk metrics
    metrics = risk_mgr.get_risk_metrics(current_price=155, position_size=position_size, 
                                        entry_price=150)
    print(f"\nRisk Metrics: {metrics}")
