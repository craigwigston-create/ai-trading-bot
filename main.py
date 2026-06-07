"""AI Trading Bot - Main Entry Point

This is the main script to run the complete trading bot workflow.
Steps:
1. Collect market data
2. Engineer features (technical indicators)
3. Train ML model
4. Backtest strategy
5. [Optional] Run live/paper trading
"""

import sys
import logging
from pathlib import Path
from dotenv import load_dotenv
import os

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.logger import setup_logging
from src.data_collector import DataCollector
from src.features import FeatureEngineer
from src.model import TradingModel
from src.backtest import Backtester
from src.trader import LiveTrader
from src.risk_manager import RiskManager

# Setup logging
logger = setup_logging(log_file='trading.log', level=logging.INFO)

# Load environment variables
load_dotenv()


def main():
    """Main entry point for the trading bot."""
    
    print("\n" + "="*70)
    print(" AI TRADING BOT - COMPLETE WORKFLOW")
    print("="*70)
    
    # Configuration
    symbol = os.getenv('SYMBOL', 'AAPL')
    initial_capital = 10000
    training_years = int(os.getenv('TRAINING_YEARS', 2))
    confidence_threshold = float(os.getenv('CONFIDENCE_THRESHOLD', 0.75))
    position_size = int(os.getenv('POSITION_SIZE', 10))
    
    logger.info(f"Configuration: Symbol={symbol}, Capital=${initial_capital}")
    
    # ========== STEP 1: Collect Data ==========
    print("\n[1] COLLECTING MARKET DATA...")
    print("-" * 70)
    
    try:
        from datetime import datetime, timedelta
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365*training_years)).strftime('%Y-%m-%d')
        
        collector = DataCollector(symbol, start_date, end_date)
        data = collector.fetch_data()
        logger.info(f"✓ Downloaded {len(data)} trading days")
        print(f"✓ Downloaded {len(data)} trading days of data")
    except Exception as e:
        logger.error(f"Failed to collect data: {e}")
        print(f"✗ Failed to collect data: {e}")
        return
    
    # ========== STEP 2: Engineer Features ==========
    print("\n[2] ENGINEERING FEATURES...")
    print("-" * 70)
    
    try:
        engineer = FeatureEngineer(data)
        data_features = engineer.add_technical_indicators()
        data_features = engineer.create_target()
        logger.info(f"✓ Created {len(data_features.columns)} features")
        print(f"✓ Created technical indicators and target variable")
    except Exception as e:
        logger.error(f"Failed to engineer features: {e}")
        print(f"✗ Failed to engineer features: {e}")
        return
    
    # ========== STEP 3: Train Model ==========
    print("\n[3] TRAINING MACHINE LEARNING MODEL...")
    print("-" * 70)
    
    try:
        model = TradingModel(model_type='random_forest')
        X_train, X_val, X_test, y_train, y_val, y_test = model.prepare_data(data_features)
        model.train(X_train, y_train)
        
        # Evaluate on validation set
        val_metrics = model.evaluate(X_val, y_val, "Validation")
        
        # Evaluate on test set
        test_metrics = model.evaluate(X_test, y_test, "Test")
        
        # Save model
        Path('data/models').mkdir(parents=True, exist_ok=True)
        model.save('data/models/trading_model.pkl')
        logger.info("✓ Model trained and saved")
        print("✓ Model trained successfully")
        print(f"  Validation Accuracy: {val_metrics['accuracy']:.2%}")
        print(f"  Test Accuracy: {test_metrics['accuracy']:.2%}")
    except Exception as e:
        logger.error(f"Failed to train model: {e}")
        print(f"✗ Failed to train model: {e}")
        return
    
    # ========== STEP 4: Backtest ==========
    print("\n[4] BACKTESTING STRATEGY...")
    print("-" * 70)
    
    try:
        backtester = Backtester(
            initial_capital=initial_capital,
            position_size=position_size,
            confidence_threshold=confidence_threshold
        )
        portfolio_values = backtester.run_backtest(data_features, model, start_idx=200)
        backtester.print_results()
        
        # Save results
        Path('data').mkdir(exist_ok=True)
        backtester.save_results('data/backtest_results.csv')
        logger.info("✓ Backtest completed and results saved")
        print("✓ Backtest results saved to data/backtest_results.csv")
    except Exception as e:
        logger.error(f"Failed to run backtest: {e}")
        print(f"✗ Failed to run backtest: {e}")
        return
    
    # ========== STEP 5: Live Trading (Optional) ==========
    print("\n[5] LIVE/PAPER TRADING (Optional)")
    print("-" * 70)
    print("\nTo enable live trading:")
    print("1. Set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env")
    print("2. Run: python main.py --live")
    print("\n⚠️  WARNING: Test with paper trading first!")
    print("   Use ALPACA_BASE_URL=https://paper-api.alpaca.markets")
    
    # Check if --live flag is passed
    if '--live' in sys.argv:
        print("\n[5] STARTING LIVE TRADING...")
        print("-" * 70)
        
        try:
            trader = LiveTrader(symbol=symbol, paper_trading=True)
            
            # Get account info
            account = trader.get_account_info()
            if account:
                print(f"✓ Connected to Alpaca")
                print(f"  Account Equity: ${account['equity']:,.2f}")
                print(f"  Buying Power: ${account['buying_power']:,.2f}")
            else:
                print("✗ Could not connect to Alpaca")
                return
            
            # Run trading loop
            print(f"\nStarting trading bot for {symbol}...")
            print("Press Ctrl+C to stop\n")
            trader.run_trading_loop(
                model=model,
                interval_minutes=int(os.getenv('INTERVAL_MINUTES', 60))
            )
        
        except Exception as e:
            logger.error(f"Failed to start live trading: {e}")
            print(f"✗ Failed to start live trading: {e}")
    
    # ========== Summary ==========
    print("\n" + "="*70)
    print(" TRADING BOT WORKFLOW COMPLETE")
    print("="*70)
    print(f"\nModel saved to: data/models/trading_model.pkl")
    print(f"Backtest results saved to: data/backtest_results.csv")
    print(f"Logs saved to: logs/trading.log")
    print(f"\nNext steps:")
    print(f"1. Review backtest results")
    print(f"2. Adjust strategy if needed")
    print(f"3. Start paper trading (recommended)")
    print(f"4. Scale to live trading after validation")
    print("\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Trading bot stopped by user")
        logger.info("Trading bot stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"\n✗ Fatal error: {e}")
        sys.exit(1)
