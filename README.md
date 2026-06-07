# AI Trading Bot 🤖📈

A complete, production-ready AI trading bot that uses machine learning to predict market movements and execute trades automatically.

## Features

- **Automated Data Collection**: Pulls real-time market data from Yahoo Finance
- **Advanced Feature Engineering**: Technical indicators (RSI, MACD, Bollinger Bands, Moving Averages)
- **Machine Learning**: Trained on historical data using Random Forest classification
- **Backtesting Engine**: Test strategies on historical data before going live
- **Live Trading**: Execute real trades via Alpaca API (paper trading available)
- **Risk Management**: Position sizing, stop losses, drawdown limits
- **Logging & Monitoring**: Track all trades and signals
- **Paper Trading**: Test with simulated money first

## Project Structure

```
ai-trading-bot/
├── data/                      # Data storage
│   ├── raw_data.csv
│   └── models/
├── src/
│   ├── data_collector.py      # Market data fetching
│   ├── features.py            # Feature engineering
│   ├── model.py               # ML model training
│   ├── backtest.py            # Backtesting engine
│   ├── trader.py              # Live trading execution
│   ├── risk_manager.py        # Risk management
│   └── logger.py              # Logging utilities
├── tests/                      # Unit tests
├── main.py                     # Main entry point
├── requirements.txt            # Python dependencies
├── .env.example               # Environment variables template
└── README.md                  # This file
```

## Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/craigwigston-create/ai-trading-bot.git
cd ai-trading-bot
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment
```bash
cp .env.example .env
# Edit .env with your API keys (optional for backtesting)
```

### 5. Run the Bot
```bash
python main.py
```

## Configuration

### Environment Variables (.env)
```
# Alpaca API (for live trading)
ALPACA_API_KEY=your_api_key
ALPACA_SECRET_KEY=your_secret_key
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Trading Parameters
SYMBOL=AAPL
POSITION_SIZE=10
MAX_RISK_PER_TRADE=0.02
STOP_LOSS_PERCENT=0.05
MAX_DAILY_DRAWDOWN=0.10

# Model Parameters
CONFIDENCE_THRESHOLD=0.75
MODEL_PATH=data/models/trading_model.pkl

# Logging
LOG_FILE=trading.log
```

## Usage Examples

### Backtest Strategy
```python
from src.data_collector import DataCollector
from src.features import FeatureEngineer
from src.model import TradingModel
from src.backtest import Backtester

# Collect data
collector = DataCollector("AAPL", "2022-01-01", "2024-06-01")
data = collector.fetch_data()

# Engineer features
engineer = FeatureEngineer(data)
data = engineer.add_technical_indicators()
data = engineer.create_target()

# Train model
model = TradingModel()
X_train, X_test, y_train, y_test = model.prepare_data(data)
model.train(X_train, y_train)
model.evaluate(X_test, y_test)

# Backtest
backtester = Backtester(initial_capital=10000)
portfolio_values = backtester.run_backtest(data, model)
backtester.print_results(portfolio_values)
```

### Paper Trading (Simulated)
```python
from src.trader import LiveTrader

trader = LiveTrader(
    api_key="YOUR_KEY",
    secret_key="YOUR_SECRET",
    paper_trading=True  # Use simulated money
)
trader.model = model
trader.run_trading_loop(interval_minutes=60)
```

## Important Notes ⚠️

1. **Start with Paper Trading**: Always test with simulated money first
2. **Backtest First**: Never trade live without backtesting
3. **Risk Management**: Never risk more than you can afford to lose
4. **Monitor Closely**: Watch the bot for the first week
5. **Past Performance ≠ Future Results**: Markets change
6. **Start Small**: Begin with $100-1000, scale up gradually
7. **Keep Learning**: Adjust your strategy as markets evolve

## Trading Strategy

The bot uses:
1. **Technical Indicators**: RSI, MACD, Bollinger Bands, Moving Averages
2. **Machine Learning**: Random Forest classifier trained on historical data
3. **Signal Generation**: Buy/Sell signals with confidence scores
4. **Risk Management**: Position sizing based on risk tolerance
5. **Stop Losses**: Automatic exits if price drops 5%

## Performance Metrics

The bot tracks:
- Win rate (% of winning trades)
- Profit/Loss
- Return on Investment (ROI)
- Maximum Drawdown
- Sharpe Ratio (risk-adjusted returns)
- Number of trades

## Troubleshooting

### Model Accuracy < 53%
- The model isn't better than random guessing
- Try different indicators or time periods
- Collect more data
- Adjust hyperparameters

### Backtest Shows Losses
- Market conditions may have changed
- Try a different stock/timeframe
- Adjust confidence threshold
- Add more features

### Live Trading Not Working
- Check API keys in .env
- Verify market hours (stocks: 9:30 AM - 4 PM EST)
- Ensure account has buying power
- Check logs for errors

## API Keys Setup

### Alpaca (Free Paper Trading)
1. Create account: https://alpaca.markets
2. Get API keys from dashboard
3. Add to .env file
4. Paper trading is completely free and risk-free

## Testing

Run unit tests:
```bash
pytest tests/
```

## Contributing

Pull requests welcome! Areas for improvement:
- Additional ML models (LSTM, XGBoost, ensemble)
- More technical indicators
- Cryptocurrency support
- Options trading
- Multi-symbol portfolio

## Disclaimer

This is an educational project. Trading involves risk. Past performance does not guarantee future results. Never invest money you can't afford to lose. Test thoroughly with paper trading first.

## License

MIT License - see LICENSE file

## Support

- Issues: GitHub Issues
- Questions: Start a Discussion
- Email: your-email@example.com

---

**Happy Trading! ��** Start with paper trading, backtest thoroughly, and scale gradually.
