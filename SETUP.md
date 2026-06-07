# AI Trading Bot - Setup & Installation Guide

## Quick Start (5 minutes)

### 1. Clone Repository
```bash
git clone https://github.com/craigwigston-create/ai-trading-bot.git
cd ai-trading-bot
```

### 2. Create Virtual Environment
```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment
```bash
# Copy example config
cp .env.example .env

# Edit .env with your settings (optional - defaults work)
nano .env
```

### 5. Run Trading Bot
```bash
# Run complete workflow (data → train → backtest)
python main.py

# Run with live trading enabled
python main.py --live
```

---

## What Gets Created

After running `python main.py`, you'll get:

```
data/
├── raw_data.csv              # Historical market data
├── backtest_results.csv      # Backtest trades & results
└── models/
    ├── trading_model.pkl     # Trained ML model
    └── trading_model_scaler.pkl

logs/
├── trading.log               # Detailed logs
└── trades.json               # Trade records (if live)
```

---

## File Descriptions

### Core Source Files (`src/`)

| File | Purpose |
|------|---------|
| `data_collector.py` | Downloads market data from Yahoo Finance |
| `features.py` | Calculates technical indicators (RSI, MACD, Bollinger Bands, etc.) |
| `model.py` | Trains Random Forest ML model on historical data |
| `backtest.py` | Simulates trading strategy on historical data |
| `trader.py` | Executes live/paper trades via Alpaca API |
| `risk_manager.py` | Position sizing, stop losses, drawdown limits |
| `logger.py` | Logging configuration |

### Main Entry Point

| File | Purpose |
|------|---------|
| `main.py` | Orchestrates entire workflow (5 steps) |

---

## Workflow Explanation

### Step 1: Data Collection
- Downloads 2 years of daily OHLCV (Open, High, Low, Close, Volume) data
- Uses Yahoo Finance (free, no API key required)
- Saves to `data/raw_data.csv`

### Step 2: Feature Engineering
- Calculates 16 technical indicators:
  - Moving Averages (SMA 20/50/200, EMA 12/26)
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - Bollinger Bands
  - Volatility, Daily Returns, ATR
- Creates target: 1 if price ↑, 0 if price ↓

### Step 3: Model Training
- Trains Random Forest classifier on historical data
- 80% training, 10% validation, 10% test data
- Evaluates on validation & test sets
- Saves model to disk

### Step 4: Backtesting
- Simulates trading on historical data using trained model
- Tracks portfolio value over time
- Shows: Total trades, Win rate, ROI, Max drawdown
- Saves detailed trade log

### Step 5: Live Trading (Optional)
- Connects to Alpaca API (requires API key)
- Executes BUY/SELL orders based on model predictions
- Supports paper trading (recommended for testing)
- Logs all trades

---

## Configuration (.env)

Key settings:

```env
# Which stock to trade
SYMBOL=AAPL

# Shares per trade
POSITION_SIZE=10

# Risk management
MAX_RISK_PER_TRADE=0.02        # 2% per trade
MAX_DAILY_DRAWDOWN=0.10         # Stop if -10% loss

# Model
CONFIDENCE_THRESHOLD=0.75       # Only trade if 75%+ confident

# Alpaca (for live trading)
ALPACA_API_KEY=pk_...
ALPACA_SECRET_KEY=sk_...
ALPACA_BASE_URL=https://paper-api.alpaca.markets  # Paper (safe!)
```

---

## Example Output

```
======================================================================
 AI TRADING BOT - COMPLETE WORKFLOW
======================================================================

[1] COLLECTING MARKET DATA...
✓ Downloaded 500 trading days of data

[2] ENGINEERING FEATURES...
✓ Created technical indicators and target variable

[3] TRAINING MACHINE LEARNING MODEL...
✓ Model trained successfully
  Validation Accuracy: 54.32%
  Test Accuracy: 53.89%

[4] BACKTESTING STRATEGY...
🤖 BUY @ 2024-06-01: 10 @ $150.32 (confidence: 81%)
💰 SELL @ 2024-06-02: 10 @ $152.15 (P/L: $18.30)
...

============================================================
  BACKTEST RESULTS
============================================================
Initial Capital:      $10,000.00
Final Value:          $11,450.32
Profit/Loss:          $1,450.32
ROI:                  +14.50%

Total Trades:         45
Buy Signals:          22
Sell Signals:         23

Win Rate:             58.3%
Max Drawdown:         -3.25%
============================================================
```

---

## Getting API Keys

### Alpaca (FREE - Recommended)

1. Sign up: https://alpaca.markets
2. Verify email
3. Dashboard → API Keys → Generate new key
4. Copy Key & Secret to `.env`
5. Use `https://paper-api.alpaca.markets` for paper trading

**Paper Trading**: Use simulated money, zero risk, same as real trading

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'alpaca_trade_api'"
```bash
pip install -r requirements.txt
```

### "ALPACA_API_KEY not set in .env"
```bash
cp .env.example .env
# Add your keys to .env
```

### "Model accuracy is 50%"
This means the model isn't better than random guessing. Try:
- Different stock/timeframe
- More training data
- Different indicators
- Adjust model parameters

### "No trades executed in backtest"
Try lowering `CONFIDENCE_THRESHOLD` (e.g., 0.60 instead of 0.75)

---

## Next Steps

1. **Run backtest first** ← Always test strategy before trading real money
2. **Paper trade** ← Use simulated money for 1-2 weeks
3. **Validate results** ← Check if backtest predictions match live performance
4. **Scale gradually** ← Start small ($100-500), increase when profitable
5. **Monitor actively** ← Watch bot for first week

---

## Important Warnings ⚠️

- ⚠️ **Past performance ≠ Future results**
- ⚠️ **Markets are unpredictable**
- ⚠️ **Always use stop losses**
- ⚠️ **Never risk money you can't afford to lose**
- ⚠️ **Test with paper trading first**

---

## Support

- **Issues**: GitHub Issues
- **Questions**: Open a Discussion
- **Bugs**: Report with error message from `logs/trading.log`

---

**Happy Trading! Start small, test thoroughly, scale gradually.** 📈
