# Predictive Stock Analysis System

A comprehensive machine learning system for stock price prediction and analysis using multiple models including Linear Regression, Random Forest, Gradient Boosting, and LSTM neural networks.

## Features

- **Multiple ML Models**: Linear Regression, Random Forest, Gradient Boosting, LSTM, and Ensemble methods
- **Technical Indicators**: RSI, MACD, Bollinger Bands, Moving Averages, and more
- **Advanced Feature Engineering**: 50+ technical and statistical features
- **Interactive Dashboard**: Real-time visualization and analysis with Plotly Dash
- **Backtesting**: Historical performance evaluation
- **Command-line Interface**: Easy-to-use CLI for predictions

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd PredictiveStockAnalysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Note: For TA-Lib, you may need to install system-level dependencies first:
- macOS: `brew install ta-lib`
- Ubuntu: `sudo apt-get install ta-lib`
- Windows: Download from [TA-Lib website](http://ta-lib.org)

## Usage

### Command Line Interface

#### Run Stock Prediction
```bash
python main.py predict AAPL --model all --save
```

Options:
- `--start-date YYYY-MM-DD`: Start date for historical data
- `--end-date YYYY-MM-DD`: End date for historical data
- `--model {linear|random_forest|gradient_boosting|lstm|ensemble|all}`: Model to use
- `--save`: Save results to CSV file

#### Launch Interactive Dashboard
```bash
python main.py dashboard
```
Then open http://localhost:8050 in your browser.

#### Train Models
```bash
python main.py train AAPL --save-model
```

#### Run Backtesting
```bash
python main.py backtest AAPL --strategy momentum
```

### Python API

```python
from stocks import StockDataCollector
from feature_engineering import FeatureEngineer
from models import StockPredictionModels

# Collect data
collector = StockDataCollector('AAPL', start_date='2022-01-01')
data = collector.prepare_features()

# Engineer features
engineer = FeatureEngineer()
result = engineer.engineer_all_features(data)

# Train models
trainer = StockPredictionModels()
trainer.train_random_forest(X_train, y_train, X_val, y_val)

# Make predictions
predictions = trainer.predict('random_forest', X_test)
```

## Project Structure

```
PredictiveStockAnalysis/
├── stocks.py              # Stock data collection and technical indicators
├── feature_engineering.py # Advanced feature engineering
├── models.py              # Machine learning models
├── visualization.py       # Plotting and visualization
├── dashboard.py          # Interactive Dash application
├── config.py             # Configuration settings
├── main.py               # Main entry point and CLI
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Models

### 1. Linear Regression
- Simple baseline model
- Fast training and inference
- Good for linear relationships

### 2. Random Forest
- Ensemble of decision trees
- Handles non-linear patterns
- Feature importance analysis

### 3. Gradient Boosting
- Sequential ensemble learning
- High accuracy
- Robust to overfitting

### 4. LSTM Neural Network
- Deep learning model
- Captures temporal dependencies
- Best for sequential patterns

### 5. Ensemble Model
- Combines multiple models
- Weighted averaging
- Improved robustness

## Features Generated

The system generates 50+ features including:
- **Price Features**: OHLC ratios, gaps, price position
- **Technical Indicators**: SMA, EMA, RSI, MACD, Bollinger Bands
- **Volume Indicators**: OBV, MFI, Volume ratios
- **Statistical Features**: Rolling statistics, Z-scores, correlations
- **Pattern Recognition**: Candlestick patterns (when TA-Lib available)

## Configuration

Edit `config.py` to customize:
- Data collection settings
- Model hyperparameters
- Feature engineering options
- Dashboard settings
- Backtesting parameters

## Performance Metrics

The system evaluates models using:
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² Score
- Directional Accuracy
- Sharpe Ratio (for backtesting)

## Troubleshooting

### ModuleNotFoundError
Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### TA-Lib Installation Issues
TA-Lib requires system-level installation. If you can't install it, the system will still work with basic pattern detection.

### Memory Issues
For large datasets, consider:
- Reducing the date range
- Using fewer features
- Decreasing model complexity

## Contributing

Contributions are welcome! Please feel free to submit pull requests.

## Disclaimer

This system is for educational and research purposes only. Stock market prediction is inherently uncertain, and this tool should not be used as the sole basis for investment decisions. Always consult with financial professionals and do your own research.

## License

MIT License - See LICENSE file for details