# Predictive Stock Analysis System

A comprehensive machine learning system for stock price prediction and analysis using multiple models including Linear Regression, Random Forest, Gradient Boosting, and LSTM neural networks.

> **Notebook-first workflow**  
> Every former `.py` module is now a Jupyter notebook (`*.ipynb`). Launch them in Jupyter Lab/Notebook or VS Code to run cells instead of invoking scripts from the command line.

## Features

- **Multiple ML Models**: Linear Regression, Random Forest, Gradient Boosting, LSTM, and Ensemble methods
- **Technical Indicators**: RSI, MACD, Bollinger Bands, Moving Averages, and more
- **Advanced Feature Engineering**: 50+ technical and statistical features
- **Interactive Dashboard**: Real-time visualization and analysis with Plotly Dash
- **Backtesting**: Historical performance evaluation
- **Notebook-first Workflow**: Run end-to-end experiments entirely inside Jupyter notebooks

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

### Launch the notebooks

1. Install dependencies (see [Installation](#installation)).
2. Start Jupyter Lab/Notebook from the repo root:
   ```bash
   jupyter lab
   # or
   jupyter notebook
   ```
3. Open the notebook that matches the workflow you need:
   - `main.ipynb`: end-to-end pipeline (data load → feature engineering → training → inference)
   - `dashboard.ipynb`: Dash/Plotly dashboard utilities
   - `feature_engineering.ipynb`: feature generation experiments
   - `models.ipynb`: individual model training/evaluation
   - `visualization.ipynb`: ad-hoc plotting
   - `stocks.ipynb`: data download and indicator calculations
   - `config.ipynb`: tweak configuration objects
   - `test_structure.ipynb`: structural/unit test scaffolding

Execute cells sequentially to reproduce the original CLI behavior. Key entry points that were previously exposed as `python main.py <command>` are annotated within `main.ipynb`.

### Python API

All reusable classes/functions remain accessible by importing directly from within a notebook cell, e.g.:

```python
from stocks import StockDataCollector
from feature_engineering import FeatureEngineer
from models import StockPredictionModels
```

## Project Structure

```
PredictiveStockAnalysis/
├── stocks.ipynb              # Stock data collection and technical indicators
├── feature_engineering.ipynb # Advanced feature engineering
├── models.ipynb              # Machine learning models
├── visualization.ipynb       # Plotting and visualization
├── dashboard.ipynb           # Interactive Dash application
├── config.ipynb              # Configuration settings
├── main.ipynb                # Main entry point / pipeline driver
├── test_structure.ipynb      # Structural/unit test harness
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

Open `config.ipynb` (or import `config` inside another notebook) to customize:
- Data collection settings
- Model hyperparameters
- Feature engineering options
- Dashboard settings
- Backtesting parameters

## MCP Integration

Codex CLI discovers MCP servers via `~/.config/codex/config.toml`. A matching template is kept in `.codex/config.toml`; copy or merge it into the global config if you need other MCP entries.

1. Ensure the global config contains the `predictive_stock_server` entry (see `.codex/config.toml` for reference). The repo path is already encoded via the `cwd` property.
2. Install Node.js/npm so `npx` can download and run the `mcp-server` package.
3. Export any required secrets (e.g., `ALPHA_VANTAGE_API_KEY`, `OPENAI_API_KEY`) before starting Codex. They are whitelisted in the MCP config and will be forwarded to the server.
4. Launch Codex CLI inside this repository. In the MCP menu, select `predictive_stock_server` and Codex will execute `npx -y mcp-server` here.
5. Once connected, the MCP server can surface repository-specific tools and data to the client.

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
