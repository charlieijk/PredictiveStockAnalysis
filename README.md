# Predictive Stock Analysis System

A comprehensive machine learning suite for stock price prediction and analysis using Linear Regression, Random Forest, Gradient Boosting, LSTM neural networks, and a PyTorch asymmetric world model.

> **Dual workflow**  
> Automate runs through the CLI pipeline or explore interactively in the companion notebooks. Both paths share the same modules, so results stay consistent.

## Features

- **Multiple ML Models**: Linear Regression, Random Forest, Gradient Boosting, LSTM, and Ensemble methods
- **Technical Indicators**: RSI, MACD, Bollinger Bands, Moving Averages, and more
- **Advanced Feature Engineering**: 50+ technical and statistical features
- **Interactive Dashboard**: Real-time visualization and analysis with Plotly Dash
- **Backtesting**: Historical performance evaluation
- **Dual Workflow**: Automate via the CLI pipeline or run experiments in notebooks
- **Asymmetric World Model**: Backward GRU + FiLM-modulated forward predictor with neuron diagnostics

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

### End-to-end CLI pipeline

1. Install dependencies.
2. Run the orchestrator:
   ```bash
   python main.py AAPL --dashboard
   ```
   Key arguments:
   - `symbol` (required): ticker to process.
   - `--start-date/--end-date` (optional): override the 3-year default window.
   - `--dashboard`: automatically launch the Dash UI after training.
3. Outputs: raw/engineered datasets in `data/`, trained models in `models/`, evaluation reports/plots in `output/`, and logs in `logs/`.

### Standalone dashboard

Spin up the Plotly Dash interface with the latest saved artifacts:

```bash
python dashboard.py
```
The app listens on `http://127.0.0.1:8050` by default (see `config.DASHBOARD_CONFIG` to customize port/host).

### Notebook workflow

Prefer notebooks for experimentation? Launch Jupyter from the repo root:

```bash
jupyter lab
# or
jupyter notebook
```

Open the notebook that fits your task (`main.ipynb`, `feature_engineering.ipynb`, `models.ipynb`, etc.) and execute cells sequentially. Notebooks import the same underlying modules (`stocks.py`, `feature_engineering.py`, `models.py`, `asymmetric_world_model.py`, …), so the computations mirror CLI runs. Every module remains importable inside a cell:

```python
from stocks import StockDataCollector
from feature_engineering import FeatureEngineer
from models import StockPredictionModels
```

### Run the test suite

Unit tests validate the config scaffolding, feature engineering utilities, and asymmetric world model shim:

```bash
pytest
```

## Project Structure

```
PredictiveStockAnalysis/
├── main.py                     # CLI pipeline orchestrator
├── stocks.py                   # Data collection + indicator calculations
├── feature_engineering.py      # Feature engineering utilities
├── models.py                   # Sklearn/TensorFlow trainers + ensembling
├── asymmetric_world_model.py   # PyTorch asymmetric world model + trainer
├── visualization.py            # Plotting helpers
├── dashboard.py                # Plotly Dash UI
├── config.py                   # Global configuration/state directories
├── api/                        # FastAPI service skeleton
├── tests/                      # Pytest-based regression tests
├── data/, models/, output/, logs/   # Runtime artifacts (auto-created)
├── notebooks (*.ipynb)         # Interactive equivalents of the modules above
├── requirements.txt            # Python dependencies
├── requirements_space.txt      # Minimal set for HF Spaces
├── scripts/                    # Developer tooling (MCP installer, etc.)
└── README.md                   # This document
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

### 6. Asymmetric World Model (PyTorch)
- Backward GRU reconstructs past states, forward FiLM network predicts next state/return
- Includes narrow bottleneck (8 units) with log-normal initialisation
- Logs neuron activation variances and perturbation-based feature importance
- Enable/disable via `TRAINING_CONFIG['train_asymmetric_world_model']`

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
- Asymmetric world-model settings (`MODEL_CONFIG['asymmetric_world_model']`)

## MCP Integration

Codex CLI discovers MCP servers via `~/.config/codex/config.toml`. A matching template is kept in `.codex/config.toml`; copy or merge it into the global config if you need other MCP entries.

1. Run the installer to copy the MCP entry into your global Codex config (this injects the correct absolute path for your clone):
   ```bash
   python scripts/install_mcp_config.py
   # Use --force if you want to overwrite an existing entry.
   ```
   (If you prefer manual edits, copy `.codex/config.toml` into `~/.config/codex/config.toml` and replace `__PROJECT_ROOT__` with the absolute path to this repository.)
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
