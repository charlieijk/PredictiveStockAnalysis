"""
Configuration file for Stock Price Prediction Project
Contains all hyperparameters, settings, and constants
"""

import os
from datetime import datetime, timedelta

# Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
LOG_DIR = os.path.join(BASE_DIR, 'logs')

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODEL_DIR, OUTPUT_DIR, LOG_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Data collection settings
DATA_CONFIG = {
    'default_symbols': ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA'],
    'start_date': (datetime.now() - timedelta(days=1095)).strftime('%Y-%m-%d'),  # 3 years
    'end_date': datetime.now().strftime('%Y-%m-%d'),
    'test_size': 0.2,
    'validation_size': 0.1,
    'sequence_length': 60,  # Days to look back for LSTM
}

# Technical indicators settings
INDICATORS_CONFIG = {
    'moving_averages': {
        'windows': [5, 10, 20, 50, 100, 200]
    },
    'rsi': {
        'period': 14
    },
    'macd': {
        'fast': 12,
        'slow': 26,
        'signal': 9
    },
    'bollinger_bands': {
        'window': 20,
        'num_std': 2
    },
    'volatility': {
        'window': 20
    }
}

# Feature engineering settings
FEATURE_CONFIG = {
    'scaling_method': 'robust',  # 'standard', 'minmax', 'robust'
    'feature_selection': {
        'method': 'mutual_info',  # 'correlation', 'mutual_info', 'rfe'
        'top_k_features': 30
    },
    'lag_features': [1, 2, 3, 5, 7, 10, 15, 20],
    'rolling_windows': [5, 10, 20, 30],
    'target_days_ahead': 1,  # Predict 1 day ahead
}

# Model configurations
MODEL_CONFIG = {
    'linear_regression': {
        'normalize': True,
        'fit_intercept': True
    },
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 15,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'random_state': 42,
        'n_jobs': -1
    },
    'gradient_boosting': {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 5,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'subsample': 0.8,
        'random_state': 42
    },
    'lstm': {
        'units': [128, 64, 32],  # Units for each LSTM layer
        'dropout': 0.2,
        'recurrent_dropout': 0.2,
        'dense_units': [32, 16],  # Units for dense layers after LSTM
        'activation': 'tanh',
        'optimizer': 'adam',
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 100,
        'early_stopping_patience': 15,
        'reduce_lr_patience': 10,
        'reduce_lr_factor': 0.5,
        'min_lr': 0.00001
    },
    'ensemble': {
        'voting': 'soft',  # 'hard' or 'soft'
        'weights': None  # Will be optimized based on validation performance
    },
    'asymmetric_world_model': {
        'sequence_length': 16,
        'backward_hidden_dim': 96,
        'backward_layers': 2,
        'forward_hidden_dims': [128, 64],
        'bottleneck_dim': 8,
        'dropout': 0.1,
        'lr': 1e-3,
        'weight_decay': 1e-5,
        'reconstruction_weight': 1.0,
        'prediction_weight': 1.0,
        'next_state_weight': 0.5,
        'batch_size': 64,
        'epochs': 50,
        'early_stopping_patience': 7,
        'train_fraction': 0.7,
        'val_fraction': 0.15,
        'activation_logging_batches': 6,
        'perturbation_std': 0.2,
        'clip_grad_norm': 1.0,
        'log_dir': LOG_DIR
    }
}

# Training settings
TRAINING_CONFIG = {
    'cross_validation': {
        'enabled': True,
        'n_splits': 5,
        'shuffle': False  # Keep temporal order for time series
    },
    'hyperparameter_tuning': {
        'enabled': False,
        'method': 'grid',  # 'grid', 'random', 'bayesian'
        'n_iter': 50,  # For random search
        'cv_splits': 3
    },
    'metrics': ['mse', 'rmse', 'mae', 'mape', 'r2', 'directional_accuracy'],
    'save_best_only': True,
    'verbose': 1,
    'train_lstm': False,
    'train_asymmetric_world_model': True
}

# Backtesting settings
BACKTESTING_CONFIG = {
    'initial_capital': 100000,
    'position_size': 0.95,  # Use 95% of available capital
    'stop_loss': 0.02,  # 2% stop loss
    'take_profit': 0.05,  # 5% take profit
    'commission': 0.001,  # 0.1% commission
    'slippage': 0.001,  # 0.1% slippage
    'rebalance_frequency': 'daily',  # 'daily', 'weekly', 'monthly'
    'benchmark_symbol': 'SPY'  # S&P 500 ETF as benchmark
}

# Visualization settings
VISUALIZATION_CONFIG = {
    'figure_size': (15, 8),
    'style': 'seaborn-v0_8-darkgrid',
    'color_palette': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
    'save_plots': True,
    'plot_format': 'png',
    'dpi': 100
}

# Dashboard settings
DASHBOARD_CONFIG = {
    'host': '127.0.0.1',
    'port': 8050,
    'debug': True,
    'auto_refresh_interval': 60,  # Seconds
    'cache_timeout': 300  # Seconds
}

# Logging settings
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'date_format': '%Y-%m-%d %H:%M:%S',
    'log_file': os.path.join(LOG_DIR, f'stock_prediction_{datetime.now().strftime("%Y%m%d")}.log')
}

# API Keys (use environment variables in production)
API_KEYS = {
    'alpha_vantage': os.getenv('ALPHA_VANTAGE_API_KEY', ''),
    'finnhub': os.getenv('FINNHUB_API_KEY', ''),
    'news_api': os.getenv('NEWS_API_KEY', '')
}

# Performance thresholds
PERFORMANCE_THRESHOLDS = {
    'min_accuracy': 0.55,  # Minimum directional accuracy
    'max_drawdown': 0.15,  # Maximum acceptable drawdown
    'min_sharpe_ratio': 1.0,  # Minimum Sharpe ratio
    'min_profit_factor': 1.2  # Minimum profit factor
}

# Feature lists (for reference and selection)
TECHNICAL_FEATURES = [
    'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50', 'SMA_100', 'SMA_200',
    'EMA_5', 'EMA_10', 'EMA_20', 'EMA_50', 'EMA_100', 'EMA_200',
    'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Histogram',
    'BB_Upper', 'BB_Middle', 'BB_Lower', 'BB_Width', 'BB_Position',
    'Volatility_20', 'Volume_MA_20', 'OBV', 'Volume_ROC'
]

PRICE_FEATURES = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'Price_Change', 'High_Low_Ratio', 'Close_Open_Ratio',
    'Returns'
]

# Model selection criteria
MODEL_SELECTION_CRITERIA = {
    'primary_metric': 'directional_accuracy',  # Main metric for model selection
    'secondary_metrics': ['rmse', 'sharpe_ratio'],
    'weight_primary': 0.5,
    'weight_secondary': 0.5
}

# Real-time prediction settings
REALTIME_CONFIG = {
    'update_frequency': 60,  # Seconds
    'pre_market_start': '04:00:00',
    'market_open': '09:30:00',
    'market_close': '16:00:00',
    'after_hours_end': '20:00:00',
    'timezone': 'US/Eastern',
    'enable_paper_trading': True
}

# Risk management settings
RISK_CONFIG = {
    'max_position_size': 0.2,  # Maximum 20% in single position
    'max_sector_exposure': 0.4,  # Maximum 40% in single sector
    'max_correlation': 0.7,  # Maximum correlation between positions
    'var_confidence': 0.95,  # Value at Risk confidence level
    'risk_free_rate': 0.04  # Current risk-free rate (4%)
}

# Database settings (for storing predictions and performance)
DATABASE_CONFIG = {
    'db_type': 'sqlite',  # 'sqlite', 'postgresql', 'mysql'
    'db_name': 'stock_predictions.db',
    'db_path': os.path.join(DATA_DIR, 'stock_predictions.db'),
    'table_predictions': 'predictions',
    'table_performance': 'model_performance',
    'table_trades': 'trades'
}

# Notification settings
NOTIFICATION_CONFIG = {
    'enable_email': False,
    'email_recipient': '',
    'email_sender': '',
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'alert_on_prediction': True,
    'alert_on_error': True,
    'alert_threshold': 0.03  # Alert if predicted move > 3%
}

# System settings
SYSTEM_CONFIG = {
    'random_seed': 42,
    'n_jobs': -1,  # Use all CPU cores
    'memory_limit': '4GB',
    'gpu_enabled': False,
    'cache_enabled': True,
    'profiling_enabled': False
}
