"""
Real-Time Trading API with FastAPI

Provides REST API endpoints for:
- Live predictions
- Model serving
- Portfolio management
- Real-time data streaming
- Backtesting
- Risk monitoring
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import joblib
import logging
from pathlib import Path
import asyncio

# Add parent directory to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from stocks import StockDataCollector
from feature_engineering import FeatureEngineer
from models import StockPredictionModels
from portfolio_optimization import MeanVarianceOptimizer, RiskParityOptimizer
from risk_management import VaRCalculator, CVaRCalculator, RiskMonitor, RiskLimits
from backtesting import BacktestEngine, BacktestConfig

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Stock Prediction Trading API",
    description="Real-time stock prediction and trading system API",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
models_cache: Dict[str, Any] = {}
predictions_cache: Dict[str, Dict] = {}


# Pydantic models for request/response validation
class PredictionRequest(BaseModel):
    symbol: str = Field(..., description="Stock symbol (e.g., AAPL)")
    model_type: str = Field("ensemble", description="Model type: linear, rf, gb, lstm, ensemble")
    lookback_days: int = Field(60, ge=1, le=500, description="Days of historical data")


class PredictionResponse(BaseModel):
    symbol: str
    prediction: float
    confidence: Optional[float] = None
    timestamp: datetime
    model_type: str
    features_used: int


class BacktestRequest(BaseModel):
    symbol: str
    start_date: str
    end_date: str
    initial_capital: float = 100000.0
    strategy: str = "sma_crossover"  # sma_crossover, rsi, etc.


class PortfolioOptimizationRequest(BaseModel):
    symbols: List[str]
    method: str = "max_sharpe"  # max_sharpe, min_variance, risk_parity
    lookback_days: int = 252


class RiskAnalysisRequest(BaseModel):
    portfolio_value: float
    positions: Dict[str, float]  # {symbol: value}
    confidence_level: float = 0.95


# Helper functions
def load_model(symbol: str, model_type: str):
    """Load trained model from cache or disk."""
    cache_key = f"{symbol}_{model_type}"

    if cache_key in models_cache:
        return models_cache[cache_key]

    models_dir = Path("models")
    model_files = {
        "linear": f"{symbol}_linear_regression",
        "rf": f"{symbol}_random_forest",
        "gb": f"{symbol}_gradient_boosting",
        "lstm": f"{symbol}_lstm",
        "ensemble": f"{symbol}_ensemble"
    }

    model_file = models_dir / f"{model_files.get(model_type)}.pkl"

    if not model_file.exists() and model_type == "lstm":
        model_file = models_dir / f"{model_files.get(model_type)}.keras"

    if not model_file.exists():
        raise FileNotFoundError(f"Model not found: {model_file}")

    # Load model
    if model_type == "lstm":
        from tensorflow import keras
        model = keras.models.load_model(model_file)
    else:
        model = joblib.load(model_file)

    models_cache[cache_key] = model
    logger.info(f"Loaded model: {cache_key}")

    return model


async def fetch_and_prepare_data(symbol: str, lookback_days: int = 60):
    """Fetch and prepare data for prediction."""
    # Fetch data
    collector = StockDataCollector(symbol=symbol)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days * 2)  # Extra buffer

    df = collector.fetch_stock_data(
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d")
    )

    # Feature engineering
    df = collector.prepare_features(df)
    engineer = FeatureEngineer(df)
    df = engineer.engineer_all_features()

    # Feature selection
    X, y, selected_features = engineer.select_features(k=30)

    return X, y, selected_features, df


# API Endpoints

@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Stock Prediction Trading API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "backtest": "/backtest",
            "optimize_portfolio": "/portfolio/optimize",
            "risk_analysis": "/risk/analyze",
            "live_data": "/ws/live/{symbol}"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": len(models_cache)
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Generate stock price prediction.

    Returns prediction for next trading day.
    """
    try:
        symbol = request.symbol.upper()
        model_type = request.model_type.lower()

        logger.info(f"Prediction request: {symbol} using {model_type}")

        # Fetch and prepare data
        X, y, selected_features, df = await fetch_and_prepare_data(
            symbol, request.lookback_days
        )

        # Load model
        model = load_model(symbol, model_type)

        # Make prediction
        if model_type == "lstm":
            # LSTM needs 3D input
            X_last = X[-1:].reshape(1, 1, -1)
            prediction = model.predict(X_last, verbose=0)[0][0]
        else:
            X_last = X[-1:]
            prediction = model.predict(X_last)[0]

        # Calculate confidence (using prediction variance if ensemble)
        confidence = None
        if model_type == "ensemble" and hasattr(model, 'estimators_'):
            predictions = [est.predict(X_last)[0] for est in model.estimators_]
            confidence = 1.0 - (np.std(predictions) / np.mean(predictions))

        response = PredictionResponse(
            symbol=symbol,
            prediction=float(prediction),
            confidence=confidence,
            timestamp=datetime.now(),
            model_type=model_type,
            features_used=len(selected_features)
        )

        # Cache prediction
        predictions_cache[symbol] = response.dict()

        return response

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/backtest")
async def run_backtest(request: BacktestRequest):
    """
    Run backtest on historical data.

    Tests trading strategy performance.
    """
    try:
        symbol = request.symbol.upper()

        # Fetch historical data
        collector = StockDataCollector(symbol=symbol)
        df = collector.fetch_stock_data(
            start_date=request.start_date,
            end_date=request.end_date
        )

        # Generate signals based on strategy
        if request.strategy == "sma_crossover":
            df['SMA_50'] = df['Close'].rolling(50).mean()
            df['SMA_200'] = df['Close'].rolling(200).mean()
            signals = pd.Series(0, index=df.index)
            signals[df['SMA_50'] > df['SMA_200']] = 1
            signals[df['SMA_50'] < df['SMA_200']] = -1

        elif request.strategy == "rsi":
            # Calculate RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            signals = pd.Series(0, index=df.index)
            signals[rsi < 30] = 1  # Oversold - buy
            signals[rsi > 70] = -1  # Overbought - sell

        else:
            raise HTTPException(status_code=400, detail=f"Unknown strategy: {request.strategy}")

        # Run backtest
        config = BacktestConfig(initial_capital=request.initial_capital)
        engine = BacktestEngine(config)
        metrics, equity_curve = engine.run(df, signals)

        return {
            "symbol": symbol,
            "strategy": request.strategy,
            "period": {
                "start": request.start_date,
                "end": request.end_date
            },
            "metrics": metrics.to_dict(),
            "trades": len(engine.trades),
            "final_value": equity_curve['equity'].iloc[-1]
        }

    except Exception as e:
        logger.error(f"Backtest error: {e}")
        raise HTTPException(status_code=500, detail=f"Backtest failed: {str(e)}")


@app.post("/portfolio/optimize")
async def optimize_portfolio(request: PortfolioOptimizationRequest):
    """
    Optimize portfolio allocation.

    Returns optimal weights for given symbols.
    """
    try:
        # Fetch returns for all symbols
        returns_dict = {}

        for symbol in request.symbols:
            collector = StockDataCollector(symbol=symbol)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=request.lookback_days)

            df = collector.fetch_stock_data(
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d")
            )

            returns_dict[symbol] = df['Close'].pct_change().dropna()

        # Combine into DataFrame
        returns_df = pd.DataFrame(returns_dict)
        returns_df = returns_df.dropna()

        # Optimize based on method
        if request.method == "max_sharpe":
            optimizer = MeanVarianceOptimizer(returns_df)
            weights, stats = optimizer.optimize_max_sharpe()

        elif request.method == "min_variance":
            optimizer = MeanVarianceOptimizer(returns_df)
            weights, stats = optimizer.optimize_min_variance()

        elif request.method == "risk_parity":
            optimizer = RiskParityOptimizer(returns_df)
            weights, stats = optimizer.optimize()

        else:
            raise HTTPException(status_code=400, detail=f"Unknown method: {request.method}")

        # Format response
        portfolio = {
            symbol: float(weight)
            for symbol, weight in zip(request.symbols, weights)
        }

        return {
            "method": request.method,
            "weights": portfolio,
            "expected_return": stats['return'],
            "volatility": stats['volatility'],
            "sharpe_ratio": stats['sharpe_ratio']
        }

    except Exception as e:
        logger.error(f"Portfolio optimization error: {e}")
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")


@app.post("/risk/analyze")
async def analyze_risk(request: RiskAnalysisRequest):
    """
    Analyze portfolio risk.

    Calculates VaR, CVaR, and other risk metrics.
    """
    try:
        # Fetch returns for all positions
        returns_dict = {}

        for symbol in request.positions.keys():
            collector = StockDataCollector(symbol=symbol)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=252)  # 1 year

            df = collector.fetch_stock_data(
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d")
            )

            returns_dict[symbol] = df['Close'].pct_change().dropna()

        returns_df = pd.DataFrame(returns_dict).dropna()

        # Calculate portfolio returns
        weights = np.array([
            request.positions[symbol] / request.portfolio_value
            for symbol in returns_df.columns
        ])

        portfolio_returns = (returns_df * weights).sum(axis=1)

        # Risk calculations
        var_calc = VaRCalculator(confidence_level=request.confidence_level)
        cvar_calc = CVaRCalculator(confidence_level=request.confidence_level)

        var_result = var_calc.historical_var(
            portfolio_returns,
            portfolio_value=request.portfolio_value
        )

        cvar_result = cvar_calc.calculate_cvar(
            portfolio_returns,
            portfolio_value=request.portfolio_value
        )

        # Additional metrics
        annual_vol = portfolio_returns.std() * np.sqrt(252)

        return {
            "portfolio_value": request.portfolio_value,
            "var": var_result,
            "cvar": cvar_result,
            "annual_volatility": float(annual_vol),
            "positions": request.positions,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Risk analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Risk analysis failed: {str(e)}")


# WebSocket endpoint for live data streaming
@app.websocket("/ws/live/{symbol}")
async def websocket_live_data(websocket: WebSocket, symbol: str):
    """
    WebSocket endpoint for streaming live predictions.

    Sends predictions every N seconds.
    """
    await websocket.accept()

    try:
        while True:
            # Fetch latest data and make prediction
            try:
                X, y, selected_features, df = await fetch_and_prepare_data(symbol, lookback_days=60)
                model = load_model(symbol, "ensemble")

                # Predict
                X_last = X[-1:]
                prediction = model.predict(X_last)[0]

                # Send update
                await websocket.send_json({
                    "symbol": symbol,
                    "prediction": float(prediction),
                    "current_price": float(df['Close'].iloc[-1]),
                    "timestamp": datetime.now().isoformat()
                })

            except Exception as e:
                await websocket.send_json({
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })

            # Wait before next update
            await asyncio.sleep(60)  # Update every minute

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for {symbol}")


# Background tasks
@app.on_event("startup")
async def startup_event():
    """Run on API startup."""
    logger.info("Trading API starting up...")

    # Pre-load common models
    common_symbols = ["AAPL", "GOOGL", "MSFT"]
    for symbol in common_symbols:
        try:
            load_model(symbol, "ensemble")
            logger.info(f"Pre-loaded model for {symbol}")
        except Exception as e:
            logger.warning(f"Could not pre-load {symbol}: {e}")

    logger.info("Trading API ready!")


@app.on_event("shutdown")
async def shutdown_event():
    """Run on API shutdown."""
    logger.info("Trading API shutting down...")
    models_cache.clear()
    predictions_cache.clear()


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)

    uvicorn.run(
        "trading_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
