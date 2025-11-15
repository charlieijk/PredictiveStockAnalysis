# stocks.py

"""Utilities for downloading stock data and creating technical indicators."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging
import os

import numpy as np
import pandas as pd
import yfinance as yf

from feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)


@dataclass
class StockDataCollector:
    """Download OHLCV data and enrich it with technical indicators."""

    symbol: str
    start_date: str = "2020-01-01"
    end_date: Optional[str] = None
    interval: str = "1d"
    data: Optional[pd.DataFrame] = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.symbol = self.symbol.upper()
        if not self.start_date:
            self.start_date = "2020-01-01"
        if not self.end_date:
            self.end_date = datetime.now().strftime("%Y-%m-%d")

    # ------------------------------------------------------------------
    # Data fetching helpers
    # ------------------------------------------------------------------
    def fetch_stock_data(self, force: bool = False) -> pd.DataFrame:
        """Download historical OHLCV data from Yahoo Finance."""
        if self.data is not None and not force:
            return self.data.copy()

        logger.info(
            "Fetching data for %s from %s to %s (interval=%s)",
            self.symbol,
            self.start_date,
            self.end_date,
            self.interval,
        )

        df = yf.download(
            self.symbol,
            start=self.start_date,
            end=self.end_date,
            interval=self.interval,
            auto_adjust=True,
            progress=False,
        )

        if df.empty:
            raise ValueError(
                f"No data returned for {self.symbol} between {self.start_date} and {self.end_date}"
            )

        # yfinance sometimes returns a MultiIndex column (field, ticker) even for
        # single symbols. Flatten so downstream code always receives 1-D Series.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

        df.reset_index(inplace=True)
        df.rename(columns=str.title, inplace=True)
        df["Returns"] = df["Close"].pct_change()
        df["Log_Returns"] = np.log1p(df["Returns"].fillna(0.0))

        self.data = df
        logger.info("Successfully fetched %d rows", len(df))
        return self.data.copy()

    def fetch_raw_data(self, force: bool = False) -> pd.DataFrame:
        """Backward compatible alias used by existing notebooks."""
        return self.fetch_stock_data(force=force)

    def _require_data(self) -> pd.DataFrame:
        if self.data is None:
            raise ValueError("No data loaded. Call fetch_stock_data() first.")
        return self.data

    # ------------------------------------------------------------------
    # Indicator calculations
    # ------------------------------------------------------------------
    def calculate_moving_averages(self, windows: Optional[List[int]] = None) -> pd.DataFrame:
        data = self._require_data()
        windows = windows or [10, 20, 50, 200]
        for window in windows:
            data[f"SMA_{window}"] = data["Close"].rolling(window=window).mean()
            data[f"EMA_{window}"] = data["Close"].ewm(span=window, adjust=False).mean()
        logger.info("Calculated moving averages for windows: %s", windows)
        return data

    def calculate_rsi(self, period: int = 14) -> pd.DataFrame:
        data = self._require_data()
        delta = data["Close"].diff()
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        avg_gain = pd.Series(gain, index=data.index).rolling(period).mean()
        avg_loss = pd.Series(loss, index=data.index).rolling(period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        data[f"RSI_{period}"] = 100 - (100 / (1 + rs))
        logger.info("Calculated RSI_%d", period)
        return data

    def calculate_macd(
        self, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> pd.DataFrame:
        data = self._require_data()
        ema_fast = data["Close"].ewm(span=fast, adjust=False).mean()
        ema_slow = data["Close"].ewm(span=slow, adjust=False).mean()
        data["MACD"] = ema_fast - ema_slow
        data["MACD_Signal"] = data["MACD"].ewm(span=signal, adjust=False).mean()
        data["MACD_Histogram"] = data["MACD"] - data["MACD_Signal"]
        logger.info("Calculated MACD (%d, %d, %d)", fast, slow, signal)
        return data

    def calculate_bollinger_bands(self, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
        data = self._require_data()
        rolling = data["Close"].rolling(window=window)
        mean = rolling.mean()
        std = rolling.std()
        data["BB_Middle"] = mean
        data["BB_Upper"] = mean + num_std * std
        data["BB_Lower"] = mean - num_std * std
        data["BB_Width"] = (data["BB_Upper"] - data["BB_Lower"]) / data["BB_Middle"]
        data["BB_Position"] = (data["Close"] - data["BB_Lower"]) / (
            data["BB_Upper"] - data["BB_Lower"]
        )
        logger.info("Calculated Bollinger Bands (window=%d, std=%.1f)", window, num_std)
        return data

    def calculate_volatility(self, window: int = 20) -> pd.DataFrame:
        data = self._require_data()
        data["Returns"] = data["Returns"].fillna(0.0)
        data[f"Volatility_{window}"] = data["Returns"].rolling(window).std() * np.sqrt(252)
        data["ATR"] = (
            (data["High"] - data["Low"]).rolling(window).mean()
        )
        logger.info("Calculated volatility metrics (window=%d)", window)
        return data

    def calculate_volume_indicators(self, window: int = 20) -> pd.DataFrame:
        data = self._require_data()
        data[f"Volume_MA_{window}"] = data["Volume"].rolling(window).mean()
        obv = [0]
        for i in range(1, len(data)):
            if data["Close"].iloc[i] > data["Close"].iloc[i - 1]:
                obv.append(obv[-1] + data["Volume"].iloc[i])
            elif data["Close"].iloc[i] < data["Close"].iloc[i - 1]:
                obv.append(obv[-1] - data["Volume"].iloc[i])
            else:
                obv.append(obv[-1])
        data["OBV"] = obv
        data["Volume_ROC"] = data["Volume"].pct_change(periods=10) * 100
        logger.info("Calculated volume indicators")
        return data

    def add_price_features(self, lags: Optional[List[int]] = None) -> pd.DataFrame:
        data = self._require_data()
        lags = lags or [1, 3, 5, 10]
        data["Price_Change"] = data["Close"] - data["Open"]
        data["High_Low_Ratio"] = data["High"] / data["Low"]
        data["Close_Open_Ratio"] = data["Close"] / data["Open"]
        for lag in lags:
            data[f"Close_Lag_{lag}"] = data["Close"].shift(lag)
            data[f"Returns_Lag_{lag}"] = data["Returns"].shift(lag)
        logger.info("Added price/lag features with lags=%s", lags)
        return data

    # ------------------------------------------------------------------
    # Aggregated workflows
    # ------------------------------------------------------------------
    def prepare_features(self) -> pd.DataFrame:
        """Compute all indicators and a default next-day return target."""
        data = self.fetch_stock_data()
        self.calculate_moving_averages()
        self.calculate_rsi()
        self.calculate_macd()
        self.calculate_bollinger_bands()
        self.calculate_volatility()
        self.calculate_volume_indicators()
        self.add_price_features()
        data = self._require_data()
        data["Target"] = data["Close"].shift(-1) / data["Close"] - 1
        data.dropna(inplace=True)
        logger.info("Feature preparation complete. Shape: %s", data.shape)
        return data.copy()

    def get_engineered_features(
        self,
        engineer: Optional[FeatureEngineer] = None,
        **engineer_kwargs: Any,
    ) -> Dict[str, Any]:
        """Run the advanced FeatureEngineer on top of the processed dataframe."""
        if engineer is None:
            engineer = FeatureEngineer()
        base_df = self.prepare_features()
        return engineer.engineer_all_features(base_df, **engineer_kwargs)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def save_to_csv(self, output_dir: str = "data") -> str:
        data = self._require_data()
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(output_dir, f"{self.symbol}_stock_data_{timestamp}.csv")
        data.to_csv(path, index=False)
        logger.info("Data saved to %s", path)
        return path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    collector = StockDataCollector("AAPL", start_date="2021-01-01")
    df = collector.prepare_features()
    print("Prepared dataset shape:", df.shape)
    engineered = collector.get_engineered_features()
    print("Engineered feature matrix:", engineered["features"].shape)
