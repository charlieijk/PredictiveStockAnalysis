"""
Stock Data Collection Module
Handles fetching historical stock data and calculating technical indicators
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Tuple, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StockDataCollector:
    """Handles stock data collection and technical indicator calculations"""
    
    def __init__(self, symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None):
        """
        Initialize the data collector
        
        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
            start_date: Start date for data collection (YYYY-MM-DD format)
            end_date: End date for data collection (YYYY-MM-DD format)
        """
        self.symbol = symbol.upper()
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.start_date = start_date or (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
        self.data = None
        
    def fetch_stock_data(self) -> pd.DataFrame:
        """
        Fetch historical stock data from Yahoo Finance
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            logger.info(f"Fetching data for {self.symbol} from {self.start_date} to {self.end_date}")
            ticker = yf.Ticker(self.symbol)
            self.data = ticker.history(start=self.start_date, end=self.end_date)
            
            if self.data.empty:
                raise ValueError(f"No data found for symbol {self.symbol}")
            
            # Reset index to have Date as a column
            self.data.reset_index(inplace=True)
            logger.info(f"Successfully fetched {len(self.data)} days of data")
            return self.data
            
        except Exception as e:
            logger.error(f"Error fetching data: {str(e)}")
            raise
    
    def calculate_moving_averages(self, windows: List[int] = [10, 20, 50, 200]) -> pd.DataFrame:
        """
        Calculate Simple Moving Averages (SMA) and Exponential Moving Averages (EMA)
        
        Args:
            windows: List of window sizes for moving averages
            
        Returns:
            DataFrame with added moving average columns
        """
        if self.data is None:
            raise ValueError("No data available. Please fetch stock data first.")
        
        for window in windows:
            # Simple Moving Average
            self.data[f'SMA_{window}'] = self.data['Close'].rolling(window=window).mean()
            # Exponential Moving Average
            self.data[f'EMA_{window}'] = self.data['Close'].ewm(span=window, adjust=False).mean()
            
        logger.info(f"Calculated moving averages for windows: {windows}")
        return self.data
    
    def calculate_rsi(self, period: int = 14) -> pd.DataFrame:
        """
        Calculate Relative Strength Index (RSI)
        
        Args:
            period: Period for RSI calculation (default: 14)
            
        Returns:
            DataFrame with added RSI column
        """
        if self.data is None:
            raise ValueError("No data available. Please fetch stock data first.")
        
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        self.data[f'RSI_{period}'] = 100 - (100 / (1 + rs))
        
        logger.info(f"Calculated RSI with period {period}")
        return self.data
    
    def calculate_macd(self, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """
        Calculate MACD (Moving Average Convergence Divergence)
        
        Args:
            fast: Fast EMA period (default: 12)
            slow: Slow EMA period (default: 26)
            signal: Signal line EMA period (default: 9)
            
        Returns:
            DataFrame with MACD columns
        """
        if self.data is None:
            raise ValueError("No data available. Please fetch stock data first.")
        
        exp1 = self.data['Close'].ewm(span=fast, adjust=False).mean()
        exp2 = self.data['Close'].ewm(span=slow, adjust=False).mean()
        
        self.data['MACD'] = exp1 - exp2
        self.data['MACD_Signal'] = self.data['MACD'].ewm(span=signal, adjust=False).mean()
        self.data['MACD_Histogram'] = self.data['MACD'] - self.data['MACD_Signal']
        
        logger.info(f"Calculated MACD with periods ({fast}, {slow}, {signal})")
        return self.data
    
    def calculate_bollinger_bands(self, window: int = 20, num_std: float = 2) -> pd.DataFrame:
        """
        Calculate Bollinger Bands
        
        Args:
            window: Window size for moving average (default: 20)
            num_std: Number of standard deviations (default: 2)
            
        Returns:
            DataFrame with Bollinger Bands columns
        """
        if self.data is None:
            raise ValueError("No data available. Please fetch stock data first.")
        
        rolling_mean = self.data['Close'].rolling(window=window).mean()
        rolling_std = self.data['Close'].rolling(window=window).std()
        
        self.data['BB_Middle'] = rolling_mean
        self.data['BB_Upper'] = rolling_mean + (rolling_std * num_std)
        self.data['BB_Lower'] = rolling_mean - (rolling_std * num_std)
        self.data['BB_Width'] = self.data['BB_Upper'] - self.data['BB_Lower']
        self.data['BB_Position'] = (self.data['Close'] - self.data['BB_Lower']) / (self.data['BB_Upper'] - self.data['BB_Lower'])
        
        logger.info(f"Calculated Bollinger Bands with window {window} and {num_std} std")
        return self.data
    
    def calculate_volatility(self, window: int = 20) -> pd.DataFrame:
        """
        Calculate historical volatility
        
        Args:
            window: Window size for volatility calculation
            
        Returns:
            DataFrame with volatility column
        """
        if self.data is None:
            raise ValueError("No data available. Please fetch stock data first.")
        
        # Calculate daily returns
        self.data['Returns'] = self.data['Close'].pct_change()
        
        # Calculate rolling volatility (annualized)
        self.data[f'Volatility_{window}'] = self.data['Returns'].rolling(window=window).std() * np.sqrt(252)
        
        logger.info(f"Calculated volatility with window {window}")
        return self.data
    
    def calculate_volume_indicators(self) -> pd.DataFrame:
        """
        Calculate volume-based indicators
        
        Returns:
            DataFrame with volume indicator columns
        """
        if self.data is None:
            raise ValueError("No data available. Please fetch stock data first.")
        
        # Volume Moving Average
        self.data['Volume_MA_20'] = self.data['Volume'].rolling(window=20).mean()
        
        # On-Balance Volume (OBV)
        obv = []
        obv_value = 0
        for i in range(len(self.data)):
            if i == 0:
                obv.append(0)
            else:
                if self.data['Close'].iloc[i] > self.data['Close'].iloc[i-1]:
                    obv_value += self.data['Volume'].iloc[i]
                elif self.data['Close'].iloc[i] < self.data['Close'].iloc[i-1]:
                    obv_value -= self.data['Volume'].iloc[i]
                obv.append(obv_value)
        self.data['OBV'] = obv
        
        # Volume Rate of Change
        self.data['Volume_ROC'] = self.data['Volume'].pct_change(periods=10) * 100
        
        logger.info("Calculated volume indicators")
        return self.data
    
    def add_price_features(self) -> pd.DataFrame:
        """
        Add price-based features for ML models
        
        Returns:
            DataFrame with additional price features
        """
        if self.data is None:
            raise ValueError("No data available. Please fetch stock data first.")
        
        # Price change features
        self.data['Price_Change'] = self.data['Close'] - self.data['Open']
        self.data['High_Low_Ratio'] = self.data['High'] / self.data['Low']
        self.data['Close_Open_Ratio'] = self.data['Close'] / self.data['Open']
        
        # Lagged features
        for lag in [1, 3, 5, 10]:
            self.data[f'Close_Lag_{lag}'] = self.data['Close'].shift(lag)
            self.data[f'Returns_Lag_{lag}'] = self.data['Returns'].shift(lag)
        
        logger.info("Added price-based features")
        return self.data
    
    def prepare_features(self) -> pd.DataFrame:
        """
        Prepare all features for machine learning
        
        Returns:
            DataFrame with all calculated features
        """
        logger.info("Preparing all features...")
        
        # Fetch data if not already available
        if self.data is None:
            self.fetch_stock_data()
        
        # Calculate all technical indicators
        self.calculate_moving_averages()
        self.calculate_rsi()
        self.calculate_macd()
        self.calculate_bollinger_bands()
        self.calculate_volatility()
        self.calculate_volume_indicators()
        self.add_price_features()
        
        # Add target variable (next day's return)
        self.data['Target'] = self.data['Close'].shift(-1) / self.data['Close'] - 1
        
        # Drop rows with NaN values
        self.data.dropna(inplace=True)
        
        logger.info(f"Feature preparation complete. Shape: {self.data.shape}")
        return self.data
    
    def save_data(self, filepath: str = None) -> str:
        """
        Save the processed data to a CSV file
        
        Args:
            filepath: Path to save the file (optional)
            
        Returns:
            Path where the file was saved
        """
        if self.data is None:
            raise ValueError("No data available to save.")
        
        if filepath is None:
            filepath = f"{self.symbol}_stock_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        self.data.to_csv(filepath, index=False)
        logger.info(f"Data saved to {filepath}")
        return filepath


# Example usage
if __name__ == "__main__":
    # Initialize collector for Apple stock
    collector = StockDataCollector('AAPL', start_date='2022-01-01')
    
    # Fetch and prepare all features
    data = collector.prepare_features()
    
    # Display first few rows and info
    print("\nData shape:", data.shape)
    print("\nFirst 5 rows:")
    print(data.head())
    print("\nColumn names:")
    print(data.columns.tolist())
    print("\nData info:")
    print(data.info())
    
    # Save the data
    filepath = collector.save_data()
    print(f"\nData saved to: {filepath}")