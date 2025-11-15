"""
Multi-Asset Data Collection

Supports data collection from multiple asset classes:
- Cryptocurrencies (Binance, Coinbase, CoinGecko)
- Forex pairs (OANDA, Alpha Vantage)
- Commodities (Yahoo Finance, Fred)
- Indices and ETFs
- Multi-timeframe support (1m, 5m, 15m, 1h, 4h, 1d)
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
import time
import requests

logger = logging.getLogger(__name__)


@dataclass
class AssetConfig:
    """Configuration for asset data collection."""

    symbol: str
    asset_type: str  # 'stock', 'crypto', 'forex', 'commodity'
    exchange: Optional[str] = None
    base_currency: str = 'USD'
    timeframe: str = '1d'  # 1m, 5m, 15m, 1h, 4h, 1d, 1w
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class CryptoDataCollector:
    """
    Cryptocurrency data collector.

    Supports:
    - Binance API (free, high rate limits)
    - CoinGecko API (free, comprehensive)
    - Yahoo Finance (for major crypto)
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize crypto data collector.

        Args:
            api_key: Optional API key for premium features
        """
        self.api_key = api_key
        self.binance_base_url = "https://api.binance.com/api/v3"
        self.coingecko_base_url = "https://api.coingecko.com/api/v3"

    def fetch_binance_data(
        self,
        symbol: str,
        interval: str = '1d',
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Fetch data from Binance.

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            interval: Kline interval (1m, 5m, 15m, 1h, 4h, 1d, 1w)
            start_time: Start time in milliseconds
            end_time: End time in milliseconds
            limit: Number of records to fetch (max 1000)

        Returns:
            DataFrame with OHLCV data
        """
        # Map common intervals to Binance format
        interval_map = {
            '1m': '1m', '5m': '5m', '15m': '15m',
            '1h': '1h', '4h': '4h',
            '1d': '1d', '1w': '1w'
        }

        binance_interval = interval_map.get(interval, '1d')

        # Build request
        url = f"{self.binance_base_url}/klines"
        params = {
            'symbol': symbol.upper(),
            'interval': binance_interval,
            'limit': limit
        }

        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time

        try:
            logger.info(f"Fetching {symbol} from Binance ({interval})")
            response = requests.get(url, params=params)
            response.raise_for_status()

            data = response.json()

            # Convert to DataFrame
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])

            # Process data
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            # Convert to numeric
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])

            # Rename columns
            df = df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })

            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

            logger.info(f"Fetched {len(df)} records for {symbol}")

            return df

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch Binance data: {e}")
            raise

    def fetch_coingecko_data(
        self,
        coin_id: str,
        vs_currency: str = 'usd',
        days: int = 365
    ) -> pd.DataFrame:
        """
        Fetch data from CoinGecko.

        Args:
            coin_id: CoinGecko coin ID (e.g., 'bitcoin', 'ethereum')
            vs_currency: Quote currency
            days: Number of days of history

        Returns:
            DataFrame with price data
        """
        url = f"{self.coingecko_base_url}/coins/{coin_id}/market_chart"
        params = {
            'vs_currency': vs_currency,
            'days': days,
            'interval': 'daily' if days > 90 else 'hourly'
        }

        try:
            logger.info(f"Fetching {coin_id} from CoinGecko")
            response = requests.get(url, params=params)
            response.raise_for_status()

            data = response.json()

            # Extract prices
            prices = pd.DataFrame(data['prices'], columns=['timestamp', 'Close'])
            prices['timestamp'] = pd.to_datetime(prices['timestamp'], unit='ms')
            prices.set_index('timestamp', inplace=True)

            # Extract volumes
            volumes = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'Volume'])
            volumes['timestamp'] = pd.to_datetime(volumes['timestamp'], unit='ms')
            volumes.set_index('timestamp', inplace=True)

            # Merge
            df = prices.join(volumes)

            # Create OHLC from close prices (approximation)
            df['Open'] = df['Close']
            df['High'] = df['Close']
            df['Low'] = df['Close']

            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

            logger.info(f"Fetched {len(df)} records for {coin_id}")

            return df

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch CoinGecko data: {e}")
            raise

    def fetch_crypto_yahoo(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch crypto data from Yahoo Finance.

        Args:
            symbol: Yahoo symbol (e.g., 'BTC-USD', 'ETH-USD')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with OHLCV data
        """
        try:
            import yfinance as yf

            logger.info(f"Fetching {symbol} from Yahoo Finance")

            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)

            if df.empty:
                raise ValueError(f"No data found for {symbol}")

            logger.info(f"Fetched {len(df)} records for {symbol}")

            return df

        except Exception as e:
            logger.error(f"Failed to fetch Yahoo Finance data: {e}")
            raise


class ForexDataCollector:
    """
    Forex data collector.

    Supports:
    - Yahoo Finance (major pairs)
    - Alpha Vantage API
    - OANDA API (requires account)
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize forex data collector.

        Args:
            api_key: API key for Alpha Vantage or OANDA
        """
        self.api_key = api_key
        self.alpha_vantage_base_url = "https://www.alphavantage.co/query"

    def fetch_forex_yahoo(
        self,
        pair: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch forex data from Yahoo Finance.

        Args:
            pair: Forex pair (e.g., 'EURUSD=X', 'GBPUSD=X')
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with price data
        """
        try:
            import yfinance as yf

            # Ensure correct format
            if not pair.endswith('=X'):
                pair = f"{pair}=X"

            logger.info(f"Fetching {pair} from Yahoo Finance")

            ticker = yf.Ticker(pair)
            df = ticker.history(start=start_date, end=end_date)

            if df.empty:
                raise ValueError(f"No data found for {pair}")

            logger.info(f"Fetched {len(df)} records for {pair}")

            return df

        except Exception as e:
            logger.error(f"Failed to fetch forex data: {e}")
            raise

    def fetch_alpha_vantage_forex(
        self,
        from_currency: str,
        to_currency: str = 'USD',
        outputsize: str = 'full'
    ) -> pd.DataFrame:
        """
        Fetch forex data from Alpha Vantage.

        Args:
            from_currency: Base currency (e.g., 'EUR')
            to_currency: Quote currency (e.g., 'USD')
            outputsize: 'compact' (100 days) or 'full' (20+ years)

        Returns:
            DataFrame with daily forex data
        """
        if not self.api_key:
            raise ValueError("Alpha Vantage API key required")

        url = self.alpha_vantage_base_url
        params = {
            'function': 'FX_DAILY',
            'from_symbol': from_currency,
            'to_symbol': to_currency,
            'apikey': self.api_key,
            'outputsize': outputsize
        }

        try:
            logger.info(f"Fetching {from_currency}/{to_currency} from Alpha Vantage")
            response = requests.get(url, params=params)
            response.raise_for_status()

            data = response.json()

            if 'Error Message' in data:
                raise ValueError(f"API Error: {data['Error Message']}")

            if 'Note' in data:
                raise ValueError(f"API Limit: {data['Note']}")

            # Extract time series
            time_series = data.get('Time Series FX (Daily)', {})

            # Convert to DataFrame
            df = pd.DataFrame.from_dict(time_series, orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()

            # Rename columns
            df.columns = ['Open', 'High', 'Low', 'Close']

            # Convert to numeric
            df = df.astype(float)

            # Add volume (N/A for forex)
            df['Volume'] = 0

            logger.info(f"Fetched {len(df)} records")

            return df

        except Exception as e:
            logger.error(f"Failed to fetch Alpha Vantage data: {e}")
            raise


class CommodityDataCollector:
    """
    Commodity data collector.

    Supports:
    - Yahoo Finance (Gold, Silver, Oil, etc.)
    - FRED (Federal Reserve Economic Data)
    """

    def __init__(self, fred_api_key: Optional[str] = None):
        """
        Initialize commodity data collector.

        Args:
            fred_api_key: FRED API key (free from https://fred.stlouisfed.org/)
        """
        self.fred_api_key = fred_api_key
        self.fred_base_url = "https://api.stlouisfed.org/fred/series/observations"

        # Common commodity symbols
        self.yahoo_symbols = {
            'gold': 'GC=F',
            'silver': 'SI=F',
            'crude_oil': 'CL=F',
            'natural_gas': 'NG=F',
            'copper': 'HG=F',
            'platinum': 'PL=F',
            'palladium': 'PA=F'
        }

    def fetch_commodity_yahoo(
        self,
        commodity: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch commodity data from Yahoo Finance.

        Args:
            commodity: Commodity name or Yahoo symbol
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with price data
        """
        try:
            import yfinance as yf

            # Map common names to Yahoo symbols
            symbol = self.yahoo_symbols.get(commodity.lower(), commodity)

            logger.info(f"Fetching {commodity} ({symbol}) from Yahoo Finance")

            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)

            if df.empty:
                raise ValueError(f"No data found for {commodity}")

            logger.info(f"Fetched {len(df)} records")

            return df

        except Exception as e:
            logger.error(f"Failed to fetch commodity data: {e}")
            raise

    def fetch_fred_data(
        self,
        series_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch economic data from FRED.

        Args:
            series_id: FRED series ID (e.g., 'GOLDAMGBD228NLBM' for gold)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with data
        """
        if not self.fred_api_key:
            raise ValueError("FRED API key required")

        params = {
            'series_id': series_id,
            'api_key': self.fred_api_key,
            'file_type': 'json'
        }

        if start_date:
            params['observation_start'] = start_date
        if end_date:
            params['observation_end'] = end_date

        try:
            logger.info(f"Fetching {series_id} from FRED")
            response = requests.get(self.fred_base_url, params=params)
            response.raise_for_status()

            data = response.json()

            # Convert to DataFrame
            df = pd.DataFrame(data['observations'])
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)

            # Convert value to numeric
            df['value'] = pd.to_numeric(df['value'], errors='coerce')

            # Create OHLC (same as close for FRED data)
            df = df.rename(columns={'value': 'Close'})
            df['Open'] = df['Close']
            df['High'] = df['Close']
            df['Low'] = df['Close']
            df['Volume'] = 0

            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            df = df.dropna()

            logger.info(f"Fetched {len(df)} records")

            return df

        except Exception as e:
            logger.error(f"Failed to fetch FRED data: {e}")
            raise


class MultiAssetDataManager:
    """
    Unified interface for multi-asset data collection.

    Automatically routes requests to appropriate collectors.
    """

    def __init__(
        self,
        alpha_vantage_key: Optional[str] = None,
        fred_key: Optional[str] = None
    ):
        """
        Initialize multi-asset manager.

        Args:
            alpha_vantage_key: Alpha Vantage API key
            fred_key: FRED API key
        """
        self.crypto_collector = CryptoDataCollector()
        self.forex_collector = ForexDataCollector(api_key=alpha_vantage_key)
        self.commodity_collector = CommodityDataCollector(fred_api_key=fred_key)

    def fetch_data(self, config: AssetConfig) -> pd.DataFrame:
        """
        Fetch data for any asset type.

        Args:
            config: Asset configuration

        Returns:
            DataFrame with OHLCV data
        """
        asset_type = config.asset_type.lower()

        try:
            if asset_type == 'crypto':
                # Try Binance first for crypto
                if config.exchange == 'binance' or config.exchange is None:
                    return self.crypto_collector.fetch_binance_data(
                        symbol=config.symbol,
                        interval=config.timeframe
                    )
                else:
                    # Fallback to Yahoo
                    return self.crypto_collector.fetch_crypto_yahoo(
                        symbol=config.symbol,
                        start_date=config.start_date,
                        end_date=config.end_date
                    )

            elif asset_type == 'forex':
                return self.forex_collector.fetch_forex_yahoo(
                    pair=config.symbol,
                    start_date=config.start_date,
                    end_date=config.end_date
                )

            elif asset_type == 'commodity':
                return self.commodity_collector.fetch_commodity_yahoo(
                    commodity=config.symbol,
                    start_date=config.start_date,
                    end_date=config.end_date
                )

            elif asset_type == 'stock':
                # Use yfinance for stocks
                import yfinance as yf
                ticker = yf.Ticker(config.symbol)
                return ticker.history(
                    start=config.start_date,
                    end=config.end_date
                )

            else:
                raise ValueError(f"Unknown asset type: {asset_type}")

        except Exception as e:
            logger.error(f"Failed to fetch data for {config.symbol}: {e}")
            raise

    def fetch_multiple_assets(
        self,
        configs: List[AssetConfig]
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple assets.

        Args:
            configs: List of asset configurations

        Returns:
            Dictionary of {symbol: DataFrame}
        """
        results = {}

        for config in configs:
            try:
                logger.info(f"Fetching {config.symbol} ({config.asset_type})")
                df = self.fetch_data(config)
                results[config.symbol] = df
                time.sleep(0.5)  # Rate limiting
            except Exception as e:
                logger.error(f"Failed to fetch {config.symbol}: {e}")
                continue

        return results


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    manager = MultiAssetDataManager()

    # Define assets to fetch
    assets = [
        AssetConfig(symbol='BTCUSDT', asset_type='crypto', exchange='binance', timeframe='1d'),
        AssetConfig(symbol='EURUSD', asset_type='forex'),
        AssetConfig(symbol='gold', asset_type='commodity'),
        AssetConfig(symbol='AAPL', asset_type='stock')
    ]

    print("Fetching multi-asset data...")
    data = manager.fetch_multiple_assets(assets)

    for symbol, df in data.items():
        print(f"\n{symbol}:")
        print(df.tail())
        print(f"Shape: {df.shape}")
