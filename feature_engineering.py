"""
Advanced Feature Engineering Module
Handles feature creation, selection, and preprocessing for ML models
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, mutual_info_regression, RFE
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from typing import Tuple, List, Optional, Dict
import logging
from scipy import stats
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    logging.warning("TA-Lib not available. Using basic pattern detection.")
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Advanced feature engineering for stock price prediction"""
    
    def __init__(self, scaling_method: str = 'robust'):
        """
        Initialize the feature engineer
        
        Args:
            scaling_method: Method for scaling features ('standard', 'minmax', 'robust')
        """
        self.scaling_method = scaling_method
        self.scaler = self._get_scaler(scaling_method)
        self.feature_names = None
        self.selected_features = None
        self.pca = None
        
    def _get_scaler(self, method: str):
        """Get the appropriate scaler based on method"""
        scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }
        return scalers.get(method, RobustScaler())
    
    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create advanced technical features
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with additional advanced features
        """
        logger.info("Creating advanced technical features...")
        
        # Ensure we have necessary columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")
        
        # Price-based features
        df = self._create_price_features(df)
        
        # Volume-based features
        df = self._create_volume_features(df)
        
        # Volatility features
        df = self._create_volatility_features(df)
        
        # Pattern recognition features
        df = self._create_pattern_features(df)
        
        # Market microstructure features
        df = self._create_microstructure_features(df)
        
        # Momentum indicators
        df = self._create_momentum_features(df)
        
        # Statistical features
        df = self._create_statistical_features(df)
        
        logger.info(f"Created {len(df.columns)} total features")
        return df
    
    def _create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create price-based features"""
        # Price ratios
        df['HL_Ratio'] = df['High'] / df['Low']
        df['CO_Ratio'] = df['Close'] / df['Open']
        df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        
        # Gap features
        df['Gap'] = df['Open'] - df['Close'].shift(1)
        df['Gap_Percentage'] = (df['Gap'] / df['Close'].shift(1)) * 100
        
        # Price position within day's range
        df['Price_Position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
        
        # Distance from various MAs
        for period in [10, 20, 50]:
            ma = df['Close'].rolling(period).mean()
            df[f'Distance_MA{period}'] = (df['Close'] - ma) / ma * 100
        
        # Price channels
        df['Highest_20'] = df['High'].rolling(20).max()
        df['Lowest_20'] = df['Low'].rolling(20).min()
        df['Price_Channel_Position'] = (df['Close'] - df['Lowest_20']) / (df['Highest_20'] - df['Lowest_20'])
        
        return df
    
    def _create_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create volume-based features"""
        # Volume-weighted average price (VWAP)
        df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
        df['Price_VWAP_Ratio'] = df['Close'] / df['VWAP']
        
        # Money Flow Index (MFI)
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        money_flow = typical_price * df['Volume']
        
        positive_flow = pd.Series(0, index=df.index)
        negative_flow = pd.Series(0, index=df.index)
        
        for i in range(1, len(df)):
            if typical_price.iloc[i] > typical_price.iloc[i-1]:
                positive_flow.iloc[i] = money_flow.iloc[i]
            else:
                negative_flow.iloc[i] = money_flow.iloc[i]
        
        positive_flow_sum = positive_flow.rolling(14).sum()
        negative_flow_sum = negative_flow.rolling(14).sum()
        
        mfi_ratio = positive_flow_sum / negative_flow_sum
        df['MFI'] = 100 - (100 / (1 + mfi_ratio))
        
        # Accumulation/Distribution Line
        clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
        df['AD_Line'] = (clv * df['Volume']).cumsum()
        
        # Chaikin Money Flow
        df['CMF'] = (clv * df['Volume']).rolling(20).sum() / df['Volume'].rolling(20).sum()
        
        return df
    
    def _create_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create volatility-based features"""
        # Average True Range (ATR)
        high_low = df['High'] - df['Low']
        high_close = abs(df['High'] - df['Close'].shift())
        low_close = abs(df['Low'] - df['Close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = true_range.rolling(14).mean()
        df['ATR_Percentage'] = (df['ATR'] / df['Close']) * 100
        
        # Normalized ATR
        df['NATR'] = (df['ATR'] / df['Close']) * 100
        
        # Keltner Channels
        ema20 = df['Close'].ewm(span=20).mean()
        df['KC_Upper'] = ema20 + (df['ATR'] * 2)
        df['KC_Lower'] = ema20 - (df['ATR'] * 2)
        df['KC_Position'] = (df['Close'] - df['KC_Lower']) / (df['KC_Upper'] - df['KC_Lower'])
        
        # Historical Volatility (different periods)
        for period in [5, 10, 20, 60]:
            returns = df['Close'].pct_change()
            df[f'HV_{period}'] = returns.rolling(period).std() * np.sqrt(252)
        
        # Parkinson Volatility
        df['Parkinson_Vol'] = np.sqrt(
            (1/(4*np.log(2))) * ((np.log(df['High']/df['Low']))**2).rolling(20).mean()
        ) * np.sqrt(252)
        
        return df
    
    def _create_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create candlestick pattern features"""
        # Using TA-Lib for pattern recognition if available
        use_simple = True
        if TALIB_AVAILABLE:
            try:
                # Bullish patterns
                df['HAMMER'] = talib.CDLHAMMER(df['Open'], df['High'], df['Low'], df['Close'])
                df['MORNING_STAR'] = talib.CDLMORNINGSTAR(df['Open'], df['High'], df['Low'], df['Close'])
                df['BULLISH_ENGULFING'] = talib.CDLENGULFING(df['Open'], df['High'], df['Low'], df['Close'])

                # Bearish patterns
                df['SHOOTING_STAR'] = talib.CDLSHOOTINGSTAR(df['Open'], df['High'], df['Low'], df['Close'])
                df['EVENING_STAR'] = talib.CDLEVENINGSTAR(df['Open'], df['High'], df['Low'], df['Close'])
                df['BEARISH_ENGULFING'] = talib.CDLENGULFING(df['Open'], df['High'], df['Low'], df['Close']) * -1

                # Neutral/Reversal patterns
                df['DOJI'] = talib.CDLDOJI(df['Open'], df['High'], df['Low'], df['Close'])
                df['SPINNING_TOP'] = talib.CDLSPINNINGTOP(df['Open'], df['High'], df['Low'], df['Close'])

                use_simple = False

            except Exception as e:
                logger.warning(f"TA-Lib error: {str(e)}, using simple pattern detection")

        if use_simple:
            # Simple pattern detection without TA-Lib
            df['Body_Size'] = abs(df['Close'] - df['Open'])
            df['Upper_Shadow'] = df['High'] - df[['Close', 'Open']].max(axis=1)
            df['Lower_Shadow'] = df[['Close', 'Open']].min(axis=1) - df['Low']
            df['Body_to_Shadow_Ratio'] = df['Body_Size'] / (df['Upper_Shadow'] + df['Lower_Shadow'] + 0.001)
        
        return df
    
    def _create_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create market microstructure features"""
        # Bid-Ask Spread Proxy (using High-Low as proxy)
        df['Spread_Proxy'] = (df['High'] - df['Low']) / df['Close']
        
        # Amihud Illiquidity Measure
        df['Illiquidity'] = abs(df['Close'].pct_change()) / (df['Volume'] + 1)
        df['Illiquidity_MA'] = df['Illiquidity'].rolling(20).mean()
        
        # Roll Measure (serial covariance of price changes)
        returns = df['Close'].pct_change()
        df['Roll_Measure'] = returns.rolling(20).apply(
            lambda x: 2 * np.sqrt(-np.cov(x[:-1], x[1:])[0, 1]) if len(x) > 1 else 0
        )
        
        # Kyle's Lambda (price impact)
        df['Kyle_Lambda'] = abs(returns) / np.log(df['Volume'] + 1)
        
        return df
    
    def _create_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create momentum indicators"""
        # Rate of Change (ROC)
        for period in [5, 10, 20]:
            df[f'ROC_{period}'] = ((df['Close'] - df['Close'].shift(period)) / df['Close'].shift(period)) * 100
        
        # Stochastic Oscillator
        for period in [14, 28]:
            low_min = df['Low'].rolling(period).min()
            high_max = df['High'].rolling(period).max()
            df[f'Stoch_{period}'] = ((df['Close'] - low_min) / (high_max - low_min)) * 100
            df[f'Stoch_{period}_MA3'] = df[f'Stoch_{period}'].rolling(3).mean()
        
        # Williams %R
        df['Williams_R'] = ((df['High'].rolling(14).max() - df['Close']) / 
                           (df['High'].rolling(14).max() - df['Low'].rolling(14).min())) * -100
        
        # Commodity Channel Index (CCI)
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        sma = typical_price.rolling(20).mean()
        mean_deviation = typical_price.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())))
        df['CCI'] = (typical_price - sma) / (0.015 * mean_deviation)
        
        # Ultimate Oscillator
        bp = df['Close'] - df[['Low', 'Close']].shift(1).min(axis=1)
        tr = df[['High', 'Close']].shift(1).max(axis=1) - df[['Low', 'Close']].shift(1).min(axis=1)
        
        avg7 = (bp.rolling(7).sum() / tr.rolling(7).sum()) * 100
        avg14 = (bp.rolling(14).sum() / tr.rolling(14).sum()) * 100
        avg28 = (bp.rolling(28).sum() / tr.rolling(28).sum()) * 100
        
        df['Ultimate_Oscillator'] = (avg7 * 4 + avg14 * 2 + avg28) / 7
        
        return df
    
    def _create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create statistical features"""
        returns = df['Close'].pct_change()
        
        # Rolling statistics
        for period in [5, 10, 20, 60]:
            df[f'Return_Mean_{period}'] = returns.rolling(period).mean()
            df[f'Return_Std_{period}'] = returns.rolling(period).std()
            df[f'Return_Skew_{period}'] = returns.rolling(period).skew()
            df[f'Return_Kurt_{period}'] = returns.rolling(period).kurt()
            
            # Z-score
            df[f'Z_Score_{period}'] = (df['Close'] - df['Close'].rolling(period).mean()) / df['Close'].rolling(period).std()
        
        # Hurst Exponent (trend strength)
        def hurst_exponent(ts, lag=20):
            lags = range(2, lag)
            tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0] * 2.0
        
        df['Hurst_Exponent'] = df['Close'].rolling(60).apply(
            lambda x: hurst_exponent(x.values) if len(x) >= 20 else 0.5
        )
        
        # Autocorrelation
        for lag in [1, 5, 10]:
            df[f'Autocorr_{lag}'] = returns.rolling(20).apply(
                lambda x: x.autocorr(lag=lag) if len(x) > lag else 0
            )
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame, 
                                  feature_cols: List[str], 
                                  max_interactions: int = 10) -> pd.DataFrame:
        """
        Create interaction features between important features
        
        Args:
            df: DataFrame with features
            feature_cols: List of feature column names
            max_interactions: Maximum number of interaction features to create
            
        Returns:
            DataFrame with interaction features
        """
        logger.info(f"Creating interaction features from {len(feature_cols)} features")
        
        # Select most important features for interactions
        if len(feature_cols) > 10:
            # Use correlation with target to select top features
            if 'Target' in df.columns:
                correlations = df[feature_cols].corrwith(df['Target']).abs()
                top_features = correlations.nlargest(10).index.tolist()
            else:
                top_features = feature_cols[:10]
        else:
            top_features = feature_cols
        
        # Create polynomial interactions
        interaction_count = 0
        for i, feat1 in enumerate(top_features):
            for feat2 in top_features[i+1:]:
                if interaction_count >= max_interactions:
                    break
                
                # Multiplication interaction
                df[f'{feat1}_X_{feat2}'] = df[feat1] * df[feat2]
                
                # Ratio interaction (with small constant to avoid division by zero)
                df[f'{feat1}_DIV_{feat2}'] = df[feat1] / (df[feat2] + 0.001)
                
                interaction_count += 2
        
        logger.info(f"Created {interaction_count} interaction features")
        return df
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, 
                       method: str = 'mutual_info', 
                       k: int = 30) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select top k features using specified method
        
        Args:
            X: Feature DataFrame
            y: Target Series
            method: Selection method ('mutual_info', 'correlation', 'rfe')
            k: Number of features to select
            
        Returns:
            Selected features DataFrame and list of selected feature names
        """
        logger.info(f"Selecting top {k} features using {method} method")
        
        # Remove any infinite or NaN values
        X_clean = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        if method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_regression, k=min(k, X_clean.shape[1]))
            X_selected = selector.fit_transform(X_clean, y)
            selected_features = X_clean.columns[selector.get_support()].tolist()
            
        elif method == 'correlation':
            correlations = X_clean.corrwith(y).abs()
            selected_features = correlations.nlargest(min(k, len(correlations))).index.tolist()
            X_selected = X_clean[selected_features]
            
        elif method == 'rfe':
            estimator = RandomForestRegressor(n_estimators=50, random_state=42)
            selector = RFE(estimator, n_features_to_select=min(k, X_clean.shape[1]))
            X_selected = selector.fit_transform(X_clean, y)
            selected_features = X_clean.columns[selector.support_].tolist()
            
        else:
            raise ValueError(f"Unknown selection method: {method}")
        
        self.selected_features = selected_features
        logger.info(f"Selected {len(selected_features)} features")
        
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index), selected_features
    
    def apply_pca(self, X: pd.DataFrame, n_components: float = 0.95) -> Tuple[pd.DataFrame, PCA]:
        """
        Apply PCA for dimensionality reduction
        
        Args:
            X: Feature DataFrame
            n_components: Number of components or variance to preserve
            
        Returns:
            Transformed DataFrame and fitted PCA object
        """
        logger.info(f"Applying PCA with n_components={n_components}")
        
        # Scale features before PCA
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply PCA
        self.pca = PCA(n_components=n_components)
        X_pca = self.pca.fit_transform(X_scaled)
        
        # Create DataFrame with PCA components
        pca_cols = [f'PCA_{i+1}' for i in range(X_pca.shape[1])]
        X_pca_df = pd.DataFrame(X_pca, columns=pca_cols, index=X.index)
        
        logger.info(f"PCA reduced {X.shape[1]} features to {X_pca.shape[1]} components")
        logger.info(f"Explained variance ratio: {self.pca.explained_variance_ratio_.sum():.2%}")
        
        return X_pca_df, self.pca
    
    def scale_features(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Scale features using the configured scaler
        
        Args:
            X: Feature DataFrame
            fit: Whether to fit the scaler (True for training, False for testing)
            
        Returns:
            Scaled DataFrame
        """
        if fit:
            X_scaled = self.scaler.fit_transform(X)
            logger.info(f"Fitted and transformed features using {self.scaling_method} scaler")
        else:
            X_scaled = self.scaler.transform(X)
            logger.info(f"Transformed features using {self.scaling_method} scaler")
        
        return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    def prepare_sequences_lstm(self, X: pd.DataFrame, y: pd.Series, 
                               sequence_length: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for LSTM model
        
        Args:
            X: Feature DataFrame
            y: Target Series
            sequence_length: Number of time steps to look back
            
        Returns:
            3D array for LSTM input and corresponding targets
        """
        logger.info(f"Preparing LSTM sequences with length {sequence_length}")
        
        X_sequences = []
        y_sequences = []
        
        for i in range(sequence_length, len(X)):
            X_sequences.append(X.iloc[i-sequence_length:i].values)
            y_sequences.append(y.iloc[i])
        
        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)
        
        logger.info(f"Created {len(X_sequences)} sequences of shape {X_sequences.shape}")
        
        return X_sequences, y_sequences
    
    def create_lagged_features(self, df: pd.DataFrame, 
                               columns: List[str], 
                               lags: List[int] = [1, 2, 3, 5, 7]) -> pd.DataFrame:
        """
        Create lagged features for specified columns
        
        Args:
            df: Input DataFrame
            columns: Columns to create lags for
            lags: List of lag periods
            
        Returns:
            DataFrame with lagged features
        """
        logger.info(f"Creating lagged features for {len(columns)} columns")
        
        for col in columns:
            if col in df.columns:
                for lag in lags:
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        return df
    
    def create_rolling_features(self, df: pd.DataFrame, 
                               columns: List[str], 
                               windows: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """
        Create rolling window features
        
        Args:
            df: Input DataFrame
            columns: Columns to create rolling features for
            windows: List of window sizes
            
        Returns:
            DataFrame with rolling features
        """
        logger.info(f"Creating rolling features for {len(columns)} columns")
        
        for col in columns:
            if col in df.columns:
                for window in windows:
                    df[f'{col}_roll_mean_{window}'] = df[col].rolling(window).mean()
                    df[f'{col}_roll_std_{window}'] = df[col].rolling(window).std()
                    df[f'{col}_roll_min_{window}'] = df[col].rolling(window).min()
                    df[f'{col}_roll_max_{window}'] = df[col].rolling(window).max()
        
        return df
    
    def create_target_features(self, df: pd.DataFrame, 
                              target_col: str = 'Close',
                              horizons: List[int] = [1, 5, 10]) -> pd.DataFrame:
        """
        Create different target variables for various prediction horizons
        
        Args:
            df: Input DataFrame
            target_col: Column to create targets from
            horizons: List of prediction horizons (days ahead)
            
        Returns:
            DataFrame with target features
        """
        logger.info(f"Creating target features for horizons {horizons}")
        
        for horizon in horizons:
            # Future return
            df[f'Target_Return_{horizon}d'] = (
                df[target_col].shift(-horizon) / df[target_col] - 1
            )
            
            # Binary classification target (up/down)
            df[f'Target_Direction_{horizon}d'] = (
                df[f'Target_Return_{horizon}d'] > 0
            ).astype(int)
            
            # Multi-class target (strong down, down, neutral, up, strong up)
            conditions = [
                df[f'Target_Return_{horizon}d'] < -0.02,
                df[f'Target_Return_{horizon}d'] < -0.005,
                df[f'Target_Return_{horizon}d'] < 0.005,
                df[f'Target_Return_{horizon}d'] < 0.02
            ]
            choices = [0, 1, 2, 3]
            df[f'Target_Class_{horizon}d'] = np.select(conditions, choices, default=4)
        
        return df
    
    def remove_multicollinearity(self, X: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
        """
        Remove highly correlated features
        
        Args:
            X: Feature DataFrame
            threshold: Correlation threshold
            
        Returns:
            DataFrame with reduced multicollinearity
        """
        logger.info(f"Removing features with correlation > {threshold}")
        
        corr_matrix = X.corr().abs()
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        to_drop = [column for column in upper_tri.columns 
                  if any(upper_tri[column] > threshold)]
        
        X_reduced = X.drop(columns=to_drop)
        
        logger.info(f"Removed {len(to_drop)} highly correlated features")
        
        return X_reduced
    
    def engineer_all_features(self, df: pd.DataFrame, 
                             target_col: str = 'Close',
                             sequence_length: int = 60) -> Dict:
        """
        Complete feature engineering pipeline
        
        Args:
            df: Raw DataFrame with OHLCV data
            target_col: Column to predict
            sequence_length: Sequence length for LSTM
            
        Returns:
            Dictionary with processed features and metadata
        """
        logger.info("Starting complete feature engineering pipeline")
        
        # Create advanced features
        df = self.create_advanced_features(df)
        
        # Create lagged and rolling features
        price_cols = ['Close', 'Volume', 'Returns']
        df = self.create_lagged_features(df, price_cols, lags=[1, 2, 3, 5, 7, 10])
        df = self.create_rolling_features(df, price_cols, windows=[5, 10, 20, 30])
        
        # Create targets
        df = self.create_target_features(df, target_col=target_col, horizons=[1, 5, 10])
        
        # Drop rows with NaN
        df = df.dropna()
        
        # Separate features and targets
        feature_cols = [col for col in df.columns 
                       if not col.startswith('Target_') and col != 'Date']
        target_cols = [col for col in df.columns if col.startswith('Target_')]
        
        X = df[feature_cols]
        y = df['Target_Return_1d']  # Default to 1-day return prediction
        
        # Remove multicollinearity
        X = self.remove_multicollinearity(X, threshold=0.95)
        
        # Select top features
        X_selected, selected_features = self.select_features(X, y, method='mutual_info', k=50)
        
        # Scale features
        X_scaled = self.scale_features(X_selected, fit=True)
        
        # Prepare LSTM sequences if needed
        X_lstm, y_lstm = self.prepare_sequences_lstm(X_scaled, y, sequence_length)
        
        # Store feature names
        self.feature_names = X_scaled.columns.tolist()
        
        result = {
            'features': X_scaled,
            'features_lstm': X_lstm,
            'target': y,
            'target_lstm': y_lstm,
            'feature_names': self.feature_names,
            'selected_features': selected_features,
            'full_dataframe': df,
            'scaler': self.scaler
        }
        
        logger.info("Feature engineering pipeline complete")
        logger.info(f"Final feature shape: {X_scaled.shape}")
        logger.info(f"LSTM sequence shape: {X_lstm.shape}")
        
        return result


# Example usage
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)
    
    df = pd.DataFrame({
        'Date': dates,
        'Open': 100 + np.random.randn(len(dates)).cumsum(),
        'High': 102 + np.random.randn(len(dates)).cumsum(),
        'Low': 98 + np.random.randn(len(dates)).cumsum(),
        'Close': 100 + np.random.randn(len(dates)).cumsum(),
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    })
    
    # Initialize feature engineer
    engineer = FeatureEngineer(scaling_method='robust')
    
    # Engineer all features
    result = engineer.engineer_all_features(df)
    
    print("\nFeature Engineering Results:")
    print(f"Number of features: {len(result['feature_names'])}")
    print(f"Feature shape: {result['features'].shape}")
    print(f"LSTM sequence shape: {result['features_lstm'].shape}")
    print(f"\nTop 10 selected features:")
    print(result['selected_features'][:10])
    print(f"\nTarget shape: {result['target'].shape}")
    print(f"Target LSTM shape: {result['target_lstm'].shape}")