import numpy as np
import pandas as pd
import pytest

from feature_engineering import FeatureEngineer


def _sample_price_frame(rows: int = 60) -> pd.DataFrame:
    """Create deterministic OHLCV data for targeted feature tests."""
    index = pd.date_range("2020-01-01", periods=rows, freq="D")
    base = np.linspace(100, 150, rows)
    volume = np.linspace(1_000_000, 1_500_000, rows)

    return pd.DataFrame(
        {
            "Open": base + 0.5,
            "High": base + 1.0,
            "Low": base - 1.0,
            "Close": base,
            "Volume": volume,
        },
        index=index,
    )


def test_create_lagged_features_creates_expected_columns():
    engineer = FeatureEngineer()
    raw = _sample_price_frame(10)
    result = engineer.create_lagged_features(raw.copy(), ["Close", "Volume"], lags=[1, 2])

    assert "Close_lag_1" in result and "Close_lag_2" in result
    assert "Volume_lag_1" in result and "Volume_lag_2" in result
    # Lagged values should line up with the original time series
    assert result["Close_lag_1"].iloc[3] == pytest.approx(raw["Close"].iloc[2])
    assert result["Volume_lag_2"].iloc[4] == pytest.approx(raw["Volume"].iloc[2])
    # Original frame should remain unchanged
    assert set(raw.columns) == {"Open", "High", "Low", "Close", "Volume"}


def test_create_target_features_matches_expected_returns():
    engineer = FeatureEngineer()
    raw = _sample_price_frame(20)
    result = engineer.create_target_features(raw.copy(), target_col="Close", horizons=[1])

    expected_returns = raw["Close"].shift(-1) / raw["Close"] - 1

    pd.testing.assert_series_equal(
        result["Target_Return_1d"],
        expected_returns,
        check_names=False,
    )
    pd.testing.assert_series_equal(
        result["Target_Direction_1d"],
        (expected_returns > 0).astype(int),
        check_names=False,
    )


def test_invalid_scaling_method_raises_value_error():
    with pytest.raises(ValueError):
        FeatureEngineer(scaling_method="not-a-valid-method")
