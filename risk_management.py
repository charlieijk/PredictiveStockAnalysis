"""
Risk Management System

Comprehensive risk management tools including:
- Value at Risk (VaR) - Parametric, Historical, Monte Carlo
- Conditional Value at Risk (CVaR/Expected Shortfall)
- Maximum Drawdown tracking and analysis
- Stress testing and scenario analysis
- Correlation risk monitoring
- Position sizing and risk limits
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from scipy import stats
from scipy.optimize import minimize
import logging
import warnings
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class RiskLimits:
    """Risk limit configurations."""

    # VaR limits
    max_var_pct: float = 0.05  # Max 5% VaR
    var_confidence: float = 0.95  # 95% confidence level

    # Position limits
    max_position_size: float = 0.10  # Max 10% per position
    max_sector_exposure: float = 0.30  # Max 30% per sector
    max_leverage: float = 1.0  # No leverage by default

    # Drawdown limits
    max_drawdown: float = 0.20  # Max 20% drawdown
    max_drawdown_duration: int = 60  # Max 60 days in drawdown

    # Concentration limits
    max_correlation: float = 0.8  # Max correlation between positions
    min_num_positions: int = 5  # Minimum diversification

    # Volatility limits
    max_portfolio_vol: float = 0.25  # Max 25% annual volatility
    max_beta: float = 1.5  # Max beta vs benchmark


class VaRCalculator:
    """
    Value at Risk (VaR) calculator.

    Supports three methods:
    1. Parametric (variance-covariance)
    2. Historical simulation
    3. Monte Carlo simulation
    """

    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize VaR calculator.

        Args:
            confidence_level: Confidence level for VaR (e.g., 0.95 for 95%)
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level

    def parametric_var(
        self,
        returns: pd.Series,
        portfolio_value: float = 1.0,
        time_horizon: int = 1
    ) -> Dict[str, float]:
        """
        Calculate parametric VaR assuming normal distribution.

        Args:
            returns: Historical returns
            portfolio_value: Current portfolio value
            time_horizon: Time horizon in days

        Returns:
            Dictionary with VaR metrics
        """
        mean = returns.mean()
        std = returns.std()

        # Z-score for confidence level
        z_score = stats.norm.ppf(self.alpha)

        # VaR calculation
        var_pct = -(mean + z_score * std) * np.sqrt(time_horizon)
        var_amount = var_pct * portfolio_value

        return {
            'var_pct': var_pct,
            'var_amount': var_amount,
            'confidence': self.confidence_level,
            'time_horizon': time_horizon,
            'method': 'parametric'
        }

    def historical_var(
        self,
        returns: pd.Series,
        portfolio_value: float = 1.0,
        time_horizon: int = 1
    ) -> Dict[str, float]:
        """
        Calculate historical VaR using empirical distribution.

        Args:
            returns: Historical returns
            portfolio_value: Current portfolio value
            time_horizon: Time horizon in days

        Returns:
            Dictionary with VaR metrics
        """
        # Adjust returns for time horizon
        if time_horizon > 1:
            # Use overlapping periods
            adjusted_returns = []
            for i in range(len(returns) - time_horizon + 1):
                period_return = returns.iloc[i:i + time_horizon].sum()
                adjusted_returns.append(period_return)
            returns_to_use = pd.Series(adjusted_returns)
        else:
            returns_to_use = returns

        # Calculate VaR as percentile
        var_pct = -np.percentile(returns_to_use, self.alpha * 100)
        var_amount = var_pct * portfolio_value

        return {
            'var_pct': var_pct,
            'var_amount': var_amount,
            'confidence': self.confidence_level,
            'time_horizon': time_horizon,
            'method': 'historical'
        }

    def monte_carlo_var(
        self,
        returns: pd.Series,
        portfolio_value: float = 1.0,
        time_horizon: int = 1,
        n_simulations: int = 10000
    ) -> Dict[str, float]:
        """
        Calculate Monte Carlo VaR through simulation.

        Args:
            returns: Historical returns
            portfolio_value: Current portfolio value
            time_horizon: Time horizon in days
            n_simulations: Number of MC simulations

        Returns:
            Dictionary with VaR metrics
        """
        mean = returns.mean()
        std = returns.std()

        # Simulate returns
        simulated_returns = np.random.normal(
            mean * time_horizon,
            std * np.sqrt(time_horizon),
            n_simulations
        )

        # Calculate VaR
        var_pct = -np.percentile(simulated_returns, self.alpha * 100)
        var_amount = var_pct * portfolio_value

        return {
            'var_pct': var_pct,
            'var_amount': var_amount,
            'confidence': self.confidence_level,
            'time_horizon': time_horizon,
            'n_simulations': n_simulations,
            'method': 'monte_carlo'
        }

    def calculate_all_methods(
        self,
        returns: pd.Series,
        portfolio_value: float = 1.0,
        time_horizon: int = 1
    ) -> pd.DataFrame:
        """
        Calculate VaR using all three methods.

        Args:
            returns: Historical returns
            portfolio_value: Current portfolio value
            time_horizon: Time horizon in days

        Returns:
            DataFrame comparing all methods
        """
        results = []

        # Parametric
        param_var = self.parametric_var(returns, portfolio_value, time_horizon)
        results.append(param_var)

        # Historical
        hist_var = self.historical_var(returns, portfolio_value, time_horizon)
        results.append(hist_var)

        # Monte Carlo
        mc_var = self.monte_carlo_var(returns, portfolio_value, time_horizon)
        results.append(mc_var)

        return pd.DataFrame(results)


class CVaRCalculator:
    """
    Conditional Value at Risk (CVaR) / Expected Shortfall calculator.

    CVaR measures the expected loss given that loss exceeds VaR.
    """

    def __init__(self, confidence_level: float = 0.95):
        """Initialize CVaR calculator."""
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level

    def calculate_cvar(
        self,
        returns: pd.Series,
        portfolio_value: float = 1.0,
        time_horizon: int = 1
    ) -> Dict[str, float]:
        """
        Calculate CVaR (Expected Shortfall).

        Args:
            returns: Historical returns
            portfolio_value: Current portfolio value
            time_horizon: Time horizon in days

        Returns:
            Dictionary with CVaR metrics
        """
        # Adjust for time horizon
        if time_horizon > 1:
            adjusted_returns = []
            for i in range(len(returns) - time_horizon + 1):
                period_return = returns.iloc[i:i + time_horizon].sum()
                adjusted_returns.append(period_return)
            returns_to_use = pd.Series(adjusted_returns)
        else:
            returns_to_use = returns

        # Calculate VaR threshold
        var_threshold = -np.percentile(returns_to_use, self.alpha * 100)

        # Calculate CVaR as mean of losses beyond VaR
        tail_losses = -returns_to_use[returns_to_use < -var_threshold]

        if len(tail_losses) > 0:
            cvar_pct = tail_losses.mean()
        else:
            cvar_pct = var_threshold

        cvar_amount = cvar_pct * portfolio_value

        return {
            'cvar_pct': cvar_pct,
            'cvar_amount': cvar_amount,
            'var_pct': var_threshold,
            'confidence': self.confidence_level,
            'time_horizon': time_horizon,
            'tail_observations': len(tail_losses)
        }


class DrawdownAnalyzer:
    """
    Maximum Drawdown analyzer.

    Tracks and analyzes drawdowns in portfolio equity.
    """

    @staticmethod
    def calculate_drawdown(equity_curve: pd.Series) -> pd.DataFrame:
        """
        Calculate drawdown series.

        Args:
            equity_curve: Series of portfolio values over time

        Returns:
            DataFrame with drawdown metrics
        """
        # Running maximum
        running_max = equity_curve.expanding().max()

        # Drawdown
        drawdown = (equity_curve - running_max) / running_max

        # Drawdown in currency
        drawdown_amount = equity_curve - running_max

        df = pd.DataFrame({
            'equity': equity_curve,
            'running_max': running_max,
            'drawdown_pct': drawdown,
            'drawdown_amount': drawdown_amount
        })

        return df

    @staticmethod
    def get_max_drawdown(equity_curve: pd.Series) -> Dict[str, any]:
        """
        Get maximum drawdown statistics.

        Args:
            equity_curve: Series of portfolio values

        Returns:
            Dictionary with max drawdown metrics
        """
        dd_df = DrawdownAnalyzer.calculate_drawdown(equity_curve)

        max_dd_idx = dd_df['drawdown_pct'].idxmin()
        max_dd = abs(dd_df.loc[max_dd_idx, 'drawdown_pct'])
        max_dd_amount = abs(dd_df.loc[max_dd_idx, 'drawdown_amount'])

        # Find peak before max drawdown
        peak_idx = dd_df.loc[:max_dd_idx, 'running_max'].idxmax()
        peak_value = dd_df.loc[peak_idx, 'equity']

        # Find recovery (if any)
        recovery_idx = None
        if max_dd_idx != equity_curve.index[-1]:
            future_equity = equity_curve.loc[max_dd_idx:]
            recovered = future_equity[future_equity >= peak_value]
            if len(recovered) > 0:
                recovery_idx = recovered.index[0]

        # Calculate duration
        if peak_idx in equity_curve.index and max_dd_idx in equity_curve.index:
            drawdown_duration = (max_dd_idx - peak_idx).days if hasattr(peak_idx, 'days') else len(equity_curve.loc[peak_idx:max_dd_idx])
        else:
            drawdown_duration = 0

        recovery_duration = None
        if recovery_idx is not None:
            recovery_duration = (recovery_idx - max_dd_idx).days if hasattr(recovery_idx, 'days') else len(equity_curve.loc[max_dd_idx:recovery_idx])

        return {
            'max_drawdown_pct': max_dd,
            'max_drawdown_amount': max_dd_amount,
            'peak_date': peak_idx,
            'trough_date': max_dd_idx,
            'recovery_date': recovery_idx,
            'drawdown_duration': drawdown_duration,
            'recovery_duration': recovery_duration,
            'peak_value': peak_value,
            'trough_value': equity_curve.loc[max_dd_idx]
        }

    @staticmethod
    def get_all_drawdowns(equity_curve: pd.Series, min_dd_pct: float = 0.01) -> List[Dict]:
        """
        Identify all significant drawdown periods.

        Args:
            equity_curve: Series of portfolio values
            min_dd_pct: Minimum drawdown percentage to include

        Returns:
            List of drawdown dictionaries
        """
        dd_df = DrawdownAnalyzer.calculate_drawdown(equity_curve)

        drawdowns = []
        in_drawdown = False
        peak_idx = None
        peak_value = None

        for idx, row in dd_df.iterrows():
            if row['drawdown_pct'] == 0 and not in_drawdown:
                # New peak
                peak_idx = idx
                peak_value = row['equity']
            elif row['drawdown_pct'] < -min_dd_pct:
                in_drawdown = True
            elif in_drawdown and row['drawdown_pct'] == 0:
                # Recovery - end of drawdown
                trough_idx = dd_df.loc[peak_idx:idx, 'drawdown_pct'].idxmin()
                max_dd = abs(dd_df.loc[trough_idx, 'drawdown_pct'])

                drawdowns.append({
                    'peak_date': peak_idx,
                    'trough_date': trough_idx,
                    'recovery_date': idx,
                    'drawdown_pct': max_dd,
                    'peak_value': peak_value,
                    'trough_value': dd_df.loc[trough_idx, 'equity']
                })

                in_drawdown = False

        return drawdowns


class StressTesting:
    """
    Stress testing and scenario analysis.

    Tests portfolio under various market scenarios.
    """

    @staticmethod
    def market_crash_scenario(
        portfolio_weights: np.ndarray,
        cov_matrix: pd.DataFrame,
        crash_magnitude: float = -0.20
    ) -> Dict[str, float]:
        """
        Simulate market crash scenario.

        Args:
            portfolio_weights: Current portfolio weights
            cov_matrix: Covariance matrix
            crash_magnitude: Market drop percentage (negative)

        Returns:
            Expected portfolio loss
        """
        # Assume all assets drop proportionally in crash
        crash_returns = np.full(len(portfolio_weights), crash_magnitude)

        # Portfolio loss
        portfolio_loss = np.dot(portfolio_weights, crash_returns)

        # With correlation
        portfolio_vol = np.sqrt(np.dot(portfolio_weights.T, np.dot(cov_matrix, portfolio_weights)))

        return {
            'crash_magnitude': crash_magnitude,
            'expected_loss': portfolio_loss,
            'portfolio_vol': portfolio_vol,
            'loss_in_std_devs': abs(portfolio_loss / portfolio_vol) if portfolio_vol > 0 else 0
        }

    @staticmethod
    def volatility_shock_scenario(
        portfolio_weights: np.ndarray,
        cov_matrix: pd.DataFrame,
        vol_multiplier: float = 2.0
    ) -> Dict[str, float]:
        """
        Simulate volatility shock scenario.

        Args:
            portfolio_weights: Current portfolio weights
            cov_matrix: Covariance matrix
            vol_multiplier: Multiplier for volatility

        Returns:
            New portfolio statistics
        """
        # Current volatility
        current_vol = np.sqrt(np.dot(portfolio_weights.T, np.dot(cov_matrix, portfolio_weights)))

        # Shocked covariance
        shocked_cov = cov_matrix * (vol_multiplier ** 2)
        shocked_vol = np.sqrt(np.dot(portfolio_weights.T, np.dot(shocked_cov, portfolio_weights)))

        return {
            'vol_multiplier': vol_multiplier,
            'current_vol': current_vol,
            'shocked_vol': shocked_vol,
            'vol_increase': shocked_vol - current_vol
        }

    @staticmethod
    def correlation_breakdown_scenario(
        portfolio_weights: np.ndarray,
        returns: pd.DataFrame,
        new_correlation: float = 1.0
    ) -> Dict[str, float]:
        """
        Simulate correlation breakdown (all correlations go to 1).

        Args:
            portfolio_weights: Current portfolio weights
            returns: Historical returns
            new_correlation: New correlation level

        Returns:
            Portfolio risk under new correlation
        """
        # Current stats
        current_cov = returns.cov()
        current_vol = np.sqrt(np.dot(portfolio_weights.T, np.dot(current_cov, portfolio_weights)))

        # Create new correlation matrix
        std_devs = np.sqrt(np.diag(current_cov))
        new_corr_matrix = np.full((len(std_devs), len(std_devs)), new_correlation)
        np.fill_diagonal(new_corr_matrix, 1.0)

        # New covariance matrix
        new_cov = np.outer(std_devs, std_devs) * new_corr_matrix
        new_cov = pd.DataFrame(new_cov, index=current_cov.index, columns=current_cov.columns)

        shocked_vol = np.sqrt(np.dot(portfolio_weights.T, np.dot(new_cov, portfolio_weights)))

        return {
            'current_vol': current_vol,
            'shocked_vol': shocked_vol,
            'vol_increase_pct': (shocked_vol - current_vol) / current_vol if current_vol > 0 else 0,
            'new_correlation': new_correlation
        }

    @staticmethod
    def custom_scenario(
        portfolio_weights: np.ndarray,
        asset_shocks: Dict[str, float],
        asset_names: List[str]
    ) -> Dict[str, float]:
        """
        Run custom scenario with specific asset shocks.

        Args:
            portfolio_weights: Current portfolio weights
            asset_shocks: Dictionary of {asset: return_shock}
            asset_names: List of asset names in order

        Returns:
            Portfolio impact
        """
        # Build shock vector
        shocks = np.array([asset_shocks.get(name, 0.0) for name in asset_names])

        # Portfolio return under scenario
        portfolio_return = np.dot(portfolio_weights, shocks)

        # Individual contributions
        contributions = portfolio_weights * shocks

        return {
            'portfolio_return': portfolio_return,
            'contributions': dict(zip(asset_names, contributions)),
            'worst_contributor': asset_names[np.argmin(contributions)],
            'best_contributor': asset_names[np.argmax(contributions)]
        }


class RiskMonitor:
    """
    Real-time risk monitoring system.

    Tracks risk metrics and checks against limits.
    """

    def __init__(self, limits: RiskLimits):
        """Initialize risk monitor."""
        self.limits = limits
        self.var_calculator = VaRCalculator(confidence_level=limits.var_confidence)
        self.cvar_calculator = CVaRCalculator(confidence_level=limits.var_confidence)
        self.alerts: List[Dict] = []

    def check_portfolio_risk(
        self,
        portfolio_value: float,
        positions: Dict[str, float],
        returns: pd.DataFrame,
        equity_curve: Optional[pd.Series] = None
    ) -> Dict[str, any]:
        """
        Comprehensive portfolio risk check.

        Args:
            portfolio_value: Current portfolio value
            positions: Dictionary of {asset: value}
            returns: Historical returns DataFrame
            equity_curve: Optional equity curve for drawdown analysis

        Returns:
            Dictionary with risk metrics and limit violations
        """
        violations = []
        metrics = {}

        # Calculate portfolio weights
        total_value = sum(positions.values())
        weights = {asset: value / total_value for asset, value in positions.items()}

        # Check position size limits
        for asset, weight in weights.items():
            if weight > self.limits.max_position_size:
                violations.append({
                    'type': 'position_size',
                    'asset': asset,
                    'value': weight,
                    'limit': self.limits.max_position_size
                })

        # Calculate portfolio returns
        weights_array = np.array([weights.get(col, 0) for col in returns.columns])
        portfolio_returns = (returns * weights_array).sum(axis=1)

        # VaR check
        var_result = self.var_calculator.historical_var(portfolio_returns, portfolio_value)
        metrics['var'] = var_result

        if var_result['var_pct'] > self.limits.max_var_pct:
            violations.append({
                'type': 'var_limit',
                'value': var_result['var_pct'],
                'limit': self.limits.max_var_pct
            })

        # CVaR
        cvar_result = self.cvar_calculator.calculate_cvar(portfolio_returns, portfolio_value)
        metrics['cvar'] = cvar_result

        # Volatility check
        annual_vol = portfolio_returns.std() * np.sqrt(252)
        metrics['volatility'] = annual_vol

        if annual_vol > self.limits.max_portfolio_vol:
            violations.append({
                'type': 'volatility',
                'value': annual_vol,
                'limit': self.limits.max_portfolio_vol
            })

        # Drawdown check
        if equity_curve is not None:
            dd_stats = DrawdownAnalyzer.get_max_drawdown(equity_curve)
            metrics['max_drawdown'] = dd_stats

            if dd_stats['max_drawdown_pct'] > self.limits.max_drawdown:
                violations.append({
                    'type': 'drawdown',
                    'value': dd_stats['max_drawdown_pct'],
                    'limit': self.limits.max_drawdown
                })

        # Concentration check
        if len(positions) < self.limits.min_num_positions:
            violations.append({
                'type': 'concentration',
                'value': len(positions),
                'limit': self.limits.min_num_positions
            })

        # Store violations
        if violations:
            self.alerts.extend(violations)
            logger.warning(f"Risk limit violations detected: {len(violations)}")

        return {
            'metrics': metrics,
            'violations': violations,
            'risk_score': len(violations)
        }

    def get_risk_report(self) -> pd.DataFrame:
        """Get summary risk report."""
        if not self.alerts:
            return pd.DataFrame()

        return pd.DataFrame(self.alerts)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Generate sample returns
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    returns = pd.Series(np.random.randn(len(dates)) * 0.01, index=dates)

    print("=== VaR Calculation ===")
    var_calc = VaRCalculator(confidence_level=0.95)
    var_comparison = var_calc.calculate_all_methods(returns, portfolio_value=100000)
    print(var_comparison)

    print("\n=== CVaR Calculation ===")
    cvar_calc = CVaRCalculator(confidence_level=0.95)
    cvar_result = cvar_calc.calculate_cvar(returns, portfolio_value=100000)
    print(f"CVaR (95%): ${cvar_result['cvar_amount']:,.2f} ({cvar_result['cvar_pct']:.2%})")

    print("\n=== Drawdown Analysis ===")
    equity = 100000 * (1 + returns).cumprod()
    dd_stats = DrawdownAnalyzer.get_max_drawdown(equity)
    print(f"Max Drawdown: {dd_stats['max_drawdown_pct']:.2%}")
    print(f"Peak: {dd_stats['peak_date']}, Trough: {dd_stats['trough_date']}")

    print("\n=== Stress Testing ===")
    weights = np.array([0.3, 0.3, 0.4])
    returns_df = pd.DataFrame(np.random.randn(len(dates), 3) * 0.01, columns=['A', 'B', 'C'])
    cov = returns_df.cov()

    crash_result = StressTesting.market_crash_scenario(weights, cov, -0.20)
    print(f"Market Crash (-20%): Expected loss = {crash_result['expected_loss']:.2%}")
