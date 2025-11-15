"""
Portfolio Optimization Module

Implements various portfolio optimization techniques:
- Modern Portfolio Theory (Markowitz mean-variance)
- Risk Parity
- Hierarchical Risk Parity (HRP)
- Black-Litterman model
- Kelly Criterion position sizing
- Minimum variance portfolio
- Maximum Sharpe ratio
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from scipy.optimize import minimize, LinearConstraint, Bounds
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
import logging
import warnings

logger = logging.getLogger(__name__)


@dataclass
class PortfolioConstraints:
    """Portfolio optimization constraints."""

    # Weight constraints
    min_weight: float = 0.0  # Minimum weight per asset
    max_weight: float = 1.0  # Maximum weight per asset
    long_only: bool = True  # No short positions

    # Concentration constraints
    max_sector_weight: Optional[float] = None  # Max weight per sector
    max_position_count: Optional[int] = None  # Max number of positions

    # Turnover constraints
    max_turnover: Optional[float] = None  # Max portfolio turnover

    # Risk constraints
    max_volatility: Optional[float] = None  # Maximum portfolio volatility
    target_return: Optional[float] = None  # Target return


class PortfolioOptimizer:
    """
    Base class for portfolio optimization.

    Provides common utilities for different optimization methods.
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        constraints: Optional[PortfolioConstraints] = None,
        risk_free_rate: float = 0.02
    ):
        """
        Initialize portfolio optimizer.

        Args:
            returns: DataFrame of asset returns (rows=dates, cols=assets)
            constraints: Portfolio constraints
            risk_free_rate: Risk-free rate for Sharpe calculation
        """
        self.returns = returns
        self.constraints = constraints or PortfolioConstraints()
        self.risk_free_rate = risk_free_rate

        # Calculate statistics
        self.mean_returns = returns.mean()
        self.cov_matrix = returns.cov()
        self.n_assets = len(returns.columns)

        # Annualization factors (assuming daily returns)
        self.annualization_factor = 252
        self.annual_mean_returns = self.mean_returns * self.annualization_factor
        self.annual_cov_matrix = self.cov_matrix * self.annualization_factor

    def calculate_portfolio_stats(
        self,
        weights: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate portfolio statistics.

        Args:
            weights: Asset weights

        Returns:
            Dictionary with return, volatility, and Sharpe ratio
        """
        portfolio_return = np.dot(weights, self.annual_mean_returns)
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(self.annual_cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0

        return {
            'return': portfolio_return,
            'volatility': portfolio_vol,
            'sharpe_ratio': sharpe_ratio
        }

    def _get_bounds(self) -> Bounds:
        """Get weight bounds for optimization."""
        if self.constraints.long_only:
            return Bounds(
                lb=np.full(self.n_assets, self.constraints.min_weight),
                ub=np.full(self.n_assets, self.constraints.max_weight)
            )
        else:
            return Bounds(
                lb=np.full(self.n_assets, -1.0),
                ub=np.full(self.n_assets, 1.0)
            )

    def _get_constraints(self) -> List:
        """Get optimization constraints."""
        constraints = []

        # Weights must sum to 1
        constraints.append({
            'type': 'eq',
            'fun': lambda w: np.sum(w) - 1.0
        })

        # Target return constraint
        if self.constraints.target_return is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda w: np.dot(w, self.annual_mean_returns) - self.constraints.target_return
            })

        # Volatility constraint
        if self.constraints.max_volatility is not None:
            def vol_constraint(w):
                vol = np.sqrt(np.dot(w.T, np.dot(self.annual_cov_matrix, w)))
                return self.constraints.max_volatility - vol

            constraints.append({
                'type': 'ineq',
                'fun': vol_constraint
            })

        return constraints


class MeanVarianceOptimizer(PortfolioOptimizer):
    """
    Mean-variance optimization (Markowitz).

    Finds optimal portfolio on the efficient frontier.
    """

    def optimize_max_sharpe(self) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Find portfolio with maximum Sharpe ratio.

        Returns:
            Tuple of (optimal weights, portfolio statistics)
        """

        def negative_sharpe(weights):
            stats = self.calculate_portfolio_stats(weights)
            return -stats['sharpe_ratio']

        # Initial guess: equal weights
        x0 = np.ones(self.n_assets) / self.n_assets

        # Optimize
        result = minimize(
            negative_sharpe,
            x0,
            method='SLSQP',
            bounds=self._get_bounds(),
            constraints=self._get_constraints()
        )

        if not result.success:
            logger.warning(f"Optimization failed: {result.message}")

        weights = result.x
        stats = self.calculate_portfolio_stats(weights)

        logger.info(f"Max Sharpe portfolio: Return={stats['return']:.2%}, "
                   f"Vol={stats['volatility']:.2%}, Sharpe={stats['sharpe_ratio']:.2f}")

        return weights, stats

    def optimize_min_variance(self) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Find minimum variance portfolio.

        Returns:
            Tuple of (optimal weights, portfolio statistics)
        """

        def portfolio_variance(weights):
            return np.dot(weights.T, np.dot(self.annual_cov_matrix, weights))

        x0 = np.ones(self.n_assets) / self.n_assets

        result = minimize(
            portfolio_variance,
            x0,
            method='SLSQP',
            bounds=self._get_bounds(),
            constraints=self._get_constraints()
        )

        if not result.success:
            logger.warning(f"Optimization failed: {result.message}")

        weights = result.x
        stats = self.calculate_portfolio_stats(weights)

        logger.info(f"Min Variance portfolio: Return={stats['return']:.2%}, "
                   f"Vol={stats['volatility']:.2%}")

        return weights, stats

    def optimize_target_return(self, target_return: float) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Find minimum variance portfolio with target return.

        Args:
            target_return: Target annual return

        Returns:
            Tuple of (optimal weights, portfolio statistics)
        """
        # Temporarily set target return constraint
        original_target = self.constraints.target_return
        self.constraints.target_return = target_return

        weights, stats = self.optimize_min_variance()

        # Restore original constraint
        self.constraints.target_return = original_target

        return weights, stats

    def get_efficient_frontier(
        self,
        n_points: int = 100
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        """
        Generate the efficient frontier.

        Args:
            n_points: Number of points on the frontier

        Returns:
            Tuple of (returns, volatilities, weights_list)
        """
        # Get range of returns
        min_ret = self.annual_mean_returns.min()
        max_ret = self.annual_mean_returns.max()

        target_returns = np.linspace(min_ret, max_ret, n_points)

        frontier_returns = []
        frontier_vols = []
        frontier_weights = []

        for target_ret in target_returns:
            try:
                weights, stats = self.optimize_target_return(target_ret)
                frontier_returns.append(stats['return'])
                frontier_vols.append(stats['volatility'])
                frontier_weights.append(weights)
            except Exception as e:
                logger.debug(f"Failed to optimize for target return {target_ret}: {e}")
                continue

        return (
            np.array(frontier_returns),
            np.array(frontier_vols),
            frontier_weights
        )


class RiskParityOptimizer(PortfolioOptimizer):
    """
    Risk Parity portfolio optimization.

    Allocates capital so each asset contributes equally to portfolio risk.
    """

    def optimize(self) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Find risk parity portfolio.

        Returns:
            Tuple of (optimal weights, portfolio statistics)
        """

        def risk_budget_objective(weights):
            """Minimize difference between risk contributions."""
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(self.annual_cov_matrix, weights)))

            # Marginal contribution to risk
            marginal_contrib = np.dot(self.annual_cov_matrix, weights) / portfolio_vol

            # Risk contribution
            risk_contrib = weights * marginal_contrib

            # Target: equal risk contribution
            target_risk = portfolio_vol / self.n_assets

            # Sum of squared deviations from target
            return np.sum((risk_contrib - target_risk) ** 2)

        # Initial guess: equal weights
        x0 = np.ones(self.n_assets) / self.n_assets

        # Optimize
        result = minimize(
            risk_budget_objective,
            x0,
            method='SLSQP',
            bounds=self._get_bounds(),
            constraints=self._get_constraints()
        )

        if not result.success:
            logger.warning(f"Risk parity optimization failed: {result.message}")

        weights = result.x
        stats = self.calculate_portfolio_stats(weights)

        logger.info(f"Risk Parity portfolio: Return={stats['return']:.2%}, "
                   f"Vol={stats['volatility']:.2%}")

        return weights, stats

    def get_risk_contributions(self, weights: np.ndarray) -> pd.Series:
        """
        Calculate risk contribution of each asset.

        Args:
            weights: Asset weights

        Returns:
            Series of risk contributions
        """
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(self.annual_cov_matrix, weights)))
        marginal_contrib = np.dot(self.annual_cov_matrix, weights) / portfolio_vol
        risk_contrib = weights * marginal_contrib

        return pd.Series(risk_contrib, index=self.returns.columns)


class HierarchicalRiskParity:
    """
    Hierarchical Risk Parity (HRP) optimizer.

    Uses hierarchical clustering to build diversified portfolios.
    """

    def __init__(self, returns: pd.DataFrame):
        """
        Initialize HRP optimizer.

        Args:
            returns: DataFrame of asset returns
        """
        self.returns = returns
        self.n_assets = len(returns.columns)
        self.cov_matrix = returns.cov()

    def optimize(self) -> Tuple[np.ndarray, pd.Series]:
        """
        Compute HRP portfolio weights.

        Returns:
            Tuple of (weights array, weights Series)
        """
        # 1. Compute correlation matrix
        corr_matrix = self.returns.corr()

        # 2. Compute distance matrix
        distance_matrix = np.sqrt((1 - corr_matrix) / 2)

        # 3. Hierarchical clustering
        link = linkage(squareform(distance_matrix), method='single')

        # 4. Quasi-diagonalization
        sorted_indices = self._get_quasi_diag(link)

        # 5. Recursive bisection
        weights = self._recursive_bisection(sorted_indices)

        weights_series = pd.Series(weights, index=self.returns.columns)

        logger.info("HRP portfolio computed successfully")

        return weights, weights_series

    def _get_quasi_diag(self, link: np.ndarray) -> List[int]:
        """Get quasi-diagonal ordering from hierarchical clustering."""
        link = link.astype(int)
        sorted_indices = []
        n = len(link) + 1

        def get_children(idx):
            if idx < n:
                return [idx]
            else:
                left_child = int(link[idx - n, 0])
                right_child = int(link[idx - n, 1])
                return get_children(left_child) + get_children(right_child)

        # Start from root
        sorted_indices = get_children(2 * n - 2)

        return sorted_indices

    def _recursive_bisection(self, sorted_indices: List[int]) -> np.ndarray:
        """
        Recursively bisect the portfolio.

        Args:
            sorted_indices: Quasi-diagonal ordering

        Returns:
            Portfolio weights
        """
        weights = np.ones(self.n_assets)
        clusters = [sorted_indices]

        while len(clusters) > 0:
            clusters_new = []

            for cluster in clusters:
                if len(cluster) > 1:
                    # Split cluster
                    mid = len(cluster) // 2
                    cluster_left = cluster[:mid]
                    cluster_right = cluster[mid:]

                    # Calculate cluster variances
                    cov_left = self.cov_matrix.iloc[cluster_left, cluster_left]
                    cov_right = self.cov_matrix.iloc[cluster_right, cluster_right]

                    var_left = self._get_cluster_var(cov_left)
                    var_right = self._get_cluster_var(cov_right)

                    # Allocate weight inversely proportional to variance
                    alpha = 1 - var_left / (var_left + var_right)

                    weights[cluster_left] *= alpha
                    weights[cluster_right] *= (1 - alpha)

                    clusters_new.append(cluster_left)
                    clusters_new.append(cluster_right)

            clusters = clusters_new

        return weights

    def _get_cluster_var(self, cov: pd.DataFrame) -> float:
        """Calculate variance of a cluster with inverse variance weighting."""
        # Inverse variance weights
        inv_diag = 1 / np.diag(cov)
        weights = inv_diag / inv_diag.sum()

        # Cluster variance
        cluster_var = np.dot(weights, np.dot(cov, weights))

        return cluster_var


class BlackLittermanOptimizer:
    """
    Black-Litterman model for portfolio optimization.

    Combines market equilibrium with investor views.
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        market_caps: Optional[pd.Series] = None,
        risk_free_rate: float = 0.02,
        risk_aversion: float = 2.5
    ):
        """
        Initialize Black-Litterman optimizer.

        Args:
            returns: Historical returns
            market_caps: Market capitalization of assets (for equilibrium weights)
            risk_free_rate: Risk-free rate
            risk_aversion: Risk aversion coefficient
        """
        self.returns = returns
        self.cov_matrix = returns.cov() * 252  # Annualized
        self.risk_free_rate = risk_free_rate
        self.risk_aversion = risk_aversion

        # Market equilibrium weights
        if market_caps is not None:
            self.market_weights = market_caps / market_caps.sum()
        else:
            # Equal weights if no market caps provided
            self.market_weights = pd.Series(1 / len(returns.columns), index=returns.columns)

        # Implied equilibrium returns
        self.pi = self._calculate_implied_returns()

    def _calculate_implied_returns(self) -> pd.Series:
        """Calculate implied equilibrium returns."""
        pi = self.risk_aversion * self.cov_matrix.dot(self.market_weights)
        return pi

    def optimize(
        self,
        views: pd.DataFrame,
        view_confidences: np.ndarray,
        tau: float = 0.05
    ) -> Tuple[np.ndarray, pd.Series, pd.Series]:
        """
        Optimize portfolio with views.

        Args:
            views: DataFrame with view matrix (P matrix)
                   Each row is a view, columns are assets
            view_confidences: Array of view confidence levels (Omega diagonal)
            tau: Scalar uncertainty of prior estimate

        Returns:
            Tuple of (weights, posterior returns, posterior covariance)
        """
        P = views.values
        Q = views.sum(axis=1).values  # View returns

        # Uncertainty in views
        Omega = np.diag(view_confidences)

        # Posterior estimates
        tau_sigma = tau * self.cov_matrix

        # M matrix
        M_inv = np.linalg.inv(tau_sigma) + P.T @ np.linalg.inv(Omega) @ P
        M = np.linalg.inv(M_inv)

        # Posterior return estimate
        posterior_returns = M @ (
            np.linalg.inv(tau_sigma) @ self.pi.values +
            P.T @ np.linalg.inv(Omega) @ Q
        )

        posterior_returns = pd.Series(posterior_returns, index=self.returns.columns)

        # Posterior covariance
        posterior_cov = self.cov_matrix + M

        # Optimize weights
        weights = np.linalg.inv(self.risk_aversion * posterior_cov) @ posterior_returns

        # Normalize weights
        weights = weights / weights.sum()

        logger.info("Black-Litterman optimization complete")

        return weights.values, posterior_returns, posterior_cov


class KellyCriterion:
    """
    Kelly Criterion for optimal position sizing.

    Maximizes long-term growth rate.
    """

    @staticmethod
    def calculate_kelly_fraction(
        win_rate: float,
        avg_win: float,
        avg_loss: float
    ) -> float:
        """
        Calculate Kelly fraction for binary outcome.

        Args:
            win_rate: Probability of winning
            avg_win: Average win size
            avg_loss: Average loss size (positive number)

        Returns:
            Optimal position size (fraction of capital)
        """
        if avg_loss == 0:
            return 0.0

        b = avg_win / avg_loss  # Win/loss ratio
        p = win_rate

        kelly = (b * p - (1 - p)) / b

        # Kelly can be negative (should not bet) or > 1 (over-leveraged)
        kelly = max(0, min(kelly, 1))

        return kelly

    @staticmethod
    def calculate_kelly_multi_asset(
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Calculate Kelly optimal portfolio for multiple assets.

        Args:
            expected_returns: Array of expected returns
            cov_matrix: Covariance matrix

        Returns:
            Optimal weights
        """
        # Kelly weights = Σ^-1 * μ
        try:
            weights = np.linalg.solve(cov_matrix, expected_returns)
        except np.linalg.LinAlgError:
            logger.warning("Singular covariance matrix, using pseudo-inverse")
            weights = np.linalg.lstsq(cov_matrix, expected_returns, rcond=None)[0]

        return weights

    @staticmethod
    def fractional_kelly(
        kelly_weights: np.ndarray,
        fraction: float = 0.5
    ) -> np.ndarray:
        """
        Apply fractional Kelly for reduced risk.

        Args:
            kelly_weights: Full Kelly weights
            fraction: Fraction to use (e.g., 0.5 for half-Kelly)

        Returns:
            Fractional Kelly weights
        """
        return kelly_weights * fraction


def compare_portfolios(
    returns: pd.DataFrame,
    portfolios: Dict[str, np.ndarray]
) -> pd.DataFrame:
    """
    Compare multiple portfolio allocations.

    Args:
        returns: Historical returns
        portfolios: Dictionary of {name: weights}

    Returns:
        Comparison DataFrame
    """
    results = []

    for name, weights in portfolios.items():
        # Portfolio returns
        port_returns = (returns * weights).sum(axis=1)

        # Statistics
        annual_return = port_returns.mean() * 252
        annual_vol = port_returns.std() * np.sqrt(252)
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0

        # Drawdown
        cumulative = (1 + port_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = abs(drawdown.min())

        results.append({
            'Portfolio': name,
            'Annual Return': annual_return,
            'Volatility': annual_vol,
            'Sharpe Ratio': sharpe,
            'Max Drawdown': max_drawdown
        })

    return pd.DataFrame(results).set_index('Portfolio')


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Generate sample data
    np.random.seed(42)
    n_assets = 5
    n_days = 500

    # Random returns
    returns = pd.DataFrame(
        np.random.randn(n_days, n_assets) * 0.01,
        columns=[f'Asset_{i}' for i in range(n_assets)]
    )

    print("\n=== Mean-Variance Optimization ===")
    mvo = MeanVarianceOptimizer(returns)
    weights_sharpe, stats_sharpe = mvo.optimize_max_sharpe()
    print(f"Max Sharpe weights: {dict(zip(returns.columns, weights_sharpe))}")

    weights_minvar, stats_minvar = mvo.optimize_min_variance()
    print(f"Min Variance weights: {dict(zip(returns.columns, weights_minvar))}")

    print("\n=== Risk Parity ===")
    rp = RiskParityOptimizer(returns)
    weights_rp, stats_rp = rp.optimize()
    print(f"Risk Parity weights: {dict(zip(returns.columns, weights_rp))}")

    print("\n=== Hierarchical Risk Parity ===")
    hrp = HierarchicalRiskParity(returns)
    weights_hrp, weights_hrp_series = hrp.optimize()
    print(f"HRP weights: {weights_hrp_series.to_dict()}")

    print("\n=== Portfolio Comparison ===")
    portfolios = {
        'Max Sharpe': weights_sharpe,
        'Min Variance': weights_minvar,
        'Risk Parity': weights_rp,
        'HRP': weights_hrp,
        'Equal Weight': np.ones(n_assets) / n_assets
    }

    comparison = compare_portfolios(returns, portfolios)
    print(comparison)
