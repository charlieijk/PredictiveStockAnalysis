"""
Advanced Backtesting Framework for Trading Strategies

This module provides comprehensive backtesting capabilities including:
- Walk-forward analysis
- Monte Carlo simulations
- Transaction cost modeling
- Performance metrics (Sharpe, Sortino, Calmar, etc.)
- Strategy comparison and optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json
import warnings

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtesting parameters."""

    initial_capital: float = 100000.0
    commission_rate: float = 0.001  # 0.1% per trade
    slippage_rate: float = 0.0005  # 0.05% slippage
    position_size: float = 1.0  # Fraction of capital per position (0-1)

    # Risk management
    max_position_size: float = 0.25  # Max 25% per position
    stop_loss: Optional[float] = 0.05  # 5% stop loss
    take_profit: Optional[float] = 0.15  # 15% take profit

    # Walk-forward analysis
    train_window: int = 252  # 1 year of trading days
    test_window: int = 63  # 3 months
    walk_forward_step: int = 21  # Step by 1 month

    # Monte Carlo
    n_simulations: int = 1000
    confidence_level: float = 0.95

    # Rebalancing
    rebalance_frequency: str = "monthly"  # daily, weekly, monthly

    # Benchmark
    benchmark_symbol: str = "SPY"
    risk_free_rate: float = 0.02  # 2% annual


@dataclass
class Trade:
    """Represents a single trade."""

    entry_date: datetime
    exit_date: Optional[datetime] = None
    symbol: str = ""
    direction: str = "long"  # long or short
    entry_price: float = 0.0
    exit_price: Optional[float] = None
    shares: int = 0
    commission: float = 0.0
    slippage: float = 0.0
    pnl: Optional[float] = None
    return_pct: Optional[float] = None
    holding_period: Optional[int] = None
    exit_reason: str = "signal"  # signal, stop_loss, take_profit, end_of_period


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""

    # Return metrics
    total_return: float = 0.0
    annual_return: float = 0.0
    cumulative_return: float = 0.0

    # Risk metrics
    volatility: float = 0.0
    downside_deviation: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0

    # Risk-adjusted returns
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    information_ratio: float = 0.0

    # Win/loss metrics
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0

    # Trade metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    avg_holding_period: float = 0.0

    # Additional metrics
    expectancy: float = 0.0
    recovery_factor: float = 0.0
    tail_ratio: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {k: v for k, v in self.__dict__.items()}


class BacktestEngine:
    """
    Core backtesting engine for strategy evaluation.

    Features:
    - Event-driven simulation
    - Transaction cost modeling
    - Position management
    - Risk management rules
    - Performance tracking
    """

    def __init__(self, config: BacktestConfig):
        """
        Initialize backtest engine.

        Args:
            config: Backtesting configuration
        """
        self.config = config
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self.positions: Dict[str, int] = {}  # symbol -> shares
        self.cash = config.initial_capital
        self.current_value = config.initial_capital

    def run(
        self,
        data: pd.DataFrame,
        signals: pd.Series,
        prices: Optional[pd.Series] = None
    ) -> Tuple[PerformanceMetrics, pd.DataFrame]:
        """
        Run backtest on historical data with trading signals.

        Args:
            data: DataFrame with OHLCV data
            signals: Series with trading signals (1=buy, -1=sell, 0=hold)
            prices: Series with execution prices (default: close prices)

        Returns:
            Tuple of (performance metrics, equity curve DataFrame)
        """
        if prices is None:
            prices = data['Close']

        # Reset state
        self.trades = []
        self.equity_curve = [self.config.initial_capital]
        self.positions = {}
        self.cash = self.config.initial_capital
        self.current_value = self.config.initial_capital

        current_trade: Optional[Trade] = None

        for idx, (date, signal) in enumerate(signals.items()):
            price = prices.loc[date]

            # Check stop loss / take profit
            if current_trade is not None and current_trade.exit_date is None:
                current_price = price
                entry_price = current_trade.entry_price

                # Check stop loss
                if self.config.stop_loss is not None:
                    if current_trade.direction == "long":
                        if (entry_price - current_price) / entry_price >= self.config.stop_loss:
                            self._close_position(current_trade, date, current_price, "stop_loss")
                            current_trade = None
                            continue
                    else:  # short
                        if (current_price - entry_price) / entry_price >= self.config.stop_loss:
                            self._close_position(current_trade, date, current_price, "stop_loss")
                            current_trade = None
                            continue

                # Check take profit
                if self.config.take_profit is not None:
                    if current_trade.direction == "long":
                        if (current_price - entry_price) / entry_price >= self.config.take_profit:
                            self._close_position(current_trade, date, current_price, "take_profit")
                            current_trade = None
                            continue
                    else:  # short
                        if (entry_price - current_price) / entry_price >= self.config.take_profit:
                            self._close_position(current_trade, date, current_price, "take_profit")
                            current_trade = None
                            continue

            # Process signals
            if signal == 1 and current_trade is None:  # Buy signal
                current_trade = self._open_position(date, price, "long")

            elif signal == -1:  # Sell signal
                if current_trade is not None and current_trade.exit_date is None:
                    self._close_position(current_trade, date, price, "signal")
                    current_trade = None

            # Update equity curve
            portfolio_value = self._calculate_portfolio_value(date, data)
            self.equity_curve.append(portfolio_value)
            self.current_value = portfolio_value

        # Close any remaining positions
        if current_trade is not None and current_trade.exit_date is None:
            last_date = data.index[-1]
            last_price = prices.iloc[-1]
            self._close_position(current_trade, last_date, last_price, "end_of_period")

        # Calculate metrics
        equity_df = pd.DataFrame({
            'equity': self.equity_curve,
            'date': [data.index[0]] + list(data.index)
        }).set_index('date')

        metrics = self._calculate_metrics(equity_df, data)

        return metrics, equity_df

    def _open_position(self, date: datetime, price: float, direction: str) -> Trade:
        """Open a new position."""

        # Calculate position size
        position_value = self.current_value * min(
            self.config.position_size,
            self.config.max_position_size
        )

        # Account for slippage
        execution_price = price * (1 + self.config.slippage_rate)

        # Calculate shares
        shares = int(position_value / execution_price)

        # Calculate commission
        commission = position_value * self.config.commission_rate

        # Update cash
        self.cash -= (shares * execution_price + commission)

        trade = Trade(
            entry_date=date,
            entry_price=execution_price,
            direction=direction,
            shares=shares,
            commission=commission,
            slippage=execution_price - price
        )

        self.trades.append(trade)
        logger.debug(f"Opened {direction} position: {shares} shares @ ${execution_price:.2f}")

        return trade

    def _close_position(self, trade: Trade, date: datetime, price: float, reason: str):
        """Close an existing position."""

        # Account for slippage
        execution_price = price * (1 - self.config.slippage_rate)

        # Calculate commission
        commission = trade.shares * execution_price * self.config.commission_rate

        # Update cash
        proceeds = trade.shares * execution_price - commission
        self.cash += proceeds

        # Update trade
        trade.exit_date = date
        trade.exit_price = execution_price
        trade.commission += commission
        trade.exit_reason = reason

        # Calculate P&L
        if trade.direction == "long":
            trade.pnl = proceeds - (trade.shares * trade.entry_price)
        else:  # short
            trade.pnl = (trade.shares * trade.entry_price) - proceeds

        trade.return_pct = trade.pnl / (trade.shares * trade.entry_price)
        trade.holding_period = (date - trade.entry_date).days

        logger.debug(f"Closed position: P&L=${trade.pnl:.2f} ({trade.return_pct*100:.2f}%) - {reason}")

    def _calculate_portfolio_value(self, date: datetime, data: pd.DataFrame) -> float:
        """Calculate current portfolio value."""

        # Start with cash
        value = self.cash

        # Add value of open positions
        for trade in self.trades:
            if trade.exit_date is None:
                current_price = data.loc[date, 'Close']
                position_value = trade.shares * current_price
                value += position_value

        return value

    def _calculate_metrics(
        self,
        equity_curve: pd.DataFrame,
        data: pd.DataFrame
    ) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics."""

        metrics = PerformanceMetrics()

        # Returns
        returns = equity_curve['equity'].pct_change().dropna()
        total_return = (equity_curve['equity'].iloc[-1] / equity_curve['equity'].iloc[0]) - 1
        metrics.total_return = total_return
        metrics.cumulative_return = total_return

        # Annualized return
        trading_days = len(equity_curve)
        years = trading_days / 252
        if years > 0:
            metrics.annual_return = (1 + total_return) ** (1 / years) - 1

        # Volatility
        metrics.volatility = returns.std() * np.sqrt(252)

        # Downside deviation
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            metrics.downside_deviation = downside_returns.std() * np.sqrt(252)

        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        metrics.max_drawdown = abs(drawdown.min())

        # Drawdown duration
        is_drawdown = drawdown < 0
        drawdown_periods = is_drawdown.astype(int).groupby(
            (is_drawdown != is_drawdown.shift()).cumsum()
        ).sum()
        if len(drawdown_periods) > 0:
            metrics.max_drawdown_duration = drawdown_periods.max()

        # Risk-adjusted returns
        excess_returns = returns - self.config.risk_free_rate / 252

        if metrics.volatility > 0:
            metrics.sharpe_ratio = (metrics.annual_return - self.config.risk_free_rate) / metrics.volatility

        if metrics.downside_deviation > 0:
            metrics.sortino_ratio = (metrics.annual_return - self.config.risk_free_rate) / metrics.downside_deviation

        if metrics.max_drawdown > 0:
            metrics.calmar_ratio = metrics.annual_return / metrics.max_drawdown

        # Trade statistics
        completed_trades = [t for t in self.trades if t.pnl is not None]
        metrics.total_trades = len(completed_trades)

        if len(completed_trades) > 0:
            winning_trades = [t for t in completed_trades if t.pnl > 0]
            losing_trades = [t for t in completed_trades if t.pnl < 0]

            metrics.winning_trades = len(winning_trades)
            metrics.losing_trades = len(losing_trades)
            metrics.win_rate = len(winning_trades) / len(completed_trades)

            if len(winning_trades) > 0:
                metrics.avg_win = np.mean([t.pnl for t in winning_trades])
                metrics.largest_win = max([t.pnl for t in winning_trades])

            if len(losing_trades) > 0:
                metrics.avg_loss = np.mean([t.pnl for t in losing_trades])
                metrics.largest_loss = min([t.pnl for t in losing_trades])

            # Profit factor
            total_wins = sum([t.pnl for t in winning_trades])
            total_losses = abs(sum([t.pnl for t in losing_trades]))
            if total_losses > 0:
                metrics.profit_factor = total_wins / total_losses

            # Expectancy
            metrics.expectancy = np.mean([t.pnl for t in completed_trades])

            # Average holding period
            metrics.avg_holding_period = np.mean([t.holding_period for t in completed_trades])

            # Recovery factor
            if metrics.max_drawdown > 0:
                metrics.recovery_factor = metrics.total_return / metrics.max_drawdown

        return metrics


class WalkForwardAnalysis:
    """
    Walk-forward analysis for strategy validation.

    Splits data into multiple train/test windows and tests strategy robustness.
    """

    def __init__(self, config: BacktestConfig):
        """Initialize walk-forward analyzer."""
        self.config = config
        self.results: List[Dict[str, Any]] = []

    def run(
        self,
        data: pd.DataFrame,
        strategy: Callable[[pd.DataFrame], pd.Series]
    ) -> pd.DataFrame:
        """
        Run walk-forward analysis.

        Args:
            data: Historical OHLCV data
            strategy: Function that takes data and returns signals

        Returns:
            DataFrame with results for each window
        """
        train_window = self.config.train_window
        test_window = self.config.test_window
        step = self.config.walk_forward_step

        total_periods = len(data)
        results = []

        start_idx = 0
        while start_idx + train_window + test_window <= total_periods:
            # Define windows
            train_end_idx = start_idx + train_window
            test_end_idx = train_end_idx + test_window

            train_data = data.iloc[start_idx:train_end_idx]
            test_data = data.iloc[train_end_idx:test_end_idx]

            logger.info(f"Walk-forward window: train={train_data.index[0]} to {train_data.index[-1]}, "
                       f"test={test_data.index[0]} to {test_data.index[-1]}")

            # Generate signals on test data
            signals = strategy(test_data)

            # Run backtest
            engine = BacktestEngine(self.config)
            metrics, equity_curve = engine.run(test_data, signals)

            # Store results
            result = {
                'train_start': train_data.index[0],
                'train_end': train_data.index[-1],
                'test_start': test_data.index[0],
                'test_end': test_data.index[-1],
                'return': metrics.total_return,
                'sharpe': metrics.sharpe_ratio,
                'max_drawdown': metrics.max_drawdown,
                'win_rate': metrics.win_rate,
                'total_trades': metrics.total_trades
            }
            results.append(result)

            # Move to next window
            start_idx += step

        self.results = results
        return pd.DataFrame(results)

    def get_summary_stats(self) -> Dict[str, float]:
        """Get summary statistics across all windows."""
        if not self.results:
            return {}

        df = pd.DataFrame(self.results)

        return {
            'mean_return': df['return'].mean(),
            'std_return': df['return'].std(),
            'mean_sharpe': df['sharpe'].mean(),
            'mean_max_drawdown': df['max_drawdown'].mean(),
            'mean_win_rate': df['win_rate'].mean(),
            'consistency': (df['return'] > 0).sum() / len(df),
            'total_windows': len(df)
        }


class MonteCarloSimulation:
    """
    Monte Carlo simulation for strategy robustness testing.

    Simulates multiple scenarios using historical return distributions.
    """

    def __init__(self, config: BacktestConfig):
        """Initialize Monte Carlo simulator."""
        self.config = config
        self.simulations: List[np.ndarray] = []

    def run(
        self,
        returns: pd.Series,
        n_periods: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run Monte Carlo simulations.

        Args:
            returns: Historical returns
            n_periods: Number of periods to simulate (default: same as returns)

        Returns:
            Dictionary with simulation results
        """
        if n_periods is None:
            n_periods = len(returns)

        n_sims = self.config.n_simulations
        initial_capital = self.config.initial_capital

        # Calculate statistics from historical returns
        mean_return = returns.mean()
        std_return = returns.std()

        simulations = []
        final_values = []

        for i in range(n_sims):
            # Simulate returns
            simulated_returns = np.random.normal(mean_return, std_return, n_periods)

            # Calculate equity curve
            equity = initial_capital * (1 + simulated_returns).cumprod()
            simulations.append(equity)
            final_values.append(equity[-1])

        self.simulations = simulations

        # Calculate confidence intervals
        percentile_low = (1 - self.config.confidence_level) / 2 * 100
        percentile_high = (1 - (1 - self.config.confidence_level) / 2) * 100

        final_values_array = np.array(final_values)

        results = {
            'mean_final_value': np.mean(final_values_array),
            'median_final_value': np.median(final_values_array),
            'std_final_value': np.std(final_values_array),
            'min_final_value': np.min(final_values_array),
            'max_final_value': np.max(final_values_array),
            f'percentile_{percentile_low:.1f}': np.percentile(final_values_array, percentile_low),
            f'percentile_{percentile_high:.1f}': np.percentile(final_values_array, percentile_high),
            'probability_profit': (final_values_array > initial_capital).sum() / n_sims,
            'var_95': initial_capital - np.percentile(final_values_array, 5),
            'cvar_95': initial_capital - final_values_array[final_values_array <= np.percentile(final_values_array, 5)].mean()
        }

        return results

    def get_percentile_paths(self, percentiles: List[float]) -> Dict[float, np.ndarray]:
        """Get equity curves for specific percentiles."""
        if not self.simulations:
            return {}

        final_values = [sim[-1] for sim in self.simulations]
        paths = {}

        for p in percentiles:
            target_value = np.percentile(final_values, p)
            closest_idx = np.argmin([abs(fv - target_value) for fv in final_values])
            paths[p] = self.simulations[closest_idx]

        return paths


class StrategyComparison:
    """
    Compare multiple strategies side-by-side.
    """

    def __init__(self, config: BacktestConfig):
        """Initialize strategy comparator."""
        self.config = config
        self.results: Dict[str, Tuple[PerformanceMetrics, pd.DataFrame]] = {}

    def add_strategy(
        self,
        name: str,
        data: pd.DataFrame,
        signals: pd.Series
    ):
        """
        Add a strategy to compare.

        Args:
            name: Strategy name
            data: Historical data
            signals: Trading signals
        """
        engine = BacktestEngine(self.config)
        metrics, equity_curve = engine.run(data, signals)
        self.results[name] = (metrics, equity_curve)
        logger.info(f"Added strategy '{name}': Return={metrics.total_return:.2%}, Sharpe={metrics.sharpe_ratio:.2f}")

    def get_comparison_table(self) -> pd.DataFrame:
        """Get comparison table of all strategies."""
        if not self.results:
            return pd.DataFrame()

        comparison_data = []
        for name, (metrics, _) in self.results.items():
            row = {'Strategy': name}
            row.update(metrics.to_dict())
            comparison_data.append(row)

        return pd.DataFrame(comparison_data).set_index('Strategy')

    def get_best_strategy(self, metric: str = 'sharpe_ratio') -> Tuple[str, PerformanceMetrics]:
        """
        Get the best performing strategy.

        Args:
            metric: Metric to optimize for

        Returns:
            Tuple of (strategy name, metrics)
        """
        if not self.results:
            raise ValueError("No strategies added")

        best_name = None
        best_value = float('-inf')
        best_metrics = None

        for name, (metrics, _) in self.results.items():
            value = getattr(metrics, metric)
            if value > best_value:
                best_value = value
                best_name = name
                best_metrics = metrics

        return best_name, best_metrics


def save_backtest_results(
    metrics: PerformanceMetrics,
    equity_curve: pd.DataFrame,
    trades: List[Trade],
    output_dir: str = "output/backtests"
):
    """
    Save backtest results to files.

    Args:
        metrics: Performance metrics
        equity_curve: Equity curve DataFrame
        trades: List of trades
        output_dir: Output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save metrics
    metrics_file = output_path / f"metrics_{timestamp}.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics.to_dict(), f, indent=2)

    # Save equity curve
    equity_file = output_path / f"equity_curve_{timestamp}.csv"
    equity_curve.to_csv(equity_file)

    # Save trades
    if trades:
        trades_data = []
        for trade in trades:
            trades_data.append({
                'entry_date': trade.entry_date,
                'exit_date': trade.exit_date,
                'symbol': trade.symbol,
                'direction': trade.direction,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'shares': trade.shares,
                'pnl': trade.pnl,
                'return_pct': trade.return_pct,
                'holding_period': trade.holding_period,
                'exit_reason': trade.exit_reason
            })

        trades_df = pd.DataFrame(trades_data)
        trades_file = output_path / f"trades_{timestamp}.csv"
        trades_df.to_csv(trades_file, index=False)

    logger.info(f"Backtest results saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    prices = 100 * (1 + np.random.randn(len(dates)).cumsum() * 0.01)

    data = pd.DataFrame({
        'Close': prices,
        'Open': prices * (1 + np.random.randn(len(dates)) * 0.005),
        'High': prices * (1 + abs(np.random.randn(len(dates))) * 0.01),
        'Low': prices * (1 - abs(np.random.randn(len(dates))) * 0.01),
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    }, index=dates)

    # Simple moving average crossover strategy
    data['SMA_50'] = data['Close'].rolling(50).mean()
    data['SMA_200'] = data['Close'].rolling(200).mean()
    signals = pd.Series(0, index=data.index)
    signals[data['SMA_50'] > data['SMA_200']] = 1
    signals[data['SMA_50'] < data['SMA_200']] = -1

    # Run backtest
    config = BacktestConfig(initial_capital=100000)
    engine = BacktestEngine(config)
    metrics, equity_curve = engine.run(data, signals)

    print("\nBacktest Results:")
    print(f"Total Return: {metrics.total_return:.2%}")
    print(f"Annual Return: {metrics.annual_return:.2%}")
    print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {metrics.max_drawdown:.2%}")
    print(f"Win Rate: {metrics.win_rate:.2%}")
    print(f"Total Trades: {metrics.total_trades}")
