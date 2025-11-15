"""
Visualization Module for Stock Price Prediction
Handles all plotting and visual analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

logger = logging.getLogger(__name__)


class StockVisualizer:
    """Handles all visualization for stock prediction project"""

    def __init__(self, figure_size: Tuple[int, int] = (15, 8)):
        """
        Initialize visualizer

        Args:
            figure_size: Default figure size for plots
        """
        self.figure_size = figure_size

    def plot_stock_price(self, df: pd.DataFrame, symbol: str = "Stock",
                        indicators: List[str] = None) -> go.Figure:
        """
        Plot stock price with technical indicators

        Args:
            df: DataFrame with stock data
            symbol: Stock symbol for title
            indicators: List of indicators to plot

        Returns:
            Plotly figure
        """
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=(f'{symbol} Price', 'Volume', 'Technical Indicators'),
            row_heights=[0.5, 0.2, 0.3]
        )

        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df.index if 'Date' not in df.columns else df['Date'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Price'
            ),
            row=1, col=1
        )

        # Add moving averages if available
        if indicators:
            colors = ['blue', 'orange', 'green', 'red', 'purple']
            for i, indicator in enumerate(indicators[:5]):
                if indicator in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df.index if 'Date' not in df.columns else df['Date'],
                            y=df[indicator],
                            mode='lines',
                            name=indicator,
                            line=dict(color=colors[i % len(colors)], width=1)
                        ),
                        row=1, col=1
                    )

        # Volume chart
        colors = ['red' if row['Close'] < row['Open'] else 'green'
                 for idx, row in df.iterrows()]

        fig.add_trace(
            go.Bar(
                x=df.index if 'Date' not in df.columns else df['Date'],
                y=df['Volume'],
                name='Volume',
                marker_color=colors,
                showlegend=False
            ),
            row=2, col=1
        )

        # RSI if available
        if 'RSI_14' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index if 'Date' not in df.columns else df['Date'],
                    y=df['RSI_14'],
                    mode='lines',
                    name='RSI',
                    line=dict(color='purple', width=1)
                ),
                row=3, col=1
            )

            # Add RSI levels
            fig.add_hline(y=70, line_dash="dash", line_color="red",
                         row=3, col=1, opacity=0.5)
            fig.add_hline(y=30, line_dash="dash", line_color="green",
                         row=3, col=1, opacity=0.5)

        # Update layout
        fig.update_layout(
            title=f'{symbol} Stock Analysis',
            xaxis_title='Date',
            yaxis_title='Price',
            template='plotly_dark',
            height=800,
            showlegend=True,
            hovermode='x unified'
        )

        fig.update_xaxes(rangeslider_visible=False)

        return fig

    def plot_predictions(self, actual: np.ndarray, predictions: Dict[str, np.ndarray]) -> go.Figure:
        """
        Plot actual vs predicted values

        Args:
            actual: Actual values
            predictions: Dictionary of model predictions

        Returns:
            Plotly figure
        """
        fig = go.Figure()

        # Plot actual values
        fig.add_trace(go.Scatter(
            x=list(range(len(actual))),
            y=actual,
            mode='lines',
            name='Actual',
            line=dict(color='blue', width=2)
        ))

        # Plot predictions
        colors = ['red', 'green', 'orange', 'purple', 'brown']
        for i, (model_name, preds) in enumerate(predictions.items()):
            fig.add_trace(go.Scatter(
                x=list(range(len(preds))),
                y=preds,
                mode='lines',
                name=model_name,
                line=dict(color=colors[i % len(colors)], width=1)
            ))

        fig.update_layout(
            title='Model Predictions vs Actual',
            xaxis_title='Time Step',
            yaxis_title='Returns',
            template='plotly_dark',
            height=500,
            hovermode='x unified'
        )

        return fig

    def plot_model_comparison(self, comparison_data: pd.DataFrame) -> go.Figure:
        """
        Plot model performance comparison

        Args:
            comparison_data: DataFrame with model metrics

        Returns:
            Plotly figure
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('RMSE Comparison', 'R² Score Comparison',
                          'Directional Accuracy', 'Overfitting Analysis'),
            specs=[[{'type': 'bar'}, {'type': 'bar'}],
                   [{'type': 'bar'}, {'type': 'scatter'}]]
        )

        def _add_bar(name: str, column: str, row: int, col: int, showlegend: bool = True) -> bool:
            """Utility to add a bar trace when the column exists."""
            if column in comparison_data.columns:
                fig.add_trace(
                    go.Bar(name=name, x=comparison_data['Model'],
                           y=comparison_data[column], showlegend=showlegend),
                    row=row, col=col
                )
                return True
            return False

        # RMSE Comparison (fallback to overall RMSE if split metrics unavailable)
        _add_bar('Train RMSE', 'Train_RMSE', row=1, col=1)
        if not _add_bar('Val RMSE', 'Val_RMSE', row=1, col=1):
            _add_bar('Test RMSE', 'RMSE', row=1, col=1)

        # R² Score
        _add_bar('Train R²', 'Train_R2', row=1, col=2, showlegend=False)
        if not _add_bar('Val R²', 'Val_R2', row=1, col=2, showlegend=False):
            _add_bar('Test R²', 'R2', row=1, col=2, showlegend=False)

        # Directional Accuracy
        _add_bar('Train Accuracy', 'Train_DirectionalAcc', row=2, col=1, showlegend=False)
        if not _add_bar('Val Accuracy', 'Val_DirectionalAcc', row=2, col=1, showlegend=False):
            _add_bar('Directional Accuracy', 'Directional_Accuracy', row=2, col=1, showlegend=False)

        # Overfitting Analysis (only if both axes available)
        train_r2_col = 'Train_R2' if 'Train_R2' in comparison_data.columns else None
        val_r2_col = None
        if 'Val_R2' in comparison_data.columns:
            val_r2_col = 'Val_R2'
        elif 'R2' in comparison_data.columns:
            val_r2_col = 'R2'

        if train_r2_col and val_r2_col:
            fig.add_trace(
                go.Scatter(x=comparison_data[train_r2_col], y=comparison_data[val_r2_col],
                          mode='markers+text', text=comparison_data['Model'],
                          textposition='top center', showlegend=False,
                          marker=dict(size=10)),
                row=2, col=2
            )

            # Add diagonal reference line
            fig.add_trace(
                go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                          line=dict(dash='dash', color='gray'),
                          showlegend=False),
                row=2, col=2
            )
        else:
            fig.add_annotation(
                text='Insufficient data for overfitting plot',
                x=0.82, y=0.2, xref='paper', yref='paper',
                showarrow=False, font=dict(color='gray')
            )

        fig.update_layout(height=700, template='plotly_dark', title='Model Comparison')

        return fig

    def plot_feature_importance(self, importance_data: pd.DataFrame, top_n: int = 15) -> go.Figure:
        """
        Plot feature importance

        Args:
            importance_data: DataFrame with feature importance
            top_n: Number of top features to display

        Returns:
            Plotly figure
        """
        # Sort and take top N
        importance_sorted = importance_data.nlargest(top_n, 'importance')

        fig = go.Figure(data=[
            go.Bar(
                x=importance_sorted['importance'],
                y=importance_sorted['feature'],
                orientation='h',
                marker=dict(
                    color=importance_sorted['importance'],
                    colorscale='Viridis',
                    showscale=True
                )
            )
        ])

        fig.update_layout(
            title=f'Top {top_n} Feature Importance',
            xaxis_title='Importance',
            yaxis_title='Feature',
            template='plotly_dark',
            height=500
        )

        return fig

    def plot_backtesting_results(self, results: Dict) -> go.Figure:
        """
        Plot backtesting performance

        Args:
            results: Dictionary with backtesting results

        Returns:
            Plotly figure
        """
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            subplot_titles=('Portfolio Value', 'Returns', 'Drawdown'),
            row_heights=[0.5, 0.25, 0.25]
        )

        dates = results['dates']

        # Portfolio value
        fig.add_trace(
            go.Scatter(x=dates, y=results['portfolio_value'],
                      mode='lines', name='Portfolio',
                      line=dict(color='blue', width=2)),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=dates, y=results['benchmark_value'],
                      mode='lines', name='Benchmark',
                      line=dict(color='gray', width=1)),
            row=1, col=1
        )

        # Returns
        fig.add_trace(
            go.Bar(x=dates, y=results['returns'],
                  marker_color=['green' if r > 0 else 'red' for r in results['returns']],
                  name='Daily Returns', showlegend=False),
            row=2, col=1
        )

        # Drawdown
        fig.add_trace(
            go.Scatter(x=dates, y=results['drawdown'],
                      fill='tozeroy', fillcolor='rgba(255, 0, 0, 0.3)',
                      line=dict(color='red', width=1),
                      name='Drawdown', showlegend=False),
            row=3, col=1
        )

        fig.update_layout(
            height=800,
            template='plotly_dark',
            title='Backtesting Results',
            hovermode='x unified'
        )

        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_yaxes(title_text="Value ($)", row=1, col=1)
        fig.update_yaxes(title_text="Return (%)", row=2, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=3, col=1)

        return fig

    def plot_correlation_matrix(self, df: pd.DataFrame, features: List[str] = None) -> go.Figure:
        """
        Plot correlation heatmap

        Args:
            df: DataFrame with features
            features: List of features to include

        Returns:
            Plotly figure
        """
        if features:
            correlation = df[features].corr()
        else:
            correlation = df.corr()

        fig = go.Figure(data=go.Heatmap(
            z=correlation.values,
            x=correlation.columns,
            y=correlation.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(correlation.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title='Correlation')
        ))

        fig.update_layout(
            title='Feature Correlation Matrix',
            template='plotly_dark',
            height=700,
            width=900
        )

        return fig

    def plot_residuals(self, actual: np.ndarray, predicted: np.ndarray,
                      model_name: str = "Model") -> go.Figure:
        """
        Plot residual analysis

        Args:
            actual: Actual values
            predicted: Predicted values
            model_name: Name of the model

        Returns:
            Plotly figure
        """
        residuals = actual - predicted

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Residuals vs Predicted', 'Residuals Distribution',
                          'Q-Q Plot', 'Residuals Over Time'),
            specs=[[{'type': 'scatter'}, {'type': 'histogram'}],
                   [{'type': 'scatter'}, {'type': 'scatter'}]]
        )

        # Residuals vs Predicted
        fig.add_trace(
            go.Scatter(x=predicted, y=residuals, mode='markers',
                      marker=dict(size=5, color='blue', opacity=0.5),
                      showlegend=False),
            row=1, col=1
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)

        # Residuals Distribution
        fig.add_trace(
            go.Histogram(x=residuals, nbinsx=30, showlegend=False),
            row=1, col=2
        )

        # Q-Q Plot
        from scipy import stats
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(residuals)))
        sample_quantiles = np.sort(residuals)

        fig.add_trace(
            go.Scatter(x=theoretical_quantiles, y=sample_quantiles,
                      mode='markers', marker=dict(size=5),
                      showlegend=False),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=[-3, 3], y=[-3, 3], mode='lines',
                      line=dict(dash='dash', color='red'),
                      showlegend=False),
            row=2, col=1
        )

        # Residuals Over Time
        fig.add_trace(
            go.Scatter(y=residuals, mode='lines', showlegend=False),
            row=2, col=2
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=2)

        fig.update_layout(
            title=f'{model_name} - Residual Analysis',
            template='plotly_dark',
            height=700
        )

        fig.update_xaxes(title_text="Predicted", row=1, col=1)
        fig.update_yaxes(title_text="Residuals", row=1, col=1)
        fig.update_xaxes(title_text="Theoretical Quantiles", row=2, col=1)
        fig.update_yaxes(title_text="Sample Quantiles", row=2, col=1)
        fig.update_xaxes(title_text="Time", row=2, col=2)

        return fig

    def plot_training_history(self, history: Dict) -> go.Figure:
        """
        Plot training history for neural networks

        Args:
            history: Training history dictionary

        Returns:
            Plotly figure
        """
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Loss', 'Metrics'),
            shared_xaxes=True
        )

        # Loss
        if 'loss' in history:
            fig.add_trace(
                go.Scatter(y=history['loss'], mode='lines',
                          name='Train Loss', line=dict(color='blue')),
                row=1, col=1
            )

        if 'val_loss' in history:
            fig.add_trace(
                go.Scatter(y=history['val_loss'], mode='lines',
                          name='Val Loss', line=dict(color='red')),
                row=1, col=1
            )

        # Metrics (e.g., accuracy, mae)
        metric_keys = [k for k in history.keys()
                      if k not in ['loss', 'val_loss', 'epoch']]

        for key in metric_keys:
            if key.startswith('val_'):
                color = 'red'
                name = f'Val {key[4:]}'
            else:
                color = 'blue'
                name = f'Train {key}'

            fig.add_trace(
                go.Scatter(y=history[key], mode='lines',
                          name=name, line=dict(color=color)),
                row=2, col=1
            )

        fig.update_layout(
            title='Training History',
            template='plotly_dark',
            height=600,
            hovermode='x unified'
        )

        fig.update_xaxes(title_text="Epoch", row=2, col=1)
        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_yaxes(title_text="Metric Value", row=2, col=1)

        return fig


# Example usage
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)

    df = pd.DataFrame({
        'Date': dates,
        'Open': 100 + np.random.randn(len(dates)).cumsum(),
        'High': 102 + np.random.randn(len(dates)).cumsum(),
        'Low': 98 + np.random.randn(len(dates)).cumsum(),
        'Close': 100 + np.random.randn(len(dates)).cumsum(),
        'Volume': np.random.randint(1000000, 10000000, len(dates)),
        'SMA_20': 100 + np.random.randn(len(dates)).cumsum() * 0.8,
        'SMA_50': 100 + np.random.randn(len(dates)).cumsum() * 0.6,
        'RSI_14': 30 + np.random.rand(len(dates)) * 40
    })

    # Initialize visualizer
    viz = StockVisualizer()

    # Create plots
    fig1 = viz.plot_stock_price(df, 'SAMPLE', ['SMA_20', 'SMA_50'])
    fig1.show()

    # Sample predictions
    actual = np.random.randn(100)
    predictions = {
        'Linear Regression': actual + np.random.randn(100) * 0.1,
        'Random Forest': actual + np.random.randn(100) * 0.15
    }

    fig2 = viz.plot_predictions(actual, predictions)
    fig2.show()
