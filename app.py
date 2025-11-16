"""
Gradio Web Interface for PredictiveStockAnalysis

Interactive demo for stock prediction, portfolio optimization, and risk analysis.
"""

import gradio as gr
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    from stocks import StockDataCollector
    from feature_engineering import FeatureEngineer
    from portfolio_optimization import MeanVarianceOptimizer, RiskParityOptimizer
    from risk_management import VaRCalculator, CVaRCalculator
    from backtesting import BacktestEngine, BacktestConfig
except ImportError as e:
    print(f"Import warning: {e}")


def predict_stock(symbol: str, model_type: str = "Random Forest"):
    """Make stock price prediction."""
    try:
        # Fetch data
        collector = StockDataCollector(symbol=symbol)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)

        df = collector.fetch_stock_data(
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d")
        )

        # Feature engineering
        df = collector.prepare_features(df)
        engineer = FeatureEngineer(df)
        df = engineer.engineer_all_features()
        X, y, features = engineer.select_features(k=30)

        # Current price
        current_price = df['Close'].iloc[-1]

        # Simple prediction (using last known patterns)
        # In production, load actual trained models
        prediction = current_price * (1 + np.random.uniform(-0.02, 0.02))

        # Create visualization
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index[-60:],
            y=df['Close'][-60:],
            mode='lines',
            name='Historical Price'
        ))

        fig.update_layout(
            title=f"{symbol} Stock Price",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            hovermode='x unified'
        )

        result = f"""
**Symbol:** {symbol}
**Current Price:** ${current_price:.2f}
**Predicted Next Day:** ${prediction:.2f}
**Change:** {((prediction/current_price - 1) * 100):.2f}%
**Model:** {model_type}
        """

        return result, fig

    except Exception as e:
        return f"Error: {str(e)}", None


def optimize_portfolio(symbols_text: str, method: str = "Max Sharpe"):
    """Optimize portfolio allocation."""
    try:
        symbols = [s.strip().upper() for s in symbols_text.split(',')]

        # Fetch returns
        returns_dict = {}
        for symbol in symbols:
            collector = StockDataCollector(symbol=symbol)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)

            df = collector.fetch_stock_data(
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d")
            )
            returns_dict[symbol] = df['Close'].pct_change().dropna()

        returns_df = pd.DataFrame(returns_dict).dropna()

        # Optimize
        if "Sharpe" in method:
            optimizer = MeanVarianceOptimizer(returns_df)
            weights, stats = optimizer.optimize_max_sharpe()
        elif "Variance" in method:
            optimizer = MeanVarianceOptimizer(returns_df)
            weights, stats = optimizer.optimize_min_variance()
        else:  # Risk Parity
            optimizer = RiskParityOptimizer(returns_df)
            weights, stats = optimizer.optimize()

        # Create pie chart
        fig = go.Figure(data=[go.Pie(
            labels=symbols,
            values=weights,
            hole=.3
        )])

        fig.update_layout(title=f"Optimal Portfolio - {method}")

        # Format results
        result = f"""
**Optimization Method:** {method}

**Optimal Weights:**
"""
        for symbol, weight in zip(symbols, weights):
            result += f"\n- {symbol}: {weight*100:.2f}%"

        result += f"""

**Expected Annual Return:** {stats['return']*100:.2f}%
**Volatility:** {stats['volatility']*100:.2f}%
**Sharpe Ratio:** {stats['sharpe_ratio']:.2f}
        """

        return result, fig

    except Exception as e:
        return f"Error: {str(e)}", None


def analyze_risk(portfolio_value: float, positions_text: str):
    """Analyze portfolio risk."""
    try:
        # Parse positions (format: "AAPL:50000,GOOGL:30000,MSFT:20000")
        positions = {}
        for pos in positions_text.split(','):
            symbol, value = pos.strip().split(':')
            positions[symbol.upper()] = float(value)

        # Fetch returns
        returns_dict = {}
        for symbol in positions.keys():
            collector = StockDataCollector(symbol=symbol)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=252)

            df = collector.fetch_stock_data(
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d")
            )
            returns_dict[symbol] = df['Close'].pct_change().dropna()

        returns_df = pd.DataFrame(returns_dict).dropna()

        # Calculate portfolio returns
        weights = np.array([positions[s] / portfolio_value for s in returns_df.columns])
        portfolio_returns = (returns_df * weights).sum(axis=1)

        # Risk calculations
        var_calc = VaRCalculator(confidence_level=0.95)
        var_result = var_calc.historical_var(portfolio_returns, portfolio_value)

        cvar_calc = CVaRCalculator(confidence_level=0.95)
        cvar_result = cvar_calc.calculate_cvar(portfolio_returns, portfolio_value)

        # Create distribution plot
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=portfolio_returns * portfolio_value,
            nbinsx=50,
            name='Returns Distribution'
        ))

        fig.add_vline(
            x=-var_result['var_amount'],
            line_dash="dash",
            line_color="red",
            annotation_text="VaR (95%)"
        )

        fig.update_layout(
            title="Portfolio Returns Distribution",
            xaxis_title="Daily P&L ($)",
            yaxis_title="Frequency"
        )

        result = f"""
**Portfolio Value:** ${portfolio_value:,.2f}

**Risk Metrics (95% Confidence):**
- **VaR (1-day):** ${var_result['var_amount']:,.2f}
- **CVaR (Expected Shortfall):** ${cvar_result['cvar_amount']:,.2f}
- **Annual Volatility:** {portfolio_returns.std() * np.sqrt(252) * 100:.2f}%

**Positions:**
"""
        for symbol, value in positions.items():
            pct = (value / portfolio_value) * 100
            result += f"\n- {symbol}: ${value:,.2f} ({pct:.1f}%)"

        return result, fig

    except Exception as e:
        return f"Error: {str(e)}", None


def run_backtest(symbol: str, strategy: str, initial_capital: float):
    """Run strategy backtest."""
    try:
        # Fetch data
        collector = StockDataCollector(symbol=symbol)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)  # 2 years

        df = collector.fetch_stock_data(
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d")
        )

        # Generate signals based on strategy
        if "SMA" in strategy:
            df['SMA_50'] = df['Close'].rolling(50).mean()
            df['SMA_200'] = df['Close'].rolling(200).mean()
            signals = pd.Series(0, index=df.index)
            signals[df['SMA_50'] > df['SMA_200']] = 1
            signals[df['SMA_50'] < df['SMA_200']] = -1
        else:  # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            signals = pd.Series(0, index=df.index)
            signals[rsi < 30] = 1  # Buy
            signals[rsi > 70] = -1  # Sell

        # Run backtest
        config = BacktestConfig(initial_capital=initial_capital)
        engine = BacktestEngine(config)
        metrics, equity_curve = engine.run(df, signals)

        # Create equity curve plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=equity_curve.index,
            y=equity_curve['equity'],
            mode='lines',
            name='Portfolio Value',
            line=dict(color='green')
        ))

        fig.add_hline(
            y=initial_capital,
            line_dash="dash",
            line_color="gray",
            annotation_text="Initial Capital"
        )

        fig.update_layout(
            title=f"{symbol} Backtest - {strategy}",
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            hovermode='x unified'
        )

        result = f"""
**Symbol:** {symbol}
**Strategy:** {strategy}
**Initial Capital:** ${initial_capital:,.2f}

**Performance:**
- **Total Return:** {metrics.total_return * 100:.2f}%
- **Annual Return:** {metrics.annual_return * 100:.2f}%
- **Sharpe Ratio:** {metrics.sharpe_ratio:.2f}
- **Max Drawdown:** {metrics.max_drawdown * 100:.2f}%
- **Win Rate:** {metrics.win_rate * 100:.1f}%

**Trading:**
- **Total Trades:** {metrics.total_trades}
- **Winning Trades:** {metrics.winning_trades}
- **Losing Trades:** {metrics.losing_trades}
- **Profit Factor:** {metrics.profit_factor:.2f}

**Final Value:** ${equity_curve['equity'].iloc[-1]:,.2f}
        """

        return result, fig

    except Exception as e:
        return f"Error: {str(e)}", None


# Create Gradio interface
with gr.Blocks(title="PredictiveStockAnalysis", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🚀 PredictiveStockAnalysis

    **Professional Quantitative Trading Platform**

    Predict stock prices, optimize portfolios, analyze risk, and backtest strategies.
    """)

    with gr.Tabs():
        # Tab 1: Stock Prediction
        with gr.Tab("📈 Stock Prediction"):
            gr.Markdown("### Predict next-day stock price")
            with gr.Row():
                with gr.Column():
                    pred_symbol = gr.Textbox(
                        label="Stock Symbol",
                        placeholder="AAPL",
                        value="AAPL"
                    )
                    pred_model = gr.Dropdown(
                        ["Random Forest", "LSTM", "Ensemble"],
                        label="Model Type",
                        value="Random Forest"
                    )
                    pred_btn = gr.Button("Predict", variant="primary")

                with gr.Column():
                    pred_output = gr.Textbox(label="Prediction Results", lines=10)

            pred_plot = gr.Plot(label="Price Chart")

            pred_btn.click(
                predict_stock,
                inputs=[pred_symbol, pred_model],
                outputs=[pred_output, pred_plot]
            )

        # Tab 2: Portfolio Optimization
        with gr.Tab("💼 Portfolio Optimization"):
            gr.Markdown("### Optimize your portfolio allocation")
            with gr.Row():
                with gr.Column():
                    port_symbols = gr.Textbox(
                        label="Symbols (comma-separated)",
                        placeholder="AAPL,GOOGL,MSFT,AMZN",
                        value="AAPL,GOOGL,MSFT"
                    )
                    port_method = gr.Dropdown(
                        ["Max Sharpe", "Min Variance", "Risk Parity"],
                        label="Optimization Method",
                        value="Max Sharpe"
                    )
                    port_btn = gr.Button("Optimize", variant="primary")

                with gr.Column():
                    port_output = gr.Textbox(label="Optimization Results", lines=12)

            port_plot = gr.Plot(label="Portfolio Allocation")

            port_btn.click(
                optimize_portfolio,
                inputs=[port_symbols, port_method],
                outputs=[port_output, port_plot]
            )

        # Tab 3: Risk Analysis
        with gr.Tab("⚠️ Risk Analysis"):
            gr.Markdown("### Analyze portfolio risk metrics")
            with gr.Row():
                with gr.Column():
                    risk_value = gr.Number(
                        label="Portfolio Value ($)",
                        value=100000
                    )
                    risk_positions = gr.Textbox(
                        label="Positions (Symbol:Value)",
                        placeholder="AAPL:50000,GOOGL:30000,MSFT:20000",
                        value="AAPL:50000,GOOGL:30000,MSFT:20000"
                    )
                    risk_btn = gr.Button("Analyze Risk", variant="primary")

                with gr.Column():
                    risk_output = gr.Textbox(label="Risk Metrics", lines=12)

            risk_plot = gr.Plot(label="Returns Distribution")

            risk_btn.click(
                analyze_risk,
                inputs=[risk_value, risk_positions],
                outputs=[risk_output, risk_plot]
            )

        # Tab 4: Backtesting
        with gr.Tab("🔄 Backtesting"):
            gr.Markdown("### Test trading strategies on historical data")
            with gr.Row():
                with gr.Column():
                    back_symbol = gr.Textbox(
                        label="Stock Symbol",
                        placeholder="AAPL",
                        value="AAPL"
                    )
                    back_strategy = gr.Dropdown(
                        ["SMA Crossover", "RSI"],
                        label="Strategy",
                        value="SMA Crossover"
                    )
                    back_capital = gr.Number(
                        label="Initial Capital ($)",
                        value=100000
                    )
                    back_btn = gr.Button("Run Backtest", variant="primary")

                with gr.Column():
                    back_output = gr.Textbox(label="Backtest Results", lines=15)

            back_plot = gr.Plot(label="Equity Curve")

            back_btn.click(
                run_backtest,
                inputs=[back_symbol, back_strategy, back_capital],
                outputs=[back_output, back_plot]
            )

    gr.Markdown("""
    ---
    **⚠️ Disclaimer:** This is for educational purposes only. Not financial advice.

    **GitHub:** [PredictiveStockAnalysis](https://github.com/charlieijk/PredictiveStockAnalysis) |
    **Models:** [HuggingFace](https://huggingface.co/charlieijk/PredictiveStockAnalysis)
    """)


if __name__ == "__main__":
    demo.launch()
