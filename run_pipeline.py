"""Utility to run the stock prediction pipeline without argparse."""

from datetime import datetime

from main import StockPredictionPipeline
import config


def run_default(symbol: str = "AAPL",
                start_date: str | None = None,
                end_date: str | None = None,
                launch_dashboard: bool = False):
    pipeline = StockPredictionPipeline(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
    )

    performance = pipeline.run_pipeline()
    print("\nModel performance summary:\n", performance)

    if launch_dashboard:
        from dashboard import app
        app.run(
            debug=config.DASHBOARD_CONFIG['debug'],
            host=config.DASHBOARD_CONFIG['host'],
            port=config.DASHBOARD_CONFIG['port']
        )


if __name__ == "__main__":
    run_default()
