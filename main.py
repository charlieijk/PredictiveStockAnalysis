"""
Main Pipeline for Stock Price Prediction Project
Orchestrates data collection, feature engineering, model training, and evaluation
"""

import argparse
import logging
import json
import os
from datetime import datetime
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from stocks import StockDataCollector
from feature_engineering import FeatureEngineer
from models import StockPredictionModels
from visualization import StockVisualizer
import config

# Setup logging
logging.basicConfig(
    level=config.LOGGING_CONFIG['level'],
    format=config.LOGGING_CONFIG['format'],
    handlers=[
        logging.FileHandler(config.LOGGING_CONFIG['log_file']),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class StockPredictionPipeline:
    """Main pipeline for stock prediction workflow"""
    
    def __init__(self, symbol: str, start_date: str = None, end_date: str = None):
        """
        Initialize pipeline
        
        Args:
            symbol: Stock symbol
            start_date: Start date for data collection
            end_date: End date for data collection
        """
        self.symbol = symbol
        self.start_date = start_date or config.DATA_CONFIG['start_date']
        self.end_date = end_date or config.DATA_CONFIG['end_date']
        
        # Initialize components
        self.collector = StockDataCollector(symbol, start_date, end_date)
        self.engineer = FeatureEngineer(scaling_method=config.FEATURE_CONFIG['scaling_method'])
        self.trainer = StockPredictionModels(config.MODEL_CONFIG)
        self.visualizer = StockVisualizer()
        
        # Storage for results
        self.raw_data = None
        self.engineered_data = None
        self.predictions = {}
        self.performance = {}
        
    def run_pipeline(self):
        """Run the complete pipeline"""
        logger.info(f"Starting pipeline for {self.symbol}")
        logger.info("=" * 50)
        
        # Step 1: Data Collection
        self.collect_data()
        
        # Step 2: Feature Engineering
        self.engineer_features()
        
        # Step 3: Model Training
        self.train_models()
        
        # Step 4: Evaluation
        self.evaluate_models()
        
        # Step 5: Visualization
        self.create_visualizations()
        
        # Step 6: Save Results
        self.save_results()
        
        logger.info("=" * 50)
        logger.info("Pipeline completed successfully!")
        
        return self.performance
    
    def collect_data(self):
        """Collect and prepare stock data"""
        logger.info("Step 1: Collecting stock data...")
        
        try:
            # Fetch raw data
            self.raw_data = self.collector.fetch_stock_data()
            logger.info(f"Collected {len(self.raw_data)} days of data")
            
            # Calculate technical indicators
            self.collector.calculate_moving_averages()
            self.collector.calculate_rsi()
            self.collector.calculate_macd()
            self.collector.calculate_bollinger_bands()
            self.collector.calculate_volatility()
            self.collector.calculate_volume_indicators()
            self.collector.add_price_features()
            
            self.raw_data = self.collector.data
            
            # Save raw data
            data_path = os.path.join(config.DATA_DIR, 
                                     f"{self.symbol}_raw_data_{datetime.now().strftime('%Y%m%d')}.csv")
            self.raw_data.to_csv(data_path, index=False)
            logger.info(f"Raw data saved to {data_path}")
            
        except Exception as e:
            logger.error(f"Error in data collection: {str(e)}")
            raise
    
    def engineer_features(self):
        """Engineer features for model training"""
        logger.info("Step 2: Engineering features...")
        
        try:
            # Create advanced features
            self.engineered_data = self.engineer.engineer_all_features(
                self.raw_data,
                target_col='Close',
                sequence_length=config.DATA_CONFIG['sequence_length']
            )
            
            logger.info(f"Created {len(self.engineered_data['feature_names'])} features")
            logger.info(f"Selected top {len(self.engineered_data['selected_features'])} features")
            
            # Save feature names
            features_path = os.path.join(config.DATA_DIR, 
                                        f"{self.symbol}_features_{datetime.now().strftime('%Y%m%d')}.json")
            with open(features_path, 'w') as f:
                json.dump({
                    'all_features': self.engineered_data['feature_names'],
                    'selected_features': self.engineered_data['selected_features']
                }, f, indent=4)
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {str(e)}")
            raise
    
    def train_models(self):
        """Train all models"""
        logger.info("Step 3: Training models...")
        
        # Prepare data splits
        features = self.engineered_data['features']
        target = self.engineered_data['target']
        features_lstm = self.engineered_data['features_lstm']
        target_lstm = self.engineered_data['target_lstm']
        
        # Calculate split indices
        train_size = int(len(features) * (1 - config.DATA_CONFIG['test_size']))
        val_size = int(train_size * config.DATA_CONFIG['validation_size'])
        
        # Traditional ML data split
        X_train = features[:train_size-val_size].values
        y_train = target[:train_size-val_size].values
        X_val = features[train_size-val_size:train_size].values
        y_val = target[train_size-val_size:train_size].values
        X_test = features[train_size:].values
        y_test = target[train_size:].values
        
        # LSTM data split
        lstm_train_size = int(len(features_lstm) * (1 - config.DATA_CONFIG['test_size']))
        lstm_val_size = int(lstm_train_size * config.DATA_CONFIG['validation_size'])
        
        X_train_lstm = features_lstm[:lstm_train_size-lstm_val_size]
        y_train_lstm = target_lstm[:lstm_train_size-lstm_val_size]
        X_val_lstm = features_lstm[lstm_train_size-lstm_val_size:lstm_train_size]
        y_val_lstm = target_lstm[lstm_train_size-lstm_val_size:lstm_train_size]
        X_test_lstm = features_lstm[lstm_train_size:]
        y_test_lstm = target_lstm[lstm_train_size:]
        
        # Store test data for later evaluation
        self.X_test = X_test
        self.y_test = y_test
        self.X_test_lstm = X_test_lstm
        self.y_test_lstm = y_test_lstm
        
        # Train models
        models_to_train = [
            ('Linear Regression', 'linear'),
            ('Random Forest', 'random_forest'),
            ('Gradient Boosting', 'gradient_boosting')
        ]
        
        for model_name, model_type in models_to_train:
            try:
                logger.info(f"Training {model_name}...")
                
                if model_type == 'linear':
                    self.trainer.train_linear_regression(X_train, y_train, X_val, y_val)
                elif model_type == 'random_forest':
                    self.trainer.train_random_forest(X_train, y_train, X_val, y_val)
                elif model_type == 'gradient_boosting':
                    self.trainer.train_gradient_boosting(X_train, y_train, X_val, y_val)
                
                logger.info(f"{model_name} training completed")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
        
        # Train LSTM if enabled
        if config.TRAINING_CONFIG.get('train_lstm', True):
            try:
                logger.info("Training LSTM model...")
                self.trainer.train_lstm(X_train_lstm, y_train_lstm, 
                                       X_val_lstm, y_val_lstm)
                logger.info("LSTM training completed")
            except Exception as e:
                logger.error(f"Error training LSTM: {str(e)}")
        
        # Train ensemble
        try:
            logger.info("Training Ensemble model...")
            self.trainer.train_ensemble(X_train, y_train, X_val, y_val)
            logger.info("Ensemble training completed")
        except Exception as e:
            logger.error(f"Error training Ensemble: {str(e)}")
    
    def evaluate_models(self):
        """Evaluate all trained models"""
        logger.info("Step 4: Evaluating models...")
        
        # Make predictions
        for model_name in self.trainer.models.keys():
            try:
                if 'lstm' in model_name.lower() or 'gru' in model_name.lower():
                    self.predictions[model_name] = self.trainer.predict(
                        model_name, self.X_test_lstm
                    )
                else:
                    self.predictions[model_name] = self.trainer.predict(
                        model_name, self.X_test
                    )
                
                logger.info(f"Generated predictions for {model_name}")
                
            except Exception as e:
                logger.error(f"Error predicting with {model_name}: {str(e)}")
        
        # Calculate performance metrics for each model
        comparison_data = []
        for model_name, predictions in self.predictions.items():
            if 'lstm' in model_name.lower():
                metrics = self.trainer.calculate_metrics(self.y_test_lstm, predictions)
            else:
                metrics = self.trainer.calculate_metrics(self.y_test, predictions)

            comparison_data.append({
                'Model': model_name,
                'RMSE': metrics['rmse'],
                'MAE': metrics['mae'],
                'R2': metrics['r2'],
                'Directional_Accuracy': metrics['directional_accuracy']
            })

        self.performance = pd.DataFrame(comparison_data).sort_values('RMSE')

        
        logger.info("\nModel Performance Summary:")
        logger.info("-" * 50)
        print(self.performance.to_string())
        
        # Identify best model
        best_model = self.performance.iloc[0]['Model']
        logger.info(f"\nBest performing model: {best_model}")
        
        # Save model comparison
        comparison_path = os.path.join(config.OUTPUT_DIR,
                                      f"{self.symbol}_model_comparison_{datetime.now().strftime('%Y%m%d')}.csv")
        self.performance.to_csv(comparison_path, index=False)
        
    def create_visualizations(self):
        """Create and save visualizations"""
        logger.info("Step 5: Creating visualizations...")
        
        try:
            # Prepare data for visualization
            viz_data = {
                'stock_data': self.raw_data,
                'symbol': self.symbol,
                'indicators': ['SMA_20', 'SMA_50', 'SMA_200'],
                'predictions': self.predictions,
                'actual': self.y_test if len(self.predictions) > 0 else None,
                'comparison': self.performance,
                'training_history': self.trainer.models.get('lstm_history', {})
            }
            
            # Create visualizations
            # Stock price plot
            stock_fig = self.visualizer.plot_stock_price(
                self.raw_data,
                self.symbol,
                ['SMA_20', 'SMA_50', 'SMA_200']
            )

            # Predictions plot
            if self.predictions:
                predictions_fig = self.visualizer.plot_predictions(
                    self.y_test,
                    self.predictions
                )

            # Model comparison
            if not self.performance.empty:
                comparison_fig = self.visualizer.plot_model_comparison(self.performance)
            
            logger.info(f"Visualizations saved to {config.OUTPUT_DIR}")
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")
    
    def save_results(self):
        """Save all results and models"""
        logger.info("Step 6: Saving results...")
        
        # Save models
        for model_name in self.trainer.models.keys():
            try:
                model_path = os.path.join(config.MODEL_DIR,
                                         f"{self.symbol}_{model_name}_{datetime.now().strftime('%Y%m%d')}")
                self.trainer.save_model(model_name, model_path)
                logger.info(f"Saved {model_name} model")
            except Exception as e:
                logger.error(f"Error saving {model_name}: {str(e)}")
        
        # Save predictions
        if self.predictions:
            predictions_df = pd.DataFrame(self.predictions)
            predictions_df['actual'] = self.y_test[:len(predictions_df)]
            predictions_path = os.path.join(config.OUTPUT_DIR,
                                           f"{self.symbol}_predictions_{datetime.now().strftime('%Y%m%d')}.csv")
            predictions_df.to_csv(predictions_path, index=False)
            logger.info(f"Predictions saved to {predictions_path}")
        
        # Save summary report
        self.generate_report()
    
    def generate_report(self):
        """Generate summary report"""
        report_path = os.path.join(config.OUTPUT_DIR,
                                  f"{self.symbol}_report_{datetime.now().strftime('%Y%m%d')}.txt")
        
        with open(report_path, 'w') as f:
            f.write(f"Stock Price Prediction Report\n")
            f.write(f"{'=' * 50}\n\n")
            f.write(f"Symbol: {self.symbol}\n")
            f.write(f"Date Range: {self.start_date} to {self.end_date}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"Data Statistics:\n")
            f.write(f"- Total days: {len(self.raw_data)}\n")
            f.write(f"- Features created: {len(self.engineered_data['feature_names'])}\n")
            f.write(f"- Features selected: {len(self.engineered_data['selected_features'])}\n\n")
            
            f.write(f"Model Performance:\n")
            f.write(self.performance.to_string())
            f.write(f"\n\nBest Model: {self.performance.iloc[0]['Model']}\n")
            
            f.write(f"\nTop 10 Selected Features:\n")
            for i, feature in enumerate(self.engineered_data['selected_features'][:10]):
                f.write(f"  {i+1}. {feature}\n")

        logger.info(f"Report saved to {report_path}")


def main():
    """Main function to run the pipeline"""
    parser = argparse.ArgumentParser(description='Stock Price Prediction Pipeline')

    parser.add_argument('symbol', type=str, help='Stock symbol (e.g., AAPL, GOOGL)')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--dashboard', action='store_true', help='Launch dashboard after training')

    args = parser.parse_args()

    # Create pipeline
    pipeline = StockPredictionPipeline(
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date
    )

    # Run pipeline
    try:
        results = pipeline.run_pipeline()

        # Launch dashboard if requested
        if args.dashboard:
            logger.info("Launching dashboard...")
            from dashboard import app
            app.run(
                debug=config.DASHBOARD_CONFIG['debug'],
                host=config.DASHBOARD_CONFIG['host'],
                port=config.DASHBOARD_CONFIG['port']
            )

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()