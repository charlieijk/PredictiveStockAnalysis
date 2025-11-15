"""
Machine Learning Models Module
Contains implementations of various ML models for stock price prediction
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any
import logging
import joblib
import os

# Sklearn imports
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler

# TensorFlow/Keras imports (with error handling)
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logging.warning("TensorFlow not available. LSTM models will not work.")

logger = logging.getLogger(__name__)


class StockPredictionModels:
    """Handles training and prediction for various ML models"""

    def __init__(self, config: Dict = None):
        """
        Initialize the model trainer

        Args:
            config: Configuration dictionary for models
        """
        self.config = config or {}
        self.models = {}
        self.scalers = {}
        self.metrics = {}
        self.histories = {}
        self.model_dir = 'models'
        os.makedirs(self.model_dir, exist_ok=True)

    def train_linear_regression(self, X_train: np.ndarray, y_train: np.ndarray,
                                X_val: np.ndarray = None, y_val: np.ndarray = None,
                                model_type: str = 'linear') -> LinearRegression:
        """
        Train linear regression model

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            model_type: Type of linear model ('linear', 'ridge', 'lasso')

        Returns:
            Trained model
        """
        logger.info(f"Training {model_type} regression model")

        # Select model type
        if model_type == 'ridge':
            model = Ridge(alpha=self.config.get('ridge_alpha', 1.0))
        elif model_type == 'lasso':
            model = Lasso(alpha=self.config.get('lasso_alpha', 1.0))
        else:
            model = LinearRegression()

        # Train model
        model.fit(X_train, y_train)

        # Store model
        self.models['linear_regression'] = model

        # Evaluate if validation data provided
        if X_val is not None and y_val is not None:
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)

            self.metrics['linear_regression'] = {
                'train': self.calculate_metrics(y_train, train_pred),
                'val': self.calculate_metrics(y_val, val_pred)
            }

            logger.info(f"Linear Regression - Train RMSE: {self.metrics['linear_regression']['train']['rmse']:.4f}")
            logger.info(f"Linear Regression - Val RMSE: {self.metrics['linear_regression']['val']['rmse']:.4f}")

        return model

    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray = None, y_val: np.ndarray = None) -> RandomForestRegressor:
        """
        Train Random Forest model

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets

        Returns:
            Trained model
        """
        logger.info("Training Random Forest model")

        # Get config or use defaults
        rf_config = self.config.get('random_forest', {})

        model = RandomForestRegressor(
            n_estimators=rf_config.get('n_estimators', 100),
            max_depth=rf_config.get('max_depth', 15),
            min_samples_split=rf_config.get('min_samples_split', 5),
            min_samples_leaf=rf_config.get('min_samples_leaf', 2),
            max_features=rf_config.get('max_features', 'sqrt'),
            random_state=rf_config.get('random_state', 42),
            n_jobs=rf_config.get('n_jobs', -1)
        )

        # Train model
        model.fit(X_train, y_train)

        # Store model
        self.models['random_forest'] = model

        # Evaluate if validation data provided
        if X_val is not None and y_val is not None:
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)

            self.metrics['random_forest'] = {
                'train': self.calculate_metrics(y_train, train_pred),
                'val': self.calculate_metrics(y_val, val_pred)
            }

            logger.info(f"Random Forest - Train RMSE: {self.metrics['random_forest']['train']['rmse']:.4f}")
            logger.info(f"Random Forest - Val RMSE: {self.metrics['random_forest']['val']['rmse']:.4f}")

        return model

    def train_gradient_boosting(self, X_train: np.ndarray, y_train: np.ndarray,
                               X_val: np.ndarray = None, y_val: np.ndarray = None) -> GradientBoostingRegressor:
        """
        Train Gradient Boosting model

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets

        Returns:
            Trained model
        """
        logger.info("Training Gradient Boosting model")

        # Get config or use defaults
        gb_config = self.config.get('gradient_boosting', {})

        model = GradientBoostingRegressor(
            n_estimators=gb_config.get('n_estimators', 100),
            learning_rate=gb_config.get('learning_rate', 0.1),
            max_depth=gb_config.get('max_depth', 5),
            min_samples_split=gb_config.get('min_samples_split', 5),
            min_samples_leaf=gb_config.get('min_samples_leaf', 2),
            subsample=gb_config.get('subsample', 0.8),
            random_state=gb_config.get('random_state', 42)
        )

        # Train model
        model.fit(X_train, y_train)

        # Store model
        self.models['gradient_boosting'] = model

        # Evaluate if validation data provided
        if X_val is not None and y_val is not None:
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)

            self.metrics['gradient_boosting'] = {
                'train': self.calculate_metrics(y_train, train_pred),
                'val': self.calculate_metrics(y_val, val_pred)
            }

            logger.info(f"Gradient Boosting - Train RMSE: {self.metrics['gradient_boosting']['train']['rmse']:.4f}")
            logger.info(f"Gradient Boosting - Val RMSE: {self.metrics['gradient_boosting']['val']['rmse']:.4f}")

        return model

    def train_lstm(self, X_train: np.ndarray, y_train: np.ndarray,
                  X_val: np.ndarray = None, y_val: np.ndarray = None) -> Optional[Any]:
        """
        Train LSTM model

        Args:
            X_train: Training features (3D array for LSTM)
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets

        Returns:
            Trained model or None if TensorFlow not available
        """
        if not TF_AVAILABLE:
            logger.error("TensorFlow not available. Cannot train LSTM model.")
            return None

        logger.info("Training LSTM model")

        # Get config or use defaults
        lstm_config = self.config.get('lstm', {})

        # Build model
        model = Sequential()

        # LSTM layers
        units = lstm_config.get('units', [128, 64, 32])
        dropout = lstm_config.get('dropout', 0.2)

        for i, unit in enumerate(units):
            if i == 0:
                model.add(LSTM(unit, return_sequences=(i < len(units) - 1),
                              input_shape=(X_train.shape[1], X_train.shape[2])))
            else:
                model.add(LSTM(unit, return_sequences=(i < len(units) - 1)))

            model.add(Dropout(dropout))

        # Dense layers
        dense_units = lstm_config.get('dense_units', [32, 16])
        for unit in dense_units:
            model.add(Dense(unit, activation='relu'))
            model.add(Dropout(dropout / 2))

        # Output layer
        model.add(Dense(1))

        # Compile model
        optimizer = Adam(learning_rate=lstm_config.get('learning_rate', 0.001))
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

        # Callbacks
        callbacks = [
            EarlyStopping(
                patience=lstm_config.get('early_stopping_patience', 15),
                restore_best_weights=True,
                monitor='val_loss' if X_val is not None else 'loss'
            ),
            ReduceLROnPlateau(
                patience=lstm_config.get('reduce_lr_patience', 10),
                factor=lstm_config.get('reduce_lr_factor', 0.5),
                min_lr=lstm_config.get('min_lr', 0.00001),
                monitor='val_loss' if X_val is not None else 'loss'
            )
        ]

        # Add model checkpoint
        checkpoint_path = os.path.join(self.model_dir, 'lstm_best.h5')
        callbacks.append(
            ModelCheckpoint(
                checkpoint_path,
                save_best_only=True,
                monitor='val_loss' if X_val is not None else 'loss'
            )
        )

        # Train model
        validation_data = (X_val, y_val) if X_val is not None else None
        history = model.fit(
            X_train, y_train,
            epochs=lstm_config.get('epochs', 100),
            batch_size=lstm_config.get('batch_size', 32),
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=lstm_config.get('verbose', 1)
        )

        # Store model and history separately to avoid treating history as a model
        self.models['lstm'] = model
        self.histories['lstm'] = history.history

        # Evaluate if validation data provided
        if X_val is not None and y_val is not None:
            train_pred = model.predict(X_train).flatten()
            val_pred = model.predict(X_val).flatten()

            self.metrics['lstm'] = {
                'train': self.calculate_metrics(y_train, train_pred),
                'val': self.calculate_metrics(y_val, val_pred)
            }

            logger.info(f"LSTM - Train RMSE: {self.metrics['lstm']['train']['rmse']:.4f}")
            logger.info(f"LSTM - Val RMSE: {self.metrics['lstm']['val']['rmse']:.4f}")

        return model

    def train_ensemble(self, X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray = None, y_val: np.ndarray = None,
                      models: list = None) -> Dict:
        """
        Train ensemble of models

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            models: List of model names to include in ensemble

        Returns:
            Dictionary of trained models
        """
        logger.info("Training ensemble models")

        if models is None:
            models = ['linear_regression', 'random_forest', 'gradient_boosting']

        ensemble_models = {}

        # Train individual models
        if 'linear_regression' in models:
            ensemble_models['linear_regression'] = self.train_linear_regression(
                X_train, y_train, X_val, y_val
            )

        if 'random_forest' in models:
            ensemble_models['random_forest'] = self.train_random_forest(
                X_train, y_train, X_val, y_val
            )

        if 'gradient_boosting' in models:
            ensemble_models['gradient_boosting'] = self.train_gradient_boosting(
                X_train, y_train, X_val, y_val
            )

        # Create ensemble predictions
        if X_val is not None and y_val is not None:
            ensemble_preds = []
            weights = []

            for model_name, model in ensemble_models.items():
                if model_name in self.models:
                    pred = self.predict(model_name, X_val)
                    ensemble_preds.append(pred)

                    # Use inverse RMSE as weight
                    rmse = self.metrics[model_name]['val']['rmse']
                    weights.append(1 / rmse if rmse > 0 else 0)

            # Normalize weights
            weights = np.array(weights)
            weights = weights / weights.sum()

            # Weighted average ensemble
            ensemble_pred = np.average(ensemble_preds, axis=0, weights=weights)

            self.metrics['ensemble'] = {
                'val': self.calculate_metrics(y_val, ensemble_pred),
                'weights': dict(zip(ensemble_models.keys(), weights))
            }

            logger.info(f"Ensemble - Val RMSE: {self.metrics['ensemble']['val']['rmse']:.4f}")
            logger.info(f"Ensemble weights: {self.metrics['ensemble']['weights']}")

        self.models['ensemble'] = ensemble_models
        return ensemble_models

    def predict(self, model_name: str, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using a trained model

        Args:
            model_name: Name of the model
            X: Features for prediction

        Returns:
            Predictions
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")

        model = self.models[model_name]

        if model_name == 'ensemble':
            # Ensemble prediction
            predictions = []
            weights = self.metrics['ensemble'].get('weights', {})

            for sub_model_name, sub_model in model.items():
                pred = sub_model.predict(X)
                weight = weights.get(sub_model_name, 1.0 / len(model))
                predictions.append(pred * weight)

            return np.sum(predictions, axis=0)

        elif model_name == 'lstm' and TF_AVAILABLE:
            return model.predict(X).flatten()

        else:
            return model.predict(X)

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Calculate various metrics for model evaluation

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # Regression metrics
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['r2'] = r2_score(y_true, y_pred)

        # MAPE (Mean Absolute Percentage Error)
        mask = y_true != 0
        if np.any(mask):
            metrics['mape'] = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            metrics['mape'] = np.inf

        # Directional Accuracy
        if len(y_true) > 1:
            true_direction = np.diff(y_true) > 0
            pred_direction = np.diff(y_pred) > 0
            metrics['directional_accuracy'] = np.mean(true_direction == pred_direction)
        else:
            metrics['directional_accuracy'] = 0

        return metrics

    def cross_validate(self, X: np.ndarray, y: np.ndarray,
                      model_type: str = 'random_forest',
                      n_splits: int = 5) -> Dict:
        """
        Perform time series cross-validation

        Args:
            X: Features
            y: Targets
            model_type: Type of model to validate
            n_splits: Number of CV splits

        Returns:
            Cross-validation results
        """
        logger.info(f"Performing {n_splits}-fold cross-validation for {model_type}")

        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_metrics = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Train model based on type
            if model_type == 'linear_regression':
                self.train_linear_regression(X_train, y_train, X_val, y_val)
            elif model_type == 'random_forest':
                self.train_random_forest(X_train, y_train, X_val, y_val)
            elif model_type == 'gradient_boosting':
                self.train_gradient_boosting(X_train, y_train, X_val, y_val)

            # Get metrics
            fold_metrics = self.metrics[model_type]['val']
            cv_metrics.append(fold_metrics)

            logger.info(f"Fold {fold + 1} - RMSE: {fold_metrics['rmse']:.4f}")

        # Aggregate metrics
        aggregated = {}
        for metric_name in cv_metrics[0].keys():
            values = [m[metric_name] for m in cv_metrics]
            aggregated[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'values': values
            }

        logger.info(f"CV Average RMSE: {aggregated['rmse']['mean']:.4f} (+/-{aggregated['rmse']['std']:.4f})")

        return aggregated

    def hyperparameter_tuning(self, X_train: np.ndarray, y_train: np.ndarray,
                            model_type: str = 'random_forest',
                            param_grid: Dict = None, cv: int = 3) -> Dict:
        """
        Perform hyperparameter tuning

        Args:
            X_train: Training features
            y_train: Training targets
            model_type: Type of model
            param_grid: Parameter grid for search
            cv: Number of CV folds

        Returns:
            Best parameters and model
        """
        logger.info(f"Performing hyperparameter tuning for {model_type}")

        # Default parameter grids
        if param_grid is None:
            if model_type == 'random_forest':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [10, 15, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            elif model_type == 'gradient_boosting':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0]
                }
            else:
                raise ValueError(f"No default param_grid for {model_type}")

        # Select base model
        if model_type == 'random_forest':
            base_model = RandomForestRegressor(random_state=42)
        elif model_type == 'gradient_boosting':
            base_model = GradientBoostingRegressor(random_state=42)
        else:
            raise ValueError(f"Hyperparameter tuning not supported for {model_type}")

        # Grid search with time series CV
        tscv = TimeSeriesSplit(n_splits=cv)
        grid_search = GridSearchCV(
            base_model, param_grid, cv=tscv,
            scoring='neg_mean_squared_error',
            n_jobs=-1, verbose=1
        )

        # Fit grid search
        grid_search.fit(X_train, y_train)

        # Store best model
        self.models[f'{model_type}_tuned'] = grid_search.best_estimator_

        results = {
            'best_params': grid_search.best_params_,
            'best_score': -grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }

        logger.info(f"Best parameters: {results['best_params']}")
        logger.info(f"Best CV RMSE: {np.sqrt(results['best_score']):.4f}")

        return results

    def save_model(self, model_name: str, filepath: str = None):
        """
        Save a trained model

        Args:
            model_name: Name of the model to save
            filepath: Path to save the model
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")

        if filepath is None:
            filepath = os.path.join(self.model_dir, f'{model_name}.pkl')

        if model_name == 'lstm' and TF_AVAILABLE:
            # Ensure Keras models use a supported extension
            if not filepath.endswith(('.keras', '.h5')):
                filepath = f"{filepath}.keras"
            self.models[model_name].save(filepath)
        else:
            # Ensure sklearn models are saved as .pkl for consistency
            if not filepath.endswith('.pkl'):
                filepath = f"{filepath}.pkl"
            joblib.dump(self.models[model_name], filepath)

        logger.info(f"Model {model_name} saved to {filepath}")

    def load_model(self, model_name: str, filepath: str = None):
        """
        Load a saved model

        Args:
            model_name: Name to assign to loaded model
            filepath: Path to load the model from
        """
        if filepath is None:
            filepath = os.path.join(self.model_dir, f'{model_name}.pkl')

        if model_name == 'lstm' and TF_AVAILABLE:
            # Adjust extension for Keras models if needed
            if filepath.endswith('.pkl'):
                filepath = filepath[:-4] + '.keras'
            elif not filepath.endswith(('.keras', '.h5')):
                filepath = f"{filepath}.keras"
            self.models[model_name] = load_model(filepath)
        else:
            # Ensure sklearn model paths end with .pkl
            if not filepath.endswith('.pkl'):
                filepath = f"{filepath}.pkl"
            self.models[model_name] = joblib.load(filepath)

        logger.info(f"Model {model_name} loaded from {filepath}")

    def get_feature_importance(self, model_name: str = 'random_forest') -> pd.DataFrame:
        """
        Get feature importance from tree-based models

        Args:
            model_name: Name of the model

        Returns:
            DataFrame with feature importance
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")

        model = self.models[model_name]

        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            return pd.DataFrame({
                'feature': range(len(importance)),
                'importance': importance
            }).sort_values('importance', ascending=False)
        else:
            logger.warning(f"Model {model_name} does not have feature_importances_")
            return pd.DataFrame()


# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20

    X_train = np.random.randn(n_samples, n_features)
    y_train = np.sum(X_train[:, :5], axis=1) + np.random.randn(n_samples) * 0.1

    X_test = np.random.randn(200, n_features)
    y_test = np.sum(X_test[:, :5], axis=1) + np.random.randn(200) * 0.1

    # Initialize trainer
    config = {
        'random_forest': {
            'n_estimators': 100,
            'max_depth': 10
        },
        'lstm': {
            'units': [64, 32],
            'epochs': 10,
            'batch_size': 32
        }
    }

    trainer = StockPredictionModels(config)

    # Train models
    trainer.train_linear_regression(X_train, y_train, X_test, y_test)
    trainer.train_random_forest(X_train, y_train, X_test, y_test)
    trainer.train_gradient_boosting(X_train, y_train, X_test, y_test)

    # Train ensemble
    trainer.train_ensemble(X_train, y_train, X_test, y_test)

    # Display metrics
    print("\nModel Performance:")
    for model_name, metrics in trainer.metrics.items():
        if 'val' in metrics:
            print(f"{model_name}: RMSE={metrics['val']['rmse']:.4f}, R�={metrics['val']['r2']:.4f}")

    # Cross-validation
    cv_results = trainer.cross_validate(X_train, y_train, 'random_forest', n_splits=3)
    print(f"\nCross-validation RMSE: {cv_results['rmse']['mean']:.4f} (�{cv_results['rmse']['std']:.4f})")

    # Save models
    trainer.save_model('random_forest')
    print("\nModel saved successfully")
