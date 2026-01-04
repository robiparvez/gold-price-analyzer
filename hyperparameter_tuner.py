"""
Hyperparameter Tuner for Gold Price Prediction Models
Uses Optuna for automated hyperparameter optimization with XGBoost integration
"""

import logging

import numpy as np
import optuna
import pandas as pd
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

# Configure logging
logger = logging.getLogger(__name__)


class HyperparameterTuner:
    """
    Hyperparameter tuner for XGBoost and other models using Optuna.
    Optimizes model performance on gold price prediction tasks.
    """

    def __init__(self, n_trials: int = 50, timeout: int = 3600):
        """
        Initialize the hyperparameter tuner.

        Args:
            n_trials: Number of optimization trials
            timeout: Maximum optimization time in seconds
        """
        self.n_trials = n_trials
        self.timeout = timeout
        self.best_params = {}
        self.best_score = None
        self.study = None

    def prepare_training_data(
        self,
        df: pd.DataFrame,
        target_col: str = "price_bdt_per_gram",
        test_size: float = 0.2,
    ) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Prepare training and testing data.

        Args:
            df: DataFrame with historical data
            target_col: Target column name
            test_size: Test set size ratio

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        if df.empty:
            logger.error("Empty DataFrame provided")
            return None, None, None, None  # type: ignore[return-value]

        # Select features
        feature_cols = [
            col
            for col in df.columns
            if col not in [target_col, "date", "source", "metal"]
        ]

        X = df[feature_cols].fillna(0)
        y = df[target_col].values

        # Scale features
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42
        )

        logger.info(
            f"Prepared training data: X_train={X_train.shape}, X_test={X_test.shape}"
        )

        return X_train, X_test, y_train, y_test

    def objective_xgboost(
        self,
        trial: optuna.Trial,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> float:
        """
        Objective function for XGBoost hyperparameter optimization.

        Args:
            trial: Optuna trial object
            X_train: Training features
            y_train: Training target
            X_test: Testing features
            y_test: Testing target

        Returns:
            Mean Absolute Percentage Error (negative for minimization)
        """
        try:
            # Suggest hyperparameters
            params = {
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3),
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 7),
                "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            }

            # Train model
            model = XGBRegressor(**params, random_state=42, n_jobs=-1, verbosity=0)
            model.fit(X_train, y_train)

            # Evaluate
            y_pred = model.predict(X_test)

            # Calculate Mean Absolute Percentage Error (MAPE)
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

            logger.info(f"Trial {trial.number}: MAPE = {mape:.2f}%")

            return mape

        except Exception as e:
            logger.error(f"Error in trial {trial.number}: {e}")
            return float("inf")

    def optimize_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> dict:
        """
        Optimize XGBoost hyperparameters using Optuna.

        Args:
            X_train: Training features
            y_train: Training target
            X_test: Testing features
            y_test: Testing target

        Returns:
            Dictionary with best parameters and score
        """
        logger.info(
            f"Starting XGBoost hyperparameter optimization ({self.n_trials} trials)..."
        )

        # Create study with pruning
        sampler = TPESampler(seed=42)
        pruner = MedianPruner()

        self.study = optuna.create_study(
            direction="minimize", sampler=sampler, pruner=pruner
        )

        # Optimize
        def objective(trial):
            return self.objective_xgboost(trial, X_train, y_train, X_test, y_test)

        try:
            self.study.optimize(
                objective,
                n_trials=self.n_trials,
                timeout=self.timeout,
                show_progress_bar=True,
            )

            # Get best results
            self.best_params = self.study.best_params
            self.best_score = self.study.best_value

            logger.info("Optimization complete!")
            logger.info(f"Best MAPE: {self.best_score:.2f}%")
            logger.info(f"Best parameters: {self.best_params}")

            return {
                "best_params": self.best_params,
                "best_score": self.best_score,
                "n_trials": len(self.study.trials),
            }

        except Exception as e:
            logger.error(f"Error during optimization: {e}")
            return None  # type: ignore[return-value]

    def train_optimized_model(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> XGBRegressor | None:
        """
        Train XGBoost model with optimized hyperparameters.

        Args:
            X_train: Training features
            y_train: Training target

        Returns:
            Trained XGBoost model
        """
        if not self.best_params:
            logger.error("No optimized parameters available. Run optimize() first.")
            return None

        try:
            logger.info("Training XGBoost with optimized parameters...")

            model = XGBRegressor(**self.best_params, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)

            logger.info("Model training complete")
            return model

        except Exception as e:
            logger.error(f"Error training model: {e}")
            return None

    def evaluate_model(
        self, model: XGBRegressor, X_test: np.ndarray, y_test: np.ndarray
    ) -> dict:
        """
        Evaluate model performance.

        Args:
            model: Trained XGBoost model
            X_test: Testing features
            y_test: Testing target

        Returns:
            Dictionary with evaluation metrics
        """
        try:
            y_pred = model.predict(X_test)

            # Calculate metrics
            mae = np.mean(np.abs(y_test - y_pred))
            rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            r2 = 1 - (
                np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
            )

            metrics = {"mae": mae, "rmse": rmse, "mape": mape, "r2_score": r2}

            logger.info(f"Model Evaluation Metrics: {metrics}")
            return metrics

        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return None  # type: ignore[return-value]

    def get_feature_importance(
        self, model: XGBRegressor, feature_names: list
    ) -> dict[str, float]:
        """
        Get feature importance from trained model.

        Args:
            model: Trained XGBoost model
            feature_names: List of feature names

        Returns:
            Dictionary with feature importance scores
        """
        try:
            importance_dict = dict(zip(feature_names, model.feature_importances_))

            # Sort by importance
            sorted_importance = dict(
                sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            )

            logger.info(f"Top features: {list(sorted_importance.items())[:5]}")
            return sorted_importance

        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return {}

    def full_optimization_pipeline(self, df: pd.DataFrame) -> dict:
        """
        Run complete optimization pipeline.

        Args:
            df: DataFrame with historical data

        Returns:
            Dictionary with optimization results
        """
        logger.info("Starting full optimization pipeline...")

        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_training_data(df)

        if X_train is None:
            logger.error("Failed to prepare training data")
            return None  # type: ignore[return-value]

        # Optimize hyperparameters
        optimization_result = self.optimize_xgboost(X_train, y_train, X_test, y_test)  # type: ignore[arg-type]

        if not optimization_result:
            logger.error("Optimization failed")
            return None  # type: ignore[return-value]

        # Train optimized model
        model = self.train_optimized_model(X_train, y_train)  # type: ignore[arg-type]

        if model is None:
            logger.error("Model training failed")
            return None  # type: ignore[return-value]

        # Evaluate
        metrics = self.evaluate_model(model, X_test, y_test)  # type: ignore[arg-type]

        # Get feature importance
        feature_cols = [
            col
            for col in df.columns
            if col not in ["price_bdt_per_gram", "date", "source", "metal"]
        ]
        importance = self.get_feature_importance(model, feature_cols)

        result = {
            "optimization": optimization_result,
            "model": model,
            "metrics": metrics,
            "feature_importance": importance,
        }

        logger.info("Full optimization pipeline complete")
        return result
