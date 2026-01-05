"""Model Orchestrator managing multiple time series models."""

import logging
from datetime import datetime

import numpy as np
import pandas as pd

from ml_models.classical import ARIMAModel, ETSModel
from ml_models.deep_learning import GRUModel, LSTMModel, TCNModel
from ml_models.hybrid import HybridLSTMARIMAModel
from ml_models.ml_enhanced import CatBoostModel, LightGBMModel, SVRModel
from ml_models.time_series_base import ForecastResult

logger = logging.getLogger(__name__)


class ModelOrchestrator:
    """Orchestrator managing multiple time series forecasting models.

    Manages training, prediction, and evaluation of 9 different models:
    - Classical: ARIMA, ETS
    - ML Enhanced: LightGBM, CatBoost, SVR
    - Deep Learning: LSTM, GRU, TCN
    - Hybrid: LSTM-ARIMA

    Features:
    - Auto-selection of best model based on validation metrics
    - Model caching for fast reuse
    - Performance tracking and comparison
    - Ensemble forecasting with dynamic weighting
    """

    def __init__(self, cache_models: bool = True, auto_select_top_k: int = 3):
        """Initialize orchestrator.

        Args:
            cache_models: Whether to cache trained models.
            auto_select_top_k: Number of top models to use in ensemble.
        """
        self.cache_models = cache_models
        self.auto_select_top_k = auto_select_top_k

        # Dictionary to store model instances
        self.models: dict[str, any] = {}
        self._init_models()

        # Cache trained models
        self._trained_models: dict[str, any] = {}
        self._model_metrics: dict[str, dict[str, float]] = {}
        self._ensemble_weights: dict[str, float] = {}

        self.logger = logging.getLogger(__name__)

    def _init_models(self) -> None:
        """Initialize all model instances with default parameters."""
        # Classical models
        self.models["arima"] = ARIMAModel()
        self.models["ets"] = ETSModel()

        # ML Enhanced models
        self.models["lightgbm"] = LightGBMModel()
        self.models["catboost"] = CatBoostModel()
        self.models["svr"] = SVRModel()

        # Deep Learning models
        self.models["lstm"] = LSTMModel(epochs=20, batch_size=16)
        self.models["gru"] = GRUModel(epochs=20, batch_size=16)
        self.models["tcn"] = TCNModel(epochs=20, batch_size=16)

        # Hybrid model
        self.models["hybrid_lstm_arima"] = HybridLSTMARIMAModel(lstm_epochs=20)

        logger.info(f"Initialized {len(self.models)} models")

    def fit_all(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Fit all models.

        Args:
            X: Training features.
            y: Training targets.

        Returns:
            Dictionary with training results for each model.
        """
        results = {}

        for model_name, model in self.models.items():
            try:
                logger.info(f"Training {model_name}...")
                model.fit(X, y)

                if self.cache_models:
                    self._trained_models[model_name] = model

                results[model_name] = {
                    "status": "success",
                    "metrics": (
                        model._diagnostics.copy()
                        if hasattr(model, "_diagnostics")
                        else {}
                    ),
                }

                logger.info(f"{model_name} training completed successfully")

            except Exception as e:
                logger.error(f"Training {model_name} failed: {e}")
                results[model_name] = {
                    "status": "failed",
                    "error": str(e),
                }

        return results

    def predict_all(self, steps: int = 7) -> dict:
        """Generate predictions from all models.

        Args:
            steps: Number of steps ahead to forecast.

        Returns:
            Dictionary with ForecastResult for each model.
        """
        predictions = {}

        for model_name, model in self.models.items():
            try:
                result = model.predict(steps=steps)
                predictions[model_name] = result

                # Store metrics
                if hasattr(result, "predictions"):
                    self._model_metrics[model_name] = {
                        "mean_pred": float(np.mean(result.predictions)),
                        "std_pred": float(np.std(result.predictions)),
                        "min_pred": float(np.min(result.predictions)),
                        "max_pred": float(np.max(result.predictions)),
                    }

            except Exception as e:
                logger.warning(f"Prediction from {model_name} failed: {e}")
                predictions[model_name] = None

        return predictions

    def get_best_model(self, steps: int = 7) -> tuple:
        """Get best performing model.

        Args:
            steps: Number of steps ahead to forecast.

        Returns:
            Tuple of (model_name, ForecastResult).
        """
        if not self._model_metrics:
            return None, None

        # Simple heuristic: model with lowest variance (most stable)
        model_name = min(
            self._model_metrics.keys(),
            key=lambda x: self._model_metrics[x]["std_pred"],
        )

        return model_name, self.models[model_name].predict(steps=steps)

    def get_ensemble_forecast(
        self,
        steps: int = 7,
        method: str = "weighted_mean",
    ) -> ForecastResult:
        """Generate ensemble forecast from top models.

        Args:
            steps: Number of steps ahead to forecast.
            method: Ensemble method ('weighted_mean', 'median', 'equal_weight').

        Returns:
            ForecastResult with ensemble predictions.
        """
        # Get predictions from all models
        all_predictions = self.predict_all(steps)

        # Filter valid predictions
        valid_models = {
            name: pred for name, pred in all_predictions.items() if pred is not None
        }

        if not valid_models:
            raise ValueError("No valid model predictions")

        # Select top models
        if len(valid_models) > self.auto_select_top_k:
            # Sort by prediction stability (std)
            sorted_models = sorted(
                self._model_metrics.items(),
                key=lambda x: x[1]["std_pred"],
            )
            top_models = {
                name: valid_models[name]
                for name, _ in sorted_models[: self.auto_select_top_k]
            }
        else:
            top_models = valid_models

        # Ensemble predictions
        if method == "weighted_mean":
            # Weights inversely proportional to variance
            variances = {
                name: self._model_metrics[name]["std_pred"]
                for name in top_models.keys()
            }
            total_var = sum(variances.values())
            weights = {
                name: (
                    (total_var - v) / (total_var * (len(variances) - 1))
                    if total_var > 0
                    else 1.0 / len(variances)
                )
                for name, v in variances.items()
            }

        elif method == "equal_weight":
            weights = {name: 1.0 / len(top_models) for name in top_models.keys()}

        elif method == "median":
            # Just use median, weights don't matter
            weights = {name: 1.0 / len(top_models) for name in top_models.keys()}

        else:
            raise ValueError(f"Unknown ensemble method: {method}")

        # Combine predictions
        ensemble_preds = []
        for step_idx in range(steps):
            if method == "median":
                step_values = [
                    top_models[name].predictions[step_idx] for name in top_models.keys()
                ]
                ensemble_preds.append(float(np.median(step_values)))
            else:
                step_value = sum(
                    weights[name] * top_models[name].predictions[step_idx]
                    for name in top_models.keys()
                )
                ensemble_preds.append(float(step_value))

        # Confidence intervals from ensemble
        ensemble_std = np.std(ensemble_preds)
        min_margin = max(np.mean(ensemble_preds) * 0.02, 0.1)
        margin = float(max(1.96 * ensemble_std, min_margin))

        # Get dates from first valid model
        first_result = next(iter(top_models.values()))

        return ForecastResult(
            dates=first_result.dates,
            predictions=ensemble_preds,
            lower_bound=[float(max(0, p - margin)) for p in ensemble_preds],
            upper_bound=[float(p + margin) for p in ensemble_preds],
            confidence_level=0.95,
            model_name="Ensemble",
            metadata={
                "num_models": len(top_models),
                "models_used": list(top_models.keys()),
                "ensemble_method": method,
                "weights": weights,
            },
        )

    def get_model_comparison(self) -> pd.DataFrame:
        """Get comparison of all models.

        Returns:
            DataFrame with model metrics comparison.
        """
        comparison_data = {}

        for model_name, metrics in self._model_metrics.items():
            comparison_data[model_name] = metrics

        return pd.DataFrame(comparison_data).T

    def get_orchestrator_stats(self) -> dict:
        """Get orchestrator statistics.

        Returns:
            Dictionary with statistics.
        """
        valid_models = sum(
            1 for metrics in self._model_metrics.values() if metrics is not None
        )

        return {
            "total_models": len(self.models),
            "successfully_trained": len(self._trained_models),
            "models_with_predictions": valid_models,
            "timestamp": datetime.now().isoformat(),
        }
