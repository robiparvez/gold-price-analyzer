"""Hybrid LSTM-ARIMA forecasting model.

Combines LSTM neural network with ARIMA for improved predictions.
"""

import logging
from datetime import datetime

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from components.business_day_utils import generate_business_day_dates
from ml_models.deep_learning import LSTMModel
from ml_models.time_series_base import (
    BaseTimeSeriesModel,
    ForecastResult,
    ModelMetadata,
)

logger = logging.getLogger(__name__)


class HybridLSTMARIMAModel(BaseTimeSeriesModel):
    """Hybrid model that combines LSTM and ARIMA.

    The hybrid approach:
    1. Train LSTM to capture non-linear patterns
    2. Compute residuals from LSTM predictions
    3. Fit ARIMA to residuals to capture remaining patterns
    4. Combine LSTM + ARIMA forecasts for final prediction

    Attributes:
        lstm_lookback: Number of past time steps for LSTM.
        forecast_horizon: Number of steps ahead to predict.
        lstm_units: Units in LSTM layers.
        arima_order: (p, d, q) for ARIMA model.
        lstm_weight: Weight for LSTM forecast in ensemble.
        arima_weight: Weight for ARIMA forecast in ensemble.
    """

    def __init__(
        self,
        lstm_lookback: int = 14,
        forecast_horizon: int = 7,
        lstm_units: int = 32,
        arima_order: tuple = (1, 1, 1),
        lstm_weight: float = 0.6,
        arima_weight: float = 0.4,
        lstm_epochs: int = 30,
        batch_size: int = 16,
        validation_split: float = 0.2,
        early_stopping_patience: int = 8,
    ):
        """Initialize hybrid model.

        Args:
            lstm_lookback: Past time steps for LSTM.
            forecast_horizon: Steps ahead to predict.
            lstm_units: Units in LSTM layers.
            arima_order: (p, d, q) tuple for ARIMA.
            lstm_weight: Weight for LSTM in ensemble (0-1).
            arima_weight: Weight for ARIMA in ensemble (0-1).
            lstm_epochs: Training epochs for LSTM.
            batch_size: Batch size for LSTM.
            validation_split: Validation split for LSTM.
            early_stopping_patience: Patience for early stopping.
        """
        super().__init__()

        self.lstm_lookback = lstm_lookback
        self.forecast_horizon = forecast_horizon
        self.lstm_units = lstm_units
        self.arima_order = arima_order

        # Normalize weights
        total_weight = lstm_weight + arima_weight
        self.lstm_weight = lstm_weight / total_weight
        self.arima_weight = arima_weight / total_weight

        self.hyperparameters = {
            "lstm_lookback": lstm_lookback,
            "forecast_horizon": forecast_horizon,
            "lstm_units": lstm_units,
            "arima_order": arima_order,
            "lstm_weight": self.lstm_weight,
            "arima_weight": self.arima_weight,
            "lstm_epochs": lstm_epochs,
            "batch_size": batch_size,
            "validation_split": validation_split,
        }

        self.version = "1.0.0"
        self.model_name = "Hybrid-LSTM-ARIMA"

        # Initialize submodels
        self.lstm_model = LSTMModel(
            lookback=lstm_lookback,
            forecast_horizon=forecast_horizon,
            units_per_layer=lstm_units,
            num_layers=1,
            epochs=lstm_epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            early_stopping_patience=early_stopping_patience,
            dropout_rate=0.2,
        )

        self._arima_model = None
        self._training_data = None
        self._lstm_residuals = None
        self._diagnostics: dict[str, any] = {}

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_data: tuple | None = None,
    ) -> None:
        """Fit hybrid model.

        Args:
            X: Training features (dates index, price values).
            y: Target values (price series).
            validation_data: Optional pre-split validation data.

        Raises:
            ValueError: If input data is invalid.
        """
        try:
            if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.Series):
                raise ValueError("X must be DataFrame and y must be Series")

            if len(X) < self.lstm_lookback + self.forecast_horizon:
                raise ValueError(
                    f"Not enough data: need at least {self.lstm_lookback + self.forecast_horizon} samples"
                )

            # Store training data
            self._training_data = y.copy()

            # Step 1: Fit LSTM model
            logger.info("Fitting LSTM component...")
            self.lstm_model.fit(X, y, validation_data)

            # Step 2: Get LSTM predictions on training data to compute residuals
            from components.sliding_window import create_sliding_windows

            X_windows, y_windows = create_sliding_windows(
                y.values,
                lookback=self.lstm_lookback,
                forecast_horizon=self.forecast_horizon,
            )

            # Predict with LSTM on training windows
            lstm_preds = []
            for i in range(len(X_windows)):
                # Get LSTM prediction (use first value)
                try:
                    lstm_result = self.lstm_model.predict(steps=1)
                    lstm_preds.append(lstm_result.predictions[0])
                except Exception:
                    # Fallback: use mean of window
                    lstm_preds.append(np.mean(X_windows[i]))

            # Compute residuals
            y_aligned = y.values[
                self.lstm_lookback : self.lstm_lookback + len(lstm_preds)
            ]
            self._lstm_residuals = y_aligned - np.array(lstm_preds)

            # Step 3: Fit ARIMA on residuals
            logger.info("Fitting ARIMA component on LSTM residuals...")
            try:
                self._arima_model = ARIMA(
                    self._lstm_residuals,
                    order=self.arima_order,
                )
                self._arima_model = self._arima_model.fit()
            except Exception as e:
                logger.warning(
                    f"ARIMA fitting failed ({e}), using simple residual mean"
                )
                self._arima_model = None

            # Store diagnostics
            self._diagnostics.update(
                {
                    "lstm_component": self.lstm_model._diagnostics,
                    "lstm_weight": self.lstm_weight,
                    "arima_weight": self.arima_weight,
                    "residuals_mean": float(np.mean(self._lstm_residuals)),
                    "residuals_std": float(np.std(self._lstm_residuals)),
                }
            )

            logger.info(
                f"Hybrid model fitted with LSTM + ARIMA{self.arima_order}, "
                f"weights: LSTM={self.lstm_weight:.2f}, ARIMA={self.arima_weight:.2f}"
            )

        except Exception as e:
            logger.error(f"Fitting failed: {e}")
            raise ValueError(f"Fitting failed: {e}") from e

    def predict(self, steps: int = 7) -> ForecastResult:
        """Generate forecasts.

        Args:
            steps: Number of steps ahead to forecast.

        Returns:
            ForecastResult with predictions and confidence intervals.

        Raises:
            ValueError: If model not fitted.
        """
        if self.lstm_model._fitted_model is None:
            raise ValueError("Model must be fitted before prediction")

        try:
            # Step 1: LSTM forecast
            lstm_result = self.lstm_model.predict(steps=steps)
            lstm_preds = lstm_result.predictions

            # Step 2: ARIMA forecast on residuals
            arima_preds = []
            if self._arima_model is not None:
                try:
                    arima_forecast = self._arima_model.get_forecast(steps=steps)
                    arima_preds = arima_forecast.predicted_mean.values.tolist()
                except Exception:
                    # Fallback: use residual mean
                    arima_preds = [np.mean(self._lstm_residuals)] * steps
            else:
                # No ARIMA model: use residual mean
                arima_preds = [np.mean(self._lstm_residuals)] * steps

            # Step 3: Combine forecasts
            combined_preds = [
                self.lstm_weight * lp + self.arima_weight * ap
                for lp, ap in zip(lstm_preds, arima_preds)
            ]

            # Create forecast dates (business days only)
            last_date = self._training_data.index[-1]
            forecast_dates = generate_business_day_dates(
                start_date=last_date,
                num_days=steps,
                exclude_time=True,
            )

            # Confidence intervals
            recent_std = (
                np.std(combined_preds[-3:])
                if len(combined_preds) >= 3
                else np.std(self._training_data.values) * 0.05
            )
            min_margin = max(np.mean(combined_preds) * 0.02, 0.1)
            margin = max(1.96 * recent_std, min_margin)

            return ForecastResult(
                dates=forecast_dates,
                predictions=combined_preds,
                lower_bound=[max(0, p - margin) for p in combined_preds],
                upper_bound=[p + margin for p in combined_preds],
                confidence_level=0.95,
                model_name="Hybrid-LSTM-ARIMA",
                metadata={
                    "lstm_weight": self.lstm_weight,
                    "arima_weight": self.arima_weight,
                    "arima_order": self.arima_order,
                    "lstm_lookback": self.lstm_lookback,
                    "forecast_horizon": self.forecast_horizon,
                },
            )

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise ValueError(f"Prediction failed: {e}") from e

    def get_metadata(self) -> ModelMetadata:
        """Get model metadata.

        Returns:
            ModelMetadata with model details.
        """
        start_date = (
            str(self._training_data.index[0]) if self._training_data is not None else ""
        )
        end_date = (
            str(self._training_data.index[-1])
            if self._training_data is not None
            else ""
        )

        return ModelMetadata(
            model_name="Hybrid-LSTM-ARIMA",
            model_type="hybrid",
            purity="22K",
            training_samples=(
                len(self._training_data) if self._training_data is not None else 0
            ),
            training_date_range=(start_date, end_date),
            trained_at=datetime.now().isoformat(),
            hyperparameters=self.hyperparameters,
            metrics=self._diagnostics,
            version=self.version,
        )
