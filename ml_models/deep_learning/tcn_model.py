"""TCN (Temporal Convolutional Network) forecasting model.

This module provides TCN implementation for time-series forecasting.
"""

import logging
from datetime import datetime

import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers

from components.business_day_utils import generate_business_day_dates
from ml_models.time_series_base import (
    BaseTimeSeriesModel,
    ForecastResult,
    ModelMetadata,
)

logger = logging.getLogger(__name__)


class TCNModel(BaseTimeSeriesModel):
    """Temporal Convolution Network with causal dilated convolutions.

    TCN uses dilated causal convolutions to capture long-range dependencies
    efficiently without the vanishing gradient problem of RNNs.

    Attributes:
        lookback: Number of past time steps to use.
        forecast_horizon: Number of steps ahead to predict.
        filters: Number of filters per convolutional layer.
        kernel_size: Size of convolutional kernel.
        num_layers: Number of convolutional layers (1-2).
        dilation_base: Base for dilation rates (exponential growth).
        learning_rate: Adam optimizer learning rate.
        epochs: Number of training epochs.
        batch_size: Batch size for training.
        validation_split: Fraction of data to use for validation.
        early_stopping_patience: Patience for early stopping.
        dropout_rate: Dropout rate for regularization.
    """

    def __init__(
        self,
        lookback: int = 14,
        forecast_horizon: int = 7,
        filters: int = 32,
        kernel_size: int = 3,
        num_layers: int = 1,
        dilation_base: int = 2,
        learning_rate: float = 0.001,
        epochs: int = 50,
        batch_size: int = 16,
        validation_split: float = 0.2,
        early_stopping_patience: int = 10,
        dropout_rate: float = 0.2,
    ):
        """Initialize TCN model.

        Args:
            lookback: Past time steps to use as input.
            forecast_horizon: Steps ahead to predict.
            filters: Number of filters in convolutional layers.
            kernel_size: Kernel size for convolutions.
            num_layers: Number of convolutional layers (1-2).
            dilation_base: Base for exponential dilation growth.
            learning_rate: Learning rate for optimizer.
            epochs: Maximum training epochs.
            batch_size: Batch size for training.
            validation_split: Validation split fraction.
            early_stopping_patience: Patience for early stopping.
            dropout_rate: Dropout rate.
        """
        super().__init__()

        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        self.filters = filters
        self.kernel_size = kernel_size
        self.num_layers = min(max(num_layers, 1), 2)
        self.dilation_base = dilation_base
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.early_stopping_patience = early_stopping_patience
        self.dropout_rate = dropout_rate

        self.hyperparameters = {
            "lookback": lookback,
            "forecast_horizon": forecast_horizon,
            "filters": filters,
            "kernel_size": kernel_size,
            "num_layers": self.num_layers,
            "dilation_base": dilation_base,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size,
            "dropout_rate": dropout_rate,
        }

        self.version = "1.0.0"
        self.model_name = "TCN"

        self._model = None
        self._fitted_model = None
        self._training_data = None
        self._feature_mean = None
        self._feature_std = None
        self._target_mean = None
        self._target_std = None
        self._diagnostics: dict[str, any] = {}

    def _calculate_receptive_field(self) -> int:
        """Calculate receptive field size.

        Returns:
            Receptive field size in time steps.
        """
        receptive_field = 1
        for layer in range(self.num_layers):
            dilation = self.dilation_base**layer
            receptive_field += (self.kernel_size - 1) * dilation
        return receptive_field

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_data: tuple | None = None,
    ) -> None:
        """Fit TCN model.

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

            if len(X) < self.lookback + self.forecast_horizon:
                raise ValueError(
                    f"Not enough data: need at least {self.lookback + self.forecast_horizon} samples"
                )

            # Store training data
            self._training_data = y.copy()

            # Create sliding windows
            from components.sliding_window import augment_data, create_sliding_windows

            X_windows, y_windows = create_sliding_windows(
                y.values,
                lookback=self.lookback,
                forecast_horizon=self.forecast_horizon,
            )

            # Normalize data
            self._feature_mean = np.mean(X_windows)
            self._feature_std = np.std(X_windows)
            if self._feature_std == 0:
                self._feature_std = 1.0

            self._target_mean = np.mean(y_windows)
            self._target_std = np.std(y_windows)
            if self._target_std == 0:
                self._target_std = 1.0

            X_norm = (X_windows - self._feature_mean) / self._feature_std
            y_norm = (y_windows - self._target_mean) / self._target_std

            # Data augmentation for small datasets
            if len(X_norm) < 100:
                X_norm, y_norm = augment_data(
                    X_norm, y_norm, noise_std=0.01, num_augmentations=2
                )

            # Reshape for Conv1D (samples, timesteps, features)
            X_norm = np.expand_dims(X_norm, -1)

            # Build TCN model
            self._model = keras.Sequential()

            # Input layer
            self._model.add(keras.Input(shape=(self.lookback, 1)))

            # Convolutional layers with causal dilation
            for layer in range(self.num_layers):
                dilation = self.dilation_base**layer

                # Causal padding: pad left side only
                self._model.add(
                    layers.Conv1D(
                        filters=self.filters,
                        kernel_size=self.kernel_size,
                        padding="causal",
                        dilation_rate=dilation,
                        activation="relu",
                    )
                )

                if self.dropout_rate > 0:
                    self._model.add(layers.Dropout(self.dropout_rate))

            # Global average pooling to reduce to single value per filter
            self._model.add(layers.GlobalAveragePooling1D())

            # Dense output layer
            self._model.add(layers.Dense(units=self.forecast_horizon))

            # Compile model
            self._model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
                loss="mse",
                metrics=["mae"],
            )

            # Early stopping
            early_stopping = keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.early_stopping_patience,
                restore_best_weights=True,
            )

            # Train model
            history = self._model.fit(
                X_norm,
                y_norm,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=self.validation_split,
                callbacks=[early_stopping],
                verbose=0,
            )

            self._fitted_model = self._model

            # Store diagnostics
            receptive_field = self._calculate_receptive_field()
            self._diagnostics.update(
                {
                    "lookback": self.lookback,
                    "forecast_horizon": self.forecast_horizon,
                    "receptive_field": receptive_field,
                    "training_samples": len(X_norm),
                    "final_train_loss": float(history.history["loss"][-1]),
                    "final_val_loss": float(history.history["val_loss"][-1]),
                    "epochs_trained": len(history.history["loss"]),
                }
            )

            logger.info(
                f"TCN model fitted with {len(X_norm)} samples, "
                f"receptive_field={receptive_field}, "
                f"val_loss={self._diagnostics['final_val_loss']:.4f}"
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
        if self._fitted_model is None:
            raise ValueError("Model must be fitted before prediction")

        try:
            predictions = []
            last_sequence = np.expand_dims(
                (self._training_data.values[-self.lookback :] - self._feature_mean)
                / self._feature_std,
                axis=-1,
            )

            # Generate predictions iteratively
            for _ in range(steps):
                # Predict next values
                pred_norm = self._fitted_model.predict(
                    np.expand_dims(last_sequence, 0), verbose=0
                )[0]

                # Denormalize
                pred = pred_norm * self._target_std + self._target_mean

                # Use first value for single-step update
                next_val = pred[0]
                predictions.append(float(next_val))

                # Update sequence
                next_val_norm = (next_val - self._feature_mean) / self._feature_std
                last_sequence = np.vstack([last_sequence[1:], [[next_val_norm]]])

            # Create forecast dates (business days only)
            last_date = self._training_data.index[-1]
            forecast_dates = generate_business_day_dates(
                start_date=last_date,
                num_days=steps,
                exclude_time=True,
            )

            # Confidence intervals
            recent_std = (
                np.std(predictions[-3:])
                if len(predictions) >= 3
                else np.std(self._training_data.values) * 0.05
            )
            min_margin = max(np.mean(predictions) * 0.02, 0.1)
            margin = max(1.96 * recent_std, min_margin)

            return ForecastResult(
                dates=forecast_dates,
                predictions=predictions,
                lower_bound=[max(0, p - margin) for p in predictions],
                upper_bound=[p + margin for p in predictions],
                confidence_level=0.95,
                model_name="TCN",
                metadata={
                    "lookback": self.lookback,
                    "forecast_horizon": self.forecast_horizon,
                    "filters": self.filters,
                    "receptive_field": self._calculate_receptive_field(),
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
            model_name="TCN",
            model_type="deep_learning",
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
