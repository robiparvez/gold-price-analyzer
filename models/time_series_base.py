"""Base classes for time-series forecasting models.

This module provides abstract base classes for implementing forecasting models
with standardized interfaces for training, prediction, and metadata management.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ModelMetadata:
    """Metadata for trained models."""

    model_name: str
    model_type: str  # "classical", "ml_enhanced", "deep_learning", "hybrid"
    purity: str
    training_samples: int
    training_date_range: tuple[str, str]  # (start_date, end_date)
    trained_at: str
    model_path: str | None = None
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    feature_names: list[str] = field(default_factory=list)
    data_statistics: dict[str, float] = field(default_factory=dict)
    version: str = "1.0.0"

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "purity": self.purity,
            "training_samples": self.training_samples,
            "training_date_range": self.training_date_range,
            "trained_at": self.trained_at,
            "model_path": self.model_path,
            "hyperparameters": self.hyperparameters,
            "metrics": self.metrics,
            "feature_names": self.feature_names,
            "data_statistics": self.data_statistics,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelMetadata":
        """Create metadata from dictionary."""
        return cls(**data)


@dataclass
class ForecastResult:
    """Result from forecasting model."""

    dates: list[str]
    predictions: list[float]
    lower_bound: list[float]
    upper_bound: list[float]
    confidence_level: float = 0.95
    model_name: str = ""
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "dates": self.dates,
            "predictions": self.predictions,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "confidence_level": self.confidence_level,
            "model_name": self.model_name,
            "metadata": self.metadata or {},
        }


class BaseTimeSeriesModel(ABC):
    """Abstract base class for time-series forecasting models.

    All forecasting models must implement this interface to ensure consistency
    across different model types and facilitate ensemble methods.
    """

    def __init__(self, purity: str = "22K", **kwargs: Any) -> None:
        """Initialize the model.

        Args:
            purity: Gold purity (e.g., "22K", "21K", "18K")
            **kwargs: Additional model-specific parameters
        """
        self.purity = purity
        self.model: Any = None
        self.metadata: ModelMetadata | None = None
        self.is_fitted: bool = False
        self.kwargs = kwargs

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
        validation_data: tuple[pd.DataFrame, pd.Series] | None = None,
    ) -> "BaseTimeSeriesModel":
        """Train the model on historical data.

        Args:
            X: Feature DataFrame with DatetimeIndex
            y: Target variable (price). If None, extract from X
            validation_data: Optional (X_val, y_val) for early stopping

        Returns:
            Self for method chaining

        Raises:
            ValueError: If data is insufficient or invalid
        """
        pass

    @abstractmethod
    def predict(
        self, horizon: int = 7, X_future: pd.DataFrame | None = None
    ) -> ForecastResult:
        """Generate forecast for future time steps.

        Args:
            horizon: Number of days to forecast ahead
            X_future: Optional future features (for models requiring exogenous variables)

        Returns:
            ForecastResult with predictions and confidence intervals

        Raises:
            RuntimeError: If model not fitted
            ValueError: If horizon is invalid
        """
        pass

    @abstractmethod
    def get_metadata(self) -> ModelMetadata:
        """Get model metadata and training information.

        Returns:
            ModelMetadata with training details and metrics
        """
        pass

    def validate_data(self, X: pd.DataFrame, y: pd.Series | None = None, min_samples: int = 60) -> None:
        """Validate input data meets requirements.

        Args:
            X: Input DataFrame
            y: Target series (optional, used for length check if provided)
            min_samples: Minimum required samples

        Raises:
            ValueError: If data is invalid or insufficient
        """
        # Check length using y if provided, else X
        data_len = len(y) if y is not None else len(X)
        if data_len < min_samples:
            raise ValueError(
                f"Insufficient data: {data_len} samples < {min_samples} required"
            )

        # Validate index type
        data_index = y.index if y is not None else X.index
        if not isinstance(data_index, pd.DatetimeIndex):
            raise ValueError("Data must have DatetimeIndex")

        # Check for NaN values
        if y is not None and y.isnull().any():
            raise ValueError("Target data contains NaN values. Please clean data first.")
        if hasattr(X, 'isnull') and X.isnull().any().any():
            raise ValueError("Feature data contains NaN values. Please clean data first.")

    def _create_metadata(
        self,
        model_name: str,
        model_type: str,
        X: pd.DataFrame,
        hyperparameters: dict[str, Any],
        metrics: dict[str, float] | None = None,
        feature_names: list[str] | None = None,
    ) -> ModelMetadata:
        """Create metadata for the model.

        Args:
            model_name: Name of the model
            model_type: Type category
            X: Training data
            hyperparameters: Model hyperparameters
            metrics: Training/validation metrics
            feature_names: List of feature names used

        Returns:
            ModelMetadata instance
        """
        date_range = (
            X.index.min().strftime("%Y-%m-%d"),
            X.index.max().strftime("%Y-%m-%d"),
        )

        data_stats = {
            "mean": float(X["price_bdt_per_gram"].mean())
            if "price_bdt_per_gram" in X.columns
            else 0.0,
            "std": float(X["price_bdt_per_gram"].std())
            if "price_bdt_per_gram" in X.columns
            else 0.0,
            "min": float(X["price_bdt_per_gram"].min())
            if "price_bdt_per_gram" in X.columns
            else 0.0,
            "max": float(X["price_bdt_per_gram"].max())
            if "price_bdt_per_gram" in X.columns
            else 0.0,
        }

        return ModelMetadata(
            model_name=model_name,
            model_type=model_type,
            purity=self.purity,
            training_samples=len(X),
            training_date_range=date_range,
            trained_at=datetime.now().isoformat(),
            hyperparameters=hyperparameters,
            metrics=metrics or {},
            feature_names=feature_names or [],
            data_statistics=data_stats,
        )

    def calculate_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> dict[str, float]:
        """Calculate evaluation metrics.

        Args:
            y_true: Actual values
            y_pred: Predicted values

        Returns:
            Dictionary with MAE, RMSE, MAPE, R2
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

        # MAPE with safe division
        mape = float(
            np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1e-10)))
            * 100
        )

        r2 = float(r2_score(y_true, y_pred))

        return {"mae": mae, "rmse": rmse, "mape": mape, "r2": r2}


class EnsembleModel(BaseTimeSeriesModel):
    """Ensemble of multiple forecasting models with weighted averaging."""

    def __init__(
        self, models: list[BaseTimeSeriesModel], weights: list[float] | None = None
    ) -> None:
        """Initialize ensemble.

        Args:
            models: List of fitted models
            weights: Optional weights for each model (auto-normalize). If None, use equal weights.
        """
        if not models:
            raise ValueError("At least one model required for ensemble")

        # Use first model's purity
        super().__init__(purity=models[0].purity)

        self.models = models
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            # Normalize weights
            total = sum(weights)
            self.weights = [w / total for w in weights]

        self.is_fitted = all(m.is_fitted for m in models)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
        validation_data: tuple[pd.DataFrame, pd.Series] | None = None,
    ) -> "EnsembleModel":
        """Fit all models in ensemble.

        Args:
            X: Training features
            y: Target variable
            validation_data: Optional validation data

        Returns:
            Self for method chaining
        """
        for model in self.models:
            if not model.is_fitted:
                model.fit(X, y, validation_data)

        self.is_fitted = True
        return self

    def predict(
        self, horizon: int = 7, X_future: pd.DataFrame | None = None
    ) -> ForecastResult:
        """Generate weighted ensemble forecast.

        Args:
            horizon: Forecast horizon
            X_future: Optional future features

        Returns:
            ForecastResult with ensemble predictions
        """
        if not self.is_fitted:
            raise RuntimeError("Ensemble not fitted. Call fit() first.")

        # Get predictions from all models
        forecasts = [model.predict(horizon, X_future) for model in self.models]

        # Weighted average of predictions
        predictions = np.average(
            [f.predictions for f in forecasts], axis=0, weights=self.weights
        )

        # Conservative confidence intervals: use min/max of all bounds
        lower_bounds = np.array([f.lower_bound for f in forecasts])
        upper_bounds = np.array([f.upper_bound for f in forecasts])

        lower_bound = np.min(lower_bounds, axis=0)
        upper_bound = np.max(upper_bounds, axis=0)

        # Use dates from first forecast
        dates = forecasts[0].dates

        return ForecastResult(
            dates=dates,
            predictions=predictions.tolist(),
            lower_bound=lower_bound.tolist(),
            upper_bound=upper_bound.tolist(),
            confidence_level=0.90,  # Conservative for ensemble
            model_name=f"Ensemble({len(self.models)} models)",
            metadata={
                "component_models": [m.__class__.__name__ for m in self.models],
                "weights": self.weights,
            },
        )

    def get_metadata(self) -> ModelMetadata:
        """Get ensemble metadata.

        Returns:
            Metadata for the ensemble
        """
        component_metadata = [m.get_metadata() for m in self.models]

        return ModelMetadata(
            model_name=f"Ensemble_{len(self.models)}_models",
            model_type="ensemble",
            purity=self.purity,
            training_samples=max(m.training_samples for m in component_metadata),
            training_date_range=component_metadata[0].training_date_range,
            trained_at=datetime.now().isoformat(),
            hyperparameters={
                "weights": self.weights,
                "component_models": [m.model_name for m in component_metadata],
            },
            metrics={},
        )
