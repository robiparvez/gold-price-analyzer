"""Centralized metrics calculation utilities.

This module consolidates all metric calculations (MAE, RMSE, MAPE, R²)
to eliminate code duplication across ML models and services.
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


@dataclass
class ForecastMetrics:
    """Container for forecast evaluation metrics."""

    mae: float  # Mean Absolute Error
    mse: float  # Mean Squared Error
    rmse: float  # Root Mean Squared Error
    mape: float  # Mean Absolute Percentage Error
    r2: float  # R-squared (coefficient of determination)

    def to_dict(self) -> dict[str, float]:
        """Convert metrics to dictionary."""
        return {
            "mae": round(self.mae, 4),
            "mse": round(self.mse, 4),
            "rmse": round(self.rmse, 4),
            "mape": round(self.mape, 4),
            "r2": round(self.r2, 4),
        }

    def __str__(self) -> str:
        """Human-readable string representation."""
        return (
            f"MAE={self.mae:.2f}, RMSE={self.rmse:.2f}, "
            f"MAPE={self.mape:.2f}%, R²={self.r2:.4f}"
        )


class MetricsCalculator:
    """Unified metrics calculator for time series forecasting.

    Provides consistent calculation of evaluation metrics across
    all ML models and services.

    Example:
        >>> calculator = MetricsCalculator()
        >>> metrics = calculator.calculate_all(y_true, y_pred)
        >>> print(metrics.rmse)
    """

    @staticmethod
    def calculate_mae(
        y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series
    ) -> float:
        """Calculate Mean Absolute Error.

        Args:
            y_true: Actual values.
            y_pred: Predicted values.

        Returns:
            MAE value.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        return float(mean_absolute_error(y_true, y_pred))

    @staticmethod
    def calculate_mse(
        y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series
    ) -> float:
        """Calculate Mean Squared Error.

        Args:
            y_true: Actual values.
            y_pred: Predicted values.

        Returns:
            MSE value.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        return float(mean_squared_error(y_true, y_pred))

    @staticmethod
    def calculate_rmse(
        y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series
    ) -> float:
        """Calculate Root Mean Squared Error.

        Args:
            y_true: Actual values.
            y_pred: Predicted values.

        Returns:
            RMSE value.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))

    @staticmethod
    def calculate_mape(
        y_true: np.ndarray | pd.Series,
        y_pred: np.ndarray | pd.Series,
        epsilon: float = 1e-10,
    ) -> float:
        """Calculate Mean Absolute Percentage Error.

        Args:
            y_true: Actual values.
            y_pred: Predicted values.
            epsilon: Small value to avoid division by zero.

        Returns:
            MAPE value as percentage.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        # Avoid division by zero
        denominator = np.where(y_true != 0, y_true, epsilon)
        mape = np.mean(np.abs((y_true - y_pred) / denominator)) * 100
        return float(mape)

    @staticmethod
    def calculate_r2(
        y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series
    ) -> float:
        """Calculate R-squared (coefficient of determination).

        Args:
            y_true: Actual values.
            y_pred: Predicted values.

        Returns:
            R² value.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        return float(r2_score(y_true, y_pred))

    @classmethod
    def calculate_all(
        cls, y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series
    ) -> ForecastMetrics:
        """Calculate all forecast evaluation metrics.

        Args:
            y_true: Actual values.
            y_pred: Predicted values.

        Returns:
            ForecastMetrics dataclass with all metrics.
        """
        return ForecastMetrics(
            mae=cls.calculate_mae(y_true, y_pred),
            mse=cls.calculate_mse(y_true, y_pred),
            rmse=cls.calculate_rmse(y_true, y_pred),
            mape=cls.calculate_mape(y_true, y_pred),
            r2=cls.calculate_r2(y_true, y_pred),
        )

    @classmethod
    def calculate_all_dict(
        cls, y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series
    ) -> dict[str, float]:
        """Calculate all metrics and return as dictionary.

        Args:
            y_true: Actual values.
            y_pred: Predicted values.

        Returns:
            Dictionary with all metrics.
        """
        return cls.calculate_all(y_true, y_pred).to_dict()


# Convenience functions for backward compatibility
def calculate_forecast_metrics(
    y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series
) -> dict[str, float]:
    """Calculate all forecast metrics (convenience function).

    Args:
        y_true: Actual values.
        y_pred: Predicted values.

    Returns:
        Dictionary with mae, mse, rmse, mape, r2.
    """
    return MetricsCalculator.calculate_all_dict(y_true, y_pred)
