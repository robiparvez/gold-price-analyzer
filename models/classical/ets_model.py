"""Exponential Smoothing (ETS) forecasting model.

This module provides ETS implementation optimized for limited historical data.
Supports Error-Trend-Seasonal decomposition with automatic component selection.
"""

import logging
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from models.time_series_base import BaseTimeSeriesModel, ForecastResult, ModelMetadata

logger = logging.getLogger(__name__)


class ETSModel(BaseTimeSeriesModel):
    """Exponential Smoothing (ETS) model with automatic component selection.

    Features:
    - Error-Trend-Seasonal decomposition
    - Automatic component selection (add/mul/none)
    - Holt-Winters exponential smoothing
    - Damped trend support
    - Conservative defaults for small datasets

    Attributes:
        trend: Trend component ('add', 'mul', None, 'auto')
        seasonal: Seasonal component ('add', 'mul', None, 'auto')
        seasonal_periods: Number of periods in season (default 7 for daily)
        damped_trend: Whether to use damped trend
        use_auto: Whether to automatically select components
        smoothing_level: Alpha parameter (0-1), None for optimization
        smoothing_trend: Beta parameter (0-1), None for optimization
        smoothing_seasonal: Gamma parameter (0-1), None for optimization
    """

    def __init__(
        self,
        trend: str | None = "auto",
        seasonal: str | None = "auto",
        seasonal_periods: int = 7,
        damped_trend: bool = False,
        use_auto: bool = True,
        smoothing_level: float | None = None,
        smoothing_trend: float | None = None,
        smoothing_seasonal: float | None = None,
        **kwargs,
    ):
        """Initialize ETS model.

        Args:
            trend: Trend component ('add', 'mul', None, 'auto').
            seasonal: Seasonal component ('add', 'mul', None, 'auto').
            seasonal_periods: Number of periods in season.
            damped_trend: Whether to use damped trend.
            use_auto: Whether to automatically select components.
            smoothing_level: Alpha parameter (None for optimization).
            smoothing_trend: Beta parameter (None for optimization).
            smoothing_seasonal: Gamma parameter (None for optimization).
            **kwargs: Additional arguments passed to BaseTimeSeriesModel.
        """
        super().__init__(purity="22K", **kwargs)

        # Store model metadata
        self.model_type = "ets"
        self.version = "1.0.0"
        self.hyperparameters = {
            "trend": trend,
            "seasonal": seasonal,
            "seasonal_periods": seasonal_periods,
            "damped_trend": damped_trend,
            "use_auto": use_auto,
            "smoothing_level": smoothing_level,
            "smoothing_trend": smoothing_trend,
            "smoothing_seasonal": smoothing_seasonal,
        }

        # ETS-specific parameters
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.damped_trend = damped_trend
        self.use_auto = use_auto
        self.smoothing_level = smoothing_level
        self.smoothing_trend = smoothing_trend
        self.smoothing_seasonal = smoothing_seasonal

        self._model = None
        self._fitted_model = None
        self._training_data = None
        self._diagnostics: dict[str, any] = {}

    def _detect_seasonality(self, data: pd.Series) -> dict[str, any]:
        """Detect seasonal patterns in data.

        Args:
            data: Time series data.

        Returns:
            Dictionary with seasonality detection results.
        """
        # Check if we have enough data for seasonal detection
        min_samples = self.seasonal_periods * 2
        if len(data) < min_samples:
            return {
                "has_seasonality": False,
                "reason": f"Insufficient data ({len(data)} < {min_samples})",
            }

        # Calculate coefficient of variation for seasonal windows
        try:
            # Reshape into seasonal periods
            n_periods = len(data) // self.seasonal_periods
            trimmed = data[: n_periods * self.seasonal_periods]
            reshaped = trimmed.values.reshape(n_periods, self.seasonal_periods)

            # Calculate std dev across periods for each position
            seasonal_std = np.std(reshaped, axis=0)
            seasonal_mean = np.mean(reshaped, axis=0)

            # Coefficient of variation
            cv = np.mean(seasonal_std / (seasonal_mean + 1e-8))

            # If variation across seasonal positions is significant, likely seasonal
            has_seasonality = cv > 0.1

            return {
                "has_seasonality": has_seasonality,
                "coefficient_of_variation": cv,
                "seasonal_periods_tested": self.seasonal_periods,
            }

        except Exception as e:
            logger.debug(f"Seasonality detection failed: {e}")
            return {
                "has_seasonality": False,
                "reason": f"Detection error: {e}",
            }

    def _detect_trend(self, data: pd.Series) -> dict[str, any]:
        """Detect trend in data using simple linear regression.

        Args:
            data: Time series data.

        Returns:
            Dictionary with trend detection results.
        """
        # Fit linear trend
        x = np.arange(len(data))
        y = data.values

        # Linear regression
        coeffs = np.polyfit(x, y, 1)
        slope = coeffs[0]

        # R-squared for trend strength
        y_pred = np.polyval(coeffs, x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / (ss_tot + 1e-8))

        # Significant trend if slope is non-zero and R² > 0.3
        has_trend = abs(slope) > 1e-6 and r_squared > 0.3

        return {
            "has_trend": has_trend,
            "slope": slope,
            "r_squared": r_squared,
            "trend_strength": (
                "strong"
                if r_squared > 0.7
                else "moderate" if r_squared > 0.3 else "weak"
            ),
        }

    def _auto_select_components(self, data: pd.Series) -> tuple[str | None, str | None]:
        """Automatically select trend and seasonal components.

        Args:
            data: Time series data.

        Returns:
            Tuple of (trend, seasonal) component types.
        """
        # Detect trend
        trend_info = self._detect_trend(data)
        self._diagnostics["trend_detection"] = trend_info

        # Detect seasonality
        seasonal_info = self._detect_seasonality(data)
        self._diagnostics["seasonal_detection"] = seasonal_info

        # Select components
        trend = "add" if trend_info["has_trend"] else None
        seasonal = "add" if seasonal_info["has_seasonality"] else None

        logger.info(
            f"Auto-selected components: trend={trend}, seasonal={seasonal} "
            f"(trend R²={trend_info['r_squared']:.3f}, "
            f"seasonal CV={seasonal_info.get('coefficient_of_variation', 0):.3f})"
        )

        return trend, seasonal

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ETSModel":
        """Fit ETS model to training data.

        Args:
            X: Feature dataframe with datetime index.
            y: Target variable (prices).

        Returns:
            Self (fitted model).

        Raises:
            ValueError: If data is invalid or model fitting fails.
        """
        # Validate data
        self.validate_data(X, y)

        # Store training data
        self._training_data = y.copy()

        # Auto-select components if needed
        if self.use_auto:
            if self.trend == "auto" or self.seasonal == "auto":
                auto_trend, auto_seasonal = self._auto_select_components(y)

                if self.trend == "auto":
                    self.trend = auto_trend
                    self.hyperparameters["trend"] = auto_trend

                if self.seasonal == "auto":
                    self.seasonal = auto_seasonal
                    self.hyperparameters["seasonal"] = auto_seasonal

        # Fit model
        try:
            # Build model
            self._model = ExponentialSmoothing(
                y,
                trend=self.trend,
                seasonal=self.seasonal,
                seasonal_periods=self.seasonal_periods if self.seasonal else None,
                damped_trend=self.damped_trend,
            )

            # Fit with optional smoothing parameters
            fit_params = {}
            if self.smoothing_level is not None:
                fit_params["smoothing_level"] = self.smoothing_level
            if self.smoothing_trend is not None:
                fit_params["smoothing_trend"] = self.smoothing_trend
            if self.smoothing_seasonal is not None:
                fit_params["smoothing_seasonal"] = self.smoothing_seasonal

            self._fitted_model = self._model.fit(**fit_params)

            # Store diagnostics
            if self._fitted_model is not None:
                self._diagnostics.update(
                    {
                        "aic": self._fitted_model.aic,
                        "bic": self._fitted_model.bic,
                        "aicc": self._fitted_model.aicc,
                        "smoothing_level": self._fitted_model.params["smoothing_level"],
                        "smoothing_trend": self._fitted_model.params.get(
                            "smoothing_trend"
                        ),
                        "smoothing_seasonal": self._fitted_model.params.get(
                            "smoothing_seasonal"
                        ),
                        "residual_std": np.std(self._fitted_model.resid),
                        "residual_mean": np.mean(self._fitted_model.resid),
                    }
                )

            logger.info(
                f"ETS(trend={self.trend}, seasonal={self.seasonal}) fitted. "
                f"AIC={self._diagnostics['aic']:.2f}"
            )

        except Exception as e:
            logger.error(f"Failed to fit ETS model: {e}")
            raise ValueError(f"Model fitting failed: {e}") from e

        return self

    def predict(self, steps: int) -> ForecastResult:
        """Generate forecasts for specified number of steps.

        Args:
            steps: Number of time steps to forecast.

        Returns:
            ForecastResult with predictions and confidence intervals.

        Raises:
            ValueError: If model is not fitted or prediction fails.
        """
        if self._fitted_model is None:
            raise ValueError("Model must be fitted before prediction")

        try:
            # Generate forecast
            forecast = self._fitted_model.forecast(steps=steps)

            # Get prediction intervals (simulate via residuals)
            # ETS doesn't provide native confidence intervals, so approximate
            residual_std = self._diagnostics["residual_std"]

            # 95% confidence interval (1.96 * std)
            margin = 1.96 * residual_std
            lower_bounds = (forecast - margin).tolist()
            upper_bounds = (forecast + margin).tolist()

            # Create date range
            last_date = self._training_data.index[-1]
            if isinstance(last_date, date | datetime):
                forecast_dates = pd.date_range(
                    start=last_date + timedelta(days=1),
                    periods=steps,
                    freq="D",
                )
            else:
                forecast_dates = range(
                    len(self._training_data), len(self._training_data) + steps
                )

            # Build result
            return ForecastResult(
                dates=[str(d) for d in forecast_dates],
                predictions=forecast.tolist(),
                lower_bound=lower_bounds,
                upper_bound=upper_bounds,
                confidence_level=0.95,
                model_name=f"ETS({self.trend},{self.seasonal})",
                metadata={
                    "trend": self.trend,
                    "seasonal": self.seasonal,
                    "seasonal_periods": self.seasonal_periods,
                    "damped_trend": self.damped_trend,
                    **self._diagnostics,
                },
            )

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise ValueError(f"Prediction failed: {e}") from e

    def get_metadata(self) -> ModelMetadata:
        """Get model metadata and diagnostics.

        Returns:
            ModelMetadata with model details and performance info.
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
            model_name=f"ETS({self.trend},{self.seasonal})",
            model_type="classical",
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
