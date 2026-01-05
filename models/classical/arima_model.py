"""ARIMA/SARIMA forecasting model with automatic parameter selection.

This module provides an ARIMA implementation optimized for limited historical data
(6-month constraint). Uses conservative parameter ranges and automatic selection.
"""

import logging
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf, adfuller, pacf

from models.time_series_base import BaseTimeSeriesModel, ForecastResult, ModelMetadata

logger = logging.getLogger(__name__)


class ARIMAModel(BaseTimeSeriesModel):
    """ARIMA/SARIMA model with automatic parameter selection.

    Features:
    - Automatic order selection via AIC/BIC minimization
    - Stationarity testing (ADF test)
    - ACF/PACF analysis for parameter suggestions
    - Conservative parameter ranges for small datasets
    - Seasonal ARIMA support (optional)
    - Residual diagnostics

    Attributes:
        order: ARIMA order (p, d, q) - auto-selected if None
        seasonal_order: Seasonal order (P, D, Q, s) - None for non-seasonal
        use_auto: Whether to use automatic order selection
        max_p: Maximum AR order (default 2 for small datasets)
        max_d: Maximum differencing order (default 1)
        max_q: Maximum MA order (default 2 for small datasets)
        information_criterion: 'aic' or 'bic' for model selection
        trend: Trend parameter ('n', 'c', 't', 'ct')
    """

    def __init__(
        self,
        order: tuple[int, int, int] | None = None,
        seasonal_order: tuple[int, int, int, int] | None = None,
        use_auto: bool = True,
        max_p: int = 2,
        max_d: int = 1,
        max_q: int = 2,
        information_criterion: str = "aic",
        trend: str = "c",
        **kwargs,
    ):
        """Initialize ARIMA model.

        Args:
            order: ARIMA order (p, d, q). If None, auto-selected.
            seasonal_order: Seasonal order (P, D, Q, s). None for non-seasonal.
            use_auto: Whether to automatically select best order.
            max_p: Maximum AR order to test.
            max_d: Maximum differencing order to test.
            max_q: Maximum MA order to test.
            information_criterion: 'aic' or 'bic' for model selection.
            trend: Trend parameter ('n', 'c', 't', 'ct').
            **kwargs: Additional arguments passed to BaseTimeSeriesModel.
        """
        super().__init__(purity="22K", **kwargs)

        # Store model metadata
        self.model_type = "arima"
        self.version = "1.0.0"
        self.hyperparameters = {
            "order": order,
            "seasonal_order": seasonal_order,
            "use_auto": use_auto,
            "max_p": max_p,
            "max_d": max_d,
            "max_q": max_q,
            "information_criterion": information_criterion,
            "trend": trend,
        }

        # ARIMA-specific parameters
        self.order = order
        self.seasonal_order = seasonal_order
        self.use_auto = use_auto
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        self.information_criterion = information_criterion
        self.trend = trend

        self._model = None
        self._fitted_model = None
        self._training_data = None
        self._diagnostics = {}

    def _check_stationarity(self, data: pd.Series) -> dict[str, Any]:
        """Perform Augmented Dickey-Fuller test for stationarity.

        Args:
            data: Time series data.

        Returns:
            Dictionary with test results (adf_statistic, p_value, is_stationary).
        """
        result = adfuller(data.dropna(), autolag="AIC")
        return {
            "adf_statistic": result[0],
            "p_value": result[1],
            "critical_values": result[4],
            "is_stationary": result[1] < 0.05,
        }

    def _suggest_orders(self, data: pd.Series) -> dict[str, Any]:
        """Suggest ARIMA orders based on ACF/PACF analysis.

        Args:
            data: Time series data.

        Returns:
            Dictionary with suggested orders and analysis results.
        """
        # Check stationarity
        stationarity = self._check_stationarity(data)

        # Calculate ACF and PACF
        acf_values = acf(data.dropna(), nlags=min(20, len(data) // 3))
        pacf_values = pacf(data.dropna(), nlags=min(20, len(data) // 3))

        # Suggest p based on PACF (significant lags)
        pacf_cutoff = 1.96 / np.sqrt(len(data))
        suggested_p = np.sum(np.abs(pacf_values[1:]) > pacf_cutoff)
        suggested_p = min(suggested_p, self.max_p)

        # Suggest q based on ACF (significant lags)
        acf_cutoff = 1.96 / np.sqrt(len(data))
        suggested_q = np.sum(np.abs(acf_values[1:]) > acf_cutoff)
        suggested_q = min(suggested_q, self.max_q)

        # Suggest d based on stationarity
        suggested_d = 0 if stationarity["is_stationary"] else 1

        return {
            "suggested_p": suggested_p,
            "suggested_q": suggested_q,
            "suggested_d": suggested_d,
            "stationarity": stationarity,
            "acf_cutoff": acf_cutoff,
            "pacf_cutoff": pacf_cutoff,
        }

    def _auto_select_order(self, data: pd.Series) -> tuple[int, int, int]:
        """Automatically select best ARIMA order using grid search.

        Args:
            data: Time series data.

        Returns:
            Best order (p, d, q) based on information criterion.
        """
        # Get suggestions from ACF/PACF
        suggestions = self._suggest_orders(data)
        self._diagnostics["order_suggestions"] = suggestions

        # Grid search over parameter space
        best_score = np.inf
        best_order = (0, 0, 0)

        # Start with suggested values and expand search
        p_range = range(
            max(0, suggestions["suggested_p"] - 1),
            min(suggestions["suggested_p"] + 2, self.max_p + 1),
        )
        d_range = range(
            suggestions["suggested_d"],
            min(suggestions["suggested_d"] + 2, self.max_d + 1),
        )
        q_range = range(
            max(0, suggestions["suggested_q"] - 1),
            min(suggestions["suggested_q"] + 2, self.max_q + 1),
        )

        for p in p_range:
            for d in d_range:
                for q in q_range:
                    try:
                        model = ARIMA(data, order=(p, d, q), trend=self.trend)
                        fitted = model.fit()

                        score = (
                            fitted.aic
                            if self.information_criterion == "aic"
                            else fitted.bic
                        )

                        if score < best_score:
                            best_score = score
                            best_order = (p, d, q)
                    except Exception as e:
                        logger.debug(f"Failed to fit ARIMA({p},{d},{q}): {e}")
                        continue

        logger.info(
            f"Auto-selected order: {best_order} "
            f"({self.information_criterion.upper()}={best_score:.2f})"
        )

        return best_order

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ARIMAModel":
        """Fit ARIMA model to training data.

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

        # Auto-select order if needed
        if self.use_auto and self.order is None:
            self.order = self._auto_select_order(y)
            self.hyperparameters["order"] = self.order
        elif self.order is None:
            # Default to (1, 1, 1) if no order specified
            self.order = (1, 1, 1)
            self.hyperparameters["order"] = self.order

        # Fit model
        try:
            # Adjust trend parameter for differencing order
            # When d > 0, constant cannot be used - use 'n' (none) instead
            trend_param = "n" if self.order[1] > 0 else self.trend

            if self.seasonal_order is not None:
                self._model = SARIMAX(
                    y,
                    order=self.order,
                    seasonal_order=self.seasonal_order,
                    trend=trend_param,
                )
            else:
                self._model = ARIMA(y, order=self.order, trend=trend_param)

            self._fitted_model = self._model.fit()

            # Store diagnostics
            self._diagnostics.update(
                {
                    "aic": self._fitted_model.aic,
                    "bic": self._fitted_model.bic,
                    "hqic": self._fitted_model.hqic,
                    "converged": self._fitted_model.mle_retvals["converged"],
                    "residual_std": np.std(self._fitted_model.resid),
                    "residual_mean": np.mean(self._fitted_model.resid),
                }
            )

            logger.info(
                f"ARIMA{self.order} fitted successfully. "
                f"AIC={self._diagnostics['aic']:.2f}, "
                f"BIC={self._diagnostics['bic']:.2f}"
            )

        except Exception as e:
            logger.error(f"Failed to fit ARIMA model: {e}")
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

            # Get confidence intervals (95% default)
            forecast_df = self._fitted_model.get_forecast(steps=steps)
            conf_int = forecast_df.conf_int(alpha=0.05)

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
                lower_bound=conf_int.iloc[:, 0].tolist(),
                upper_bound=conf_int.iloc[:, 1].tolist(),
                confidence_level=0.95,
                model_name=f"ARIMA{self.order}",
                metadata={
                    "order": self.order,
                    "seasonal_order": self.seasonal_order,
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
            model_name=f"ARIMA{self.order}",
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
