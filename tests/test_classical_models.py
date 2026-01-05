"""Tests for classical time-series models (ARIMA, ETS)."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from models.classical import ARIMAModel, ETSModel


# Fixtures
@pytest.fixture
def small_time_series():  # type: ignore[misc]
    """Create small time series (90 days)."""
    dates = pd.date_range(start="2024-01-01", periods=90, freq="D")
    # Trend + noise
    trend = np.linspace(100, 110, 90)
    noise = np.random.normal(0, 1, 90)
    values = trend + noise

    df = pd.DataFrame({"date": dates})
    series = pd.Series(values, index=dates, name="price")
    return df, series


@pytest.fixture
def medium_time_series():  # type: ignore[misc]
    """Create medium time series (180 days, 6 months)."""
    dates = pd.date_range(start="2024-01-01", periods=180, freq="D")
    # Trend + seasonal + noise
    t = np.arange(180)
    trend = 100 + 0.05 * t
    seasonal = 5 * np.sin(2 * np.pi * t / 7)  # Weekly seasonality
    noise = np.random.normal(0, 1, 180)
    values = trend + seasonal + noise

    df = pd.DataFrame({"date": dates})
    series = pd.Series(values, index=dates, name="price")
    return df, series


@pytest.fixture
def stationary_series():  # type: ignore[misc]
    """Create stationary time series (no trend)."""
    dates = pd.date_range(start="2024-01-01", periods=120, freq="D")
    values = np.random.normal(100, 5, 120)

    df = pd.DataFrame({"date": dates})
    series = pd.Series(values, index=dates, name="price")
    return df, series


# ARIMA Model Tests
class TestARIMAModel:
    """Test suite for ARIMAModel."""

    def test_init_with_defaults(self):
        """Test ARIMA initialization with default parameters."""
        model = ARIMAModel()

        assert model.model_type == "arima"
        assert model.version == "1.0.0"
        assert model.use_auto is True
        assert model.max_p == 2
        assert model.max_d == 1
        assert model.max_q == 2
        assert model.information_criterion == "aic"

    def test_init_with_custom_order(self):
        """Test ARIMA initialization with custom order."""
        model = ARIMAModel(order=(1, 1, 1), use_auto=False)

        assert model.order == (1, 1, 1)
        assert model.use_auto is False

    def test_init_with_seasonal(self):
        """Test ARIMA initialization with seasonal component."""
        model = ARIMAModel(seasonal_order=(1, 0, 1, 7))

        assert model.seasonal_order == (1, 0, 1, 7)

    def test_check_stationarity_stationary(self, stationary_series):
        """Test stationarity check on stationary data."""
        df, series = stationary_series
        model = ARIMAModel()

        result = model._check_stationarity(series)

        assert "adf_statistic" in result
        assert "p_value" in result
        assert "is_stationary" in result
        assert isinstance(result["is_stationary"], bool | np.bool_)

    def test_suggest_orders(self, medium_time_series):
        """Test automatic order suggestion."""
        df, series = medium_time_series
        model = ARIMAModel()

        suggestions = model._suggest_orders(series)

        assert "suggested_p" in suggestions
        assert "suggested_q" in suggestions
        assert "suggested_d" in suggestions
        assert "stationarity" in suggestions

        # Check ranges
        assert 0 <= suggestions["suggested_p"] <= model.max_p
        assert 0 <= suggestions["suggested_q"] <= model.max_q
        assert 0 <= suggestions["suggested_d"] <= model.max_d

    def test_auto_select_order(self, medium_time_series):
        """Test automatic order selection via grid search."""
        df, series = medium_time_series
        model = ARIMAModel(max_p=2, max_d=1, max_q=2)

        best_order = model._auto_select_order(series)

        assert isinstance(best_order, tuple)
        assert len(best_order) == 3
        assert all(isinstance(x, int) for x in best_order)
        assert 0 <= best_order[0] <= 2  # p
        assert 0 <= best_order[1] <= 1  # d
        assert 0 <= best_order[2] <= 2  # q

    def test_fit_with_auto_selection(self, medium_time_series):
        """Test fitting ARIMA with automatic order selection."""
        df, series = medium_time_series
        model = ARIMAModel(use_auto=True)

        fitted_model = model.fit(df, series)

        assert fitted_model == model
        assert model._fitted_model is not None
        assert model.order is not None
        assert "aic" in model._diagnostics
        assert "bic" in model._diagnostics

    def test_fit_with_custom_order(self, small_time_series):
        """Test fitting ARIMA with custom order."""
        df, series = small_time_series
        model = ARIMAModel(order=(1, 1, 1), use_auto=False)

        fitted_model = model.fit(df, series)

        assert fitted_model == model
        assert model._fitted_model is not None
        assert model.order == (1, 1, 1)

    def test_predict_single_step(self, medium_time_series):
        """Test single-step forecast."""
        df, series = medium_time_series
        model = ARIMAModel(order=(1, 1, 1), use_auto=False)
        model.fit(df, series)

        result = model.predict(steps=1)

        assert result.model_name.startswith("ARIMA")
        assert len(result.predictions) == 1
        assert len(result.lower_bound) == 1
        assert len(result.upper_bound) == 1
        assert result.lower_bound[0] < result.predictions[0]
        assert result.predictions[0] < result.upper_bound[0]

    def test_predict_multi_step(self, medium_time_series):
        """Test multi-step forecast."""
        df, series = medium_time_series
        model = ARIMAModel(order=(1, 1, 1), use_auto=False)
        model.fit(df, series)

        steps = 7
        result = model.predict(steps=steps)

        assert len(result.predictions) == steps
        assert len(result.lower_bound) == steps
        assert len(result.upper_bound) == steps
        assert len(result.dates) == steps

    def test_predict_without_fit_raises_error(self):
        """Test that predicting without fitting raises error."""
        model = ARIMAModel()

        with pytest.raises(ValueError, match="Model must be fitted"):
            model.predict(steps=1)

    def test_get_metadata(self, medium_time_series):
        """Test metadata extraction."""
        df, series = medium_time_series
        model = ARIMAModel(order=(1, 1, 1), use_auto=False)
        model.fit(df, series)

        metadata = model.get_metadata()

        assert metadata.model_type == "classical"
        assert "order" in metadata.hyperparameters
        assert "aic" in metadata.metrics
        assert metadata.training_samples == 180

    def test_forecast_dates_are_future(self, medium_time_series):
        """Test that forecast dates are in the future."""
        df, series = medium_time_series
        model = ARIMAModel(order=(1, 1, 1), use_auto=False)
        model.fit(df, series)

        result = model.predict(steps=5)

        last_train_date = series.index[-1]
        # Dates are strings, so parse them for comparison
        result_dates = [
            (
                datetime.fromisoformat(d.split()[0])
                if " " in d
                else datetime.fromisoformat(d)
            )
            for d in result.dates
        ]
        assert all(d.date() > last_train_date.date() for d in result_dates)

    def test_diagnostics_stored(self, medium_time_series):
        """Test that diagnostics are properly stored."""
        df, series = medium_time_series
        model = ARIMAModel(order=(1, 1, 1), use_auto=False)
        model.fit(df, series)

        assert "aic" in model._diagnostics
        assert "bic" in model._diagnostics
        assert "residual_std" in model._diagnostics
        assert "converged" in model._diagnostics


# ETS Model Tests
class TestETSModel:
    """Test suite for ETSModel."""

    def test_init_with_defaults(self):
        """Test ETS initialization with default parameters."""
        model = ETSModel()

        assert model.model_type == "ets"
        assert model.version == "1.0.0"
        assert model.trend == "auto"
        assert model.seasonal == "auto"
        assert model.seasonal_periods == 7
        assert model.use_auto is True

    def test_init_with_custom_components(self):
        """Test ETS initialization with custom components."""
        model = ETSModel(trend="add", seasonal="mul", seasonal_periods=12)

        assert model.trend == "add"
        assert model.seasonal == "mul"
        assert model.seasonal_periods == 12

    def test_detect_seasonality_insufficient_data(self):
        """Test seasonality detection with insufficient data."""
        dates = pd.date_range(start="2024-01-01", periods=10, freq="D")
        series = pd.Series(np.random.randn(10), index=dates)

        model = ETSModel(seasonal_periods=7)
        result = model._detect_seasonality(series)

        assert result["has_seasonality"] is False
        assert "reason" in result

    def test_detect_seasonality_with_pattern(self):
        """Test seasonality detection with clear pattern."""
        dates = pd.date_range(start="2024-01-01", periods=84, freq="D")  # 12 weeks
        t = np.arange(84)
        seasonal = 10 * np.sin(2 * np.pi * t / 7)
        series = pd.Series(seasonal, index=dates)

        model = ETSModel(seasonal_periods=7)
        result = model._detect_seasonality(series)

        assert "has_seasonality" in result
        assert "coefficient_of_variation" in result

    def test_detect_trend_with_trend(self):
        """Test trend detection with clear trend."""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        trend = np.linspace(100, 150, 100)
        series = pd.Series(trend, index=dates)

        model = ETSModel()
        result = model._detect_trend(series)

        assert result["has_trend"] == True  # noqa: E712
        assert result["slope"] > 0
        assert result["r_squared"] > 0.9

    def test_detect_trend_without_trend(self, stationary_series):
        """Test trend detection without trend."""
        df, series = stationary_series
        model = ETSModel()

        result = model._detect_trend(series)

        assert "has_trend" in result
        assert "slope" in result
        assert "r_squared" in result

    def test_auto_select_components(self, medium_time_series):
        """Test automatic component selection."""
        df, series = medium_time_series
        model = ETSModel()

        trend, seasonal = model._auto_select_components(series)

        assert trend in [None, "add", "mul"]
        assert seasonal in [None, "add", "mul"]
        assert "trend_detection" in model._diagnostics
        assert "seasonal_detection" in model._diagnostics

    def test_fit_with_auto_components(self, medium_time_series):
        """Test fitting ETS with automatic component selection."""
        df, series = medium_time_series
        model = ETSModel(trend="auto", seasonal="auto", use_auto=True)

        fitted_model = model.fit(df, series)

        assert fitted_model == model
        assert model._fitted_model is not None
        assert model.trend in [None, "add", "mul"]
        assert model.seasonal in [None, "add", "mul"]

    def test_fit_with_custom_components(self, small_time_series):
        """Test fitting ETS with custom components."""
        df, series = small_time_series
        model = ETSModel(trend="add", seasonal=None, use_auto=False)

        fitted_model = model.fit(df, series)

        assert fitted_model == model
        assert model._fitted_model is not None
        assert model.trend == "add"
        assert model.seasonal is None

    def test_predict_single_step(self, medium_time_series):
        """Test single-step forecast."""
        df, series = medium_time_series
        model = ETSModel(trend="add", seasonal=None)
        model.fit(df, series)

        result = model.predict(steps=1)

        assert result.model_name.startswith("ETS")
        assert len(result.predictions) == 1
        assert len(result.lower_bound) == 1
        assert len(result.upper_bound) == 1

    def test_predict_multi_step(self, medium_time_series):
        """Test multi-step forecast."""
        df, series = medium_time_series
        model = ETSModel(trend="add", seasonal=None)
        model.fit(df, series)

        steps = 7
        result = model.predict(steps=steps)

        assert len(result.predictions) == steps
        assert len(result.lower_bound) == steps
        assert len(result.upper_bound) == steps
        assert len(result.dates) == steps

    def test_predict_without_fit_raises_error(self):
        """Test that predicting without fitting raises error."""
        model = ETSModel()

        with pytest.raises(ValueError, match="Model must be fitted"):
            model.predict(steps=1)

    def test_get_metadata(self, medium_time_series):
        """Test metadata extraction."""
        df, series = medium_time_series
        model = ETSModel(trend="add", seasonal=None)
        model.fit(df, series)

        metadata = model.get_metadata()

        assert metadata.model_type == "classical"
        assert "trend" in metadata.hyperparameters
        assert "aic" in metadata.metrics
        assert metadata.training_samples == 180

    def test_confidence_intervals_valid(self, medium_time_series):
        """Test that confidence intervals are properly bounded."""
        df, series = medium_time_series
        model = ETSModel(trend="add", seasonal=None)
        model.fit(df, series)

        result = model.predict(steps=5)

        for i in range(5):
            assert result.lower_bound[i] < result.predictions[i]
            assert result.predictions[i] < result.upper_bound[i]

    def test_diagnostics_stored(self, medium_time_series):
        """Test that diagnostics are properly stored."""
        df, series = medium_time_series
        model = ETSModel(trend="add", seasonal=None)
        model.fit(df, series)

        assert "aic" in model._diagnostics
        assert "bic" in model._diagnostics
        assert "smoothing_level" in model._diagnostics
        assert "residual_std" in model._diagnostics


# Integration Tests
class TestClassicalModelsIntegration:
    """Integration tests for classical models."""

    def test_arima_and_ets_comparable_results(self, medium_time_series):
        """Test that ARIMA and ETS produce comparable forecasts."""
        df, series = medium_time_series

        arima = ARIMAModel(order=(1, 1, 1), use_auto=False)
        ets = ETSModel(trend="add", seasonal=None)

        arima.fit(df, series)
        ets.fit(df, series)

        arima_result = arima.predict(steps=7)
        ets_result = ets.predict(steps=7)

        # Both should produce forecasts in similar range
        arima_mean = np.mean(arima_result.predictions)
        ets_mean = np.mean(ets_result.predictions)

        # Allow 20% difference (different methodologies)
        assert abs(arima_mean - ets_mean) / arima_mean < 0.2

    def test_models_handle_small_dataset(self, small_time_series):
        """Test that models can handle small datasets (90 days)."""
        df, series = small_time_series

        arima = ARIMAModel(order=(1, 0, 1), use_auto=False)
        ets = ETSModel(trend="add", seasonal=None)

        # Both should fit without errors
        arima.fit(df, series)
        ets.fit(df, series)

        # Both should predict
        arima_result = arima.predict(steps=3)
        ets_result = ets.predict(steps=3)

        assert len(arima_result.predictions) == 3
        assert len(ets_result.predictions) == 3

    def test_forecast_dates_consistency(self, medium_time_series):
        """Test that forecast dates are consistent across models."""
        df, series = medium_time_series

        arima = ARIMAModel(order=(1, 1, 1), use_auto=False)
        ets = ETSModel(trend="add", seasonal=None)

        arima.fit(df, series)
        ets.fit(df, series)

        arima_result = arima.predict(steps=5)
        ets_result = ets.predict(steps=5)

        # Same forecast dates
        assert arima_result.dates == ets_result.dates
