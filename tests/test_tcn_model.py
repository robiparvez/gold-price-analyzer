"""Tests for Temporal Convolution Network (TCN) model."""

import numpy as np
import pandas as pd
import pytest

from models.deep_learning import TCNModel


@pytest.fixture
def small_time_series():
    """Create small time series (90 days)."""
    dates = pd.date_range(start="2025-10-07", periods=90, freq="D")
    prices = np.random.uniform(70000, 75000, 90)
    prices = pd.Series(prices, index=dates)
    X = pd.DataFrame({"price": prices.values}, index=dates)
    return X, prices


@pytest.fixture
def medium_time_series():
    """Create medium time series (180 days)."""
    dates = pd.date_range(start="2025-07-09", periods=180, freq="D")
    trend = np.linspace(70000, 75000, 180)
    noise = np.random.normal(0, 500, 180)
    prices = trend + noise
    prices = pd.Series(prices, index=dates)
    X = pd.DataFrame({"price": prices.values}, index=dates)
    return X, prices


class TestTCNModel:
    """Test TCN model."""

    def test_init(self):
        """Test model initialization."""
        model = TCNModel()
        assert model.lookback == 14
        assert model.forecast_horizon == 7
        assert model.filters == 32
        assert model.kernel_size == 3
        assert model.num_layers == 1

    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        model = TCNModel(
            lookback=21,
            forecast_horizon=14,
            filters=64,
            kernel_size=5,
            num_layers=2,
        )
        assert model.lookback == 21
        assert model.filters == 64
        assert model.num_layers == 2

    def test_receptive_field_calculation(self):
        """Test receptive field size calculation."""
        model = TCNModel(kernel_size=3, num_layers=1, dilation_base=2)
        rf = model._calculate_receptive_field()
        # RF = 1 + (3-1)*1 = 3
        assert rf == 3

        model2 = TCNModel(kernel_size=3, num_layers=2, dilation_base=2)
        rf2 = model2._calculate_receptive_field()
        # RF = 1 + (3-1)*1 + (3-1)*2 = 1 + 2 + 4 = 7
        assert rf2 == 7

    def test_fit(self, medium_time_series):
        """Test model fitting."""
        X, y = medium_time_series
        model = TCNModel(epochs=5, batch_size=8, validation_split=0.2)
        model.fit(X, y)

        assert model._fitted_model is not None
        assert model._feature_mean is not None
        assert "receptive_field" in model._diagnostics
        assert model._diagnostics["receptive_field"] > 0

    def test_predict(self, medium_time_series):
        """Test prediction."""
        X, y = medium_time_series
        model = TCNModel(lookback=10, forecast_horizon=7, epochs=3)
        model.fit(X, y)

        result = model.predict(steps=7)

        assert len(result.predictions) == 7
        assert len(result.lower_bound) == 7
        assert len(result.upper_bound) == 7
        assert all(
            lb < pred < ub
            for lb, pred, ub in zip(
                result.lower_bound, result.predictions, result.upper_bound
            )
        )

    def test_get_metadata(self, medium_time_series):
        """Test metadata extraction."""
        X, y = medium_time_series
        model = TCNModel(epochs=3)
        model.fit(X, y)

        metadata = model.get_metadata()
        assert metadata.model_name == "TCN"
        assert metadata.model_type == "deep_learning"
        assert metadata.training_samples > 0
        assert "receptive_field" in metadata.metrics


class TestTCNIntegration:
    """Integration tests for TCN model."""

    def test_tcn_with_different_configurations(self, medium_time_series):
        """Test TCN with different layer configurations."""
        X, y = medium_time_series

        # Single layer
        model1 = TCNModel(lookback=10, num_layers=1, epochs=2)
        model1.fit(X, y)
        result1 = model1.predict(steps=5)
        assert len(result1.predictions) == 5

        # Two layers
        model2 = TCNModel(lookback=10, num_layers=2, epochs=2)
        model2.fit(X, y)
        result2 = model2.predict(steps=5)
        assert len(result2.predictions) == 5

        # Results should differ
        assert not np.allclose(result1.predictions, result2.predictions)

    def test_tcn_with_small_dataset(self, small_time_series):
        """Test TCN with small 90-day dataset."""
        X, y = small_time_series

        model = TCNModel(lookback=7, forecast_horizon=3, epochs=2)
        model.fit(X, y)
        result = model.predict(steps=3)

        assert result is not None
        assert len(result.predictions) == 3

    def test_receptive_field_in_metadata(self, medium_time_series):
        """Test that receptive field is in metadata."""
        X, y = medium_time_series

        model = TCNModel(kernel_size=3, num_layers=2, epochs=2)
        model.fit(X, y)
        result = model.predict(steps=3)

        assert "receptive_field" in result.metadata
        assert result.metadata["receptive_field"] > 0
