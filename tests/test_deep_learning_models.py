"""Tests for deep learning models (LSTM and GRU)."""

import numpy as np
import pandas as pd
import pytest

from models.deep_learning import GRUModel, LSTMModel


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
    # Create more realistic time series with trend
    trend = np.linspace(70000, 75000, 180)
    noise = np.random.normal(0, 500, 180)
    prices = trend + noise
    prices = pd.Series(prices, index=dates)
    X = pd.DataFrame({"price": prices.values}, index=dates)
    return X, prices


class TestLSTMModel:
    """Test LSTM model."""

    def test_init(self):
        """Test model initialization."""
        model = LSTMModel()
        assert model.lookback == 14
        assert model.forecast_horizon == 7
        assert model.units_per_layer == 32
        assert model.num_layers == 1
        assert model.learning_rate == 0.001
        assert model.version == "1.0.0"

    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        model = LSTMModel(
            lookback=21,
            forecast_horizon=14,
            units_per_layer=64,
            num_layers=2,
            learning_rate=0.0005,
        )
        assert model.lookback == 21
        assert model.forecast_horizon == 14
        assert model.units_per_layer == 64
        assert model.num_layers == 2

    def test_fit(self, medium_time_series):
        """Test model fitting."""
        X, y = medium_time_series
        model = LSTMModel(epochs=5, batch_size=8, validation_split=0.2)
        model.fit(X, y)

        assert model._fitted_model is not None
        assert model._feature_mean is not None
        assert model._target_std is not None
        assert "final_val_loss" in model._diagnostics

    def test_predict(self, medium_time_series):
        """Test prediction."""
        X, y = medium_time_series
        model = LSTMModel(lookback=10, forecast_horizon=7, epochs=3)
        model.fit(X, y)

        result = model.predict(steps=7)

        assert len(result.predictions) == 7
        assert len(result.lower_bound) == 7
        assert len(result.upper_bound) == 7
        assert len(result.dates) == 7
        assert all(
            lb < pred < ub
            for lb, pred, ub in zip(
                result.lower_bound, result.predictions, result.upper_bound
            )
        )

    def test_get_metadata(self, medium_time_series):
        """Test metadata extraction."""
        X, y = medium_time_series
        model = LSTMModel(epochs=3)
        model.fit(X, y)

        metadata = model.get_metadata()
        assert metadata.model_name == "LSTM"
        assert metadata.model_type == "deep_learning"
        assert metadata.training_samples > 0
        assert metadata.version == "1.0.0"


class TestGRUModel:
    """Test GRU model."""

    def test_init(self):
        """Test model initialization."""
        model = GRUModel()
        assert model.lookback == 14
        assert model.forecast_horizon == 7
        assert model.units_per_layer == 32
        assert model.num_layers == 1
        assert model.version == "1.0.0"

    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        model = GRUModel(
            lookback=28,
            forecast_horizon=14,
            units_per_layer=48,
            num_layers=2,
            dropout_rate=0.3,
        )
        assert model.lookback == 28
        assert model.num_layers == 2
        assert model.dropout_rate == 0.3

    def test_fit(self, medium_time_series):
        """Test model fitting."""
        X, y = medium_time_series
        model = GRUModel(epochs=5, batch_size=8)
        model.fit(X, y)

        assert model._fitted_model is not None
        assert model._feature_mean is not None
        assert "final_val_loss" in model._diagnostics
        assert model._diagnostics["final_val_loss"] >= 0

    def test_predict(self, medium_time_series):
        """Test prediction."""
        X, y = medium_time_series
        model = GRUModel(lookback=10, forecast_horizon=7, epochs=3)
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
        model = GRUModel(epochs=3)
        model.fit(X, y)

        metadata = model.get_metadata()
        assert metadata.model_name == "GRU"
        assert metadata.model_type == "deep_learning"
        assert metadata.training_samples > 0


class TestDeepLearningIntegration:
    """Integration tests for deep learning models."""

    def test_lstm_vs_gru_on_same_data(self, medium_time_series):
        """Test both models on same data."""
        X, y = medium_time_series

        lstm_model = LSTMModel(lookback=10, epochs=3)
        gru_model = GRUModel(lookback=10, epochs=3)

        lstm_model.fit(X, y)
        gru_model.fit(X, y)

        lstm_result = lstm_model.predict(steps=5)
        gru_result = gru_model.predict(steps=5)

        assert len(lstm_result.predictions) == 5
        assert len(gru_result.predictions) == 5

        # Models should produce different forecasts
        assert not np.allclose(lstm_result.predictions, gru_result.predictions)

    def test_models_with_small_dataset(self, small_time_series):
        """Test models with small 90-day dataset."""
        X, y = small_time_series

        lstm = LSTMModel(lookback=7, forecast_horizon=3, epochs=2)
        gru = GRUModel(lookback=7, forecast_horizon=3, epochs=2)

        # Should handle augmentation for small datasets
        lstm.fit(X, y)
        gru.fit(X, y)

        lstm_result = lstm.predict(steps=3)
        gru_result = gru.predict(steps=3)

        assert lstm_result is not None
        assert gru_result is not None

    def test_deep_learning_produces_realistic_bounds(self, medium_time_series):
        """Test that confidence intervals are realistic."""
        X, y = medium_time_series

        model = LSTMModel(lookback=10, epochs=3)
        model.fit(X, y)
        result = model.predict(steps=7)

        # Bounds should have reasonable width relative to predictions
        widths = [ub - lb for lb, ub in zip(result.lower_bound, result.upper_bound)]
        avg_width = np.mean(widths)
        avg_pred = np.mean(result.predictions)

        # Width should be 2-10% of average prediction
        assert 0.02 * avg_pred <= avg_width <= 0.1 * avg_pred
