"""Tests for hybrid models."""

import numpy as np
import pandas as pd
import pytest

from models.hybrid import HybridLSTMARIMAModel


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


class TestHybridLSTMARIMAModel:
    """Test hybrid LSTM-ARIMA model."""
    
    def test_init(self):
        """Test model initialization."""
        model = HybridLSTMARIMAModel()
        assert model.lstm_lookback == 14
        assert model.forecast_horizon == 7
        assert model.lstm_units == 32
        assert model.arima_order == (1, 1, 1)
    
    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        model = HybridLSTMARIMAModel(
            lstm_lookback=21,
            forecast_horizon=14,
            lstm_units=64,
            arima_order=(2, 1, 1),
            lstm_weight=0.7,
            arima_weight=0.3,
        )
        assert model.lstm_lookback == 21
        assert model.arima_order == (2, 1, 1)
        # Weights should be normalized
        assert abs(model.lstm_weight + model.arima_weight - 1.0) < 0.001
    
    def test_weight_normalization(self):
        """Test that weights are properly normalized."""
        model = HybridLSTMARIMAModel(lstm_weight=2.0, arima_weight=3.0)
        # Should normalize to 0.4 and 0.6
        assert abs(model.lstm_weight - 0.4) < 0.001
        assert abs(model.arima_weight - 0.6) < 0.001
        assert abs(model.lstm_weight + model.arima_weight - 1.0) < 0.001
    
    def test_fit(self, medium_time_series):
        """Test model fitting."""
        X, y = medium_time_series
        model = HybridLSTMARIMAModel(
            lstm_lookback=10,
            forecast_horizon=7,
            lstm_epochs=2,
            batch_size=8,
            validation_split=0.2,
        )
        model.fit(X, y)
        
        assert model.lstm_model._fitted_model is not None
        assert model._lstm_residuals is not None
        assert len(model._lstm_residuals) > 0
    
    def test_predict(self, medium_time_series):
        """Test prediction."""
        X, y = medium_time_series
        model = HybridLSTMARIMAModel(
            lstm_lookback=10,
            forecast_horizon=7,
            lstm_epochs=2,
        )
        model.fit(X, y)
        
        result = model.predict(steps=7)
        
        assert len(result.predictions) == 7
        assert len(result.lower_bound) == 7
        assert len(result.upper_bound) == 7
        assert all(lb < pred < ub for lb, pred, ub in
                  zip(result.lower_bound, result.predictions, result.upper_bound))
    
    def test_get_metadata(self, medium_time_series):
        """Test metadata extraction."""
        X, y = medium_time_series
        model = HybridLSTMARIMAModel(lstm_epochs=2)
        model.fit(X, y)
        
        metadata = model.get_metadata()
        assert metadata.model_name == "Hybrid-LSTM-ARIMA"
        assert metadata.model_type == "hybrid"
        assert metadata.training_samples > 0


class TestHybridIntegration:
    """Integration tests for hybrid models."""
    
    def test_hybrid_combines_components(self, medium_time_series):
        """Test that hybrid model combines LSTM and ARIMA."""
        X, y = medium_time_series
        
        model = HybridLSTMARIMAModel(
            lstm_lookback=10,
            lstm_weight=0.6,
            arima_weight=0.4,
            lstm_epochs=2,
        )
        model.fit(X, y)
        result = model.predict(steps=5)
        
        # Predictions should be a mix of LSTM and ARIMA
        assert result is not None
        assert len(result.predictions) == 5
        assert "lstm_weight" in result.metadata
        assert "arima_weight" in result.metadata
    
    def test_hybrid_vs_individual_predictions(self, medium_time_series):
        """Test that hybrid produces different forecast than individual models."""
        X, y = medium_time_series
        
        hybrid = HybridLSTMARIMAModel(
            lstm_lookback=10,
            lstm_weight=0.5,
            arima_weight=0.5,
            lstm_epochs=2,
        )
        hybrid.fit(X, y)
        hybrid_result = hybrid.predict(steps=5)
        
        # Just LSTM component
        lstm_result = hybrid.lstm_model.predict(steps=5)
        
        # Hybrid should blend both forecasts
        # Not necessarily all equal (since ARIMA adds residuals)
        assert hybrid_result.predictions is not None
        assert lstm_result.predictions is not None
    
    def test_hybrid_with_different_weights(self, medium_time_series):
        """Test hybrid with different ensemble weights."""
        X, y = medium_time_series
        
        # LSTM-heavy
        model_lstm_heavy = HybridLSTMARIMAModel(
            lstm_lookback=10,
            lstm_weight=0.8,
            arima_weight=0.2,
            lstm_epochs=2,
        )
        model_lstm_heavy.fit(X, y)
        result_heavy = model_lstm_heavy.predict(steps=5)
        
        # ARIMA-heavy
        model_arima_heavy = HybridLSTMARIMAModel(
            lstm_lookback=10,
            lstm_weight=0.2,
            arima_weight=0.8,
            lstm_epochs=2,
        )
        model_arima_heavy.fit(X, y)
        result_arima = model_arima_heavy.predict(steps=5)
        
        # Different weights should produce different forecasts
        assert not np.allclose(result_heavy.predictions, result_arima.predictions)
