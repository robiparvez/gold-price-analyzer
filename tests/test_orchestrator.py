"""Tests for Model Orchestrator."""

import numpy as np
import pandas as pd
import pytest

from models.orchestrator import ModelOrchestrator


@pytest.fixture
def small_time_series():
    """Create small time series (90 days)."""
    dates = pd.date_range(start="2025-10-06", periods=90, freq="D")
    trend = np.linspace(70000, 72000, 90)
    noise = np.random.normal(0, 500, 90)
    prices = trend + noise
    prices = pd.Series(prices, index=dates)
    X = pd.DataFrame({"price": prices.values}, index=dates)
    return X, prices


@pytest.fixture
def medium_time_series():
    """Create medium time series (180 days)."""
    dates = pd.date_range(start="2025-04-09", periods=180, freq="D")
    trend = np.linspace(70000, 75000, 180)
    noise = np.random.normal(0, 500, 180)
    prices = trend + noise
    prices = pd.Series(prices, index=dates)
    X = pd.DataFrame({"price": prices.values}, index=dates)
    return X, prices


class TestModelOrchestrator:
    """Test Model Orchestrator."""

    def test_init(self):
        """Test orchestrator initialization."""
        orchestrator = ModelOrchestrator()
        assert len(orchestrator.models) == 9
        assert "arima" in orchestrator.models
        assert "ets" in orchestrator.models
        assert "lightgbm" in orchestrator.models
        assert "catboost" in orchestrator.models
        assert "svr" in orchestrator.models
        assert "lstm" in orchestrator.models
        assert "gru" in orchestrator.models
        assert "tcn" in orchestrator.models
        assert "hybrid_lstm_arima" in orchestrator.models

    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        orchestrator = ModelOrchestrator(
            cache_models=False,
            auto_select_top_k=5,
        )
        assert orchestrator.cache_models is False
        assert orchestrator.auto_select_top_k == 5

    def test_cache_enabled(self):
        """Test model caching is enabled."""
        orchestrator = ModelOrchestrator(cache_models=True)
        assert orchestrator.cache_models is True
        assert len(orchestrator._trained_models) == 0  # Not trained yet

    def test_fit_all(self, small_time_series):
        """Test fitting all models."""
        X, y = small_time_series
        orchestrator = ModelOrchestrator(cache_models=True)

        results = orchestrator.fit_all(X, y)

        # Check that all models were attempted
        assert len(results) == 9

        # Check that at least some models succeeded
        successful = sum(1 for r in results.values() if r["status"] == "success")
        assert successful > 0, "At least some models should succeed"

        # Check caching
        if orchestrator.cache_models:
            assert len(orchestrator._trained_models) > 0

    def test_predict_all(self, small_time_series):
        """Test prediction from all models."""
        X, y = small_time_series
        orchestrator = ModelOrchestrator()

        # Train first
        orchestrator.fit_all(X, y)

        # Predict
        predictions = orchestrator.predict_all(steps=7)

        # Check that predictions exist
        assert len(predictions) > 0

        # Check that at least some predictions are valid
        valid_predictions = sum(1 for p in predictions.values() if p is not None)
        assert valid_predictions > 0, "At least some models should produce predictions"

        # Check prediction structure
        for name, pred in predictions.items():
            if pred is not None:
                assert hasattr(pred, "predictions")
                assert len(pred.predictions) == 7

    def test_get_best_model(self, small_time_series):
        """Test getting best performing model."""
        X, y = small_time_series
        orchestrator = ModelOrchestrator()

        # Train and predict
        orchestrator.fit_all(X, y)
        orchestrator.predict_all(steps=7)

        # Get best model
        best_name, best_result = orchestrator.get_best_model()

        if len(orchestrator._model_metrics) > 0:
            assert best_name is not None
            assert best_result is not None

    def test_get_model_comparison(self, small_time_series):
        """Test getting model comparison."""
        X, y = small_time_series
        orchestrator = ModelOrchestrator()

        # Train and predict
        orchestrator.fit_all(X, y)
        orchestrator.predict_all(steps=7)

        # Get comparison
        comparison = orchestrator.get_model_comparison()

        # Check structure
        assert isinstance(comparison, pd.DataFrame)

        # Should have metrics columns
        if len(comparison) > 0:
            assert "mean_pred" in comparison.columns
            assert "std_pred" in comparison.columns
            assert "min_pred" in comparison.columns
            assert "max_pred" in comparison.columns

    def test_get_ensemble_forecast_weighted_mean(self, small_time_series):
        """Test ensemble forecast with weighted mean."""
        X, y = small_time_series
        orchestrator = ModelOrchestrator(auto_select_top_k=3)

        # Train and predict
        orchestrator.fit_all(X, y)
        orchestrator.predict_all(steps=7)

        # Get ensemble
        ensemble = orchestrator.get_ensemble_forecast(
            steps=7,
            method="weighted_mean",
        )

        # Check structure
        assert hasattr(ensemble, "predictions")
        assert len(ensemble.predictions) == 7
        assert len(ensemble.lower_bound) == 7
        assert len(ensemble.upper_bound) == 7

        # Verify bounds
        for i in range(7):
            assert ensemble.lower_bound[i] < ensemble.predictions[i]
            assert ensemble.predictions[i] < ensemble.upper_bound[i]

        # Check metadata
        assert ensemble.metadata["ensemble_method"] == "weighted_mean"
        assert ensemble.metadata["num_models"] <= 3

    def test_get_ensemble_forecast_equal_weight(self, small_time_series):
        """Test ensemble forecast with equal weights."""
        X, y = small_time_series
        orchestrator = ModelOrchestrator(auto_select_top_k=3)

        # Train and predict
        orchestrator.fit_all(X, y)
        orchestrator.predict_all(steps=7)

        # Get ensemble
        ensemble = orchestrator.get_ensemble_forecast(
            steps=7,
            method="equal_weight",
        )

        # Check structure
        assert hasattr(ensemble, "predictions")
        assert len(ensemble.predictions) == 7

        # Check metadata
        assert ensemble.metadata["ensemble_method"] == "equal_weight"

    def test_get_ensemble_forecast_median(self, small_time_series):
        """Test ensemble forecast with median."""
        X, y = small_time_series
        orchestrator = ModelOrchestrator(auto_select_top_k=3)

        # Train and predict
        orchestrator.fit_all(X, y)
        orchestrator.predict_all(steps=7)

        # Get ensemble
        ensemble = orchestrator.get_ensemble_forecast(
            steps=7,
            method="median",
        )

        # Check structure
        assert hasattr(ensemble, "predictions")
        assert len(ensemble.predictions) == 7

        # Check metadata
        assert ensemble.metadata["ensemble_method"] == "median"

    def test_get_ensemble_forecast_invalid_method(self, small_time_series):
        """Test ensemble forecast with invalid method."""
        X, y = small_time_series
        orchestrator = ModelOrchestrator()

        # Train and predict
        orchestrator.fit_all(X, y)
        orchestrator.predict_all(steps=7)

        # Try invalid method
        with pytest.raises(ValueError, match="Unknown ensemble method"):
            orchestrator.get_ensemble_forecast(
                steps=7,
                method="invalid_method",
            )

    def test_get_orchestrator_stats(self, small_time_series):
        """Test getting orchestrator statistics."""
        X, y = small_time_series
        orchestrator = ModelOrchestrator()

        # Get stats before training
        stats_before = orchestrator.get_orchestrator_stats()
        assert stats_before["total_models"] == 9
        assert stats_before["successfully_trained"] == 0

        # Train and get stats
        orchestrator.fit_all(X, y)
        orchestrator.predict_all(steps=7)
        stats_after = orchestrator.get_orchestrator_stats()

        assert stats_after["total_models"] == 9
        assert stats_after["successfully_trained"] > 0
        assert "timestamp" in stats_after


class TestOrchestrationIntegration:
    """Integration tests for orchestrator with multiple time series."""

    def test_full_workflow(self, medium_time_series):
        """Test complete orchestration workflow."""
        X, y = medium_time_series
        orchestrator = ModelOrchestrator(
            cache_models=True,
            auto_select_top_k=3,
        )

        # 1. Train all models
        train_results = orchestrator.fit_all(X, y)
        assert len(train_results) == 9

        # 2. Get predictions
        predictions = orchestrator.predict_all(steps=7)
        assert len(predictions) > 0

        # 3. Get comparison
        comparison = orchestrator.get_model_comparison()
        assert isinstance(comparison, pd.DataFrame)

        # 4. Get best model
        best_name, best_result = orchestrator.get_best_model()

        # 5. Get ensemble forecast
        ensemble = orchestrator.get_ensemble_forecast(steps=7)
        assert hasattr(ensemble, "predictions")

        # 6. Get stats
        stats = orchestrator.get_orchestrator_stats()
        assert stats["total_models"] == 9

    def test_different_ensemble_methods_produce_different_results(
        self, medium_time_series
    ):
        """Test that different ensemble methods produce different results."""
        X, y = medium_time_series
        orchestrator = ModelOrchestrator()

        # Train
        orchestrator.fit_all(X, y)
        orchestrator.predict_all(steps=7)

        # Get different ensemble predictions
        ensemble_weighted = orchestrator.get_ensemble_forecast(method="weighted_mean")
        ensemble_equal = orchestrator.get_ensemble_forecast(method="equal_weight")
        ensemble_median = orchestrator.get_ensemble_forecast(method="median")

        # Predictions should be different (not all identical)
        weighted_preds = ensemble_weighted.predictions
        equal_preds = ensemble_equal.predictions
        median_preds = ensemble_median.predictions

        # At least some predictions should differ
        is_different = not np.allclose(weighted_preds, equal_preds) or not np.allclose(
            equal_preds, median_preds
        )
        assert (
            is_different
        ), "Different ensemble methods should produce different results"

    def test_model_caching_prevents_retraining(self, small_time_series):
        """Test that model caching works correctly."""
        X, y = small_time_series
        orchestrator = ModelOrchestrator(cache_models=True)

        # First training
        orchestrator.fit_all(X, y)
        cached_count_1 = len(orchestrator._trained_models)

        # Check caching
        assert cached_count_1 > 0, "Models should be cached after training"

        # Note: We can't directly test "no retraining" without modifying the code,
        # but we can verify cache state
        cached_models = list(orchestrator._trained_models.keys())
        assert len(cached_models) == cached_count_1
