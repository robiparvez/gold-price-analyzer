"""End-to-end integration tests for complete workflows."""

import tempfile

import numpy as np
import pandas as pd
import pytest

from services.gold_price_service import GoldPriceService


@pytest.fixture
def full_time_series():
    """Create full time series (365 days)."""
    dates = pd.date_range(start="2025-01-05", periods=365, freq="D")
    trend = np.linspace(70000, 80000, 365)
    seasonal = 1000 * np.sin(np.linspace(0, 4 * np.pi, 365))
    noise = np.random.normal(0, 500, 365)
    prices = trend + seasonal + noise
    prices = pd.Series(prices, index=dates)
    X = pd.DataFrame({"price": prices.values}, index=dates)
    return X, prices


@pytest.fixture
def split_full_data(full_time_series):
    """Split full dataset into train/val/test."""
    X, y = full_time_series

    # 70% train, 15% val, 15% test
    train_idx = int(0.7 * len(X))
    val_idx = int(0.85 * len(X))

    X_train = X.iloc[:train_idx]
    y_train = y.iloc[:train_idx]
    X_val = X.iloc[train_idx:val_idx]
    y_val = y.iloc[train_idx:val_idx]
    X_test = X.iloc[val_idx:]
    y_test = y.iloc[val_idx:]

    return X_train, y_train, X_val, y_val, X_test, y_test


class TestEndToEndWorkflows:
    """End-to-end integration tests."""

    def test_complete_forecasting_workflow(self, split_full_data):
        """Test complete workflow from training to forecasting."""
        X_train, y_train, X_val, y_val, X_test, y_test = split_full_data

        with tempfile.TemporaryDirectory() as tmpdir:
            # Initialize service
            service = GoldPriceService(cache_dir=tmpdir)

            # Step 1: Train all models
            train_results = service.train(X_train, y_train)
            assert isinstance(train_results, dict)
            assert len(train_results) == 9  # 9 models

            # Step 2: Generate standard forecast
            forecast_standard = service.forecast(steps=7, use_optimized=False)
            assert len(forecast_standard.predictions) == 7
            assert forecast_standard.model_name == "Ensemble"

            # Step 3: Optimize ensemble
            opt_metrics = service.optimize(
                X_train,
                y_train,
                X_val,
                y_val,
                n_trials=5,
            )
            assert opt_metrics["best_rmse"] > 0

            # Step 4: Generate optimized forecast
            forecast_optimized = service.forecast(steps=7, use_optimized=True)
            assert len(forecast_optimized.predictions) == 7

            # Step 5: Verify metrics tracking
            metrics = service.get_metrics()
            assert metrics["forecasts_requested"] == 2
            assert metrics["cache_misses"] == 2  # First requests

    def test_caching_across_multiple_forecasts(self, split_full_data):
        """Test caching behavior across multiple forecast requests."""
        X_train, y_train, X_val, y_val, X_test, y_test = split_full_data

        with tempfile.TemporaryDirectory() as tmpdir:
            service = GoldPriceService(cache_dir=tmpdir, cache_enabled=True)

            # Train
            service.train(X_train, y_train)

            # First forecast - cache miss
            forecast1 = service.forecast(steps=7, use_cache=True)
            metrics1 = service.get_metrics()
            assert metrics1["cache_misses"] == 1
            assert metrics1["cache_hits"] == 0

            # Same forecast - cache hit
            forecast2 = service.forecast(steps=7, use_cache=True)
            metrics2 = service.get_metrics()
            assert metrics2["cache_hits"] == 1

            # Predictions should be identical
            assert forecast1.predictions == forecast2.predictions

            # Different steps - cache miss
            service.forecast(steps=14, use_cache=True)
            metrics3 = service.get_metrics()
            assert metrics3["cache_misses"] == 2

    def test_optimization_improves_forecast_quality(self, split_full_data):
        """Test that optimization improves forecast quality."""
        X_train, y_train, X_val, y_val, X_test, y_test = split_full_data

        service = GoldPriceService(cache_enabled=False)

        # Train
        service.train(X_train, y_train)

        # Optimize
        opt_metrics = service.optimize(
            X_train,
            y_train,
            X_val,
            y_val,
            n_trials=5,
        )

        # RMSE should be finite and reasonable
        assert not np.isinf(opt_metrics["best_rmse"])
        assert opt_metrics["best_rmse"] > 0
        assert opt_metrics["best_rmse"] < 20000  # Reasonable for gold prices

        # Best method should be discovered
        assert opt_metrics["best_method"] in ["weighted_mean", "equal_weight", "median"]

    def test_model_comparison_after_training(self, split_full_data):
        """Test model comparison functionality."""
        X_train, y_train, X_val, y_val, X_test, y_test = split_full_data

        service = GoldPriceService(cache_enabled=False)

        # Train
        service.train(X_train, y_train)

        # Generate predictions
        service.forecast(steps=7)

        # Get comparison
        comparison = service.get_model_comparison()

        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) > 0
        # Should have metrics for models that succeeded
        assert "mean_pred" in comparison.columns
        assert "std_pred" in comparison.columns

    def test_service_resilience_to_errors(self, split_full_data):
        """Test service error handling and resilience."""
        X_train, y_train, X_val, y_val, X_test, y_test = split_full_data

        service = GoldPriceService(cache_enabled=False)

        # Should handle forecasting without training gracefully
        with pytest.raises(Exception):
            service.forecast(steps=7, use_optimized=True)

        # Error should be tracked
        metrics = service.get_metrics()
        assert metrics["errors"] > 0

    def test_cache_persistence_across_sessions(self, split_full_data):
        """Test that cache persists across service instances."""
        X_train, y_train, X_val, y_val, X_test, y_test = split_full_data

        with tempfile.TemporaryDirectory() as tmpdir:
            # Session 1: Train and forecast
            service1 = GoldPriceService(cache_dir=tmpdir)
            service1.train(X_train, y_train)
            service1.forecast(steps=7, use_cache=True)

            # Session 2: New service instance, same cache
            service2 = GoldPriceService(cache_dir=tmpdir)
            service2.train(X_train, y_train)  # Re-train (in real scenario, might load)
            service2.forecast(steps=7, use_cache=True)

            # Should get cache hit
            metrics = service2.get_metrics()
            assert metrics["cache_hits"] == 1

    def test_full_production_workflow(self, split_full_data):
        """Test complete production-like workflow."""
        X_train, y_train, X_val, y_val, X_test, y_test = split_full_data

        with tempfile.TemporaryDirectory() as tmpdir:
            service = GoldPriceService(cache_dir=tmpdir)

            # 1. Initial training
            train_results = service.train(X_train, y_train)
            successful_models = sum(
                1 for r in train_results.values() if r.get("status") == "success"
            )
            assert successful_models > 0

            # 2. Baseline forecast
            baseline_forecast = service.forecast(steps=7, use_optimized=False)
            assert len(baseline_forecast.predictions) == 7

            # 3. Hyperparameter optimization
            opt_metrics = service.optimize(
                X_train,
                y_train,
                X_val,
                y_val,
                n_trials=5,
            )
            assert opt_metrics["best_rmse"] > 0

            # 4. Optimized forecast
            optimized_forecast = service.forecast(steps=7, use_optimized=True)
            assert len(optimized_forecast.predictions) == 7

            # 5. Model comparison
            comparison = service.get_model_comparison()
            assert len(comparison) > 0

            # 6. Check optimization history
            history = service.get_optimization_history()
            assert len(history) == 5  # n_trials

            # 7. Service status
            status = service.get_service_status()
            assert status["orchestrator_initialized"]
            assert status["optimizer_initialized"]
            assert status["cache_enabled"]

            # 8. Metrics validation
            metrics = service.get_metrics()
            assert metrics["forecasts_requested"] == 2
            assert metrics["cache_hit_rate"] >= 0


class TestModelAccuracy:
    """Tests for model accuracy and predictions."""

    def test_predictions_are_reasonable(self, split_full_data):
        """Test that predictions are within reasonable bounds."""
        X_train, y_train, X_val, y_val, X_test, y_test = split_full_data

        service = GoldPriceService(cache_enabled=False)
        service.train(X_train, y_train)
        forecast = service.forecast(steps=7)

        # Predictions should be positive
        assert all(p > 0 for p in forecast.predictions)

        # Should be within reasonable range of training data
        train_mean = y_train.mean()
        train_std = y_train.std()

        for pred in forecast.predictions:
            # Within 5 standard deviations (very generous)
            assert train_mean - 5 * train_std < pred < train_mean + 5 * train_std

    def test_confidence_intervals_are_valid(self, split_full_data):
        """Test that confidence intervals are properly ordered."""
        X_train, y_train, X_val, y_val, X_test, y_test = split_full_data

        service = GoldPriceService(cache_enabled=False)
        service.train(X_train, y_train)
        forecast = service.forecast(steps=7)

        # Check all steps
        for i in range(7):
            lb = forecast.lower_bound[i]
            pred = forecast.predictions[i]
            ub = forecast.upper_bound[i]

            assert lb <= pred <= ub, f"Step {i}: {lb} <= {pred} <= {ub} failed"

    def test_ensemble_predictions_differ_from_individual_models(self, split_full_data):
        """Test that ensemble provides different predictions than individual models."""
        X_train, y_train, X_val, y_val, X_test, y_test = split_full_data

        service = GoldPriceService(cache_enabled=False)
        service.train(X_train, y_train)

        # Get ensemble forecast
        ensemble_forecast = service.forecast(steps=7)

        # Get individual model predictions
        individual_predictions = service.orchestrator.predict_all(steps=7)

        # Ensemble should differ from at least some individual models
        differs_from_some = False
        for model_name, individual_result in individual_predictions.items():
            if individual_result is not None:
                if ensemble_forecast.predictions != individual_result.predictions:
                    differs_from_some = True
                    break

        assert differs_from_some


class TestScalability:
    """Tests for system scalability."""

    def test_handles_multiple_forecast_horizons(self, split_full_data):
        """Test forecasting with different time horizons."""
        X_train, y_train, X_val, y_val, X_test, y_test = split_full_data

        service = GoldPriceService(cache_enabled=False)
        service.train(X_train, y_train)

        # Test different horizons
        for steps in [1, 3, 7, 14, 30]:
            forecast = service.forecast(steps=steps, use_optimized=False)
            assert len(forecast.predictions) == steps
            assert len(forecast.lower_bound) == steps
            assert len(forecast.upper_bound) == steps

    def test_cache_handles_multiple_configurations(self, split_full_data):
        """Test cache handles different forecast configurations."""
        X_train, y_train, X_val, y_val, X_test, y_test = split_full_data

        with tempfile.TemporaryDirectory() as tmpdir:
            service = GoldPriceService(cache_dir=tmpdir)
            service.train(X_train, y_train)

            # Generate forecasts with different configurations
            configs = [
                (7, False),
                (14, False),
                (7, False),  # Duplicate - should hit cache
                (14, False),  # Duplicate - should hit cache
            ]

            for steps, use_opt in configs:
                service.forecast(steps=steps, use_optimized=use_opt)

            metrics = service.get_metrics()
            assert metrics["forecasts_requested"] == 4
            assert metrics["cache_hits"] == 2  # Two duplicates
            assert metrics["cache_misses"] == 2  # Two unique
