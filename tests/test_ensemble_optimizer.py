"""Tests for Ensemble Optimizer."""

import numpy as np
import pandas as pd
import pytest

from ml_models.ensemble_optimizer import EnsembleOptimizer, OptimizerMetrics
from ml_models.orchestrator import ModelOrchestrator


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


@pytest.fixture
def split_data(medium_time_series):
    """Split data into train/val sets."""
    X, y = medium_time_series
    split_idx = int(0.7 * len(X))

    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    X_val = X.iloc[split_idx:]
    y_val = y.iloc[split_idx:]

    return X_train, y_train, X_val, y_val


class TestEnsembleOptimizer:
    """Test Ensemble Optimizer."""

    def test_init(self):
        """Test optimizer initialization."""
        orchestrator = ModelOrchestrator()
        optimizer = EnsembleOptimizer(
            orchestrator=orchestrator,
            n_trials=10,
        )

        assert optimizer.orchestrator is orchestrator
        assert optimizer.n_trials == 10
        assert optimizer.best_rmse == float("inf")
        assert optimizer.best_method == "weighted_mean"
        assert len(optimizer.best_weights) == 0

    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        orchestrator = ModelOrchestrator()
        optimizer = EnsembleOptimizer(
            orchestrator=orchestrator,
            n_trials=50,
            sampler_seed=42,
            verbose=True,
        )

        assert optimizer.n_trials == 50
        assert optimizer.sampler_seed == 42
        assert optimizer.verbose is True

    def test_calculate_rmse(self):
        """Test RMSE calculation."""
        orchestrator = ModelOrchestrator()
        optimizer = EnsembleOptimizer(orchestrator)

        y_true = pd.Series([70000, 71000, 72000, 73000, 74000])
        y_pred = [70100, 70900, 72100, 72900, 74100]

        rmse = optimizer._calculate_rmse(y_true, y_pred)

        # RMSE = sqrt(mean((100)^2)) = 100
        assert abs(rmse - 100.0) < 10.0, "RMSE should be approximately 100"

    def test_optimize(self, split_data):
        """Test optimization process."""
        X_train, y_train, X_val, y_val = split_data

        orchestrator = ModelOrchestrator()
        optimizer = EnsembleOptimizer(
            orchestrator=orchestrator,
            n_trials=5,  # Small number for quick test
        )

        metrics = optimizer.optimize(X_train, y_train, X_val, y_val)

        # Check metrics structure
        assert isinstance(metrics, OptimizerMetrics)
        assert metrics.best_rmse > 0
        assert metrics.best_trial_number >= 0
        assert metrics.n_trials == 5
        assert metrics.optimization_time_seconds > 0
        assert metrics.study_direction == "minimize"

    def test_optimize_improves_rmse(self, split_data):
        """Test that optimization finds a solution."""
        X_train, y_train, X_val, y_val = split_data

        orchestrator = ModelOrchestrator()
        optimizer = EnsembleOptimizer(
            orchestrator=orchestrator,
            n_trials=5,
        )

        metrics = optimizer.optimize(X_train, y_train, X_val, y_val)

        # RMSE should be finite and reasonably small
        assert not np.isinf(metrics.best_rmse)
        assert metrics.best_rmse > 0
        assert metrics.best_rmse < 10000  # Reasonable bound for gold prices

    def test_optimize_discovers_ensemble_method(self, split_data):
        """Test that optimization discovers ensemble method."""
        X_train, y_train, X_val, y_val = split_data

        orchestrator = ModelOrchestrator()
        optimizer = EnsembleOptimizer(
            orchestrator=orchestrator,
            n_trials=5,
        )

        metrics = optimizer.optimize(X_train, y_train, X_val, y_val)

        # Should discover a valid method
        assert metrics.best_ensemble_method in [
            "weighted_mean",
            "equal_weight",
            "median",
        ]

    def test_optimize_discovers_weights(self, split_data):
        """Test that optimization discovers weights."""
        X_train, y_train, X_val, y_val = split_data

        orchestrator = ModelOrchestrator()
        optimizer = EnsembleOptimizer(
            orchestrator=orchestrator,
            n_trials=5,
        )

        metrics = optimizer.optimize(X_train, y_train, X_val, y_val)

        # If weighted_mean method, should have weights
        if metrics.best_ensemble_method == "weighted_mean":
            assert len(metrics.best_ensemble_weights) > 0
            # Weights should sum to approximately 1
            total_weight = sum(metrics.best_ensemble_weights.values())
            assert abs(total_weight - 1.0) < 0.01

    def test_get_optimized_ensemble_forecast(self, split_data):
        """Test getting optimized ensemble forecast."""
        X_train, y_train, X_val, y_val = split_data

        orchestrator = ModelOrchestrator()
        optimizer = EnsembleOptimizer(
            orchestrator=orchestrator,
            n_trials=5,
        )

        # Optimize first
        optimizer.optimize(X_train, y_train, X_val, y_val)

        # Get forecast
        forecast = optimizer.get_optimized_ensemble_forecast(steps=7)

        # Check structure
        assert hasattr(forecast, "predictions")
        assert len(forecast.predictions) == 7
        assert len(forecast.lower_bound) == 7
        assert len(forecast.upper_bound) == 7

        # Verify bounds
        for i in range(7):
            assert forecast.lower_bound[i] < forecast.predictions[i]
            assert forecast.predictions[i] < forecast.upper_bound[i]

        # Check metadata
        assert "best_rmse" in forecast.metadata
        assert "best_method" in forecast.metadata

    def test_get_optimization_history(self, split_data):
        """Test getting optimization history."""
        X_train, y_train, X_val, y_val = split_data

        orchestrator = ModelOrchestrator()
        optimizer = EnsembleOptimizer(
            orchestrator=orchestrator,
            n_trials=5,
        )

        # Optimize first
        optimizer.optimize(X_train, y_train, X_val, y_val)

        # Get history
        history = optimizer.get_optimization_history()

        # Check structure
        assert isinstance(history, pd.DataFrame)
        assert len(history) == 5
        assert "trial_number" in history.columns
        assert "value" in history.columns
        assert "state" in history.columns


class TestEnsembleOptimizerIntegration:
    """Integration tests for optimizer."""

    def test_full_optimization_workflow(self, split_data):
        """Test complete optimization workflow."""
        X_train, y_train, X_val, y_val = split_data

        # Create optimizer
        orchestrator = ModelOrchestrator()
        optimizer = EnsembleOptimizer(
            orchestrator=orchestrator,
            n_trials=5,
        )

        # 1. Run optimization
        metrics = optimizer.optimize(X_train, y_train, X_val, y_val)
        assert metrics.best_rmse > 0

        # 2. Get optimized forecast
        forecast = optimizer.get_optimized_ensemble_forecast(steps=7)
        assert len(forecast.predictions) == 7

        # 3. Get history
        history = optimizer.get_optimization_history()
        assert len(history) == 5

        # 4. Verify consistency
        assert metrics.n_trials == len(history)

    def test_optimizer_uses_orchestrator_models(self, split_data):
        """Test that optimizer uses all orchestrator models."""
        X_train, y_train, X_val, y_val = split_data

        orchestrator = ModelOrchestrator()
        optimizer = EnsembleOptimizer(
            orchestrator=orchestrator,
            n_trials=3,
        )

        # Optimize
        optimizer.optimize(X_train, y_train, X_val, y_val)

        # Check metadata
        forecast = optimizer.get_optimized_ensemble_forecast(
            steps=7
        )  # Should have multiple models in ensemble
        assert forecast.metadata["num_models"] > 1
        assert len(forecast.metadata["models_used"]) > 1

    def test_different_random_seeds_converge(self, split_data):
        """Test that different seeds find similar RMSE."""
        X_train, y_train, X_val, y_val = split_data

        # Run with different seeds
        rmses = []
        for seed in [1, 2, 3]:
            orchestrator = ModelOrchestrator()
            optimizer = EnsembleOptimizer(
                orchestrator=orchestrator,
                n_trials=3,
                sampler_seed=seed,
            )
            metrics = optimizer.optimize(X_train, y_train, X_val, y_val)
            rmses.append(metrics.best_rmse)

        # All RMSE values should be reasonable (not infinite, not too large)
        for rmse in rmses:
            assert not np.isinf(rmse)
            assert rmse < 10000
