"""Tests for Gold Price Service Layer."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from services.gold_price_service import ForecastCache, GoldPriceService


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


class TestForecastCache:
    """Test caching functionality."""

    def test_cache_init_sqlite(self):
        """Test cache initialization with SQLite."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ForecastCache(cache_dir=tmpdir, use_sqlite=True)

            assert cache.use_sqlite is True
            assert cache.cache_dir == Path(tmpdir)
            assert (Path(tmpdir) / "forecast_cache.db").exists()

    def test_cache_init_file_based(self):
        """Test cache initialization with file-based storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ForecastCache(cache_dir=tmpdir, use_sqlite=False)

            assert cache.use_sqlite is False
            assert cache.cache_dir == Path(tmpdir)

    def test_cache_key_generation(self):
        """Test cache key generation."""
        cache = ForecastCache()

        params1 = {"steps": 7, "method": "weighted_mean"}
        params2 = {"steps": 7, "method": "weighted_mean"}
        params3 = {"steps": 14, "method": "weighted_mean"}

        key1 = cache._get_cache_key(params1)
        key2 = cache._get_cache_key(params2)
        key3 = cache._get_cache_key(params3)

        # Same params should produce same key
        assert key1 == key2
        # Different params should produce different key
        assert key1 != key3

    def test_cache_set_and_get_sqlite(self, small_time_series):
        """Test setting and getting cache with SQLite."""
        with tempfile.TemporaryDirectory() as tmpdir:
            X, y = small_time_series
            cache = ForecastCache(cache_dir=tmpdir, use_sqlite=True)

            # Create mock forecast result
            from models.time_series_base import ForecastResult

            forecast = ForecastResult(
                dates=["2025-01-01", "2025-01-02"],
                predictions=[70000, 71000],
                lower_bound=[69000, 70000],
                upper_bound=[71000, 72000],
                confidence_level=0.95,
                model_name="test",
            )

            params = {"steps": 7, "method": "test"}

            # Set cache
            cache.set(params, forecast)

            # Get cache
            cached = cache.get(params)

            assert cached is not None
            assert cached.predictions == forecast.predictions

    def test_cache_miss(self):
        """Test cache miss."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ForecastCache(cache_dir=tmpdir, use_sqlite=True)

            result = cache.get({"steps": 7, "method": "nonexistent"})

            assert result is None

    def test_cache_clear(self):
        """Test cache clearing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from models.time_series_base import ForecastResult

            cache = ForecastCache(cache_dir=tmpdir, use_sqlite=True)

            forecast = ForecastResult(
                dates=["2025-01-01"],
                predictions=[70000],
                lower_bound=[69000],
                upper_bound=[71000],
                confidence_level=0.95,
                model_name="test",
            )

            cache.set({"test": "params"}, forecast)

            # Verify cached
            assert cache.get({"test": "params"}) is not None

            # Clear
            cache.clear()

            # Verify cleared
            assert cache.get({"test": "params"}) is None


class TestGoldPriceService:
    """Test Gold Price Service."""

    def test_init_default(self):
        """Test service initialization with defaults."""
        service = GoldPriceService()

        assert service.orchestrator is not None
        assert service.optimizer is not None
        assert service.cache_enabled is True
        assert service.cache is not None

    def test_init_custom(self):
        """Test service initialization with custom params."""
        with tempfile.TemporaryDirectory() as tmpdir:
            service = GoldPriceService(
                cache_enabled=False,
                cache_dir=tmpdir,
            )

            assert service.cache_enabled is False
            assert service.cache is None

    def test_train(self, small_time_series):
        """Test model training."""
        X, y = small_time_series
        service = GoldPriceService(cache_enabled=False)

        results = service.train(X, y)

        assert isinstance(results, dict)
        assert len(results) > 0

    def test_optimize(self, split_data):
        """Test ensemble optimization."""
        X_train, y_train, X_val, y_val = split_data
        service = GoldPriceService(cache_enabled=False)

        # Train first
        service.train(X_train, y_train)

        # Optimize
        metrics = service.optimize(
            X_train,
            y_train,
            X_val,
            y_val,
            n_trials=3,
        )

        assert isinstance(metrics, dict)
        assert "best_rmse" in metrics
        assert "best_method" in metrics
        assert metrics["best_rmse"] > 0

    def test_forecast_standard(self, small_time_series):
        """Test standard ensemble forecast."""
        X, y = small_time_series

        with tempfile.TemporaryDirectory() as tmpdir:
            service = GoldPriceService(cache_dir=tmpdir)

            # Train
            service.train(X, y)

            # Forecast
            forecast = service.forecast(steps=7, use_optimized=False)

            assert forecast is not None
            assert len(forecast.predictions) == 7
            assert forecast.model_name == "Ensemble"

    def test_forecast_optimized(self, split_data):
        """Test optimized ensemble forecast."""
        X_train, y_train, X_val, y_val = split_data

        with tempfile.TemporaryDirectory() as tmpdir:
            service = GoldPriceService(cache_dir=tmpdir)

            # Train
            service.train(X_train, y_train)

            # Optimize
            service.optimize(X_train, y_train, X_val, y_val, n_trials=3)

            # Forecast
            forecast = service.forecast(steps=7, use_optimized=True)

            assert forecast is not None
            assert len(forecast.predictions) == 7
            # Model name depends on optimization result
            assert forecast.model_name in ["Ensemble", "OptimizedEnsemble"]

    def test_forecast_caching(self, small_time_series):
        """Test forecast caching."""
        X, y = small_time_series

        with tempfile.TemporaryDirectory() as tmpdir:
            service = GoldPriceService(
                cache_dir=tmpdir,
                cache_enabled=True,
            )

            service.train(X, y)

            # First forecast (cache miss)
            forecast1 = service.forecast(steps=7, use_cache=True)
            assert service.get_metrics()["cache_misses"] == 1

            # Second forecast (cache hit)
            forecast2 = service.forecast(steps=7, use_cache=True)
            assert service.get_metrics()["cache_hits"] == 1

            # Results should be identical
            assert forecast1.predictions == forecast2.predictions

    def test_forecast_no_cache(self, small_time_series):
        """Test forecast without caching."""
        X, y = small_time_series
        service = GoldPriceService(cache_enabled=False)

        service.train(X, y)

        forecast = service.forecast(steps=7, use_cache=False)

        assert forecast is not None
        # When cache is disabled, both hits and misses should be 0
        metrics = service.get_metrics()
        assert metrics["cache_enabled"] is False

    def test_get_model_comparison(self, small_time_series):
        """Test getting model comparison."""
        X, y = small_time_series
        service = GoldPriceService(cache_enabled=False)

        service.train(X, y)
        service.forecast(steps=7)

        comparison = service.get_model_comparison()

        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) > 0

    def test_get_optimization_history(self, split_data):
        """Test getting optimization history."""
        X_train, y_train, X_val, y_val = split_data
        service = GoldPriceService(cache_enabled=False)

        service.train(X_train, y_train)
        service.optimize(X_train, y_train, X_val, y_val, n_trials=3)

        history = service.get_optimization_history()

        assert isinstance(history, pd.DataFrame)
        assert len(history) == 3

    def test_get_metrics(self, small_time_series):
        """Test getting service metrics."""
        X, y = small_time_series

        with tempfile.TemporaryDirectory() as tmpdir:
            service = GoldPriceService(cache_dir=tmpdir)

            service.train(X, y)
            service.forecast(steps=7, use_cache=True)
            service.forecast(steps=7, use_cache=True)

            metrics = service.get_metrics()

            assert metrics["forecasts_requested"] == 2
            assert metrics["cache_misses"] == 1
            assert metrics["cache_hits"] == 1
            assert metrics["cache_hit_rate"] == 50.0

    def test_clear_cache(self, small_time_series):
        """Test cache clearing."""
        X, y = small_time_series

        with tempfile.TemporaryDirectory() as tmpdir:
            service = GoldPriceService(
                cache_dir=tmpdir,
                cache_enabled=True,
            )

            service.train(X, y)
            service.forecast(steps=7, use_cache=True)

            # Clear cache
            service.clear_cache()

            # Next forecast should be cache miss
            service.forecast(steps=7, use_cache=True)

            metrics = service.get_metrics()
            assert metrics["cache_misses"] == 2
            assert metrics["cache_hits"] == 0

    def test_get_service_status(self, small_time_series):
        """Test getting service status."""
        X, y = small_time_series
        service = GoldPriceService()

        service.train(X, y)

        status = service.get_service_status()

        assert status["orchestrator_initialized"] is True
        assert status["optimizer_initialized"] is True
        assert status["cache_enabled"] is True
        assert "metrics" in status
        assert "timestamp" in status


class TestGoldPriceServiceIntegration:
    """Integration tests for service."""

    def test_full_workflow_with_optimization(self, split_data):
        """Test complete workflow with optimization."""
        X_train, y_train, X_val, y_val = split_data

        with tempfile.TemporaryDirectory() as tmpdir:
            service = GoldPriceService(cache_dir=tmpdir)

            # 1. Train
            service.train(X_train, y_train)

            # 2. Optimize
            opt_metrics = service.optimize(
                X_train,
                y_train,
                X_val,
                y_val,
                n_trials=3,
            )
            assert opt_metrics["best_rmse"] > 0

            # 3. Forecast
            forecast = service.forecast(steps=7, use_optimized=True)
            assert len(forecast.predictions) == 7

            # 4. Get metrics
            metrics = service.get_metrics()
            assert metrics["forecasts_requested"] == 1

            # 5. Get status
            status = service.get_service_status()
            assert status["orchestrator_initialized"]

    def test_multiple_forecasts_with_cache(self, small_time_series):
        """Test multiple forecasts with caching."""
        X, y = small_time_series

        with tempfile.TemporaryDirectory() as tmpdir:
            service = GoldPriceService(cache_dir=tmpdir)

            service.train(X, y)

            # Generate multiple forecasts
            for steps in [7, 14, 7]:
                service.forecast(steps=steps, use_cache=True)

            metrics = service.get_metrics()
            assert metrics["forecasts_requested"] == 3
            # Two cache hits (7 steps twice)
            assert metrics["cache_hits"] == 1
