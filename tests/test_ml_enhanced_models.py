"""Tests for enhanced ML models (LightGBM, CatBoost, SVR)."""

import numpy as np
import pandas as pd
import pytest

from ml_models.ml_enhanced import CatBoostModel, LightGBMModel, SVRModel


# Fixtures
@pytest.fixture
def medium_time_series():  # type: ignore[misc]
    """Create medium time series (180 days, 6 months)."""
    dates = pd.date_range(start="2024-01-01", periods=180, freq="D")
    # Trend + seasonal + noise
    t = np.arange(180)
    trend = 100 + 0.05 * t
    seasonal = 5 * np.sin(2 * np.pi * t / 7)
    noise = np.random.normal(0, 1, 180)
    values = trend + seasonal + noise

    df = pd.DataFrame({"date": dates})
    series = pd.Series(values, index=dates, name="price")
    return df, series


@pytest.fixture
def small_time_series():  # type: ignore[misc]
    """Create small time series (90 days)."""
    dates = pd.date_range(start="2024-01-01", periods=90, freq="D")
    trend = np.linspace(100, 110, 90)
    noise = np.random.normal(0, 1, 90)
    values = trend + noise

    df = pd.DataFrame({"date": dates})
    series = pd.Series(values, index=dates, name="price")
    return df, series


# LightGBM Tests
class TestLightGBMModel:
    """Test suite for LightGBMModel."""

    def test_init(self):
        """Test LightGBM initialization."""
        model = LightGBMModel()

        assert model.model_type == "ml_enhanced"
        assert model.version == "1.0.0"
        assert model.n_estimators == 50
        assert model.max_depth == 3
        assert model.learning_rate == 0.05

    def test_fit(self, medium_time_series):
        """Test LightGBM fitting."""
        df, series = medium_time_series
        model = LightGBMModel()

        fitted = model.fit(df, series)

        assert fitted == model
        assert model._fitted_model is not None
        assert len(model._feature_names) > 0
        assert "n_features" in model._diagnostics

    def test_predict(self, medium_time_series):
        """Test LightGBM prediction."""
        df, series = medium_time_series
        model = LightGBMModel()
        model.fit(df, series)

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
        df, series = medium_time_series
        model = LightGBMModel()
        model.fit(df, series)

        metadata = model.get_metadata()

        assert metadata.model_type == "ml_enhanced"
        assert metadata.training_samples > 0
        assert "n_features" in metadata.metrics


# CatBoost Tests
class TestCatBoostModel:
    """Test suite for CatBoostModel."""

    def test_init(self):
        """Test CatBoost initialization."""
        model = CatBoostModel()

        assert model.model_type == "ml_enhanced"
        assert model.version == "1.0.0"
        assert model.iterations == 50
        assert model.depth == 4
        assert model.learning_rate == 0.03

    def test_fit(self, medium_time_series):
        """Test CatBoost fitting."""
        df, series = medium_time_series
        model = CatBoostModel()

        fitted = model.fit(df, series)

        assert fitted == model
        assert model._fitted_model is not None
        assert len(model._feature_names) > 0
        assert "n_features" in model._diagnostics
        assert "n_categorical_features" in model._diagnostics

    def test_predict(self, medium_time_series):
        """Test CatBoost prediction."""
        df, series = medium_time_series
        model = CatBoostModel()
        model.fit(df, series)

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

    def test_categorical_features_detected(self, medium_time_series):
        """Test that categorical features are properly detected."""
        df, series = medium_time_series
        model = CatBoostModel()
        model.fit(df, series)

        # CatBoost with all numeric features (no categorical features detected)
        assert isinstance(model._categorical_features, list)
        assert "n_categorical_features" in model._diagnostics

    def test_get_metadata(self, medium_time_series):
        """Test metadata extraction."""
        df, series = medium_time_series
        model = CatBoostModel()
        model.fit(df, series)

        metadata = model.get_metadata()

        assert metadata.model_type == "ml_enhanced"
        assert metadata.training_samples > 0
        assert "n_features" in metadata.metrics


# SVR Tests
class TestSVRModel:
    """Test suite for SVRModel."""

    def test_init_with_defaults(self):
        """Test SVR initialization."""
        model = SVRModel()

        assert model.model_type == "ml_enhanced"
        assert model.version == "1.0.0"
        assert model.kernel == "rbf"
        assert model.C == 1.0
        assert model.epsilon == 0.01

    def test_init_with_custom_kernel(self):
        """Test SVR with custom kernel."""
        model = SVRModel(kernel="linear")

        assert model.kernel == "linear"

    def test_fit(self, medium_time_series):
        """Test SVR fitting."""
        df, series = medium_time_series
        model = SVRModel()

        fitted = model.fit(df, series)

        assert fitted == model
        assert model._fitted_model is not None
        assert len(model._feature_names) > 0
        assert model._y_mean is not None
        assert model._y_std is not None

    def test_predict(self, medium_time_series):
        """Test SVR prediction."""
        df, series = medium_time_series
        model = SVRModel()
        model.fit(df, series)

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
        df, series = medium_time_series
        model = SVRModel()
        model.fit(df, series)

        metadata = model.get_metadata()

        assert metadata.model_type == "ml_enhanced"
        assert metadata.training_samples > 0
        assert "n_features" in metadata.metrics
        assert "kernel" in metadata.metrics


# Integration Tests
class TestMLModelsIntegration:
    """Integration tests for ML models."""

    def test_all_models_handle_same_data(self, medium_time_series):
        """Test that all three models can fit the same data."""
        df, series = medium_time_series

        lgb = LightGBMModel()
        cat = CatBoostModel()
        svr = SVRModel()

        # All should fit without errors
        lgb.fit(df, series)
        cat.fit(df, series)
        svr.fit(df, series)

        # All should generate predictions
        lgb_result = lgb.predict(steps=5)
        cat_result = cat.predict(steps=5)
        svr_result = svr.predict(steps=5)

        assert len(lgb_result.predictions) == 5
        assert len(cat_result.predictions) == 5
        assert len(svr_result.predictions) == 5

    def test_models_produce_different_forecasts(self, medium_time_series):
        """Test that models produce reasonably different forecasts."""
        df, series = medium_time_series

        lgb = LightGBMModel(n_estimators=10)
        cat = CatBoostModel(iterations=10)
        svr = SVRModel()

        lgb.fit(df, series)
        cat.fit(df, series)
        svr.fit(df, series)

        lgb_result = lgb.predict(steps=1)
        cat_result = cat.predict(steps=1)
        svr_result = svr.predict(steps=1)

        # Models should produce different (but reasonable) predictions
        predictions = [
            lgb_result.predictions[0],
            cat_result.predictions[0],
            svr_result.predictions[0],
        ]

        # Check they're not all identical
        assert not (predictions[0] == predictions[1] == predictions[2])

        # Check they're in reasonable range (within 50% of mean)
        mean_price = series.mean()
        assert all(0.5 * mean_price < p < 1.5 * mean_price for p in predictions)

    def test_models_with_small_dataset(self, small_time_series):
        """Test that models handle small datasets gracefully."""
        df, series = small_time_series

        lgb = LightGBMModel(n_estimators=10)
        cat = CatBoostModel(iterations=10)
        svr = SVRModel()

        # All should fit without errors despite small dataset
        lgb.fit(df, series)
        cat.fit(df, series)
        svr.fit(df, series)

        # All should predict
        lgb_result = lgb.predict(steps=3)
        cat_result = cat.predict(steps=3)
        svr_result = svr.predict(steps=3)

        assert len(lgb_result.predictions) == 3
        assert len(cat_result.predictions) == 3
        assert len(svr_result.predictions) == 3
