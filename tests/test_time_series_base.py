"""Tests for time-series base model classes and model registry."""

import tempfile
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from models.model_registry import ModelRegistry
from models.time_series_base import (
    BaseTimeSeriesModel,
    EnsembleModel,
    ForecastResult,
    ModelMetadata,
)


class MockModel(BaseTimeSeriesModel):
    """Mock implementation of BaseTimeSeriesModel for testing."""

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
        validation_data: tuple[pd.DataFrame, pd.Series] | None = None,
    ) -> "MockModel":
        """Fit the mock model."""
        self.validate_data(X, min_samples=10)
        self.is_fitted = True
        self.metadata = self._create_metadata(
            model_name="MockModel",
            model_type="test",
            X=X,
            hyperparameters={"param1": 1.0},
            metrics={"mae": 10.5, "rmse": 15.2},
        )
        return self

    def predict(
        self, horizon: int = 7, X_future: pd.DataFrame | None = None
    ) -> ForecastResult:
        """Generate mock predictions."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")

        dates = [
            (datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d")
            for i in range(1, horizon + 1)
        ]
        predictions = [100.0 + i * 2 for i in range(horizon)]
        lower_bound = [p - 5 for p in predictions]
        upper_bound = [p + 5 for p in predictions]

        return ForecastResult(
            dates=dates,
            predictions=predictions,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            model_name="MockModel",
        )

    def get_metadata(self) -> ModelMetadata:
        """Get model metadata."""
        if self.metadata is None:
            raise RuntimeError("Model not fitted")
        return self.metadata


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Create sample time-series data for testing."""
    dates = pd.date_range(start="2025-01-01", end="2025-06-30", freq="D")
    prices = 100 + np.cumsum(np.random.randn(len(dates)) * 2)

    df = pd.DataFrame(
        {"date": dates, "price_bdt_per_gram": prices, "purity": "22K"}
    )
    df.set_index("date", inplace=True)
    return df


@pytest.fixture
def temp_registry():  # type: ignore[misc]
    """Create a temporary model registry for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = ModelRegistry(base_path=tmpdir)
        yield registry


class TestModelMetadata:
    """Tests for ModelMetadata dataclass."""

    def test_creation(self) -> None:
        """Test creating ModelMetadata."""
        metadata = ModelMetadata(
            model_name="TestModel",
            model_type="classical",
            purity="22K",
            training_samples=100,
            training_date_range=("2025-01-01", "2025-06-30"),
            trained_at="2026-01-05T10:00:00",
            hyperparameters={"p": 1, "d": 1, "q": 1},
            metrics={"mae": 5.5, "rmse": 7.2},
        )

        assert metadata.model_name == "TestModel"
        assert metadata.purity == "22K"
        assert metadata.training_samples == 100
        assert metadata.metrics["mae"] == 5.5

    def test_to_dict(self) -> None:
        """Test converting metadata to dictionary."""
        metadata = ModelMetadata(
            model_name="TestModel",
            model_type="classical",
            purity="22K",
            training_samples=100,
            training_date_range=("2025-01-01", "2025-06-30"),
            trained_at="2026-01-05T10:00:00",
        )

        data = metadata.to_dict()
        assert isinstance(data, dict)
        assert data["model_name"] == "TestModel"
        assert data["purity"] == "22K"

    def test_from_dict(self) -> None:
        """Test creating metadata from dictionary."""
        data = {
            "model_name": "TestModel",
            "model_type": "classical",
            "purity": "22K",
            "training_samples": 100,
            "training_date_range": ("2025-01-01", "2025-06-30"),
            "trained_at": "2026-01-05T10:00:00",
            "hyperparameters": {},
            "metrics": {},
            "feature_names": [],
            "data_statistics": {},
            "model_path": None,
            "version": "1.0.0",
        }

        metadata = ModelMetadata.from_dict(data)
        assert metadata.model_name == "TestModel"
        assert metadata.training_samples == 100


class TestForecastResult:
    """Tests for ForecastResult dataclass."""

    def test_creation(self) -> None:
        """Test creating ForecastResult."""
        result = ForecastResult(
            dates=["2026-01-06", "2026-01-07"],
            predictions=[100.0, 102.0],
            lower_bound=[95.0, 97.0],
            upper_bound=[105.0, 107.0],
            model_name="TestModel",
        )

        assert len(result.dates) == 2
        assert result.predictions[0] == 100.0
        assert result.model_name == "TestModel"

    def test_to_dict(self) -> None:
        """Test converting result to dictionary."""
        result = ForecastResult(
            dates=["2026-01-06"],
            predictions=[100.0],
            lower_bound=[95.0],
            upper_bound=[105.0],
        )

        data = result.to_dict()
        assert isinstance(data, dict)
        assert "dates" in data
        assert "predictions" in data


class TestBaseTimeSeriesModel:
    """Tests for BaseTimeSeriesModel."""

    def test_initialization(self) -> None:
        """Test model initialization."""
        model = MockModel(purity="22K")
        assert model.purity == "22K"
        assert not model.is_fitted
        assert model.metadata is None

    def test_fit_and_predict(self, sample_data: pd.DataFrame) -> None:
        """Test fitting and predicting."""
        model = MockModel(purity="22K")
        model.fit(sample_data)

        assert model.is_fitted
        assert model.metadata is not None

        result = model.predict(horizon=7)
        assert len(result.dates) == 7
        assert len(result.predictions) == 7

    def test_validate_data_insufficient(self) -> None:
        """Test data validation with insufficient samples."""
        model = MockModel()
        small_data = pd.DataFrame(
            {
                "price_bdt_per_gram": [100.0, 101.0],
                "date": pd.date_range(start="2025-01-01", periods=2),
            }
        ).set_index("date")

        with pytest.raises(ValueError, match="Insufficient data"):
            model.validate_data(small_data, min_samples=60)

    def test_validate_data_no_datetime_index(self) -> None:
        """Test validation fails without DatetimeIndex."""
        model = MockModel()
        # Create data with enough samples but no DatetimeIndex
        bad_data = pd.DataFrame({"price_bdt_per_gram": [100.0 + i for i in range(70)]})

        with pytest.raises(ValueError, match="DatetimeIndex"):
            model.validate_data(bad_data, min_samples=60)

    def test_predict_before_fit(self) -> None:
        """Test predicting before fitting raises error."""
        model = MockModel()

        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict()

    def test_calculate_metrics(self, sample_data: pd.DataFrame) -> None:
        """Test metric calculation."""
        model = MockModel()
        model.fit(sample_data)

        y_true = np.array([100, 102, 104, 106])
        y_pred = np.array([101, 103, 103, 107])

        metrics = model.calculate_metrics(y_true, y_pred)

        assert "mae" in metrics
        assert "rmse" in metrics
        assert "mape" in metrics
        assert "r2" in metrics
        assert metrics["mae"] >= 0
        assert metrics["rmse"] >= 0


class TestEnsembleModel:
    """Tests for EnsembleModel."""

    def test_initialization(self) -> None:
        """Test ensemble initialization."""
        model1 = MockModel(purity="22K")
        model2 = MockModel(purity="22K")

        ensemble = EnsembleModel(models=[model1, model2])
        assert len(ensemble.models) == 2
        assert len(ensemble.weights) == 2
        assert sum(ensemble.weights) == pytest.approx(1.0)

    def test_custom_weights(self) -> None:
        """Test ensemble with custom weights."""
        model1 = MockModel(purity="22K")
        model2 = MockModel(purity="22K")

        ensemble = EnsembleModel(models=[model1, model2], weights=[0.6, 0.4])
        assert ensemble.weights[0] == pytest.approx(0.6)
        assert ensemble.weights[1] == pytest.approx(0.4)

    def test_fit_and_predict(self, sample_data: pd.DataFrame) -> None:
        """Test ensemble fitting and prediction."""
        model1 = MockModel(purity="22K")
        model2 = MockModel(purity="22K")

        ensemble = EnsembleModel(models=[model1, model2])
        ensemble.fit(sample_data)

        assert ensemble.is_fitted
        assert model1.is_fitted
        assert model2.is_fitted

        result = ensemble.predict(horizon=7)
        assert len(result.predictions) == 7
        assert "Ensemble" in result.model_name


class TestModelRegistry:
    """Tests for ModelRegistry."""

    def test_initialization(self, temp_registry: ModelRegistry) -> None:
        """Test registry initialization."""
        assert temp_registry.base_path.exists()
        assert temp_registry.classical_dir.exists()
        assert temp_registry.ml_enhanced_dir.exists()

    def test_register_and_get_model(
        self, temp_registry: ModelRegistry, sample_data: pd.DataFrame
    ) -> None:
        """Test registering and retrieving model metadata."""
        model = MockModel(purity="22K")
        model.fit(sample_data)

        metadata = model.get_metadata()
        model_path = str(temp_registry.get_model_path("test", "MockModel", "22K"))

        temp_registry.register_model(metadata, model_path)

        retrieved = temp_registry.get_model_metadata("MockModel", "22K")
        assert retrieved is not None
        assert retrieved.model_name == "MockModel"
        assert retrieved.purity == "22K"

    def test_list_models(
        self, temp_registry: ModelRegistry, sample_data: pd.DataFrame
    ) -> None:
        """Test listing models with filters."""
        model1 = MockModel(purity="22K")
        model1.fit(sample_data)
        metadata1 = model1.get_metadata()
        temp_registry.register_model(
            metadata1, str(temp_registry.get_model_path("test", "MockModel", "22K"))
        )

        model2 = MockModel(purity="21K")
        model2.fit(sample_data)
        metadata2 = model2.get_metadata()
        temp_registry.register_model(
            metadata2, str(temp_registry.get_model_path("test", "MockModel", "21K"))
        )

        # List all models
        all_models = temp_registry.list_models()
        assert len(all_models) == 2

        # Filter by purity
        models_22k = temp_registry.list_models(purity="22K")
        assert len(models_22k) == 1
        assert models_22k[0].purity == "22K"

    def test_delete_model(
        self, temp_registry: ModelRegistry, sample_data: pd.DataFrame
    ) -> None:
        """Test deleting a model."""
        model = MockModel(purity="22K")
        model.fit(sample_data)
        metadata = model.get_metadata()
        temp_registry.register_model(
            metadata, str(temp_registry.get_model_path("test", "MockModel", "22K"))
        )

        # Verify it exists
        assert temp_registry.get_model_metadata("MockModel", "22K") is not None

        # Delete it
        deleted = temp_registry.delete_model("MockModel", "22K")
        assert deleted is True

        # Verify it's gone
        assert temp_registry.get_model_metadata("MockModel", "22K") is None

    def test_storage_stats(
        self, temp_registry: ModelRegistry, sample_data: pd.DataFrame
    ) -> None:
        """Test getting storage statistics."""
        model = MockModel(purity="22K")
        model.fit(sample_data)
        metadata = model.get_metadata()
        temp_registry.register_model(
            metadata, str(temp_registry.get_model_path("test", "MockModel", "22K"))
        )

        stats = temp_registry.get_storage_stats()
        assert "total_models" in stats
        assert "total_size_bytes" in stats
        assert stats["total_models"] >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
