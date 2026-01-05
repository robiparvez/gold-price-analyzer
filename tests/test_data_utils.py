"""Tests for data validation and preprocessing utilities."""

import numpy as np
import pandas as pd
import pytest

from components.data_utils import (
    DatasetInfo,
    adaptive_train_test_split,
    adaptive_train_val_test_split,
    analyze_dataset,
    augment_time_series_jittering,
    create_sliding_windows,
    fill_missing_dates,
    get_recommended_models,
    remove_outliers,
    time_series_cv_split,
)


@pytest.fixture
def small_dataset() -> pd.DataFrame:
    """Create a small dataset (50 samples)."""
    dates = pd.date_range(start="2025-11-01", periods=50, freq="D")
    prices = 100 + np.cumsum(np.random.randn(50) * 2)
    df = pd.DataFrame(
        {"date": dates, "price_bdt_per_gram": prices, "purity": "22K"}
    )
    df.set_index("date", inplace=True)
    return df


@pytest.fixture
def medium_dataset() -> pd.DataFrame:
    """Create a medium dataset (100 samples)."""
    dates = pd.date_range(start="2025-07-01", periods=100, freq="D")
    prices = 100 + np.cumsum(np.random.randn(100) * 2)
    df = pd.DataFrame(
        {"date": dates, "price_bdt_per_gram": prices, "purity": "22K"}
    )
    df.set_index("date", inplace=True)
    return df


@pytest.fixture
def large_dataset() -> pd.DataFrame:
    """Create a large dataset (180 samples - 6 months)."""
    dates = pd.date_range(start="2025-07-01", periods=180, freq="D")
    prices = 100 + np.cumsum(np.random.randn(180) * 2)
    df = pd.DataFrame(
        {"date": dates, "price_bdt_per_gram": prices, "purity": "22K"}
    )
    df.set_index("date", inplace=True)
    return df


@pytest.fixture
def dataset_with_gaps() -> pd.DataFrame:
    """Create dataset with missing dates."""
    dates = pd.date_range(start="2025-10-01", periods=100, freq="D")
    # Remove some dates to create gaps
    dates = dates.drop([dates[10], dates[20], dates[30], dates[40]])
    prices = 100 + np.cumsum(np.random.randn(len(dates)) * 2)
    df = pd.DataFrame(
        {"date": dates, "price_bdt_per_gram": prices, "purity": "22K"}
    )
    df.set_index("date", inplace=True)
    return df


class TestAnalyzeDataset:
    """Tests for analyze_dataset function."""

    def test_small_dataset_analysis(self, small_dataset: pd.DataFrame) -> None:
        """Test analyzing a small dataset."""
        info = analyze_dataset(small_dataset)

        assert isinstance(info, DatasetInfo)
        assert info.total_samples == 50
        assert info.data_quality == "insufficient"
        assert info.num_days == 50
        assert not info.has_gaps

    def test_medium_dataset_analysis(self, medium_dataset: pd.DataFrame) -> None:
        """Test analyzing a medium dataset."""
        info = analyze_dataset(medium_dataset)

        assert info.total_samples == 100
        assert info.data_quality == "limited"
        assert info.recommended_min_samples == 60

    def test_large_dataset_analysis(self, large_dataset: pd.DataFrame) -> None:
        """Test analyzing a large dataset."""
        info = analyze_dataset(large_dataset)

        assert info.total_samples == 180
        assert info.data_quality == "excellent"
        assert info.price_mean > 0
        assert info.price_std > 0

    def test_dataset_with_gaps(self, dataset_with_gaps: pd.DataFrame) -> None:
        """Test analyzing dataset with date gaps."""
        info = analyze_dataset(dataset_with_gaps)

        assert info.has_gaps
        assert info.gap_count > 0

    def test_invalid_dataframe(self) -> None:
        """Test with invalid DataFrame."""
        df = pd.DataFrame({"price": [100, 101]})

        with pytest.raises(ValueError, match="DatetimeIndex"):
            analyze_dataset(df)


class TestAdaptiveTrainTestSplit:
    """Tests for adaptive_train_test_split function."""

    def test_small_dataset_split(self, small_dataset: pd.DataFrame) -> None:
        """Test splitting small dataset uses 80/20."""
        train, test = adaptive_train_test_split(small_dataset)

        assert len(train) == 40  # 80% of 50
        assert len(test) == 10  # 20% of 50
        assert train.index.max() < test.index.min()  # Time order preserved

    def test_medium_dataset_split(self, medium_dataset: pd.DataFrame) -> None:
        """Test splitting medium dataset uses 75/25."""
        train, test = adaptive_train_test_split(medium_dataset)

        assert len(train) == 75  # 75% of 100
        assert len(test) == 25

    def test_large_dataset_split(self, large_dataset: pd.DataFrame) -> None:
        """Test splitting large dataset uses 70/30."""
        train, test = adaptive_train_test_split(large_dataset)

        # 70% of 180 = 126, but int() truncates so we get 125/55 split
        assert len(train) + len(test) == 180
        assert len(train) >= 125  # Close to 70%
        assert len(test) >= 54  # Close to 30%

    def test_custom_test_size(self, medium_dataset: pd.DataFrame) -> None:
        """Test with custom test size."""
        train, test = adaptive_train_test_split(medium_dataset, test_size=0.15)

        assert len(test) == 15
        assert len(train) == 85


class TestAdaptiveTrainValTestSplit:
    """Tests for adaptive_train_val_test_split function."""

    def test_small_dataset_no_val(self, small_dataset: pd.DataFrame) -> None:
        """Test small dataset returns only train/test."""
        result = adaptive_train_val_test_split(small_dataset)

        assert len(result) == 2  # Only train and test
        train, test = result
        assert len(train) + len(test) == 50

    def test_medium_dataset_with_val(self, medium_dataset: pd.DataFrame) -> None:
        """Test medium dataset returns train/val/test."""
        result = adaptive_train_val_test_split(medium_dataset)

        assert len(result) == 3
        train, val, test = result
        assert len(train) == 60  # 60%
        assert len(val) == 20  # 20%
        assert len(test) == 20  # 20%

    def test_large_dataset_split(self, large_dataset: pd.DataFrame) -> None:
        """Test large dataset uses 70/15/15 split."""
        result = adaptive_train_val_test_split(large_dataset)

        assert len(result) == 3
        train, val, test = result
        # Due to int() truncation, exact split may vary by 1
        assert len(train) + len(val) + len(test) == 180
        assert len(train) >= 125  # Approximately 70%
        assert len(val) >= 26  # Approximately 15%
        assert len(test) >= 26  # Approximately 15%


class TestTimeSeriesCVSplit:
    """Tests for time_series_cv_split function."""

    def test_default_splits(self, large_dataset: pd.DataFrame) -> None:
        """Test creating default CV splits."""
        splits = time_series_cv_split(large_dataset)

        assert len(splits) == 5  # Default n_splits
        assert all(isinstance(s, tuple) for s in splits)
        assert all(len(s) == 2 for s in splits)

        # Check time order
        for train, test in splits:
            assert train.index.max() < test.index.min()

    def test_small_dataset_adaptive_splits(self, small_dataset: pd.DataFrame) -> None:
        """Test CV splits adapt to small dataset."""
        splits = time_series_cv_split(small_dataset, n_splits=5)

        # Should reduce number of splits for small data
        assert len(splits) >= 3
        assert all(len(test) >= 5 for _, test in splits)

    def test_custom_test_size(self, medium_dataset: pd.DataFrame) -> None:
        """Test CV with custom test size."""
        splits = time_series_cv_split(medium_dataset, n_splits=3, test_size=10)

        assert all(len(test) == 10 for _, test in splits)


class TestAugmentTimeSeriesJittering:
    """Tests for augment_time_series_jittering function."""

    def test_single_augmentation(self, medium_dataset: pd.DataFrame) -> None:
        """Test augmenting dataset once."""
        augmented = augment_time_series_jittering(
            medium_dataset, noise_level=0.005, n_augmentations=1
        )

        assert len(augmented) == 200  # Original + 1 augmentation
        assert "price_bdt_per_gram" in augmented.columns

    def test_multiple_augmentations(self, small_dataset: pd.DataFrame) -> None:
        """Test multiple augmentations."""
        augmented = augment_time_series_jittering(
            small_dataset, noise_level=0.01, n_augmentations=3
        )

        assert len(augmented) == 200  # Original + 3 augmentations

    def test_no_negative_prices(self, small_dataset: pd.DataFrame) -> None:
        """Test that augmentation doesn't create negative prices."""
        augmented = augment_time_series_jittering(
            small_dataset, noise_level=0.1, n_augmentations=5
        )

        assert (augmented["price_bdt_per_gram"] >= 0).all()


class TestCreateSlidingWindows:
    """Tests for create_sliding_windows function."""

    def test_default_windows(self, medium_dataset: pd.DataFrame) -> None:
        """Test creating default sliding windows."""
        X, y = create_sliding_windows(medium_dataset, lookback=7, horizon=7)

        assert X.shape[1] == 7  # Lookback
        assert y.shape[1] == 7  # Horizon
        assert len(X) == len(y)
        assert len(X) > 0

    def test_custom_window_sizes(self, large_dataset: pd.DataFrame) -> None:
        """Test custom lookback and horizon."""
        X, y = create_sliding_windows(large_dataset, lookback=14, horizon=3)

        assert X.shape[1] == 14
        assert y.shape[1] == 3

    def test_step_size(self, medium_dataset: pd.DataFrame) -> None:
        """Test different step sizes."""
        X_step1, y_step1 = create_sliding_windows(medium_dataset, step=1)
        X_step7, y_step7 = create_sliding_windows(medium_dataset, step=7)

        # Step=7 should produce fewer windows
        assert len(X_step7) < len(X_step1)


class TestFillMissingDates:
    """Tests for fill_missing_dates function."""

    def test_fill_gaps_linear(self, dataset_with_gaps: pd.DataFrame) -> None:
        """Test filling gaps with linear interpolation."""
        original_len = len(dataset_with_gaps)
        # Fill weekends=True to actually fill all dates
        filled = fill_missing_dates(dataset_with_gaps, method="linear", fill_weekends=True)

        # Should have more samples after filling (including removed gaps)
        assert len(filled) >= original_len
        assert not filled["price_bdt_per_gram"].isnull().any()

    def test_fill_forward(self, dataset_with_gaps: pd.DataFrame) -> None:
        """Test forward fill method."""
        filled = fill_missing_dates(dataset_with_gaps, method="ffill")

        assert not filled["price_bdt_per_gram"].isnull().any()

    def test_weekends_excluded(self, dataset_with_gaps: pd.DataFrame) -> None:
        """Test that weekends can be excluded."""
        filled = fill_missing_dates(
            dataset_with_gaps, method="linear", fill_weekends=False
        )

        # All dates should be weekdays
        assert all(filled.index.dayofweek < 5)


class TestRemoveOutliers:
    """Tests for remove_outliers function."""

    def test_remove_extreme_outliers(self, medium_dataset: pd.DataFrame) -> None:
        """Test removing extreme outliers."""
        # Add an extreme outlier
        df = medium_dataset.copy()
        df.loc[df.index[50], "price_bdt_per_gram"] = 10000  # Extreme value

        cleaned = remove_outliers(df, n_std=3.0)

        # Outlier should be removed
        assert len(cleaned) < len(df)
        assert 10000 not in cleaned["price_bdt_per_gram"].values

    def test_no_outliers_removed(self, medium_dataset: pd.DataFrame) -> None:
        """Test when no outliers exist."""
        cleaned = remove_outliers(medium_dataset, n_std=3.0)

        # All data should remain
        assert len(cleaned) == len(medium_dataset)


class TestGetRecommendedModels:
    """Tests for get_recommended_models function."""

    def test_small_dataset_recommendations(self, small_dataset: pd.DataFrame) -> None:
        """Test recommendations for small dataset."""
        info = analyze_dataset(small_dataset)
        recommendations = get_recommended_models(info)

        assert "recommended_models" in recommendations
        assert "prophet" in recommendations["recommended_models"]
        assert len(recommendations["warnings"]) > 0
        assert recommendations["train_config"]["use_validation_set"] is False

    def test_medium_dataset_recommendations(
        self, medium_dataset: pd.DataFrame
    ) -> None:
        """Test recommendations for medium dataset."""
        info = analyze_dataset(medium_dataset)
        recommendations = get_recommended_models(info)

        assert "lightgbm" in recommendations["recommended_models"]
        assert "transformer" not in recommendations["recommended_models"]
        assert recommendations["train_config"]["n_cv_splits"] >= 3

    def test_large_dataset_recommendations(self, large_dataset: pd.DataFrame) -> None:
        """Test recommendations for large dataset."""
        info = analyze_dataset(large_dataset)
        recommendations = get_recommended_models(info)

        assert "lstm" in recommendations["recommended_models"]
        assert "tcn" in recommendations["recommended_models"]
        assert recommendations["train_config"]["use_validation_set"] is True
        assert recommendations["train_config"]["enable_hyperparameter_tuning"] is True

    def test_recommendations_with_gaps(
        self, dataset_with_gaps: pd.DataFrame
    ) -> None:
        """Test that gap warnings are included."""
        info = analyze_dataset(dataset_with_gaps)
        recommendations = get_recommended_models(info)

        # Should have warning about gaps
        assert any("gap" in w.lower() for w in recommendations["warnings"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
