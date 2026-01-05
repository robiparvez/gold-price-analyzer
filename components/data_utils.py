"""Data validation and preprocessing utilities for time-series forecasting.

This module provides functions for dataset quality checking, adaptive splitting,
time-series cross-validation, and data augmentation strategies optimized for
limited historical data (6-month horizon).
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)


@dataclass
class DatasetInfo:
    """Information about a time-series dataset."""

    total_samples: int
    date_range: tuple[str, str]
    num_days: int
    has_gaps: bool
    gap_count: int
    missing_percentage: float
    price_mean: float
    price_std: float
    price_range: tuple[float, float]
    recommended_min_samples: int
    data_quality: str  # "excellent", "good", "limited", "insufficient"


def analyze_dataset(df: pd.DataFrame, purity: str | None = None) -> DatasetInfo:
    """Analyze time-series dataset and provide quality metrics.

    Args:
        df: DataFrame with DatetimeIndex and price_bdt_per_gram column
        purity: Optional purity filter

    Returns:
        DatasetInfo with comprehensive dataset statistics

    Raises:
        ValueError: If DataFrame is invalid
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have DatetimeIndex")

    if "price_bdt_per_gram" not in df.columns:
        raise ValueError("DataFrame must have 'price_bdt_per_gram' column")

    # Filter by purity if specified
    if purity and "purity" in df.columns:
        df = df[df["purity"] == purity].copy()

    if len(df) == 0:
        raise ValueError(f"No data available for purity {purity}")

    # Calculate metrics
    total_samples = len(df)
    date_range = (
        df.index.min().strftime("%Y-%m-%d"),
        df.index.max().strftime("%Y-%m-%d"),
    )
    num_days = (df.index.max() - df.index.min()).days + 1

    # Check for gaps
    expected_dates = pd.date_range(start=df.index.min(), end=df.index.max(), freq="D")
    gap_count = len(set(expected_dates) - set(df.index))
    has_gaps = gap_count > 0

    # Missing values
    missing_count = df["price_bdt_per_gram"].isnull().sum()
    missing_percentage = (missing_count / total_samples) * 100

    # Price statistics
    price_mean = float(df["price_bdt_per_gram"].mean())
    price_std = float(df["price_bdt_per_gram"].std())
    price_range = (
        float(df["price_bdt_per_gram"].min()),
        float(df["price_bdt_per_gram"].max()),
    )

    # Determine data quality
    if total_samples >= 150:
        data_quality = "excellent"
        recommended_min_samples = 120
    elif total_samples >= 120:
        data_quality = "good"
        recommended_min_samples = 90
    elif total_samples >= 60:
        data_quality = "limited"
        recommended_min_samples = 60
    else:
        data_quality = "insufficient"
        recommended_min_samples = 60

    return DatasetInfo(
        total_samples=total_samples,
        date_range=date_range,
        num_days=num_days,
        has_gaps=has_gaps,
        gap_count=gap_count,
        missing_percentage=missing_percentage,
        price_mean=price_mean,
        price_std=price_std,
        price_range=price_range,
        recommended_min_samples=recommended_min_samples,
        data_quality=data_quality,
    )


def adaptive_train_test_split(
    df: pd.DataFrame, test_size: float | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data adaptively based on dataset size.

    For limited data (< 100 days): 80/20 split
    For moderate data (100-150 days): 75/25 split
    For good data (> 150 days): 70/30 split

    Args:
        df: DataFrame with DatetimeIndex
        test_size: Optional fixed test size (0-1). If None, use adaptive sizing

    Returns:
        Tuple of (train_df, test_df)
    """
    total_samples = len(df)

    if test_size is None:
        # Adaptive sizing
        if total_samples < 100:
            test_size = 0.20
        elif total_samples < 150:
            test_size = 0.25
        else:
            test_size = 0.30

    split_index = int(total_samples * (1 - test_size))

    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    logger.info(
        f"Split data: {len(train_df)} train samples, {len(test_df)} test samples "
        f"(test_size={test_size:.2%})"
    )

    return train_df, test_df


def adaptive_train_val_test_split(
    df: pd.DataFrame,
) -> (
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | tuple[pd.DataFrame, pd.DataFrame]
):
    """Split data into train/val/test with adaptive strategy.

    - If < 100 samples: train/test only (no validation set)
    - If 100-150 samples: 60/20/20 split
    - If > 150 samples: 70/15/15 split

    Args:
        df: DataFrame with DatetimeIndex

    Returns:
        Tuple of (train_df, val_df, test_df) or (train_df, test_df) if no val set
    """
    total_samples = len(df)

    if total_samples < 100:
        # No validation set for small datasets
        logger.info(
            f"Small dataset ({total_samples} samples): using train/test split only"
        )
        return adaptive_train_test_split(df, test_size=0.20)

    elif total_samples < 150:
        # 60/20/20 split
        train_end = int(total_samples * 0.60)
        val_end = int(total_samples * 0.80)

        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()

        logger.info(
            f"Split data: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test"
        )
        return train_df, val_df, test_df

    else:
        # 70/15/15 split
        train_end = int(total_samples * 0.70)
        val_end = int(total_samples * 0.85)

        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()

        logger.info(
            f"Split data: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test"
        )
        return train_df, val_df, test_df


def time_series_cv_split(
    df: pd.DataFrame, n_splits: int = 5, test_size: int | None = None
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Perform time-series cross-validation split.

    Args:
        df: DataFrame with DatetimeIndex
        n_splits: Number of splits
        test_size: Size of test set. If None, adaptive based on data size

    Returns:
        List of (train, test) DataFrame tuples
    """
    total_samples = len(df)

    # Adaptive test size
    if test_size is None:
        if total_samples < 100:
            test_size = max(7, total_samples // 10)  # Min 7 days or 10% of data
        else:
            test_size = 7  # Standard 7-day forecast horizon

    # Adjust n_splits if data is too small
    max_splits = (total_samples - test_size) // test_size
    n_splits = min(n_splits, max(3, max_splits))

    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)

    splits = []
    for train_idx, test_idx in tscv.split(df):
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()
        splits.append((train_df, test_df))

    logger.info(f"Created {len(splits)} time-series CV splits")
    return splits


def augment_time_series_jittering(
    df: pd.DataFrame, noise_level: float = 0.005, n_augmentations: int = 1
) -> pd.DataFrame:
    """Augment time-series data with jittering (adding small noise).

    Args:
        df: DataFrame with price_bdt_per_gram column
        noise_level: Standard deviation of Gaussian noise (as fraction of price std)
        n_augmentations: Number of augmented versions to create

    Returns:
        DataFrame with original + augmented data
    """
    if "price_bdt_per_gram" not in df.columns:
        raise ValueError("DataFrame must have 'price_bdt_per_gram' column")

    price_std = df["price_bdt_per_gram"].std()
    noise_std = price_std * noise_level

    augmented_dfs = [df.copy()]

    for i in range(n_augmentations):
        aug_df = df.copy()
        noise = np.random.normal(0, noise_std, len(aug_df))
        aug_df["price_bdt_per_gram"] = aug_df["price_bdt_per_gram"] + noise

        # Ensure no negative prices
        aug_df["price_bdt_per_gram"] = aug_df["price_bdt_per_gram"].clip(lower=0)

        augmented_dfs.append(aug_df)

    result = pd.concat(augmented_dfs, ignore_index=True)
    logger.info(
        f"Augmented dataset from {len(df)} to {len(result)} samples "
        f"(noise_level={noise_level:.3%})"
    )

    return result


def create_sliding_windows(
    df: pd.DataFrame,
    lookback: int = 7,
    horizon: int = 7,
    step: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Create sliding window sequences for time-series modeling.

    Args:
        df: DataFrame with price_bdt_per_gram column
        lookback: Number of past days to use as input
        horizon: Number of future days to predict
        step: Step size for sliding window

    Returns:
        Tuple of (X, y) where X is (n_samples, lookback) and y is (n_samples, horizon)
    """
    if "price_bdt_per_gram" not in df.columns:
        raise ValueError("DataFrame must have 'price_bdt_per_gram' column")

    prices = df["price_bdt_per_gram"].values

    X, y = [], []

    for i in range(0, len(prices) - lookback - horizon + 1, step):
        X.append(prices[i : i + lookback])
        y.append(prices[i + lookback : i + lookback + horizon])

    X = np.array(X)
    y = np.array(y)

    logger.info(
        f"Created {len(X)} sliding windows (lookback={lookback}, horizon={horizon}, step={step})"
    )

    return X, y


def fill_missing_dates(
    df: pd.DataFrame, method: str = "linear", fill_weekends: bool = False
) -> pd.DataFrame:
    """Fill missing dates in time-series data.

    Args:
        df: DataFrame with DatetimeIndex
        method: Interpolation method ('linear', 'ffill', 'bfill')
        fill_weekends: Whether to fill weekend gaps (if False, keep weekends empty)

    Returns:
        DataFrame with filled dates
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have DatetimeIndex")

    # Create complete date range
    full_date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq="D")

    # Reindex to include all dates
    df_filled = df.reindex(full_date_range)

    if not fill_weekends:
        # Keep weekends as NaN (gold markets often closed on weekends)
        weekday_mask = df_filled.index.dayofweek < 5  # Monday=0, Friday=4
        df_filled = df_filled[weekday_mask]

    # Fill missing values
    if method == "linear":
        df_filled = df_filled.interpolate(method="linear")
    elif method == "ffill":
        df_filled = df_filled.fillna(method="ffill")
    elif method == "bfill":
        df_filled = df_filled.fillna(method="bfill")
    else:
        raise ValueError(f"Invalid method: {method}")

    # Forward/backward fill any remaining NaN at edges
    df_filled = df_filled.fillna(method="ffill").fillna(method="bfill")

    logger.info(f"Filled missing dates: {len(df)} -> {len(df_filled)} samples")

    return df_filled


def remove_outliers(
    df: pd.DataFrame, column: str = "price_bdt_per_gram", n_std: float = 3.0
) -> pd.DataFrame:
    """Remove outliers using standard deviation method.

    Args:
        df: DataFrame
        column: Column to check for outliers
        n_std: Number of standard deviations for outlier threshold

    Returns:
        DataFrame with outliers removed
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")

    mean = df[column].mean()
    std = df[column].std()

    lower_bound = mean - n_std * std
    upper_bound = mean + n_std * std

    df_clean = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)].copy()

    removed_count = len(df) - len(df_clean)
    if removed_count > 0:
        logger.warning(
            f"Removed {removed_count} outliers from {column} "
            f"(bounds: {lower_bound:.2f} - {upper_bound:.2f})"
        )

    return df_clean


def get_recommended_models(dataset_info: DatasetInfo) -> dict[str, Any]:
    """Get recommended models based on dataset characteristics.

    Args:
        dataset_info: Information about the dataset

    Returns:
        Dictionary with recommended models and configurations
    """
    recommendations = {
        "recommended_models": [],
        "avoid_models": [],
        "train_config": {},
        "warnings": [],
    }

    total_samples = dataset_info.total_samples

    # Determine which models to use
    if total_samples >= 150:
        recommendations["recommended_models"] = [
            "prophet",
            "arima",
            "lightgbm",
            "lstm",
            "gru",
            "tcn",
            "catboost",
            "svr",
        ]
        recommendations["train_config"] = {
            "use_validation_set": True,
            "enable_hyperparameter_tuning": True,
            "n_cv_splits": 5,
        }

    elif total_samples >= 120:
        recommendations["recommended_models"] = [
            "prophet",
            "arima",
            "lightgbm",
            "lstm",
            "gru",
            "catboost",
        ]
        recommendations["avoid_models"] = ["tcn", "svr", "stacked_lstm", "transformer"]
        recommendations["train_config"] = {
            "use_validation_set": True,
            "enable_hyperparameter_tuning": False,  # Limited tuning
            "n_cv_splits": 4,
        }
        recommendations["warnings"].append(
            "Dataset size is moderate. Avoid complex deep learning models."
        )

    elif total_samples >= 60:
        recommendations["recommended_models"] = [
            "prophet",
            "arima",
            "lightgbm",
            "simple_lstm",
        ]
        recommendations["avoid_models"] = [
            "gru",
            "tcn",
            "transformer",
            "catboost",
            "svr",
            "ensemble",
        ]
        recommendations["train_config"] = {
            "use_validation_set": False,  # Too small for 3-way split
            "enable_hyperparameter_tuning": False,
            "n_cv_splits": 3,
        }
        recommendations["warnings"].extend(
            [
                "Limited dataset. Use only simple models.",
                "Avoid deep learning models to prevent overfitting.",
                "Consider data augmentation techniques.",
            ]
        )

    else:
        recommendations["recommended_models"] = ["prophet"]
        recommendations["avoid_models"] = [
            "arima",
            "lightgbm",
            "lstm",
            "gru",
            "catboost",
            "svr",
            "any_deep_learning",
        ]
        recommendations["train_config"] = {
            "use_validation_set": False,
            "enable_hyperparameter_tuning": False,
            "n_cv_splits": None,  # Not enough data for CV
        }
        recommendations["warnings"].extend(
            [
                f"Insufficient data ({total_samples} samples < 60 required).",
                "Only Prophet or naive baseline recommended.",
                "Collect more historical data before using advanced models.",
            ]
        )

    # Check for data quality issues
    if dataset_info.has_gaps:
        recommendations["warnings"].append(
            f"Dataset has {dataset_info.gap_count} date gaps. Consider filling them."
        )

    if dataset_info.missing_percentage > 5:
        recommendations["warnings"].append(
            f"High missing data percentage: {dataset_info.missing_percentage:.1f}%"
        )

    return recommendations
