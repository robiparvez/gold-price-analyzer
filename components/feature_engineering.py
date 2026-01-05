"""Feature engineering utilities for machine learning models.

Provides functions to create time-series features from price data,
optimized for limited datasets (6-month constraint).
"""

import numpy as np
import pandas as pd


def create_lag_features(
    series: pd.Series,
    lags: list[int] | None = None,
    dropna: bool = True,
) -> pd.DataFrame:
    """Create lagged features from time series.
    
    Args:
        series: Input time series.
        lags: List of lag values. Defaults to [1, 3, 7, 14, 30].
        dropna: Whether to drop rows with NaN values.
        
    Returns:
        DataFrame with lag features.
    """
    if lags is None:
        # Use conservative lags for small datasets
        available_length = len(series)
        lags = [lag for lag in [1, 3, 7, 14, 30] if lag < available_length // 3]
    
    df = pd.DataFrame({"price": series})
    
    for lag in lags:
        df[f"lag_{lag}"] = series.shift(lag)
    
    if dropna:
        df = df.dropna()
    
    return df


def create_rolling_features(
    series: pd.Series,
    windows: list[int] | None = None,
    features: list[str] | None = None,
    dropna: bool = True,
) -> pd.DataFrame:
    """Create rolling window features.
    
    Args:
        series: Input time series.
        windows: Window sizes. Defaults to [3, 7, 14].
        features: Features to compute ('mean', 'std', 'min', 'max', 'range').
        dropna: Whether to drop rows with NaN values.
        
    Returns:
        DataFrame with rolling features.
    """
    if windows is None:
        available_length = len(series)
        windows = [w for w in [3, 7, 14] if w < available_length // 2]
    
    if features is None:
        features = ["mean", "std"]
    
    df = pd.DataFrame(index=series.index)
    
    for window in windows:
        for feature in features:
            col_name = f"rolling_{window}_{feature}"
            
            if feature == "mean":
                df[col_name] = series.rolling(window=window).mean()
            elif feature == "std":
                df[col_name] = series.rolling(window=window).std()
            elif feature == "min":
                df[col_name] = series.rolling(window=window).min()
            elif feature == "max":
                df[col_name] = series.rolling(window=window).max()
            elif feature == "range":
                rolling_min = series.rolling(window=window).min()
                rolling_max = series.rolling(window=window).max()
                df[col_name] = rolling_max - rolling_min
    
    if dropna:
        df = df.dropna()
    
    return df


def create_diff_features(
    series: pd.Series,
    periods: list[int] | None = None,
    dropna: bool = True,
) -> pd.DataFrame:
    """Create differencing features.
    
    Args:
        series: Input time series.
        periods: Differencing periods. Defaults to [1, 7].
        dropna: Whether to drop rows with NaN values.
        
    Returns:
        DataFrame with differencing features.
    """
    if periods is None:
        periods = [1, 7]
    
    df = pd.DataFrame(index=series.index)
    
    for period in periods:
        df[f"diff_{period}"] = series.diff(period)
    
    if dropna:
        df = df.dropna()
    
    return df


def create_time_features(
    dates: pd.DatetimeIndex,
    include_cyclical: bool = True,
) -> pd.DataFrame:
    """Create time-based features from dates.
    
    Args:
        dates: DatetimeIndex.
        include_cyclical: Whether to include cyclical features (sin/cos).
        
    Returns:
        DataFrame with time features.
    """
    df = pd.DataFrame(index=dates)
    
    # Basic time features
    df["day_of_week"] = dates.dayofweek
    df["day_of_month"] = dates.day
    df["month"] = dates.month
    df["quarter"] = dates.quarter
    df["week_of_year"] = dates.isocalendar().week
    
    if include_cyclical:
        # Cyclical encoding for day of week
        df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
        
        # Cyclical encoding for month
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        
        # Cyclical encoding for day of year
        day_of_year = dates.dayofyear
        df["day_of_year_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
        df["day_of_year_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    
    return df


def create_all_features(
    series: pd.Series,
    include_lags: bool = True,
    include_rolling: bool = True,
    include_diff: bool = True,
    include_time: bool = True,
    dropna: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Create comprehensive feature set.
    
    Args:
        series: Input time series.
        include_lags: Whether to include lag features.
        include_rolling: Whether to include rolling features.
        include_diff: Whether to include differencing features.
        include_time: Whether to include time features.
        dropna: Whether to drop rows with NaN values.
        
    Returns:
        Tuple of (features DataFrame, feature names list).
    """
    features = pd.DataFrame(index=series.index)
    feature_names = []
    
    if include_lags:
        lag_features = create_lag_features(series, dropna=False)
        lag_cols = [col for col in lag_features.columns if col.startswith("lag_")]
        features[lag_cols] = lag_features[lag_cols]
        feature_names.extend(lag_cols)
    
    if include_rolling:
        rolling_features = create_rolling_features(series, dropna=False)
        rolling_cols = [col for col in rolling_features.columns if col.startswith("rolling_")]
        features[rolling_cols] = rolling_features[rolling_cols]
        feature_names.extend(rolling_cols)
    
    if include_diff:
        diff_features = create_diff_features(series, dropna=False)
        diff_cols = [col for col in diff_features.columns if col.startswith("diff_")]
        features[diff_cols] = diff_features[diff_cols]
        feature_names.extend(diff_cols)
    
    if include_time:
        time_features = create_time_features(series.index)
        time_cols = [col for col in time_features.columns]
        features[time_cols] = time_features[time_cols]
        feature_names.extend(time_cols)
    
    if dropna:
        features = features.dropna()
    
    return features, feature_names


def normalize_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame | None = None,
    method: str = "minmax",
) -> tuple[pd.DataFrame, pd.DataFrame | None, dict]:
    """Normalize features using specified method.
    
    Args:
        X_train: Training features.
        X_test: Test features (optional).
        method: 'minmax' or 'zscore'.
        
    Returns:
        Tuple of (normalized X_train, normalized X_test, normalization params).
    """
    params = {}
    
    if method == "minmax":
        x_min = X_train.min()
        x_max = X_train.max()
        x_range = x_max - x_min
        x_range[x_range == 0] = 1  # Avoid division by zero
        
        X_train_norm = (X_train - x_min) / x_range
        params = {"min": x_min, "max": x_max, "range": x_range}
        
        if X_test is not None:
            X_test_norm = (X_test - x_min) / x_range
        else:
            X_test_norm = None
    
    elif method == "zscore":
        mean = X_train.mean()
        std = X_train.std()
        std[std == 0] = 1  # Avoid division by zero
        
        X_train_norm = (X_train - mean) / std
        params = {"mean": mean, "std": std}
        
        if X_test is not None:
            X_test_norm = (X_test - mean) / std
        else:
            X_test_norm = None
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return X_train_norm, X_test_norm, params


def select_important_features(
    X: pd.DataFrame,
    y: pd.Series,
    n_features: int | None = None,
    method: str = "correlation",
) -> list[str]:
    """Select important features based on criterion.
    
    Args:
        X: Feature matrix.
        y: Target variable.
        n_features: Number of features to select. If None, keep all with |correlation| > 0.3.
        method: 'correlation' or 'variance'.
        
    Returns:
        List of selected feature names.
    """
    if method == "correlation":
        # Select features with highest absolute correlation with target
        correlations = X.corrwith(y).abs().sort_values(ascending=False)
        
        if n_features is None:
            # Keep features with correlation > 0.3
            selected = correlations[correlations > 0.3].index.tolist()
            # If too few, keep at least top 5
            if len(selected) < 5:
                selected = correlations.head(max(5, len(X.columns) // 3)).index.tolist()
        else:
            selected = correlations.head(n_features).index.tolist()
    
    elif method == "variance":
        # Select features with highest variance
        variances = X.var().sort_values(ascending=False)
        
        if n_features is None:
            # Keep top 30% by variance
            n_keep = max(5, len(X.columns) // 3)
            selected = variances.head(n_keep).index.tolist()
        else:
            selected = variances.head(n_features).index.tolist()
    
    else:
        raise ValueError(f"Unknown feature selection method: {method}")
    
    return selected
