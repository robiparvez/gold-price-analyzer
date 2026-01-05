"""Sliding window dataset utilities for deep learning models."""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def create_sliding_windows(
    data: np.ndarray | pd.Series,
    lookback: int,
    forecast_horizon: int = 1,
    gap: int = 0,
) -> tuple:
    """Create sliding windows for time series prediction.
    
    Args:
        data: Time series data (1D array or Series).
        lookback: Number of past time steps to use as input.
        forecast_horizon: Number of steps ahead to predict.
        gap: Number of steps to skip between input and target.
        
    Returns:
        Tuple of (X, y) where:
        - X: (n_samples, lookback) input windows
        - y: (n_samples, forecast_horizon) target values
    """
    if isinstance(data, pd.Series):
        data = data.values
    
    if len(data) < lookback + gap + forecast_horizon:
        raise ValueError(
            f"Data length {len(data)} too short for lookback={lookback}, "
            f"gap={gap}, forecast_horizon={forecast_horizon}"
        )
    
    X, y = [], []
    
    for i in range(len(data) - lookback - gap - forecast_horizon + 1):
        # Input window
        X.append(data[i : i + lookback])
        
        # Target window (after gap)
        target_start = i + lookback + gap
        target_end = target_start + forecast_horizon
        y.append(data[target_start : target_end])
    
    return np.array(X), np.array(y)


def create_multivariate_windows(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    lookback: int,
    forecast_horizon: int = 1,
) -> tuple:
    """Create sliding windows for multivariate time series.
    
    Args:
        X: Multivariate features (n_samples, n_features).
        y: Target values (n_samples,).
        lookback: Number of past time steps to use as input.
        forecast_horizon: Number of steps ahead to predict.
        
    Returns:
        Tuple of (X_windows, y_windows) where:
        - X_windows: (n_samples, lookback, n_features) input windows
        - y_windows: (n_samples, forecast_horizon) target windows
    """
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values
    
    if len(X) != len(y):
        raise ValueError("X and y must have same length")
    
    if len(X) < lookback + forecast_horizon:
        raise ValueError(
            f"Data length {len(X)} too short for lookback={lookback}, "
            f"forecast_horizon={forecast_horizon}"
        )
    
    X_windows, y_windows = [], []
    
    for i in range(len(X) - lookback - forecast_horizon + 1):
        # Input window
        X_windows.append(X[i : i + lookback])
        
        # Target window
        y_start = i + lookback
        y_end = y_start + forecast_horizon
        y_windows.append(y[y_start : y_end])
    
    return np.array(X_windows), np.array(y_windows)


def augment_data(
    X: np.ndarray,
    y: np.ndarray,
    noise_std: float = 0.01,
    shift_range: float = 0.05,
    num_augmentations: int = 2,
) -> tuple:
    """Augment dataset by adding noise and shifting values.
    
    Args:
        X: Input windows (n_samples, lookback, ...).
        y: Target windows (n_samples, forecast_horizon, ...).
        noise_std: Standard deviation of Gaussian noise as fraction of mean.
        shift_range: Range for random shift as fraction of mean.
        num_augmentations: Number of augmented copies to create.
        
    Returns:
        Tuple of (X_augmented, y_augmented).
    """
    X_list = [X]
    y_list = [y]
    
    for _ in range(num_augmentations):
        # Add Gaussian noise
        noise_X = np.random.normal(0, noise_std * np.mean(np.abs(X)), X.shape)
        X_aug = X + noise_X
        
        noise_y = np.random.normal(0, noise_std * np.mean(np.abs(y)), y.shape)
        y_aug = y + noise_y
        
        # Add small random shifts
        shift = np.random.uniform(-shift_range, shift_range)
        X_aug = X_aug * (1 + shift)
        y_aug = y_aug * (1 + shift)
        
        X_list.append(X_aug)
        y_list.append(y_aug)
    
    return np.vstack(X_list), np.vstack(y_list)


def normalize_for_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
) -> tuple:
    """Normalize data for LSTM using training statistics.
    
    Args:
        X_train: Training features.
        y_train: Training targets.
        X_test: Test features (optional).
        y_test: Test targets (optional).
        
    Returns:
        Tuple of (X_train_norm, y_train_norm, X_params, X_test_norm, y_test_norm).
    """
    # Normalize X (features)
    X_mean = np.mean(X_train)
    X_std = np.std(X_train)
    if X_std == 0:
        X_std = 1.0
    
    X_train_norm = (X_train - X_mean) / X_std
    X_params = {"mean": X_mean, "std": X_std}
    
    X_test_norm = None
    if X_test is not None:
        X_test_norm = (X_test - X_mean) / X_std
    
    # Normalize y (targets)
    y_mean = np.mean(y_train)
    y_std = np.std(y_train)
    if y_std == 0:
        y_std = 1.0
    
    y_train_norm = (y_train - y_mean) / y_std
    y_params = {"mean": y_mean, "std": y_std}
    
    y_test_norm = None
    if y_test is not None:
        y_test_norm = (y_test - y_mean) / y_std
    
    return X_train_norm, y_train_norm, X_params, X_test_norm, y_test_norm, y_params


def denormalize(
    data: np.ndarray,
    params: dict,
) -> np.ndarray:
    """Denormalize data using stored parameters.
    
    Args:
        data: Normalized data.
        params: Normalization parameters (mean, std).
        
    Returns:
        Denormalized data.
    """
    mean = params.get("mean", 0)
    std = params.get("std", 1)
    return data * std + mean
