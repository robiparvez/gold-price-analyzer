"""Support Vector Regression (SVR) forecasting model."""

import logging
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.svm import SVR

from components.feature_engineering import create_all_features, normalize_features
from models.time_series_base import BaseTimeSeriesModel, ForecastResult, ModelMetadata

logger = logging.getLogger(__name__)


class SVRModel(BaseTimeSeriesModel):
    """Support Vector Regression model for time-series forecasting.
    
    Features:
    - Automatic feature engineering (lags, rolling, differencing, time features)
    - RBF kernel with feature normalization
    - Conservative hyperparameters for small datasets
    - Memory-efficient for limited data
    - Multi-step ahead forecasting
    
    Attributes:
        kernel: Kernel type ('rbf', 'linear', 'poly')
        C: Regularization parameter (default 1.0)
        epsilon: Epsilon-tube (default 0.01)
        gamma: Kernel coefficient ('scale' or float)
    """
    
    def __init__(
        self,
        kernel: str = "rbf",
        C: float = 1.0,
        epsilon: float = 0.01,
        gamma: str | float = "scale",
        **kwargs,
    ):
        """Initialize SVR model.
        
        Args:
            kernel: Kernel type ('rbf', 'linear', 'poly').
            C: Regularization parameter.
            epsilon: Epsilon-tube parameter.
            gamma: Kernel coefficient ('scale' or float).
            **kwargs: Additional arguments.
        """
        super().__init__(purity="22K", **kwargs)
        
        self.model_type = "ml_enhanced"
        self.version = "1.0.0"
        self.hyperparameters = {
            "kernel": kernel,
            "C": C,
            "epsilon": epsilon,
            "gamma": gamma,
        }
        
        self._model = None
        self._fitted_model = None
        self._training_data = None
        self._features = None
        self._feature_names = None
        self._normalization_params = None
        self._y_mean = None
        self._y_std = None
        self._diagnostics: dict[str, Any] = {}
        
        # Store hyperparameters
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.gamma = gamma
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SVRModel":
        """Fit SVR model to training data.
        
        Args:
            X: Feature dataframe with datetime index.
            y: Target variable (prices).
            
        Returns:
            Self (fitted model).
            
        Raises:
            ValueError: If data is invalid or model fitting fails.
        """
        # Validate data
        self.validate_data(X, y)
        
        # Store training data
        self._training_data = y.copy()
        
        try:
            # Create features from time series
            features, feature_names = create_all_features(
                y,
                include_lags=True,
                include_rolling=True,
                include_diff=True,
                include_time=True,
                dropna=True,
            )
            
            self._feature_names = feature_names
            
            # Normalize features (SVR requires normalized input)
            X_norm, _, norm_params = normalize_features(features, method="zscore")
            self._normalization_params = norm_params
            
            # Store features for prediction
            self._features = features.copy()
            
            # Get target values aligned with features
            y_aligned = y[features.index].values
            
            # Normalize target for better convergence
            self._y_mean = np.mean(y_aligned)
            self._y_std = np.std(y_aligned)
            if self._y_std == 0:
                self._y_std = 1.0
            y_normalized = (y_aligned - self._y_mean) / self._y_std
            
            # Create and fit SVR model
            self._model = SVR(
                kernel=self.kernel,
                C=self.C,
                epsilon=self.epsilon,
                gamma=self.gamma,
            )
            
            self._fitted_model = self._model.fit(X_norm, y_normalized)
            
            # Store diagnostics
            self._diagnostics.update({
                "n_features": len(feature_names),
                "training_samples": len(X_norm),
                "feature_names": feature_names,
                "target_mean": float(self._y_mean),
                "target_std": float(self._y_std),
                "kernel": self.kernel,
            })
            
            logger.info(
                f"SVR fitted with {len(feature_names)} features and "
                f"{len(X_norm)} training samples"
            )
            
        except Exception as e:
            logger.error(f"Failed to fit SVR model: {e}")
            raise ValueError(f"Model fitting failed: {e}") from e
        
        return self
    
    def predict(self, steps: int) -> ForecastResult:
        """Generate forecasts for specified number of steps.
        
        Args:
            steps: Number of time steps to forecast.
            
        Returns:
            ForecastResult with predictions.
            
        Raises:
            ValueError: If model is not fitted or prediction fails.
        """
        if self._fitted_model is None:
            raise ValueError("Model must be fitted before prediction")
        
        try:
            predictions = []
            last_features = self._features.iloc[-1].copy()
            last_price = self._training_data.iloc[-1]
            
            # Iterative forecasting
            for _ in range(steps):
                # Fill any NaN values in features with last price
                for col in last_features.index:
                    if pd.isna(last_features[col]):
                        last_features[col] = last_price
                
                # Create feature DataFrame for prediction
                feature_df = pd.DataFrame([last_features])
                
                # Normalize using stored parameters
                if self._normalization_params:
                    mean = self._normalization_params.get("mean", 0)
                    std = self._normalization_params.get("std", 1)
                    X_norm = (feature_df - mean) / std
                else:
                    X_norm = feature_df
                
                # Fill any remaining NaNs with 0
                X_norm = X_norm.fillna(0)
                
                # Predict (normalized)
                pred_normalized = self._fitted_model.predict(X_norm.values)[0]
                
                # Denormalize
                pred = pred_normalized * self._y_std + self._y_mean
                predictions.append(float(pred))
                
                # Update features for next iteration
                last_features["lag_1"] = pred
                last_price = pred
            
            # Create forecast dates
            last_date = self._training_data.index[-1]
            forecast_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=steps,
                freq="D",
            )
            
            # Use simple bounds based on std of recent predictions
            # Ensure margin is meaningful: at least 2% of mean prediction or 1.96 * std
            mean_pred = np.mean(predictions)
            recent_std = np.std(predictions[-3:]) if len(predictions) >= 3 else self._y_std
            min_margin = max(mean_pred * 0.02, 0.1)  # At least 2% of mean or 0.1
            margin = max(1.96 * recent_std, min_margin)
            lower_bounds = [max(0, p - margin) for p in predictions]
            upper_bounds = [p + margin for p in predictions]
            
            return ForecastResult(
                dates=[str(d) for d in forecast_dates],
                predictions=predictions,
                lower_bound=lower_bounds,
                upper_bound=upper_bounds,
                confidence_level=0.95,
                model_name="SVR",
                metadata={
                    "n_features": self._diagnostics.get("n_features", 0),
                    "kernel": self.kernel,
                    "target_std": self._diagnostics.get("target_std", 1.0),
                },
            )
        
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise ValueError(f"Prediction failed: {e}") from e
    
    def get_metadata(self) -> ModelMetadata:
        """Get model metadata.
        
        Returns:
            ModelMetadata with model details.
        """
        start_date = str(self._training_data.index[0]) if self._training_data is not None else ""
        end_date = str(self._training_data.index[-1]) if self._training_data is not None else ""
        
        return ModelMetadata(
            model_name="SVR",
            model_type="ml_enhanced",
            purity="22K",
            training_samples=len(self._training_data) if self._training_data is not None else 0,
            training_date_range=(start_date, end_date),
            trained_at=datetime.now().isoformat(),
            hyperparameters=self.hyperparameters,
            metrics=self._diagnostics,
            version=self.version,
        )
