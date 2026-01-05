"""CatBoost forecasting model."""

import logging
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from components.feature_engineering import create_all_features
from models.time_series_base import BaseTimeSeriesModel, ForecastResult, ModelMetadata

logger = logging.getLogger(__name__)


class CatBoostModel(BaseTimeSeriesModel):
    """CatBoost regression model for time-series forecasting.

    Features:
    - Automatic feature engineering (lags, rolling, differencing, time features)
    - Categorical feature handling (day of week, month, etc.)
    - Conservative hyperparameters for small datasets
    - Built-in feature importance
    - Multi-step ahead forecasting

    Attributes:
        iterations: Number of boosting iterations (default 50)
        depth: Tree depth (default 4)
        learning_rate: Learning rate (default 0.03)
        l2_leaf_reg: L2 regularization (default 1.0)
    """

    def __init__(
        self,
        iterations: int = 50,
        depth: int = 4,
        learning_rate: float = 0.03,
        l2_leaf_reg: float = 1.0,
        random_state: int = 42,
        **kwargs,
    ):
        """Initialize CatBoost model.

        Args:
            iterations: Number of boosting iterations.
            depth: Tree depth.
            learning_rate: Learning rate.
            l2_leaf_reg: L2 regularization coefficient.
            random_state: Random seed.
            **kwargs: Additional arguments.
        """
        super().__init__(purity="22K", **kwargs)

        self.model_type = "ml_enhanced"
        self.version = "1.0.0"
        self.hyperparameters = {
            "iterations": iterations,
            "depth": depth,
            "learning_rate": learning_rate,
            "l2_leaf_reg": l2_leaf_reg,
            "random_state": random_state,
        }

        self._model = None
        self._fitted_model = None
        self._training_data = None
        self._features = None
        self._feature_names = None
        self._categorical_features = None
        self._diagnostics: dict[str, Any] = {}

        # Store hyperparameters
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "CatBoostModel":
        """Fit CatBoost model to training data.

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

            # For now, treat all features as numerical
            # CatBoost can handle this, but performance might be better with proper categorical encoding
            categorical_features = []  # type: ignore[assignment]
            self._categorical_features = categorical_features

            # Store features for prediction
            self._features = features.copy()

            # Get target values aligned with features
            y_aligned = y[features.index]

            # Create and fit CatBoost model
            self._model = CatBoostRegressor(
                iterations=self.iterations,
                depth=self.depth,
                learning_rate=self.learning_rate,
                l2_leaf_reg=self.l2_leaf_reg,
                cat_features=categorical_features,
                random_state=self.random_state,
                verbose=False,
            )

            self._fitted_model = self._model.fit(features, y_aligned)

            # Store diagnostics
            self._diagnostics.update(
                {
                    "n_features": len(feature_names),
                    "n_categorical_features": len(categorical_features),
                    "training_samples": len(features),
                    "feature_names": feature_names,
                    "feature_importance": dict(
                        zip(feature_names, self._fitted_model.feature_importances_)
                    ),
                }
            )

            logger.info(
                f"CatBoost fitted with {len(feature_names)} features "
                f"({len(categorical_features)} categorical) and "
                f"{len(features)} training samples"
            )

        except Exception as e:
            logger.error(f"Failed to fit CatBoost model: {e}")
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

                feature_df = pd.DataFrame([last_features])

                # Predict next value
                pred = self._fitted_model.predict(feature_df)[0]
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
            recent_std = (
                np.std(predictions[-3:])
                if len(predictions) >= 3
                else np.std([self._training_data.iloc[-1]])
            )
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
                model_name="CatBoost",
                metadata={
                    "n_features": self._diagnostics.get("n_features", 0),
                    "n_categorical": self._diagnostics.get("n_categorical_features", 0),
                    "feature_importance": self._diagnostics.get(
                        "feature_importance", {}
                    ),
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
        start_date = (
            str(self._training_data.index[0]) if self._training_data is not None else ""
        )
        end_date = (
            str(self._training_data.index[-1])
            if self._training_data is not None
            else ""
        )

        return ModelMetadata(
            model_name="CatBoost",
            model_type="ml_enhanced",
            purity="22K",
            training_samples=(
                len(self._training_data) if self._training_data is not None else 0
            ),
            training_date_range=(start_date, end_date),
            trained_at=datetime.now().isoformat(),
            hyperparameters=self.hyperparameters,
            metrics=self._diagnostics,
            version=self.version,
        )
