"""Gold Price Service Layer - Unified API for forecasting."""

import hashlib
import logging
import pickle
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

from models.ensemble_optimizer import EnsembleOptimizer
from models.orchestrator import ModelOrchestrator
from models.time_series_base import ForecastResult

logger = logging.getLogger(__name__)


class ForecastCache:
    """Cache for storing forecast results."""

    def __init__(self, cache_dir: str = ".cache", use_sqlite: bool = True):
        """Initialize cache.

        Args:
            cache_dir: Directory for cache files.
            use_sqlite: Whether to use SQLite for caching (vs file-based).
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.use_sqlite = use_sqlite

        if use_sqlite:
            self.db_path = self.cache_dir / "forecast_cache.db"
            self._init_db()

    def _init_db(self) -> None:
        """Initialize SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS forecasts (
                    cache_key TEXT PRIMARY KEY,
                    result BLOB,
                    created_at TIMESTAMP,
                    accessed_at TIMESTAMP
                )
            """
            )
            conn.commit()

    def _get_cache_key(self, params: dict) -> str:
        """Generate cache key from parameters.

        Args:
            params: Dictionary of parameters.

        Returns:
            Cache key hash.
        """
        # Convert params to sorted JSON-like string
        param_str = str(sorted(params.items()))
        return hashlib.md5(param_str.encode()).hexdigest()

    def get(self, params: dict) -> ForecastResult | None:
        """Get cached forecast.

        Args:
            params: Parameters used for forecasting.

        Returns:
            Cached ForecastResult or None if not found.
        """
        cache_key = self._get_cache_key(params)

        if self.use_sqlite:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(
                        "SELECT result FROM forecasts WHERE cache_key = ?",
                        (cache_key,),
                    )
                    row = cursor.fetchone()
                    if row:
                        # Update accessed_at
                        conn.execute(
                            "UPDATE forecasts SET accessed_at = ? WHERE cache_key = ?",
                            (datetime.now(), cache_key),
                        )
                        conn.commit()
                        return pickle.loads(row[0])
            except Exception as e:
                logger.warning(f"Failed to retrieve from cache: {e}")

        else:
            # File-based cache
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            if cache_file.exists():
                try:
                    with open(cache_file, "rb") as f:
                        return pickle.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load cache file: {e}")

        return None

    def set(self, params: dict, result: ForecastResult) -> None:
        """Cache forecast result.

        Args:
            params: Parameters used for forecasting.
            result: ForecastResult to cache.
        """
        cache_key = self._get_cache_key(params)

        if self.use_sqlite:
            try:
                result_blob = pickle.dumps(result)
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO forecasts
                        (cache_key, result, created_at, accessed_at)
                        VALUES (?, ?, ?, ?)
                        """,
                        (cache_key, result_blob, datetime.now(), datetime.now()),
                    )
                    conn.commit()
            except Exception as e:
                logger.warning(f"Failed to cache result: {e}")

        else:
            # File-based cache
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            try:
                with open(cache_file, "wb") as f:
                    pickle.dump(result, f)
            except Exception as e:
                logger.warning(f"Failed to save cache file: {e}")

    def clear(self) -> None:
        """Clear all cached forecasts."""
        if self.use_sqlite:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("DELETE FROM forecasts")
                    conn.commit()
            except Exception as e:
                logger.warning(f"Failed to clear cache: {e}")

        else:
            # Remove all pickle files
            for cache_file in self.cache_dir.glob("*.pkl"):
                try:
                    cache_file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete cache file: {e}")


class GoldPriceService:
    """Unified service for gold price forecasting.

    Provides high-level API wrapping ModelOrchestrator and EnsembleOptimizer
    with caching, error handling, and metrics tracking.
    """

    def __init__(
        self,
        orchestrator: ModelOrchestrator | None = None,
        optimizer: EnsembleOptimizer | None = None,
        cache_enabled: bool = True,
        cache_dir: str = ".cache",
        use_sqlite_cache: bool = True,
    ):
        """Initialize service.

        Args:
            orchestrator: ModelOrchestrator instance. Created if None.
            optimizer: EnsembleOptimizer instance. Created if None.
            cache_enabled: Whether to cache forecast results.
            cache_dir: Directory for cache files.
            use_sqlite_cache: Whether to use SQLite for caching.
        """
        self.orchestrator = orchestrator or ModelOrchestrator()
        self.optimizer = optimizer or EnsembleOptimizer(self.orchestrator)

        self.cache_enabled = cache_enabled
        self.cache = (
            ForecastCache(cache_dir, use_sqlite_cache) if cache_enabled else None
        )

        # Metrics tracking
        self._metrics = {
            "forecasts_requested": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "errors": 0,
        }

        self.logger = logging.getLogger(__name__)

    def train(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Train all models.

        Args:
            X: Training features.
            y: Training targets.

        Returns:
            Dictionary with training results.
        """
        try:
            self.logger.info("Training orchestrator...")
            results = self.orchestrator.fit_all(X, y)
            self.logger.info("Orchestrator training completed")
            return results
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            self._metrics["errors"] += 1
            raise

    def optimize(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        n_trials: int = 50,
    ) -> dict:
        """Optimize ensemble.

        Args:
            X_train: Training features.
            y_train: Training targets.
            X_val: Validation features.
            y_val: Validation targets.
            n_trials: Number of optimization trials.

        Returns:
            Dictionary with optimization metrics.
        """
        try:
            self.logger.info(f"Starting optimization with {n_trials} trials...")

            # Create new optimizer with specified trials
            self.optimizer = EnsembleOptimizer(
                self.orchestrator,
                n_trials=n_trials,
            )

            metrics = self.optimizer.optimize(X_train, y_train, X_val, y_val)

            self.logger.info(
                f"Optimization completed. Best RMSE: {metrics.best_rmse:.4f}"
            )

            return {
                "best_rmse": metrics.best_rmse,
                "best_method": metrics.best_ensemble_method,
                "best_weights": metrics.best_ensemble_weights,
                "optimization_time": metrics.optimization_time_seconds,
            }
        except Exception as e:
            self.logger.error(f"Optimization failed: {e}")
            self._metrics["errors"] += 1
            raise

    def forecast(
        self,
        steps: int = 7,
        use_optimized: bool = True,
        use_cache: bool = True,
    ) -> ForecastResult:
        """Generate price forecast.

        Args:
            steps: Number of steps ahead to forecast.
            use_optimized: Whether to use optimized ensemble (requires optimize() first).
            use_cache: Whether to use cache.

        Returns:
            ForecastResult with predictions and confidence intervals.
        """
        self._metrics["forecasts_requested"] += 1

        try:
            # Check cache
            cache_key = {
                "steps": steps,
                "method": "optimized" if use_optimized else "standard",
            }

            if use_cache and self.cache_enabled:
                cached_result = self.cache.get(cache_key)
                if cached_result:
                    self.logger.info("Forecast retrieved from cache")
                    self._metrics["cache_hits"] += 1
                    return cached_result

            self._metrics["cache_misses"] += 1

            # Generate forecast
            if use_optimized:
                self.logger.info("Generating optimized ensemble forecast...")
                forecast = self.optimizer.get_optimized_ensemble_forecast(steps=steps)
            else:
                self.logger.info("Generating standard ensemble forecast...")
                forecast = self.orchestrator.get_ensemble_forecast(steps=steps)

            # Cache result
            if use_cache and self.cache_enabled:
                self.cache.set(cache_key, forecast)

            return forecast

        except Exception as e:
            self.logger.error(f"Forecasting failed: {e}")
            self._metrics["errors"] += 1
            raise

    def get_model_comparison(self) -> pd.DataFrame:
        """Get comparison of all models.

        Returns:
            DataFrame with model metrics.
        """
        try:
            return self.orchestrator.get_model_comparison()
        except Exception as e:
            self.logger.error(f"Failed to get model comparison: {e}")
            self._metrics["errors"] += 1
            raise

    def get_optimization_history(self) -> pd.DataFrame:
        """Get optimization trial history.

        Returns:
            DataFrame with trial results.
        """
        try:
            return self.optimizer.get_optimization_history()
        except Exception as e:
            self.logger.error(f"Failed to get optimization history: {e}")
            self._metrics["errors"] += 1
            raise

    def get_metrics(self) -> dict:
        """Get service metrics.

        Returns:
            Dictionary with service metrics.
        """
        total_requests = self._metrics["forecasts_requested"]
        if total_requests > 0:
            cache_hit_rate = self._metrics["cache_hits"] / total_requests * 100
        else:
            cache_hit_rate = 0.0

        return {
            "forecasts_requested": self._metrics["forecasts_requested"],
            "cache_hits": self._metrics["cache_hits"],
            "cache_misses": self._metrics["cache_misses"],
            "cache_hit_rate": cache_hit_rate,
            "errors": self._metrics["errors"],
            "cache_enabled": self.cache_enabled,
        }

    def clear_cache(self) -> None:
        """Clear all cached forecasts."""
        if self.cache_enabled:
            self.cache.clear()
            self.logger.info("Cache cleared")

    def get_service_status(self) -> dict:
        """Get service status.

        Returns:
            Dictionary with service status info.
        """
        return {
            "orchestrator_initialized": self.orchestrator is not None,
            "optimizer_initialized": self.optimizer is not None,
            "cache_enabled": self.cache_enabled,
            "metrics": self.get_metrics(),
            "timestamp": datetime.now().isoformat(),
        }
