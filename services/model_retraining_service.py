"""Model Retraining Service - Automated model maintenance and version control.

This module provides scheduled model retraining using latest BAJUS data,
version control for model changes, and automated accuracy improvement tracking.
"""

import hashlib
import json
import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from core import DatabaseConnectionManager

logger = logging.getLogger(__name__)


@dataclass
class ModelVersion:
    """Represents a specific version of a trained model."""

    version_id: str
    model_name: str
    created_at: str
    training_samples: int
    training_date_range: tuple[str, str]
    validation_mae: float
    validation_mape: float
    validation_rmse: float
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    is_active: bool = False
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version_id": self.version_id,
            "model_name": self.model_name,
            "created_at": self.created_at,
            "training_samples": self.training_samples,
            "training_date_range": self.training_date_range,
            "validation_mae": self.validation_mae,
            "validation_mape": self.validation_mape,
            "validation_rmse": self.validation_rmse,
            "hyperparameters": self.hyperparameters,
            "is_active": self.is_active,
            "notes": self.notes,
        }


@dataclass
class RetrainingResult:
    """Result from a model retraining operation."""

    success: bool
    version_id: str | None
    previous_mae: float
    new_mae: float
    improvement: float
    improvement_percentage: float
    training_time_seconds: float
    models_trained: int
    message: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "version_id": self.version_id,
            "previous_mae": round(self.previous_mae, 2),
            "new_mae": round(self.new_mae, 2),
            "improvement": round(self.improvement, 2),
            "improvement_percentage": round(self.improvement_percentage, 2),
            "training_time_seconds": round(self.training_time_seconds, 2),
            "models_trained": self.models_trained,
            "message": self.message,
        }


class ModelRetrainingService:
    """Service for automated model retraining and version control.

    Provides:
    - Scheduled weekly model retraining
    - Model version control with rollback capability
    - Performance comparison between versions
    - Automated deployment of improved models
    """

    def __init__(
        self,
        db_path: str = "data/gold_prices.db",
        models_dir: str = "models/saved",
        min_improvement_threshold: float = 5.0,  # Minimum MAE improvement in BDT
    ):
        """Initialize the retraining service.

        Args:
            db_path: Path to DuckDB database.
            models_dir: Directory to save model files.
            min_improvement_threshold: Minimum MAE improvement required to deploy new model.
        """
        self.db_path = Path(db_path)
        self.db_manager = DatabaseConnectionManager(db_path)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.min_improvement_threshold = min_improvement_threshold
        self._init_versioning_tables()

    def _init_versioning_tables(self) -> None:
        """Initialize model versioning tables."""
        try:
            with self.db_manager.get_connection() as conn:
                # Table for model versions
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS model_versions (
                        id INTEGER PRIMARY KEY,
                        version_id TEXT NOT NULL UNIQUE,
                        model_name TEXT NOT NULL,
                        created_at TIMESTAMP NOT NULL,
                        training_samples INTEGER NOT NULL,
                        training_start_date DATE NOT NULL,
                        training_end_date DATE NOT NULL,
                        validation_mae REAL NOT NULL,
                        validation_mape REAL NOT NULL,
                        validation_rmse REAL NOT NULL,
                        hyperparameters TEXT,
                        model_path TEXT,
                        is_active BOOLEAN DEFAULT FALSE,
                        notes TEXT,
                        CONSTRAINT unique_version UNIQUE(version_id)
                    )
                """
                )

                # Table for retraining history
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS retraining_history (
                        id INTEGER PRIMARY KEY,
                        retrain_date TIMESTAMP NOT NULL,
                        trigger_type TEXT NOT NULL,
                        previous_version_id TEXT,
                        new_version_id TEXT,
                        previous_mae REAL,
                        new_mae REAL,
                        improvement REAL,
                        deployed BOOLEAN DEFAULT FALSE,
                        training_time_seconds REAL,
                        notes TEXT
                    )
                """
                )

                # Table for scheduled tasks
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS scheduled_tasks (
                        id INTEGER PRIMARY KEY,
                        task_type TEXT NOT NULL,
                        frequency TEXT NOT NULL,
                        last_run TIMESTAMP,
                        next_run TIMESTAMP,
                        is_enabled BOOLEAN DEFAULT TRUE,
                        config TEXT
                    )
                """
                )

                conn.commit()
                logger.info("Model versioning tables initialized")

        except Exception as e:
            logger.error(f"Failed to initialize versioning tables: {e}")
            raise

    def _generate_version_id(self) -> str:
        """Generate unique version ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hash_suffix = hashlib.md5(str(datetime.now().timestamp()).encode()).hexdigest()[
            :6
        ]
        return f"v{timestamp}_{hash_suffix}"

    def get_training_data(
        self,
        days: int = 180,
        purity: str = "22K",
    ) -> tuple[pd.DataFrame, pd.Series] | tuple[None, None]:
        """Get training data from database.

        Args:
            days: Number of days of historical data.
            purity: Gold purity to filter.

        Returns:
            Tuple of (features DataFrame, target Series) or (None, None).
        """
        try:
            with self.db_manager.get_connection() as conn:
                # Try historical_prices first
                df = conn.execute(
                    f"""
                    SELECT date, price_bdt_per_gram
                    FROM historical_prices
                    WHERE purity = '{purity}'
                      AND date >= CURRENT_DATE - INTERVAL '{days} days'
                    ORDER BY date
                """
                ).fetchdf()

                if df.empty:
                    # Fall back to prices table
                    df = conn.execute(
                        f"""
                        SELECT CAST(date AS DATE) as date, price_bdt_per_gram
                        FROM prices
                        WHERE purity = '{purity}'
                          AND date >= CURRENT_DATE - INTERVAL '{days} days'
                        ORDER BY date
                    """
                    ).fetchdf()

                if df.empty or len(df) < 30:
                    logger.warning(f"Insufficient training data: {len(df)} records")
                    return None, None

                # Prepare data
                df = df.drop_duplicates(subset=["date"]).sort_values("date")
                df.set_index("date", inplace=True)

                # Create features (simple for now - price history)
                X = df[["price_bdt_per_gram"]].copy()
                y = df["price_bdt_per_gram"].copy()

                return X, y

        except Exception as e:
            logger.error(f"Failed to get training data: {e}")
            return None, None

    def retrain_models(
        self,
        days: int = 180,
        purity: str = "22K",
        trigger_type: str = "manual",
        n_optimization_trials: int = 30,
    ) -> RetrainingResult:
        """Retrain all models with latest data.

        Args:
            days: Days of historical data to use.
            purity: Gold purity to train on.
            trigger_type: What triggered retraining (manual, scheduled, accuracy_drop).
            n_optimization_trials: Number of Optuna trials.

        Returns:
            RetrainingResult with training outcome.
        """
        import time

        start_time = time.time()
        version_id = self._generate_version_id()

        try:
            # Get training data
            X, y = self.get_training_data(days=days, purity=purity)

            if X is None or y is None:
                return RetrainingResult(
                    success=False,
                    version_id=None,
                    previous_mae=0,
                    new_mae=0,
                    improvement=0,
                    improvement_percentage=0,
                    training_time_seconds=time.time() - start_time,
                    models_trained=0,
                    message="Insufficient training data",
                )

            # Get current active model performance
            current_version = self.get_active_version()
            previous_mae = (
                current_version.validation_mae if current_version else float("inf")
            )

            # Split data
            split_idx = int(len(X) * 0.85)
            X_train, y_train = X[:split_idx], y[:split_idx]
            X_val, y_val = X[split_idx:], y[split_idx:]

            # Import and train models
            from services.gold_price_service import GoldPriceService

            service = GoldPriceService(cache_enabled=False)

            # Train all models
            train_results = service.train(X_train, y_train)

            # Optimize ensemble
            optimization_results = service.optimize(
                X_train, y_train, X_val, y_val, n_trials=n_optimization_trials
            )

            new_mae = optimization_results.get("best_rmse", float("inf"))

            # Calculate improvement
            improvement = previous_mae - new_mae
            improvement_pct = (
                (improvement / previous_mae * 100) if previous_mae > 0 else 0
            )

            # Determine if we should deploy
            should_deploy = improvement >= self.min_improvement_threshold

            # Save model version
            model_path = self._save_model_version(
                version_id=version_id,
                service=service,
                training_samples=len(X),
                training_range=(str(X.index.min()), str(X.index.max())),
                validation_mae=new_mae,
                validation_mape=optimization_results.get("best_rmse", 0)
                / y.mean()
                * 100,
                validation_rmse=optimization_results.get("best_rmse", 0),
                hyperparameters=optimization_results.get("best_weights", {}),
                notes=f"Retrained via {trigger_type}",
            )

            # Deploy if improved
            if should_deploy:
                self._deploy_version(version_id)
                message = (
                    f"Model improved by {improvement:.0f} BDT - deployed {version_id}"
                )
            else:
                message = f"Model did not improve enough ({improvement:.0f} BDT < {self.min_improvement_threshold} BDT threshold)"

            # Log retraining
            self._log_retraining(
                trigger_type=trigger_type,
                previous_version_id=(
                    current_version.version_id if current_version else None
                ),
                new_version_id=version_id,
                previous_mae=previous_mae,
                new_mae=new_mae,
                improvement=improvement,
                deployed=should_deploy,
                training_time=time.time() - start_time,
                notes=message,
            )

            training_time = time.time() - start_time

            return RetrainingResult(
                success=True,
                version_id=version_id,
                previous_mae=previous_mae,
                new_mae=new_mae,
                improvement=improvement,
                improvement_percentage=improvement_pct,
                training_time_seconds=training_time,
                models_trained=(
                    len(train_results) if isinstance(train_results, dict) else 9
                ),
                message=message,
            )

        except Exception as e:
            logger.error(f"Retraining failed: {e}")
            return RetrainingResult(
                success=False,
                version_id=None,
                previous_mae=0,
                new_mae=0,
                improvement=0,
                improvement_percentage=0,
                training_time_seconds=time.time() - start_time,
                models_trained=0,
                message=f"Retraining failed: {str(e)}",
            )

    def _save_model_version(
        self,
        version_id: str,
        service: Any,
        training_samples: int,
        training_range: tuple[str, str],
        validation_mae: float,
        validation_mape: float,
        validation_rmse: float,
        hyperparameters: dict,
        notes: str = "",
    ) -> str:
        """Save model version to disk and database."""
        try:
            # Save model to disk
            model_path = self.models_dir / f"{version_id}.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(
                    {
                        "orchestrator": service.orchestrator,
                        "optimizer": service.optimizer,
                        "version_id": version_id,
                        "created_at": datetime.now().isoformat(),
                    },
                    f,
                )

            # Save to database
            with self.db_manager.get_connection() as conn:
                max_id = conn.execute(
                    "SELECT COALESCE(MAX(id), 0) FROM model_versions"
                ).fetchone()[0]

                # Convert numpy types to Python types
                validation_mae_py = float(validation_mae)
                validation_mape_py = float(validation_mape)
                validation_rmse_py = float(validation_rmse)

                conn.execute(
                    """
                    INSERT INTO model_versions
                    (id, version_id, model_name, created_at, training_samples,
                     training_start_date, training_end_date, validation_mae,
                     validation_mape, validation_rmse, hyperparameters, model_path, notes)
                    VALUES (?, ?, 'ensemble', CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    [
                        max_id + 1,
                        version_id,
                        training_samples,
                        training_range[0],
                        training_range[1],
                        validation_mae_py,
                        validation_mape_py,
                        validation_rmse_py,
                        json.dumps(hyperparameters),
                        str(model_path),
                        notes,
                    ],
                )
                conn.commit()

            logger.info(f"Saved model version {version_id} to {model_path}")
            return str(model_path)

        except Exception as e:
            logger.error(f"Failed to save model version: {e}")
            raise

    def _deploy_version(self, version_id: str) -> bool:
        """Deploy a model version as active."""
        try:
            with self.db_manager.get_connection() as conn:
                # Deactivate all versions
                conn.execute("UPDATE model_versions SET is_active = FALSE")

                # Activate new version
                conn.execute(
                    "UPDATE model_versions SET is_active = TRUE WHERE version_id = ?",
                    [version_id],
                )
                conn.commit()

            logger.info(f"Deployed model version {version_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to deploy version: {e}")
            return False

    def _log_retraining(
        self,
        trigger_type: str,
        previous_version_id: str | None,
        new_version_id: str,
        previous_mae: float,
        new_mae: float,
        improvement: float,
        deployed: bool,
        training_time: float,
        notes: str,
    ) -> None:
        """Log retraining event."""
        try:
            with self.db_manager.get_connection() as conn:
                max_id = conn.execute(
                    "SELECT COALESCE(MAX(id), 0) FROM retraining_history"
                ).fetchone()[0]

                conn.execute(
                    """
                    INSERT INTO retraining_history
                    (id, retrain_date, trigger_type, previous_version_id, new_version_id,
                     previous_mae, new_mae, improvement, deployed, training_time_seconds, notes)
                    VALUES (?, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    [
                        max_id + 1,
                        trigger_type,
                        previous_version_id,
                        new_version_id,
                        float(previous_mae) if previous_mae != float("inf") else None,
                        float(new_mae),
                        float(improvement),
                        deployed,
                        float(training_time),
                        notes,
                    ],
                )
                conn.commit()

        except Exception as e:
            logger.error(f"Failed to log retraining: {e}")

    def get_active_version(self) -> ModelVersion | None:
        """Get currently active model version."""
        try:
            with self.db_manager.get_connection() as conn:
                result = conn.execute(
                    """
                    SELECT version_id, model_name, created_at, training_samples,
                           training_start_date, training_end_date, validation_mae,
                           validation_mape, validation_rmse, hyperparameters, notes
                    FROM model_versions
                    WHERE is_active = TRUE
                    LIMIT 1
                """
                ).fetchone()

                if result:
                    return ModelVersion(
                        version_id=result[0],
                        model_name=result[1],
                        created_at=str(result[2]),
                        training_samples=result[3],
                        training_date_range=(str(result[4]), str(result[5])),
                        validation_mae=result[6],
                        validation_mape=result[7],
                        validation_rmse=result[8],
                        hyperparameters=json.loads(result[9]) if result[9] else {},
                        is_active=True,
                        notes=result[10] or "",
                    )
                return None

        except Exception as e:
            logger.error(f"Failed to get active version: {e}")
            return None

    def get_version_history(self, limit: int = 10) -> list[ModelVersion]:
        """Get model version history."""
        try:
            with self.db_manager.get_connection() as conn:
                results = conn.execute(
                    f"""
                    SELECT version_id, model_name, created_at, training_samples,
                           training_start_date, training_end_date, validation_mae,
                           validation_mape, validation_rmse, hyperparameters, is_active, notes
                    FROM model_versions
                    ORDER BY created_at DESC
                    LIMIT {limit}
                """
                ).fetchall()

                versions = []
                for row in results:
                    versions.append(
                        ModelVersion(
                            version_id=row[0],
                            model_name=row[1],
                            created_at=str(row[2]),
                            training_samples=row[3],
                            training_date_range=(str(row[4]), str(row[5])),
                            validation_mae=row[6],
                            validation_mape=row[7],
                            validation_rmse=row[8],
                            hyperparameters=json.loads(row[9]) if row[9] else {},
                            is_active=row[10],
                            notes=row[11] or "",
                        )
                    )
                return versions

        except Exception as e:
            logger.error(f"Failed to get version history: {e}")
            return []

    def rollback_to_version(self, version_id: str) -> bool:
        """Rollback to a previous model version.

        Args:
            version_id: Version ID to rollback to.

        Returns:
            True if successful.
        """
        try:
            # Check version exists
            with self.db_manager.get_connection() as conn:
                result = conn.execute(
                    "SELECT model_path FROM model_versions WHERE version_id = ?",
                    [version_id],
                ).fetchone()

                if not result:
                    logger.error(f"Version {version_id} not found")
                    return False

                model_path = result[0]

                if not Path(model_path).exists():
                    logger.error(f"Model file not found: {model_path}")
                    return False

            # Deploy the version
            if self._deploy_version(version_id):
                logger.info(f"Rolled back to version {version_id}")

                # Log rollback
                self._log_retraining(
                    trigger_type="rollback",
                    previous_version_id=None,
                    new_version_id=version_id,
                    previous_mae=0,
                    new_mae=0,
                    improvement=0,
                    deployed=True,
                    training_time=0,
                    notes=f"Rollback to {version_id}",
                )
                return True

            return False

        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False

    def load_model_version(self, version_id: str | None = None) -> dict | None:
        """Load a model version from disk.

        Args:
            version_id: Version to load. Uses active version if None.

        Returns:
            Dictionary with model components or None.
        """
        try:
            if version_id is None:
                active = self.get_active_version()
                if active:
                    version_id = active.version_id
                else:
                    logger.warning("No active model version found")
                    return None

            with self.db_manager.get_connection() as conn:
                result = conn.execute(
                    "SELECT model_path FROM model_versions WHERE version_id = ?",
                    [version_id],
                ).fetchone()

                if not result:
                    logger.error(f"Version {version_id} not found")
                    return None

                model_path = Path(result[0])

                if not model_path.exists():
                    logger.error(f"Model file not found: {model_path}")
                    return None

                with open(model_path, "rb") as f:
                    return pickle.load(f)

        except Exception as e:
            logger.error(f"Failed to load model version: {e}")
            return None

    def get_retraining_schedule(self) -> dict[str, Any]:
        """Get current retraining schedule."""
        try:
            with self.db_manager.get_connection() as conn:
                result = conn.execute(
                    """
                    SELECT task_type, frequency, last_run, next_run, is_enabled, config
                    FROM scheduled_tasks
                    WHERE task_type = 'model_retraining'
                """
                ).fetchone()

                if result:
                    return {
                        "task_type": result[0],
                        "frequency": result[1],
                        "last_run": str(result[2]) if result[2] else None,
                        "next_run": str(result[3]) if result[3] else None,
                        "is_enabled": result[4],
                        "config": json.loads(result[5]) if result[5] else {},
                    }
                return {"is_enabled": False, "message": "No schedule configured"}

        except Exception as e:
            logger.error(f"Failed to get retraining schedule: {e}")
            return {"error": str(e)}

    def set_retraining_schedule(
        self,
        frequency: str = "weekly",
        enabled: bool = True,
        config: dict | None = None,
    ) -> bool:
        """Set model retraining schedule.

        Args:
            frequency: Retraining frequency (daily, weekly, monthly).
            enabled: Whether schedule is enabled.
            config: Additional configuration.

        Returns:
            True if successful.
        """
        try:
            from datetime import timedelta

            # Calculate next run
            now = datetime.now()
            if frequency == "daily":
                next_run = now + timedelta(days=1)
            elif frequency == "weekly":
                next_run = now + timedelta(weeks=1)
            elif frequency == "monthly":
                next_run = now + timedelta(days=30)
            else:
                next_run = now + timedelta(weeks=1)

            with self.db_manager.get_connection() as conn:
                # Check if exists
                existing = conn.execute(
                    "SELECT id FROM scheduled_tasks WHERE task_type = 'model_retraining'"
                ).fetchone()

                if existing:
                    conn.execute(
                        """
                        UPDATE scheduled_tasks
                        SET frequency = ?, next_run = ?, is_enabled = ?, config = ?
                        WHERE task_type = 'model_retraining'
                    """,
                        [frequency, next_run, enabled, json.dumps(config or {})],
                    )
                else:
                    max_id = conn.execute(
                        "SELECT COALESCE(MAX(id), 0) FROM scheduled_tasks"
                    ).fetchone()[0]

                    conn.execute(
                        """
                        INSERT INTO scheduled_tasks
                        (id, task_type, frequency, next_run, is_enabled, config)
                        VALUES (?, 'model_retraining', ?, ?, ?, ?)
                    """,
                        [
                            max_id + 1,
                            frequency,
                            next_run,
                            enabled,
                            json.dumps(config or {}),
                        ],
                    )

                conn.commit()

            logger.info(f"Set retraining schedule: {frequency}, enabled={enabled}")
            return True

        except Exception as e:
            logger.error(f"Failed to set retraining schedule: {e}")
            return False

    def check_and_run_scheduled_tasks(self) -> list[dict]:
        """Check and run any due scheduled tasks.

        Returns:
            List of task execution results.
        """
        results = []

        try:
            with self.db_manager.get_connection() as conn:
                due_tasks = conn.execute(
                    """
                    SELECT task_type, config FROM scheduled_tasks
                    WHERE is_enabled = TRUE AND next_run <= CURRENT_TIMESTAMP
                """
                ).fetchall()

                for task_type, config_str in due_tasks:
                    config = json.loads(config_str) if config_str else {}

                    if task_type == "model_retraining":
                        result = self.retrain_models(
                            days=config.get("days", 180),
                            purity=config.get("purity", "22K"),
                            trigger_type="scheduled",
                            n_optimization_trials=config.get("n_trials", 30),
                        )
                        results.append({"task": task_type, "result": result.to_dict()})

                        # Update last_run and next_run
                        schedule = self.get_retraining_schedule()
                        self.set_retraining_schedule(
                            frequency=schedule.get("frequency", "weekly"),
                            enabled=True,
                            config=config,
                        )

        except Exception as e:
            logger.error(f"Failed to check scheduled tasks: {e}")
            results.append({"error": str(e)})

        return results

    def get_performance_comparison(self, limit: int = 5) -> pd.DataFrame:
        """Get performance comparison across recent model versions.

        Args:
            limit: Number of versions to compare.

        Returns:
            DataFrame with version comparison.
        """
        try:
            with self.db_manager.get_connection() as conn:
                df = conn.execute(
                    f"""
                    SELECT
                        version_id,
                        created_at,
                        training_samples,
                        validation_mae,
                        validation_mape,
                        validation_rmse,
                        is_active
                    FROM model_versions
                    ORDER BY created_at DESC
                    LIMIT {limit}
                """
                ).fetchdf()

                return df

        except Exception as e:
            logger.error(f"Failed to get performance comparison: {e}")
            return pd.DataFrame()
