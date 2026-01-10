"""Forecast Validation Service - Tracks accuracy against BAJUS prices.

This module provides automated validation of forecasts against actual BAJUS prices,
calculates accuracy metrics, and triggers alerts when discrepancies exceed thresholds.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from scraper import GoldPriceScraper

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result from a single forecast validation."""

    date: str
    purity: str
    forecasted_price: float
    actual_price: float
    absolute_error: float
    percentage_error: float
    within_threshold: bool
    threshold_bdt: float = 100.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "date": self.date,
            "purity": self.purity,
            "forecasted_price": self.forecasted_price,
            "actual_price": self.actual_price,
            "absolute_error": self.absolute_error,
            "percentage_error": self.percentage_error,
            "within_threshold": self.within_threshold,
            "threshold_bdt": self.threshold_bdt,
        }


@dataclass
class ValidationMetrics:
    """Aggregate validation metrics."""

    total_validations: int = 0
    within_threshold_count: int = 0
    mae: float = 0.0  # Mean Absolute Error
    mape: float = 0.0  # Mean Absolute Percentage Error
    rmse: float = 0.0  # Root Mean Square Error
    max_error: float = 0.0
    min_error: float = float("inf")
    accuracy_rate: float = 0.0  # % within threshold
    validation_period_start: str = ""
    validation_period_end: str = ""
    errors_by_purity: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_validations": self.total_validations,
            "within_threshold_count": self.within_threshold_count,
            "mae": round(self.mae, 2),
            "mape": round(self.mape, 2),
            "rmse": round(self.rmse, 2),
            "max_error": round(self.max_error, 2),
            "min_error": (
                round(self.min_error, 2) if self.min_error != float("inf") else 0.0
            ),
            "accuracy_rate": round(self.accuracy_rate, 2),
            "validation_period_start": self.validation_period_start,
            "validation_period_end": self.validation_period_end,
            "errors_by_purity": self.errors_by_purity,
        }


class ForecastValidationService:
    """Service for validating forecasts against actual BAJUS prices.

    Provides:
    - Automated daily comparison between forecasts and actual prices
    - MAE/MAPE/RMSE tracking for accuracy improvement
    - Alerts when discrepancies exceed threshold (default: 100 BDT)
    - Historical validation data storage for trend analysis
    """

    def __init__(
        self,
        db_path: str = "data/gold_prices.db",
        threshold_bdt: float = 100.0,
        scraper: GoldPriceScraper | None = None,
    ):
        """Initialize the validation service.

        Args:
            db_path: Path to DuckDB database.
            threshold_bdt: Maximum acceptable discrepancy in BDT.
            scraper: GoldPriceScraper instance for fetching actual prices.
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        self.threshold_bdt = threshold_bdt
        self.scraper = scraper or GoldPriceScraper()
        self._init_validation_tables()

    def _init_validation_tables(self) -> None:
        """Initialize validation tracking tables."""
        try:
            with duckdb.connect(str(self.db_path)) as conn:
                # Table for storing forecasts before validation
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS forecast_log (
                        id INTEGER PRIMARY KEY,
                        forecast_date DATE NOT NULL,
                        target_date DATE NOT NULL,
                        purity TEXT NOT NULL,
                        forecasted_price REAL NOT NULL,
                        lower_bound REAL,
                        upper_bound REAL,
                        model_name TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        CONSTRAINT unique_forecast UNIQUE(forecast_date, target_date, purity)
                    )
                """
                )

                # Table for validation results
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS validation_results (
                        id INTEGER PRIMARY KEY,
                        validation_date DATE NOT NULL,
                        purity TEXT NOT NULL,
                        forecasted_price REAL NOT NULL,
                        actual_price REAL NOT NULL,
                        absolute_error REAL NOT NULL,
                        percentage_error REAL NOT NULL,
                        within_threshold BOOLEAN NOT NULL,
                        threshold_bdt REAL NOT NULL,
                        model_name TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        CONSTRAINT unique_validation UNIQUE(validation_date, purity)
                    )
                """
                )

                # Table for daily accuracy metrics
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS daily_accuracy_metrics (
                        id INTEGER PRIMARY KEY,
                        date DATE NOT NULL,
                        mae REAL NOT NULL,
                        mape REAL NOT NULL,
                        rmse REAL NOT NULL,
                        accuracy_rate REAL NOT NULL,
                        total_validations INTEGER NOT NULL,
                        within_threshold_count INTEGER NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        CONSTRAINT unique_daily_metrics UNIQUE(date)
                    )
                """
                )

                # Table for alerts
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS validation_alerts (
                        id INTEGER PRIMARY KEY,
                        alert_date TIMESTAMP NOT NULL,
                        alert_type TEXT NOT NULL,
                        purity TEXT NOT NULL,
                        discrepancy_bdt REAL NOT NULL,
                        forecasted_price REAL NOT NULL,
                        actual_price REAL NOT NULL,
                        message TEXT NOT NULL,
                        acknowledged BOOLEAN DEFAULT FALSE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """
                )

                conn.commit()
                logger.info("Validation tables initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize validation tables: {e}")
            raise

    def log_forecast(
        self,
        target_date: str,
        purity: str,
        forecasted_price: float,
        lower_bound: float | None = None,
        upper_bound: float | None = None,
        model_name: str = "ensemble",
    ) -> bool:
        """Log a forecast for later validation.

        Args:
            target_date: Date the forecast is for (YYYY-MM-DD).
            purity: Gold purity (22K, 21K, 18K, Traditional).
            forecasted_price: Predicted price in BDT.
            lower_bound: Lower confidence bound.
            upper_bound: Upper confidence bound.
            model_name: Name of the forecasting model.

        Returns:
            True if logged successfully.
        """
        try:
            with duckdb.connect(str(self.db_path)) as conn:
                # Get next ID
                max_id = conn.execute(
                    "SELECT COALESCE(MAX(id), 0) FROM forecast_log"
                ).fetchone()[0]

                conn.execute(
                    """
                    INSERT INTO forecast_log
                    (id, forecast_date, target_date, purity, forecasted_price,
                     lower_bound, upper_bound, model_name)
                    VALUES (?, CURRENT_DATE, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (forecast_date, target_date, purity) DO UPDATE SET
                        forecasted_price = EXCLUDED.forecasted_price,
                        lower_bound = EXCLUDED.lower_bound,
                        upper_bound = EXCLUDED.upper_bound,
                        model_name = EXCLUDED.model_name
                """,
                    [
                        max_id + 1,
                        target_date,
                        purity,
                        forecasted_price,
                        lower_bound,
                        upper_bound,
                        model_name,
                    ],
                )
                conn.commit()
                logger.info(
                    f"Logged forecast for {target_date} - {purity}: {forecasted_price} BDT"
                )
                return True

        except Exception as e:
            logger.error(f"Failed to log forecast: {e}")
            return False

    def fetch_actual_bajus_price(self, purity: str = "22K") -> float | None:
        """Fetch current actual price from BAJUS.

        Args:
            purity: Gold purity to fetch.

        Returns:
            Current BAJUS price or None if failed.
        """
        try:
            prices = self.scraper.get_latest_prices()
            if purity in prices:
                return prices[purity]
            logger.warning(f"No BAJUS price found for purity: {purity}")
            return None
        except Exception as e:
            logger.error(f"Failed to fetch BAJUS price: {e}")
            return None

    def validate_forecast(
        self,
        purity: str = "22K",
        forecasted_price: float | None = None,
        actual_price: float | None = None,
        model_name: str = "ensemble",
    ) -> ValidationResult | None:
        """Validate a forecast against actual BAJUS price.

        Args:
            purity: Gold purity to validate.
            forecasted_price: The forecasted price (fetches from log if None).
            actual_price: The actual BAJUS price (fetches live if None).
            model_name: Name of the model used.

        Returns:
            ValidationResult or None if validation failed.
        """
        try:
            today = datetime.now().strftime("%Y-%m-%d")

            # Get forecasted price from log if not provided
            if forecasted_price is None:
                with duckdb.connect(str(self.db_path)) as conn:
                    result = conn.execute(
                        """
                        SELECT forecasted_price FROM forecast_log
                        WHERE target_date = ? AND purity = ?
                        ORDER BY created_at DESC LIMIT 1
                    """,
                        [today, purity],
                    ).fetchone()

                    if result:
                        forecasted_price = result[0]
                    else:
                        logger.warning(f"No forecast found for {today} - {purity}")
                        return None

            # Get actual price from BAJUS if not provided
            if actual_price is None:
                actual_price = self.fetch_actual_bajus_price(purity)
                if actual_price is None:
                    return None

            # Calculate errors
            absolute_error = abs(forecasted_price - actual_price)
            percentage_error = (absolute_error / actual_price) * 100
            within_threshold = absolute_error <= self.threshold_bdt

            # Create validation result
            validation = ValidationResult(
                date=today,
                purity=purity,
                forecasted_price=forecasted_price,
                actual_price=actual_price,
                absolute_error=absolute_error,
                percentage_error=percentage_error,
                within_threshold=within_threshold,
                threshold_bdt=self.threshold_bdt,
            )

            # Store validation result
            self._store_validation_result(validation, model_name)

            # Create alert if threshold exceeded
            if not within_threshold:
                self._create_alert(validation)

            logger.info(
                f"Validation: {purity} - Forecast: {forecasted_price:.0f}, "
                f"Actual: {actual_price:.0f}, Error: {absolute_error:.0f} BDT "
                f"({'✓' if within_threshold else '✗'})"
            )

            return validation

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return None

    def _store_validation_result(
        self, validation: ValidationResult, model_name: str
    ) -> None:
        """Store validation result in database."""
        try:
            with duckdb.connect(str(self.db_path)) as conn:
                max_id = conn.execute(
                    "SELECT COALESCE(MAX(id), 0) FROM validation_results"
                ).fetchone()[0]

                conn.execute(
                    """
                    INSERT INTO validation_results
                    (id, validation_date, purity, forecasted_price, actual_price,
                     absolute_error, percentage_error, within_threshold,
                     threshold_bdt, model_name)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (validation_date, purity) DO UPDATE SET
                        forecasted_price = EXCLUDED.forecasted_price,
                        actual_price = EXCLUDED.actual_price,
                        absolute_error = EXCLUDED.absolute_error,
                        percentage_error = EXCLUDED.percentage_error,
                        within_threshold = EXCLUDED.within_threshold
                """,
                    [
                        max_id + 1,
                        validation.date,
                        validation.purity,
                        validation.forecasted_price,
                        validation.actual_price,
                        validation.absolute_error,
                        validation.percentage_error,
                        validation.within_threshold,
                        validation.threshold_bdt,
                        model_name,
                    ],
                )
                conn.commit()

        except Exception as e:
            logger.error(f"Failed to store validation result: {e}")

    def _create_alert(self, validation: ValidationResult) -> None:
        """Create alert for threshold exceedance."""
        try:
            message = (
                f"⚠️ ALERT: {validation.purity} gold price discrepancy of "
                f"{validation.absolute_error:.0f} BDT exceeds {validation.threshold_bdt} BDT threshold. "
                f"Forecast: {validation.forecasted_price:.0f} BDT, "
                f"Actual: {validation.actual_price:.0f} BDT"
            )

            with duckdb.connect(str(self.db_path)) as conn:
                max_id = conn.execute(
                    "SELECT COALESCE(MAX(id), 0) FROM validation_alerts"
                ).fetchone()[0]

                conn.execute(
                    """
                    INSERT INTO validation_alerts
                    (id, alert_date, alert_type, purity, discrepancy_bdt,
                     forecasted_price, actual_price, message)
                    VALUES (?, CURRENT_TIMESTAMP, 'THRESHOLD_EXCEEDED', ?, ?, ?, ?, ?)
                """,
                    [
                        max_id + 1,
                        validation.purity,
                        validation.absolute_error,
                        validation.forecasted_price,
                        validation.actual_price,
                        message,
                    ],
                )
                conn.commit()

            logger.warning(message)

        except Exception as e:
            logger.error(f"Failed to create alert: {e}")

    def validate_all_purities(self) -> list[ValidationResult]:
        """Validate forecasts for all gold purities.

        Returns:
            List of validation results.
        """
        purities = ["22K", "21K", "18K", "Traditional"]
        results = []

        for purity in purities:
            result = self.validate_forecast(purity=purity)
            if result:
                results.append(result)

        # Update daily metrics
        if results:
            self._update_daily_metrics(results)

        return results

    def _update_daily_metrics(self, validations: list[ValidationResult]) -> None:
        """Update daily accuracy metrics."""
        try:
            if not validations:
                return

            # Calculate metrics
            errors = [v.absolute_error for v in validations]
            pct_errors = [v.percentage_error for v in validations]

            mae = sum(errors) / len(errors)
            mape = sum(pct_errors) / len(pct_errors)
            rmse = (sum(e**2 for e in errors) / len(errors)) ** 0.5
            within_count = sum(1 for v in validations if v.within_threshold)
            accuracy_rate = (within_count / len(validations)) * 100

            today = datetime.now().strftime("%Y-%m-%d")

            with duckdb.connect(str(self.db_path)) as conn:
                max_id = conn.execute(
                    "SELECT COALESCE(MAX(id), 0) FROM daily_accuracy_metrics"
                ).fetchone()[0]

                conn.execute(
                    """
                    INSERT INTO daily_accuracy_metrics
                    (id, date, mae, mape, rmse, accuracy_rate,
                     total_validations, within_threshold_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (date) DO UPDATE SET
                        mae = EXCLUDED.mae,
                        mape = EXCLUDED.mape,
                        rmse = EXCLUDED.rmse,
                        accuracy_rate = EXCLUDED.accuracy_rate,
                        total_validations = EXCLUDED.total_validations,
                        within_threshold_count = EXCLUDED.within_threshold_count
                """,
                    [
                        max_id + 1,
                        today,
                        mae,
                        mape,
                        rmse,
                        accuracy_rate,
                        len(validations),
                        within_count,
                    ],
                )
                conn.commit()

            logger.info(
                f"Daily metrics updated: MAE={mae:.2f}, MAPE={mape:.2f}%, "
                f"Accuracy={accuracy_rate:.1f}%"
            )

        except Exception as e:
            logger.error(f"Failed to update daily metrics: {e}")

    def get_validation_metrics(self, days: int = 30) -> ValidationMetrics:
        """Get aggregate validation metrics for specified period.

        Args:
            days: Number of days to include.

        Returns:
            ValidationMetrics object.
        """
        try:
            with duckdb.connect(str(self.db_path)) as conn:
                # Get validation results
                df = conn.execute(
                    f"""
                    SELECT * FROM validation_results
                    WHERE validation_date >= CURRENT_DATE - INTERVAL '{days} days'
                    ORDER BY validation_date
                """
                ).fetchdf()

                if df.empty:
                    return ValidationMetrics()

                # Calculate metrics
                errors = df["absolute_error"].values
                pct_errors = df["percentage_error"].values

                metrics = ValidationMetrics(
                    total_validations=len(df),
                    within_threshold_count=int(df["within_threshold"].sum()),
                    mae=float(errors.mean()),
                    mape=float(pct_errors.mean()),
                    rmse=float((errors**2).mean() ** 0.5),
                    max_error=float(errors.max()),
                    min_error=float(errors.min()),
                    accuracy_rate=(df["within_threshold"].sum() / len(df)) * 100,
                    validation_period_start=str(df["validation_date"].min()),
                    validation_period_end=str(df["validation_date"].max()),
                    errors_by_purity={
                        purity: float(group["absolute_error"].mean())
                        for purity, group in df.groupby("purity")
                    },
                )

                return metrics

        except Exception as e:
            logger.error(f"Failed to get validation metrics: {e}")
            return ValidationMetrics()

    def get_recent_alerts(
        self, days: int = 7, unacknowledged_only: bool = False
    ) -> pd.DataFrame:
        """Get recent validation alerts.

        Args:
            days: Number of days to look back.
            unacknowledged_only: Only return unacknowledged alerts.

        Returns:
            DataFrame with alerts.
        """
        try:
            with duckdb.connect(str(self.db_path)) as conn:
                query = f"""
                    SELECT * FROM validation_alerts
                    WHERE alert_date >= CURRENT_DATE - INTERVAL '{days} days'
                """
                if unacknowledged_only:
                    query += " AND acknowledged = FALSE"
                query += " ORDER BY alert_date DESC"

                return conn.execute(query).fetchdf()

        except Exception as e:
            logger.error(f"Failed to get alerts: {e}")
            return pd.DataFrame()

    def acknowledge_alert(self, alert_id: int) -> bool:
        """Acknowledge an alert.

        Args:
            alert_id: ID of alert to acknowledge.

        Returns:
            True if successful.
        """
        try:
            with duckdb.connect(str(self.db_path)) as conn:
                conn.execute(
                    "UPDATE validation_alerts SET acknowledged = TRUE WHERE id = ?",
                    [alert_id],
                )
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to acknowledge alert: {e}")
            return False

    def get_validation_history(
        self, days: int = 30, purity: str | None = None
    ) -> pd.DataFrame:
        """Get validation history.

        Args:
            days: Number of days to include.
            purity: Filter by purity (optional).

        Returns:
            DataFrame with validation history.
        """
        try:
            with duckdb.connect(str(self.db_path)) as conn:
                query = f"""
                    SELECT
                        validation_date as date,
                        purity,
                        forecasted_price,
                        actual_price,
                        absolute_error,
                        percentage_error,
                        within_threshold,
                        model_name
                    FROM validation_results
                    WHERE validation_date >= CURRENT_DATE - INTERVAL '{days} days'
                """
                if purity:
                    query += f" AND purity = '{purity}'"
                query += " ORDER BY validation_date DESC"

                return conn.execute(query).fetchdf()

        except Exception as e:
            logger.error(f"Failed to get validation history: {e}")
            return pd.DataFrame()

    def generate_daily_report(self) -> dict[str, Any]:
        """Generate a daily forecast vs actual comparison report.

        Returns:
            Dictionary with report data.
        """
        today = datetime.now().strftime("%Y-%m-%d")
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

        try:
            with duckdb.connect(str(self.db_path)) as conn:
                # Get today's validations
                today_df = conn.execute(
                    """
                    SELECT * FROM validation_results
                    WHERE validation_date = ?
                """,
                    [today],
                ).fetchdf()

                # Get yesterday's validations
                yesterday_df = conn.execute(
                    """
                    SELECT * FROM validation_results
                    WHERE validation_date = ?
                """,
                    [yesterday],
                ).fetchdf()

                # Get 7-day metrics
                weekly_metrics = self.get_validation_metrics(days=7)

                # Get 30-day metrics
                monthly_metrics = self.get_validation_metrics(days=30)

                # Get recent alerts
                alerts = self.get_recent_alerts(days=1)

                report = {
                    "report_date": today,
                    "today_validations": (
                        today_df.to_dict("records") if not today_df.empty else []
                    ),
                    "yesterday_validations": (
                        yesterday_df.to_dict("records")
                        if not yesterday_df.empty
                        else []
                    ),
                    "weekly_metrics": weekly_metrics.to_dict(),
                    "monthly_metrics": monthly_metrics.to_dict(),
                    "alerts_today": len(alerts),
                    "alert_details": (
                        alerts.to_dict("records") if not alerts.empty else []
                    ),
                    "improvement_trend": self._calculate_improvement_trend(),
                }

                return report

        except Exception as e:
            logger.error(f"Failed to generate daily report: {e}")
            return {"error": str(e)}

    def _calculate_improvement_trend(self) -> dict[str, Any]:
        """Calculate if accuracy is improving over time."""
        try:
            with duckdb.connect(str(self.db_path)) as conn:
                # Get last 14 days of metrics
                df = conn.execute(
                    """
                    SELECT date, mae, accuracy_rate
                    FROM daily_accuracy_metrics
                    WHERE date >= CURRENT_DATE - INTERVAL '14 days'
                    ORDER BY date
                """
                ).fetchdf()

                if len(df) < 2:
                    return {
                        "trend": "insufficient_data",
                        "message": "Need more data for trend analysis",
                    }

                # Compare first half vs second half
                mid = len(df) // 2
                first_half_mae = df.iloc[:mid]["mae"].mean()
                second_half_mae = df.iloc[mid:]["mae"].mean()

                improvement = first_half_mae - second_half_mae
                improving = improvement > 0

                return {
                    "trend": "improving" if improving else "declining",
                    "mae_change": round(improvement, 2),
                    "first_period_mae": round(first_half_mae, 2),
                    "second_period_mae": round(second_half_mae, 2),
                    "message": (
                        f"MAE improved by {improvement:.0f} BDT"
                        if improving
                        else f"MAE increased by {abs(improvement):.0f} BDT - consider model retraining"
                    ),
                }

        except Exception as e:
            logger.error(f"Failed to calculate improvement trend: {e}")
            return {"trend": "error", "message": str(e)}
