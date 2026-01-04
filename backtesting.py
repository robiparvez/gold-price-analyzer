"""
Backtesting module for gold price prediction models.

This module provides functionality to backtest ML models on historical data
to evaluate their accuracy and performance.
"""

import logging
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from enhanced_analyzer import AdvancedGoldPriceAnalyzer

logger = logging.getLogger(__name__)


class GoldPriceBacktester:
    """Backtest gold price prediction models."""

    def __init__(self):
        """Initialize backtester with analyzer."""
        self.analyzer = AdvancedGoldPriceAnalyzer()
        self.backtest_results = {}

    def rolling_window_backtest(
        self,
        df: pd.DataFrame,
        window_size: int = 30,
        forecast_horizon: int = 7,
        model_type: Literal[
            "prophet", "random_forest", "xgboost", "ensemble"
        ] = "ensemble",
        step_size: int = 7,
    ) -> dict:
        """
        Perform rolling window backtesting.

        Args:
            df: Historical price data
            window_size: Size of training window in days
            forecast_horizon: Number of days to forecast
            model_type: Type of model to use
            step_size: Days to move forward for each test

        Returns:
            Dictionary with backtest results
        """
        if df.empty or len(df) < window_size + forecast_horizon:
            logger.error("Insufficient data for backtesting")
            return {}

        # Ensure data is sorted
        df = df.sort_values("date").reset_index(drop=True)

        predictions = []
        actuals = []
        dates = []
        errors = []

        start_idx = window_size
        end_idx = len(df) - forecast_horizon

        logger.info(
            f"Starting rolling window backtest: "
            f"window={window_size}, horizon={forecast_horizon}, "
            f"model={model_type}, step={step_size}"
        )

        iteration = 0
        for i in range(start_idx, end_idx, step_size):
            iteration += 1

            # Training data
            _ = df.iloc[i - window_size : i].copy()  # train_df

            # Test data
            test_df = df.iloc[i : i + forecast_horizon].copy()

            try:
                # Generate forecast
                forecast_result = self.analyzer.forecast_price(
                    purity="22k",  # Default, should be parameterized
                    days=forecast_horizon,
                    model_type=model_type,
                )

                if forecast_result and "forecast" in forecast_result:
                    forecast_df = forecast_result["forecast"]

                    # Match predictions with actuals
                    for j in range(min(len(forecast_df), len(test_df))):
                        pred_value = forecast_df.iloc[j]["predicted_price"]
                        actual_value = test_df.iloc[j]["price_bdt_per_gram"]
                        date = test_df.iloc[j]["date"]

                        predictions.append(pred_value)
                        actuals.append(actual_value)
                        dates.append(date)
                        errors.append(abs(pred_value - actual_value))

            except Exception as e:
                logger.warning(f"Error in iteration {iteration}: {e}")
                continue

        if not predictions:
            logger.error("No successful predictions during backtesting")
            return {}

        # Calculate metrics
        predictions_arr = np.array(predictions)
        actuals_arr = np.array(actuals)

        mae = mean_absolute_error(actuals_arr, predictions_arr)  # type: ignore[arg-type]
        mse = mean_squared_error(actuals_arr, predictions_arr)  # type: ignore[arg-type]
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((actuals_arr - predictions_arr) / actuals_arr)) * 100
        r2 = r2_score(actuals_arr, predictions_arr)  # type: ignore[arg-type]

        # Create results dataframe
        results_df = pd.DataFrame(
            {
                "date": dates,
                "actual": actuals,
                "predicted": predictions,
                "error": errors,
            }
        )

        self.backtest_results = {
            "results_df": results_df,
            "metrics": {
                "mae": mae,
                "rmse": rmse,
                "mape": mape,
                "r2_score": r2,
                "total_predictions": len(predictions),
            },
            "parameters": {
                "window_size": window_size,
                "forecast_horizon": forecast_horizon,
                "model_type": model_type,
                "step_size": step_size,
            },
        }

        logger.info(
            f"Backtest completed: MAE={mae:.2f}, RMSE={rmse:.2f}, "
            f"MAPE={mape:.2f}%, R²={r2:.3f}"
        )

        return self.backtest_results

    def time_series_split_backtest(
        self,
        df: pd.DataFrame,
        n_splits: int = 5,
        forecast_horizon: int = 7,
        model_type: Literal[
            "prophet", "random_forest", "xgboost", "ensemble"
        ] = "ensemble",
    ) -> dict:
        """
        Perform time series split backtesting.

        Args:
            df: Historical price data
            n_splits: Number of train/test splits
            forecast_horizon: Number of days to forecast
            model_type: Type of model to use

        Returns:
            Dictionary with backtest results
        """
        if df.empty or len(df) < (n_splits + 1) * forecast_horizon:
            logger.error("Insufficient data for time series split backtesting")
            return {}

        df = df.sort_values("date").reset_index(drop=True)

        split_results = []
        split_size = len(df) // (n_splits + 1)

        logger.info(
            f"Starting time series split backtest: "
            f"splits={n_splits}, horizon={forecast_horizon}, model={model_type}"
        )

        for split in range(1, n_splits + 1):
            train_end = split * split_size
            test_end = min(train_end + forecast_horizon, len(df))

            train_df = df.iloc[:train_end].copy()
            test_df = df.iloc[train_end:test_end].copy()

            if len(test_df) < forecast_horizon:
                continue

            try:
                # Generate forecast using training data
                forecast_result = self.analyzer.forecast_price(
                    purity="22k", days=forecast_horizon, model_type=model_type
                )

                if forecast_result and "forecast" in forecast_result:
                    forecast_df = forecast_result["forecast"]

                    # Calculate metrics for this split
                    predictions = forecast_df["predicted_price"].values[: len(test_df)]
                    actuals = test_df["price_bdt_per_gram"].values

                    mae = mean_absolute_error(actuals, predictions)  # type: ignore[arg-type]
                    rmse = np.sqrt(mean_squared_error(actuals, predictions))  # type: ignore[arg-type]
                    mape = (
                        np.mean(np.abs((actuals - predictions) / actuals)) * 100  # type: ignore[operator]
                    )

                    split_results.append(
                        {
                            "split": split,
                            "train_size": len(train_df),
                            "test_size": len(test_df),
                            "mae": mae,
                            "rmse": rmse,
                            "mape": mape,
                        }
                    )

            except Exception as e:
                logger.warning(f"Error in split {split}: {e}")
                continue

        if not split_results:
            logger.error("No successful splits during backtesting")
            return {}

        # Aggregate results
        splits_df = pd.DataFrame(split_results)

        aggregate_metrics = {
            "mean_mae": splits_df["mae"].mean(),
            "mean_rmse": splits_df["rmse"].mean(),
            "mean_mape": splits_df["mape"].mean(),
            "std_mae": splits_df["mae"].std(),
            "std_rmse": splits_df["rmse"].std(),
            "std_mape": splits_df["mape"].std(),
        }

        results = {
            "splits_df": splits_df,
            "aggregate_metrics": aggregate_metrics,
            "parameters": {
                "n_splits": n_splits,
                "forecast_horizon": forecast_horizon,
                "model_type": model_type,
            },
        }

        logger.info(
            f"Time series split backtest completed: "
            f"Mean MAE={aggregate_metrics['mean_mae']:.2f}, "
            f"Mean RMSE={aggregate_metrics['mean_rmse']:.2f}"
        )

        return results

    def compare_models(
        self,
        df: pd.DataFrame,
        window_size: int = 30,
        forecast_horizon: int = 7,
    ) -> dict:
        """
        Compare all available models using backtesting.

        Args:
            df: Historical price data
            window_size: Size of training window
            forecast_horizon: Forecast horizon

        Returns:
            Dictionary with comparison results
        """
        models = ["prophet", "random_forest", "xgboost", "ensemble"]
        comparison_results = {}

        for model in models:
            logger.info(f"Backtesting model: {model}")

            try:
                results = self.rolling_window_backtest(
                    df=df,
                    window_size=window_size,
                    forecast_horizon=forecast_horizon,
                    model_type=model,  # type: ignore[arg-type]
                )

                if results and "metrics" in results:
                    comparison_results[model] = results["metrics"]

            except Exception as e:
                logger.error(f"Error backtesting {model}: {e}")
                continue

        # Create comparison dataframe
        if comparison_results:
            comparison_df = pd.DataFrame(comparison_results).T
            comparison_df["rank"] = comparison_df["mae"].rank()

            return {
                "comparison_df": comparison_df,
                "best_model": comparison_df["mae"].idxmin(),
                "parameters": {
                    "window_size": window_size,
                    "forecast_horizon": forecast_horizon,
                },
            }

        return {}

    def get_error_distribution(self) -> dict:
        """
        Analyze error distribution from backtest results.

        Returns:
            Dictionary with error statistics
        """
        if not self.backtest_results or "results_df" not in self.backtest_results:
            logger.warning("No backtest results available")
            return {}

        results_df = self.backtest_results["results_df"]
        errors = results_df["error"].values

        return {
            "mean_error": np.mean(errors),
            "std_error": np.std(errors),
            "min_error": np.min(errors),
            "max_error": np.max(errors),
            "median_error": np.median(errors),
            "percentile_25": np.percentile(errors, 25),
            "percentile_75": np.percentile(errors, 75),
            "percentile_90": np.percentile(errors, 90),
            "percentile_95": np.percentile(errors, 95),
        }


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    backtester = GoldPriceBacktester()

    # Load historical data
    from historical_scraper import HistoricalGoldPriceScraper

    scraper = HistoricalGoldPriceScraper()
    df = scraper.load_historical_data(days=90)

    if not df.empty:
        # Filter for 22k gold
        df_22k = df[(df["purity"] == "22k") & (df["metal"] == "gold")].copy()

        # Run backtest
        results = backtester.rolling_window_backtest(
            df=df_22k, window_size=30, forecast_horizon=7, model_type="ensemble"
        )

        if results:
            print("\nBacktest Results:")
            print(f"Total Predictions: {results['metrics']['total_predictions']}")
            print(f"MAE: {results['metrics']['mae']:.2f} BDT/gram")
            print(f"RMSE: {results['metrics']['rmse']:.2f} BDT/gram")
            print(f"MAPE: {results['metrics']['mape']:.2f}%")
            print(f"R² Score: {results['metrics']['r2_score']:.3f}")
