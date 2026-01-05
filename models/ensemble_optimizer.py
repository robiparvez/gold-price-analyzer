"""Ensemble Optimizer using Optuna for hyperparameter tuning."""

import logging
import time
from dataclasses import dataclass

import numpy as np
import optuna
import pandas as pd
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.trial import Trial

from models.orchestrator import ModelOrchestrator
from models.time_series_base import ForecastResult

logger = logging.getLogger(__name__)


@dataclass
class OptimizerMetrics:
    """Metrics from optimization trial."""

    best_rmse: float
    best_trial_number: int
    best_ensemble_weights: dict
    best_ensemble_method: str
    study_direction: str
    n_trials: int
    optimization_time_seconds: float


class EnsembleOptimizer:
    """Optimizer for ensemble weights using Optuna.

    Automatically discovers optimal ensemble weights and methods
    using Bayesian optimization with early stopping.
    """

    def __init__(
        self,
        orchestrator: ModelOrchestrator,
        n_trials: int = 50,
        sampler_seed: int | None = None,
        verbose: bool = False,
    ):
        """Initialize optimizer.

        Args:
            orchestrator: ModelOrchestrator instance.
            n_trials: Number of optimization trials.
            sampler_seed: Random seed for TPE sampler.
            verbose: Whether to print optimization progress.
        """
        self.orchestrator = orchestrator
        self.n_trials = n_trials
        self.sampler_seed = sampler_seed
        self.verbose = verbose

        self.best_weights: dict[str, float] = {}
        self.best_rmse = float("inf")
        self.best_method = "weighted_mean"
        self.study: optuna.Study | None = None
        self.logger = logging.getLogger(__name__)

    def _calculate_rmse(self, y_true: pd.Series, y_pred: list) -> float:
        """Calculate RMSE between predictions and actual values.

        Args:
            y_true: Actual values.
            y_pred: Predicted values.

        Returns:
            RMSE value.
        """
        if len(y_true) < len(y_pred):
            # Use all available actual values
            y_true_subset = y_true.values[-len(y_pred) :]
        else:
            y_true_subset = y_true.values[-len(y_pred) :]

        return float(np.sqrt(np.mean((y_true_subset - np.array(y_pred)) ** 2)))

    def _objective(self, trial: Trial, X_val: pd.DataFrame, y_val: pd.Series) -> float:
        """Objective function for Optuna optimization.

        Args:
            trial: Optuna trial object.
            X_val: Validation features.
            y_val: Validation targets.

        Returns:
            RMSE to minimize.
        """
        # Get all model predictions
        all_predictions = self.orchestrator.predict_all(steps=7)
        valid_models = {
            name: pred for name, pred in all_predictions.items() if pred is not None
        }

        if not valid_models:
            return float("inf")

        # Try different ensemble methods
        method = trial.suggest_categorical(
            "method", ["weighted_mean", "equal_weight", "median"]
        )

        # For weighted_mean, optimize weights
        if method == "weighted_mean":
            # Suggest weights for each model
            weights = {}
            for model_name in valid_models.keys():
                weights[model_name] = trial.suggest_float(
                    f"weight_{model_name}",
                    0.0,
                    1.0,
                )

            # Normalize weights
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {name: w / total_weight for name, w in weights.items()}
            else:
                # Equal weights if all are 0
                weights = {name: 1.0 / len(weights) for name in weights.keys()}

        else:
            # Equal or median weights don't need optimization
            weights = {name: 1.0 / len(valid_models) for name in valid_models.keys()}

        # Calculate ensemble prediction
        ensemble_preds = []
        for step_idx in range(7):
            if method == "median":
                step_values = [
                    valid_models[name].predictions[step_idx]
                    for name in valid_models.keys()
                ]
                ensemble_preds.append(float(np.median(step_values)))
            else:
                step_value = sum(
                    weights[name] * valid_models[name].predictions[step_idx]
                    for name in valid_models.keys()
                )
                ensemble_preds.append(float(step_value))

        # Calculate RMSE on validation set
        rmse = self._calculate_rmse(y_val, ensemble_preds)

        return rmse

    def optimize(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> OptimizerMetrics:
        """Run optimization.

        Args:
            X_train: Training features.
            y_train: Training targets.
            X_val: Validation features.
            y_val: Validation targets.

        Returns:
            OptimizerMetrics with results.
        """
        start_time = time.time()

        # Train orchestrator on training data
        logger.info("Training orchestrator...")
        self.orchestrator.fit_all(X_train, y_train)

        # Create Optuna study
        sampler = TPESampler(seed=self.sampler_seed)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=2)

        self.study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
            pruner=pruner,
        )

        # Run optimization
        logger.info(f"Starting optimization with {self.n_trials} trials...")
        if self.study is None:
            raise RuntimeError("Study not initialized")

        self.study.optimize(
            lambda trial: self._objective(trial, X_val, y_val),
            n_trials=self.n_trials,
            show_progress_bar=self.verbose,
        )

        # Extract best results
        best_trial = self.study.best_trial
        self.best_rmse = best_trial.value
        self.best_method = best_trial.params.get("method", "weighted_mean")

        # Reconstruct best weights
        self.best_weights = {}
        for param_name, param_value in best_trial.params.items():
            if param_name.startswith("weight_"):
                model_name = param_name.replace("weight_", "")
                self.best_weights[model_name] = param_value

        # Normalize weights if they exist
        if self.best_weights:
            total = sum(self.best_weights.values())
            if total > 0:
                self.best_weights = {
                    name: w / total for name, w in self.best_weights.items()
                }

        elapsed_time = time.time() - start_time

        metrics = OptimizerMetrics(
            best_rmse=self.best_rmse,
            best_trial_number=best_trial.number,
            best_ensemble_weights=self.best_weights.copy(),
            best_ensemble_method=self.best_method,
            study_direction="minimize",
            n_trials=self.n_trials,
            optimization_time_seconds=elapsed_time,
        )

        logger.info(f"Optimization complete! Best RMSE: {self.best_rmse:.4f}")
        logger.info(f"Best ensemble method: {self.best_method}")
        logger.info(f"Best weights: {self.best_weights}")

        return metrics

    def get_optimized_ensemble_forecast(self, steps: int = 7) -> ForecastResult:
        """Get ensemble forecast using optimized weights.

        Args:
            steps: Number of steps ahead to forecast.

        Returns:
            ForecastResult with optimized ensemble predictions.
        """
        if not self.best_weights and self.best_method == "median":
            # Use orchestrator's ensemble with median method
            forecast = self.orchestrator.get_ensemble_forecast(
                steps=steps, method="median"
            )
            # Add best_rmse to metadata
            forecast.metadata["best_rmse"] = self.best_rmse
            forecast.metadata["best_method"] = self.best_method
            return forecast

        # Get predictions
        all_predictions = self.orchestrator.predict_all(steps=steps)
        valid_models = {
            name: pred for name, pred in all_predictions.items() if pred is not None
        }

        if not valid_models:
            raise ValueError("No valid model predictions")

        # Use best weights
        weights = (
            self.best_weights
            if self.best_weights
            else {name: 1.0 / len(valid_models) for name in valid_models.keys()}
        )

        # Generate ensemble
        ensemble_preds = []
        for step_idx in range(steps):
            if self.best_method == "median":
                step_values = [
                    valid_models[name].predictions[step_idx]
                    for name in valid_models.keys()
                ]
                ensemble_preds.append(float(np.median(step_values)))
            else:
                step_value = sum(
                    weights.get(name, 0) * valid_models[name].predictions[step_idx]
                    for name in valid_models.keys()
                )
                ensemble_preds.append(float(step_value))

        # Confidence intervals
        ensemble_std = np.std(ensemble_preds)
        min_margin = max(np.mean(ensemble_preds) * 0.02, 0.1)
        margin = float(max(1.96 * ensemble_std, min_margin))

        # Get dates from first result
        first_result = next(iter(valid_models.values()))

        return ForecastResult(
            dates=first_result.dates,
            predictions=ensemble_preds,
            lower_bound=[float(max(0, p - margin)) for p in ensemble_preds],
            upper_bound=[float(p + margin) for p in ensemble_preds],
            confidence_level=0.95,
            model_name="OptimizedEnsemble",
            metadata={
                "best_rmse": self.best_rmse,
                "best_method": self.best_method,
                "best_weights": weights,
                "num_models": len(valid_models),
                "models_used": list(valid_models.keys()),
            },
        )

    def get_optimization_history(self) -> pd.DataFrame:
        """Get optimization trial history.

        Returns:
            DataFrame with trial results.
        """
        if self.study is None:
            return pd.DataFrame()

        trials_data = []
        for trial in self.study.trials:
            trials_data.append(
                {
                    "trial_number": trial.number,
                    "value": trial.value,
                    "params": str(trial.params),
                    "state": trial.state.name,
                }
            )

        return pd.DataFrame(trials_data)
