"""Model registry for managing and discovering forecasting models.

This module provides a centralized registry for all forecasting models,
enabling dynamic model discovery, instantiation, and metadata management.
"""

import json
import logging
from pathlib import Path
from typing import Any

from models.time_series_base import ModelMetadata

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Registry for time-series forecasting models."""

    def __init__(self, base_path: str = "ml_models") -> None:
        """Initialize the model registry.

        Args:
            base_path: Base directory for storing models
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.classical_dir = self.base_path / "classical"
        self.ml_enhanced_dir = self.base_path / "ml_enhanced"
        self.deep_learning_dir = self.base_path / "deep_learning"
        self.hybrid_dir = self.base_path / "hybrid"
        self.optuna_dir = self.base_path / "optuna_studies"

        for directory in [
            self.classical_dir,
            self.ml_enhanced_dir,
            self.deep_learning_dir,
            self.hybrid_dir,
            self.optuna_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

        self.manifest_path = self.base_path / "model_manifest.json"
        self.manifest: dict[str, dict[str, Any]] = self._load_manifest()

    def _load_manifest(self) -> dict[str, dict[str, Any]]:
        """Load model manifest from disk.

        Returns:
            Dictionary mapping model keys to metadata
        """
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load manifest: {e}. Creating new one.")
                return {}
        return {}

    def _save_manifest(self) -> None:
        """Save model manifest to disk."""
        try:
            with open(self.manifest_path, "w", encoding="utf-8") as f:
                json.dump(self.manifest, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save manifest: {e}")

    def get_model_path(self, model_type: str, model_name: str, purity: str) -> Path:
        """Get the file path for a model.

        Args:
            model_type: Type of model (classical, ml_enhanced, deep_learning, hybrid)
            model_name: Name of the model
            purity: Gold purity

        Returns:
            Path to model file
        """
        type_dirs = {
            "classical": self.classical_dir,
            "ml_enhanced": self.ml_enhanced_dir,
            "deep_learning": self.deep_learning_dir,
            "hybrid": self.hybrid_dir,
        }

        directory = type_dirs.get(model_type, self.base_path)
        return directory / f"{model_name}_{purity.lower()}"

    def register_model(self, metadata: ModelMetadata, model_path: str) -> None:
        """Register a trained model in the manifest.

        Args:
            metadata: Model metadata
            model_path: Path to saved model file
        """
        key = f"{metadata.model_name}_{metadata.purity}"

        # Update metadata with path
        metadata.model_path = model_path

        # Store in manifest
        self.manifest[key] = metadata.to_dict()
        self._save_manifest()

        logger.info(f"Registered model: {key} at {model_path}")

    def get_model_metadata(self, model_name: str, purity: str) -> ModelMetadata | None:
        """Get metadata for a registered model.

        Args:
            model_name: Name of the model
            purity: Gold purity

        Returns:
            ModelMetadata if found, None otherwise
        """
        key = f"{model_name}_{purity}"
        data = self.manifest.get(key)

        if data:
            return ModelMetadata.from_dict(data)
        return None

    def list_models(
        self, purity: str | None = None, model_type: str | None = None
    ) -> list[ModelMetadata]:
        """List all registered models with optional filtering.

        Args:
            purity: Filter by purity (e.g., "22K")
            model_type: Filter by model type

        Returns:
            List of ModelMetadata objects
        """
        models = []

        for key, data in self.manifest.items():
            metadata = ModelMetadata.from_dict(data)

            # Apply filters
            if purity and metadata.purity != purity:
                continue
            if model_type and metadata.model_type != model_type:
                continue

            models.append(metadata)

        return models

    def delete_model(self, model_name: str, purity: str) -> bool:
        """Delete a model from the registry.

        Args:
            model_name: Name of the model
            purity: Gold purity

        Returns:
            True if deleted, False if not found
        """
        key = f"{model_name}_{purity}"

        if key in self.manifest:
            metadata = self.manifest[key]

            # Delete model file if exists
            if metadata.get("model_path"):
                try:
                    path = Path(metadata["model_path"])
                    if path.exists():
                        if path.is_dir():
                            import shutil

                            shutil.rmtree(path)
                        else:
                            path.unlink()
                    logger.info(f"Deleted model file: {path}")
                except Exception as e:
                    logger.warning(f"Failed to delete model file: {e}")

            # Remove from manifest
            del self.manifest[key]
            self._save_manifest()

            logger.info(f"Deleted model: {key}")
            return True

        return False

    def cleanup_old_models(self, keep_latest: int = 2) -> None:
        """Remove old model versions, keeping only the latest N versions per model type.

        Args:
            keep_latest: Number of latest models to keep per (model_name, purity) pair
        """
        # Group models by (model_name, purity)
        model_groups: dict[str, list[ModelMetadata]] = {}

        for key, data in self.manifest.items():
            metadata = ModelMetadata.from_dict(data)
            group_key = f"{metadata.model_name}_{metadata.purity}"

            if group_key not in model_groups:
                model_groups[group_key] = []
            model_groups[group_key].append(metadata)

        # Sort each group by trained_at and delete old ones
        for group_key, models in model_groups.items():
            if len(models) <= keep_latest:
                continue

            # Sort by training date (newest first)
            sorted_models = sorted(models, key=lambda m: m.trained_at, reverse=True)

            # Delete old models
            for old_model in sorted_models[keep_latest:]:
                self.delete_model(old_model.model_name, old_model.purity)
                logger.info(
                    f"Cleaned up old model: {old_model.model_name}_{old_model.purity}"
                )

    def get_best_model(
        self, purity: str, metric: str = "mae", model_type: str | None = None
    ) -> ModelMetadata | None:
        """Get the best performing model for a given purity.

        Args:
            purity: Gold purity
            metric: Metric to use for comparison (mae, rmse, mape, r2)
            model_type: Optional filter by model type

        Returns:
            Best model metadata or None
        """
        models = self.list_models(purity=purity, model_type=model_type)

        if not models:
            return None

        # Filter models that have the metric
        models_with_metric = [m for m in models if metric in m.metrics]

        if not models_with_metric:
            return None

        # For r2, higher is better; for others, lower is better
        reverse = metric == "r2"

        best_model = sorted(
            models_with_metric, key=lambda m: m.metrics[metric], reverse=reverse
        )[0]

        return best_model

    def get_storage_stats(self) -> dict[str, Any]:
        """Get storage statistics for models.

        Returns:
            Dictionary with storage information
        """
        total_size = 0
        model_count = len(self.manifest)

        for key, data in self.manifest.items():
            model_path = data.get("model_path")
            if model_path:
                path = Path(model_path)
                if path.exists():
                    if path.is_dir():
                        # Sum all files in directory
                        total_size += sum(
                            f.stat().st_size for f in path.rglob("*") if f.is_file()
                        )
                    else:
                        total_size += path.stat().st_size

        return {
            "total_models": model_count,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "models_by_type": self._count_models_by_type(),
            "models_by_purity": self._count_models_by_purity(),
        }

    def _count_models_by_type(self) -> dict[str, int]:
        """Count models by type."""
        counts: dict[str, int] = {}
        for data in self.manifest.values():
            model_type = data.get("model_type", "unknown")
            counts[model_type] = counts.get(model_type, 0) + 1
        return counts

    def _count_models_by_purity(self) -> dict[str, int]:
        """Count models by purity."""
        counts: dict[str, int] = {}
        for data in self.manifest.values():
            purity = data.get("purity", "unknown")
            counts[purity] = counts.get(purity, 0) + 1
        return counts


# Global registry instance
_registry: ModelRegistry | None = None


def get_registry(base_path: str = "ml_models") -> ModelRegistry:
    """Get the global model registry instance.

    Args:
        base_path: Base directory for models

    Returns:
        ModelRegistry instance
    """
    global _registry
    if _registry is None:
        _registry = ModelRegistry(base_path)
    return _registry
