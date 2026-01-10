"""Core module containing shared utilities, constants, and base classes."""

from core.constants import (
    CACHE_TTL_SECONDS,
    DATABASE_PATH,
    DEFAULT_FORECAST_DAYS,
    DEFAULT_LOOKBACK_DAYS,
    DEFAULT_PURITY,
    PURITIES,
    normalize_purity,
)
from core.database import DatabaseConnectionManager, get_db_manager
from core.exceptions import (
    BAJUSConnectionError,
    BAJUSParseError,
    CalculationError,
    ConfigurationError,
    DatabaseConnectionError,
    DatabaseError,
    DatabaseQueryError,
    DataError,
    DataFetchError,
    DataParseError,
    DataValidationError,
    ForecastError,
    GoldPriceAnalyzerError,
    InsufficientDataError,
    InvalidJewelryTypeError,
    InvalidPurityError,
    ModelError,
    ModelLoadError,
    ModelNotFoundError,
    ModelPredictionError,
    ModelSaveError,
    ModelTrainingError,
    PriceCalculationError,
    RecordNotFoundError,
    RetrainingError,
    ScrapingError,
    ServiceError,
    ValidationError,
)
from core.metrics import MetricsCalculator
from core.validators import (
    validate_date,
    validate_days,
    validate_percentage,
    validate_positive_int,
    validate_price,
    validate_purity,
    validate_quantity,
    validate_string,
)

__all__ = [
    # Constants
    "DATABASE_PATH",
    "DEFAULT_PURITY",
    "PURITIES",
    "DEFAULT_FORECAST_DAYS",
    "DEFAULT_LOOKBACK_DAYS",
    "CACHE_TTL_SECONDS",
    # Functions
    "normalize_purity",
    # Classes
    "DatabaseConnectionManager",
    "get_db_manager",
    "MetricsCalculator",
    # Validators
    "validate_date",
    "validate_days",
    "validate_percentage",
    "validate_positive_int",
    "validate_price",
    "validate_purity",
    "validate_quantity",
    "validate_string",
    # Base Exceptions
    "GoldPriceAnalyzerError",
    # Data Exceptions
    "DataError",
    "DataFetchError",
    "DataParseError",
    "DataValidationError",
    "InsufficientDataError",
    # Database Exceptions
    "DatabaseError",
    "DatabaseConnectionError",
    "DatabaseQueryError",
    "RecordNotFoundError",
    # Model Exceptions
    "ModelError",
    "ModelTrainingError",
    "ModelPredictionError",
    "ModelNotFoundError",
    "ModelLoadError",
    "ModelSaveError",
    # Service Exceptions
    "ServiceError",
    "ForecastError",
    "ValidationError",
    "RetrainingError",
    # Calculation Exceptions
    "CalculationError",
    "PriceCalculationError",
    "InvalidPurityError",
    "InvalidJewelryTypeError",
    # Scraping Exceptions
    "ScrapingError",
    "BAJUSConnectionError",
    "BAJUSParseError",
    # Configuration Exceptions
    "ConfigurationError",
]
