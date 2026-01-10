"""Custom exceptions for the Gold Price Analyzer application.

This module provides a hierarchy of specific exceptions to enable
more precise error handling throughout the application.
"""


class GoldPriceAnalyzerError(Exception):
    """Base exception for all Gold Price Analyzer errors."""

    def __init__(self, message: str = "An error occurred in Gold Price Analyzer"):
        self.message = message
        super().__init__(self.message)


# Data-related exceptions
class DataError(GoldPriceAnalyzerError):
    """Base exception for data-related errors."""

    pass


class DataFetchError(DataError):
    """Error occurred while fetching data from an external source."""

    def __init__(self, source: str, message: str = "Failed to fetch data"):
        self.source = source
        super().__init__(f"{message} from {source}")


class DataParseError(DataError):
    """Error occurred while parsing data."""

    def __init__(self, message: str = "Failed to parse data"):
        super().__init__(message)


class DataValidationError(DataError):
    """Data failed validation checks."""

    def __init__(self, field: str, message: str = "Data validation failed"):
        self.field = field
        super().__init__(f"{message}: {field}")


class InsufficientDataError(DataError):
    """Not enough data available for the requested operation."""

    def __init__(self, required: int, available: int, operation: str = "operation"):
        self.required = required
        self.available = available
        super().__init__(
            f"Insufficient data for {operation}. "
            f"Required: {required}, Available: {available}"
        )


# Database-related exceptions
class DatabaseError(GoldPriceAnalyzerError):
    """Base exception for database-related errors."""

    pass


class DatabaseConnectionError(DatabaseError):
    """Failed to connect to the database."""

    def __init__(self, db_path: str, message: str = "Database connection failed"):
        self.db_path = db_path
        super().__init__(f"{message}: {db_path}")


class DatabaseQueryError(DatabaseError):
    """Error occurred while executing a database query."""

    def __init__(self, query: str = "unknown", message: str = "Query execution failed"):
        self.query = query
        super().__init__(f"{message}")


class RecordNotFoundError(DatabaseError):
    """Requested record was not found in the database."""

    def __init__(self, table: str, identifier: str):
        self.table = table
        self.identifier = identifier
        super().__init__(f"Record not found in {table}: {identifier}")


# Model-related exceptions
class ModelError(GoldPriceAnalyzerError):
    """Base exception for ML model-related errors."""

    pass


class ModelTrainingError(ModelError):
    """Error occurred during model training."""

    def __init__(self, model_name: str, message: str = "Training failed"):
        self.model_name = model_name
        super().__init__(f"Model {model_name}: {message}")


class ModelPredictionError(ModelError):
    """Error occurred during model prediction."""

    def __init__(self, model_name: str, message: str = "Prediction failed"):
        self.model_name = model_name
        super().__init__(f"Model {model_name}: {message}")


class ModelNotFoundError(ModelError):
    """Requested model was not found."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        super().__init__(f"Model not found: {model_name}")


class ModelLoadError(ModelError):
    """Error occurred while loading a saved model."""

    def __init__(self, model_path: str, message: str = "Failed to load model"):
        self.model_path = model_path
        super().__init__(f"{message}: {model_path}")


class ModelSaveError(ModelError):
    """Error occurred while saving a model."""

    def __init__(self, model_path: str, message: str = "Failed to save model"):
        self.model_path = model_path
        super().__init__(f"{message}: {model_path}")


# Service-related exceptions
class ServiceError(GoldPriceAnalyzerError):
    """Base exception for service-related errors."""

    pass


class ForecastError(ServiceError):
    """Error occurred during forecasting."""

    def __init__(self, message: str = "Forecast generation failed"):
        super().__init__(message)


class ValidationError(ServiceError):
    """Error occurred during validation."""

    def __init__(self, message: str = "Validation failed"):
        super().__init__(message)


class RetrainingError(ServiceError):
    """Error occurred during model retraining."""

    def __init__(self, message: str = "Retraining failed"):
        super().__init__(message)


# Calculation-related exceptions
class CalculationError(GoldPriceAnalyzerError):
    """Base exception for calculation-related errors."""

    pass


class PriceCalculationError(CalculationError):
    """Error occurred during price calculation."""

    def __init__(self, message: str = "Price calculation failed"):
        super().__init__(message)


class InvalidPurityError(CalculationError):
    """Invalid gold purity specified."""

    VALID_PURITIES = ["22K", "21K", "18K", "Traditional"]

    def __init__(self, purity: str):
        self.purity = purity
        super().__init__(
            f"Invalid purity: {purity}. Valid options: {', '.join(self.VALID_PURITIES)}"
        )


class InvalidJewelryTypeError(CalculationError):
    """Invalid jewelry type specified."""

    VALID_TYPES = [
        "ring",
        "necklace",
        "bracelet",
        "earrings",
        "chain",
        "bangle",
        "other",
    ]

    def __init__(self, jewelry_type: str):
        self.jewelry_type = jewelry_type
        super().__init__(
            f"Invalid jewelry type: {jewelry_type}. "
            f"Valid options: {', '.join(self.VALID_TYPES)}"
        )


# Scraping-related exceptions
class ScrapingError(GoldPriceAnalyzerError):
    """Base exception for web scraping-related errors."""

    pass


class BAJUSConnectionError(ScrapingError):
    """Failed to connect to BAJUS website."""

    def __init__(self, message: str = "Failed to connect to BAJUS"):
        super().__init__(message)


class BAJUSParseError(ScrapingError):
    """Failed to parse BAJUS website content."""

    def __init__(self, message: str = "Failed to parse BAJUS data"):
        super().__init__(message)


# Configuration-related exceptions
class ConfigurationError(GoldPriceAnalyzerError):
    """Error in application configuration."""

    def __init__(self, setting: str, message: str = "Invalid configuration"):
        self.setting = setting
        super().__init__(f"{message}: {setting}")
