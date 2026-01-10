"""Centralized constants and configuration for the Gold Price Analyzer.

This module consolidates all magic numbers, configuration values, and
standardized patterns used across the codebase.
"""

# =============================================================================
# Database Configuration
# =============================================================================
DATABASE_PATH = "data/gold_prices.db"
DATA_DIRECTORY = "data"

# =============================================================================
# Gold Purity Standards
# =============================================================================
# All purity values should be uppercase for consistency
PURITIES = ("22K", "21K", "18K", "Traditional")
DEFAULT_PURITY = "22K"

# Purity normalization mapping (handles common variations)
PURITY_MAPPING = {
    "22k": "22K",
    "22K": "22K",
    "22 karat": "22K",
    "22karat": "22K",
    "21k": "21K",
    "21K": "21K",
    "21 karat": "21K",
    "21karat": "21K",
    "18k": "18K",
    "18K": "18K",
    "18 karat": "18K",
    "18karat": "18K",
    "traditional": "Traditional",
    "Traditional": "Traditional",
    "সনাতন": "Traditional",
}


def normalize_purity(purity: str) -> str:
    """Normalize purity value to standard uppercase format.

    Args:
        purity: Raw purity string (e.g., "22k", "22K", "22 karat").

    Returns:
        Normalized purity string (e.g., "22K").
    """
    if not purity:
        return DEFAULT_PURITY

    # Try direct mapping first
    normalized = PURITY_MAPPING.get(purity.strip())
    if normalized:
        return normalized

    # Try lowercase lookup
    normalized = PURITY_MAPPING.get(purity.strip().lower())
    if normalized:
        return normalized

    # Fallback: check for numeric patterns
    purity_clean = purity.strip()
    if "22" in purity_clean:
        return "22K"
    elif "21" in purity_clean:
        return "21K"
    elif "18" in purity_clean:
        return "18K"
    elif "traditional" in purity_clean.lower() or "সনাতন" in purity_clean:
        return "Traditional"

    return DEFAULT_PURITY


# =============================================================================
# Forecasting Configuration
# =============================================================================
DEFAULT_FORECAST_DAYS = 7
DEFAULT_LOOKBACK_DAYS = 14
MAX_FORECAST_DAYS = 30
MIN_TRAINING_SAMPLES = 60

# ML Model Hyperparameters
LSTM_EPOCHS = 20
LSTM_BATCH_SIZE = 16
ARIMA_MAX_P = 2
ARIMA_MAX_D = 1
ARIMA_MAX_Q = 2

# =============================================================================
# Caching Configuration
# =============================================================================
CACHE_TTL_SECONDS = 300  # 5 minutes for price data
CACHE_TTL_HISTORICAL = 1800  # 30 minutes for historical data
CACHE_TTL_FORECAST = 3600  # 1 hour for forecast data

# =============================================================================
# Validation Thresholds
# =============================================================================
FORECAST_ERROR_THRESHOLD_BDT = 100.0  # Alert threshold for forecast errors
MIN_IMPROVEMENT_THRESHOLD_BDT = 5.0  # Minimum improvement for model deployment

# =============================================================================
# Network Configuration
# =============================================================================
REQUEST_TIMEOUT_SECONDS = 30
RETRY_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 2

# User agent for web scraping
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
)
