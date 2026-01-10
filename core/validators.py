"""Input validation utilities for the Gold Price Analyzer.

This module provides reusable validation functions for common inputs
like prices, quantities, dates, and other domain-specific values.
"""

from datetime import date, datetime
from typing import Any

from core.constants import PURITIES


def validate_price(
    value: float | int | Any,
    field_name: str = "price",
    min_value: float = 0.0,
    max_value: float = 100000.0,
) -> float:
    """Validate a price value.

    Args:
        value: The value to validate.
        field_name: Name of the field for error messages.
        min_value: Minimum allowed value.
        max_value: Maximum allowed value.

    Returns:
        Validated float value.

    Raises:
        ValueError: If validation fails.
    """
    try:
        price = float(value)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"{field_name} must be a number, got {type(value).__name__}"
        ) from e

    if price < min_value:
        raise ValueError(f"{field_name} must be >= {min_value}, got {price}")

    if price > max_value:
        raise ValueError(f"{field_name} must be <= {max_value}, got {price}")

    return price


def validate_quantity(
    value: float | int | Any,
    field_name: str = "quantity",
    min_value: float = 0.001,
    max_value: float = 10000.0,
) -> float:
    """Validate a quantity value (grams, etc.).

    Args:
        value: The value to validate.
        field_name: Name of the field for error messages.
        min_value: Minimum allowed value.
        max_value: Maximum allowed value.

    Returns:
        Validated float value.

    Raises:
        ValueError: If validation fails.
    """
    try:
        qty = float(value)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"{field_name} must be a number, got {type(value).__name__}"
        ) from e

    if qty < min_value:
        raise ValueError(f"{field_name} must be >= {min_value}, got {qty}")

    if qty > max_value:
        raise ValueError(f"{field_name} must be <= {max_value}, got {qty}")

    return qty


def validate_purity(value: str | Any) -> str:
    """Validate a gold purity value.

    Args:
        value: The purity value to validate.

    Returns:
        Validated and normalized purity string.

    Raises:
        ValueError: If validation fails.
    """
    if not isinstance(value, str):
        raise ValueError(f"Purity must be a string, got {type(value).__name__}")

    normalized = value.strip().upper() if value else ""

    # Check for direct match
    if normalized in PURITIES:
        return normalized

    # Check for common variations
    purity_map = {
        "22K": "22K",
        "21K": "21K",
        "18K": "18K",
        "TRADITIONAL": "Traditional",
    }

    if normalized in purity_map:
        return purity_map[normalized]

    # Check for numeric prefix
    for purity in ("22K", "21K", "18K"):
        if purity[:2] in normalized:
            return purity

    valid_options = ", ".join(PURITIES)
    raise ValueError(f"Invalid purity: {value}. Valid options: {valid_options}")


def validate_days(
    value: int | Any,
    field_name: str = "days",
    min_value: int = 1,
    max_value: int = 365,
) -> int:
    """Validate a number of days.

    Args:
        value: The value to validate.
        field_name: Name of the field for error messages.
        min_value: Minimum allowed value.
        max_value: Maximum allowed value.

    Returns:
        Validated integer value.

    Raises:
        ValueError: If validation fails.
    """
    try:
        days = int(value)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"{field_name} must be an integer, got {type(value).__name__}"
        ) from e

    if days < min_value:
        raise ValueError(f"{field_name} must be >= {min_value}, got {days}")

    if days > max_value:
        raise ValueError(f"{field_name} must be <= {max_value}, got {days}")

    return days


def validate_date(
    value: date | datetime | str | Any,
    field_name: str = "date",
    allow_future: bool = True,
    allow_past: bool = True,
) -> date:
    """Validate a date value.

    Args:
        value: The value to validate.
        field_name: Name of the field for error messages.
        allow_future: Whether to allow future dates.
        allow_past: Whether to allow past dates.

    Returns:
        Validated date object.

    Raises:
        ValueError: If validation fails.
    """
    if isinstance(value, datetime):
        validated_date = value.date()
    elif isinstance(value, date):
        validated_date = value
    elif isinstance(value, str):
        try:
            validated_date = datetime.strptime(value.strip(), "%Y-%m-%d").date()
        except ValueError:
            try:
                validated_date = datetime.fromisoformat(value.strip()).date()
            except ValueError as e:
                raise ValueError(
                    f"{field_name} must be a valid date string (YYYY-MM-DD), got {value}"
                ) from e
    else:
        raise ValueError(f"{field_name} must be a date, got {type(value).__name__}")

    today = date.today()

    if not allow_future and validated_date > today:
        raise ValueError(f"{field_name} cannot be in the future: {validated_date}")

    if not allow_past and validated_date < today:
        raise ValueError(f"{field_name} cannot be in the past: {validated_date}")

    return validated_date


def validate_string(
    value: str | Any,
    field_name: str = "value",
    min_length: int = 0,
    max_length: int = 1000,
    allow_empty: bool = False,
) -> str:
    """Validate a string value.

    Args:
        value: The value to validate.
        field_name: Name of the field for error messages.
        min_length: Minimum allowed length.
        max_length: Maximum allowed length.
        allow_empty: Whether to allow empty strings.

    Returns:
        Validated string value.

    Raises:
        ValueError: If validation fails.
    """
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string, got {type(value).__name__}")

    if not allow_empty and not value.strip():
        raise ValueError(f"{field_name} cannot be empty")

    if len(value) < min_length:
        raise ValueError(f"{field_name} must be at least {min_length} characters")

    if len(value) > max_length:
        raise ValueError(f"{field_name} must be at most {max_length} characters")

    return value


def validate_percentage(
    value: float | int | Any,
    field_name: str = "percentage",
    min_value: float = 0.0,
    max_value: float = 100.0,
) -> float:
    """Validate a percentage value.

    Args:
        value: The value to validate.
        field_name: Name of the field for error messages.
        min_value: Minimum allowed value.
        max_value: Maximum allowed value.

    Returns:
        Validated float value.

    Raises:
        ValueError: If validation fails.
    """
    try:
        pct = float(value)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"{field_name} must be a number, got {type(value).__name__}"
        ) from e

    if pct < min_value:
        raise ValueError(f"{field_name} must be >= {min_value}%, got {pct}%")

    if pct > max_value:
        raise ValueError(f"{field_name} must be <= {max_value}%, got {pct}%")

    return pct


def validate_positive_int(
    value: int | Any,
    field_name: str = "value",
    max_value: int | None = None,
) -> int:
    """Validate a positive integer.

    Args:
        value: The value to validate.
        field_name: Name of the field for error messages.
        max_value: Optional maximum allowed value.

    Returns:
        Validated integer value.

    Raises:
        ValueError: If validation fails.
    """
    try:
        val = int(value)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"{field_name} must be an integer, got {type(value).__name__}"
        ) from e

    if val <= 0:
        raise ValueError(f"{field_name} must be positive, got {val}")

    if max_value is not None and val > max_value:
        raise ValueError(f"{field_name} must be <= {max_value}, got {val}")

    return val
