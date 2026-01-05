"""Utility functions for business day calculations.

Handles Bangladesh gold market business days:
- Sunday through Thursday (5 business days per week)
- Excluding Fridays and Saturdays (market closed)
"""

from datetime import datetime, timedelta


def is_business_day(date: datetime) -> bool:
    """Check if a date is a business day in Bangladesh.

    Business days are Sunday through Thursday (weekday 0-4).
    Fridays (5) and Saturdays (6) are excluded.

    Args:
        date: DateTime object to check.

    Returns:
        True if the date is a business day, False otherwise.
    """
    # weekday(): Monday=0, ..., Sunday=6
    # In Bangladesh: Sunday=6 is business day, Friday=4 and Saturday=5 are not
    # Python weekday: Monday=0, Tuesday=1, ..., Sunday=6
    weekday = date.weekday()
    # Business days: Sunday(6), Monday(0), Tuesday(1), Wednesday(2), Thursday(3)
    # Non-business: Friday(4), Saturday(5)
    return weekday != 4 and weekday != 5  # Exclude Friday and Saturday


def get_next_business_day(date: datetime) -> datetime:
    """Get the next business day after the given date.

    Args:
        date: Starting date.

    Returns:
        Next business day (datetime).
    """
    current = date + timedelta(days=1)
    while not is_business_day(current):
        current += timedelta(days=1)
    return current


def generate_business_day_dates(
    start_date: datetime | None = None,
    num_days: int = 7,
    exclude_time: bool = True,
) -> list[str]:
    """Generate list of business day dates.

    Generates dates starting from current date (or specified start_date),
    skipping Fridays and Saturdays, returning exactly num_days business days.

    Args:
        start_date: Starting date. If None, uses today.
        num_days: Number of business days to generate (default: 7).
        exclude_time: If True, return dates as YYYY-MM-DD format without time.
                     If False, return full datetime strings.

    Returns:
        List of date strings in chronological order.

    Raises:
        ValueError: If num_days is less than 1.
    """
    if num_days < 1:
        raise ValueError("num_days must be at least 1")

    # Use current date if not specified
    if start_date is None:
        start_date = datetime.now()

    # Ensure start_date is a datetime
    if isinstance(start_date, str):
        start_date = datetime.fromisoformat(start_date)

    dates = []
    # Start from the day after the given date
    current = start_date

    while len(dates) < num_days:
        current = get_next_business_day(current)

        # Format date
        if exclude_time:
            date_str = current.strftime("%Y-%m-%d")
        else:
            date_str = str(current)

        dates.append(date_str)

    return dates


def validate_forecast_dates(dates: list[str]) -> dict:
    """Validate forecast dates for business day compliance.

    Checks that:
    - No dates are Fridays or Saturdays
    - All dates are in proper format (YYYY-MM-DD)
    - No dates are from previous months/years
    - Dates are in chronological order

    Args:
        dates: List of date strings to validate.

    Returns:
        Dictionary with validation results:
        {
            'is_valid': bool,
            'errors': list[str],
            'warnings': list[str],
            'statistics': {
                'total_dates': int,
                'invalid_weekend_dates': list[str],
                'dates_with_time': list[str],
                'out_of_order_indices': list[int],
            }
        }
    """
    errors = []
    warnings = []
    invalid_weekends = []
    dates_with_time = []
    out_of_order_indices = []

    if not dates:
        return {
            "is_valid": False,
            "errors": ["Empty date list provided"],
            "warnings": [],
            "statistics": {
                "total_dates": 0,
                "invalid_weekend_dates": [],
                "dates_with_time": [],
                "out_of_order_indices": [],
            },
        }

    # Parse and validate each date
    parsed_dates = []
    for i, date_str in enumerate(dates):
        try:
            # Check if date includes time component
            if "T" in date_str or " " in date_str:
                dates_with_time.append(date_str)
                # Still try to parse it
                parsed_date = datetime.fromisoformat(
                    date_str.split("T")[0].split(" ")[0]
                )
            else:
                parsed_date = datetime.strptime(date_str, "%Y-%m-%d")

            # Check for Friday (4) or Saturday (5)
            if not is_business_day(parsed_date):
                day_name = parsed_date.strftime("%A")
                invalid_weekends.append(
                    f"Date {date_str} is a {day_name} (weekend in Bangladesh)"
                )

            parsed_dates.append(parsed_date)

        except (ValueError, AttributeError) as e:
            errors.append(f"Invalid date format at index {i}: {date_str} ({str(e)})")

    # Check chronological order
    for i in range(1, len(parsed_dates)):
        if parsed_dates[i] <= parsed_dates[i - 1]:
            out_of_order_indices.append(i)

    # Check that no dates are from previous month/year
    current_date = datetime.now()
    current_year_month = (current_date.year, current_date.month)

    for date_str, parsed_date in zip(dates, parsed_dates):
        date_year_month = (parsed_date.year, parsed_date.month)
        if date_year_month < current_year_month:
            warnings.append(f"Date {date_str} is from a previous month/year")

    if dates_with_time:
        warnings.append(
            f"Found {len(dates_with_time)} dates with time component (should be YYYY-MM-DD only)"
        )

    is_valid = len(errors) == 0 and len(invalid_weekends) == 0

    return {
        "is_valid": is_valid,
        "errors": errors + invalid_weekends,
        "warnings": warnings,
        "statistics": {
            "total_dates": len(dates),
            "invalid_weekend_dates": invalid_weekends,
            "dates_with_time": dates_with_time,
            "out_of_order_indices": out_of_order_indices,
        },
    }


def format_forecast_date(date: datetime, include_day_name: bool = False) -> str:
    """Format a date for display in forecast tables.

    Args:
        date: DateTime object to format.
        include_day_name: If True, include day name (e.g., "2026-01-06 (Tuesday)").
                         If False, just date (e.g., "2026-01-06").

    Returns:
        Formatted date string.
    """
    if include_day_name:
        return date.strftime("%Y-%m-%d (%a)")
    return date.strftime("%Y-%m-%d")
