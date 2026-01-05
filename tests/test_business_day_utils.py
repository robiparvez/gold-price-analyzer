"""Unit tests for business day utilities."""

from datetime import datetime

import pytest

from components.business_day_utils import (
    format_forecast_date,
    generate_business_day_dates,
    get_next_business_day,
    is_business_day,
    validate_forecast_dates,
)


class TestIsBusinessDay:
    """Test business day detection."""

    def test_sunday_is_business_day(self):
        """Sunday should be a business day."""
        # January 5, 2026 is a Sunday
        date = datetime(2026, 1, 5)
        assert is_business_day(date)

    def test_monday_is_business_day(self):
        """Monday should be a business day."""
        # January 6, 2026 is a Tuesday
        date = datetime(2026, 1, 6)
        assert is_business_day(date)

    def test_wednesday_is_business_day(self):
        """Wednesday should be a business day."""
        # January 7, 2026 is a Wednesday
        date = datetime(2026, 1, 7)
        assert is_business_day(date)

    def test_thursday_is_business_day(self):
        """Thursday should be a business day."""
        # January 8, 2026 is a Thursday
        date = datetime(2026, 1, 8)
        assert is_business_day(date)

    def test_friday_is_not_business_day(self):
        """Friday should NOT be a business day."""
        # January 9, 2026 is a Friday
        date = datetime(2026, 1, 9)
        assert not is_business_day(date)

    def test_saturday_is_not_business_day(self):
        """Saturday should NOT be a business day."""
        # January 10, 2026 is a Saturday
        date = datetime(2026, 1, 10)
        assert not is_business_day(date)


class TestGetNextBusinessDay:
    """Test next business day calculation."""

    def test_sunday_to_monday(self):
        """Sunday should return Monday."""
        # January 5, 2026 is Sunday -> should return Monday Jan 6
        sunday = datetime(2026, 1, 5)
        next_day = get_next_business_day(sunday)
        assert next_day.date().isoformat() == "2026-01-06"

    def test_thursday_to_sunday(self):
        """Thursday should skip Friday/Saturday and return Sunday."""
        # January 8, 2026 is Thursday -> should return Sunday Jan 11
        thursday = datetime(2026, 1, 8)
        next_day = get_next_business_day(thursday)
        assert next_day.date().isoformat() == "2026-01-11"

    def test_friday_to_sunday(self):
        """Friday should skip Saturday and return Sunday."""
        # January 9, 2026 is Friday -> should return Sunday Jan 11
        friday = datetime(2026, 1, 9)
        next_day = get_next_business_day(friday)
        assert next_day.date().isoformat() == "2026-01-11"

    def test_saturday_to_sunday(self):
        """Saturday should return Sunday."""
        # January 10, 2026 is Saturday -> should return Sunday Jan 11
        saturday = datetime(2026, 1, 10)
        next_day = get_next_business_day(saturday)
        assert next_day.date().isoformat() == "2026-01-11"


class TestGenerateBusinessDayDates:
    """Test business day date generation."""

    def test_generate_7_business_days_from_sunday(self):
        """Generate 7 business days starting from Sunday 2026-01-05."""
        start = datetime(2026, 1, 5)  # Sunday
        dates = generate_business_day_dates(start_date=start, num_days=7)

        expected = [
            "2026-01-06",  # Tuesday
            "2026-01-07",  # Wednesday
            "2026-01-08",  # Thursday
            "2026-01-11",  # Sunday (skips Fri/Sat)
            "2026-01-12",  # Monday
            "2026-01-13",  # Tuesday
            "2026-01-14",  # Wednesday
        ]

        assert dates == expected

    def test_generate_5_business_days(self):
        """Generate 5 business days."""
        start = datetime(2026, 1, 5)
        dates = generate_business_day_dates(start_date=start, num_days=5)

        expected = [
            "2026-01-06",
            "2026-01-07",
            "2026-01-08",
            "2026-01-11",
            "2026-01-12",
        ]

        assert dates == expected

    def test_dates_in_yyyy_mm_dd_format(self):
        """Verify dates are in YYYY-MM-DD format without time."""
        start = datetime(2026, 1, 5)
        dates = generate_business_day_dates(start_date=start, num_days=3)

        for date_str in dates:
            assert len(date_str) == 10  # YYYY-MM-DD format
            assert date_str.count("-") == 2
            assert ":" not in date_str  # No time component
            assert "T" not in date_str  # No ISO format time

    def test_invalid_num_days_raises_error(self):
        """Test that num_days < 1 raises ValueError."""
        start = datetime(2026, 1, 5)

        with pytest.raises(ValueError):
            generate_business_day_dates(start_date=start, num_days=0)

        with pytest.raises(ValueError):
            generate_business_day_dates(start_date=start, num_days=-5)

    def test_no_december_dates(self):
        """Verify forecast doesn't include December 2025 dates."""
        start = datetime(2026, 1, 5)
        dates = generate_business_day_dates(start_date=start, num_days=30)

        for date_str in dates:
            year, month, day = date_str.split("-")
            assert year == "2026"
            assert month != "12"  # No December dates


class TestValidateForecastDates:
    """Test forecast date validation."""

    def test_valid_business_day_dates(self):
        """Test validation of valid business day dates."""
        dates = [
            "2026-01-06",
            "2026-01-07",
            "2026-01-08",
            "2026-01-11",
        ]

        result = validate_forecast_dates(dates)
        assert result["is_valid"]
        assert len(result["errors"]) == 0
        assert result["statistics"]["invalid_weekend_dates"] == []

    def test_invalid_friday_dates(self):
        """Test detection of Friday in dates."""
        dates = ["2026-01-06", "2026-01-09"]  # Jan 9 is Friday

        result = validate_forecast_dates(dates)
        assert not result["is_valid"]
        assert len(result["errors"]) > 0
        assert any("Friday" in error for error in result["errors"])

    def test_invalid_saturday_dates(self):
        """Test detection of Saturday in dates."""
        dates = ["2026-01-06", "2026-01-10"]  # Jan 10 is Saturday

        result = validate_forecast_dates(dates)
        assert not result["is_valid"]
        assert len(result["errors"]) > 0
        assert any("Saturday" in error for error in result["errors"])

    def test_dates_with_time_component(self):
        """Test warning for dates with time component."""
        dates = ["2026-01-06", "2026-01-07 10:00:00"]

        result = validate_forecast_dates(dates)
        assert len(result["warnings"]) > 0
        assert any("time component" in warning for warning in result["warnings"])

    def test_empty_date_list(self):
        """Test validation of empty date list."""
        result = validate_forecast_dates([])

        assert not result["is_valid"]
        assert len(result["errors"]) > 0


class TestFormatForecastDate:
    """Test date formatting."""

    def test_format_without_day_name(self):
        """Test formatting without day name."""
        date = datetime(2026, 1, 6)
        formatted = format_forecast_date(date, include_day_name=False)
        assert formatted == "2026-01-06"

    def test_format_with_day_name(self):
        """Test formatting with day name."""
        date = datetime(2026, 1, 6)  # Tuesday
        formatted = format_forecast_date(date, include_day_name=True)
        assert formatted == "2026-01-06 (Tue)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
