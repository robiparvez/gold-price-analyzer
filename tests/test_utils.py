"""Tests for utility functions."""

import pytest

from utils import format_price_bdt, format_price_bdt_short


class TestFormatPriceBDT:
    """Test cases for format_price_bdt function."""

    def test_format_with_decimals(self):
        """Test formatting with decimal places."""
        assert "100.00 BDT" in format_price_bdt(100)
        assert "1,000.00 BDT" in format_price_bdt(1000)
        assert "15,000.50 BDT" in format_price_bdt(15000.50)

    def test_format_without_decimals(self):
        """Test formatting without decimal places."""
        assert "100 BDT" in format_price_bdt(100, decimals=0)
        assert "1,000 BDT" in format_price_bdt(1000, decimals=0)
        assert "15,000 BDT" in format_price_bdt(15000.50, decimals=0)

    def test_large_numbers(self):
        """Test formatting large numbers."""
        assert "100,000" in format_price_bdt(100000)
        assert "1,000,000" in format_price_bdt(1000000)
        assert "123,456.78 BDT" in format_price_bdt(123456.78)

    def test_small_numbers(self):
        """Test formatting small numbers."""
        assert "10.00 BDT" in format_price_bdt(10)
        assert "1.50 BDT" in format_price_bdt(1.50)


class TestFormatPriceBDTShort:
    """Test cases for format_price_bdt_short function."""

    def test_less_than_thousand(self):
        """Test numbers less than 1000."""
        assert "100 BDT" in format_price_bdt_short(100)
        assert "500 BDT" in format_price_bdt_short(500.75)

    def test_thousands(self):
        """Test thousands formatting."""
        result = format_price_bdt_short(1500)
        assert "1.5K BDT" in result or "1.50K BDT" in result

        result = format_price_bdt_short(10000)
        assert "10K BDT" in result or "10.0K BDT" in result

    def test_lakhs(self):
        """Test lakhs formatting."""
        result = format_price_bdt_short(150000)
        assert "1.5L BDT" in result or "1.50L BDT" in result

        result = format_price_bdt_short(1000000)
        assert "10L BDT" in result or "10.0L BDT" in result

    def test_edge_cases(self):
        """Test edge cases."""
        assert "0 BDT" in format_price_bdt_short(0)
        result = format_price_bdt_short(999)
        assert "999 BDT" in result


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
