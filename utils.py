"""
Utility functions for the Gold Price Analyzer
"""

import locale


def format_price_bdt(price: int | float, decimals: int = 2) -> str:
    """
    Format a price value in Bangladeshi Taka (BDT) format.

    Args:
        price: The price value to format
        decimals: Number of decimal places (default: 2)

    Returns:
        Formatted price string with BDT currency indicator

    Examples:
        >>> format_price_bdt(15000.50)
        '15,000.50 BDT'
        >>> format_price_bdt(15000, decimals=0)
        '15,000 BDT'
    """
    try:
        # Set locale for proper number formatting
        # Try Bangladesh locale first, fall back to C locale with comma separators
        try:
            locale.setlocale(locale.LC_NUMERIC, "bn_BD")
        except locale.Error:
            # Fallback to C locale with manual comma formatting
            locale.setlocale(locale.LC_NUMERIC, "C")

        # Format the number with comma separators
        if decimals == 0:
            formatted_number = locale.format_string("%.0f", price, grouping=True)
        else:
            formatted_number = locale.format_string(
                f"%.{decimals}f", price, grouping=True
            )

        return f"{formatted_number} BDT"

    except (locale.Error, ValueError, TypeError):
        # Fallback formatting if locale fails
        formatted_number = f"{price:,.{decimals}f}" if decimals > 0 else f"{price:,.0f}"
        return f"{formatted_number} BDT"


def format_price_bdt_short(price: int | float) -> str:
    """
    Format a price value in compact BDT format for charts and small displays.

    Args:
        price: The price value to format

    Returns:
        Formatted price string in compact form

    Examples:
        >>> format_price_bdt_short(15000.50)
        '15K BDT'
        >>> format_price_bdt_short(1500000)
        '1.5M BDT'
    """
    if price >= 1_000_000:
        return f"{price / 1_000_000:.1f}M BDT"
    elif price >= 1_000:
        return f"{price / 1_000:.0f}K BDT"
    else:
        return f"{price:.0f} BDT"
