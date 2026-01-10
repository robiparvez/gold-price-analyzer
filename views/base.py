"""Base view utilities and shared components for Streamlit views.

This module provides base classes and utilities shared across all view modules.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

import streamlit as st

logger = logging.getLogger(__name__)


class BaseView(ABC):
    """Abstract base class for view components.

    Provides common functionality for all views in the application.
    """

    def __init__(self):
        """Initialize the view."""
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def render(self, **kwargs: Any) -> None:
        """Render the view.

        Args:
            **kwargs: View-specific parameters.
        """
        pass

    def show_error(self, message: str, icon: str = "❌") -> None:
        """Display an error message.

        Args:
            message: Error message to display.
            icon: Emoji icon for the toast notification.
        """
        st.error(message)
        st.toast(message[:50], icon=icon)

    def show_success(self, message: str, icon: str = "✅") -> None:
        """Display a success message.

        Args:
            message: Success message to display.
            icon: Emoji icon for the toast notification.
        """
        st.success(message)
        st.toast(message[:50], icon=icon)

    def show_info(self, message: str) -> None:
        """Display an info message.

        Args:
            message: Info message to display.
        """
        st.info(message)

    def show_warning(self, message: str) -> None:
        """Display a warning message.

        Args:
            message: Warning message to display.
        """
        st.warning(message)


def display_metric_card(
    label: str,
    value: str | float,
    delta: float | None = None,
    delta_color: str = "normal",
) -> None:
    """Display a single metric card.

    Args:
        label: Metric label.
        value: Metric value.
        delta: Optional delta value.
        delta_color: Color for delta ("normal", "inverse", "off").
    """
    st.metric(
        label=label,
        value=value,
        delta=f"{delta:+.2f}" if delta is not None else None,
        delta_color=delta_color,
    )


def format_price_display(price: float, currency: str = "BDT") -> str:
    """Format a price for display.

    Args:
        price: Price value.
        currency: Currency code.

    Returns:
        Formatted price string.
    """
    return f"৳{price:,.0f}" if currency == "BDT" else f"{currency} {price:,.2f}"


def create_section_header(title: str, icon: str = "") -> None:
    """Create a section header with optional icon.

    Args:
        title: Section title.
        icon: Optional emoji icon.
    """
    header = f"### {icon} {title}" if icon else f"### {title}"
    st.markdown(header)


def wrap_in_tab_content(func):
    """Decorator to wrap view content in tab styling.

    Args:
        func: Function to wrap.

    Returns:
        Wrapped function.
    """

    def wrapper(*args, **kwargs):
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        result = func(*args, **kwargs)
        st.markdown("</div>", unsafe_allow_html=True)
        return result

    return wrapper
