"""Current Analysis View - Displays current price data and metrics.

This view shows the current gold price analysis including:
- Current price metrics
- 30-day price trend chart
- Statistical analysis
"""

import logging
from typing import Any

import streamlit as st

from views.base import BaseView, create_section_header

logger = logging.getLogger(__name__)


class CurrentAnalysisView(BaseView):
    """View for displaying current price analysis."""

    def __init__(
        self, analyzer: Any, load_price_data: callable, load_historical_data: callable
    ):
        """Initialize the current analysis view.

        Args:
            analyzer: AdvancedGoldPriceAnalyzer instance.
            load_price_data: Function to load current price data.
            load_historical_data: Function to load historical data.
        """
        super().__init__()
        self.analyzer = analyzer
        self.load_price_data = load_price_data
        self.load_historical_data = load_historical_data

    def render(self, settings: dict[str, Any]) -> None:
        """Render the current analysis view.

        Args:
            settings: Dictionary with user settings (metal, purity, days).
        """
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)

        with st.spinner("Loading current price data..."):
            try:
                # Load current data
                df = self.load_price_data(days=30)

                if df.empty:
                    st.error(
                        "❌ No current data available. Please check your data source or try refreshing."
                    )
                    return

                # Calculate statistics
                stats = self.analyzer.calculate_statistics(
                    df, settings["metal"], settings["purity"]
                )

                if not stats:
                    st.warning("No statistics available for selected metal and purity")
                    return

                # Display metrics
                create_section_header("Current Price Metrics", "💰")
                self._display_metric_cards(stats)

                # Load data for trend
                trend_df = self.load_historical_data(days=365)

                # Price chart
                create_section_header("Recent Price Trend (30 days)", "📊")
                trends_df = self.analyzer.calculate_trends(
                    trend_df, settings["metal"], settings["purity"]
                )

                if not trends_df.empty:
                    chart = self.analyzer.create_price_chart(
                        trend_df, settings["metal"], settings["purity"]
                    )
                    st.plotly_chart(chart, config={"responsive": True})
                    st.toast("Price trend loaded successfully!", icon="✅")
                else:
                    st.warning("No trend data available")

            except Exception as e:
                self.logger.error(f"Error loading current data: {e}")
                st.error(f"Error loading current data: {str(e)}")
                st.toast("Failed to load price data", icon="❌")

        st.markdown("</div>", unsafe_allow_html=True)

    def _display_metric_cards(self, stats: dict[str, Any]) -> None:
        """Display metric cards for price statistics.

        Args:
            stats: Dictionary with price statistics.
        """
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="Current Price",
                value=f"৳{stats.get('current_price', 0):,.0f}",
                delta=(
                    f"{stats.get('daily_change', 0):+.0f}"
                    if stats.get("daily_change")
                    else None
                ),
            )

        with col2:
            st.metric(
                label="Average Price",
                value=f"৳{stats.get('avg_price', 0):,.0f}",
            )

        with col3:
            st.metric(
                label="Max Price",
                value=f"৳{stats.get('max_price', 0):,.0f}",
            )

        with col4:
            st.metric(
                label="Min Price",
                value=f"৳{stats.get('min_price', 0):,.0f}",
            )


def render_current_analysis_tab(
    analyzer: Any,
    load_price_data: callable,
    load_historical_data: callable,
    settings: dict[str, Any],
) -> None:
    """Render the current analysis tab (functional interface).

    Args:
        analyzer: AdvancedGoldPriceAnalyzer instance.
        load_price_data: Function to load current price data.
        load_historical_data: Function to load historical data.
        settings: User settings dictionary.
    """
    view = CurrentAnalysisView(analyzer, load_price_data, load_historical_data)
    view.render(settings)
