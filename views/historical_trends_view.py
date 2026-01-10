"""Historical Trends View - Tab 3 of the dashboard.

Displays historical price data with moving averages, candlestick charts,
and trading signal detection (Golden/Death Cross).
"""

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from views.base import BaseView


class HistoricalTrendsView(BaseView):
    """View for historical trends analysis."""

    def __init__(
        self,
        analyzer: Any,
        load_historical_func: Any,
        fetch_historical_func: Any,
        create_ohlc_func: Any,
        format_price_func: Any,
    ):
        """Initialize the historical trends view.

        Args:
            analyzer: AdvancedGoldPriceAnalyzer instance.
            load_historical_func: Function to load historical data.
            fetch_historical_func: Function to fetch and save historical data.
            create_ohlc_func: Function to create OHLC data.
            format_price_func: Function to format price display.
        """
        self.analyzer = analyzer
        self.load_historical_data = load_historical_func
        self.fetch_and_save_historical_data = fetch_historical_func
        self.create_ohlc_data = create_ohlc_func
        self.format_price_bdt = format_price_func

    def render(self, settings: dict[str, Any]) -> None:
        """Render the historical trends tab.

        Args:
            settings: Dashboard settings dictionary.
        """
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)

        with st.spinner("Loading historical data..."):
            try:
                self._render_historical_content(settings)
            except Exception as e:
                st.error(f"Error loading historical data: {str(e)}")

        st.markdown("</div>", unsafe_allow_html=True)

    def _render_historical_content(self, settings: dict[str, Any]) -> None:
        """Render the main historical content.

        Args:
            settings: Dashboard settings dictionary.
        """
        if not settings["use_historical"]:
            st.info("Historical data analysis is disabled. Enable it in the sidebar.")
            return

        historical_df = self.load_historical_data(settings["days"])

        if historical_df.empty:
            st.warning(
                "No historical data in local storage. Fetching from external sources..."
            )
            historical_df = self.fetch_and_save_historical_data(settings["days"])

        if historical_df.empty:
            st.error("Unable to load historical data")
            return

        # Filter by selected metal and purity
        filtered_df = historical_df[
            (historical_df["metal"] == settings["metal"])
            & (historical_df["purity"] == settings["purity"])
        ]

        if filtered_df.empty:
            st.warning(
                f"No historical data available for {settings['metal']} {settings['purity']}"
            )
            return

        self._display_historical_analysis(filtered_df, settings)

    def _display_historical_analysis(
        self, filtered_df: pd.DataFrame, settings: dict[str, Any]
    ) -> None:
        """Display the full historical analysis.

        Args:
            filtered_df: Filtered historical data.
            settings: Dashboard settings dictionary.
        """
        st.markdown(f"### 📈 Historical Trends ({settings['days']} days)")

        # Chart type selection
        chart_type = st.radio(
            "Chart Type:",
            ["Line Chart", "Candlestick Chart"],
            horizontal=True,
            help="Choose between line chart or candlestick chart for price visualization",
        )

        # Display statistics
        self._display_statistics(filtered_df)

        # Process and display chart
        processed_df = self.analyzer.preprocess_historical_data(filtered_df)

        if processed_df.empty:
            st.warning("Unable to process historical data for visualization")
            return

        self._display_chart(processed_df, settings, chart_type)
        self._display_trading_signals(processed_df)
        self._display_data_sources(filtered_df)

    def _display_statistics(self, filtered_df: pd.DataFrame) -> None:
        """Display historical statistics.

        Args:
            filtered_df: Filtered historical data.
        """
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Records", len(filtered_df))

        with col2:
            min_date = pd.to_datetime(filtered_df["date"]).min().strftime("%Y-%m-%d")
            max_date = pd.to_datetime(filtered_df["date"]).max().strftime("%Y-%m-%d")
            st.metric("Date Range", f"{min_date} to {max_date}")

        with col3:
            min_price = self.format_price_bdt(
                filtered_df["price_bdt_per_gram"].min(), decimals=0
            )
            max_price = self.format_price_bdt(
                filtered_df["price_bdt_per_gram"].max(), decimals=0
            )
            st.metric("Price Range", f"{min_price} - {max_price}")

        with col4:
            avg_price = filtered_df["price_bdt_per_gram"].mean()
            st.metric("Average Price", self.format_price_bdt(avg_price, decimals=0))

    def _display_chart(
        self, processed_df: pd.DataFrame, settings: dict[str, Any], chart_type: str
    ) -> None:
        """Display the historical price chart.

        Args:
            processed_df: Processed historical data.
            settings: Dashboard settings dictionary.
            chart_type: Type of chart to display.
        """
        fig = go.Figure()

        if chart_type == "Candlestick Chart":
            self._add_candlestick_trace(fig, processed_df)
        else:
            self._add_line_trace(fig, processed_df)

        # Add moving averages
        self._add_moving_averages(fig, processed_df)

        fig.update_layout(
            title=f"Historical Gold Prices - {settings['purity']} ({settings['days']} days)",
            xaxis_title="Date",
            yaxis_title="Price (BDT/gram)",
            yaxis=dict(tickformat=",.0f", ticksuffix=" BDT"),
            height=500,
        )

        st.plotly_chart(fig, config={"responsive": True})

    def _add_candlestick_trace(
        self, fig: go.Figure, processed_df: pd.DataFrame
    ) -> None:
        """Add candlestick trace to the chart.

        Args:
            fig: Plotly figure.
            processed_df: Processed historical data.
        """
        ohlc_df = self.create_ohlc_data(processed_df)

        if not ohlc_df.empty:
            fig.add_trace(
                go.Candlestick(
                    x=ohlc_df["date"],
                    open=ohlc_df["open"],
                    high=ohlc_df["high"],
                    low=ohlc_df["low"],
                    close=ohlc_df["close"],
                    name="Price",
                    increasing_line_color="#26a69a",
                    decreasing_line_color="#ef5350",
                )
            )

    def _add_line_trace(self, fig: go.Figure, processed_df: pd.DataFrame) -> None:
        """Add line trace to the chart.

        Args:
            fig: Plotly figure.
            processed_df: Processed historical data.
        """
        fig.add_trace(
            go.Scatter(
                x=processed_df["date"],
                y=processed_df["price_bdt_per_gram"],
                mode="lines",
                name="Historical Price",
                line=dict(color="#1f77b4", width=2),
            )
        )

    def _add_moving_averages(self, fig: go.Figure, processed_df: pd.DataFrame) -> None:
        """Add moving average traces to the chart.

        Args:
            fig: Plotly figure.
            processed_df: Processed historical data.
        """
        if "ma_7" in processed_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=processed_df["date"],
                    y=processed_df["ma_7"],
                    mode="lines",
                    name="7-day MA",
                    line=dict(color="#ff7f0e", width=1, dash="dash"),
                )
            )

        if "ma_30" in processed_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=processed_df["date"],
                    y=processed_df["ma_30"],
                    mode="lines",
                    name="30-day MA",
                    line=dict(color="#2ca02c", width=1, dash="dot"),
                )
            )

    def _display_trading_signals(self, processed_df: pd.DataFrame) -> None:
        """Display trading signals section.

        Args:
            processed_df: Processed historical data.
        """
        st.markdown("---")
        st.markdown("### 🎯 Trading Signals (Golden/Death Cross)")

        # Detect crossovers
        signal_df = self.analyzer.detect_ma_crossovers(
            processed_df, short_window=7, long_window=30
        )
        latest_signal = self.analyzer.get_latest_signal(signal_df)

        # Display latest signal
        col1, col2, col3 = st.columns(3)

        with col1:
            signal_type = latest_signal.get("signal", "Hold")
            if "Buy" in signal_type:
                st.success(f"🟢 **{signal_type}**")
            elif "Sell" in signal_type:
                st.error(f"🔴 **{signal_type}**")
            else:
                st.info(f"⚪ **{signal_type}**")

        with col2:
            signal_date = latest_signal.get("date")
            if signal_date:
                date_str = (
                    signal_date.strftime("%Y-%m-%d")
                    if hasattr(signal_date, "strftime")
                    else str(signal_date)
                )
                st.metric("Signal Date", date_str)

        with col3:
            st.metric(
                "Signal Strength",
                f"{latest_signal.get('strength', 0):.2f}%",
                help="Based on price momentum",
            )

        # Explain the signal
        self._display_signal_explanation()

        # Recent signals table
        self._display_recent_signals(signal_df)

    def _display_signal_explanation(self) -> None:
        """Display explanation of trading signals."""
        with st.expander("💡 What does this mean?"):
            st.markdown(
                """
            **Golden Cross (Buy Signal)**: Occurs when the short-term moving average (7-day) crosses above
            the long-term moving average (30-day). This suggests upward momentum and potential buying opportunity.

            **Death Cross (Sell Signal)**: Occurs when the short-term moving average crosses below
            the long-term moving average. This suggests downward momentum and potential selling opportunity.

            **Hold**: No recent crossover detected. Market is stable with current trend.

            ⚠️ These signals should be used as one of many factors in making investment decisions,
            not as the sole basis for trading.
            """
            )

    def _display_recent_signals(self, signal_df: pd.DataFrame) -> None:
        """Display recent trading signals.

        Args:
            signal_df: DataFrame with signal information.
        """
        recent_signals = signal_df[signal_df["signal"] != "Hold"].tail(10)

        if not recent_signals.empty:
            st.markdown("#### Recent Signals (Last 10)")
            display_signals = recent_signals[
                ["date", "signal", "price_bdt_per_gram", "ma_short", "ma_long"]
            ].copy()
            display_signals.columns = [
                "Date",
                "Signal",
                "Price (BDT/g)",
                "MA-7",
                "MA-30",
            ]
            st.dataframe(display_signals, use_container_width=True)

    def _display_data_sources(self, filtered_df: pd.DataFrame) -> None:
        """Display data sources information.

        Args:
            filtered_df: Filtered historical data.
        """
        sources = filtered_df["source"].value_counts()
        st.markdown("### 📊 Data Sources")
        for source, count in sources.items():
            st.write(f"- **{source.title()}**: {count} records")


def render_historical_trends_tab(
    settings: dict[str, Any],
    analyzer: Any,
    load_historical_func: Any,
    fetch_historical_func: Any,
    create_ohlc_func: Any,
    format_price_func: Any,
) -> None:
    """Render the historical trends tab.

    Args:
        settings: Dashboard settings dictionary.
        analyzer: AdvancedGoldPriceAnalyzer instance.
        load_historical_func: Function to load historical data.
        fetch_historical_func: Function to fetch and save historical data.
        create_ohlc_func: Function to create OHLC data.
        format_price_func: Function to format price display.
    """
    view = HistoricalTrendsView(
        analyzer=analyzer,
        load_historical_func=load_historical_func,
        fetch_historical_func=fetch_historical_func,
        create_ohlc_func=create_ohlc_func,
        format_price_func=format_price_func,
    )
    view.render(settings)
