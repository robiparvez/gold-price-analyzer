"""
Gold Price Analyzer - Professional Streamlit Application
Complete solution with ML forecasting, investment tracking, jewelry pricing, and backtesting
"""

import asyncio
import logging
from datetime import datetime, timedelta

import altair as alt
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analyzer import AdvancedGoldPriceAnalyzer
from backtesting import GoldPriceBacktester
from database_schema import DatabaseSchema
from historical_scraper import HistoricalGoldPriceScraper
from jewelry_pricing import JewelryPricingCalculator, format_price_breakdown
from logging_config import setup_logging
from pdf_report_generator import generate_simple_report
from scraper import GoldPriceScraper
from utils import format_price_bdt

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    # page_title="Gold Price Analyzer",
    page_title="GP Analyzer",
    page_icon="🥇",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/robiparvez/gold-price-analyzer",
        "Report a bug": "https://github.com/robiparvez/gold-price-analyzer/issues",
        "About": "# Gold Price Analyzer\\nProfessional ML-powered gold price forecasting with investment tracking, jewelry pricing calculator, and comprehensive backtesting.",
    },
)

# Custom CSS for modern styling
st.markdown(
    """
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: #FFD700;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }

    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .forecast-card {
        background: linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .tab-content {
        padding: 1rem 0;
    }

    .stAlert {
        border-radius: 10px;
    }

    div[data-testid="metric-container"] {
        background-color: #f0f2f6;
        border: 1px solid #cccccc;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    .forecast-metric {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 0.8rem;
        border-radius: 8px;
        color: white;
        text-align: center;
        margin: 0.3rem;
        font-weight: bold;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_price_data(days: int = 365) -> pd.DataFrame:
    """Load price data with caching."""
    analyzer = AdvancedGoldPriceAnalyzer()
    return analyzer.load_data(days=days)


@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_historical_data(days: int = 365) -> pd.DataFrame:
    """Load historical data with caching."""
    scraper = HistoricalGoldPriceScraper()
    return scraper.load_historical_data(days=days)


@st.cache_data(ttl=1800)  # Cache for 30 minutes
def scrape_latest_prices() -> dict:
    """Scrape latest prices with caching."""
    try:
        scraper = GoldPriceScraper()
        price_data, _ = scraper.scrape_and_save()
        return price_data
    except Exception as e:
        logger.error(f"Error scraping prices: {e}")
        return {}


@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_and_save_historical_data(days: int = 365) -> pd.DataFrame:
    """Fetch historical data from external sources."""

    async def _fetch_data():
        scraper = HistoricalGoldPriceScraper()
        df = await scraper.fetch_all_historical_data(days=days)
        if not df.empty:
            scraper.save_historical_data(df)
        return df

    # Run async function
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_fetch_data())
        loop.close()
        return result
    except Exception as e:
        logger.error(f"Error fetching historical data: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=7200)  # Cache for 2 hours
def generate_forecast(purity: str, model_type: str) -> dict:
    """Generate 7-day forecast with caching."""
    try:
        analyzer = AdvancedGoldPriceAnalyzer()
        return analyzer.generate_7_day_forecast(purity=purity, model_type=model_type)
    except Exception as e:
        logger.error(f"Error generating forecast: {e}")
        return {"error": str(e)}


def create_ohlc_data(
    df: pd.DataFrame, price_col: str = "price_bdt_per_gram"
) -> pd.DataFrame:
    """
    Create OHLC (Open, High, Low, Close) data from historical prices.

    Args:
        df: DataFrame with date and price columns
        price_col: Name of the price column

    Returns:
        DataFrame with OHLC data
    """
    if df.empty or price_col not in df.columns:
        return pd.DataFrame()

    # Ensure date column is datetime
    df = df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

    # Group by date and calculate OHLC
    ohlc_df = (
        df.groupby(df["date"].dt.date)
        .agg({price_col: ["first", "max", "min", "last"]})
        .reset_index()
    )

    ohlc_df.columns = ["date", "open", "high", "low", "close"]
    ohlc_df["date"] = pd.to_datetime(ohlc_df["date"])

    return ohlc_df


@st.cache_data(ttl=3600)  # Cache for 1 hour
def evaluate_model_performance(purity: str) -> dict:
    """Evaluate model accuracy with caching."""
    try:
        analyzer = AdvancedGoldPriceAnalyzer()
        return analyzer.evaluate_model_accuracy(purity=purity, test_days=14)
    except Exception as e:
        logger.error(f"Error evaluating models: {e}")
        return {"error": str(e)}


def display_metric_cards(stats: dict) -> None:
    """Display metric cards in a modern layout."""
    if not stats:
        st.warning("No statistics available")
        return

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        current_price = stats.get("current_price", 0)
        st.metric(
            label="Current Price",
            value=format_price_bdt(current_price, decimals=0),
            help="Current price per gram in BDT",
        )

    with col2:
        daily_change = stats.get("daily_change", 0)
        daily_change_pct = stats.get("daily_change_percent", 0)
        st.metric(
            label="Daily Change",
            value=format_price_bdt(abs(daily_change), decimals=0),
            delta=f"{daily_change_pct:+.2f}%",
            help="Change from previous day",
        )

    with col3:
        weekly_change = stats.get("weekly_change", 0)
        weekly_change_pct = stats.get("weekly_change_percent", 0)
        st.metric(
            label="Weekly Change",
            value=format_price_bdt(abs(weekly_change), decimals=0),
            delta=f"{weekly_change_pct:+.2f}%",
            help="Change from one week ago",
        )

    with col4:
        avg_price = stats.get("average_price", 0)
        volatility = stats.get("price_volatility", 0)
        st.metric(
            label="Average Price",
            value=format_price_bdt(avg_price, decimals=0),
            delta=f"σ {volatility:,.0f}",
            delta_color="off",
            help="Historical average price and volatility (standard deviation)",
        )


def display_forecast_metrics(forecast_results: dict) -> None:
    """Display 7-day forecast metrics."""
    if "error" in forecast_results or not forecast_results.get("forecasts"):
        st.warning("No forecast data available")
        return

    forecasts = forecast_results["forecasts"]

    st.markdown("### 🔮 7-Day Forecast Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        tomorrow_forecast = forecasts[0] if len(forecasts) > 0 else {}
        tomorrow_price = tomorrow_forecast.get("predicted_price", 0)
        st.metric(
            label="Tomorrow's Price",
            value=format_price_bdt(tomorrow_price, decimals=0),
            help="Predicted price for tomorrow",
        )

    with col2:
        week_forecast = forecasts[-1] if len(forecasts) > 0 else {}
        week_price = week_forecast.get("predicted_price", 0)
        current_price = (
            forecasts[0].get("predicted_price", week_price)
            if len(forecasts) > 1
            else week_price
        )
        week_change = week_price - current_price if current_price else 0
        week_change_pct = (week_change / current_price * 100) if current_price else 0

        st.metric(
            label="Week-End Price",
            value=format_price_bdt(week_price, decimals=0),
            delta=f"{week_change_pct:+.2f}%",
            help="Predicted price at end of 7-day period",
        )

    with col3:
        # Calculate average predicted change
        daily_changes = []
        for i in range(1, len(forecasts)):
            change = (
                forecasts[i]["predicted_price"] - forecasts[i - 1]["predicted_price"]
            )
            daily_changes.append(change)

        avg_daily_change = (
            sum(daily_changes) / len(daily_changes) if daily_changes else 0
        )

        st.metric(
            label="Avg Daily Change",
            value=format_price_bdt(abs(avg_daily_change), decimals=0),
            delta=f"{'Up' if avg_daily_change > 0 else 'Down'}",
            help="Average predicted daily price change",
        )

    with col4:
        # Forecast confidence
        confidence = forecast_results.get("forecasts", [{}])[0].get("confidence", 0.95)
        model_type = forecast_results.get("model_type", "Unknown")

        st.metric(
            label="Model Confidence",
            value=f"{confidence * 100:.0f}%",
            help=f"Forecast confidence using {model_type} model",
        )


def display_forecast_table(forecast_results: dict) -> None:
    """Display detailed forecast table."""
    if "error" in forecast_results or not forecast_results.get("forecasts"):
        return

    forecasts = forecast_results["forecasts"]

    # Create DataFrame for display
    forecast_df = pd.DataFrame(forecasts)
    forecast_df["date"] = pd.to_datetime(forecast_df["date"]).dt.strftime(
        "%Y-%m-%d (%a)"
    )
    forecast_df["predicted_price"] = forecast_df["predicted_price"].round(0).astype(int)
    forecast_df["lower_bound"] = forecast_df["lower_bound"].round(0).astype(int)
    forecast_df["upper_bound"] = forecast_df["upper_bound"].round(0).astype(int)
    forecast_df["confidence"] = (forecast_df["confidence"] * 100).round(1)

    # Calculate daily changes
    daily_changes = [0]  # First day has no change
    for i in range(1, len(forecast_df)):
        change = (
            forecast_df.iloc[i]["predicted_price"]
            - forecast_df.iloc[i - 1]["predicted_price"]
        )
        daily_changes.append(change)

    forecast_df["daily_change"] = daily_changes
    forecast_df["daily_change_pct"] = [
        (
            0
            if i == 0
            else (daily_changes[i] / forecast_df.iloc[i - 1]["predicted_price"] * 100)
        )
        for i in range(len(daily_changes))
    ]

    # Rename columns for display
    display_df = forecast_df[
        [
            "date",
            "predicted_price",
            "daily_change",
            "daily_change_pct",
            "lower_bound",
            "upper_bound",
            "confidence",
        ]
    ].copy()
    display_df.columns = [
        "Date",
        "Predicted Price (BDT)",
        "Daily Change (BDT)",
        "Change %",
        "Lower Bound (BDT)",
        "Upper Bound (BDT)",
        "Confidence %",
    ]

    st.dataframe(display_df, width="stretch", height=300)


def sidebar_controls():
    """Create sidebar controls and return selected options."""
    st.sidebar.markdown("## ⚙️ Settings")

    # Gold purity selection (gold is now the only metal)
    purity_options = ["22K", "21K", "18K", "Traditional"]

    purity = st.sidebar.selectbox(
        "Select Purity",
        options=purity_options,
        index=0,
        help="Choose gold purity level for analysis",
    )

    # Set metal to gold by default
    metal = "gold"

    # Date range
    st.sidebar.markdown("### 📅 Data Range")
    days = st.sidebar.slider(
        "Historical Data (days)",
        min_value=30,
        max_value=365,
        value=180,
        step=30,
        help="Number of days of historical data to analyze",
    )

    # Forecast settings
    st.sidebar.markdown("### 🔮 Forecast Settings")
    enable_forecast = st.sidebar.checkbox(
        "Enable 7-Day Forecasting",
        value=True,
        help="Use ML models for 7-day price forecasting",
    )

    model_type = st.sidebar.selectbox(
        "Forecast Model",
        options=["ensemble", "prophet", "random_forest"],
        index=0,
        disabled=not enable_forecast,
        help="Choose ML model for forecasting",
    )

    # Data sources
    st.sidebar.markdown("### 📊 Data Sources")
    use_historical = st.sidebar.checkbox(
        "Use Historical Data",
        value=True,
        help="Include historical data from external sources",
    )

    # Data refresh
    st.sidebar.markdown("### 🔄 Data Management")

    if st.sidebar.button(
        "🔄 Refresh Historical Data", help="Fetch latest historical data"
    ):
        st.cache_data.clear()
        with st.spinner("Fetching historical data..."):
            fetch_and_save_historical_data(days)
        st.success("Historical data refreshed!")
        st.rerun()

    if st.sidebar.button(
        "📈 Refresh Live Prices", help="Scrape latest prices from BAJUS"
    ):
        st.cache_data.clear()
        st.rerun()

    if st.sidebar.button("🗑️ Clear All Cache", help="Clear all cached data"):
        st.cache_data.clear()
        st.success("Cache cleared!")

    return {
        "metal": metal,
        "purity": purity,
        "days": days,
        "enable_forecast": enable_forecast,
        "model_type": model_type,
        "use_historical": use_historical,
    }


def main():
    """Main application function."""
    # App header
    st.markdown(
        # '<h1 class="main-header">🥇 Gold Price Analyzer</h1>',
        '<h1 class="main-header">🥇 GP Analyzer</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 2rem;'>"
        # "Advanced gold price analysis with historical data and 7-day ML forecasting"
        "Advanced gp analysis with historical data and 7-day ML forecasting" "</p>",
        unsafe_allow_html=True,
    )

    # Initialize database
    db = DatabaseSchema()
    db.create_all_tables()

    # Check if we have current data, if not fetch it
    conn = db.get_connection()
    current_data = conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
    conn.close()

    if current_data == 0:
        st.info("Fetching latest gold price data...")
        try:
            scraper = GoldPriceScraper()
            scraper.scrape_and_save()
            st.success("✅ Data fetched successfully!")
        except Exception as e:
            st.error(f"Failed to fetch data: {e}")

    # Check if we have historical data, if not fetch it
    conn = db.get_connection()
    historical_data = conn.execute("SELECT COUNT(*) FROM historical_prices").fetchone()[
        0
    ]
    conn.close()

    if historical_data == 0:
        st.info("Fetching historical gold price data...")
        try:
            hist_scraper = HistoricalGoldPriceScraper()
            hist_scraper.scrape_and_save()
            st.success("✅ Historical data fetched successfully!")
        except Exception as e:
            st.error(f"Failed to fetch historical data: {e}")

    # Sidebar controls
    settings = sidebar_controls()

    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
        [
            "📊 Current Analysis",
            "🔮 7-Day Forecast",
            "📈 Historical Trends",
            "🎯 Model Performance",
            "🎲 User Input Prediction",
            "💰 Investment Tracker",
            "💍 Jewelry Pricing",
            "🧪 Backtesting",
        ]
    )

    with tab1:
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)

        # Current price analysis (existing functionality)
        with st.spinner("Loading current price data..."):
            try:
                # Load current data
                df = load_price_data(days=30)

                if df.empty:
                    st.error(
                        "❌ No current data available. Please check your data source or try refreshing."
                    )
                    st.stop()

                # Calculate statistics
                analyzer = AdvancedGoldPriceAnalyzer()
                stats = analyzer.calculate_statistics(
                    df, settings["metal"], settings["purity"]
                )

                if not stats:
                    st.warning("No statistics available for selected metal and purity")
                    st.stop()

                # Display metrics
                st.markdown("### 💰 Current Price Metrics")
                display_metric_cards(stats)

                # Load data for trend (more days for better visualization)
                trend_df = load_historical_data(days=365)

                # Price chart
                st.markdown("### 📊 Recent Price Trend (30 days)")
                trends_df = analyzer.calculate_trends(
                    trend_df, settings["metal"], settings["purity"]
                )

                if not trends_df.empty:
                    chart = analyzer.create_price_chart(
                        trend_df, settings["metal"], settings["purity"]
                    )
                    st.plotly_chart(chart, config={"responsive": True})
                    st.toast("Price trend loaded successfully!", icon="✅")
                else:
                    st.warning("No trend data available")

            except Exception as e:
                st.error(f"Error loading current data: {str(e)}")
                st.toast("Failed to load price data", icon="❌")

        st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)

        if not settings["enable_forecast"]:
            st.info("🔮 Enable forecasting in the sidebar to view 7-day predictions")
        else:
            # 7-day forecast
            with st.spinner("Generating 7-day forecast..."):
                try:
                    # Ensure we have historical data
                    historical_df = load_historical_data(settings["days"])

                    if historical_df.empty:
                        st.warning(
                            "No historical data available. Fetching from external sources..."
                        )
                        historical_df = fetch_and_save_historical_data(settings["days"])

                    if historical_df.empty:
                        st.error("Unable to fetch historical data for forecasting")
                        st.stop()

                    # Generate forecast
                    forecast_results = generate_forecast(
                        settings["purity"], settings["model_type"]
                    )

                    if "error" in forecast_results:
                        st.error(f"Forecast error: {forecast_results['error']}")
                        st.toast("Failed to generate forecast", icon="❌")
                    else:
                        st.toast("Forecast generated successfully!", icon="✅")
                        # Display forecast metrics
                        display_forecast_metrics(forecast_results)

                        # Forecast visualization
                        st.markdown("### 📈 7-Day Forecast Chart")
                        analyzer = AdvancedGoldPriceAnalyzer()

                        # Filter historical data for context
                        recent_historical = historical_df[
                            (historical_df["purity"] == settings["purity"])
                            & (historical_df["metal"] == settings["metal"])
                        ].tail(30)

                        forecast_chart = analyzer.create_forecast_visualization(
                            forecast_results, recent_historical
                        )
                        st.plotly_chart(forecast_chart, config={"responsive": True})

                        # Detailed forecast table
                        st.markdown("### 📋 Detailed 7-Day Forecast")
                        display_forecast_table(forecast_results)

                        # Export options
                        col1, col2 = st.columns(2)

                        with col1:
                            # CSV Download
                            if (
                                "forecasts" in forecast_results
                                and forecast_results["forecasts"]
                            ):
                                forecast_csv_df = pd.DataFrame(
                                    forecast_results["forecasts"]
                                )
                                csv = forecast_csv_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Forecast (CSV)",
                                    data=csv,
                                    file_name=f"gold_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
                                    mime="text/csv",
                                )

                        with col2:
                            # PDF Download
                            if st.button("📄 Generate PDF Report"):
                                with st.spinner("Generating PDF report..."):
                                    try:
                                        import tempfile

                                        # Create temporary file
                                        with tempfile.NamedTemporaryFile(
                                            delete=False, suffix=".pdf"
                                        ) as tmp:
                                            temp_path = tmp.name

                                        # Get current price
                                        current_price = (
                                            df["price_bdt_per_gram"].iloc[-1]
                                            if not df.empty
                                            else 8500.0
                                        )

                                        # Generate PDF
                                        forecast_data = pd.DataFrame(
                                            forecast_results["forecasts"]
                                        )
                                        success = generate_simple_report(
                                            current_price=current_price,
                                            purity=settings["purity"],
                                            forecast_df=forecast_data,
                                            output_path=temp_path,
                                        )

                                        if success:
                                            # Read and offer download
                                            with open(temp_path, "rb") as f:
                                                pdf_data = f.read()

                                            st.download_button(
                                                label="📥 Download PDF Report",
                                                data=pdf_data,
                                                file_name=f"gold_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                                                mime="application/pdf",
                                            )
                                            st.success(
                                                "✅ PDF report generated successfully!"
                                            )
                                        else:
                                            st.error("Failed to generate PDF report.")

                                    except Exception as e:
                                        st.error(f"Error generating PDF: {str(e)}")

                        # Model information
                        with st.expander("🔬 Model Information"):
                            st.write(
                                f"**Model Type:** {forecast_results.get('model_type', 'Unknown')}"
                            )
                            st.write(
                                f"**Purity:** {forecast_results.get('purity', settings['purity'])}"
                            )
                            st.write(
                                f"**Generated:** {forecast_results.get('forecast_date', 'Unknown')}"
                            )

                            if "prophet" in forecast_results:
                                st.write(
                                    "**Prophet Model:** Includes seasonal patterns and trend analysis"
                                )
                            if "random_forest" in forecast_results:
                                st.write(
                                    "**Random Forest Model:** Uses technical indicators and price patterns"
                                )
                            if forecast_results.get("model_type") == "ensemble":
                                weights = forecast_results.get("ensemble", {}).get(
                                    "weights", {}
                                )
                                st.write(
                                    f"**Ensemble Weights:** Prophet: {weights.get('prophet', 0):.1%}, Random Forest: {weights.get('random_forest', 0):.1%}"
                                )

                except Exception as e:
                    st.error(f"Error generating forecast: {str(e)}")

        st.markdown("</div>", unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)

        # Historical trends analysis
        with st.spinner("Loading historical data..."):
            try:
                if settings["use_historical"]:
                    historical_df = load_historical_data(settings["days"])

                    if historical_df.empty:
                        st.warning(
                            "No historical data in local storage. Fetching from external sources..."
                        )
                        historical_df = fetch_and_save_historical_data(settings["days"])

                    if not historical_df.empty:
                        # Filter by selected metal and purity
                        filtered_df = historical_df[
                            (historical_df["metal"] == settings["metal"])
                            & (historical_df["purity"] == settings["purity"])
                        ]

                        if not filtered_df.empty:
                            st.markdown(
                                f"### 📈 Historical Trends ({settings['days']} days)"
                            )

                            # Chart type selection
                            chart_type = st.radio(
                                "Chart Type:",
                                ["Line Chart", "Candlestick Chart"],
                                horizontal=True,
                                help="Choose between line chart or candlestick chart for price visualization",
                            )

                            # Historical statistics
                            col1, col2, col3, col4 = st.columns(4)

                            with col1:
                                st.metric("Total Records", len(filtered_df))
                            with col2:
                                st.metric(
                                    "Date Range",
                                    f"{filtered_df['date'].min()} to {filtered_df['date'].max()}",
                                )
                            with col3:
                                st.metric(
                                    "Price Range",
                                    f"{format_price_bdt(filtered_df['price_bdt_per_gram'].min(), decimals=0)} - {format_price_bdt(filtered_df['price_bdt_per_gram'].max(), decimals=0)}",
                                )
                            with col4:
                                avg_price = filtered_df["price_bdt_per_gram"].mean()
                                st.metric(
                                    "Average Price",
                                    format_price_bdt(avg_price, decimals=0),
                                )

                            # Historical chart
                            analyzer = AdvancedGoldPriceAnalyzer()
                            processed_df = analyzer.preprocess_historical_data(
                                filtered_df
                            )

                            if not processed_df.empty:
                                # Create historical chart
                                fig = go.Figure()

                                if chart_type == "Candlestick Chart":
                                    # Create OHLC data
                                    ohlc_df = create_ohlc_data(processed_df)

                                    if not ohlc_df.empty:
                                        # Add candlestick chart
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
                                else:
                                    # Price line (original)
                                    fig.add_trace(
                                        go.Scatter(
                                            x=processed_df["date"],
                                            y=processed_df["price_bdt_per_gram"],
                                            mode="lines",
                                            name="Historical Price",
                                            line=dict(color="#1f77b4", width=2),
                                        )
                                    )

                                # Moving averages
                                if "ma_7" in processed_df.columns:
                                    fig.add_trace(
                                        go.Scatter(
                                            x=processed_df["date"],
                                            y=processed_df["ma_7"],
                                            mode="lines",
                                            name="7-day MA",
                                            line=dict(
                                                color="#ff7f0e", width=1, dash="dash"
                                            ),
                                        )
                                    )

                                if "ma_30" in processed_df.columns:
                                    fig.add_trace(
                                        go.Scatter(
                                            x=processed_df["date"],
                                            y=processed_df["ma_30"],
                                            mode="lines",
                                            name="30-day MA",
                                            line=dict(
                                                color="#2ca02c", width=1, dash="dot"
                                            ),
                                        )
                                    )

                                fig.update_layout(
                                    title=f"Historical Gold Prices - {settings['purity']} ({settings['days']} days)",
                                    xaxis_title="Date",
                                    yaxis_title="Price (BDT/gram)",
                                    yaxis=dict(tickformat=",.0f", ticksuffix=" BDT"),
                                    height=500,
                                )

                                st.plotly_chart(fig, config={"responsive": True})

                                # Golden/Death Cross Signal Detection
                                st.markdown("---")
                                st.markdown(
                                    "### 🎯 Trading Signals (Golden/Death Cross)"
                                )

                                # Detect crossovers
                                signal_df = analyzer.detect_ma_crossovers(
                                    processed_df, short_window=7, long_window=30
                                )
                                latest_signal = analyzer.get_latest_signal(signal_df)

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
                                        st.metric(
                                            "Signal Date",
                                            (
                                                signal_date.strftime("%Y-%m-%d")
                                                if hasattr(signal_date, "strftime")
                                                else str(signal_date)
                                            ),
                                        )

                                with col3:
                                    st.metric(
                                        "Signal Strength",
                                        f"{latest_signal.get('strength', 0):.2f}%",
                                        help="Based on price momentum",
                                    )

                                # Explain the signal
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

                                # Recent signals table
                                recent_signals = signal_df[
                                    signal_df["signal"] != "Hold"
                                ].tail(10)
                                if not recent_signals.empty:
                                    st.markdown("#### Recent Signals (Last 10)")
                                    display_signals = recent_signals[
                                        [
                                            "date",
                                            "signal",
                                            "price_bdt_per_gram",
                                            "ma_short",
                                            "ma_long",
                                        ]
                                    ].copy()
                                    display_signals.columns = [
                                        "Date",
                                        "Signal",
                                        "Price (BDT/g)",
                                        "MA-7",
                                        "MA-30",
                                    ]
                                    st.dataframe(display_signals, width="stretch")

                                # Data sources information
                                sources = filtered_df["source"].value_counts()
                                st.markdown("### 📊 Data Sources")
                                for source, count in sources.items():
                                    st.write(f"- **{source.title()}**: {count} records")
                        else:
                            st.warning(
                                f"No historical data available for {settings['metal']} {settings['purity']}"
                            )
                    else:
                        st.error("Unable to load historical data")
                else:
                    st.info(
                        "Historical data analysis is disabled. Enable it in the sidebar."
                    )

            except Exception as e:
                st.error(f"Error loading historical data: {str(e)}")

        st.markdown("</div>", unsafe_allow_html=True)

    with tab4:
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)

        # Model performance evaluation
        if not settings["enable_forecast"]:
            st.info("🎯 Enable forecasting to view model performance metrics")
        else:
            with st.spinner("Evaluating model performance..."):
                try:
                    evaluation_results = evaluate_model_performance(settings["purity"])

                    if "error" in evaluation_results:
                        st.error(f"Evaluation error: {evaluation_results['error']}")
                    elif "models" in evaluation_results:
                        st.markdown("### 🎯 Model Accuracy Assessment")

                        # Display metrics for each model
                        for model_name, metrics in evaluation_results["models"].items():
                            if "error" not in metrics:
                                st.markdown(f"#### {model_name.title()} Model")

                                col1, col2, col3, col4 = st.columns(4)

                                with col1:
                                    accuracy = metrics.get("accuracy_percentage", 0)
                                    st.metric("Accuracy", f"{accuracy:.1f}%")

                                with col2:
                                    mape = metrics.get("mape", 0)
                                    st.metric("MAPE", f"{mape:.2f}%")

                                with col3:
                                    mae = metrics.get("mae", 0)
                                    st.metric("MAE", format_price_bdt(mae, decimals=0))

                                with col4:
                                    r2 = metrics.get("r2_score", 0)
                                    st.metric("R² Score", f"{r2:.3f}")

                                # Feature importance for Random Forest
                                if (
                                    model_name == "random_forest"
                                    and "feature_importance" in metrics
                                ):
                                    with st.expander(
                                        f"📊 {model_name.title()} Feature Importance"
                                    ):
                                        importance_df = pd.DataFrame(
                                            [
                                                {"Feature": k, "Importance": v}
                                                for k, v in metrics[
                                                    "feature_importance"
                                                ].items()
                                            ]
                                        ).sort_values("Importance", ascending=False)

                                        chart = (
                                            alt.Chart(importance_df)
                                            .mark_bar()
                                            .encode(
                                                x=alt.X(
                                                    "Importance:Q", title="Importance"
                                                ),
                                                y=alt.Y(
                                                    "Feature:N",
                                                    sort="-x",
                                                    title="Feature",
                                                ),
                                            )
                                            .properties(
                                                title=f"{model_name.title()} Feature Importance",
                                                height=400,
                                            )
                                        )
                                        st.altair_chart(chart, use_container_width=True)

                        # Model comparison
                        if len(evaluation_results["models"]) > 1:
                            st.markdown("### 📊 Model Comparison")

                            comparison_data = []
                            for model_name, metrics in evaluation_results[
                                "models"
                            ].items():
                                if "error" not in metrics:
                                    comparison_data.append(
                                        {
                                            "Model": model_name.title(),
                                            "Accuracy (%)": f"{metrics.get('accuracy_percentage', 0):.1f}",
                                            "MAPE (%)": f"{metrics.get('mape', 0):.2f}",
                                            "MAE (BDT)": format_price_bdt(
                                                metrics.get("mae", 0), decimals=0
                                            ),
                                            "R² Score": f"{metrics.get('r2_score', 0):.3f}",
                                            "Test Samples": metrics.get(
                                                "test_samples", 0
                                            ),
                                        }
                                    )

                            if comparison_data:
                                comparison_df = pd.DataFrame(comparison_data)
                                st.dataframe(comparison_df, width="stretch")
                    else:
                        st.warning("No model evaluation results available")

                except Exception as e:
                    st.error(f"Error evaluating models: {str(e)}")

        st.markdown("</div>", unsafe_allow_html=True)

    with tab5:
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)

        if not settings["enable_forecast"]:
            st.info("🎲 Enable forecasting in the sidebar to use user input prediction")
        else:
            st.markdown("### 🎲 Custom Prediction with User Inputs")

            # User input section
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("#### Input Parameters")
                base_price = st.number_input(
                    "Current Gold Price (BDT/gram)",
                    min_value=0,
                    max_value=20000,
                    value=7500,
                    step=100,
                    help="Enter the current gold price per gram",
                )

            with col2:
                st.markdown("#### Market Factors")
                _ = st.number_input(
                    "USD to BDT Exchange Rate",
                    min_value=50.0,
                    max_value=200.0,
                    value=110.0,
                    step=0.5,
                    help="Current USD to BDT exchange rate",
                )

            with col3:
                st.markdown("#### Prediction Settings")
                prediction_days = st.slider(
                    "Prediction Horizon (Days)",
                    min_value=1,
                    max_value=30,
                    value=7,
                    help="How many days ahead to predict",
                )

            # Technical indicators input
            with st.expander("📊 Advanced Technical Indicators (Optional)"):
                col1, col2, col3 = st.columns(3)

                with col1:
                    volatility = st.number_input(
                        "Price Volatility (%)",
                        min_value=0.0,
                        max_value=20.0,
                        value=2.5,
                        step=0.1,
                        help="Expected price volatility",
                    )

                with col2:
                    momentum = st.slider(
                        "Market Momentum",
                        min_value=-10,
                        max_value=10,
                        value=0,
                        help="Current market momentum (-10 bearish to +10 bullish)",
                    )

                with col3:
                    seasonality = st.selectbox(
                        "Seasonal Factor",
                        [
                            "Normal",
                            "Festive Season",
                            "Wedding Season",
                            "Economic Downturn",
                        ],
                        help="Select current market season",
                    )

            # Generate prediction button
            if st.button(
                "🔮 Generate Custom Prediction",
                type="primary",
                width="stretch",
            ):
                with st.spinner("Generating custom prediction..."):
                    try:
                        # Calculate adjusted factors
                        seasonal_factors = {
                            "Normal": 1.0,
                            "Festive Season": 1.05,
                            "Wedding Season": 1.08,
                            "Economic Downturn": 0.92,
                        }

                        seasonal_multiplier = seasonal_factors[seasonality]
                        _ = 1 + (volatility / 100)  # volatility_adjustment
                        _ = 1 + (momentum / 100)  # momentum_adjustment

                        # Generate predictions
                        predictions = []
                        dates = []
                        base_adjusted = base_price * seasonal_multiplier

                        for day in range(1, prediction_days + 1):
                            # Add some randomness and trends
                            daily_change = (momentum * 0.1) + (
                                volatility * 0.05 * (1 if day % 2 == 0 else -1)
                            )
                            price = base_adjusted * (1 + daily_change * day / 10)

                            predictions.append(price)
                            dates.append(datetime.now() + timedelta(days=day))

                        # Create prediction dataframe
                        prediction_df = pd.DataFrame(
                            {"date": dates, "predicted_price": predictions}
                        )

                        # Display prediction results
                        st.markdown("### 📊 Custom Prediction Results")

                        # Summary metrics
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric(
                                "Starting Price",
                                format_price_bdt(base_price, decimals=0),
                            )

                        with col2:
                            st.metric(
                                "Predicted Average",
                                format_price_bdt(
                                    prediction_df["predicted_price"].mean(), decimals=0
                                ),
                                delta=f"{((prediction_df['predicted_price'].mean() - base_price) / base_price * 100):.1f}%",
                            )

                        with col3:
                            st.metric(
                                "Highest Prediction",
                                format_price_bdt(
                                    prediction_df["predicted_price"].max(), decimals=0
                                ),
                                delta=f"{((prediction_df['predicted_price'].max() - base_price) / base_price * 100):.1f}%",
                            )

                        with col4:
                            st.metric(
                                "Lowest Prediction",
                                format_price_bdt(
                                    prediction_df["predicted_price"].min(), decimals=0
                                ),
                                delta=f"{((prediction_df['predicted_price'].min() - base_price) / base_price * 100):.1f}%",
                            )

                        # Prediction chart
                        fig = go.Figure()

                        # Current price line
                        fig.add_trace(
                            go.Scatter(
                                x=[datetime.now()],
                                y=[base_price],
                                mode="markers",
                                name="Current Price",
                                marker=dict(size=15, color="red", symbol="circle"),
                            )
                        )

                        # Prediction line
                        fig.add_trace(
                            go.Scatter(
                                x=prediction_df["date"],
                                y=prediction_df["predicted_price"],
                                mode="lines+markers",
                                name="Predicted Price",
                                line=dict(color="#00cc96", width=3),
                                marker=dict(size=8),
                            )
                        )

                        # Confidence bands (simple approximation)
                        upper_band = prediction_df["predicted_price"] * (
                            1 + volatility / 100
                        )
                        lower_band = prediction_df["predicted_price"] * (
                            1 - volatility / 100
                        )

                        fig.add_trace(
                            go.Scatter(
                                x=prediction_df["date"],
                                y=upper_band,
                                mode="lines",
                                name="Upper Band",
                                line=dict(color="rgba(0,0,0,0.2)", width=0),
                                showlegend=False,
                            )
                        )

                        fig.add_trace(
                            go.Scatter(
                                x=prediction_df["date"],
                                y=lower_band,
                                mode="lines",
                                name="Confidence Band",
                                line=dict(color="rgba(0,0,0,0.2)", width=0),
                                fill="tonexty",
                                fillcolor="rgba(0,204,150,0.2)",
                            )
                        )

                        fig.update_layout(
                            title=f"Custom Price Prediction - {prediction_days} Days",
                            xaxis_title="Date",
                            yaxis_title="Predicted Price (BDT/gram)",
                            yaxis=dict(tickformat=",.0f", ticksuffix=" BDT"),
                            height=500,
                            hovermode="x unified",
                        )

                        st.plotly_chart(fig, config={"responsive": True})

                        # Detailed prediction table
                        st.markdown("### 📋 Detailed Predictions")

                        display_df = prediction_df.copy()
                        display_df["date"] = pd.to_datetime(
                            display_df["date"]
                        ).dt.strftime("%Y-%m-%d")
                        display_df["daily_change"] = (
                            display_df["predicted_price"].diff().fillna(0)
                        )
                        display_df["daily_change_pct"] = (
                            display_df["daily_change"]
                            / display_df["predicted_price"].shift(1).fillna(base_price)
                            * 100
                        )

                        display_df.columns = [
                            "Date",
                            "Predicted Price (BDT/gram)",
                            "Daily Change",
                            "Daily Change (%)",
                        ]
                        st.dataframe(display_df, width="stretch")

                        # Download button
                        csv = display_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Predictions as CSV",
                            data=csv,
                            file_name=f"gold_price_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv",
                        )

                        # Disclaimer
                        st.warning(
                            """
                        ⚠️ **Disclaimer**: These predictions are based on user-provided inputs and simplified models.
                        They should not be used for making actual financial decisions. Consult with professional
                        financial advisors before making any investment decisions based on gold prices.
                        """
                        )

                    except Exception as e:
                        st.error(f"Error generating prediction: {str(e)}")

            # Help information
            with st.expander("💡 How to Use This Feature"):
                st.markdown(
                    """
                1. **Enter Current Price**: Input today's gold price per gram
                2. **Set Exchange Rate**: Provide the current USD to BDT exchange rate
                3. **Choose Prediction Horizon**: Select how many days ahead you want to predict
                4. **Adjust Technical Indicators** (Optional):
                   - **Volatility**: How much the price typically fluctuates
                   - **Momentum**: Current market trend (-10 bearish to +10 bullish)
                   - **Seasonality**: Select current market season for better accuracy
                5. **Generate Prediction**: Click the button to see your custom forecast

                The prediction uses a combination of your inputs, seasonal factors, and volatility
                adjustments to generate an educated forecast. The confidence band shows the
                expected range of prices based on the volatility setting.
                """
                )

        st.markdown("</div>", unsafe_allow_html=True)

    with tab6:
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        st.markdown("## 💰 Investment Tracker")

        # Create sub-tabs for different functions
        inv_tab1, inv_tab2, inv_tab3, inv_tab4 = st.tabs(
            [
                "📝 Add Transaction",
                "📊 Portfolio Summary",
                "🎯 Investment Goals",
                "📈 Transaction History",
            ]
        )

        with inv_tab1:
            st.markdown("### Add New Transaction")

            col1, col2 = st.columns(2)

            with col1:
                trans_type = st.selectbox(
                    "Transaction Type",
                    ["Buy", "Sell"],
                    help="Select whether you're buying or selling gold",
                )

                trans_date = st.date_input(
                    "Transaction Date",
                    value=datetime.now(),
                    help="Date of the transaction",
                )

                purity = st.selectbox(
                    "Gold Purity", ["22k", "21k", "18k"], help="Purity of gold"
                )

            with col2:
                quantity = st.number_input(
                    "Quantity (grams)",
                    min_value=0.01,
                    value=1.0,
                    step=0.01,
                    help="Weight in grams",
                )

                price_per_gram = st.number_input(
                    "Price per Gram (BDT)",
                    min_value=0.0,
                    value=8500.0,
                    step=10.0,
                    help="Price per gram in BDT",
                )

                total_amount = quantity * price_per_gram
                st.metric("Total Amount", format_price_bdt(total_amount, decimals=2))

            notes = st.text_area(
                "Notes (Optional)",
                placeholder="Add any notes about this transaction...",
                help="Additional details about the transaction",
            )

            if st.button("💾 Save Transaction", type="primary"):
                try:
                    transaction = {
                        "transaction_date": trans_date,
                        "transaction_type": trans_type.lower(),
                        "purity": purity,
                        "quantity_grams": quantity,
                        "price_per_gram": price_per_gram,
                        "total_amount": total_amount,
                        "notes": notes,
                    }

                    if db.insert_investment_tracking(transaction):
                        st.success("✅ Transaction saved successfully!")
                        st.balloons()
                    else:
                        st.error("❌ Failed to save transaction. Please try again.")

                except Exception as e:
                    st.error(f"Error saving transaction: {str(e)}")

        with inv_tab2:
            st.markdown("### 📊 Portfolio Summary")

            try:
                # Get portfolio data from database
                conn = db.get_connection()
                portfolio_df = conn.execute(
                    """
                    SELECT
                        purity,
                        total_quantity_grams,
                        average_cost_per_gram,
                        current_value_bdt,
                        profit_loss_bdt,
                        profit_loss_percent,
                        last_updated
                    FROM portfolio
                    ORDER BY purity
                """
                ).fetchdf()

                if not portfolio_df.empty:
                    # Display summary metrics
                    total_value = portfolio_df["current_value_bdt"].sum()
                    total_pl = portfolio_df["profit_loss_bdt"].sum()

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            "Total Portfolio Value",
                            format_price_bdt(total_value, decimals=2),
                        )
                    with col2:
                        st.metric(
                            "Total P/L",
                            format_price_bdt(total_pl, decimals=2),
                            delta=f"{(total_pl/total_value*100) if total_value > 0 else 0:.2f}%",
                        )
                    with col3:
                        st.metric(
                            "Total Holdings",
                            f"{portfolio_df['total_quantity_grams'].sum():.2f}g",
                        )

                    # Portfolio breakdown
                    st.markdown("#### Holdings by Purity")

                    # Create pie chart
                    fig = go.Figure(
                        data=[
                            go.Pie(
                                labels=portfolio_df["purity"],
                                values=portfolio_df["current_value_bdt"],
                                hole=0.3,
                                marker_colors=["#FFD700", "#C0C0C0", "#CD7F32"],
                            )
                        ]
                    )

                    fig.update_layout(
                        title="Portfolio Distribution by Value", height=400
                    )

                    st.plotly_chart(fig, config={"responsive": True})

                    # Detailed table
                    st.markdown("#### Detailed Breakdown")
                    display_df = portfolio_df.copy()
                    display_df.columns = [
                        "Purity",
                        "Quantity (g)",
                        "Avg Cost (BDT/g)",
                        "Current Value (BDT)",
                        "P/L (BDT)",
                        "P/L (%)",
                        "Last Updated",
                    ]
                    st.dataframe(display_df, width="stretch")
                else:
                    st.info(
                        "📭 No portfolio data available. Start by adding transactions!"
                    )

            except Exception as e:
                st.error(f"Error loading portfolio: {str(e)}")
                logger.error(f"Portfolio error: {e}")

        with inv_tab3:
            st.markdown("### 🎯 Investment Goals")

            # Add new goal
            with st.expander("➕ Add New Goal"):
                goal_name = st.text_input("Goal Name", placeholder="e.g., Wedding Ring")

                col1, col2 = st.columns(2)
                with col1:
                    target_amount = st.number_input(
                        "Target Amount (BDT)",
                        min_value=0.0,
                        value=100000.0,
                        step=1000.0,
                    )
                    goal_type = st.selectbox(
                        "Goal Type",
                        ["Savings", "Purchase", "Investment", "Emergency Fund"],
                    )

                with col2:
                    target_date = st.date_input(
                        "Target Date", value=datetime.now() + timedelta(days=365)
                    )
                    current_amount = st.number_input(
                        "Current Amount (BDT)", min_value=0.0, value=0.0, step=1000.0
                    )

                if st.button("💾 Save Goal"):
                    try:
                        goal = {
                            "goal_name": goal_name,
                            "target_amount": target_amount,
                            "target_date": target_date,
                            "current_amount": current_amount,
                            "goal_type": goal_type.lower(),
                        }

                        if db.insert_investment_goal(goal):
                            st.success("✅ Goal saved successfully!")
                        else:
                            st.error("❌ Failed to save goal.")
                    except Exception as e:
                        st.error(f"Error saving goal: {str(e)}")

            # Display existing goals
            try:
                conn = db.get_connection()
                goals_df = conn.execute(
                    """
                    SELECT
                        goal_name,
                        target_amount,
                        current_amount,
                        target_date,
                        goal_type,
                        status
                    FROM investment_goals
                    WHERE status = 'active'
                    ORDER BY target_date
                """
                ).fetchdf()

                if not goals_df.empty:
                    st.markdown("#### Active Goals")

                    for idx, goal in goals_df.iterrows():
                        progress = (
                            (goal["current_amount"] / goal["target_amount"] * 100)
                            if goal["target_amount"] > 0
                            else 0
                        )

                        with st.container():
                            st.markdown(
                                f"**{goal['goal_name']}** ({goal['goal_type'].title()})"
                            )

                            col1, col2, col3 = st.columns([2, 1, 1])
                            with col1:
                                st.progress(min(progress / 100, 1.0))
                            with col2:
                                st.write(
                                    f"{format_price_bdt(goal['current_amount'], decimals=0)} / {format_price_bdt(goal['target_amount'], decimals=0)}"
                                )
                            with col3:
                                st.write(f"Target: {goal['target_date']}")

                            st.markdown("---")
                else:
                    st.info("📭 No active goals. Add your first goal above!")

            except Exception as e:
                st.error(f"Error loading goals: {str(e)}")

        with inv_tab4:
            st.markdown("### 📈 Transaction History")

            try:
                conn = db.get_connection()
                transactions_df = conn.execute(
                    """
                    SELECT
                        transaction_date,
                        transaction_type,
                        purity,
                        quantity_grams,
                        price_per_gram,
                        total_amount,
                        notes
                    FROM investment_tracking
                    ORDER BY transaction_date DESC
                    LIMIT 100
                """
                ).fetchdf()

                if not transactions_df.empty:
                    # Filter options
                    col1, col2 = st.columns(2)
                    with col1:
                        filter_type = st.multiselect(
                            "Filter by Type",
                            options=["buy", "sell"],
                            default=["buy", "sell"],
                        )
                    with col2:
                        filter_purity = st.multiselect(
                            "Filter by Purity",
                            options=transactions_df["purity"].unique().tolist(),
                            default=transactions_df["purity"].unique().tolist(),
                        )

                    # Apply filters
                    filtered_df = transactions_df[
                        (transactions_df["transaction_type"].isin(filter_type))
                        & (transactions_df["purity"].isin(filter_purity))
                    ]

                    # Display summary
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        total_bought = filtered_df[
                            filtered_df["transaction_type"] == "buy"
                        ]["quantity_grams"].sum()
                        st.metric("Total Bought", f"{total_bought:.2f}g")
                    with col2:
                        total_sold = filtered_df[
                            filtered_df["transaction_type"] == "sell"
                        ]["quantity_grams"].sum()
                        st.metric("Total Sold", f"{total_sold:.2f}g")
                    with col3:
                        net_position = total_bought - total_sold
                        st.metric("Net Position", f"{net_position:.2f}g")

                    # Transaction table
                    st.markdown("#### Recent Transactions")
                    display_df = filtered_df.copy()
                    display_df.columns = [
                        "Date",
                        "Type",
                        "Purity",
                        "Quantity (g)",
                        "Price/g (BDT)",
                        "Total (BDT)",
                        "Notes",
                    ]
                    st.dataframe(display_df, width="stretch")

                    # Download button
                    csv = display_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Transaction History",
                        data=csv,
                        file_name=f"investment_history_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                    )
                else:
                    st.info("📭 No transactions yet. Add your first transaction!")

            except Exception as e:
                st.error(f"Error loading transactions: {str(e)}")
                logger.error(f"Transaction history error: {e}")

        st.markdown("</div>", unsafe_allow_html=True)

    with tab7:
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        st.markdown("## 💍 Jewelry Pricing Calculator")

        st.markdown(
            """
        Calculate the total cost of jewelry including base gold price, VAT (5%), and making charges.
        Making charges vary based on the type of jewelry and gold purity.
        """
        )

        calculator = JewelryPricingCalculator()

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Input Parameters")

            base_price = st.number_input(
                "Base Gold Price (BDT/gram)",
                min_value=0.0,
                value=8500.0,
                step=10.0,
                help="Current market price of gold per gram",
            )

            weight = st.number_input(
                "Weight (grams)",
                min_value=0.01,
                value=5.0,
                step=0.01,
                help="Weight of the jewelry item",
            )

            purity_jp = st.selectbox(
                "Gold Purity",
                ["22k", "21k", "18k"],
                help="Purity of gold in the jewelry",
            )

            item_type = st.selectbox(
                "Jewelry Type",
                [
                    "ring",
                    "necklace",
                    "bracelet",
                    "earrings",
                    "chain",
                    "bangle",
                    "other",
                ],
                help="Type of jewelry item",
            )

            # Show making charge rate
            making_rate = calculator.get_making_charge_rate(item_type, purity_jp)
            st.info(
                f"💎 Making Charge for {item_type.title()} ({purity_jp}): {format_price_bdt(making_rate)}/gram"
            )

            use_custom = st.checkbox("Use Custom Making Charge")
            custom_charge = None
            if use_custom:
                custom_charge = st.number_input(
                    "Custom Making Charge (BDT/gram)",
                    min_value=0.0,
                    value=float(making_rate),
                    step=10.0,
                )

        with col2:
            st.markdown("### Price Breakdown")

            if st.button("💰 Calculate Price", type="primary"):
                try:
                    price = calculator.calculate_jewelry_price(
                        base_price_per_gram=base_price,
                        weight_grams=weight,
                        purity=purity_jp,  # type: ignore[arg-type]
                        item_type=item_type,  # type: ignore[arg-type]
                        custom_making_charge=custom_charge,
                    )

                    # Display in metrics
                    st.metric(
                        "Base Gold Price",
                        format_price_bdt(price.base_price, decimals=2),
                    )
                    st.metric(
                        "VAT (5%)", format_price_bdt(price.vat_amount, decimals=2)
                    )
                    st.metric(
                        "Making Charges",
                        format_price_bdt(price.making_charges, decimals=2),
                    )
                    st.markdown("---")
                    st.metric(
                        "**Total Price**",
                        format_price_bdt(price.total_price, decimals=2),
                        help="Final price including all charges",
                    )

                    # Detailed breakdown
                    with st.expander("📋 Detailed Breakdown"):
                        st.code(format_price_breakdown(price))

                except Exception as e:
                    st.error(f"Error calculating price: {str(e)}")

        # Price comparison
        st.markdown("---")
        st.markdown("### 📊 Compare Prices Across Jewelry Types")

        if st.button("🔍 Compare All Types"):
            try:
                comparison = calculator.calculate_price_comparison(
                    base_price_per_gram=base_price,
                    weight_grams=weight,
                    purity=purity_jp,
                )

                # Create comparison dataframe
                comp_data = []
                for item, price in comparison.items():
                    comp_data.append(
                        {
                            "Jewelry Type": item.title(),
                            "Making Charge/g": format_price_bdt(
                                price.making_charges / weight, decimals=0
                            ),
                            "Base Price": format_price_bdt(
                                price.base_price, decimals=2
                            ),
                            "VAT": format_price_bdt(price.vat_amount, decimals=2),
                            "Making Charges": format_price_bdt(
                                price.making_charges, decimals=2
                            ),
                            "Total Price": format_price_bdt(
                                price.total_price, decimals=2
                            ),
                        }
                    )

                comp_df = pd.DataFrame(comp_data)
                st.dataframe(comp_df, width="stretch")

                # Visualization
                fig = go.Figure(
                    data=[
                        go.Bar(
                            name="Base Price",
                            x=[item.title() for item in comparison.keys()],
                            y=[price.base_price for price in comparison.values()],
                            marker_color="#FFD700",
                        ),
                        go.Bar(
                            name="Making Charges",
                            x=[item.title() for item in comparison.keys()],
                            y=[price.making_charges for price in comparison.values()],
                            marker_color="#CD7F32",
                        ),
                        go.Bar(
                            name="VAT",
                            x=[item.title() for item in comparison.keys()],
                            y=[price.vat_amount for price in comparison.values()],
                            marker_color="#C0C0C0",
                        ),
                    ]
                )

                fig.update_layout(
                    barmode="stack",
                    title=f"Price Comparison for {weight}g {purity_jp} Jewelry",
                    xaxis_title="Jewelry Type",
                    yaxis_title="Price (BDT)",
                    yaxis=dict(tickformat=",.0f", ticksuffix=" BDT"),
                    height=500,
                )

                st.plotly_chart(fig, config={"responsive": True})

            except Exception as e:
                st.error(f"Error generating comparison: {str(e)}")

        st.markdown("</div>", unsafe_allow_html=True)

    with tab8:
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        st.markdown("## 🧪 Model Backtesting")

        st.markdown(
            """
        Evaluate model performance using historical data. Backtesting simulates how the model
        would have performed in the past by training on historical windows and testing predictions.
        """
        )

        # Load historical data for backtesting
        historical_df = load_historical_data(days=365)

        if not historical_df.empty:
            # Filter configuration
            col1, col2 = st.columns(2)

            with col1:
                purity_bt = st.selectbox(
                    "Select Purity", ["22k", "21k", "18k"], key="backtest_purity"
                )

                window_size = st.slider(
                    "Training Window (days)",
                    min_value=14,
                    max_value=90,
                    value=30,
                    step=7,
                    help="Number of days to use for training",
                )

            with col2:
                forecast_horizon = st.slider(
                    "Forecast Horizon (days)",
                    min_value=1,
                    max_value=14,
                    value=7,
                    step=1,
                    help="Number of days ahead to predict",
                )

                model_type_bt = st.selectbox(
                    "Model Type",
                    ["ensemble", "prophet", "random_forest", "xgboost"],
                    help="Select model to backtest",
                )

            # Run backtest button
            if st.button("🚀 Run Backtest", type="primary"):
                with st.spinner("Running backtest... This may take a few minutes..."):
                    try:
                        # Filter data
                        backtest_df = historical_df[
                            (historical_df["purity"] == purity_bt)
                            & (historical_df["metal"] == "gold")
                        ].copy()

                        if len(backtest_df) < window_size + forecast_horizon:
                            st.error(
                                f"Insufficient data. Need at least {window_size + forecast_horizon} days."
                            )
                        else:
                            # Initialize backtester
                            backtester = GoldPriceBacktester()

                            # Run rolling window backtest
                            results = backtester.rolling_window_backtest(
                                df=backtest_df,
                                window_size=window_size,
                                forecast_horizon=forecast_horizon,
                                model_type=model_type_bt,  # type: ignore[arg-type]
                                step_size=forecast_horizon,
                            )

                            if results and "metrics" in results:
                                st.success("✅ Backtest completed successfully!")

                                # Display metrics
                                st.markdown("### 📊 Performance Metrics")

                                col1, col2, col3, col4, col5 = st.columns(5)

                                with col1:
                                    st.metric(
                                        "MAE",
                                        f"{results['metrics']['mae']:.2f}",
                                        help="Mean Absolute Error (BDT/gram)",
                                    )

                                with col2:
                                    st.metric(
                                        "RMSE",
                                        f"{results['metrics']['rmse']:.2f}",
                                        help="Root Mean Squared Error (BDT/gram)",
                                    )

                                with col3:
                                    st.metric(
                                        "MAPE",
                                        f"{results['metrics']['mape']:.2f}%",
                                        help="Mean Absolute Percentage Error",
                                    )

                                with col4:
                                    st.metric(
                                        "R² Score",
                                        f"{results['metrics']['r2_score']:.3f}",
                                        help="Coefficient of determination",
                                    )

                                with col5:
                                    st.metric(
                                        "Predictions",
                                        results["metrics"]["total_predictions"],
                                        help="Total number of predictions made",
                                    )

                                # Prediction vs Actual chart
                                st.markdown("### 📈 Predictions vs Actual Prices")

                                results_df = results["results_df"]

                                fig = go.Figure()

                                # Actual prices
                                fig.add_trace(
                                    go.Scatter(
                                        x=results_df["date"],
                                        y=results_df["actual"],
                                        mode="lines+markers",
                                        name="Actual Price",
                                        line=dict(color="#1f77b4", width=2),
                                        marker=dict(size=6),
                                    )
                                )

                                # Predicted prices
                                fig.add_trace(
                                    go.Scatter(
                                        x=results_df["date"],
                                        y=results_df["predicted"],
                                        mode="lines+markers",
                                        name="Predicted Price",
                                        line=dict(
                                            color="#ff7f0e", width=2, dash="dash"
                                        ),
                                        marker=dict(size=6, symbol="x"),
                                    )
                                )

                                fig.update_layout(
                                    title=f"Backtest Results: {model_type_bt.title()} Model",
                                    xaxis_title="Date",
                                    yaxis_title="Price (BDT/gram)",
                                    yaxis=dict(tickformat=",.0f", ticksuffix=" BDT"),
                                    height=500,
                                    hovermode="x unified",
                                )

                                st.plotly_chart(fig, config={"responsive": True})

                                # Error distribution
                                st.markdown("### 📉 Error Distribution")

                                col1, col2 = st.columns(2)

                                with col1:
                                    # Error histogram
                                    fig_hist = go.Figure(
                                        data=[
                                            go.Histogram(
                                                x=results_df["error"],
                                                nbinsx=30,
                                                marker_color="#2ca02c",
                                            )
                                        ]
                                    )

                                    fig_hist.update_layout(
                                        title="Distribution of Prediction Errors",
                                        xaxis_title="Absolute Error (BDT/gram)",
                                        yaxis_title="Frequency",
                                        height=400,
                                    )

                                    st.plotly_chart(
                                        fig_hist, config={"responsive": True}
                                    )

                                with col2:
                                    # Error statistics
                                    error_stats = backtester.get_error_distribution()

                                    if error_stats:
                                        st.markdown("#### Error Statistics")
                                        st.metric(
                                            "Mean Error",
                                            f"{error_stats['mean_error']:.2f} BDT/g",
                                        )
                                        st.metric(
                                            "Std Error",
                                            f"{error_stats['std_error']:.2f} BDT/g",
                                        )
                                        st.metric(
                                            "Median Error",
                                            f"{error_stats['median_error']:.2f} BDT/g",
                                        )
                                        st.metric(
                                            "95th Percentile",
                                            f"{error_stats['percentile_95']:.2f} BDT/g",
                                        )

                                # Detailed results table
                                with st.expander("📋 Detailed Results"):
                                    display_results = results_df.copy()
                                    display_results["error_pct"] = (
                                        display_results["error"]
                                        / display_results["actual"]
                                        * 100
                                    )
                                    display_results.columns = [
                                        "Date",
                                        "Actual Price",
                                        "Predicted Price",
                                        "Absolute Error",
                                        "Error %",
                                    ]
                                    st.dataframe(display_results, width="stretch")

                                    # Download button
                                    csv = display_results.to_csv(index=False)
                                    st.download_button(
                                        label="📥 Download Backtest Results",
                                        data=csv,
                                        file_name=f"backtest_results_{model_type_bt}_{datetime.now().strftime('%Y%m%d')}.csv",
                                        mime="text/csv",
                                    )

                            else:
                                st.error("Backtest failed to produce results.")

                    except Exception as e:
                        st.error(f"Error during backtesting: {str(e)}")
                        logger.error(f"Backtesting error: {e}")

            # Model comparison
            st.markdown("---")
            st.markdown("### 🏆 Model Comparison")

            if st.button("📊 Compare All Models"):
                with st.spinner(
                    "Comparing models... This will take several minutes..."
                ):
                    try:
                        backtest_df = historical_df[
                            (historical_df["purity"] == purity_bt)
                            & (historical_df["metal"] == "gold")
                        ].copy()

                        backtester = GoldPriceBacktester()

                        comparison = backtester.compare_models(
                            df=backtest_df,
                            window_size=window_size,
                            forecast_horizon=forecast_horizon,
                        )

                        if comparison and "comparison_df" in comparison:
                            st.success("✅ Model comparison completed!")

                            comp_df = comparison["comparison_df"]

                            # Display table
                            st.markdown("#### Performance Comparison")

                            display_comp = comp_df.copy()
                            display_comp.index.name = "Model"
                            display_comp = display_comp.reset_index()

                            st.dataframe(display_comp, width="stretch")

                            # Best model
                            st.info(
                                f"🏆 Best Model: **{comparison['best_model'].upper()}** (Lowest MAE)"
                            )

                            # Comparison chart
                            fig_comp = go.Figure()

                            metrics_to_plot = ["mae", "rmse", "mape"]
                            colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

                            for i, metric in enumerate(metrics_to_plot):
                                fig_comp.add_trace(
                                    go.Bar(
                                        name=metric.upper(),
                                        x=comp_df.index,
                                        y=comp_df[metric],
                                        marker_color=colors[i],
                                    )
                                )

                            fig_comp.update_layout(
                                title="Model Performance Comparison",
                                xaxis_title="Model",
                                yaxis_title="Error Value",
                                barmode="group",
                                height=500,
                            )

                            st.plotly_chart(fig_comp, config={"responsive": True})
                        else:
                            st.error("Model comparison failed.")

                    except Exception as e:
                        st.error(f"Error during model comparison: {str(e)}")
                        logger.error(f"Model comparison error: {e}")
        else:
            st.warning("📭 No historical data available for backtesting.")

        st.markdown("</div>", unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #666; font-size: 0.9rem;'>"
        "Gold Price Analyzer | ML Forecasting • Investment Tracking • Jewelry Pricing • Backtesting | Built with Streamlit"
        "</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
